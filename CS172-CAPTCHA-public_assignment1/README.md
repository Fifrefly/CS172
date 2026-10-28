# CS172 CAPTCHA 识别

用 ResNet 一次预测 5 位数字验证码。数据由 `src/cs172/generate_captcha.py` 随机生成，模型输出形状为 `[batch, 10, 5]` 的 logits，损失为 5 个位置交叉熵之和。分析与消融结果见 `main.ipynb`。

## 目录结构

```
src/cs172/            可导入的包
  generate_captcha.py   单张验证码生成器
  datasets.py           CAPTCHA（目录）/ ZIP_CAPTCHA（zip）数据集，从文件名 "12345#0.png" 解析标签
  models.py             resnet18/resnet34（TorchVision，GAP）、captcha_resnet18/34（无 GAP，flatten）
scripts/
  generate_data.py      批量生成数据集（目录或 zip）
  train.py              训练入口
  evaluate.py           在一个或多个数据集上评测 checkpoint
  evaluate.sh           批量评测某个实验目录下的所有 run
  train.sh              本 notebook 用到的全部实验（训练 + 评测）
  slurm/                集群脚本：job.sbatch、gen_data.sbatch、submit.sh
data/                   training100k.zip、validation.zip、test/test0..test4（不随代码提交）
outputs/<实验名>/<run>/ config.json、metrics.json、checkpoints/best.pt、evaluation.json、tensorboard/
```

## 环境

```bash
pip install -e .                 # numpy matplotlib pillow tqdm torch torchvision
pip install -e ".[tensorboard]"  # 可选：训练时写 TensorBoard
```

`scripts/train.py` 需要 CUDA GPU；`scripts/evaluate.py` 在 CPU 上也能运行。所有脚本在仓库根目录执行，并需要 `PYTHONPATH=src`（`pip install -e .` 后可省略）。

## 1. 生成数据

```bash
export PYTHONPATH=src
python scripts/generate_data.py data/training100k.zip -n 100000 --zip --seed 0
python scripts/generate_data.py data/validation.zip   -n 5000   --zip --seed 10000000
```

第 i 张图使用种子 `seed + i`，所以相同参数一定生成相同数据；训练集与验证集的种子区间不重叠。同时会写出 `data/*.json` 记录生成参数。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `output_dir` | — | 输出目录；加 `--zip` 时为 `.zip` 文件 |
| `-n, --num-images` | 必填 | 图片数量 |
| `--image-size W H` | 260 80 | 图片宽高 |
| `--num-digits` | 5 | 每张图的数字个数 |
| `--font-size-range MIN MAX` | 30 40 | 字号范围 |
| `--minimum-padding H V` | 5 10 | 左右 / 顶部最小留白 |
| `--rotation-angle-range MIN MAX` | -30 30 | 旋转角度（度）；`--no-rotation` 关闭 |
| `--max-noisy-lines` | 5 | 干扰线最多条数 |
| `--noise-ratio` | 0.02 | 噪点像素比例 |
| `--border-width` | 2 | 边框宽度 |
| `--seed` | 0 | 基础随机种子 |

**数字布局。** 测试图里数字的横向位置是自由的：先把剩余宽度随机分成左边距、4 个间隙、右边距，数字按从左到右的顺序排列、互不重叠，可以挤在一起也可以相距很远；整体偏下，下沿可以贴近底部边框。生成器按此布局摆放（若把数字固定在 5 个等宽槽里，模型会学到"第 k 个输出对应第 k 个槽"，在测试图上整图准确率只有约 70%）。

## 2. 训练：`scripts/train.py`

```bash
python scripts/train.py \
  --data-dir data/training100k.zip --val-data-dir data/validation.zip \
  --output-dir outputs/my-exp --model resnet18 --epochs 20 --batch-size 64 \
  --learning-rate 1e-3 --weight-decay 1e-4 --augment --seed 42 --tensorboard
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--data-dir` | `data/training` | 训练集目录或 zip |
| `--val-data-dir` | 无 | 验证集目录或 zip；每个 epoch 评估一次，按验证整图准确率保存 `best.pt` |
| `--num-images` | 全部 | 只用训练集中随机（由 `--seed` 决定）选出的前 N 张 |
| `--output-dir` | `outputs` | 实验目录；每次运行新建 `<时间戳>-<id>/` 子目录，多个 seed 互不覆盖 |
| `--model` | `resnet18` | `resnet18`、`resnet34`（TorchVision + GAP）、`captcha_resnet18`、`captcha_resnet34`（无 GAP，保留空间信息） |
| `--image-size H W` | 80 260 | 输入尺寸（图片会 resize 到该尺寸） |
| `--epochs` | 20 | 训练轮数，学习率按 epoch 做余弦退火到 0 |
| `--batch-size` | 64 | 批大小 |
| `--learning-rate` | 1e-3 | AdamW 初始学习率 |
| `--weight-decay` | 1e-4 | AdamW 权重衰减 |
| `--workers` | `min(4, CPU 数)` | DataLoader 进程数 |
| `--seed` | 42 | 随机种子（Python / NumPy / PyTorch） |
| `--augment` | 关 | 仅训练集：`RandomAffine(±5°, 平移 5%×2%, 缩放 0.95–1.05)` + `ColorJitter(0.1)` |
| `--tensorboard` | 关 | 写 TensorBoard 日志（需要安装 tensorboard） |
| `--resume CKPT` | 无 | 从 checkpoint 继续训练（恢复优化器和调度器） |
| `--weights-only` | 关 | 与 `--resume` 合用：只加载模型权重，从头开始训练计划 |

预处理（resize、转 float32、`mean=std=0.5` 归一化）在数据集里完成，训练和评测完全一致，输入范围约 `[-1, 1]`，形状 `[3, 80, 260]`。

输出：`metrics.json`（每个 epoch 的 train/val 的 loss、逐位准确率、整图准确率、训练耗时）、`checkpoints/best.pt` 与最新 checkpoint。

## 3. 评测：`scripts/evaluate.py` / `scripts/evaluate.sh`

```bash
python scripts/evaluate.py --checkpoint outputs/my-exp/<run>/checkpoints/best.pt \
  --data data/test/test0 data/test/test1 data/test/test2 data/test/test3 data/test/test4

scripts/evaluate.sh --exp-dir outputs/my-exp --data data/test/test0 data/test/test1 data/test/test2 data/test/test3 data/test/test4
```

模型类型和输入尺寸从 run 目录里的 `config.json` 读取（也可用 `--model`、`--image-size` 覆盖）。结果写入 run 目录的 `evaluation.json`：每个测试集的逐位准确率和整图准确率。其他选项：`--batch-size`、`--workers`、`--device {auto,cpu,cuda}`、`--output-json`。

得分 = 20 × max(逐位准确率 − 0.95, 0) / 0.05 + 20 × max(整图准确率 − 0.9, 0) / 0.1，按 5 个测试集共 500 张图合并计算。

## 4. 复现 notebook 中的全部实验：`scripts/train.sh`

```bash
scripts/train.sh list                 # 列出实验名和分组
scripts/train.sh final 42             # 最终模型（10 万张、ResNet-18、增强）
scripts/train.sh size 42 43 44        # 训练集大小 1k / 2k / 5k / 10k / 20k / 50k
scripts/train.sh ablate 42 43 44      # 增强、weight decay、学习率、骨干、有无 GAP
scripts/train.sh bsx 42 43 44         # batch size 16/32/64/128/256（同一 GPU 上顺序运行，耗时可比）
```

每个实验训练完会自动在 5 个测试集上评测。基线为 ResNet-18、batch 64、学习率 1e-3、weight decay 1e-4、增强、20 epoch；每个实验只改变一个因素，seed 为 42、43、44。可用环境变量 `OUTPUT_ROOT`、`TRAIN_DATA`、`VAL_DATA`、`EPOCHS`、`WORKERS`、`PYTHON` 调整。

### 在 Slurm 集群上运行

```bash
sbatch scripts/slurm/gen_data.sbatch                 # 生成数据（CPU 作业）
scripts/slurm/submit.sh "final 42"                   # 提交一个 GPU 作业
scripts/slurm/submit.sh "size 42" "ablate 42" "bsx 42" "size 43" "ablate 43" "bsx 43"
```

`submit.sh` 按 A40（`hexm`）→ L40（`hexm_l40`）→ RTX 2080 Ti（`normal`）的顺序选择当前有空闲 GPU 的类型，并保证同时排队/运行的作业不超过 `MAX_JOBS`（默认 6）。日志写入 `slurm_logs/`。

## 5. Notebook

`main.ipynb` 用上面的数据与 `outputs/` 中保存的 `config.json`、`metrics.json`、`evaluation.json` 生成全部图表，不会重新训练模型。运行前确认 `data/` 与 `outputs/` 就位，用 `learning` kernel 从上到下 Run All。
