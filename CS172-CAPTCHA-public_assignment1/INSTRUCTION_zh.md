<style>
/* Print-specific styles to improve page breaks */
@media print {
  pre,
  .highlight,
  .codehilite {
    break-inside: auto !important;
    page-break-inside: auto !important;
    overflow: visible !important;
  }
  pre {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  h2,
  h3,
  h4 {
    break-after: avoid-page;
    page-break-after: avoid;
  }
}
</style>

# CS172 计算机视觉 I:CAPTCHA 识别

![CAPTCHAs](assets/CAPTCHAs.jpg)

CAPTCHA(验证码)通过让用户识别带噪声图像中的字符,来区分人类用户和自动化程序。在本次作业中,你将为五位数字 CAPTCHA 构建一条完整的 PyTorch 流程:生成合成图像,加载并预处理图像,设计并训练神经网络,最后评估其预测结果。目标是理解数据质量、模型设计和训练选择如何影响识别性能。

> [!CAUTION]
> 这是一项个人作业。你可以讨论思路,但所有提交的代码和分析都必须是你自己完成的。

## 学习资源

如果你对 PyTorch 或深度学习还不熟悉,请在开始之前浏览以下资源:

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html):张量、数据集、模型、自动求导和优化
- [PyTorch: Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html):自定义数据集和小批量(minibatch)加载
- [TorchVision documentation](https://docs.pytorch.org/vision/stable/):图像读写、模型组件和计算机视觉工具
- [TorchVision transforms](https://docs.pytorch.org/vision/stable/transforms.html):预处理和数据增强
- [Dive into Deep Learning](https://d2l.ai/):神经网络、优化、卷积网络和 ResNet

你不需要读完每一页。请先重点关注张量、`Dataset`、`DataLoader`、`nn.Module`、损失函数、反向传播、卷积层和残差连接。

## 项目组织

清晰的项目结构能让实现更易复用,让 notebook 更易阅读,也能把生成的文件与源代码分开。我们推荐如下组织方式:

```text
.
├── main.ipynb                 # experiment report, results, and analysis
├── README.md                  # setup and usage instructions
├── pyproject.toml             # project metadata and dependencies
├── .gitignore                 # generated and local files ignored by Git
├── src/cs172/                 # reusable Python package
│   ├── __init__.py
│   ├── generate_captcha.py    # CAPTCHA generation
│   ├── datasets.py            # data loading and preprocessing
│   ├── models.py              # neural-network components and models
├── scripts/                   # command-line entry points
│   ├── generate_data.py       # run the data-generation workflow
│   └── train.py               # run training and evaluation
├── data/                      # generated and provided datasets
└── outputs/                   # run configurations, metrics, and checkpoints
```

用 `main.ipynb` 来展示实验流程和结论,而不要把它当作实现代码的唯一存放位置。可复用的逻辑请放在 `src/cs172/` 中;脚本只应提供轻量的命令行接口,去调用这些共享代码。数据集存放在 `data/`,实验产物存放在 `outputs/`。

请用 Git 跟踪源代码、notebook、文档和轻量的配置文件。由于 `data/` 和 `outputs/` 可能变得很大,通常应通过 `.gitignore` 将它们排除。上面的文件名只是建议而非硬性要求:只要你的项目仍然易于运行、理解和复现,就可以调整目录结构。关于 `src/` 布局和可编辑安装的原因,请参阅[工程与可复现性说明](#工程与可复现性说明)。

## 快速开始

### 1. 配置环境

请使用 [Conda](https://anaconda.org/anaconda/conda) 或 [UV](https://github.com/astral-sh/uv) 来管理你的 Python 环境,Python 版本需为 3.10 或更高。我们推荐使用 *pyproject.toml* 来**管理依赖,并以可编辑模式安装你自己位于 `src/` 下的包。**这样,notebook 或脚本可以导入你的包,并且在之后修改源代码时无需重复安装即可生效。

下面是基于推荐项目结构的示例:

```bash
conda create -n cs172 "python>=3.10"  # you may choose a different environment name
conda activate cs172
# Install your own package in editable mode along with its dependencies
pip install -e .
```

### 2. 生成数据

这一部分分两个阶段完成:先实现一个生成单张 CAPTCHA 的函数,再编写一个生成并存储完整数据集的脚本。

#### 2.1 生成单张 CAPTCHA

请参考 `data/test/` 中的样例测试图像了解期望的外观。你的函数应生成一张五位数字 CAPTCHA,其中数字的颜色、位置和旋转角度均随机,并带有干扰线和像素噪声。字体大小需在 **30–40 像素**的闭区间内随机采样。干扰线最多 **5** 条,像素噪声比例为 **2%**。最终图像应带有 **2 像素**的边框。

下面的函数签名给出了一些推荐的参数及默认值:

```python
# generate_captcha.py
def generate_captcha(
    image_size=(260, 80),
    num_digits=5,
    font_size_range=(30, 40),
    ...,
    border_width=2,
    seed=None,
):
    # TODO: choose five random digits
    # TODO: draw them with random colors, positions, and rotations
    # TODO: add random interference lines and pixel noise
    # TODO: return the image and its label
    ...
```

常用的 Pillow API 包括:

- `Image.new(...)`:创建背景图,或为单个数字创建透明图像;
- `ImageDraw.Draw(image)`:获取绘图上下文;
- `ImageFont.truetype(...)` 或 `ImageFont.load_default(...)`:创建字体;
- `ImageDraw.textbbox(...)`:在定位数字之前计算其包围框;
- `ImageDraw.text(...)`:绘制数字;
- `Image.rotate(..., expand=True)`:旋转单独渲染的数字且不被裁剪;
- `Image.paste(...)`:合成旋转后的数字,或添加像素噪声;
- `ImageDraw.line(...)` 和 `ImageDraw.rectangle(...)`:添加干扰线和边框。

要保证每个数字都位于画布之内,并保持从左到右的顺序。请使用局部的、带种子的随机数生成器,使相同的种子能复现相同的图像。

#### 2.2 生成数据集

编写一个生成脚本(例如 `scripts/generate_data.py`),反复调用 `generate_captcha` 并保存图像。每个文件名必须编码标签和一个重复计数。例如,`12345#0.png` 的标签是 `12345`。如果另一张图像的标签相同,则将其命名为 `12345#1.png`,然后是 `12345#2.png`,依此类推。这个后缀可以防止标签相同的图像互相覆盖。

成千上万个小图像文件会带来大量文件系统操作,在共享存储上复制或访问都可能很慢。因此更推荐把图像直接写入同一个 ZIP 压缩包。如果你使用 ZIP 存储,后续的数据集类应当直接从压缩包中读取图像,而不是每次运行都解压出一份新的副本。

### 3. 加载并预处理数据

**补全一个 PyTorch `Dataset`:它能发现样本,将每张图像解码为 RGB,并从文件名中提取五位数字标签。**

下面是一个骨架,请按你自己的设计进行调整:

```python
# datasets.py
import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class CAPTCHA(Dataset):
    def __init__(self, source, image_size, transform: transforms.Transform | None = None):
        self.source = os.fspath(source)
        self.image_size = image_size
        self.transform = transform
        # Recommend sorting image paths so that indexing is deterministic.
        self.img_paths = sorted(self._find_images())

    def _find_images(self):
        # TODO: list image paths in a deterministic order
        # For ZIP input, store archive member names instead
        ...

    def _read_image(self, img_path):
        # TODO: decode the selected file as an RGB image/tensor
        ...

    @staticmethod
    def _parse_label(filename):
        # "12345#0.png" -> tensor([1, 2, 3, 4, 5])
        stem = os.path.splitext(os.path.basename(filename))[0]
        digits = stem.split("#", maxsplit=1)[0]
        # TODO: validate the label and convert it to torch.long
        ...

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = self._read_image(img_path)
        label = self._parse_label(img_path)

        # TODO: apply optional torchvision.transforms.v2 transforms
        # TODO: return image [3, H, W] and label [5]
        return image, label
```

请检查 `__getitem__` 的返回值是否为:

- 形状为 `[3, H, W]` 的 RGB 图像张量;
- 形状为 `[5]` 的 `torch.long` 标签张量。

请使用 [TorchVision transforms](https://docs.pytorch.org/vision/stable/transforms.html) 来构建数据变换流水线。常用操作包括:

- `Resize`:得到统一的空间尺寸;
- `ToDtype(torch.float32, scale=True)`:把整数像素转换为 `[0, 1]` 范围内的浮点值;
- `Normalize(mean, std)`:对每个颜色通道做归一化;
- 温和的随机变换,例如 `RandomAffine` 或 `ColorJitter`,用于训练时的数据增强(通过参数 `transform` 传入)。

```python
from torchvision.transforms import v2

transform = v2.Compose([
    v2.Resize((height, width)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=channel_mean, std=channel_std),
])
```

> [!NOTE]
> 编写新代码时请优先使用 `torchvision.transforms.v2`。它提供当前的变换 API,并且对张量图像的行为一致。支持的输入和操作请参阅 [Transforms v2 文档](https://docs.pytorch.org/vision/stable/transforms.html)。

**典型的模型通常接收经过缩放、转为浮点并归一化的图像,但这些确定性操作不一定要在数据集内部实现。**你可以把它们放在数据集的 transform 中、一个单独的预处理函数里,或模型的开头。请只选择一个位置,在训练和评估时一致地使用它,并写明期望的输入范围和形状。避免把同样的类型转换或归一化应用两次。

随机数据增强要放在模型之外,并且只应用于训练数据集。验证和测试数据应使用确定性的预处理,这样指标才稳定且可比。

### 4. 构建模型

**创建一个 PyTorch `nn.Module`,把一批图像映射为五位数字的预测。**模型应输出形状为 `[batch, num_classes, num_digits]` 的 logits,其中对应数字 0–9 的 `num_classes` 为 10。

你可以从 [`torchvision.models`](https://docs.pytorch.org/vision/stable/models/resnet.html) 中的标准 ResNet 入手。不过,TorchVision 的 ResNet 以全局平均池化加一个全连接层结尾。请替换它的最后一层,使其输出 `num_digits * num_classes` 个 logits,然后对输出进行 reshape:

```python
# models.py
import torch
from torch import nn
from torchvision import models


class TorchvisionResNet(nn.Module):
    def __init__(self, num_digits=5, num_classes=10):
        super().__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes

        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(
            in_features,
            num_classes * num_digits,
        )

    def forward(self, images):
        logits = self.backbone(images)
        return logits.reshape(
            images.shape[0], self.num_classes, self.num_digits
        )
```

对于一批图像,请在整个项目中统一使用以下约定:

```text
input:  [batch, 3, height, width]
output: [batch, 10 classes, 5 digits]
target: [batch, 5 digits]
```

基线模型跑通之后,请实现你自己的 ResNet 变体。CAPTCHA 是从左到右排列的,但标准 ResNet 中的全局平均池化层会把空间维度压缩掉。请研究保留最后一层特征图是否能提升识别效果。

你的自定义变体应当:

1. 实现一个带有主路径和跳跃连接(skip connection)的残差块;
2. 当空间尺寸或通道数发生变化时,在跳跃路径上使用投影(projection);
3. 把残差块堆叠成若干个阶段(stage);
4. 去掉全局平均池化(`AvgPool2d` 或 `AdaptiveAvgPool2d`);
5. 将最后的特征图展平,并映射为 `num_digits * num_classes` 个 logits;
6. 把 logits reshape 为 `[batch, num_classes, num_digits]`。

```python
# TODO: implement a residual block with two convolutional layers, batch normalization, and ReLU activation.
class BasicBlock(nn.Module):
    ...


class CAPTCHAResNet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        layers=(2, 2, 2, 2),
        num_digits=5,
        num_classes=10,
    ):
        super().__init__()
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.in_channels = 64

        # TODO: implement the ResNet stem:
        # convolution -> batch norm -> ReLU -> max pooling
        self.stem = ...

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # TODO: create a classifier for the flattened spatial features.
        # Do not add global average pooling. The classifier must produce
        # num_digits * num_classes outputs.

        # TODO (optional): initialize convolution and normalization layers
        # following torchvision.models.resnet.ResNet.

    def _make_layer(self, block, channels, num_blocks, stride=1):
        out_channels = channels * block.expansion

        # TODO: create a projection shortcut when stride changes or when
        # self.in_channels does not equal out_channels.

        # TODO: make the first block use `stride` and the projection.
        self.in_channels = out_channels

        # TODO: append the remaining blocks with stride=1 and no projection.
        for _ in range(1, num_blocks):
            ...

        # TODO: return the stage as nn.Sequential(*blocks).
        ...

    def forward(self, images):
        # TODO: run the stem and four residual stages.
        # TODO: flatten all remaining spatial features.
        # TODO: compute and reshape the logits to
        # [batch, num_classes, num_digits].
        ...
```

对于分类器,`nn.LazyLinear` 可以在第一次前向传播时推断展平后的特征维度,此后输入的空间尺寸必须保持一致。另一种做法是根据配置的图像尺寸自行计算特征维度,并构造一个普通的 `nn.Linear`。你可以使用最大池化或带步长的卷积来控制特征图的大小。请在 `main.ipynb` 中验证最终的输出形状。在其他训练设置相近的情况下,比较 TorchVision 基线与你的自定义变体。

#### 损失函数应该在哪里实现?

当 logits 的形状为 `[batch, num_classes, num_digits]`、目标的形状为 `[batch, num_digits]` 时,`torch.nn.functional.cross_entropy(logits, targets)` 会在类别维度上计算交叉熵,并在 batch 和各数字位置上取平均。

损失函数可以在以下两个位置之一实现:

- **`models.py`:**当模型具有特殊的输出结构,或需要多个模型特有的损失项时,在模型中写一个 `compute_loss` 方法会比较方便。
- **`train.py`:**把损失放在训练流程中,可以把网络结构与优化目标分开,也更便于在不同模型之间比较损失。

两种设计都可以接受。就本次作业而言,所有模型使用相同的五位置分类目标,所以在 `train.py` 里写一个小函数可能比较合适。如果损失变得与具体模型相关,把它移到模型旁边也是合理的。无论你选择哪种设计,都请只保留一份权威实现,并在训练和验证中一致地使用它。

### 5. 训练模型

**补全一个基于 epoch 的训练流程。**

```python
# train.py
for epoch in range(num_epochs):
    # TODO: train with forward pass, loss, backpropagation, and update
    # TODO: evaluate on validation data without gradients
    # TODO: report loss, digit accuracy, and full-image accuracy
    # TODO: save the latest and best validation checkpoints
    ...
```

请遵循以下常见的实验实践:

- **固定随机种子。**在生成数据、划分数据集和初始化模型之前,为 Python、NumPy 和 PyTorch 设置种子,并在 notebook 中记录该种子。固定种子有助于复现,但并不保证在不同的 PyTorch 版本、平台,或 CPU 与 GPU 之间得到完全相同的结果。关于确定性算法、CUDA 行为和性能权衡,请参阅 PyTorch 的[可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。
- **打乱训练数据。**对训练用的 `DataLoader` 设置 `shuffle=True`,使各个 epoch 的小批量不同。验证和测试的 loader 使用 `shuffle=False`。
- **监控指标。**每个 epoch 结束后,至少要记录训练集和验证集的损失、*逐位数字准确率*(per-digit accuracy)和*整图准确率*(full-image accuracy)。用验证集的整图准确率来选择最佳 checkpoint。

---

两个准确率指标的定义如下:

- **逐位数字准确率**是整个数据集中单个数字位置被预测正确的比例。对于 `N` 张五位数字图像,其分母为 `5N`。
- **整图准确率**是 **五个数字全部**预测正确的 CAPTCHA 图像所占的比例。只要有一个位置错误,该图像就算错误。

例如,如果目标是 `12345`,预测是 `12305`,则五个数字中有四个正确。这个样本对逐位数字准确率的贡献是 `4/5`,但对整图准确率的贡献是 `0`。因此整图准确率更严格,也更能反映 CAPTCHA 是否被成功解出。

---

你的训练函数或脚本(例如 `scripts/train.py`)应当暴露常见的实验设置,例如模型选择、batch size、学习率、权重衰减、数据增强、随机种子和从 checkpoint 恢复训练。请选择清晰的接口,并在你自己的 `README.md` 中说明。

### 6. 评估与分析

使用 `main.ipynb` 报告最终的测试性能,以及要求的分析和消融实验。请用清晰的图或表来呈现结果,在每次对比中保持无关设置不变,并简要解释你的结论。必选项和评分请见下面的评分标准。

## 评分标准

评分依据是已执行的 `main.ipynb` 中展示的代码、输出、图、表和分析。满分 100 分。

### 1. 数据生成 — 5 分

- 证明生成器能产生有效的五位数字 CAPTCHA 图像。
- 展示具有代表性的样本,体现随机的颜色、位置、旋转、干扰线和像素噪声。

### 2. 数据集与预处理 — 5 分

- 证明自定义的 `Dataset` 能正确加载图像并解析标签。
- 展示 `DataLoader` 的一个或多个 batch,并验证图像形状、标签形状和数据类型。
- 展示训练/验证集的划分以及所要求的预处理。

### 3. 模型 — 5 分

- 证明你实现的某个模型能接收一个 batch 并返回形状符合预期的 logits。
- 简要概括模型架构,并报告其参数量。

### 4. 训练报告 — 5 分

- 绘制训练和验证的损失、逐位数字准确率及整图准确率随 epoch 的变化曲线。
- 说明最终模型所用的配置,包括随机种子、模型、训练集大小、优化器、batch size、学习率、正则化、数据增强和 epoch 数。

### 5. 评估 — 40 分

在要求的测试数据上评估所选模型(`data/test/` 下的 5 个测试集,共 500 张图像),并报告两种准确率。得分的计算方式为:

```text
evaluation score = 20 × max(per-digit accuracy - 0.95, 0) / 0.05 + 20 × max(full-image accuracy - 0.9, 0) / 0.1
```

### 6. 分析与消融实验 — 40 分

每一项 5 分。对每个实验,都要保持无关设置不变,并使用验证集整图准确率来选择 checkpoint。随 epoch 变化的曲线应展示验证集指标;最终性能的对比应直接报告汇总后的测试结果。每个结果都要用清晰、简单的图或表呈现,并给出简明的结论。

1. **训练集大小:**评估至少三种不同的训练集大小,并绘制汇总测试性能随训练图像数量变化的曲线。
2. **数据增强与过拟合:**比较使用与不使用数据增强的训练。绘制验证指标随 epoch 的变化曲线,比较汇总测试结果,并解释其对过拟合的影响。
3. **正则化与过拟合:**比较多个权重衰减(weight decay)取值。绘制验证指标随 epoch 的变化曲线,比较汇总测试结果,并解释其对过拟合的影响。
4. **Batch size:**比较多个 batch size,使用验证曲线、汇总测试结果以及一致测得的训练时间。讨论收敛性、准确率和速度。
5. **学习率:**比较多个学习率,使用验证曲线和汇总测试结果。讨论收敛速度、稳定性和测试性能。
6. **骨干网络对比:**在可比的设置下,比较至少两个模型,例如 ResNet-18 和 ResNet-34。报告验证曲线和汇总测试结果。
7. **全局平均池化:**在可比的设置下,比较有无全局平均池化的模型。报告验证曲线和汇总测试结果,并分析保留空间信息是否有助于 CAPTCHA 识别。
8. **预测可视化:**将预测结果与真实标签并排展示。如果整图准确率低于 100%,则**必须**包含并分析失败案例。

## 提交

需要提交:

1. 已完成的 `main.ipynb` 导出的 PDF。
2. `ChineseName-StudentID.zip`,其中包含**不含 `data/`** 的完整项目。

## 工程与可复现性说明

这些做法对于一个最简方案并非必需,但它们能让项目更易于维护、复现和评估。

### 将可复用代码与入口脚本分开

小型原型可以把所有 Python 文件都放在 notebook 旁边。这种扁平布局上手容易,但随着项目增长,各部分的职责和导入关系会变得不清晰。[Python Packaging User Guide 对 `src` 布局与扁平布局的比较](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)解释了二者在导入和安装行为上的差异。推荐的布局把可复用代码与可执行的工作流分开:

- `src/cs172/` 存放可导入的包:数据集、模型、指标以及其他可复用的函数。
- `scripts/` 存放入口脚本。每个脚本应当解析参数并调用包中的函数,而不是重复实现它们。

例如,`scripts/train.py` 和 `main.ipynb` 都可以从 `cs172` 导入同一个数据集和模型。这能确保 notebook 的结果和命令行实验使用完全相同的代码。请包含 `src/cs172/__init__.py` 以明确包的边界;参见 [Python 教程中关于包的部分](https://docs.python.org/3/tutorial/modules.html#packages)。

### 为什么使用 `pyproject.toml`?

[`pyproject.toml`](https://packaging.python.org/en/latest/specifications/pyproject-toml/) 是现代的、标准化的 Python 项目配置文件。对本次作业而言,它取代了旧项目通常放在 `setup.py` 或 `setup.cfg` 中的项目元数据和打包配置。它的 `[project]` 表声明了包名、版本、所支持的 Python 版本和直接的运行时依赖等信息,也可以包含构建系统以及各工具的配置。

使用 `pyproject.toml` 是良好的实践,因为它是声明式的且与具体工具无关:打包工具无需执行任意的 setup 代码就能读取项目元数据,配置也可以统一保存在一种有文档规范的格式中。它还支持隔离构建,并能配合 setuptools、Hatchling 和 Flit 等现代构建后端使用。

运行 `pip install -e .` 会读取项目配置、安装其依赖,并以**可编辑模式**安装本地包,因此对源代码的修改会立刻对脚本和 notebook 生效。详情请参阅 pip 关于[本地项目与可编辑安装](https://pip.pypa.io/en/stable/topics/local-project-installs/)的指南。

`requirements.txt` 的作用不同;`pyproject.toml` 并不能直接取代它。requirements 文件列出某个特定环境要安装的包,并且可以固定确切版本以便复现。相比之下,`pyproject.toml` 定义的是项目本身及其直接依赖的约束。一个项目可以两者并用,但本次作业以 `pyproject.toml` 作为权威的项目配置。

### 为什么需要 Python 3.10 或更高版本?

本项目使用了现代的内置类型注解语法。Python 3.9 引入了带参数的内置集合类型,如 `list[str]` 和 `tuple[int, int]`;Python 3.10 增加了 `str | None` 这类注解中使用的 `|` 联合类型运算符。这些写法比 `typing` 中较旧的等价形式(如 `List[str]` 和 `Optional[str]`)更简短,也更易读。完整的语法和语义请参阅 [Python 3.10 `typing` 文档](https://docs.python.org/3.10/library/typing.html)。类型注解能提升可读性并便于静态分析,但 Python 不会在运行时自动强制执行它们。

### 将配置与结果放在一起

为每个实验创建单独的输出目录。把随机种子、数据集描述、模型配置、超参数、指标和 checkpoint 保存在一起。这样可以避免意外覆盖,并使 `main.ipynb` 中的每个表格或图都能追溯到某一次具体的运行。PyTorch 的[可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)解释了种子控制、确定性操作及其局限。

```text
outputs/<experiment-name>/
├── config.json
├── metrics.json or metrics.csv
├── best.pt
└── latest.pt
```

### 让 checkpoint 可恢复训练

仅保存权重的 checkpoint 足以用于推理。若要忠实地恢复训练,还应保存当前 epoch、优化器状态、学习率调度器状态和最佳验证分数。加载时使用 `map_location`,这样 checkpoint 才能在不同设备上打开。关于推理用 checkpoint、可恢复训练的 checkpoint 和跨设备加载,请参阅 PyTorch 的[保存与加载模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)教程。

请根据验证性能而不是测试性能来选择最佳 checkpoint。只有在模型和超参数的选择全部完成之后,才用测试集评估所选模型。
