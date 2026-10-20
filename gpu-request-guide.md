# GPU 申请说明

## 申请规则

- 在服务器上通过 Slurm 运行代码、测试和实验。
- Python 命令使用 `tap` Conda 环境(已验证 Python 版本为 `3.9.25`)。
- 申请 **A40 GPU**:分区 `hexm`,账号 `hexm`,资源类型 `gpu:NVIDIAA40`。
- 申请 **L40 GPU**:分区 `hexm_l40`,账号 `hexm`,资源类型 `gpu:NVIDIAL40`。
- 同时使用的 GPU 总数不超过 **8 张 L40、12 张 A40**(两者独立计算),要把已有占用和排队中的任务算进去。互相独立的任务尽量并行;任务支持多卡时可以用多张 GPU,但仍受上述上限约束。
- 用 `--time=0` 申请不限时长。
- 运行中加排队中的作业总数不超过分区当前的限制:**每人 6 个作业**。

## SSH 访问

```sshconfig
Host sist-ai-cluster
    HostName 10.15.89.191
    User zhaowb2024
    Port 22112
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

```bash
ssh sist-ai-cluster
```

非交互式 SSH 要用登录 shell 来初始化命令路径:

```bash
ssh sist-ai-cluster 'bash -lc "squeue -u zhaowb2024"'
```

## Conda 环境初始化

在每个分配到的 shell 或批处理脚本里初始化 `tap`:

```bash
module load anaconda3/python3.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate tap
cd /public/home/zhaowb2024/projects
```

## Slurm 任务

Slurm 版本 `19.05.5`。A40 容量:`hexm01` 有 8 张,`hexm02` 有 4 张。申请资源前先查看当前占用:

```bash
squeue -u zhaowb2024 -o '%.18i %.12P %.10T %.20b %j'
sinfo -p hexm -N -o '%N|%t|%G'
scontrol show node hexm01,hexm02
```

### 交互式单卡任务

申请一个 shell,按上面的方式初始化 `tap`,再运行命令,结束后退出 shell:

```bash
srun -p hexm -A hexm -N 1 -n 1 -c 8 --mem=64G \
  --gres=gpu:NVIDIAA40:1 --time=0 --pty bash -l
```

### 批处理任务

在服务器上保存一个运行脚本,例如 `job.sbatch`。把 `TASK_RUNNER.sh` 换成实际脚本,用数组索引选择每个实验:

```bash
#!/bin/bash -l
set -eo pipefail
module load anaconda3/python3.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate tap
cd /public/home/zhaowb2024/projects
srun bash TASK_RUNNER.sh "${SLURM_ARRAY_TASK_ID}"
```

A40 预算充足时,提交 4 个各用 1 张 A40 的独立任务(用 L40 时改成 `-p hexm_l40` 和 `--gres=gpu:NVIDIAL40:1`):

```bash
mkdir -p slurm_logs
sbatch -p hexm -A hexm -N 1 -n 1 -c 8 --mem=64G \
  --gres=gpu:NVIDIAA40:1 --time=0 --array=0-3%4 \
  --chdir=/public/home/zhaowb2024/projects \
  --output=slurm_logs/%A_%a.out --error=slurm_logs/%A_%a.err job.sbatch
```

数组并发数(`%4`)要根据剩余 GPU 额度调整。示例里的 8 个 CPU 和 64 GB 内存与之前成功过的申请一致,请按实际任务调整大小。较大的实验网格要分批提交,并保持在提交数量限制之内。

### 查看和取消任务

用实际的作业 ID:

```bash
scontrol show job JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES
scancel JOB_ID
```

2026 年 10 月 13 日通过 SSH 验证过:`tap` 激活成功,4 任务的资源申请通过了 Slurm 的 `--test-only` 检查。执行前要重新确认资源是否可用。
