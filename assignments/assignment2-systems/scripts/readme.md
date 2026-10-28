# 租卡后的实验运行指南

按当前仓库中的实际入口整理。下列命令供你在租用的 Linux GPU 服务器上执行；本文件没有附带实测结果，也不会修改实验实现。

## 1. 选择机器和实验顺序

| 机器 | 适合运行的实验 |
|---|---|
| 同机两张 A100/A800 80GB，或两张 RTX PRO 6000 96GB | XL DDP、优化器状态分片、FSDP，以及下列单卡实验 |
| 单张 80/96GB GPU | 普通 attention、FlashAttention 对比、梯度检查点 |
| 同机六张 16/24GB GPU | 完整的 2/4/6 卡通信实验；例如六张 RTX 3090 24GB |

默认 XL FP32 DDP 仅参数、梯度和 AdamW 状态就约 50.8 GiB/卡，另外还需要激活和临时张量。两张 48GB 的显存不会在 DDP 中合并。80/96GB 是容量建议，不是所有配置均能运行的实测保证。

普通 attention 的极大配置可能 OOM，相关脚本会记录 `oom` 并继续；完整运行脚本不等于每个配置都成功。双卡实例上的单卡实验只使用其中一张卡，如需节省费用，可以分阶段租用。数学推导的四个 `*_calcs` 目录不需要 GPU。

建议顺序：环境检查 → 小模型正确性测试 → 需要的单卡实验 → 双卡数值计时 → Nsight 截图 → 六卡通信 → 下载结果。

## 2. 上传项目、准备环境

先将整个项目上传到服务器，包含 `cs336-basics/`、`cs336_systems/`、`scripts/`、`tests/`、`pyproject.toml` 和 `uv.lock`。在服务器上重新创建环境，不复制本机 `.venv`。

下面的项目目录是 AutoDL 常用位置的示例，改成你实际上传的位置。后续所有命令均在项目根目录、同一个 Bash 会话中执行。

```bash
cd /root/autodl-tmp/assignment2-systems
uv --version
uv sync --locked
uv run --no-sync python --version
```

如果没有 `uv`，先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装，再执行上面的命令。若锁文件与项目配置不一致，应先确认本地项目与上传文件一致，而不是临时替换依赖版本。

可选：为避免 SSH 断开中断实验，先进入已安装的 tmux，再执行本节初始化与后续命令：

```bash
tmux new -s cs336
```

断开时按 `Ctrl+b`，松开后按 `d`；重新连接服务器后恢复：

```bash
tmux attach -t cs336
```

## 3. 创建本轮结果目录并检查设备

每轮重新实验先执行这一段，避免覆盖之前备份的日志。新开终端后也需要重新设置这些会话变量。

```bash
set -o pipefail
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "logs/$RUN_TAG" "profiles/$RUN_TAG" "results/$RUN_TAG"
nvidia-smi | tee "logs/$RUN_TAG/nvidia-smi.txt"
nvidia-smi -L | tee "logs/$RUN_TAG/gpu-list.txt"
nvidia-smi topo -m | tee "logs/$RUN_TAG/topology.txt"
uv pip show torch triton | tee "logs/$RUN_TAG/packages.txt"
uv run --no-sync python -m torch.utils.collect_env 2>&1 | tee "logs/$RUN_TAG/environment.txt"
```

确认 GPU 数量、型号、空闲显存及互联符合所选实验。当前项目使用 PyTorch 2.11 和 CUDA 13.0 wheel；CUDA 13.x 的基础驱动要求为 580 系列或更新，具体 JIT 功能还取决于完整软件组合。`nvidia-smi` 中的 CUDA Version 表示驱动支持能力，不等于 PyTorch 实际使用的运行时版本，后者可在环境报告中核对。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

如果服务器上有 Git 工作区，可额外记录代码版本和未提交修改：

```bash
git rev-parse HEAD > "logs/$RUN_TAG/commit.txt"
git diff > "results/$RUN_TAG/source.patch"
```

后续通过 `tee` 同时显示并保存输出。若命令失败，先检查日志，不继续复制可能残留的旧 CSV。实验脚本通常会覆盖自己目录中的原始 CSV，因此每次成功运行后立即按下面的命令备份。

## 4. 先做正确性检查

这些是仓库现有测试入口。先确认对应实现和适配器已完成，避免直接使用 XL 模型排查基础错误。

双卡实例：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync pytest tests/test_ddp.py tests/test_sharded_optimizer.py tests/test_fsdp.py -q --tb=short 2>&1 | tee "logs/$RUN_TAG/distributed-tests.log"
```

单卡或双卡实例的第一张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync pytest tests/test_attention.py -q --tb=short 2>&1 | tee "logs/$RUN_TAG/attention-tests.log"
```

不要在正式性能测量的同时运行其他测试或训练任务。

## 5. 单卡实验

### 5.1 普通 PyTorch attention：eager 与 compiled

默认 batch=8，序列长度最高 16384，遍历多种 head dimension，分别测 eager 和 compiled。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -u -m scripts.pytorch_attention.benchmark_attn 2>&1 | tee "logs/$RUN_TAG/pytorch-attention.log"
cp -v scripts/pytorch_attention/*_benchmark_attn.csv "results/$RUN_TAG/"
```

最大配置显存需求高，编译及完整的重复测量也会消耗时间；查看 CSV 的 `status` 区分成功测量和 OOM。

### 5.2 FlashAttention：PyTorch 与 Triton

遍历序列长度 128 到 65536、多个 head dimension，以及 BF16/FP32，分别测前向、反向和前向加反向。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -u -m scripts.flash_benchmarking.benchmark_flash 2>&1 | tee "logs/$RUN_TAG/flash-attention.log"
cp -v scripts/flash_benchmarking/*_benchmark_flash.csv "results/$RUN_TAG/"
```

朴素 PyTorch attention 最大 FP32 配置即使在 80/96GB 卡上也可能 OOM；这不等于 Triton 配置失败。按每行及各阶段的记录判断，不把 `OOM` 当作耗时。

### 5.3 梯度检查点显存

默认 XL 的 32 个 Transformer block、FP32、batch=4、序列长 2048，依次测试不同 checkpoint 粒度。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -u -m scripts.gradient_checkpointing.benchmark_checkpoint 2>&1 | tee "logs/$RUN_TAG/checkpoint.log"
cp -v scripts/gradient_checkpointing/*_benchmark_checkpoint.csv "results/$RUN_TAG/"
```

输出的 `peak_mem_MiB` 是参数、梯度和临时激活共同形成的 allocated 峰值；较大粒度能否容纳仍需看实测状态。

### 5.4 基础 autocast 示例（可选）

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -u -m scripts.benchmark_script 2>&1 | tee "logs/$RUN_TAG/autocast-example.log"
```

当前入口只运行 `run_autocast_toy_model`，不是完整语言模型计时。文件中的 `benchmarking_script` 当前没有被入口调用；若要做那部分实验，需要先自行确认入口、模型规模、是否反向及精度配置。启用后生成的内存快照写在运行时工作目录。

## 6. 双卡实验：普通运行与数值结果

本节默认 XL、全局 batch=4、每卡 batch=2、序列长 512。脚本已通过 `spawn` 启动两个进程，**直接运行模块即可，不再套 `torchrun`**。

`CUDA_VISIBLE_DEVICES=0,1` 表示让这条命令使用可见的前两张 GPU，并不创建额外 GPU。分布式脚本使用固定端口 29500，按顺序运行，不要同时启动多个。

### 6.1 Naive DDP

文件名确实是 `becnmark_naive_ddp`，保留仓库现有拼写：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -u -m scripts.naive_ddp_benchmarking.becnmark_naive_ddp 2>&1 | tee "logs/$RUN_TAG/naive-ddp.log"
cp scripts/naive_ddp_benchmarking/benchmark_dist.csv "results/$RUN_TAG/naive-ddp.csv"
```

### 6.2 Flat DDP

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -u -m scripts.minimal_ddp_flat_benchmarking.becnmark_flat_ddp 2>&1 | tee "logs/$RUN_TAG/flat-ddp.log"
cp scripts/minimal_ddp_flat_benchmarking/benchmark_dist.csv "results/$RUN_TAG/flat-ddp.csv"
```

以上两组 CSV 含每个 rank 的迭代及同步计时。比较时保持同机、同配置，不能只挑不同机器上最快的一行。

### 6.3 普通与分片 AdamW

脚本依次运行 `naive` 与 `sharded` 优化器，两组均使用 OverLapDDP，并记录初始化、优化器前后阶段的显存峰值和迭代耗时。

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -u -m scripts.optimizer_state_sharding_accounting.benchmark_sharder_optim 2>&1 | tee "logs/$RUN_TAG/sharded-optimizer.log"
cp scripts/optimizer_state_sharding_accounting/benchmark_dist.csv "results/$RUN_TAG/sharded-optimizer.csv"
```

显存结果的单位为 GiB，时间为 ms；优化器前后字段表示对应阶段窗口的峰值，不是两个瞬时读数。`step_comm_time_mean_ms` 也不能直接解释为所有通信的总耗时，重叠通信应结合 GPU trace 分析。

### 6.4 Overlap DDP 与 FSDP 普通运行（可选）

下面两条用于直接检查完整运行。若已准备好 Nsight、目的是采集截图，可直接运行下一节的采集命令，避免无必要地重复 XL 实验。

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -u -m scripts.ddp_overlap_individual_parameters_benchmarking.benchmark_overlap_ddp 2>&1 | tee "logs/$RUN_TAG/overlap-ddp.log"
cp scripts/ddp_overlap_individual_parameters_benchmarking/benchmark_dist.csv "results/$RUN_TAG/overlap-config.csv"
```

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -u -m scripts.fsdp_accounting.benchmark_fsdp 2>&1 | tee "logs/$RUN_TAG/fsdp.log"
cp scripts/fsdp_accounting/benchmark_dist.csv "results/$RUN_TAG/fsdp-config.csv"
```

注意当前行为：

- overlap 脚本依次运行 `naive`、`flat`、`overlap`。
- FSDP 脚本依次运行 `naive`、`flat`、`overlap`、`fsdp`，不是只运行 FSDP。
- 这两个 CSV **只有配置，没有正式耗时数据**，需要从 Nsight 获取相关耗时。
- FSDP 路径使用 BF16 计算权重、FP32 主分片和优化器更新；同脚本 DDP 路径使用 FP32。
- 当前 FSDP 是本层计算前同步聚集，尚无提前两层的异步预取；截图结论必须以实际 trace 为准。

## 7. Nsight Systems：采集双卡时间线

先检查服务器是否安装 Linux 版 Nsight Systems CLI：

```bash
command -v nsys
nsys --version
```

未安装时，先按 [NVIDIA 官方安装入口](https://developer.nvidia.com/nsight-systems/get-started) 安装。普通计时脚本不要求安装 nsys；生成这里需要的 `.nsys-rep` 则需要它。

### 7.1 DDP：无重叠与有重叠

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync nsys profile \
    --trace=cuda,nvtx \
    --pytorch=autograd-nvtx \
    --sample=none --cpuctxsw=none \
    --output="profiles/$RUN_TAG/ddp-overlap" \
    python -u -m scripts.ddp_overlap_individual_parameters_benchmarking.benchmark_overlap_ddp \
    2>&1 | tee "logs/$RUN_TAG/nsys-ddp-overlap.log"
cp scripts/ddp_overlap_individual_parameters_benchmarking/benchmark_dist.csv "results/$RUN_TAG/overlap-profile-config.csv"
```

报告：`profiles/<本轮 RUN_TAG>/ddp-overlap.nsys-rep`。在同一 GPU 上选取预热后的 naive 与 overlap 迭代，观察 backward 计算 kernel 与 NCCL kernel 是否存在时间交叠，分别截图。

### 7.2 FSDP：权重 all-gather 与计算

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync nsys profile \
    --trace=cuda,nvtx \
    --pytorch=autograd-nvtx \
    --sample=none --cpuctxsw=none \
    --output="profiles/$RUN_TAG/fsdp-accounting" \
    python -u -m scripts.fsdp_accounting.benchmark_fsdp \
    2>&1 | tee "logs/$RUN_TAG/nsys-fsdp-accounting.log"
cp scripts/fsdp_accounting/benchmark_dist.csv "results/$RUN_TAG/fsdp-profile-config.csv"
```

报告：`profiles/<本轮 RUN_TAG>/fsdp-accounting.nsys-rep`。选择 `rank*/fsdp/forward` 标记对应的 GPU 工作，观察 all-gather 与依赖该权重的计算，记录通信持续时间、是否重叠及暴露的等待，保留时间刻度与 GPU 标识。

两组采集都使用全程捕获，因为当前脚本没有调用 profiler start/stop；不要添加 `--capture-range=cudaProfilerApi`。预热也会进入报告，分析时选择正式迭代。NVTX 是 CPU 范围，不能直接将其宽度当作完整 GPU 耗时，需要查看 GPU stream 并关联 CUDA API。Nsight 本身会引入开销，普通计时与 profiler 计时应注明采集方式。[Nsight 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/)

下载报告到本机，用相同或更新版本的 Nsight Systems GUI 打开，无需在远程服务器安装桌面环境。

## 8. 同机 2/4/6 GPU 通信实验

完整覆盖需要同一个实例内六张可见 GPU。脚本内部依次测试 2、4、6 个进程及多种输入大小；可见 GPU 数不足时会跳过对应组，因此只有两张卡时不能声称完成了 4/6 卡测量。

六卡实例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run --no-sync python -u -m scripts.distributed_communication_single_node.benchmark_dist 2>&1 | tee "logs/$RUN_TAG/communication-2-4-6gpu.log"
cp scripts/distributed_communication_single_node/benchmark_dist.csv "results/$RUN_TAG/communication-2-4-6gpu.csv"
```

仅使用现有双卡实例测两卡组：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -u -m scripts.distributed_communication_single_node.benchmark_dist 2>&1 | tee "logs/$RUN_TAG/communication-2gpu.log"
cp scripts/distributed_communication_single_node/benchmark_dist.csv "results/$RUN_TAG/communication-2gpu.csv"
```

记录本次 GPU 拓扑；不同机器的通信结果不能只按显存容量直接比较。

## 9. 保存并下载结果

打包本轮日志、已备份 CSV 和 Nsight 报告：

```bash
tar -czf "cs336-results-${RUN_TAG}.tar.gz" "logs/$RUN_TAG" "results/$RUN_TAG" "profiles/$RUN_TAG"
ls -lh "cs336-results-${RUN_TAG}.tar.gz"
```

从平台文件管理页面下载压缩包；若额外启用了基础模型 benchmark 的 `.pickle` 内存快照，也要单独保存。确认结果已下载后再释放租用实例。

本指南不包含 leaderboard 的最终性能测量：该题指定双 B200 与官方 benchmark，不能将这里在 A100/A800/PRO 6000 上的 XL 脚本结果直接作为 leaderboard 成绩。
