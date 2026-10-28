# 实验运行速查

在租用的 Linux 服务器上执行，所有命令都在项目根目录、同一个 Bash 会话中运行。

## 租卡建议

**主力方案：同一实例租两张 A100/A800 80GB，完成 XL 双卡实验。显存余量优先时，选同机两张 RTX PRO 6000 Blackwell 96GB。完整的六卡通信实验单独租六张 24GB 卡。**

按当前脚本实际运行配置选择，不把 `model_zoo` 中未启用的 `10B` 预设算作默认需求：

| 实验 | 推荐配置 | 选型依据 |
| --- | --- | --- |
| Naive / Flat / Overlap DDP | 同机 **2×A100/A800 80GB**；96GB 卡也可 | XL、FP32、AdamW，每卡保留完整参数、梯度和优化器状态 |
| 优化器分片、FSDP accounting | 同机 **2×80GB 或 2×96GB** | 脚本还会跑未分片 DDP 基线，不能只按 FSDP 的分片容量选卡 |
| 梯度检查点 | **1×96GB** 余量更大；80GB 可先试跑 | XL、FP32、batch=4、序列长 2048；峰值随 checkpoint 粒度变化 |
| 普通 PyTorch attention | **1×80/96GB** | 最大 batch=8、序列长 16384；编译版峰值需实测 |
| FlashAttention 对比 | Triton 调试可用 **1×24GB**；完整 BF16 对比优先 **1×80GB** | 最大序列长 65536；朴素 FP32 最大配置在 80/96GB 上仍可能 OOM，脚本会记录 |
| 单机通信 | 同机 **6×16/24GB**，例如 6×RTX 3090 24GB | 要覆盖 2/4/6 卡；单卡最大输入约 1GB，重点是卡数和互联 |
| `*_calcs` 理论题 | 无需 GPU | Markdown 推导，不运行模型 |

### 为什么不推荐 48GB 跑主实验

当前 XL 约 34.1 亿参数：完整 FP32 参数约 **12.7 GiB**，加上梯度和 AdamW 的两个状态，合计约 **50.8 GiB/卡**。这还没包含激活、Flat DDP 的拼接缓冲和优化器临时张量。因此 **2×48GB 不能代替 2×80GB**，DDP 不会合并两张卡的显存。

FSDP 虽能降低稳态存储，当前脚本仍先在每张 GPU 上创建完整模型；单卡实验也不会自动借用另一张卡的显存。80/96GB 是选型估算，尚未通过完整默认配置的实测保证。

### 按预算安排

- **希望少换机器**：租同机 2×80GB，顺序跑单卡和双卡实验；遇到单卡显存压力时再考虑 96GB。
- **希望显存余量更大**：租同机 2×96GB。额外显存不能代替互联性能，仍需确认拓扑。
- **希望减少闲置费用**：单卡实验先租一张；双卡实验再租两张；六卡通信另租 6×24GB，完成后释放。不要为小张量通信一直租六张大显存卡。

### 下单前确认

- **同一实例、同一主机、相同 GPU 型号**；双卡基线和分片实验尽量在同一硬件上完成。
- 选择明确标注资源分配、算力是否独占的实例；“vGPU-48GB”不能仅凭名字判断性能，且容量不足以运行当前 XL FP32 DDP 基线。
- 核实 NVLink / PCIe 拓扑、NCCL 多卡通信和 Nsight 支持；不要从“80GB/96GB”推断互联方式。
- 主机内存建议至少 **64GB**，有条件可选 128GB；两个进程都会先在 CPU 创建完整模型。给依赖、日志和 profiler 报告预留充足磁盘空间。
- 价格、库存以租用页面为准，不沿用脚本头部的旧参考价。先完成小模型测试，再跑 XL，保存结果后及时释放实例。

官方参考：[AutoDL 单机多卡](https://www.autodl.com/docs/distributed_training/)、[资源分配](https://www.autodl.com/docs/gpu)、[A100 80GB 规格](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf)、[RTX PRO 6000 96GB 规格](https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000/)。

先上传完整仓库及 `cs336-basics`，不复制本机 `.venv`。下面的实验按顺序运行；双卡实例上的单卡实验只使用 GPU 0。

## 1. 环境准备

将第一行改为服务器上的实际项目目录：

```bash
cd /root/autodl-tmp/assignment2-systems
uv sync --locked
set -o pipefail
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "logs/$RUN_TAG" "profiles/$RUN_TAG" "results/$RUN_TAG"
nvidia-smi | tee "logs/$RUN_TAG/gpu.txt"
nvidia-smi topo -m | tee "logs/$RUN_TAG/topology.txt"
```

当前依赖使用 CUDA 13.0 wheel，基础驱动要求为 580 系列或更新。可选用 `tmux` 保持断线后任务运行。

## 2. 正确性测试

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run pytest tests/test_ddp.py tests/test_sharded_optimizer.py tests/test_fsdp.py -q --tb=short
CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_attention.py -q --tb=short
```

## 3. 单卡实验

```bash
# 普通 attention：eager / compiled
CUDA_VISIBLE_DEVICES=0 uv run python -u -m scripts.pytorch_attention.benchmark_attn 2>&1 | tee "logs/$RUN_TAG/attention.log"

# FlashAttention：PyTorch / Triton
CUDA_VISIBLE_DEVICES=0 uv run python -u -m scripts.flash_benchmarking.benchmark_flash 2>&1 | tee "logs/$RUN_TAG/flash.log"

# 梯度检查点显存
CUDA_VISIBLE_DEVICES=0 uv run python -u -m scripts.gradient_checkpointing.benchmark_checkpoint 2>&1 | tee "logs/$RUN_TAG/checkpoint.log"
```

CSV 保存在对应脚本目录。极大 attention 配置可能记录 `oom`，按 `status` 判断是否成功。

## 4. 双卡计时与显存实验

```bash
# Naive DDP：保留文件名原有拼写 becnmark
CUDA_VISIBLE_DEVICES=0,1 uv run python -u -m scripts.naive_ddp_benchmarking.becnmark_naive_ddp 2>&1 | tee "logs/$RUN_TAG/naive-ddp.log"

# Flat DDP
CUDA_VISIBLE_DEVICES=0,1 uv run python -u -m scripts.minimal_ddp_flat_benchmarking.becnmark_flat_ddp 2>&1 | tee "logs/$RUN_TAG/flat-ddp.log"

# 普通 / 分片 AdamW：迭代耗时与显存
CUDA_VISIBLE_DEVICES=0,1 uv run python -u -m scripts.optimizer_state_sharding_accounting.benchmark_sharder_optim 2>&1 | tee "logs/$RUN_TAG/optimizer.log"
```

默认 XL、全局 batch=4、序列长 512。CSV 为各脚本目录的 `benchmark_dist.csv`，含数值结果。脚本内部已 `spawn`，**不套 `torchrun`**；固定端口 29500，不能同时启动多组。

## 5. Nsight 采集：DDP 与 FSDP

需要服务器已安装 `nsys`：

```bash
nsys --version

# 依次采集 naive、flat、overlap DDP
CUDA_VISIBLE_DEVICES=0,1 uv run nsys profile \
    --trace=cuda,nvtx --pytorch=autograd-nvtx \
    --sample=none --cpuctxsw=none \
    --output="profiles/$RUN_TAG/ddp" \
    python -u -m scripts.ddp_overlap_individual_parameters_benchmarking.benchmark_overlap_ddp \
    2>&1 | tee "logs/$RUN_TAG/nsys-ddp.log"

# 依次采集 naive、flat、overlap、fsdp
CUDA_VISIBLE_DEVICES=0,1 uv run nsys profile \
    --trace=cuda,nvtx --pytorch=autograd-nvtx \
    --sample=none --cpuctxsw=none \
    --output="profiles/$RUN_TAG/fsdp" \
    python -u -m scripts.fsdp_accounting.benchmark_fsdp \
    2>&1 | tee "logs/$RUN_TAG/nsys-fsdp.log"
```

- 下载 `profiles/<RUN_TAG>/*.nsys-rep`，用相同或更新版本的 Nsight Systems GUI 打开。
- 选预热后的迭代，查看 GPU stream 上计算 kernel 与 NCCL kernel 的时间关系；NVTX 范围本身不是完整 GPU 耗时。
- 当前脚本没有 profiler start/stop，**不加** `--capture-range=cudaProfilerApi`。
- overlap / FSDP 脚本的 CSV 只有配置，耗时从 Nsight 获取。没有 nsys 时，可去掉 `nsys profile` 及其选项，直接运行对应 Python 模块检查流程，但不会生成报告。
- 当前 FSDP 使用 BF16 计算、FP32 主分片，DDP 使用 FP32；FSDP 的 all-gather 尚无异步预取。

## 6. 多卡通信实验

六卡实例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run python -u -m scripts.distributed_communication_single_node.benchmark_dist 2>&1 | tee "logs/$RUN_TAG/communication.log"
```

只有双卡时，将可见设备改为 `0,1`，脚本会跳过 4/6 卡组。结果在 `scripts/distributed_communication_single_node/benchmark_dist.csv`。

## 7. 备份与下载

成功完成所选实验后，备份 CSV 并打包；脚本再次运行会覆盖原始 CSV，需要比较多轮时每轮都备份。

```bash
cp --parents scripts/*/*.csv "results/$RUN_TAG/"
tar -czf "cs336-results-${RUN_TAG}.tar.gz" "logs/$RUN_TAG" "profiles/$RUN_TAG" "results/$RUN_TAG"
```

下载压缩包后再释放实例。若实验失败，先看日志；注意不要把上轮残留 CSV 当作本轮结果。

补充：`scripts.benchmark_script` 当前仅运行 autocast 小模型示例；`*_calcs` 是理论题，无需 GPU；leaderboard 需按官方双 B200 benchmark 另测。
