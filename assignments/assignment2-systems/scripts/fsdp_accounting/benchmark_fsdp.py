r"""Profile XL FSDP training and weight all-gathers on one node with two GPUs.

当前配置与实现：
- 单节点、2 GPU、xl、AdamW；全局 batch=4，每卡 batch=2，序列长 512，词表 10000。
- 默认依次运行 naive、flat、overlap、fsdp，每种实现预热 5 步，再运行 10 个带阶段标记的正式迭代。
- 三种 DDP 路径使用 FP32；FSDP 指定 compute_dtype=torch.bfloat16，并在前向启用 BF16 autocast。
- 预热与正式迭代均使用相同精度设置；loss 显式转为 FP32 计算，backward 位于 autocast 外。
- FSDP hooks 会替换参数 data，autocast 禁用权重缓存，避免缓存跨越参数聚集/分片。
- 当前 FSDP 对 Linear/Embedding 权重分片，长期保存 FP32 主分片；权重聚集和计算使用 BF16。
  梯度转为 FP32 后 reduce-scatter，AdamW 更新本地主分片；RMSNorm 保留完整 FP32 副本，
  其梯度使用 all-reduce。AdamW 的状态随它接收到的参数分片创建。
- 当前权重 all-gather 在本层前向/反向开始前同步执行，没有提前两层的异步预取。
  因此需要用 trace 检查暴露的通信等待，不能预先声称通信已被前一层计算隐藏。

AutoDL 租卡建议：
- 在同一个实例内租用 2 张相同 GPU，满足题目的单节点双卡条件。
  优先考虑同机 2 x A100/A800 80GB；若所选主机提供 2 x RTX PRO 6000 Blackwell 96GB，
  可以用更大的显存余量容纳临时张量，但仍需单独核实互联拓扑。
  价格、库存、实际型号和主机内存以租用页面为准，不使用固定价格估算：https://www.autodl.com/home
- 默认循环会先运行完整 DDP，因此应按 DDP 路径的需求选卡，而不是只按 FSDP 的理论分片量选卡。
  XL 约 34.1 亿参数，完整 FP32 参数约 12.7 GiB；完整参数、梯度和 AdamW 两个状态合计
  约 50.8 GiB/卡，尚未包含激活、flat 拼接缓冲和优化器临时张量。
  80/96GB 是容量建议，不是实测可运行保证；24/32/48GB 不适合默认循环中的 FP32 DDP 路径。
- 即使只采集 FSDP，当前脚本也先将完整 FP32 模型搬到每张 GPU，再进行分片。
  启动峰值与稳定训练峰值不同，不能依据分片后的理论容量直接推断小显存卡一定可用。
- 优先选择互联较好的同机双卡，记录 nvidia-smi topo -m；大显存或同机不等于一定有 NVLink。
  不同卡型和互联会改变 all-gather 延迟，报告应注明 GPU 型号、拓扑及软件版本。
- 建议至少 64GB 主机内存，并给 profiler 报告预留磁盘空间。两个进程都会先在 CPU 创建完整模型。
  先完成小模型正确性检查，再短时租用双卡验证 XL 的显存和采集流程，保存报告后及时停止实例。
- 平台说明：https://www.autodl.com/docs/distributed_training/
  资源分配说明：https://www.autodl.com/docs/gpu
  RTX PRO 6000 规格（NVIDIA PDF）：
  https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf

运行与 Nsight Systems 采集（均在仓库 assignment2-systems 根目录执行）：
    uv sync
    nvidia-smi
    nvidia-smi topo -m
    nsys --version

直接运行，脚本内部 spawn 两个进程，无需再套 torchrun：
    CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python -m scripts.fsdp_accounting.benchmark_fsdp

采集默认四种实现，报告中通过 rank 和实现类型区分：
    mkdir -p profiles
    CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync nsys profile \
        --trace=cuda,nvtx \
        --sample=none --cpuctxsw=none \
        --output=profiles/fsdp_accounting \
        python -m scripts.fsdp_accounting.benchmark_fsdp

先用小模型验证完整前向、反向与优化器更新：
    CS336_MODEL_SIZE=small CS336_DDP_TYPES=fsdp CUDA_VISIBLE_DEVICES=0,1 \
        uv run --no-sync python -m scripts.fsdp_accounting.benchmark_fsdp

仅研究题目 (b) 时，在上面的采集命令前设置 CS336_DDP_TYPES=fsdp。
CS336_MODEL_SIZE 默认 xl；CS336_DDP_TYPES 默认 naive,flat,overlap,fsdp。
上述 nsys 参数兼容当前不支持 --pytorch 的安装；采集程序已有的阶段 NVTX，
不自动生成 PyTorch 算子级标记。
代码没有调用 profiler start/stop，使用全程采集，不添加 --capture-range=cudaProfilerApi。
将 profiles/fsdp_accounting.nsys-rep 下载到本机，用相同或更新版本的 Nsight Systems GUI 打开。
Nsight 官方说明：https://developer.nvidia.com/nsight-systems/get-started
https://docs.nvidia.com/nsight-systems/UserGuide/

Problem (fsdp_accounting) 答题分析要点（用于整理自己的推导和实测，不是已完成的答案）：
(a) 题目要求基于 Section 6 估算峰值显存节省，并用 2-3 句话说明；允许忽略权重 all-gather
    的预分配缓冲区，不强制要求额外编写内存测量脚本。
    - 先明确与 Section 6 哪个设置比较、参数总数、设备数、各类数据的精度与统计单位。
    - 分别列出参数/FP32 主权重、梯度、AdamW 状态、激活，标记哪些在当前实现中被分片。
      RMSNorm 仍是副本；激活不会仅因权重分片就自动按 GPU 数缩小。
    - 区分长期 FP32 主分片与临时 BF16 计算权重，不要把 BF16 带来的节省全部归因于分片，
      也不要将整张 GPU 的峰值直接除以 world_size。忽略聚集缓冲也不等于忽略激活。
    - AdamW 状态在首次 step 时按需创建；若用实测辅助解释，应区分初始化、首次 step 与预热后稳态，
      并注明 allocated/reserved、bytes/GiB 的口径及哪些临时分配未包含在理论估计中。

(b) 题目要求在 XL、双 GPU 上观察权重 all-gather 是否及时完成，给出 2-3 句话的耗时分析和
    Nsight 截图；不能只交 NVTX 标记或尚未采集的脚本。
    - 在 rank*/fsdp/forward 中选择预热后的迭代，展开对应 GPU 的 CUDA stream，定位 NCCL
      all-gather kernel 与依赖该权重的计算 kernel，并用 CUDA API 关联信息确认对应关系。
      当前只有阶段级 NVTX，若层与通信对应关系不清晰，可增加层名和 gather 阶段标记。
    - 记录代表层的聚集耗时、与前层计算是否重叠，以及本层启动前暴露的通信等待；注明 GPU、
      精度、互联和选取的迭代。多 GPU 的截图需要清楚标出正在观察的 rank/GPU 和时间刻度。
    - 当前同步聚集保证结果可供随后计算使用，但不保证通信被隐藏。根据实际 GPU trace 描述
      gather 与 compute 的先后关系，不把 CPU 上的函数返回时间当成 GPU 通信结束时间。
    - 三条 DDP 路径为 FP32、FSDP 为 BF16；如额外比较总迭代耗时，必须说明精度差异，不能
      将速度变化全部归因于分片。题目 (b) 的重点是 FSDP 权重聚集，而非比较全部 DDP 变体。

当前输出与局限：
- benchmark_dist.csv 只保存模型配置，不保存迭代或通信耗时；本题耗时可以直接从 Nsight 获取。
- NVTX train/forward/backward 是 CPU 范围，边界没有做 GPU 同步，不可直接作为完整 GPU 耗时。
- 脚本没有采集峰值显存，头部容量数字只是选卡参考；没有实测数据时不要填写最终耗时或截图结论。
"""

import csv
import os
from pathlib import Path

from torch.multiprocessing.spawn import spawn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import NaiveDDP, OverLapDDP, FlatDDP
import torch
import torch.distributed as dist

from cs336_systems.fsdp import FSDP


path = Path(__file__).with_name("benchmark_dist.csv")
FILEDS = ["ddp_type", "device_name", "rank", "world_size", "step_time_mean_ms", "step_comm_time_mean_ms", "comm_ratio"]

model_zoo = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}
model_size = os.environ.get("CS336_MODEL_SIZE", "xl")
if model_size not in model_zoo:
    raise ValueError(f"Unknown CS336_MODEL_SIZE: {model_size!r}; choose from {list(model_zoo)}")
vocab_size = 10000
batch_size = 4
context_length = 512


def setup(rank, world_size, device):
    if device == "cuda":
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl" if device == "cuda" else "gloo", rank=rank, world_size=world_size)


def run_dist(rank, ddp_type, world_size, device, x_all, y_all, warmup_iter=5, repeate_iter=10):
    try:
        setup(rank, world_size, device)
        step = batch_size // world_size
        x = x_all[rank * step : (rank + 1) * step, :].to(device)
        y = y_all[rank * step : (rank + 1) * step, :].to(device)
        model = BasicsTransformerLM(vocab_size, context_length, **model_zoo[model_size]).to(device)
        if ddp_type == "naive":
            ddp_model = NaiveDDP(model)
        elif ddp_type == "flat":
            ddp_model = FlatDDP(model)
        elif ddp_type == "overlap":
            ddp_model = OverLapDDP(model)
        else:
            ddp_model = FSDP(model, torch.bfloat16)

        ddp_optimizer = AdamW(ddp_model.parameters())

        for _ in range(warmup_iter):
            ddp_optimizer.zero_grad()
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=(ddp_type == "fsdp"),
                cache_enabled=False,
            ):
                logits = ddp_model(x)
            loss = cross_entropy(logits.float(), y)
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            ddp_optimizer.step()

        for _ in range(repeate_iter):
            ddp_optimizer.zero_grad()
            with torch.cuda.nvtx.range(f"rank{rank}/{ddp_type}/train"):
                with torch.cuda.nvtx.range(f"rank{rank}/{ddp_type}/forward"):
                    with torch.autocast(
                        device_type=device,
                        dtype=torch.bfloat16,
                        enabled=(ddp_type == "fsdp"),
                        cache_enabled=False,
                    ):
                        logits = ddp_model(x)
                loss = cross_entropy(logits.float(), y)

                with torch.cuda.nvtx.range(f"rank{rank}/{ddp_type}/backward"):
                    loss.backward()

                with torch.cuda.nvtx.range(f"rank{rank}/{ddp_type}/wait_gradients"):
                    ddp_model.finish_gradient_synchronization()

                ddp_optimizer.step()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def benchmark(ddp_type, device, world_size=2):
    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    if device == "cuda":
        assert world_size <= torch.cuda.device_count(), "Not enough GPUs"
    x = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long)
    spawn(run_dist, args=(ddp_type, world_size, device, x, y), nprocs=world_size, join=True)


if __name__ == "__main__":
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_zoo ", "Batch Size", "Context Length", "Vocab Size"])
        writer.writerow([model_size, batch_size, context_length, vocab_size])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ddp_types = [value.strip() for value in os.environ.get("CS336_DDP_TYPES", "naive,flat,overlap,fsdp").split(",")]
    invalid_types = set(ddp_types) - {"naive", "flat", "overlap", "fsdp"}
    if invalid_types:
        raise ValueError(f"Unknown CS336_DDP_TYPES entries: {sorted(invalid_types)}")
    for ddp_type in ddp_types:
        print(f"Starting {ddp_type}: model={model_size}, world_size=2", flush=True)
        benchmark(ddp_type, device, world_size=2)
        print(f"Completed {ddp_type}", flush=True)
