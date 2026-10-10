import argparse
import contextlib
import statistics
import timeit

import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


# Table 1 of the assignment handout.
MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["forward", "backward", "full"],
        default="forward",
        help="Benchmark mode",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_SIZES),
        default="small",
        help="Table 1 preset",
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-grad",
        action="store_true",
        help="run --mode forward as inference under torch.no_grad()",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="emit NVTX ranges and swap in the annotated SDPA (use when profiling with nsys)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="use mixed precision for the forward pass",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    for key, value in MODEL_SIZES[args.model_size].items():
        setattr(args, key, value)

    return args


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nvtx_range(name, enabled):
    """NVTX range context manager, or a no-op when profiling is disabled."""
    return nvtx.range(name) if enabled else contextlib.nullcontext()


def build_model(args, device):
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    return model


def make_batch(args, device):
    # batch 生成不计入计时，因此只生成一次并复用
    x = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    y = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    return x, y


def run_step(args, model, x, y, optimizer):

    cast_ctx = torch.autocast(device_type=args.device, dtype=torch.bfloat16) if args.mixed_precision else contextlib.nullcontext()

    if args.mode == "forward":
        # forward-only 既可以测带 autograd 图的训练前向，也可以测纯推理（--no-grad）
        grad_ctx = torch.no_grad() if args.no_grad else contextlib.nullcontext()
        with grad_ctx, nvtx_range("forward", args.annotate), cast_ctx:
            model(x)
        return

    with nvtx_range("forward", args.annotate), cast_ctx:
        logits = model(x)

    with nvtx_range("loss", args.annotate), cast_ctx:
        loss = F.cross_entropy(
            logits.reshape(-1, args.vocab_size),
            y.reshape(-1),
        )

    with nvtx_range("backward", args.annotate):
        loss.backward()

    if args.mode == "full":
        with nvtx_range("optimizer", args.annotate):
            optimizer.step()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.annotate:
        import cs336_basics.model
        from cs336_systems.annotated_units import annotated_scaled_dot_product_attention

        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model = build_model(args, device)
    if args.mode == "forward" and args.no_grad:
        model.eval()
    else:
        model.train()

    # forward-only 其实用不到 optimizer，但保持流程统一
    optimizer = AdamW(model.parameters(), lr=args.lr)

    x, y = make_batch(args, device)

    # warm-up：不记录时间，单独打 warmup 标签，方便在 nsys 里滤掉
    with nvtx_range("warmup", args.annotate):
        for _ in range(args.warmup):
            run_step(args, model, x, y, optimizer)
            synchronize(device)

    step_times = []

    # 开始前同步，避免上一个 step 的异步 CUDA 工作混入当前计时
    synchronize(device)

    # 正式测量：benchmark 包住全部测量步，每个 step 再单独一个 range。
    # 这样 --filter-nvtx step 拿到的恰好是一个完整训练步。
    with nvtx_range("benchmark", args.annotate):
        for _ in range(args.steps):
            with nvtx_range("step", args.annotate):
                optimizer.zero_grad(set_to_none=True)

                start = timeit.default_timer()

                run_step(args, model, x, y, optimizer)

                # 题目要求：每个 step 后同步
                synchronize(device)

                elapsed = timeit.default_timer() - start

            step_times.append(elapsed)

    mean_s = statistics.mean(step_times)
    median_s = statistics.median(step_times)
    stdv_s = statistics.stdev(step_times) if len(step_times) > 1 else 0

    print(f"mode: {args.mode}")
    print(f"device: {device}")
    print(f"model_size: {args.model_size}")
    print(f"context_length: {args.context_length}")
    print(f"mean: {mean_s * 1000:.3f} ms/step")
    print(f"median: {median_s * 1000:.3f} ms/step")
    print(f"stddev: {stdv_s * 1000:.3f}")


if __name__ == "__main__":
    main()
