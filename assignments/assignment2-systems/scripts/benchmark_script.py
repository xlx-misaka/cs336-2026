import argparse
import contextlib
import csv
import statistics
import timeit
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F

from datetime import datetime
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


# Table 1 of the assignment handout.
_MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

_MEASUREMENT_SCOPES = {
    "eval": "forward_no_grad",
    "fwd": "forward_with_grad",
    "bwd": "forward+loss+backward",
    "train": "forward+loss+backward+optimizer_step",
}

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_MEM_PROFILE_DIR = _OUTPUT_DIR / "mem_use_history"
_MEM_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

_TIME_INFO = datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["fwd", "bwd", "train", "eval"],
        default="eval",
        help=(
            "eval: forward without gradients; fwd: forward with gradient recording; "
            "bwd: forward + loss + backward; train: forward + loss + backward + optimizer step"
        ),
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(_MODEL_SIZES),
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
        "--annotate",
        action="store_true",
        help="emit NVTX ranges and swap in the annotated SDPA (use when profiling with nsys)",
    )
    parser.add_argument(
        "--mem-snapshot",
        action="store_true",
        help="record a CUDA memory snapshot (adds profiling overhead to timings)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="use mixed precision for the forward pass",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="use torch.compile",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    if args.warmup < 1:
        raise ValueError("--warmup must be at least 1")

    for key, value in _MODEL_SIZES[args.model_size].items():
        setattr(args, key, value)

    return args


def _args_info(args) -> str:
    """Return a filesystem-safe summary of the essential benchmark arguments."""
    precision = "bf16" if args.mixed_precision else "fp32"
    compile = "compiled" if args.compile else "uncompiled"
    return "_".join(
        str(value)
        for value in (
            args.mode,
            args.model_size,
            args.context_length,
            precision,
            compile,
        )
    )


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nvtx_range(name, enabled):
    """NVTX range context manager, or a no-op when profiling is disabled."""
    return nvtx.range(name) if enabled else contextlib.nullcontext()


def _build_model(args, device):
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


def _annotate_units():
    import cs336_basics.model
    from scripts.annotated_units import annotated_scaled_dot_product_attention

    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def _make_batch(args, device):
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


def _run_step(args, device, model, x, y, optimizer):

    cast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.mixed_precision else contextlib.nullcontext()
    grad_ctx = torch.no_grad() if args.mode == "eval" else contextlib.nullcontext()

    with _nvtx_range("forward", args.annotate), cast_ctx, grad_ctx:
        logits = model(x)

    if args.mode == "fwd" or args.mode == "eval":
        return

    with _nvtx_range("loss", args.annotate), cast_ctx:
        loss = F.cross_entropy(
            logits.reshape(-1, args.vocab_size),
            y.reshape(-1),
        )

    with _nvtx_range("backward", args.annotate):
        loss.backward()

    if args.mode == "bwd":
        return

    with _nvtx_range("optimizer", args.annotate):
        optimizer.step()


def _save_mem_history_info(args):
    # Save a pickle file to be loaded by PyTorch's online tool.
    snapshot_path = _MEM_PROFILE_DIR / f"{_args_info(args)}_{_TIME_INFO}_memory_snapshot.pickle"

    torch.cuda.memory._dump_snapshot(str(snapshot_path))

    print("=" * 80)
    print(f"memory_used_history: {snapshot_path}")


def _append_result_row(csv_path, fieldnames, row):
    """Extend older CSV headers while preserving existing cells and unknown fields."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    if not write_header:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames
            if not existing_fields:
                raise ValueError(f"CSV has no readable header: {csv_path}")

            if set(fieldnames).issubset(existing_fields):
                fieldnames = existing_fields
                existing_rows = None
            else:
                existing_rows = list(reader)

        if existing_rows is not None:
            if any(None in existing_row for existing_row in existing_rows):
                raise ValueError(f"CSV contains rows wider than its header: {csv_path}")

            fieldnames = list(fieldnames) + [name for name in existing_fields if name not in fieldnames]
            temp_path = None
            try:
                with NamedTemporaryFile(
                    mode="w", newline="", encoding="utf-8", dir=csv_path.parent,
                    prefix=f"{csv_path.stem}_", suffix=".tmp", delete=False,
                ) as f:
                    temp_path = Path(f.name)
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                    writer.writerow(row)

                temp_path.replace(csv_path)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            return

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_results_csv(args, device, step_times, mem_peak_GiB):
    mean_s = statistics.mean(step_times)
    median_s = statistics.median(step_times)
    stdv_s = statistics.stdev(step_times) if len(step_times) > 1 else 0.0

    csv_path = _OUTPUT_DIR / "benchmark_results.csv"

    fieldnames = [
        "timestamp",
        "mode",
        "measurement_scope",
        "model_size",
        "d_model",
        "d_ff",
        "num_layers",
        "num_heads",
        "context_length",
        "batch_size",
        "vocab_size",
        "rope_theta",
        "lr",
        "seed",
        "mixed_precision",
        "compile",
        "mem_snapshot",
        "annotate",
        "warmup",
        "steps",
        "device",
        "gpu",
        "torch_version",
        "cuda_version",
        "mean_ms",
        "median_ms",
        "stddev_ms",
        "mem_peak_GiB",
    ]

    row = {
        "timestamp": _TIME_INFO,
        "mode": args.mode,
        "measurement_scope": _MEASUREMENT_SCOPES[args.mode],
        "model_size": args.model_size,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "vocab_size": args.vocab_size,
        "rope_theta": args.rope_theta,
        "lr": args.lr,
        "seed": args.seed,
        "mixed_precision": args.mixed_precision,
        "compile": args.compile,
        "mem_snapshot": args.mem_snapshot,
        "annotate": args.annotate,
        "warmup": args.warmup,
        "steps": args.steps,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or ""),
        "mean_ms": mean_s * 1000,
        "median_ms": median_s * 1000,
        "stddev_ms": stdv_s * 1000,
        "mem_peak_GiB": mem_peak_GiB,
    }

    _append_result_row(csv_path, fieldnames, row)

    print("=" * 80)
    print(f"results saved: {csv_path}")


def benchmark():
    args = _parse_args()

    device = torch.device(args.device)

    torch.manual_seed(args.seed)

    if args.annotate:
        _annotate_units()

    model = _build_model(args, device)

    if args.mode == "eval":
        model.eval()
    else:
        model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.compile:
        model = torch.compile(model)

    x, y = _make_batch(args, device)

    # warm-up：不记录时间，单独打 warmup 标签，方便在 nsys 里滤掉
    with _nvtx_range("warmup", args.annotate):
        for _ in range(args.warmup):
            _run_step(args, device, model, x, y, optimizer)
            _synchronize(device)
            optimizer.zero_grad(set_to_none=True)

    step_times = []

    # 开始前同步，避免上一个 step 的异步 CUDA 工作混入当前计时
    _synchronize(device)

    # 正式测量：benchmark 包住全部测量步，每个 step 再单独一个 range。

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

        if args.mem_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

    print("=" * 80)
    print(f"Benchmark start! mode={args.mode}, scope={_MEASUREMENT_SCOPES[args.mode]}")

    with _nvtx_range("benchmark", args.annotate):
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)

            with _nvtx_range("step", args.annotate):
                start = timeit.default_timer()

                _run_step(args, device, model, x, y, optimizer)

                # 题目要求：每个 step 后同步
                _synchronize(device)

                elapsed = timeit.default_timer() - start

            step_times.append(elapsed)

    mem_peak_gib = None
    if device.type == "cuda":
        mem_peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3

        if args.mem_snapshot:
            _save_mem_history_info(args)
            torch.cuda.memory._record_memory_history(enabled=None)

    _save_results_csv(args, device, step_times, mem_peak_gib)

    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    benchmark()
