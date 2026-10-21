import argparse
import contextlib
import csv
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from torch.cuda import nvtx

_MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

_MEASUREMENT_SCOPES = {
    "eval": "forward_no_grad",
    "fwd": "forward_with_grad",
    "bwd": "forward+loss+backward",
    "train": "forward+loss+backward+optimizer_step",
}

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
_MEM_PROFILE_DIR = _OUTPUT_DIR / "mem_use_history"

_CONFIG_FIELDS = (
    "mode",
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
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark transformer latency and peak allocated CUDA memory.")

    parser.add_argument(
        "--mode",
        choices=["fwd", "bwd", "train", "eval"],
        default="eval",
        help=("eval: forward without gradients; fwd: forward with gradient recording; bwd: forward + loss + backward; train: forward + loss + backward + optimizer step"),
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
        help="cpu, cuda, or cuda:N",
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

    args = parser.parse_args(argv)

    for name in ("steps", "vocab_size", "context_length", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.warmup < 1:
        parser.error("--warmup must be at least 1")

    if not math.isfinite(args.lr) or args.lr < 0:
        parser.error("--lr must be finite and nonnegative")
    if not math.isfinite(args.rope_theta) or args.rope_theta <= 0:
        parser.error("--rope-theta must be finite and positive")

    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if device.type not in {"cpu", "cuda"}:
        parser.error("--device must be cpu, cuda, or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            parser.error(f"CUDA device {device.index} does not exist")
    elif args.annotate or args.mem_snapshot:
        parser.error("--annotate and --mem-snapshot require a CUDA device")

    for key, value in _MODEL_SIZES[args.model_size].items():
        setattr(args, key, value)

    return args


def _args_info(args) -> str:
    """Return a filesystem-safe summary of the essential benchmark arguments."""
    precision = "bf16" if args.mixed_precision else "fp32"
    execution = "compiled" if args.compile else "uncompiled"
    return "_".join(
        str(value)
        for value in (
            args.mode,
            args.model_size,
            args.context_length,
            precision,
            execution,
        )
    )


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nvtx_range(name, enabled):
    """NVTX range context manager, or a no-op when profiling is disabled."""
    return nvtx.range(name) if enabled else contextlib.nullcontext()


def _build_model(args, device) -> BasicsTransformerLM:
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


@contextlib.contextmanager
def _annotated_attention(enabled):
    """Temporarily annotate SDPA without changing later runs in this process."""
    if not enabled:
        yield
        return

    import cs336_basics.model

    from scripts.annotated_units import annotated_scaled_dot_product_attention

    with patch.object(cs336_basics.model, "scaled_dot_product_attention", annotated_scaled_dot_product_attention):
        yield


def _make_batch(args, device):
    # Generate inputs once, outside warmup and measurement.
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
    """Execute the selected scope; gradient clearing stays outside the timer."""
    with torch.set_grad_enabled(args.mode != "eval"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.mixed_precision):
            with _nvtx_range("forward", args.annotate):
                logits = model(x)

            if args.mode in {"fwd", "eval"}:
                return

            with _nvtx_range("loss", args.annotate):
                loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))

        with _nvtx_range("backward", args.annotate):
            loss.backward()

        if args.mode == "train":
            with _nvtx_range("optimizer", args.annotate):
                optimizer.step()


@contextlib.contextmanager
def _memory_history(snapshot_path):
    """Save profiling data and stop recording even when measurement fails."""
    if snapshot_path is None:
        yield
        return

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    try:
        yield
    finally:
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)


def _append_result_row(csv_path, row):
    """Extend older CSV headers while preserving existing cells and unknown fields."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
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
                    mode="w",
                    newline="",
                    encoding="utf-8",
                    dir=csv_path.parent,
                    prefix=f"{csv_path.stem}_",
                    suffix=".tmp",
                    delete=False,
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


def _save_results_csv(args, device, timestamp, summary, device_name):
    row = {
        "timestamp": timestamp,
        "measurement_scope": _MEASUREMENT_SCOPES[args.mode],
        **{name: getattr(args, name) for name in _CONFIG_FIELDS},
        "device": str(device),
        "gpu": device_name if device.type == "cuda" else "",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or ""),
        **summary,
    }
    csv_path = _OUTPUT_DIR / "benchmark_results.csv"
    _append_result_row(csv_path, row)
    return csv_path


def benchmark(argv=None):
    args = _parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)

    torch.manual_seed(args.seed)

    precision = "bf16 autocast" if args.mixed_precision else "fp32"
    execution = "compiled" if args.compile else "eager"
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"{args.model_size} | {_MEASUREMENT_SCOPES[args.mode]} | {device} ({device_name}) | batch={args.batch_size}, seq={args.context_length} | {precision}, {execution}",
        flush=True,
    )

    device_context = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
    with device_context, _annotated_attention(args.annotate):
        model = _build_model(args, device)
        model.train(args.mode != "eval")
        optimizer = AdamW(model.parameters(), lr=args.lr) if args.mode == "train" else None
        if args.compile:
            model.compile()
        x, y = _make_batch(args, device)

        with _nvtx_range("warmup", args.annotate):
            for _ in range(args.warmup):
                model.zero_grad(set_to_none=True)
                _run_step(args, device, model, x, y, optimizer)
                _synchronize(device)

        # Release warmup gradients before measuring peak allocated memory.
        model.zero_grad(set_to_none=True)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        snapshot_path = None
        if args.mem_snapshot:
            file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_path = _MEM_PROFILE_DIR / f"{_args_info(args)}_{file_timestamp}_memory_snapshot.pickle"

        step_times_ms = []
        with _memory_history(snapshot_path), _nvtx_range("benchmark", args.annotate):
            for _ in range(args.steps):
                model.zero_grad(set_to_none=True)
                _synchronize(device)
                with _nvtx_range("step", args.annotate):
                    start = time.perf_counter()
                    _run_step(args, device, model, x, y, optimizer)
                    _synchronize(device)
                    step_times_ms.append((time.perf_counter() - start) * 1000)

            mem_peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None

    summary = {
        "mean_ms": statistics.mean(step_times_ms),
        "median_ms": statistics.median(step_times_ms),
        "stddev_ms": statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0,
        "mem_peak_GiB": mem_peak_gib,
    }
    csv_path = _save_results_csv(args, device, timestamp, summary, device_name)
    memory = f" | peak allocated={mem_peak_gib:.3f} GiB" if mem_peak_gib is not None else ""
    print(f"Latency: mean={summary['mean_ms']:.3f} ms | median={summary['median_ms']:.3f} ms | std={summary['stddev_ms']:.3f} ms{memory}")
    print(f"CSV: {csv_path}")
    if snapshot_path is not None:
        print(f"Memory snapshot: {snapshot_path}")
    return summary


if __name__ == "__main__":
    benchmark()
