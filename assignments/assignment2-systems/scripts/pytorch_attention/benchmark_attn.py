import csv
import gc
from pathlib import Path
import statistics
import timeit

from cs336_basics import model
import torch


_OUTPUT_DIR = Path(__file__).resolve().parent

_STEP = 100


def benchmark(d_model, context_length, batch_size=8, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    if device.type == "cpu":
        raise ValueError("CPU not supported")

    Q = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)
    K = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)
    V = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)

    i = torch.arange(0, context_length, device=device)[:, None]
    j = torch.arange(0, context_length, device=device)[None, :]
    mask = i >= j

    def run_forward(Q, K, V, mask=None):
        return model.scaled_dot_product_attention(Q, K, V, mask)

    warmup_iter = 5
    for _ in range(warmup_iter):
        O = run_forward(Q, K, V, mask)

        O.sum().backward()

        Q.grad = None
        K.grad = None
        V.grad = None
        del O

    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []
    fwd_peak_mems = []
    fwd_use_mems = []
    bwd_peak_mems = []

    for _ in range(_STEP):
        torch.cuda.reset_peak_memory_stats()

        start = timeit.default_timer()

        O = run_forward(Q, K, V, mask)

        torch.cuda.synchronize()

        end = timeit.default_timer()

        fwd_times.append(end - start)
        fwd_peak_mems.append(torch.cuda.max_memory_allocated())
        fwd_use_mems.append(torch.cuda.memory_allocated())

        loss = O.sum()

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

        start = timeit.default_timer()

        loss.backward()

        torch.cuda.synchronize()

        end = timeit.default_timer()

        bwd_peak_mems.append(torch.cuda.max_memory_allocated())
        bwd_times.append(end - start)

        Q.grad = None
        K.grad = None
        V.grad = None

        del O, loss

        torch.cuda.synchronize()

    fwd_time_mean = statistics.mean(fwd_times)
    bwd_time_mean = statistics.mean(bwd_times)
    fwd_use_mem_mean = statistics.mean(fwd_use_mems)
    fwd_peak_mem_mean = statistics.mean(fwd_peak_mems)
    bwd_peak_mem_mean = statistics.mean(bwd_peak_mems)

    print("=" * 40 + f"d_model = {d_model} context_length = {context_length}" + "=" * 40)
    print(f"前向传播平均耗时: {fwd_time_mean * 1000:.2f} ms")
    print(f"反向传播平均耗时: {bwd_time_mean * 1000:.2f} ms")
    print(f"前向传播平均占用内存: {fwd_use_mem_mean / 1024**2:.2f} MiB")
    print(f"前向传播平均峰值内存: {fwd_peak_mem_mean / 1024**2:.2f} MiB")
    print(f"反向传播总平均峰值内存: {bwd_peak_mem_mean / 1024**2:.2f} MiB")

    return {
        "d_model": d_model,
        "context_length": context_length,
        "batch_size": batch_size,
        "forward_ms": fwd_time_mean * 1000,
        "backward_ms": bwd_time_mean * 1000,
        "before_backward_MiB": fwd_use_mem_mean / 1024**2,
        "forward_peak_MiB": fwd_peak_mem_mean / 1024**2,
        "backward_phase_peak_MiB": bwd_peak_mem_mean / 1024**2,
        "status": "OK",
    }


def _save_results_csv(results):
    fieldnames = [
        "d_model",
        "context_length",
        "batch_size",
        "forward_ms",
        "backward_ms",
        "before_backward_MiB",
        "forward_peak_MiB",
        "backward_phase_peak_MiB",
        "status",
    ]

    res_path = _OUTPUT_DIR / "benchmark_attn.csv"

    with open(res_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 80)
    print(f"Saved results to {res_path}")


if __name__ == "__main__":
    d_models = [16, 32, 64, 128]
    context_lengths = [256, 1024, 4096, 8192, 16384]

    results = []

    for d_model in d_models:
        for context_length in context_lengths:
            try:
                results.append(benchmark(d_model, context_length))

            except torch.cuda.OutOfMemoryError:
                print("=" * 40 + f"d_model = {d_model} context_length = {context_length}" + "=" * 40)
                print("CUDA 显存不足")

                results.append(
                    {
                        "d_model": d_model,
                        "context_length": context_length,
                        "batch_size": 8,
                        "forward_ms": "",
                        "backward_ms": "",
                        "before_backward_MiB": "",
                        "forward_peak_MiB": "",
                        "backward_phase_peak_MiB": "",
                        "status": "OOM",
                    }
                )

            gc.collect()
            torch.cuda.empty_cache()

    _save_results_csv(results)

    print("=" * 40 + "所有测试完成!" + "=" * 40)
