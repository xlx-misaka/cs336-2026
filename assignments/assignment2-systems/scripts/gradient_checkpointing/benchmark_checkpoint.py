import csv
from pathlib import Path

import torch

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

_OUTPUT_DIR = Path(__file__).resolve().parent

d_model = 1280
d_ff = 5120
num_heads = 20
num_layers = 36

context_length = 512
batch_size = 4

total_size_bytes = 0

device = torch.device("cuda")


def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):  # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    # print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    # print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def k_blocks(blocks, x, i, k):
    for j in range(i, i + k):
        x = blocks[j](x)
    return x


def half_block(blocks, x, i):

    def attn_only(block, x):
        return x + block.attn(block.ln1(x))

    def ffn_only(block, x):
        return x + block.ffn(block.ln2(x))

    b = blocks[i]
    x = checkpoint(attn_only, b, x, use_reentrant=False)
    x = checkpoint(ffn_only, b, x, use_reentrant=False)
    return x


def run_model_checkpoint(blocks, x, n, k):

    if k == 0.5:
        for i in range(n):
            x = half_block(blocks, x, i)
    else:
        i = 0
        while i < n:
            x = checkpoint(k_blocks, blocks, x, i, k, use_reentrant=False)
            i += k
    return x


def benchmark(n, k):

    assert k == 0.5 or n % k == 0, "n 必须是 k 的倍数"

    blocks = torch.nn.ModuleList(
        TransformerBlock(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            positional_encoder=RotaryEmbedding(
                dim=d_model // num_heads,
                context_length=context_length,
            ),
        ).to(device)
        for _ in range(n)
    )

    x = torch.randn(
        (batch_size, context_length, d_model),
        device=device,
        requires_grad=True,
    )

    optim = torch.optim.AdamW(blocks.parameters())

    warmup_iter = 5
    for _ in range(warmup_iter):
        logits = run_model_checkpoint(blocks, x, n, k)
        assert logits is not None
        logits.mean().backward()
        optim.zero_grad()
        del logits

    x.grad = None
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        logits = run_model_checkpoint(blocks, x, n, k)
        assert logits is not None
        logits.mean().backward()

    torch.cuda.synchronize()

    mem = torch.cuda.max_memory_allocated()

    print("=" * 80)
    print(f"Total saved residual memory: {total_size_bytes / 1024**2:.2f} MiB")
    print(f"n = {n}, k = {k} :{mem / 1024**2:.2f} MiB")

    return {"num_layers": n, "k": k, "peak_mem_MiB": mem / 1024**2, "saved_residual_MiB": total_size_bytes / 1024**2}


def _save_results_csv(results):
    res_path = _OUTPUT_DIR / "benchmark_checkpoint.csv"

    with open(res_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "d_model",
                "d_ff",
                "num_heads",
                "num_layers",
                "context_length",
                "batch_size",
            ]
        )
        writer.writerow(
            [
                d_model,
                d_ff,
                num_heads,
                num_layers,
                context_length,
                batch_size,
            ]
        )

        writer.writerow([])

        writer.writerow(
            [
                "k",
                "peak_mem_MiB",
                "saved_residual_MiB",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["k"],
                    result["peak_mem_MiB"],
                    result["saved_residual_MiB"],
                ]
            )

    print(f"Saved results to {res_path}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行")

    results = []

    for k in [0.5, 1, 2, 3, 4, 6]:
        total_size_bytes = 0
        results.append(benchmark(num_layers, k))

    _save_results_csv(results)

    print("=" * 80)
