"""Measure peak memory once per checkpoint granularity using the course xl config."""

import csv
import gc
from pathlib import Path

import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

D_MODEL = 2560
D_FF = 10240
NUM_LAYERS = 32
NUM_HEADS = 32
CONTEXT_LENGTH = 2048
BATCH_SIZE = 4
K_VALUES = [0.5, 1, 2, 3, 4, 6, 8]


def k_blocks(blocks, x, start, count):
    for i in range(start, start + count):
        x = blocks[i](x)
    return x


def half_block(blocks, x, i):
    def attention(block, x):
        return x + block.attn(block.ln1(x))

    def ffn(block, x):
        return x + block.ffn(block.ln2(x))

    x = checkpoint(attention, blocks[i], x, use_reentrant=False)
    return checkpoint(ffn, blocks[i], x, use_reentrant=False)


def run_model_checkpoint(blocks, x, k):
    if k == 0.5:
        for i in range(len(blocks)):
            x = half_block(blocks, x, i)
    else:
        k = int(k)
        for start in range(0, len(blocks), k):
            count = min(k, len(blocks) - start)
            x = checkpoint(k_blocks, blocks, x, start, count, use_reentrant=False)
    return x


def benchmark(k):
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(
        [
            TransformerBlock(
                d_model=D_MODEL,
                d_ff=D_FF,
                num_heads=NUM_HEADS,
                positional_encoder=RotaryEmbedding(dim=D_MODEL // NUM_HEADS, context_length=CONTEXT_LENGTH),
            ).to("cuda")
            for _ in range(NUM_LAYERS)
        ]
    )
    x = torch.randn(BATCH_SIZE, CONTEXT_LENGTH, D_MODEL, device="cuda", requires_grad=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    output = run_model_checkpoint(blocks, x, k)
    assert output is not None
    output.mean().backward()
    torch.cuda.synchronize()
    # Total allocated peak includes parameters, gradients and temporary activations.
    return {"peak_mem_MiB": torch.cuda.max_memory_allocated() / 1024**2, "status": "ok"}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    gpu = torch.cuda.get_device_name()
    path = Path(__file__).with_name(f"{gpu}_benchmark_checkpoint.csv")
    print(f"GPU: {gpu}\nResults: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["gpu", "k", "peak_mem_MiB", "status"])
        writer.writeheader()
        for k in K_VALUES:
            row = {"gpu": gpu, "k": k}
            try:
                row.update(benchmark(k))
            except torch.cuda.OutOfMemoryError:
                row.update(peak_mem_MiB="OOM", status="oom")
            gc.collect()
            torch.cuda.empty_cache()
            writer.writerow(row)
            file.flush()
            print(f"k={k}: peak={row['peak_mem_MiB']} MiB ({row['status']})", flush=True)


if __name__ == "__main__":
    main()
