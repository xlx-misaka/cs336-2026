import torch

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

d_model = 768
d_ff = 3072
num_heads = 12
num_layers = 12

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
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t


def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
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
    print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行")

    for k in [0.5, 1, 2, 3]:
        total_size_bytes = 0
        benchmark(n=num_layers, k=k)
