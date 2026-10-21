import os
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size, context_lengrh):
    setup(rank, world_size)
    data = torch.randn(context_lengrh, dtype=torch.float32)
    dist.all_reduce(data, async_op=False)


def benchmark(world_size, context_length):
    spawn(fn=distributed_demo, args=(world_size, context_length), nprocs=world_size, join=True)


if __name__ == "__main__":
    for world_size in [2, 4, 6]:
        for context_length in [10, 20, 30]:
            benchmark(world_size, context_length)
