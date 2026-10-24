from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist


class FlatDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        for param in self.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self, optimizer: torch.optim.Optimizer):
        grads = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.grad = param.grad / self.world_size
                grads.append(param.grad)
        grads_flat = _flatten_dense_tensors(grads)
        job = dist.all_reduce(grads_flat, op=dist.ReduceOp.SUM, async_op=True)
        if job is not None:
            job.wait()
        grads = _unflatten_dense_tensors(grads_flat, grads)
        it = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.grad = grads[it]
                it += 1


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        for param in self.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self, optimizer: torch.optim.Optimizer):
        jobs = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.grad = param.grad / self.world_size
                jobs.append(dist.all_reduce(param.grad, async_op=True))
        for job in jobs:
            job.wait()
