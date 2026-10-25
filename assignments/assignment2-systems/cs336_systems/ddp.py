from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist


class OverLapDDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        def all_reduce_hook(param):
            if param.grad is None:
                return
            param.grad = param.grad / self.world_size
            self.handles.append(dist.all_reduce(param.grad, async_op=True))

        for param in self.parameters():
            dist.broadcast(param.data, src=0)
            if not param.requires_grad:
                continue
            param.register_post_accumulate_grad_hook(all_reduce_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


class FlatDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        for param in self.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        grads = []
        for param in self.parameters():
            if param.grad is None:
                continue
            param.grad = param.grad / self.world_size
            grads.append(param.grad)
        grads_flat = _flatten_dense_tensors(grads)
        dist.all_reduce(grads_flat, op=dist.ReduceOp.SUM, async_op=False)
        grads = _unflatten_dense_tensors(grads_flat, grads)
        it = 0
        for param in self.parameters():
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

    def finish_gradient_synchronization(self):
        jobs = []
        for param in self.parameters():
            if param.grad is None:
                continue
            param.grad = param.grad / self.world_size
            jobs.append(dist.all_reduce(param.grad, async_op=True))
        for job in jobs:
            job.wait()
