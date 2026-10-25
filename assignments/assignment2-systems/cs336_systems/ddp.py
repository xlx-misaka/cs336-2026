from typing import Any, overload
from collections.abc import Callable
from torch import nn, optim
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
            param.grad.div_(self.world_size)
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


class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[optim.Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer = None
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.num_element_per_rank = [0] * self.world_size
        self.rank_per_param = {}
        super().__init__(params, defaults={})

    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None, **kwargs):
        assert self.optimizer is not None, "Optimizer not initialized. Call init_optimizer first."
        self.optimizer.step(closure, **kwargs)

        for param, rank in self.rank_per_param.items():
            dist.broadcast(param.data, src=rank, async_op=False)

        return closure() if closure else None

    def add_param_group(self, param_group: dict[str, Any]):
        local_params = []
        for param in param_group["params"]:
            insert_rank = self.num_element_per_rank.index(min(self.num_element_per_rank))
            self.num_element_per_rank[insert_rank] += param.numel()
            self.rank_per_param[param] = insert_rank
            if insert_rank == self.rank:
                local_params.append(param)

        if local_params:
            local_group = {}
            for k, v in param_group.items():
                if k != "params":
                    local_group[k] = v
                else:
                    local_group["params"] = local_params

            if self.optimizer:
                self.optimizer.add_param_group(local_group)
            else:
                self.optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)

        super().add_param_group(param_group)
