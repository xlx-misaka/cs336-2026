from dataclasses import dataclass
from typing import Any, overload
from collections.abc import Callable
from torch import nn, optim
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from cs336_basics.model import Linear, Embedding, RMSNorm
import torch
import torch.distributed as dist


@dataclass
class _ShardInfo:
    shape: torch.Size
    numel: int
    shard_numel: int
    shard_master: torch.Tensor
    is_full: bool = False


class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self._shard_info_per_param: dict[nn.Parameter, _ShardInfo] = {}
        self.handles = []

        for submodule in module.modules():
            if not isinstance(submodule, (Linear, Embedding, RMSNorm)):
                continue

            param = submodule.weight
            assert isinstance(param, nn.Parameter)

            if isinstance(submodule, (Linear, Embedding)):
                self._shard_param_data(param)
                submodule.register_forward_pre_hook(self._pre_fwd_hook)
                submodule.register_forward_hook(self._after_fwd_hook)
                submodule.register_full_backward_pre_hook(self._pre_bwd_hook)

            param.register_post_accumulate_grad_hook(self._after_bwd_param_hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for submodule in self.module.modules():
            if not isinstance(submodule, (Linear, Embedding, RMSNorm)):
                continue

            param = submodule.weight
            assert isinstance(param, nn.Parameter)

    def _shard_param_data(self, param: nn.Parameter):
        origin_shape = param.shape
        origin_numel = param.numel()
        shard_numel = (origin_numel + self.world_size - 1) // self.world_size
        padd_numel = shard_numel * self.world_size

        full_data = param.data.contiguous().view(-1).to(torch.float32)
        if padd_numel > origin_numel:
            padd_data = torch.zeros(padd_numel - origin_numel, dtype=full_data.dtype, device=full_data.device)
            full_data = torch.cat([full_data, padd_data], dim=0)

        if self.world_size > 1:
            dist.broadcast(full_data, src=0)

        start = self.rank * shard_numel
        param.data = full_data[start : start + shard_numel].clone()
        self._shard_info_per_param[param] = _ShardInfo(
            shape=origin_shape,
            numel=origin_numel,
            shard_numel=shard_numel,
            shard_master=param.data,
        )

    def _gather_param_data(self, param: nn.Parameter):
        shard_info = self._shard_info_per_param[param]
        shape = shard_info.shape
        numel = shard_info.numel
        shard_numel = shard_info.shard_numel

        if shard_info.is_full:
            return

        param.data = param.data.to(dtype=self.compute_dtype or torch.float32)

        full_data = param.data
        if self.world_size > 1:
            slices = [torch.zeros(shard_numel, device=param.data.device, dtype=param.data.dtype) for _ in range(self.world_size)]
            dist.all_gather(slices, param.data)
            full_data = torch.cat(slices, dim=0)[:numel]

        full_data = full_data.view(shape).contiguous()
        param.data = full_data
        shard_info.is_full = True

    def _shard_param_grad(self, param: nn.Parameter):
        if param.grad is None:
            return

        shard_info = self._shard_info_per_param[param]
        numel = shard_info.numel
        shard_numel = shard_info.shard_numel
        padd_numel = shard_numel * self.world_size

        full_grad = param.grad.contiguous().view(-1).to(dtype=param.dtype) / self.world_size
        if padd_numel > numel:
            padd_grad = torch.zeros(padd_numel - numel, device=param.device, dtype=param.dtype)
            full_grad = torch.cat([full_grad, padd_grad])

        if self.world_size > 1:
            slices = [full_grad[i * shard_numel : (i + 1) * shard_numel] for i in range(self.world_size)]
            param.grad = torch.zeros(shard_numel, device=param.device, dtype=param.dtype)
            self.handles.append(dist.reduce_scatter(param.grad, slices, async_op=True))
        else:
            param.grad = full_grad

    def _pre_fwd_hook(self, module: nn.Module, *args: Any):
        assert isinstance(module, (Linear, Embedding))
        param = module.weight
        assert isinstance(param, nn.Parameter)

        self._gather_param_data(param)

    def _after_fwd_hook(self, module: nn.Module, *args: Any):
        assert isinstance(module, (Linear, Embedding))
        param = module.weight
        assert isinstance(param, nn.Parameter)

        shard_info = self._shard_info_per_param[param]
        shard_info.is_full = False
        param.data = shard_info.shard_master

    def _pre_bwd_hook(self, module: nn.Module, *args: Any):
        assert isinstance(module, (Linear, Embedding))
        param = module.weight
        assert isinstance(param, nn.Parameter)

        self._gather_param_data(param)

    def _after_bwd_param_hook(self, param: nn.Parameter):

        if param in self._shard_info_per_param:
            shard_info = self._shard_info_per_param[param]
            shard_info.is_full = False
            param.data = shard_info.shard_master
            self._shard_param_grad(param)

        elif param.grad is not None:
            param.grad = param.grad / self.world_size
            self.handles.append(dist.all_reduce(param.grad, async_op=True))


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
        loss = None
        if self.optimizer is not None:
            loss = self.optimizer.step(closure, **kwargs)

        for param, rank in self.rank_per_param.items():
            dist.broadcast(param.data, src=rank, async_op=False)

        return loss

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
