from typing import Any

from cs336_basics.model import Linear, Embedding, RMSNorm
from dataclasses import dataclass
from torch import nn

import torch
import torch.distributed as dist

__all__ = ["FSDP"]


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

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full_params: dict[str, torch.Tensor] = {}

        for name, param in self.module.named_parameters():
            if param in self._shard_info_per_param:
                shard_info = self._shard_info_per_param[param]
                numel = shard_info.numel
                shape = shard_info.shape
                master = shard_info.shard_master

                param_data = master.clone()

                slices = [torch.zeros_like(param_data) for _ in range(self.world_size)]
                dist.all_gather(slices, param_data)
                full_data = torch.cat(slices, dim=0)[:numel].view(shape).contiguous()
                full_params[name] = full_data

            else:
                full_params[name] = param.data.to(torch.float32).clone()

        return full_params
