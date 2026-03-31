from __future__ import annotations
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.nn as nn


def require_initialized_process_group() -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed process group must be initialized before using DDPOverlapIndividualParameters."
        )


def broadcast_module_parameters(module: nn.Module, *, src: int = 0) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=src)

def _synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


class DDPOverlapIndividualParameters(nn.Module):
    """Skeleton for overlapping backward compute with per-parameter communication.

    Expected public interface from the handout:
    - ``__init__(module: nn.Module)``
    - ``forward(*inputs, **kwargs)``
    - ``finish_gradient_synchronization()``

    Suggested realization direction:
    - broadcast rank-0 parameters at construction time
    - register one autograd hook per trainable parameter
    - launch ``dist.all_reduce(..., async_op=True)`` in each hook as soon as
      that parameter's gradient is ready
    - wait on all pending handles before ``optimizer.step()``
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        require_initialized_process_group()
        self.module = module
        broadcast_module_parameters(self.module)

        self.world_size = dist.get_world_size()
        self._pending_handles: list[dist.Work] = []
        self._trainable_parameters = [
            (name, param)
            for name, param in self.module.named_parameters()
            if param.requires_grad
        ]

        self._register_gradient_hooks()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _register_gradient_hooks(self) -> None:
        """Install one hook per trainable parameter.

        TODO:
        - Launch asynchronous communication when a gradient becomes available.
        - Decide whether to communicate ``grad`` or ``param.grad``.
        - Handle tied/shared parameters carefully if you need deduplication.
        - Record handles so ``finish_gradient_synchronization()`` can wait on them.
        """
        for _, param in self._trainable_parameters:

            def _hook(grad: torch.Tensor, *, _param: nn.Parameter):
                handle = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
                self._pending_handles.append(handle)

            param.register_hook(_hook)

    def finish_gradient_synchronization(self) -> None:
        """Wait for outstanding gradient communication and normalize gradients.

        - Wait on each async all-reduce handle started during backward.
        - Clear handle state before the next iteration.
        - Divide gradients by ``world_size`` if you use ``ReduceOp.SUM``.
        """

        for handle in self._pending_handles:
            handle.wait()

        self._pending_handles.clear()


def get_ddp_overlap_individual_parameters(module: nn.Module) -> nn.Module:
    """Convenience constructor for the overlap-by-parameter DDP skeleton."""

    return DDPOverlapIndividualParameters(module)


def get_ddp_individual_parameters(module: nn.Module) -> nn.Module:
    return get_ddp_overlap_individual_parameters(module)


def ddp_overlap_individual_parameters_on_after_backward(
    ddp_model: DDPOverlapIndividualParameters,
    optimizer: torch.optim.Optimizer,
    warmup: bool,
    device: str
) -> float | None:
    """Run the required post-backward synchronization step before optimizer.step()."""
    del optimizer

    start_time, end_time = None, None
    if not warmup:
        _synchronize_if_needed(device)
        start_time = default_timer()
    ddp_model.finish_gradient_synchronization()
    if not warmup:
        _synchronize_if_needed(device)
        end_time = default_timer()
        return end_time - start_time


def ddp_individual_parameters_on_after_backward(
    ddp_model: DDPOverlapIndividualParameters,
    optimizer: torch.optim.Optimizer,
) -> None:
    del optimizer
    ddp_model.finish_gradient_synchronization()
