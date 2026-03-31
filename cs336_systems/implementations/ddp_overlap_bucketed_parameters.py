from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def require_initialized_process_group() -> None:
    """Validate that ``torch.distributed`` is ready before wrapping a model.

    - Keep this guard if you want a friendly early error.
    - Adjust the exact validation/error message if your implementation needs it.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed process group must be initialized before using DDPOverlapBucket.")


def broadcast_module_parameters(module: nn.Module, *, src: int = 0) -> None:
    """Synchronize rank-0 parameters to every other rank.

    - Decide whether you also want to broadcast buffers.
    - Handle tied/shared parameters carefully if you need deduplication.
    """

    for param in module.parameters():
        dist.broadcast(param.data, src=src)


@dataclass(frozen=True)
class BucketAssignment:
    bucket_index: int
    parameter_names: tuple[str, ...]
    parameter_indices: tuple[int, ...]
    size_bytes: int


def bucket_parameters(module: nn.Module, bucket_size_mb: float | None) -> list[BucketAssignment]:
    """Group trainable parameters into communication buckets.

    - Implement the bucketing policy you want to benchmark.
    - Preserve the parameter order expected by your backward hooks.
    - Decide how to handle shared/tied parameters.
    """
    # no use of buckets
    if bucket_size_mb is None:
        trainable = [
            (name, param_index, param)
            for param_index, (name, param) in enumerate(module.named_parameters())
            if param.requires_grad
        ]
        total_size = sum(param.numel() * param.element_size() for _, _, param in trainable)
        return [
            BucketAssignment(
                bucket_index=0,
                parameter_names=tuple(name for name, _, _ in trainable),
                parameter_indices=tuple(i for _, i, _ in trainable),
                size_bytes=total_size,
            )
        ]
    
    bucket_size_limit = bucket_size_mb * 1024 * 1024
    buckets = []
    current_bucket_index = 0
    current_bucket_size = 0
    current_bucket_parameter_names = []
    current_bucket_parameter_indices = []

    # use buckets
    for i, (name, param) in enumerate(module.named_parameters()):
        if not param.requires_grad:
            continue
        param_size = param.numel() * param.element_size()
        if current_bucket_size + param_size > bucket_size_limit:
            # add current bucket
            bucket = BucketAssignment(
                bucket_index=current_bucket_index, 
                parameter_names=tuple(current_bucket_parameter_names), 
                parameter_indices=tuple(current_bucket_parameter_indices),
                size_bytes=current_bucket_size,
            )
            buckets.append(bucket)

            # reset current bucket
            current_bucket_index += 1
            current_bucket_size = 0
            current_bucket_parameter_names = []
            current_bucket_parameter_indices = []
        
        current_bucket_size += param_size
        current_bucket_parameter_names.append(name)
        current_bucket_parameter_indices.append(i)

    # add the last bucket
    if current_bucket_size > 0:
        bucket = BucketAssignment(
            bucket_index=current_bucket_index, 
            parameter_names=tuple(current_bucket_parameter_names), 
            parameter_indices=tuple(current_bucket_parameter_indices),
            size_bytes=current_bucket_size,
        )
        buckets.append(bucket)

    return buckets


class DDPOverlapBucket(nn.Module):
    """Skeleton for overlapping backward computation with bucketed communication.

    Required public interface:
    - ``__init__(module: nn.Module, bucket_size_mb: float | None)``
    - ``forward(*inputs, **kwargs)``
    - ``on_train_batch_start()``
    - ``finish_gradient_synchronization()``
    """

    def __init__(self, module: nn.Module, *, bucket_size_mb: float | None) -> None:
        super().__init__()
        require_initialized_process_group()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.world_size = dist.get_world_size()

        broadcast_module_parameters(self.module)

        self.buckets: list[BucketAssignment] = bucket_parameters(self.module, bucket_size_mb)
        self._parameters = list(self.module.parameters())
        self._pending_handles: list[tuple[int, dist.Work]] = []

        # - Replace these placeholders with the exact bucket state you need.
        # - Track which grads in each bucket are ready.
        # - Track which buckets have already launched communication this step.
        # - Track any flattened buffers you need to unflatten later.
        self._bucket_ready_parameter_indices: dict[int, set[int]] = {
            bucket.bucket_index: set() for bucket in self.buckets
        }

        self._bucket_flat_buffers: dict[int, torch.Tensor] = {}

        self._register_bucket_hooks()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _register_bucket_hooks(self) -> None:
        """Install hooks that can trigger async communication when a bucket is ready.

        - Register one hook per trainable parameter.
        - Mark that parameter as ready in its bucket.
        - Detect when the bucket is fully ready.
        - Flatten/pack the bucket gradients.
        - Launch ``dist.all_reduce(..., async_op=True)`` for the bucket.
        - Store handles so ``finish_gradient_synchronization()`` can wait on them.
        """

        for bucket in self.buckets:
            for parameter_index in bucket.parameter_indices:
                param = self._parameters[parameter_index]
                if not param.requires_grad:
                    continue

                def _hook(
                    grad: torch.Tensor,
                    *,
                    _bucket_index=bucket.bucket_index,
                    _parameter_index=parameter_index,
                    _bucket_parameter_indices=bucket.parameter_indices,
                ) -> torch.Tensor:
                    # - Update bucket readiness bookkeeping.
                    # - If the bucket is complete, queue async communication.
                    self._bucket_ready_parameter_indices[_bucket_index].add(_parameter_index)
                    if len(self._bucket_ready_parameter_indices[_bucket_index]) == len(self.buckets[_bucket_index].parameter_indices):
                        flattend_parameter_grads = _flatten_dense_tensors(
                            [self._parameters[i].grad for i in _bucket_parameter_indices if self._parameters[i].grad is not None]
                        )
                        self._bucket_flat_buffers[_bucket_index] = flattend_parameter_grads
                        handle = dist.all_reduce(flattend_parameter_grads, op=dist.ReduceOp.AVG, async_op=True)
                        self._pending_handles.append((_bucket_index, handle))

                    return grad

                param.register_hook(_hook)

    def on_train_batch_start(self) -> None:
        """Reset per-step bucket state before the next forward/backward pass.

        - Clear bucket readiness bookkeeping.
        - Release any flattened buffers from the previous step.
        - Reset per-bucket launch state.
        """

        for ready_indices in self._bucket_ready_parameter_indices.values():
            ready_indices.clear()
        self._bucket_flat_buffers.clear()

    def finish_gradient_synchronization(self) -> None:
        """Wait for bucket communication and restore synchronized gradients.

        - Wait on every async handle launched during backward.
        - Unflatten bucket buffers back into parameter gradients.
        - Normalize by ``world_size`` if you use sum-reduction.
        - Clear handle state before the next step.
        """

        for bucket_index, handle in self._pending_handles:
            handle.wait()
            bucket = self.buckets[bucket_index]
            grads = [
                self._parameters[i].grad 
                for i in bucket.parameter_indices
                if self._parameters[i].grad is not None
            ]
            flattened_grads = self._bucket_flat_buffers[bucket_index]
            unflattened_grads = _unflatten_dense_tensors(flattened_grads, grads)

            for grad, synced_grad in zip(grads, unflattened_grads):
                grad.copy_(synced_grad)

        self._pending_handles.clear()


def get_ddp_overlap_bucket(module: nn.Module, bucket_size_mb: float | None) -> nn.Module:
    """Convenience constructor for the bucketed-overlap skeleton."""

    return DDPOverlapBucket(module, bucket_size_mb=bucket_size_mb)


def get_ddp_bucketed(module: nn.Module, bucket_size_mb: float | None) -> nn.Module:
    return get_ddp_overlap_bucket(module, bucket_size_mb)


def ddp_overlap_bucket_on_train_batch_start(
    ddp_model: DDPOverlapBucket,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Run any per-step state reset before the next forward/backward pass."""

    del optimizer
    ddp_model.on_train_batch_start()


def ddp_bucketed_on_train_batch_start(
    ddp_model: DDPOverlapBucket,
    optimizer: torch.optim.Optimizer,
) -> None:
    ddp_overlap_bucket_on_train_batch_start(ddp_model, optimizer)


def ddp_overlap_bucket_on_after_backward(
    ddp_model: DDPOverlapBucket,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Finish synchronization before ``optimizer.step()``."""

    del optimizer
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_after_backward(
    ddp_model: DDPOverlapBucket,
    optimizer: torch.optim.Optimizer,
) -> None:
    ddp_overlap_bucket_on_after_backward(ddp_model, optimizer)
