from __future__ import annotations

from collections.abc import Iterable
from typing import Type

import torch
import torch.distributed as dist


def require_initialized_process_group() -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed process group must be initialized before using ShardedOptimizer.")


class ShardedOptimizer(torch.optim.Optimizer):
    """Skeleton wrapper for optimizer-state sharding."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ) -> None:
        require_initialized_process_group()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.optimizer_cls = optimizer_cls

        defaults = dict(kwargs)
        # Keep the superclass initialization explicit as required by the handout.
        self.local_param_groups = list()
        # - Decide how you want to represent all parameters / parameter groups.
        # - Implement add_param_group(...) so the superclass constructor can use it.
        super().__init__(params, defaults)

        # - Partition parameters across ranks.
        # - Construct the wrapped optimizer only on the local shard.
        # - Track whatever mapping you need to synchronize updated parameter shards.

        self._local_optimizer = optimizer_cls(self.local_param_groups, **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        # - Zero only the optimizer state / gradients owned by this rank.
        self._local_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Take a local optimizer step, then synchronize updated parameter shards.

        - Call the wrapped optimizer's step() on the local shard.
        - Broadcast or all-gather updated parameter shards so all ranks end with
          the same model weights.
        """

        loss = self._local_optimizer.step(closure=closure)

        for param_group in self.param_groups:
            for i, param in enumerate(param_group["params"]):
                src = i % self.world_size
                dist.broadcast(param.data, src=src)

    def add_param_group(self, param_group: dict[str, object]) -> None:
        """Add a new parameter group and assign its parameters across ranks.

        - Decide how to shard the incoming parameters across ranks.
        - Add only this rank's shard to the wrapped/local optimizer.
        - Preserve the public Optimizer.param_groups structure expected by PyTorch.
        """
        params = list(param_group["params"])
        global_param_group = dict(param_group)
        global_param_group["params"] = params
        self.param_groups.append(global_param_group)

        metadata = {k: v for k, v in param_group.items() if k != "params"}

        local_params = [param for i, param in enumerate(params) if i % self.world_size == self.rank]

        local_param_group = dict(metadata)
        local_param_group["params"] = local_params

        self.local_param_groups.append(local_param_group)
        

def get_sharded_optimizer(
    params,
    optimizer_cls: Type[torch.optim.Optimizer],
    **kwargs,
) -> torch.optim.Optimizer:
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
