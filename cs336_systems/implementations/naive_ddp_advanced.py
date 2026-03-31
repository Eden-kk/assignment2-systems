"""Advanced communication strategy variants for the naive DDP benchmark flow.

This module reuses the shared language-model training scaffold from
``naive_ddp.py`` and swaps in the two more advanced communication strategies:
- flat gradient communication
- overlap of backward computation with per-parameter communication
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .ddp_overlap_individual_parameters import (
        ddp_overlap_individual_parameters_on_after_backward,
        get_ddp_overlap_individual_parameters,
    )
    from .naive_ddp import (
        NaiveDDPConfig,
        _synchronize_if_needed,
        broadcast_model_from_rank0,
        build_lm_model,
        resolve_config,
        run_ddp_worker_with_strategy,
    )
    from cs336_systems.implementations.benchmarking import cross_entropy
except ImportError:
    from cs336_systems.implementations.ddp_overlap_individual_parameters import (
        ddp_overlap_individual_parameters_on_after_backward,
        get_ddp_overlap_individual_parameters,
    )
    from cs336_systems.implementations.naive_ddp import (
        NaiveDDPConfig,
        _synchronize_if_needed,
        broadcast_model_from_rank0,
        build_lm_model,
        resolve_config,
        run_ddp_worker_with_strategy,
    )
    from cs336_systems.implementations.benchmarking import cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced communication strategy variants for naive DDP.")
    parser.add_argument(
        "--strategy",
        choices=("flat", "overlap"),
        default="flat",
        help="Advanced communication strategy to run.",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--optimizer-lr", type=float, default=1e-3)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29500")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def average_gradients_flat(model: torch.nn.Module, warmup: bool, device: str) -> float | None:
    start_time, end_time = None, None
    if not warmup:
        _synchronize_if_needed(device)
        start_time = default_timer()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    flat_grads = _flatten_dense_tensors(grads)
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads /= dist.get_world_size()
    synced_grads = _unflatten_dense_tensors(flat_grads, grads)
    for grad, synced_grad in zip(grads, synced_grads):
        grad.copy_(synced_grad)
    if not warmup:
        _synchronize_if_needed(device)
        end_time = default_timer()
        return end_time - start_time
    return None


def run_ddp_flat_grad_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    local_batch: torch.Tensor,
    warmup: bool,
) -> tuple[float | None, float | None]:
    if not warmup:
        _synchronize_if_needed(str(local_batch.device))
        start_time = default_timer()
    optimizer.zero_grad(set_to_none=True)
    x = local_batch[..., :-1]
    target = local_batch[..., 1:]
    logits = model(x)
    loss = cross_entropy(logits, target)
    loss.backward()
    communication_time = average_gradients_flat(model, warmup=warmup, device=str(local_batch.device))
    optimizer.step()
    if not warmup:
        _synchronize_if_needed(str(local_batch.device))
        end_time = default_timer()
        return end_time - start_time, communication_time
    return None, None


def run_ddp_overlap_individual_parameters_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    local_batch: torch.Tensor,
    warmup: bool,
) -> tuple[float | None, float | None]:
    if not warmup:
        _synchronize_if_needed(str(local_batch.device))
        start_time = default_timer()
    optimizer.zero_grad(set_to_none=True)
    x = local_batch[..., :-1]
    target = local_batch[..., 1:]
    logits = model(x)
    loss = cross_entropy(logits, target)
    loss.backward()
    communication_time = ddp_overlap_individual_parameters_on_after_backward(
        model,
        optimizer,
        warmup=warmup,
        device=str(local_batch.device),
    )
    optimizer.step()
    if not warmup:
        _synchronize_if_needed(str(local_batch.device))
        end_time = default_timer()
        return end_time - start_time, communication_time
    return None, None


def build_ddp_flat_grad_model(config: NaiveDDPConfig) -> torch.nn.Module:
    model = build_lm_model(config)
    broadcast_model_from_rank0(model)
    return model


def build_ddp_overlap_individual_parameters_model(config: NaiveDDPConfig) -> torch.nn.Module:
    return get_ddp_overlap_individual_parameters(build_lm_model(config))


def ddp_flat_grad_worker(rank: int, config: NaiveDDPConfig) -> None:
    run_ddp_worker_with_strategy(
        rank,
        config,
        strategy_label="DDP Flat Grad",
        build_ddp_model=build_ddp_flat_grad_model,
        run_ddp_step=run_ddp_flat_grad_step,
    )


def ddp_overlap_individual_parameters_worker(rank: int, config: NaiveDDPConfig) -> None:
    run_ddp_worker_with_strategy(
        rank,
        config,
        strategy_label="DDP Overlap Individual Parameters",
        build_ddp_model=build_ddp_overlap_individual_parameters_model,
        run_ddp_step=run_ddp_overlap_individual_parameters_step,
    )


def run_ddp_flat_grad(config: NaiveDDPConfig) -> None:
    mp.spawn(
        ddp_flat_grad_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )


def run_ddp_overlap_individual_parameters(config: NaiveDDPConfig) -> None:
    mp.spawn(
        ddp_overlap_individual_parameters_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    if args.strategy == "flat":
        run_ddp_flat_grad(config)
    elif args.strategy == "overlap":
        run_ddp_overlap_individual_parameters(config)
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")


if __name__ == "__main__":
    main()
