"""Naive DDP correctness script using the same language model as the benchmark.

This script mirrors the data-parallel training flow from the assignment:
- rank 0 initializes the model
- parameters are broadcast to all other ranks
- each rank trains on a disjoint shard of the same global batch
- gradients are all-reduced across ranks after backward
- a single-process reference model trains on the full batch
- the final weights are compared on rank 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from timeit import default_timer
from statistics import mean

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs336_systems.implementations.benchmarking import (
    BenchmarkConfig,
    build_model,
    build_optimizer,
    cross_entropy,
    make_batch,
)


@dataclass(frozen=True)
class NaiveDDPConfig:
    world_size: int
    backend: str
    device: str
    model_size: str
    context_length: int
    batch_size: int
    vocab_size: int
    rope_theta: float
    warmup_steps: int
    num_steps: int
    optimizer_lr: float
    master_addr: str
    master_port: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive DDP correctness check using the assignment language model.")
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


def resolve_config(args: argparse.Namespace) -> NaiveDDPConfig:
    return NaiveDDPConfig(
        world_size=args.world_size,
        backend=args.backend,
        device=args.device,
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        rope_theta=args.rope_theta,
        warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        optimizer_lr=args.optimizer_lr,
        master_addr=args.master_addr,
        master_port=args.master_port,
        seed=args.seed,
    )

def _synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def make_lm_training_config(config: NaiveDDPConfig) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_size=config.model_size,  # type: ignore[arg-type]
        context_length=config.context_length,
        batch_size=config.batch_size,
        vocab_size=config.vocab_size,
        rope_theta=config.rope_theta,
        warmup_steps=0,
        measurement_steps=0,
        mode="train-step",
        device=config.device,
        dtype=torch.float32,
        precision_autocast=None,
        optimizer_lr=config.optimizer_lr,
        memory_snapshot_path=None,
        report_peak_memory=False,
        seed=config.seed,
    )


def setup_process_group(rank: int, config: NaiveDDPConfig) -> None:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    dist.init_process_group(
        backend=config.backend,
        rank=rank,
        world_size=config.world_size,
    )


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_lm_model(config: NaiveDDPConfig) -> torch.nn.Module:
    return build_model(make_lm_training_config(config))


def build_lm_optimizer(model: torch.nn.Module, config: NaiveDDPConfig) -> torch.optim.Optimizer:
    optimizer = build_optimizer(model, make_lm_training_config(config))
    if optimizer is None:
        raise ValueError("Expected a train-step optimizer for naive DDP.")
    return optimizer


def make_token_batch(config: NaiveDDPConfig) -> torch.Tensor:
    return make_batch(make_lm_training_config(config))


def unwrap_ddp_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def broadcast_model_from_rank0(model: torch.nn.Module) -> None:
    for param in model.parameters():
        dist.broadcast(param.data, src=0, async_op=False)


def shard_batch_for_rank(
    batch: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    batch_size = batch.shape[0]
    assert batch_size % world_size == 0
    local_bs = batch_size // world_size
    start = rank * local_bs
    end = start + local_bs
    return batch[start:end]


def average_gradients_individual(model: torch.nn.Module, warmup: bool, device: str) -> float | None:
    """Synchronize each parameter gradient with its own collective call."""

    start_time, end_time = None, None
    if not warmup:
        _synchronize_if_needed(device)
        start_time = default_timer()
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= dist.get_world_size()
    if not warmup:
        _synchronize_if_needed(device)
        end_time = default_timer()
        return end_time - start_time


def run_single_process_reference_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    warmup: bool,
) -> float | None:
    start_time, end_time = None, None
    if not warmup:
        _synchronize_if_needed(str(batch.device))
        start_time = default_timer()
    optimizer.zero_grad(set_to_none=True)
    x = batch[..., :-1]
    target = batch[..., 1:]
    logits = model(x)
    loss = cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
    if not warmup:
        _synchronize_if_needed(str(batch.device))
        end_time = default_timer()
        return end_time - start_time


def run_naive_ddp_step(
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
    communication_time = average_gradients_individual(model, warmup=warmup, device=str(local_batch.device))
    optimizer.step()
    if not warmup:
        _synchronize_if_needed(str(local_batch.device))
        end_time = default_timer()
        return end_time - start_time, communication_time
    return None, None

def compare_model_parameters(reference_model: torch.nn.Module, ddp_model: torch.nn.Module, *, atol: float = 1e-5) -> None:
    ref_params = dict(reference_model.named_parameters())
    ddp_params = dict(unwrap_ddp_model(ddp_model).named_parameters())
    assert ref_params.keys() == ddp_params.keys()

    max_diff = 0.0
    max_name = None
    for name in ref_params:
        ref_param = ref_params[name]
        ddp_param = ddp_params[name]
        diff = (ref_param - ddp_param).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            max_name = name
        assert torch.allclose(ref_param, ddp_param, atol=atol), f"{name} (max diff so far: {max_diff})"

    print(f"max parameter diff: {max_diff} at {max_name}")


def print_time_consumption_report(
    gathered_ddp_step_times: list[list[float]],
    gathered_ddp_communication_times: list[list[float]],
    reference_step_times: list[float],
    config: NaiveDDPConfig,
    strategy_label: str,
) -> None:
    ddp_step_means = [mean(step_times) for step_times in gathered_ddp_step_times]
    ddp_communication_means = [mean(communication_times) for communication_times in gathered_ddp_communication_times]
    ddp_communication_fractions = [
        communication_mean / step_mean if step_mean > 0 else 0.0
        for step_mean, communication_mean in zip(ddp_step_means, ddp_communication_means)
    ]

    print(f"{strategy_label} Timing Report")
    print(
        f"backend={config.backend} "
        f"device={config.device} "
        f"model_size={config.model_size} "
        f"world_size={config.world_size} "
        f"warmup_steps={config.warmup_steps} "
        f"num_steps={config.num_steps}"
    )
    print("Per-rank measured timings:")
    for rank, (step_mean, communication_mean, communication_fraction) in enumerate(
        zip(ddp_step_means, ddp_communication_means, ddp_communication_fractions)
    ):
        print(
            f"  rank {rank}: "
            f"ddp_step_mean_ms={1000.0 * step_mean:.3f} "
            f"communication_mean_ms={1000.0 * communication_mean:.3f} "
            f"communication_fraction={100.0 * communication_fraction:.2f}%"
        )

    print(
        "Aggregate DDP: "
        f"step_mean_ms={1000.0 * mean(ddp_step_means):.3f} "
        f"communication_mean_ms={1000.0 * mean(ddp_communication_means):.3f} "
        f"communication_fraction={100.0 * mean(ddp_communication_fractions):.2f}%"
    )

    if reference_step_times:
        print(f"Reference mean step ms={1000.0 * mean(reference_step_times):.3f}")


def run_ddp_worker_with_strategy(
    rank: int,
    config: NaiveDDPConfig,
    *,
    strategy_label: str,
    build_ddp_model,
    run_ddp_step,
) -> None:
    setup_process_group(rank, config)
    try:
        if config.device.startswith("cuda"):
            torch.cuda.set_device(rank)

        ddp_model = build_ddp_model(config)
        ddp_optimizer = build_lm_optimizer(ddp_model, config)

        reference_model = None
        reference_optimizer = None
        if rank == 0:
            reference_model = build_lm_model(config)
            reference_model.load_state_dict(unwrap_ddp_model(ddp_model).state_dict())
            reference_optimizer = build_lm_optimizer(reference_model, config)

        ddp_step_times: list[float] = []
        ddp_communication_times: list[float] = []
        reference_step_times: list[float] = []

        total_steps = config.warmup_steps + config.num_steps
        for step in range(total_steps):
            torch.manual_seed(config.seed + step)
            batch = make_token_batch(config)
            local_batch = shard_batch_for_rank(batch, rank=rank, world_size=config.world_size)
            ddp_step_time, ddp_communication_time = run_ddp_step(
                ddp_model,
                ddp_optimizer,
                local_batch,
                warmup=(step < config.warmup_steps),
            )

            if rank == 0:
                step_time = run_single_process_reference_step(reference_model, reference_optimizer, batch, warmup=(step < config.warmup_steps))
            else:
                step_time = None

            if step >= config.warmup_steps:
                if ddp_step_time is not None and ddp_communication_time is not None:
                    ddp_step_times.append(ddp_step_time)
                    ddp_communication_times.append(ddp_communication_time)
                if rank == 0 and step_time is not None:
                    reference_step_times.append(step_time)

        gathered_ddp_step_times: list[list[float] | None] = [None for _ in range(config.world_size)]
        gathered_ddp_communication_times: list[list[float] | None] = [None for _ in range(config.world_size)]
        dist.all_gather_object(gathered_ddp_step_times, ddp_step_times)
        dist.all_gather_object(gathered_ddp_communication_times, ddp_communication_times)

        if rank == 0:
            print_time_consumption_report(
                [times for times in gathered_ddp_step_times if times is not None],
                [times for times in gathered_ddp_communication_times if times is not None],
                reference_step_times,
                config,
                strategy_label,
            )
            compare_model_parameters(reference_model, ddp_model)
    finally:
        cleanup_process_group()

def build_naive_ddp_model(config: NaiveDDPConfig) -> torch.nn.Module:
    model = build_lm_model(config)
    broadcast_model_from_rank0(model)
    return model


def naive_ddp_worker(rank: int, config: NaiveDDPConfig) -> None:
    run_ddp_worker_with_strategy(
        rank,
        config,
        strategy_label="Naive DDP",
        build_ddp_model=build_naive_ddp_model,
        run_ddp_step=run_naive_ddp_step,
    )


def run_naive_ddp(config: NaiveDDPConfig) -> None:
    mp.spawn(
        naive_ddp_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )

def main() -> None:
    config = resolve_config(parse_args())
    run_naive_ddp(config)


if __name__ == "__main__":
    main()
