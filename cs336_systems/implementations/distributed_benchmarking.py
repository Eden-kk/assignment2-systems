"""Skeleton benchmark script for Problem (distributed_communication_single_node).

This file intentionally leaves the core benchmarking realization for you:
- setting up the exact tensor/device for each rank
- timing `dist.all_reduce(...)`
- synchronizing correctly for GPU runs
- aggregating and reporting results

The goal here is to keep only the handout-aligned structure in place.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from statistics import mean
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DTYPE_MAP = {
    "float32": torch.float32,
}


@dataclass(frozen=True)
class AllReduceBenchmarkConfig:
    world_size: int
    backend: str
    device: str
    dtype: torch.dtype
    tensor_size_mb: float
    warmup_steps: int
    measurement_steps: int
    master_addr: str
    master_port: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dist.all_reduce in a single-node multi-process setup."
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP.keys()), default="float32")
    parser.add_argument("--tensor-size-mb", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=20)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29500")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> AllReduceBenchmarkConfig:
    return AllReduceBenchmarkConfig(
        world_size=args.world_size,
        backend=args.backend,
        device=args.device,
        dtype=DTYPE_MAP[args.dtype],
        tensor_size_mb=args.tensor_size_mb,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        master_addr=args.master_addr,
        master_port=args.master_port,
        seed=args.seed,
    )


def _validate_config(config: AllReduceBenchmarkConfig) -> None:
    if config.backend == "nccl" and not config.device.startswith("cuda"):
        raise ValueError("NCCL benchmarks must use a CUDA device.")
    if config.device.startswith("mps"):
        raise ValueError("Use CPU for Gloo on a Mac. MPS is not the target device for this benchmark.")
    if config.dtype != torch.float32:
        raise ValueError("This problem asks for float32 all-reduce tensors.")


def _synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def setup_process_group(rank: int, config: AllReduceBenchmarkConfig) -> None:
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


def make_tensor(rank: int, config: AllReduceBenchmarkConfig) -> torch.Tensor:
    """Construct the per-rank float32 tensor to benchmark.

    - Convert `tensor_size_mb` into a number of elements using `config.dtype`.
    - Decide what initial values you want in the tensor.
    - Place the tensor on `config.device` with dtype `config.dtype`.
    """

    num_bytes = int(config.tensor_size_mb * 1024 * 1024)
    element_size = torch.tensor([], dtype=config.dtype).element_size()
    numel = max(1, num_bytes // element_size)
    return torch.full(
        (numel,),
        fill_value=float(rank + 1),
        dtype=config.dtype,
        device=config.device,
    )


def benchmark_all_reduce_once(tensor: torch.Tensor, config: AllReduceBenchmarkConfig) -> float:
    """Time one all-reduce call.

    - Synchronize before timing if needed.
    - Call `dist.all_reduce(...)`.
    - Synchronize again if needed.
    - Return elapsed wall-clock time in seconds.
    """

    _synchronize_if_needed(device=config.device)
    start = default_timer()

    dist.all_reduce(tensor, async_op=False)
    _synchronize_if_needed(device=config.device)

    return default_timer() - start


def process_gathered_times(gathered_times: list[list[float]], config: AllReduceBenchmarkConfig) -> None:
    """Print a simple rank-by-rank and aggregate timing report.

    Expected input:
    - `gathered_times[i]` is the list of measured step times, in seconds,
      collected from rank `i`.
    """

    per_rank_means_ms: list[float] = []

    print("All-Reduce Benchmark Report")
    print(
        f"backend={config.backend} "
        f"device={config.device} "
        f"dtype={config.dtype} "
        f"world_size={config.world_size} "
        f"tensor_size_mb={config.tensor_size_mb}"
    )
    print(f"warmup_steps={config.warmup_steps} measurement_steps={config.measurement_steps}")
    print("Per-rank timings:")

    for rank, rank_times in enumerate(gathered_times):
        rank_times_ms = [1000.0 * value for value in rank_times]
        rank_mean_ms = mean(rank_times_ms)
        per_rank_means_ms.append(rank_mean_ms)
        print(
            f"  rank {rank}: "
            f"mean={rank_mean_ms:.3f} ms "
            f"min={min(rank_times_ms):.3f} ms "
            f"max={max(rank_times_ms):.3f} ms "
            f"num_steps={len(rank_times_ms)}"
        )

    print("Aggregate:")
    print(f"  mean_of_rank_means={mean(per_rank_means_ms):.3f} ms")
    print(f"  max_rank_mean={max(per_rank_means_ms):.3f} ms")
    print(f"  min_rank_mean={min(per_rank_means_ms):.3f} ms")


def benchmark_worker(rank: int, config: AllReduceBenchmarkConfig) -> None:
    setup_process_group(rank, config)
    try:
        if config.device.startswith("cuda"):
            torch.cuda.set_device(rank)
        torch.manual_seed(config.seed + rank)

        # - Create the rank-local tensor with `make_tensor(...)`.
        # - Run `config.warmup_steps` warmup iterations.
        # - Run `config.measurement_steps` timed iterations and collect times.
        # - Package a per-rank result dictionary with the fields you care about.
        # - Gather those objects to rank 0 with `dist.all_gather_object(...)`.
        # - Print or save a rank-0 summary for the current configuration.
        tensor = make_tensor(rank, config)
        dist.barrier()

        for _ in range(config.warmup_steps):
            dist.all_reduce(tensor, async_op=False)
        dist.barrier()

        times: list[float] = []

        for _ in range(config.measurement_steps):
            tensor = make_tensor(rank, config)
            elapsed = benchmark_all_reduce_once(tensor, config)
            times.append(elapsed)
        dist.barrier()

        gathered_times: list[list[float] | None] = [None for _ in range(config.world_size)]
        dist.all_gather_object(gathered_times, times)
        dist.barrier()

        if rank == 0:
            process_gathered_times(
                [rank_times for rank_times in gathered_times if rank_times is not None],
                config,
            )
        
    finally:
        cleanup_process_group()


def benchmark_distributed_communication_single_node(config: AllReduceBenchmarkConfig) -> None:
    _validate_config(config)
    mp.spawn(
        benchmark_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )


def main() -> None:
    config = resolve_config(parse_args())
    benchmark_distributed_communication_single_node(config)


if __name__ == "__main__":
    main()
