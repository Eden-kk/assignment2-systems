"""Skeleton for CS336 Assignment 2 benchmarking_script.

This file is intentionally incomplete. It sets up the structure, CLI surface,
and model-size presets so you can fill in the actual benchmarking logic
yourself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import torch

from cs336_basics.model import BasicsTransformerLM

ModelSizeName = Literal["small", "medium", "large", "xl", "2.7B"]
BenchmarkMode = Literal["forward", "forward-backward"]


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


# Model sizes from the assignment handout.
MODEL_SPECS: dict[ModelSizeName, ModelSpec] = {
    "small": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSpec(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSpec(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelSpec(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: ModelSizeName
    context_length: int
    batch_size: int
    vocab_size: int
    rope_theta: float
    warmup_steps: int
    measurement_steps: int
    mode: BenchmarkMode
    device: str
    dtype: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skeleton CLI for benchmarking Transformer forward and backward passes."
    )
    parser.add_argument("--model-size", choices=tuple(MODEL_SPECS), default="small")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--mode", choices=("forward", "forward-backward"), default="forward-backward")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        rope_theta=args.rope_theta,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        mode=args.mode,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )


def build_model(config: BenchmarkConfig) -> BasicsTransformerLM:
    """TODO: Instantiate the assignment-1 Transformer and move it to the target device/dtype."""
    raise NotImplementedError("Implement model construction here.")


def make_batch(config: BenchmarkConfig) -> torch.Tensor:
    """TODO: Create a random integer token batch with shape [batch_size, context_length]."""
    raise NotImplementedError("Implement random batch creation here.")


def run_step(model: BasicsTransformerLM, batch: torch.Tensor, config: BenchmarkConfig) -> None:
    """TODO: Run one benchmark step and synchronize CUDA after the step completes."""
    raise NotImplementedError("Implement one forward or forward-backward step here.")


def benchmark_steps(
    model: BasicsTransformerLM,
    batch: torch.Tensor,
    config: BenchmarkConfig,
) -> list[float]:
    """TODO: Perform warmup, time measurement steps, and return per-step durations in seconds."""
    raise NotImplementedError("Implement the benchmarking loop here.")


def summarize_timings(step_times: list[float]) -> dict[str, float]:
    """TODO: Compute aggregate statistics such as mean and standard deviation."""
    raise NotImplementedError("Implement timing aggregation here.")


def main() -> None:
    config = resolve_config(parse_args())

    torch.manual_seed(config.seed)

    model = build_model(config)
    batch = make_batch(config)
    step_times = benchmark_steps(model, batch, config)
    summary = summarize_timings(step_times)

    print(summary)


if __name__ == "__main__":
    main()
