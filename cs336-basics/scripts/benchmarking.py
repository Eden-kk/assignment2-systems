"""Skeleton for CS336 Assignment 2 benchmarking_script.

This file is intentionally incomplete. It sets up the structure, CLI surface,
and model-size presets so you can fill in the actual benchmarking logic
yourself.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import statistics
from timeit import default_timer
from typing import Literal

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

ModelSizeName = Literal["small", "medium", "large", "xl", "2.7B"]
BenchmarkMode = Literal["forward", "forward-backward"]
PrecisionName = Literal["float16", "float32", "bfloat16"]
PrecisionAutocastName = Literal["none", "float16", "bfloat16"]


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
    dtype: torch.dtype
    precision_autocast: torch.dtype | None
    seed: int


DTYPE_MAP: dict[PrecisionName, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


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
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), default="float32")
    parser.add_argument(
        "--precision-autocast",
        choices=("none", "float16", "bfloat16"),
        default="none",
        help="Autocast dtype for mixed precision. Keep model params in --dtype and cast eligible ops dynamically.",
    )
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
        dtype=DTYPE_MAP[args.dtype],
        precision_autocast=None if args.precision_autocast == "none" else DTYPE_MAP[args.precision_autocast],
        seed=args.seed,
    )


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def autocast_context(config: BenchmarkConfig) -> contextlib.AbstractContextManager[None]:
    if config.precision_autocast is None:
        return contextlib.nullcontext()
    device_type = "cuda" if config.device.startswith("cuda") else config.device
    return torch.autocast(device_type=device_type, dtype=config.precision_autocast)


def build_model(config: BenchmarkConfig) -> BasicsTransformerLM:
    """Instantiate the assignment-1 Transformer and move it to the target device/dtype."""
    spec = MODEL_SPECS[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=config.rope_theta,
    )
    model.to(device=config.device, dtype=config.dtype)
    return model


def make_batch(config: BenchmarkConfig) -> torch.Tensor:
    """Create a random integer token batch with shape [batch_size, context_length]."""
    batch = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length + 1),
        device=config.device,
        dtype=torch.long,
    )
    return batch


def run_step(model: BasicsTransformerLM, batch: torch.Tensor, config: BenchmarkConfig) -> None:
    """Run one benchmark step and synchronize CUDA after the step completes."""
    x, target = batch[..., :-1], batch[..., 1:]

    if config.mode == "forward-backward":
        model.zero_grad(set_to_none=True)
        with autocast_context(config):
            logits = model(x)
            loss = cross_entropy(logits, target)
        loss.backward()
    else:
        with torch.no_grad():
            with autocast_context(config):
                _ = model(x)

    synchronize(config.device)


def benchmark_steps(
    model: BasicsTransformerLM,
    batch: torch.Tensor,
    config: BenchmarkConfig,
) -> list[float]:
    """Perform warmup, time measurement steps, and return per-step durations in seconds."""
    for _ in range(config.warmup_steps):
        run_step(model, batch, config)

    step_times = []

    for _ in range(config.measurement_steps):
        synchronize(config.device)
        start = default_timer()
        run_step(model, batch, config)
        synchronize(config.device)
        end = default_timer()
        step_times.append(end - start)

    return step_times


def summarize_timings(step_times: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(step_times),
        "stdev": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "min": min(step_times),
        "max": max(step_times),
    }
    


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
