"""Independent Bernoulli (Poisson) sampling for window-level DP-SGD."""

from __future__ import annotations

import math

import torch


def poisson_indices(num_records: int, sample_rate: float,
                    generator: torch.Generator) -> torch.Tensor:
    """Sample each record independently with probability ``sample_rate``.

    Returns unique dataset indices in ascending order. Empty draws are valid.
    The generator is CPU based so the sampling law does not depend on the
    accelerator backend.
    """
    n = int(num_records)
    q = float(sample_rate)
    if n <= 0:
        raise ValueError("num_records must be positive")
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("sample_rate must be in [0, 1]")
    if generator.device.type != "cpu":
        raise ValueError("Poisson sampler requires a CPU torch.Generator")
    return torch.nonzero(torch.rand(n, generator=generator) < q, as_tuple=False).flatten()


def nominal_steps_per_epoch(num_records: int, expected_batch_size: int) -> int:
    """Number of Poisson draws used to represent one nominal local epoch."""
    n, expected = int(num_records), int(expected_batch_size)
    if n <= 0 or expected <= 0 or expected > n:
        raise ValueError("Require 0 < expected_batch_size <= num_records")
    return math.ceil(n / expected)
