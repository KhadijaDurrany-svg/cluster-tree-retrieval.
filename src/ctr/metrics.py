"""Retrieval metrics: supporting-chunk recall and all-support, with a bootstrap."""

from __future__ import annotations

import random
from collections.abc import Sequence


def support_recall(covered: set[str], supporting: Sequence[str]) -> float:
    if not supporting:
        return 0.0
    s = set(supporting)
    return len(covered & s) / len(s)


def all_support(covered: set[str], supporting: Sequence[str]) -> float:
    if not supporting:
        return 0.0
    return 1.0 if set(supporting) <= covered else 0.0


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, confidence: float = 0.95,
                 seed: int = 0) -> tuple[float, float, float]:
    vals = list(values)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(vals) / n
    if n == 1:
        return (mean, mean, mean)
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[max(0, int((1 - confidence) / 2 * n_resamples))]
    hi = means[min(n_resamples - 1, int((1 + confidence) / 2 * n_resamples) - 1)]
    return (mean, lo, hi)


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_resamples: int = 10000,
                            seed: int = 0) -> float:
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    if n == 0:
        return 1.0
    observed = abs(sum(diffs) / n)
    rng = random.Random(seed)
    count = 0
    for _ in range(n_resamples):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / n) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_resamples + 1)
