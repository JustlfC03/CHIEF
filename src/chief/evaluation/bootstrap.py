from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def bootstrap_confidence_interval(
    metric_fn: Callable[..., float],
    *arrays: Any,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Sample-level percentile bootstrap for aligned arrays."""
    converted = [np.asarray(array) for array in arrays]
    if not converted:
        raise ValueError("At least one array is required")
    size = len(converted[0])
    if size == 0 or any(len(array) != size for array in converted):
        raise ValueError("Arrays must be non-empty and aligned on axis 0")
    estimate = float(metric_fn(*converted))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, size, size=size)
        try:
            value = float(metric_fn(*(array[indices] for array in converted)))
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return estimate, float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return estimate, float(lower), float(upper)


def bootstrap_metric_dict(
    metric_fn: Callable[..., Mapping[str, float]],
    *arrays: Any,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap all finite scalar entries returned by ``metric_fn``."""
    converted = [np.asarray(array) for array in arrays]
    if not converted:
        raise ValueError("At least one array is required")
    size = len(converted[0])
    if size == 0 or any(len(array) != size for array in converted):
        raise ValueError("Arrays must be non-empty and aligned on axis 0")
    base = dict(metric_fn(*converted))
    scalar_keys = [
        key for key, value in base.items() if isinstance(value, (int, float, np.number))
    ]
    distributions: dict[str, list[float]] = {key: [] for key in scalar_keys}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, size, size=size)
        try:
            sampled = metric_fn(*(array[indices] for array in converted))
        except (ValueError, ZeroDivisionError):
            continue
        for key in scalar_keys:
            value = float(sampled.get(key, float("nan")))
            if np.isfinite(value):
                distributions[key].append(value)
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float]] = {}
    for key in scalar_keys:
        values = distributions[key]
        if values:
            lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
        else:
            lower = upper = float("nan")
        result[key] = {
            "estimate": float(base[key]),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }
    return result


def paired_bootstrap_pvalue(
    metric_fn: Callable[..., float],
    targets: Any,
    first_predictions: Any,
    second_predictions: Any,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired bootstrap p value for a metric difference.

    Positive ``difference`` means the first system performs better. The p value
    is twice the smaller bootstrap tail probability, clipped to one.
    """
    target = np.asarray(targets)
    first = np.asarray(first_predictions)
    second = np.asarray(second_predictions)
    if len(target) == 0 or len(first) != len(target) or len(second) != len(target):
        raise ValueError("targets and both predictions must be non-empty and aligned")
    observed = float(metric_fn(target, first) - metric_fn(target, second))
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, len(target), size=len(target))
        try:
            difference = float(
                metric_fn(target[indices], first[indices])
                - metric_fn(target[indices], second[indices])
            )
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(difference):
            differences.append(difference)
    if not differences:
        pvalue = float("nan")
    else:
        values = np.asarray(differences)
        pvalue = min(1.0, 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean())))
    return {"difference": observed, "p_value": pvalue}
