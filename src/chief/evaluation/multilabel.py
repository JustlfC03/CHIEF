from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)



def _safe_binary_metric(function, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(function(y_true, y_score))


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negative = y_true == 0
    denominator = int(negative.sum())
    return float(((y_pred == 0) & negative).sum() / denominator) if denominator else float("nan")


def _threshold_array(threshold: float | Sequence[float], columns: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, columns)
    if values.shape != (columns,):
        raise ValueError(f"threshold must be scalar or have shape ({columns},)")
    return values


def fit_optimal_thresholds(
    targets: Any,
    probabilities: Any,
    *,
    objective: str = "youden",
) -> np.ndarray:
    """Fit one threshold per label on a validation set only.

    ``youden`` maximizes sensitivity + specificity - 1. ``f1`` maximizes F1.
    The caller is responsible for keeping the evaluation test set independent.
    """
    true = np.asarray(targets, dtype=int)
    score = np.asarray(probabilities, dtype=float)
    if true.shape != score.shape or true.ndim != 2:
        raise ValueError("targets and probabilities must be aligned [N,C] arrays")
    if objective not in {"youden", "f1"}:
        raise ValueError("objective must be 'youden' or 'f1'")
    thresholds = np.full(true.shape[1], 0.5, dtype=float)
    for column in range(true.shape[1]):
        y = true[:, column]
        s = score[:, column]
        if np.unique(y).size < 2:
            continue
        candidates = np.unique(np.concatenate(([0.0], s, [1.0])))
        best_value = -np.inf
        best_threshold = 0.5
        for candidate in candidates:
            pred = s >= candidate
            if objective == "f1":
                value = f1_score(y, pred, zero_division=0)
            else:
                sensitivity = recall_score(y, pred, zero_division=0)
                specificity = _specificity(y, pred)
                value = sensitivity + specificity - 1.0
            if value > best_value or (value == best_value and abs(candidate - 0.5) < abs(best_threshold - 0.5)):
                best_value = float(value)
                best_threshold = float(candidate)
        thresholds[column] = best_threshold
    return thresholds


def multilabel_metrics(
    targets: Any,
    probabilities: Any,
    *,
    threshold: float | Sequence[float] = 0.5,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    true = np.asarray(targets, dtype=int)
    score = np.asarray(probabilities, dtype=float)
    if true.shape != score.shape or true.ndim != 2:
        raise ValueError("targets and probabilities must be aligned [N,C] arrays")
    labels = list(label_names or [str(i) for i in range(true.shape[1])])
    if len(labels) != true.shape[1]:
        raise ValueError("label_names length does not match number of columns")
    thresholds = _threshold_array(threshold, true.shape[1])
    pred = score >= thresholds[None, :]
    per_label: dict[str, dict[str, float | int]] = {}
    aurocs: list[float] = []
    aps: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    specificities: list[float] = []
    baccs: list[float] = []
    gmeans: list[float] = []
    for index, name in enumerate(labels):
        y = true[:, index]
        p = pred[:, index]
        auroc = _safe_binary_metric(roc_auc_score, y, score[:, index])
        ap = _safe_binary_metric(average_precision_score, y, score[:, index])
        precision = float(precision_score(y, p, zero_division=0))
        recall = float(recall_score(y, p, zero_division=0))
        f1 = float(f1_score(y, p, zero_division=0))
        specificity = _specificity(y, p)
        bacc = float(0.5 * (recall + specificity)) if np.isfinite(specificity) and np.unique(y).size == 2 else float("nan")
        gmean = float(np.sqrt(max(recall, 0.0) * max(specificity, 0.0))) if np.isfinite(specificity) else float("nan")
        aurocs.append(auroc); aps.append(ap); precisions.append(precision); recalls.append(recall)
        f1s.append(f1); specificities.append(specificity); baccs.append(bacc); gmeans.append(gmean)
        per_label[name] = {
            "positive_cases": int(y.sum()),
            "threshold": float(thresholds[index]),
            "auroc": auroc,
            "average_precision": ap,
            "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": bacc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "gmean": gmean,
        }
    flat_true, flat_score, flat_pred = true.ravel(), score.ravel(), pred.ravel()
    finite_mean = lambda values: float(np.mean([v for v in values if np.isfinite(v)])) if any(np.isfinite(v) for v in values) else float("nan")
    micro_recall = float(recall_score(flat_true, flat_pred, zero_division=0))
    micro_specificity = _specificity(flat_true, flat_pred)
    return {
        "macro_auroc": finite_mean(aurocs),
        "micro_auroc": _safe_binary_metric(roc_auc_score, flat_true, flat_score),
        "macro_average_precision": finite_mean(aps),
        "micro_average_precision": _safe_binary_metric(average_precision_score, flat_true, flat_score),
        "subset_accuracy": float(accuracy_score(true, pred)),
        "macro_accuracy": finite_mean([float(accuracy_score(true[:, i], pred[:, i])) for i in range(true.shape[1])]),
        "macro_balanced_accuracy": finite_mean(baccs),
        "macro_precision": finite_mean(precisions),
        "macro_recall": finite_mean(recalls),
        "macro_f1": finite_mean(f1s),
        "macro_specificity": finite_mean(specificities),
        "macro_gmean": finite_mean(gmeans),
        "micro_precision": float(precision_score(flat_true, flat_pred, zero_division=0)),
        "micro_recall": micro_recall,
        "micro_f1": float(f1_score(flat_true, flat_pred, zero_division=0)),
        "micro_specificity": micro_specificity,
        "micro_gmean": float(np.sqrt(max(micro_recall, 0.0) * max(micro_specificity, 0.0))),
        "thresholds": thresholds.tolist(),
        "per_label": per_label,
    }


def multilabel_metrics_with_ci(
    targets: Any,
    probabilities: Any,
    *,
    threshold: float | Sequence[float] = 0.5,
    label_names: Sequence[str] | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Multilabel endpoints with examination-level percentile bootstrap CIs.

    The same resampled examination indices are applied to every label, which
    preserves within-examination label dependence. CIs are returned for both
    aggregate scalar endpoints and label-wise endpoints.
    """
    true = np.asarray(targets, dtype=int)
    score = np.asarray(probabilities, dtype=float)
    if true.shape != score.shape or true.ndim != 2 or len(true) == 0:
        raise ValueError("targets and probabilities must be non-empty aligned [N,C] arrays")
    point = multilabel_metrics(true, score, threshold=threshold, label_names=label_names)
    labels = list(point["per_label"])
    aggregate_keys = [
        key for key, value in point.items()
        if isinstance(value, (int, float, np.number))
    ]
    label_keys = [
        "auroc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "gmean",
    ]
    aggregate_samples: dict[str, list[float]] = {key: [] for key in aggregate_keys}
    label_samples: dict[str, dict[str, list[float]]] = {
        label: {key: [] for key in label_keys} for label in labels
    }
    rng = np.random.default_rng(seed)
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, len(true), size=len(true))
        sampled = multilabel_metrics(
            true[indices],
            score[indices],
            threshold=threshold,
            label_names=labels,
        )
        for key in aggregate_keys:
            value = float(sampled.get(key, float("nan")))
            if np.isfinite(value):
                aggregate_samples[key].append(value)
        for label in labels:
            row = sampled["per_label"][label]
            for key in label_keys:
                value = float(row.get(key, float("nan")))
                if np.isfinite(value):
                    label_samples[label][key].append(value)

    alpha = (1.0 - confidence) / 2.0

    def interval(estimate: float, values: list[float]) -> dict[str, float]:
        if values:
            lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
        else:
            lower = upper = float("nan")
        return {
            "estimate": float(estimate),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }

    aggregate = {
        key: interval(float(point[key]), aggregate_samples[key]) for key in aggregate_keys
    }
    per_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        source = dict(point["per_label"][label])
        source["confidence_intervals"] = {
            key: interval(float(source[key]), label_samples[label][key]) for key in label_keys
        }
        per_label[label] = source
    return {
        "n_samples": len(true),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "metrics": aggregate,
        "thresholds": point["thresholds"],
        "per_label": per_label,
    }

