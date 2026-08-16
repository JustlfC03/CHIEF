from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division with undefined entries represented by NaN."""
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def _classwise_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-order confusion matrix and one-vs-rest class statistics.

    A cohort may contain no ground-truth examinations for one of the predefined
    triage classes. The 3 x 3 confusion matrix and output order remain fixed,
    while statistics whose denominator is zero are undefined (NaN) and are
    excluded from the corresponding macro mean. This reproduces the manuscript
    evaluation behaviour for class-composition-shift cohorts.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=classes).astype(np.float64)
    total = matrix.sum()
    true_positive = np.diag(matrix)
    false_negative = matrix.sum(axis=1) - true_positive
    false_positive = matrix.sum(axis=0) - true_positive
    true_negative = total - true_positive - false_negative - false_positive

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)
    gmean = np.sqrt(recall * specificity)
    return matrix, precision, recall, specificity, gmean, true_positive


def _finite_nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    probabilities: Any | None = None,
    *,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape or true.ndim != 1:
        raise ValueError("y_true and y_pred must be aligned 1D arrays")
    if true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")

    classes = np.asarray(
        labels if labels is not None else sorted(set(true) | set(pred)),
        dtype=int,
    )
    if classes.ndim != 1 or classes.size == 0 or np.unique(classes).size != classes.size:
        raise ValueError("labels must be a non-empty sequence of unique class ids")
    allowed = set(classes.tolist())
    if not set(np.unique(true).tolist()).issubset(allowed):
        raise ValueError("y_true contains a class outside labels")
    if not set(np.unique(pred).tolist()).issubset(allowed):
        raise ValueError("y_pred contains a class outside labels")

    matrix, precisions, recalls, specificities, gmeans, _ = _classwise_statistics(
        true, pred, classes
    )
    # The historical manuscript evaluator kept all predefined classes in
    # macro-F1 (undefined classes contribute zero) but used NaN-aware means for
    # precision, recall/BACC, specificity and G-mean.
    f1s = np.asarray(
        f1_score(true, pred, labels=classes, average=None, zero_division=0),
        dtype=float,
    )

    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": _finite_nanmean(recalls),
        "macro_precision": _finite_nanmean(precisions),
        "macro_recall": _finite_nanmean(recalls),
        "macro_f1": float(
            f1_score(true, pred, labels=classes, average="macro", zero_division=0)
        ),
        "macro_specificity": _finite_nanmean(specificities),
        "macro_gmean": _finite_nanmean(gmeans),
        "cohen_kappa": float(cohen_kappa_score(true, pred, labels=classes)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_recall": {
            str(label): float(value) for label, value in zip(classes, recalls, strict=True)
        },
        "class_specificity": {
            str(label): float(value)
            for label, value in zip(classes, specificities, strict=True)
        },
    }
    for index, label in enumerate(classes):
        result[f"class_{label}_precision"] = float(precisions[index])
        result[f"class_{label}_recall"] = float(recalls[index])
        result[f"class_{label}_f1"] = float(f1s[index])
        result[f"class_{label}_specificity"] = float(specificities[index])
        result[f"class_{label}_gmean"] = float(gmeans[index])

    if probabilities is not None:
        scores = np.asarray(probabilities, dtype=float)
        observed = set(np.unique(true).tolist())
        expected = set(classes.tolist())
        if scores.ndim == 2 and scores.shape == (len(true), len(classes)):
            # Historical exported columns were occasionally rounded or not
            # exactly normalized. Normalize non-negative rows before AUROC.
            clipped = np.clip(scores, 0.0, None)
            row_sum = clipped.sum(axis=1, keepdims=True)
            valid_scores = np.isfinite(clipped).all() and np.isfinite(row_sum).all()
            valid_scores = bool(valid_scores and np.all(row_sum > 0))
            result["macro_auroc_ovr"] = (
                float(
                    roc_auc_score(
                        true,
                        clipped / row_sum,
                        labels=classes,
                        multi_class="ovr",
                        average="macro",
                    )
                )
                if observed == expected and valid_scores
                else float("nan")
            )
        elif scores.ndim == 1 and len(classes) == 2:
            result["auroc"] = (
                float(roc_auc_score(true, scores))
                if len(observed) == 2 and np.isfinite(scores).all()
                else float("nan")
            )
        else:
            raise ValueError("Probability shape is incompatible with classes")
    return result


def classification_metrics_with_ci(
    y_true: Any,
    y_pred: Any,
    probabilities: Any | None = None,
    *,
    labels: list[int] | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Classification metrics with sample-level percentile bootstrap CIs."""
    from .bootstrap import bootstrap_metric_dict

    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    scores = None if probabilities is None else np.asarray(probabilities, dtype=float)

    def scalar_metrics(sample_true, sample_pred, sample_scores=None):
        result = classification_metrics(
            sample_true,
            sample_pred,
            sample_scores,
            labels=labels,
        )
        return {
            key: value
            for key, value in result.items()
            if isinstance(value, (int, float, np.number))
        }

    if scores is None:
        ci = bootstrap_metric_dict(
            lambda a, b: scalar_metrics(a, b),
            true,
            pred,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
    else:
        ci = bootstrap_metric_dict(
            scalar_metrics,
            true,
            pred,
            scores,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
    point = classification_metrics(true, pred, scores, labels=labels)
    return {
        "n_samples": len(true),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "metrics": ci,
        "confusion_matrix": point["confusion_matrix"],
        "class_recall": point["class_recall"],
        "class_specificity": point["class_specificity"],
    }
