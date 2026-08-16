from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from chief.constants import CQ500_LABELS, TRIAGE_LABEL_TO_ID
from chief.data.manifests import read_csv_compatible
from chief.evaluation import (
    classification_metrics,
    classification_metrics_with_ci,
    fit_optimal_thresholds,
    generation_metrics,
    generation_metrics_with_ci,
    generation_paired_comparison,
    multilabel_metrics,
    multilabel_metrics_with_ci,
    paired_bootstrap_pvalue,
    first_match_ranks,
    retrieval_metrics,
    retrieval_metrics_from_ranks,
    retrieval_metrics_from_ranks_with_ci,
    retrieval_metrics_with_ci,
    text_relevance_matrix,
)
from chief.utils import configure_logging


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _label_id(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "normal": "negative",
            "neg": "negative",
            "nonemergencypositive": "non-emergency-positive",
            "non-emergency positive": "non-emergency-positive",
            "non emergency positive": "non-emergency-positive",
            "urgent-positive": "positive",
            "urgent positive": "positive",
            "pos": "positive",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in TRIAGE_LABEL_TO_ID:
            return TRIAGE_LABEL_TO_ID[normalized]
    return int(value)


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    direct = {str(column): str(column) for column in frame.columns}
    folded = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in direct:
            return direct[candidate]
        match = folded.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def _triage_arrays(frame: pd.DataFrame):
    true_column = _find_column(frame, ["true_label", "class_label", "triage_label"])
    if true_column is None:
        raise ValueError("Triage CSV must contain true_label, class_label, or triage_label")
    true = frame[true_column].map(_label_id).to_numpy(dtype=int)

    prediction_column = _find_column(
        frame,
        ["prediction_id", "prediction", "Pred", "pred", "predicted_label"],
    )
    if prediction_column is None:
        raise ValueError("Triage CSV must contain prediction_id, prediction, or Pred")
    pred = frame[prediction_column].map(_label_id).to_numpy(dtype=int)
    valid_ids = {0, 1, 2}
    if not set(np.unique(true)).issubset(valid_ids):
        raise ValueError(f"Triage ground-truth labels must be in {sorted(valid_ids)}")
    if not set(np.unique(pred)).issubset(valid_ids):
        raise ValueError(f"Triage predictions must be in {sorted(valid_ids)}")

    probability_candidates = [
        ["prob_negative", "prob_neg", "PROB_NEG"],
        [
            "prob_non-emergency-positive",
            "prob_non_emergency_positive",
            "prob_non",
            "PROB_NON",
        ],
        ["prob_positive", "prob_pos", "PROB_POS"],
    ]
    probability_columns = [_find_column(frame, names) for names in probability_candidates]
    probabilities = None
    if all(column is not None for column in probability_columns):
        probabilities = frame[[str(column) for column in probability_columns]].to_numpy(dtype=float)
    return true, pred, probabilities


def _read_threshold_payload(path: str | None) -> object | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("thresholds", payload) if isinstance(payload, dict) else payload


def _load_thresholds(path: str | None, labels: list[str], fallback: float) -> float | np.ndarray:
    payload = _read_threshold_payload(path)
    if payload is None:
        return fallback
    if isinstance(payload, dict):
        missing = [label for label in labels if label not in payload]
        if missing:
            raise ValueError(f"Threshold file is missing labels: {missing[:5]}")
        return np.asarray([float(payload[label]) for label in labels], dtype=float)
    values = np.asarray(payload, dtype=float)
    if values.shape != (len(labels),):
        raise ValueError("Threshold file does not match the label order")
    return values


def _load_named_threshold(path: str | None, name: str, fallback: float) -> float:
    payload = _read_threshold_payload(path)
    if isinstance(payload, dict) and name in payload:
        return float(payload[name])
    return float(fallback)


def _multilabel_arrays(frame: pd.DataFrame, labels: list[str]):
    missing_true = [label for label in labels if label not in frame]
    missing_prob = [f"prob_{label}" for label in labels if f"prob_{label}" not in frame]
    if missing_true or missing_prob:
        raise ValueError(f"Missing label columns: true={missing_true}, probability={missing_prob}")
    return (
        frame[labels].to_numpy(dtype=int),
        frame[[f"prob_{label}" for label in labels]].to_numpy(dtype=float),
    )


def _triage_comparison(
    true: np.ndarray,
    first_pred: np.ndarray,
    second_pred: np.ndarray,
    first_probabilities: np.ndarray | None,
    second_probabilities: np.ndarray | None,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_specificity",
        "macro_gmean",
        "cohen_kappa",
    ]
    result: dict[str, Any] = {}
    for offset, name in enumerate(metric_names):
        metric: Callable[[np.ndarray, np.ndarray], float] = (
            lambda y, p, metric_name=name: float(
                classification_metrics(y, p, labels=[0, 1, 2])[metric_name]
            )
        )
        result[name] = paired_bootstrap_pvalue(
            metric,
            true,
            first_pred,
            second_pred,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
    if first_probabilities is not None and second_probabilities is not None:
        def macro_auroc(y: np.ndarray, scores: np.ndarray) -> float:
            predicted = scores.argmax(axis=1)
            value = classification_metrics(
                y, predicted, scores, labels=[0, 1, 2]
            ).get("macro_auroc_ovr", float("nan"))
            return float(value)

        result["macro_auroc_ovr"] = paired_bootstrap_pvalue(
            macro_auroc,
            true,
            first_probabilities,
            second_probabilities,
            n_resamples=n_resamples,
            seed=seed + len(metric_names),
        )
    return result


def _multilabel_comparison(
    true: np.ndarray,
    first_score: np.ndarray,
    second_score: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    def macro_auroc(y, score):
        values = [
            roc_auc_score(y[:, column], score[:, column])
            for column in range(y.shape[1])
            if np.unique(y[:, column]).size == 2
        ]
        return float(np.mean(values)) if values else float("nan")

    def macro_ap(y, score):
        values = [
            average_precision_score(y[:, column], score[:, column])
            for column in range(y.shape[1])
            if np.unique(y[:, column]).size == 2
        ]
        return float(np.mean(values)) if values else float("nan")

    return {
        "macro_auroc": paired_bootstrap_pvalue(
            macro_auroc,
            true,
            first_score,
            second_score,
            n_resamples=n_resamples,
            seed=seed,
        ),
        "macro_average_precision": paired_bootstrap_pvalue(
            macro_ap,
            true,
            first_score,
            second_score,
            n_resamples=n_resamples,
            seed=seed + 1,
        ),
    }


def _retrieval_comparison(
    first: np.ndarray,
    second: np.ndarray,
    relevance: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Retrieval comparator matrices must be aligned [queries,candidates] matrices")
    if relevance.shape != first.shape:
        raise ValueError("Retrieval relevance matrix is not aligned with the score matrices")
    keys = list(retrieval_metrics(first, relevance=relevance))
    result: dict[str, Any] = {}
    for offset, key in enumerate(keys):
        metric = lambda rel, score, name=key: float(
            retrieval_metrics(score, relevance=rel)[name]
        )
        result[key] = paired_bootstrap_pvalue(
            metric,
            relevance,
            first,
            second,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
    return result



def _parse_ks(value: str) -> tuple[int, ...]:
    ks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--retrieval-ks must contain positive comma-separated integers")
    return ks


def _read_report_column(path: str, column: str) -> list[str]:
    frame = read_csv_compatible(path)
    if column not in frame:
        raise ValueError(f"Column {column!r} is missing from {path}")
    return frame[column].fillna("").astype(str).tolist()

def _overall_ground_truth(frame: pd.DataFrame) -> np.ndarray | None:
    if "true_overall_abnormality" in frame:
        return frame["true_overall_abnormality"].to_numpy(dtype=int)
    for column in ("class_label", "triage_label", "true_label"):
        if column not in frame:
            continue
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            return values.to_numpy(dtype=int) != 0
        normalized = values.fillna("").astype(str).str.strip().str.lower().str.replace("_", "-")
        return (~normalized.isin({"negative", "normal", "neg", "0"})).to_numpy(dtype=int)
    return None


def _overall_scores(
    frame: pd.DataFrame,
    label_probabilities: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, str]:
    for column in ("prob_overall_abnormality", "p_abnormal_topk", "p_abnormal_max"):
        if column in frame:
            return frame[column].to_numpy(dtype=float), column
    k = min(max(1, int(top_k)), label_probabilities.shape[1])
    partitioned = np.partition(label_probabilities, -k, axis=1)[:, -k:]
    return partitioned.mean(axis=1), "top_k_mean"


def _overall_metrics(
    true: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float,
    no_bootstrap: bool,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    true_2d = np.asarray(true, dtype=int).reshape(-1, 1)
    score_2d = np.asarray(score, dtype=float).reshape(-1, 1)
    if no_bootstrap:
        return multilabel_metrics(
            true_2d,
            score_2d,
            threshold=threshold,
            label_names=["overall_abnormality"],
        )
    return multilabel_metrics_with_ci(
        true_2d,
        score_2d,
        threshold=threshold,
        label_names=["overall_abnormality"],
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def _ordered_zero_shot_labels(frame: pd.DataFrame, labels_json: str | None) -> list[str]:
    if labels_json:
        payload = json.loads(Path(labels_json).read_text(encoding="utf-8"))
        values = payload.get("labels", payload) if isinstance(payload, dict) else payload
        labels = [
            item.get("name_zh", item.get("name", "")) if isinstance(item, dict) else str(item)
            for item in values
        ]
        labels = [label for label in labels if label]
    else:
        labels = [
            column[5:]
            for column in frame.columns
            if column.startswith("prob_")
            and column != "prob_overall_abnormality"
            and column[5:] in frame.columns
        ]
    if not labels:
        raise ValueError("No ordered zero-shot labels could be determined")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exported CHIEF predictions")
    parser.add_argument(
        "--task",
        required=True,
        choices=["triage", "cq500", "zero_shot", "generation", "retrieval"],
    )
    parser.add_argument("--predictions", required=True, help="CSV, or .npy similarity matrix")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--comparison-predictions",
        help="Aligned comparator CSV, or comparator .npy matrix for retrieval",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds-json")
    parser.add_argument(
        "--threshold-validation-predictions",
        help="Independent validation CSV used to fit per-label thresholds",
    )
    parser.add_argument("--threshold-objective", choices=["youden", "f1"], default="youden")
    parser.add_argument("--save-thresholds")
    parser.add_argument("--labels-json", help="Ordered label list for zero_shot evaluation")
    parser.add_argument("--overall-top-k", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument(
        "--rouge-tokenization", choices=["jieba", "char", "whitespace"], default="jieba"
    )
    parser.add_argument(
        "--meteor-tokenization", choices=["char", "jieba", "whitespace"], default="char"
    )
    parser.add_argument("--bertscore-model", default="bert-base-chinese")
    parser.add_argument("--bertscore-device")
    parser.add_argument("--bertscore-batch-size", type=int, default=32)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-cider", action="store_true")
    parser.add_argument("--relevance-matrix", help="Optional .npy boolean relevance matrix for retrieval")
    parser.add_argument("--query-reports", help="CSV containing query/reference reports for retrieval relevance")
    parser.add_argument("--candidate-reports", help="CSV containing candidate reports for retrieval relevance")
    parser.add_argument("--query-report-column", default="report")
    parser.add_argument("--candidate-report-column", default="report")
    parser.add_argument(
        "--reference-column", "--ranked-reference-column",
        dest="reference_column", default="gt_answer",
        help="Reference column for ranked-report CSV evaluation",
    )
    parser.add_argument(
        "--candidates-column", "--ranked-candidates-column",
        dest="candidates_column", default="similar_valid_gt",
        help="Separator-joined ranked reports column",
    )
    parser.add_argument(
        "--retrieval-separator", "--ranked-separator",
        dest="retrieval_separator", default=" || ",
    )
    parser.add_argument("--match-mode", choices=["exact", "contains", "fuzzy"], default="exact")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.99)
    parser.add_argument("--min-match-length", type=int, default=8)
    parser.add_argument("--retrieval-ks", default="1,5,10,50,100,200")
    parser.add_argument(
        "--ranked-miss-policy",
        choices=["zero", "after-list"],
        default="after-list",
        help=(
            "For ranked-report CSVs, use zero for standard MRR (miss contributes 0) "
            "or after-list to reproduce the manuscript evaluator (miss rank=top_k+1)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    configure_logging(args.verbose)
    source = Path(args.predictions)

    if args.task == "retrieval":
        ks = _parse_ks(args.retrieval_ks)
        if source.suffix.lower() == ".csv":
            frame = read_csv_compatible(source)
            missing = [
                column
                for column in (args.reference_column, args.candidates_column)
                if column not in frame
            ]
            if missing:
                raise ValueError(f"Ranked-report CSV is missing columns: {missing}")
            references = frame[args.reference_column].fillna("").astype(str).tolist()
            ranked = [
                [item.strip() for item in str(value).split(args.retrieval_separator) if item.strip()]
                for value in frame[args.candidates_column].fillna("").astype(str).tolist()
            ]
            max_candidates = max((len(items) for items in ranked), default=0)
            if max_candidates == 0:
                raise ValueError("Ranked-report CSV contains no candidates")
            if max(ks) > max_candidates:
                raise ValueError(
                    f"Ranked-report CSV contains at most {max_candidates} candidates, "
                    f"but retrieval K includes {max(ks)}"
                )
            miss_rank = (
                float(max_candidates + 1)
                if args.ranked_miss_policy == "after-list"
                else None
            )
            ranks = first_match_ranks(
                references,
                ranked,
                mode=args.match_mode,
                fuzzy_threshold=args.fuzzy_threshold,
                min_length=args.min_match_length,
                miss_rank=miss_rank,
            )
            metrics = (
                retrieval_metrics_from_ranks(ranks, ks=ks)
                if args.no_bootstrap
                else retrieval_metrics_from_ranks_with_ci(
                    ranks,
                    ks=ks,
                    n_resamples=args.n_bootstrap,
                    confidence=args.confidence,
                    seed=args.seed,
                )
            )
            metrics["matching"] = {
                "mode": args.match_mode,
                "fuzzy_threshold": args.fuzzy_threshold if args.match_mode == "fuzzy" else None,
                "min_length": args.min_match_length,
                "reference_column": args.reference_column,
                "candidates_column": args.candidates_column,
                "miss_policy": args.ranked_miss_policy,
                "max_candidates": max_candidates,
            }
        else:
            matrix = np.load(source)
            relevance = None
            relevance_source = "paired_diagonal"
            if args.relevance_matrix:
                relevance = np.load(args.relevance_matrix).astype(bool)
                relevance_source = str(Path(args.relevance_matrix).resolve())
            elif args.query_reports or args.candidate_reports:
                if not args.query_reports or not args.candidate_reports:
                    raise ValueError("--query-reports and --candidate-reports must be supplied together")
                query_reports = _read_report_column(args.query_reports, args.query_report_column)
                candidate_reports = _read_report_column(
                    args.candidate_reports, args.candidate_report_column
                )
                relevance = text_relevance_matrix(
                    query_reports,
                    candidate_reports,
                    mode=args.match_mode,
                    fuzzy_threshold=args.fuzzy_threshold,
                    min_length=args.min_match_length,
                )
                relevance_source = f"text:{args.match_mode}"
            if relevance is None:
                if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                    raise ValueError(
                        "A rectangular similarity matrix requires --relevance-matrix or report CSVs"
                    )
                relevance = np.eye(matrix.shape[0], dtype=bool)
            metrics = (
                retrieval_metrics(matrix, relevance=relevance, ks=ks)
                if args.no_bootstrap
                else retrieval_metrics_with_ci(
                    matrix,
                    relevance=relevance,
                    ks=ks,
                    n_resamples=args.n_bootstrap,
                    confidence=args.confidence,
                    seed=args.seed,
                )
            )
            metrics["relevance"] = {
                "source": relevance_source,
                "match_mode": args.match_mode if relevance_source.startswith("text:") else None,
            }
            if args.comparison_predictions:
                comparison = np.load(args.comparison_predictions)
                metrics["paired_comparison"] = _retrieval_comparison(
                    matrix,
                    comparison,
                    relevance,
                    n_resamples=args.n_bootstrap,
                    seed=args.seed,
                )
    else:
        frame = read_csv_compatible(source)
        if args.task == "triage":
            true, pred, probabilities = _triage_arrays(frame)
            metrics = (
                classification_metrics(true, pred, probabilities, labels=[0, 1, 2])
                if args.no_bootstrap
                else classification_metrics_with_ci(
                    true,
                    pred,
                    probabilities,
                    labels=[0, 1, 2],
                    n_resamples=args.n_bootstrap,
                    confidence=args.confidence,
                    seed=args.seed,
                )
            )
            if args.comparison_predictions:
                comparison = read_csv_compatible(args.comparison_predictions)
                comparison_true, comparison_pred, comparison_probabilities = _triage_arrays(comparison)
                if not np.array_equal(true, comparison_true):
                    raise ValueError("Comparator true labels are not aligned")
                metrics["paired_comparison"] = _triage_comparison(
                    true,
                    pred,
                    comparison_pred,
                    probabilities,
                    comparison_probabilities,
                    n_resamples=args.n_bootstrap,
                    seed=args.seed,
                )
        elif args.task in {"cq500", "zero_shot"}:
            labels = (
                list(CQ500_LABELS)
                if args.task == "cq500"
                else _ordered_zero_shot_labels(frame, args.labels_json)
            )
            true, probabilities = _multilabel_arrays(frame, labels)
            threshold: float | np.ndarray = _load_thresholds(
                args.thresholds_json, labels, args.threshold
            )
            overall_threshold = _load_named_threshold(
                args.thresholds_json, "overall_abnormality", args.threshold
            )
            threshold_payload: dict[str, Any] | None = None
            if args.threshold_validation_predictions:
                validation = read_csv_compatible(args.threshold_validation_predictions)
                validation_true, validation_probabilities = _multilabel_arrays(validation, labels)
                threshold = fit_optimal_thresholds(
                    validation_true,
                    validation_probabilities,
                    objective=args.threshold_objective,
                )
                threshold_mapping = {
                    label: float(value)
                    for label, value in zip(labels, threshold, strict=True)
                }
                if args.task == "zero_shot":
                    validation_overall_true = _overall_ground_truth(validation)
                    if validation_overall_true is not None:
                        validation_overall_score, _validation_score_method = _overall_scores(
                            validation,
                            validation_probabilities,
                            top_k=args.overall_top_k,
                        )
                        overall_threshold = float(
                            fit_optimal_thresholds(
                                validation_overall_true.reshape(-1, 1),
                                validation_overall_score.reshape(-1, 1),
                                objective=args.threshold_objective,
                            )[0]
                        )
                        threshold_mapping["overall_abnormality"] = overall_threshold
                threshold_payload = {
                    "label_order": labels,
                    "objective": args.threshold_objective,
                    "overall_top_k": args.overall_top_k if args.task == "zero_shot" else None,
                    "thresholds": threshold_mapping,
                    "source": str(Path(args.threshold_validation_predictions).resolve()),
                }
                if args.save_thresholds:
                    destination = Path(args.save_thresholds)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(
                        json.dumps(threshold_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            metrics = (
                multilabel_metrics(true, probabilities, threshold=threshold, label_names=labels)
                if args.no_bootstrap
                else multilabel_metrics_with_ci(
                    true,
                    probabilities,
                    threshold=threshold,
                    label_names=labels,
                    n_resamples=args.n_bootstrap,
                    confidence=args.confidence,
                    seed=args.seed,
                )
            )
            if threshold_payload is not None:
                metrics["fitted_thresholds"] = threshold_payload
            if args.task == "zero_shot":
                overall_true = _overall_ground_truth(frame)
                if overall_true is not None:
                    overall_score, overall_score_method = _overall_scores(
                        frame, probabilities, top_k=args.overall_top_k
                    )
                    metrics["overall_abnormality"] = _overall_metrics(
                        overall_true,
                        overall_score,
                        threshold=overall_threshold,
                        no_bootstrap=args.no_bootstrap,
                        n_resamples=args.n_bootstrap,
                        confidence=args.confidence,
                        seed=args.seed + 1000,
                    )
                    metrics["overall_abnormality"]["score_aggregation"] = {
                        "method": overall_score_method,
                        "top_k": min(args.overall_top_k, len(labels)),
                        "threshold": overall_threshold,
                    }
            if args.comparison_predictions:
                comparison = read_csv_compatible(args.comparison_predictions)
                comparison_true, comparison_probabilities = _multilabel_arrays(comparison, labels)
                if not np.array_equal(true, comparison_true):
                    raise ValueError("Comparator true labels are not aligned")
                metrics["paired_comparison"] = _multilabel_comparison(
                    true,
                    probabilities,
                    comparison_probabilities,
                    n_resamples=args.n_bootstrap,
                    seed=args.seed,
                )
        else:
            required = {"reference_report", "generated_report"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"Generation CSV is missing columns: {sorted(missing)}")
            kwargs = dict(
                rouge_tokenization=args.rouge_tokenization,
                meteor_tokenization=args.meteor_tokenization,
                bertscore_model=None if args.skip_bertscore else args.bertscore_model,
                bertscore_device=args.bertscore_device,
                bertscore_batch_size=args.bertscore_batch_size,
                enable_cider=not args.skip_cider,
            )
            references = frame["reference_report"].fillna("").astype(str).tolist()
            hypotheses = frame["generated_report"].fillna("").astype(str).tolist()
            metrics = (
                generation_metrics(references, hypotheses, **kwargs)
                if args.no_bootstrap
                else generation_metrics_with_ci(
                    references,
                    hypotheses,
                    n_resamples=args.n_bootstrap,
                    confidence=args.confidence,
                    seed=args.seed,
                    **kwargs,
                )
            )
            if args.comparison_predictions:
                comparison = read_csv_compatible(args.comparison_predictions)
                comparison_references = (
                    comparison["reference_report"].fillna("").astype(str).tolist()
                )
                if references != comparison_references:
                    raise ValueError("Comparator reference reports are not aligned")
                metrics["paired_comparison"] = generation_paired_comparison(
                    references,
                    hypotheses,
                    comparison["generated_report"].fillna("").astype(str).tolist(),
                    n_resamples=args.n_bootstrap,
                    seed=args.seed,
                    **kwargs,
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
