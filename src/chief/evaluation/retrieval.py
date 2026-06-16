from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any

import numpy as np

_WHITESPACE_RE = re.compile(r"[ \t\r\n\u3000]+")
_PUNCTUATION_RE = re.compile(
    r"[`~!@#$%^&*()_\-+=\[\]{}\\|;:'\",<>/?，。；：、“”‘’！￥…（）【】《》？·]+"
)


def normalize_report_text(value: Any) -> str:
    """Normalize report text for retrieval relevance matching."""
    if value is None:
        return ""
    text = str(value).strip().replace("\u3000", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = text.replace("；", ";").replace("，", ",").replace("。", ".").replace("：", ":")
    text = _PUNCTUATION_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def report_text_match(
    reference: Any,
    candidate: Any,
    *,
    mode: str = "exact",
    fuzzy_threshold: float = 0.99,
    min_length: int = 8,
) -> bool:
    """Determine whether two reports should be treated as retrieval-relevant.

    ``exact`` is the default paper-facing setting for paired image-to-report
    retrieval. ``contains`` and ``fuzzy`` reproduce the matching options found
    in the historical evaluation scripts and must be reported explicitly when
    used.
    """
    reference_n = normalize_report_text(reference)
    candidate_n = normalize_report_text(candidate)
    if not reference_n or not candidate_n:
        return False
    if mode not in {"exact", "contains", "fuzzy"}:
        raise ValueError(f"Unsupported match mode={mode!r}")
    if reference_n == candidate_n:
        return True
    if mode == "exact":
        return False
    # Match the historical evaluator exactly: the minimum-length guard is
    # applied to the reference report, while containment is accepted in either
    # direction.
    if len(reference_n) >= int(min_length) and (
        reference_n in candidate_n or candidate_n in reference_n
    ):
        return True
    if mode == "contains":
        return False
    return SequenceMatcher(None, reference_n, candidate_n).ratio() >= float(fuzzy_threshold)


def text_relevance_matrix(
    query_reports: Sequence[Any],
    candidate_reports: Sequence[Any],
    *,
    mode: str = "exact",
    fuzzy_threshold: float = 0.99,
    min_length: int = 8,
) -> np.ndarray:
    """Build a query-by-candidate relevance matrix from report text."""
    relevance = np.zeros((len(query_reports), len(candidate_reports)), dtype=bool)
    for i, query in enumerate(query_reports):
        for j, candidate in enumerate(candidate_reports):
            relevance[i, j] = report_text_match(
                query,
                candidate,
                mode=mode,
                fuzzy_threshold=fuzzy_threshold,
                min_length=min_length,
            )
    return relevance


def first_match_ranks(
    references: Sequence[Any],
    ranked_candidates: Sequence[Sequence[Any]],
    *,
    mode: str = "exact",
    fuzzy_threshold: float = 0.99,
    min_length: int = 8,
    miss_rank: float | None = None,
) -> np.ndarray:
    """Return 1-based first relevant ranks.

    By default a miss is represented as ``+inf`` and contributes zero to MRR.
    Set ``miss_rank=max_candidates+1`` only when reproducing the historical
    top-k-truncated evaluator used for the manuscript retrieval tables.
    """
    if len(references) != len(ranked_candidates):
        raise ValueError("references and ranked_candidates must have equal length")
    ranks: list[float] = []
    for reference, candidates in zip(references, ranked_candidates, strict=True):
        rank = float("inf") if miss_rank is None else float(miss_rank)
        for index, candidate in enumerate(candidates):
            if report_text_match(
                reference,
                candidate,
                mode=mode,
                fuzzy_threshold=fuzzy_threshold,
                min_length=min_length,
            ):
                rank = float(index + 1)
                break
        ranks.append(rank)
    return np.asarray(ranks, dtype=float)


def retrieval_metrics_from_ranks(
    ranks: Any,
    *,
    ks: Sequence[int] = (1, 5, 10, 50, 100, 200),
) -> dict[str, float]:
    rank_array = np.asarray(ranks, dtype=float).reshape(-1)
    if rank_array.size == 0 or np.any(rank_array <= 0) or np.any(np.isnan(rank_array)):
        raise ValueError("ranks must contain positive 1-based ranks or +inf for misses")
    reciprocal = np.divide(
        1.0,
        rank_array,
        out=np.zeros_like(rank_array, dtype=float),
        where=np.isfinite(rank_array),
    )
    result = {"mrr": float(np.mean(reciprocal))}
    for k in ks:
        cutoff = int(k)
        hits = rank_array <= cutoff
        result[f"recall@{k}"] = float(np.mean(hits))
        discounted = np.where(hits, 1.0 / np.log2(rank_array + 1.0), 0.0)
        result[f"ndcg@{k}"] = float(np.mean(discounted))
    return result


def retrieval_metrics_from_ranks_with_ci(
    ranks: Any,
    *,
    ks: Sequence[int] = (1, 5, 10, 50, 100, 200),
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    from .bootstrap import bootstrap_metric_dict

    rank_array = np.asarray(ranks, dtype=float).reshape(-1)
    metrics = bootstrap_metric_dict(
        lambda sampled: retrieval_metrics_from_ranks(sampled, ks=ks),
        rank_array,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "n_queries": int(rank_array.size),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "metrics": metrics,
    }


def retrieval_metrics(
    similarities: Any,
    *,
    relevance: Any | None = None,
    ks: Sequence[int] = (1, 5, 10, 50, 100, 200),
) -> dict[str, float]:
    scores = np.asarray(similarities, dtype=float)
    if scores.ndim != 2:
        raise ValueError("similarities must be [queries,candidates]")
    queries, candidates = scores.shape
    if relevance is None:
        if queries != candidates:
            raise ValueError("Diagonal relevance requires a square similarity matrix")
        relevant = np.eye(queries, dtype=bool)
    else:
        relevant = np.asarray(relevance, dtype=bool)
        if relevant.shape != scores.shape:
            raise ValueError("relevance shape must equal similarities shape")
    if np.any(relevant.sum(axis=1) == 0):
        raise ValueError("Every retrieval query must have at least one relevant candidate")

    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_relevance = np.take_along_axis(relevant, order, axis=1)
    first_positions = np.argmax(ranked_relevance, axis=1) + 1
    result = {"mrr": float(np.mean(1.0 / first_positions))}
    for k in ks:
        cutoff = min(int(k), candidates)
        hit = ranked_relevance[:, :cutoff].any(axis=1)
        result[f"recall@{k}"] = float(hit.mean())
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        dcg = (ranked_relevance[:, :cutoff] * discounts).sum(axis=1)
        relevant_counts = relevant.sum(axis=1)
        ideal = np.asarray(
            [discounts[: min(int(count), cutoff)].sum() for count in relevant_counts],
            dtype=float,
        )
        ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg, dtype=float), where=ideal > 0)
        result[f"ndcg@{k}"] = float(ndcg.mean())
    return result


def retrieval_metrics_with_ci(
    similarities: Any,
    *,
    relevance: Any | None = None,
    ks: Sequence[int] = (1, 5, 10, 50, 100, 200),
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Retrieval metrics with query-level percentile bootstrap CIs."""
    from .bootstrap import bootstrap_metric_dict

    scores = np.asarray(similarities, dtype=float)
    if scores.ndim != 2:
        raise ValueError("similarities must be [queries,candidates]")
    if relevance is None:
        if scores.shape[0] != scores.shape[1]:
            raise ValueError("Diagonal relevance requires a square similarity matrix")
        relevant = np.eye(scores.shape[0], dtype=bool)
    else:
        relevant = np.asarray(relevance, dtype=bool)
        if relevant.shape != scores.shape:
            raise ValueError("relevance shape must equal similarities shape")
    if np.any(relevant.sum(axis=1) == 0):
        raise ValueError("Every retrieval query must have at least one relevant candidate")

    metrics = bootstrap_metric_dict(
        lambda sampled_scores, sampled_relevance: retrieval_metrics(
            sampled_scores, relevance=sampled_relevance, ks=ks
        ),
        scores,
        relevant,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "n_queries": scores.shape[0],
        "n_candidates": scores.shape[1],
        "confidence": confidence,
        "n_resamples": n_resamples,
        "metrics": metrics,
    }
