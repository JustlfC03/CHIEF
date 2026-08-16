"""Versioned report-derived lexical label extraction utilities.

The rules are intentionally transparent and auditable. They reconstruct weak
labels from reports; they are not a substitute for radiologist adjudication.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Mention:
    """One lexical label mention and its local assertion status."""

    start: int
    end: int
    text: str
    left_context: str
    right_context: str
    negated: bool
    matched_negation: str | None
    uncertain_positive: bool = False
    matched_uncertainty: str | None = None


def _ordered_rules(expressions: Sequence[str]) -> list[str]:
    return sorted({str(rule) for rule in expressions if str(rule)}, key=len, reverse=True)


def find_mentions(
    text: str | None,
    label: str,
    negation_expressions: Sequence[str],
    *,
    uncertainty_expressions: Sequence[str] = (),
    left_window: int = 8,
) -> list[Mention]:
    """Return exact mentions with preceding-context assertion checks.

    Uncertainty rules have precedence over embedded shorter negation strings.
    Thus ``不除外蛛网膜下腔出血`` is retained as a positive weak-label mention,
    whereas ``除外蛛网膜下腔出血`` is negated. This resolves a common failure of
    the historical substring heuristic.
    """

    if left_window < 0:
        raise ValueError("left_window must be non-negative")
    if not label:
        raise ValueError("label must not be empty")

    normalized = "" if text is None else str(text)
    negations = _ordered_rules(negation_expressions)
    uncertainties = _ordered_rules(uncertainty_expressions)
    mentions: list[Mention] = []
    cursor = 0
    while True:
        start = normalized.find(label, cursor)
        if start < 0:
            break
        end = start + len(label)
        left = normalized[max(0, start - left_window) : start]
        right = normalized[end : min(len(normalized), end + left_window)]
        local_context = left + "|" + right
        matched_uncertainty = next(
            (rule for rule in uncertainties if rule in left or rule in right), None
        )
        matched_negation = None
        if matched_uncertainty is None:
            matched_negation = next(
                (rule for rule in negations if rule in left or rule in right), None
            )
        mentions.append(
            Mention(
                start=start,
                end=end,
                text=label,
                left_context=left,
                right_context=right,
                negated=matched_negation is not None,
                matched_negation=matched_negation,
                uncertain_positive=matched_uncertainty is not None,
                matched_uncertainty=matched_uncertainty,
            )
        )
        cursor = end
    return mentions


def extract_label(
    text: str | None,
    label: str,
    negation_expressions: Sequence[str],
    *,
    uncertainty_expressions: Sequence[str] = (),
    left_window: int = 8,
) -> int:
    """Return 1 when at least one mention is affirmative or not-excluded."""

    mentions = find_mentions(
        text,
        label,
        negation_expressions,
        uncertainty_expressions=uncertainty_expressions,
        left_window=left_window,
    )
    return int(any(not mention.negated for mention in mentions))


def extract_labels(
    text: str | None,
    labels: Iterable[str],
    negation_expressions: Sequence[str],
    *,
    uncertainty_expressions: Sequence[str] = (),
    left_window: int = 8,
) -> dict[str, int]:
    """Extract a deterministic binary mapping for the supplied ordered labels."""

    ordered = [str(label) for label in labels]
    if len(set(ordered)) != len(ordered):
        raise ValueError("labels must be unique")
    return {
        label: extract_label(
            text,
            label,
            negation_expressions,
            uncertainty_expressions=uncertainty_expressions,
            left_window=left_window,
        )
        for label in ordered
    }
