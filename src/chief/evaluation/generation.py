from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .bootstrap import bootstrap_confidence_interval


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def tokenize_text(text: str, mode: str = "char") -> list[str]:
    text = normalize_text(text)
    if mode == "char":
        return [char for char in text if not char.isspace()]
    if mode == "whitespace":
        return text.split()
    if mode == "jieba":
        try:
            import jieba
        except ImportError as exc:
            raise ImportError("Install requirements-eval.txt to use jieba tokenization") from exc
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    raise ValueError(f"Unsupported tokenization mode {mode!r}")


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, start=1):
            current.append(previous[index - 1] + 1 if left == right else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    lcs = _lcs_length(reference, hypothesis)
    precision = lcs / len(hypothesis)
    recall = lcs / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _corpus_bleu(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError("Install requirements-eval.txt to compute SacreBLEU") from exc
    return float(
        sacrebleu.corpus_bleu(
            [normalize_text(text) for text in hypotheses],
            [[normalize_text(text) for text in references]],
            tokenize="zh",
            smooth_method="exp",
            use_effective_order=True,
        ).score
        / 100.0
    )


def _meteor(reference: str, hypothesis: str, tokenization: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:
        raise ImportError("Install requirements-eval.txt to compute METEOR") from exc
    ref_tokens = tokenize_text(reference, tokenization)
    hyp_tokens = tokenize_text(hypothesis, tokenization)
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return float(meteor_score([ref_tokens], hyp_tokens))


def _cider(references: Sequence[str], hypotheses: Sequence[str]) -> tuple[float, np.ndarray]:
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as exc:
        raise ImportError("Install requirements-eval.txt to compute the manuscript CIDEr") from exc
    gts = {
        index: [" ".join(tokenize_text(reference, "char"))]
        for index, reference in enumerate(references)
    }
    res = {
        index: [" ".join(tokenize_text(hypothesis, "char"))]
        for index, hypothesis in enumerate(hypotheses)
    }
    score, per_sample = Cider().compute_score(gts, res)
    return float(score), np.asarray(per_sample, dtype=float)


def _bertscore(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    model: str,
    device: str | None,
    batch_size: int,
) -> np.ndarray:
    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError("Install requirements-eval.txt to compute BERTScore") from exc
    _precision, _recall, f1 = score(
        list(hypotheses),
        list(references),
        lang="zh",
        model_type=model,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    return f1.detach().cpu().numpy().astype(float)


def generation_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    rouge_tokenization: str = "jieba",
    meteor_tokenization: str = "char",
    bertscore_model: str | None = "bert-base-chinese",
    bertscore_device: str | None = None,
    bertscore_batch_size: int = 32,
    enable_cider: bool = True,
) -> dict[str, float]:
    """Compute the manuscript report-generation endpoints.

    The implementations mirror the final historical evaluator: SacreBLEU with
    Chinese tokenization, ROUGE-L F1, NLTK METEOR, pycocoevalcap CIDEr and
    BERTScore using ``bert-base-chinese`` by default.
    """
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must be non-empty and aligned")
    result = {"bleu": _corpus_bleu(references, hypotheses)}
    rouge = [
        _rouge_l(tokenize_text(ref, rouge_tokenization), tokenize_text(hyp, rouge_tokenization))
        for ref, hyp in zip(references, hypotheses, strict=True)
    ]
    meteor = [
        _meteor(ref, hyp, meteor_tokenization)
        for ref, hyp in zip(references, hypotheses, strict=True)
    ]
    result["rouge_l"] = float(np.mean(rouge))
    result["meteor"] = float(np.mean(meteor))
    if enable_cider:
        result["cider"] = _cider(references, hypotheses)[0]
    if bertscore_model is not None:
        result["bertscore_f1"] = float(
            _bertscore(
                references,
                hypotheses,
                model=bertscore_model,
                device=bertscore_device,
                batch_size=bertscore_batch_size,
            ).mean()
        )
    return result


def generation_metrics_with_ci(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    rouge_tokenization: str = "jieba",
    meteor_tokenization: str = "char",
    bertscore_model: str | None = "bert-base-chinese",
    bertscore_device: str | None = None,
    bertscore_batch_size: int = 32,
    enable_cider: bool = True,
) -> dict[str, Any]:
    """Compute report-generation metrics with bootstrap CIs (1,000 by default).

    BLEU is recalculated for each resampled corpus. ROUGE-L, METEOR and
    BERTScore use resampled per-examination scores. For CIDEr, the corpus point
    estimate and per-examination CIDEr scores are computed once; its interval
    is the percentile bootstrap of the fixed per-examination scores, matching
    the final manuscript analysis rather than rerunning CIDEr in every draw.
    """
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must be non-empty and aligned")
    refs = np.asarray(list(references), dtype=object)
    hyps = np.asarray(list(hypotheses), dtype=object)
    metrics: dict[str, dict[str, float]] = {}

    estimate, lower, upper = bootstrap_confidence_interval(
        lambda r, h: _corpus_bleu(r.tolist(), h.tolist()),
        refs,
        hyps,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )
    metrics["bleu"] = {"estimate": estimate, "ci_lower": lower, "ci_upper": upper}

    rouge_values = np.asarray([
        _rouge_l(tokenize_text(ref, rouge_tokenization), tokenize_text(hyp, rouge_tokenization))
        for ref, hyp in zip(refs, hyps, strict=True)
    ])
    meteor_values = np.asarray([
        _meteor(ref, hyp, meteor_tokenization) for ref, hyp in zip(refs, hyps, strict=True)
    ])
    for offset, (name, values) in enumerate((('rouge_l', rouge_values), ('meteor', meteor_values)), start=1):
        estimate, lower, upper = bootstrap_confidence_interval(
            lambda x: float(np.mean(x)), values,
            n_resamples=n_resamples, confidence=confidence, seed=seed + offset
        )
        metrics[name] = {"estimate": estimate, "ci_lower": lower, "ci_upper": upper}

    if enable_cider:
        cider_estimate, cider_values = _cider(refs.tolist(), hyps.tolist())
        _mean, lower, upper = bootstrap_confidence_interval(
            lambda x: float(np.mean(x)), cider_values,
            n_resamples=n_resamples, confidence=confidence, seed=seed + 3
        )
        metrics["cider"] = {"estimate": cider_estimate, "ci_lower": lower, "ci_upper": upper}
    if bertscore_model is not None:
        values = _bertscore(
            refs.tolist(), hyps.tolist(), model=bertscore_model,
            device=bertscore_device, batch_size=bertscore_batch_size
        )
        estimate, lower, upper = bootstrap_confidence_interval(
            lambda x: float(np.mean(x)), values,
            n_resamples=n_resamples, confidence=confidence, seed=seed + 4
        )
        metrics["bertscore_f1"] = {"estimate": estimate, "ci_lower": lower, "ci_upper": upper}
    return {
        "n_samples": len(refs),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "metrics": metrics,
    }


def generation_paired_comparison(
    references: Sequence[str],
    first_hypotheses: Sequence[str],
    second_hypotheses: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    rouge_tokenization: str = "jieba",
    meteor_tokenization: str = "char",
    bertscore_model: str | None = "bert-base-chinese",
    bertscore_device: str | None = None,
    bertscore_batch_size: int = 32,
    enable_cider: bool = True,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap comparison of two generated-report systems."""
    if not references or len(references) != len(first_hypotheses) or len(references) != len(second_hypotheses):
        raise ValueError("references and both hypothesis lists must be non-empty and aligned")
    refs = np.asarray(list(references), dtype=object)
    first = np.asarray(list(first_hypotheses), dtype=object)
    second = np.asarray(list(second_hypotheses), dtype=object)
    rng = np.random.default_rng(seed)

    def summarize(observed: float, differences: list[float]) -> dict[str, float]:
        values = np.asarray(differences, dtype=float)
        p_value = (
            min(1.0, 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean())))
            if values.size
            else float("nan")
        )
        return {"difference": float(observed), "p_value": p_value}

    result: dict[str, dict[str, float]] = {}
    observed_bleu = _corpus_bleu(refs.tolist(), first.tolist()) - _corpus_bleu(refs.tolist(), second.tolist())
    bleu_differences: list[float] = []
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, len(refs), size=len(refs))
        bleu_differences.append(
            _corpus_bleu(refs[indices].tolist(), first[indices].tolist())
            - _corpus_bleu(refs[indices].tolist(), second[indices].tolist())
        )
    result["bleu"] = summarize(observed_bleu, bleu_differences)

    first_rouge = np.asarray([
        _rouge_l(tokenize_text(ref, rouge_tokenization), tokenize_text(hyp, rouge_tokenization))
        for ref, hyp in zip(refs, first, strict=True)
    ])
    second_rouge = np.asarray([
        _rouge_l(tokenize_text(ref, rouge_tokenization), tokenize_text(hyp, rouge_tokenization))
        for ref, hyp in zip(refs, second, strict=True)
    ])
    first_meteor = np.asarray([
        _meteor(ref, hyp, meteor_tokenization) for ref, hyp in zip(refs, first, strict=True)
    ])
    second_meteor = np.asarray([
        _meteor(ref, hyp, meteor_tokenization) for ref, hyp in zip(refs, second, strict=True)
    ])
    for name, first_values, second_values in (
        ("rouge_l", first_rouge, second_rouge),
        ("meteor", first_meteor, second_meteor),
    ):
        differences = first_values - second_values
        bootstrap_differences = [
            float(differences[rng.integers(0, len(differences), size=len(differences))].mean())
            for _ in range(int(n_resamples))
        ]
        result[name] = summarize(float(differences.mean()), bootstrap_differences)

    if enable_cider:
        first_score, first_values = _cider(refs.tolist(), first.tolist())
        second_score, second_values = _cider(refs.tolist(), second.tolist())
        differences = first_values - second_values
        bootstrap_differences = [
            float(differences[rng.integers(0, len(differences), size=len(differences))].mean())
            for _ in range(int(n_resamples))
        ]
        result["cider"] = summarize(first_score - second_score, bootstrap_differences)
    if bertscore_model is not None:
        first_values = _bertscore(
            refs.tolist(), first.tolist(), model=bertscore_model,
            device=bertscore_device, batch_size=bertscore_batch_size
        )
        second_values = _bertscore(
            refs.tolist(), second.tolist(), model=bertscore_model,
            device=bertscore_device, batch_size=bertscore_batch_size
        )
        differences = first_values - second_values
        bootstrap_differences = [
            float(differences[rng.integers(0, len(differences), size=len(differences))].mean())
            for _ in range(int(n_resamples))
        ]
        result["bertscore_f1"] = summarize(float(differences.mean()), bootstrap_differences)
    return result
