from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_WHITESPACE = re.compile(r"\s+")
_REPEATED_PUNCT = re.compile(r"([，。；：、,.!?！？])\1+")
_SENTENCE_BREAKS = re.compile(r"[\r\n]+")


def clean_report(
    text: object,
    *,
    terminology: Mapping[str, str] | None = None,
    normalize_sentences: bool = True,
) -> str:
    """Conservative, deterministic Chinese report normalization.

    De-identification must be completed before this function is called. Clinical
    synonym replacement is applied only from an explicit versioned mapping.
    """
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\x00", " ")
    if normalize_sentences:
        value = _SENTENCE_BREAKS.sub("。", value)
    else:
        value = value.replace("\r", "\n")
    value = _WHITESPACE.sub(" ", value).strip()
    value = _REPEATED_PUNCT.sub(r"\1", value)
    if terminology:
        # Longest-first replacement prevents a short synonym from modifying a
        # longer, more specific expression before it can be matched.
        for source in sorted(terminology, key=len, reverse=True):
            value = value.replace(source, terminology[source])
    return value.strip(" ;；")


@dataclass(frozen=True)
class ReportTextPreprocessor:
    terminology: Mapping[str, str] | None = None
    normalize_sentences: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, object] | None) -> "ReportTextPreprocessor":
        cfg = cfg or {}
        mapping: dict[str, str] = {}
        mapping_path = cfg.get("terminology_map_path", cfg.get("terminology_map"))
        if mapping_path:
            with Path(str(mapping_path)).expanduser().open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            raw = payload.get("mapping", payload) if isinstance(payload, dict) else None
            if not isinstance(raw, dict):
                raise ValueError("terminology_map_path must contain a JSON object")
            mapping = {str(k): str(v) for k, v in raw.items() if str(k)}
        inline = cfg.get("terminology")
        if inline:
            if not isinstance(inline, dict):
                raise ValueError("text_preprocessing.terminology must be a mapping")
            mapping.update({str(k): str(v) for k, v in inline.items() if str(k)})
        return cls(
            terminology=mapping or None,
            normalize_sentences=bool(cfg.get("normalize_sentences", True)),
        )

    def __call__(self, text: object) -> str:
        return clean_report(
            text,
            terminology=self.terminology,
            normalize_sentences=self.normalize_sentences,
        )
