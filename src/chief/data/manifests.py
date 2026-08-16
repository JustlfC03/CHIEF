from __future__ import annotations

from pathlib import Path

import pandas as pd

from chief.constants import CQ500_LABELS, MANIFEST_ALIASES, TRIAGE_LABEL_TO_ID


class ManifestError(ValueError):
    pass


def read_csv_compatible(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read UTF-8 or common Chinese-encoded CSV files.

    The original evaluation tables include both UTF-8 and GB18030 files.  The
    fallback is intentionally limited to decoding only; parsing errors are not
    swallowed.
    """
    source = Path(path)
    try:
        return pd.read_csv(source, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(source, encoding="gb18030", **kwargs)


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    for canonical, aliases in MANIFEST_ALIASES.items():
        for alias in aliases:
            original = lower.get(alias.lower())
            if original is not None:
                rename[original] = canonical
                break
    return frame.rename(columns=rename)


def _normalize_triage(value: object, class_order: list[str] | None = None) -> int:
    if pd.isna(value):
        raise ManifestError("Missing triage label")
    if isinstance(value, (int, float)) and int(value) == value:
        label = int(value)
        if class_order is None:
            raise ManifestError(
                "Numeric triage labels are ambiguous. Set data.triage_class_order explicitly."
            )
        if label < 0 or label >= len(class_order):
            raise ManifestError(f"Numeric triage label {label} is outside class_order")
        class_name = str(class_order[label]).strip().lower().replace("_", "-")
        if class_name not in TRIAGE_LABEL_TO_ID:
            raise ManifestError(f"Unknown class in triage_class_order: {class_name!r}")
        return TRIAGE_LABEL_TO_ID[class_name]
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "normal": "negative",
        "neg": "negative",
        "non-emergency positive": "non-emergency-positive",
        "non emergency positive": "non-emergency-positive",
        "nonemergency-positive": "non-emergency-positive",
        "nonurgent-positive": "non-emergency-positive",
        "urgent-positive": "positive",
        "urgent positive": "positive",
        "pos": "positive",
    }
    text = aliases.get(text, text)
    if text not in TRIAGE_LABEL_TO_ID:
        raise ManifestError(f"Unknown triage label {value!r}")
    return TRIAGE_LABEL_TO_ID[text]


def load_manifest(path: str | Path, root: str | Path | None = None) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = read_csv_compatible(source)
    elif suffix in {".json", ".jsonl"}:
        frame = pd.read_json(source, lines=suffix == ".jsonl")
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise ManifestError(f"Unsupported manifest format: {source}")
    frame = _canonicalize_columns(frame)
    base = Path(root).expanduser().resolve() if root else source.parent
    if "image_path" in frame:

        def resolve_path(value: object) -> str:
            candidate = Path(str(value)).expanduser()
            return str(candidate if candidate.is_absolute() else (base / candidate).resolve())

        frame["image_path"] = frame["image_path"].map(resolve_path)
    if "sample_id" not in frame:
        frame["sample_id"] = [f"sample-{index:06d}" for index in range(len(frame))]
    return frame


def required_columns(task: str) -> list[str]:
    common = ["sample_id", "image_path"]
    if task in {"pretrain", "report_generation", "retrieval"}:
        return [*common, "report"]
    if task == "triage":
        return [*common, "triage_label"]
    if task == "cq500":
        return [*common, *CQ500_LABELS]
    if task in {"inference", "zero_shot"}:
        return common
    raise ManifestError(f"Unknown task {task!r}")


def validate_manifest(
    frame: pd.DataFrame,
    task: str,
    *,
    check_files: bool = True,
    allow_empty_report: bool = False,
    triage_class_order: list[str] | None = None,
) -> pd.DataFrame:
    missing = [column for column in required_columns(task) if column not in frame.columns]
    if missing:
        raise ManifestError(f"Manifest is missing columns for task={task}: {missing}")
    frame = frame.copy()
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ManifestError(f"sample_id must be unique; examples={duplicates}")
    if check_files:
        missing_paths = [value for value in frame["image_path"] if not Path(str(value)).exists()]
        if missing_paths:
            raise ManifestError(
                f"{len(missing_paths)} image paths do not exist; first={missing_paths[0]}"
            )
    if "report" in frame and not allow_empty_report:
        empty = frame["report"].fillna("").astype(str).str.strip().eq("")
        if empty.any():
            raise ManifestError(f"Found {int(empty.sum())} empty reports")
    if task == "triage":
        frame["triage_label"] = frame["triage_label"].map(lambda value: _normalize_triage(value, triage_class_order))
    if task == "cq500":
        for label in CQ500_LABELS:
            values = pd.to_numeric(frame[label], errors="coerce")
            if values.isna().any() or not values.isin([0, 1]).all():
                raise ManifestError(f"CQ500 label {label!r} must contain only 0/1")
            frame[label] = values.astype(int)
    return frame


def write_manifest(frame: pd.DataFrame, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".jsonl":
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ManifestError("Output manifest must be .csv or .jsonl")
