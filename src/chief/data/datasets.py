from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from chief.constants import CQ500_LABELS

from .manifests import load_manifest, validate_manifest
from .preprocessing import ModelInputTransform
from .text import ReportTextPreprocessor
from .volume_io import load_volume_data


class _BaseCTDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        preprocessor: ModelInputTransform,
        text_preprocessor: ReportTextPreprocessor | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.text_preprocessor = text_preprocessor or ReportTextPreprocessor()

    def __len__(self) -> int:
        return len(self.frame)

    def _base_item(self, index: int) -> tuple[pd.Series, dict[str, Any]]:
        row = self.frame.iloc[index]
        loaded = load_volume_data(row["image_path"])
        volume = self.preprocessor(loaded.tensor, loaded.spacing_zyx)
        return row, {
            "sample_id": str(row["sample_id"]),
            "image": volume,
            "image_path": str(row["image_path"]),
        }


class PairedCTReportDataset(_BaseCTDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row, item = self._base_item(index)
        item["report"] = self.text_preprocessor(row["report"])
        return item


class TriageDataset(_BaseCTDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row, item = self._base_item(index)
        item["label"] = torch.tensor(int(row["triage_label"]), dtype=torch.long)
        return item


class CQ500Dataset(_BaseCTDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row, item = self._base_item(index)
        item["labels"] = torch.tensor([float(row[label]) for label in CQ500_LABELS])
        return item


class InferenceDataset(_BaseCTDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row, item = self._base_item(index)
        if "report" in row and pd.notna(row["report"]):
            item["report"] = self.text_preprocessor(row["report"])
        return item


def build_dataset(cfg: dict[str, Any], split: str | None = None) -> Dataset[dict[str, Any]]:
    data_cfg = cfg["data"]
    task = str(cfg["task"]).lower()
    manifest = data_cfg.get("manifest")
    if not manifest:
        raise ValueError("data.manifest is required")
    frame = load_manifest(manifest, data_cfg.get("root"))
    if split and "split" in frame:
        frame = frame.loc[frame["split"].astype(str).str.lower() == split.lower()].copy()
        if frame.empty:
            raise ValueError(f"No samples found for split={split!r}")
    validation_task = (
        task
        if task in {"pretrain", "report_generation", "retrieval", "triage", "cq500"}
        else "inference"
    )
    frame = validate_manifest(
        frame,
        validation_task,
        check_files=bool(data_cfg.get("check_files", True)),
        allow_empty_report=bool(data_cfg.get("allow_empty_report", False)),
        triage_class_order=list(data_cfg.get("triage_class_order", [])) or None,
    )
    preprocessor = ModelInputTransform.from_config(data_cfg.get("preprocessing"))
    text_preprocessor = ReportTextPreprocessor.from_config(data_cfg.get("text_preprocessing"))
    if task in {"pretrain", "report_generation", "retrieval"}:
        return PairedCTReportDataset(frame, preprocessor, text_preprocessor)
    if task == "triage":
        return TriageDataset(frame, preprocessor, text_preprocessor)
    if task == "cq500":
        return CQ500Dataset(frame, preprocessor, text_preprocessor)
    return InferenceDataset(frame, preprocessor, text_preprocessor)
