from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chief.constants import CQ500_LABELS, TRIAGE_ID_TO_LABEL
from chief.data.collate import ChiefCollator
from chief.data.datasets import build_dataset
from chief.engine.checkpoint import (
    checkpoint_model_config,
    load_checkpoint,
    read_checkpoint_payload,
)
from chief.tasks.zero_shot import DEFAULT_ABNORMALITY_TEMPLATES, score_abnormalities
from chief.utils import get_logger

from .common import base_parser, initialize, load_model

LOGGER = get_logger(__name__)


def _loader(cfg: dict[str, Any], collator: ChiefCollator, *, paired: bool = False) -> DataLoader:
    data_cfg = copy.deepcopy(cfg)
    data_cfg["task"] = "retrieval" if paired else "inference"
    dataset = build_dataset(data_cfg)
    loader_cfg = cfg.get("loader", {})
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", torch.cuda.is_available())),
        collate_fn=collator,
    )


def _load_weights(
    model,
    cfg: dict[str, Any],
    checkpoint: str,
    device: torch.device,
    strict: bool,
    *,
    checkpoint_payload: Any | None = None,
) -> None:
    result = load_checkpoint(
        checkpoint,
        model=model,
        strict=strict,
        map_location=device,
        checkpoint_payload=checkpoint_payload,
    )
    if result.missing_keys or result.unexpected_keys:
        LOGGER.warning(
            "Checkpoint mismatch: missing=%s unexpected=%s",
            result.missing_keys,
            result.unexpected_keys,
        )



@lru_cache(maxsize=8)
def _read_threshold_values(path: str) -> object:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("thresholds", payload) if isinstance(payload, dict) else payload


def _load_label_thresholds(path: str | None, labels: list[str], default: float = 0.5) -> np.ndarray:
    if not path:
        return np.full(len(labels), default, dtype=float)
    values = _read_threshold_values(str(Path(path).resolve()))
    if isinstance(values, dict):
        missing = [label for label in labels if label not in values]
        if missing:
            raise ValueError(f"Threshold file is missing labels: {missing[:5]}")
        return np.asarray([float(values[label]) for label in labels], dtype=float)
    array = np.asarray(values, dtype=float)
    if array.shape != (len(labels),):
        raise ValueError("Threshold file does not match inference label order")
    return array


def _load_named_threshold(path: str | None, name: str, default: float) -> float:
    if not path:
        return float(default)
    values = _read_threshold_values(str(Path(path).resolve()))
    if isinstance(values, dict) and name in values:
        return float(values[name])
    return float(default)


def main() -> None:
    parser = base_parser("Run CHIEF inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--non-strict", action="store_true", help="Allow checkpoint key mismatch")
    args = parser.parse_args()
    cfg, device = initialize(args)
    checkpoint_payload = read_checkpoint_payload(args.checkpoint, map_location="cpu")
    saved_model_cfg = checkpoint_model_config(checkpoint_payload)
    if saved_model_cfg is not None:
        # Architecture-defining fields must come from the trained checkpoint.
        # Runtime, data and inference settings remain controlled by the YAML.
        if cfg.get("model") != saved_model_cfg:
            LOGGER.info("Restoring the exact model architecture stored in the checkpoint")
        cfg["model"] = saved_model_cfg
    model, text_tokenizer, decoder_tokenizer = load_model(cfg, device)
    _load_weights(
        model,
        cfg,
        args.checkpoint,
        device,
        strict=not args.non_strict,
        checkpoint_payload=checkpoint_payload,
    )
    model.eval()
    inference = cfg.get("inference", {})
    mode = str(inference.get("mode", cfg.get("task", "triage"))).lower()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if mode == "retrieval":
        collator = ChiefCollator(
            text_tokenizer=text_tokenizer,
            max_text_length=int(cfg.get("loader", {}).get("max_text_length", 256)),
        )
        loader = _loader(cfg, collator, paired=True)
        image_latents, text_latents, sample_ids, reports = [], [], [], []
        with torch.inference_mode():
            for batch in tqdm(loader, desc="retrieval embeddings"):
                images = batch["images"].to(device)
                text_batch = {key: value.to(device) for key, value in batch["text_batch"].items()}
                image_latents.append(model.encode_image(images).image_base.cpu())
                text_latents.append(model.encode_text(text_batch).text_base.cpu())
                sample_ids.extend(batch["sample_id"])
                reports.extend(batch["reports"])
        image = torch.nn.functional.normalize(torch.cat(image_latents), dim=-1)
        text = torch.nn.functional.normalize(torch.cat(text_latents), dim=-1)
        similarities = image @ text.t()
        top_k = min(int(inference.get("top_k", 10)), len(sample_ids))
        indices = similarities.topk(top_k, dim=1).indices
        rows = []
        ranked_rows = []
        separator = str(inference.get("ranked_report_separator", " || "))
        for query_index, query_id in enumerate(sample_ids):
            ranked_reports: list[str] = []
            for rank, candidate_index in enumerate(indices[query_index].tolist(), start=1):
                candidate_report = reports[candidate_index]
                ranked_reports.append(candidate_report)
                rows.append(
                    {
                        "query_id": query_id,
                        "rank": rank,
                        "candidate_id": sample_ids[candidate_index],
                        "similarity": float(similarities[query_index, candidate_index]),
                        "candidate_report": candidate_report,
                    }
                )
            ranked_rows.append(
                {
                    "query_id": query_id,
                    "gt_answer": reports[query_index],
                    "similar_valid_gt": separator.join(ranked_reports),
                }
            )
        pd.DataFrame(rows).to_csv(output, index=False)
        ranked_output = output.with_name(f"{output.stem}_ranked{output.suffix or '.csv'}")
        pd.DataFrame(ranked_rows).to_csv(ranked_output, index=False)
        similarity_output = output.with_suffix(".similarity.npy")
        np.save(similarity_output, similarities.numpy())
        LOGGER.info(
            "Saved retrieval results to %s, ranked report lists to %s and similarities to %s",
            output, ranked_output, similarity_output,
        )
        return

    collator = ChiefCollator()
    loader = _loader(cfg, collator)
    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    embedding_ids: list[str] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=mode):
            images = batch["images"].to(device)
            if mode == "triage":
                probabilities = model.triage_logits(images).softmax(dim=-1).cpu().numpy()
                for sample_id, values in zip(batch["sample_id"], probabilities, strict=True):
                    prediction = int(values.argmax())
                    row = {
                        "sample_id": sample_id,
                        "prediction_id": prediction,
                        "prediction": TRIAGE_ID_TO_LABEL[prediction],
                    }
                    for class_id, class_name in TRIAGE_ID_TO_LABEL.items():
                        row[f"prob_{class_name}"] = float(values[class_id])
                    rows.append(row)
            elif mode == "cq500":
                probabilities = model.cq500_logits(images).sigmoid().cpu().numpy()
                thresholds = _load_label_thresholds(
                    inference.get("thresholds_path"), list(CQ500_LABELS),
                    float(inference.get("threshold", 0.5))
                )
                for sample_id, values in zip(batch["sample_id"], probabilities, strict=True):
                    row = {"sample_id": sample_id}
                    for name, value, threshold in zip(CQ500_LABELS, values, thresholds, strict=True):
                        row[f"prob_{name}"] = float(value)
                        row[f"pred_{name}"] = int(value >= threshold)
                    rows.append(row)
            elif mode == "report_generation":
                tokenizer = decoder_tokenizer
                if tokenizer is None:
                    raise RuntimeError("Report generation requires a decoder tokenizer")
                tokens = model.generate_reports(
                    images,
                    bos_token_id=int(tokenizer.bos_token_id),
                    eos_token_id=int(tokenizer.eos_token_id),
                    max_new_tokens=int(inference.get("max_new_tokens", 256)),
                    strategy=str(inference.get("generation_strategy", "top_k_sampling")),
                    temperature=float(inference.get("generation_temperature", 0.8)),
                    top_k=int(inference.get("generation_top_k", 50)),
                )
                generated = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                rows.extend(
                    {"sample_id": sample_id, "generated_report": report}
                    for sample_id, report in zip(batch["sample_id"], generated, strict=True)
                )
            elif mode == "embedding":
                latent = model.encode_image(images).image_base.cpu().numpy()
                embeddings.append(latent)
                embedding_ids.extend(batch["sample_id"])
            elif mode == "zero_shot":
                tokenizer = decoder_tokenizer
                if tokenizer is None:
                    raise RuntimeError("Zero-shot inference requires a decoder tokenizer")
                labels = list(inference.get("labels", []))
                if not labels:
                    raise ValueError("inference.labels is required for zero_shot mode")
                templates = tuple(inference.get("templates", DEFAULT_ABNORMALITY_TEMPLATES))
                probabilities = (
                    score_abnormalities(
                        model,
                        images,
                        tokenizer,
                        labels,
                        templates=templates,
                        label_display_names=inference.get("label_display_names"),
                        yes_text=str(inference.get("yes_text", "是")),
                        no_text=str(inference.get("no_text", "否")),
                        temperature=float(inference.get("temperature", 2.0)),
                        baseline_calibration=bool(inference.get("baseline_calibration", True)),
                    )
                    .cpu()
                    .numpy()
                )
                thresholds_path = inference.get("thresholds_path")
                thresholds = _load_label_thresholds(
                    thresholds_path, labels, float(inference.get("threshold", 0.5))
                )
                overall_top_k = min(
                    max(1, int(inference.get("overall_top_k", 3))), len(labels)
                )
                overall_threshold = _load_named_threshold(
                    thresholds_path,
                    "overall_abnormality",
                    float(inference.get("overall_threshold", inference.get("threshold", 0.5))),
                )
                for sample_id, values in zip(batch["sample_id"], probabilities, strict=True):
                    row = {"sample_id": sample_id}
                    for name, value, threshold in zip(labels, values, thresholds, strict=True):
                        row[f"prob_{name}"] = float(value)
                        row[f"pred_{name}"] = int(value >= threshold)
                    overall_score = float(np.mean(np.partition(values, -overall_top_k)[-overall_top_k:]))
                    row["prob_overall_abnormality"] = overall_score
                    row["pred_overall_abnormality"] = int(overall_score >= overall_threshold)
                    rows.append(row)
            else:
                raise ValueError(f"Unsupported inference.mode={mode!r}")
    if mode == "embedding":
        np.savez_compressed(
            output, sample_id=np.asarray(embedding_ids), embedding=np.concatenate(embeddings)
        )
    else:
        pd.DataFrame(rows).to_csv(output, index=False)
    LOGGER.info("Saved %s output to %s", mode, output.resolve())


if __name__ == "__main__":
    main()
