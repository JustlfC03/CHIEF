from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from chief.constants import CQ500_LABELS
from chief.models.generation import conditional_lm_loss
from chief.models.losses import AsymmetricLossMultiLabel, hierarchy_consistency_loss


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_nested(item, device) for key, item in value.items()}
    return value


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return _move_nested(batch, device)


def compute_task_loss(
    model, batch: dict[str, Any], cfg: dict[str, Any]
) -> tuple[Tensor, dict[str, Any]]:
    task = str(cfg["task"]).lower()
    images = batch["images"]
    if task == "pretrain":
        output = model.forward_pretrain(
            images,
            batch["text_batch"],
            batch.get("decoder_batch"),
            generation_weight=float(cfg.get("loss", {}).get("generation_weight", 1.0)),
        )
        return output.loss, {
            "loss": float(output.loss.detach()),
            "contrastive": float(output.contrastive_loss.detach()),
            "generation": float(output.generation_loss.detach()),
            "decorrelation": float(output.decorrelation_loss.detach()),
            "logit_scale": float(output.logit_scale),
        }
    if task == "report_generation":
        if model.decoder is None or model.prefix_projector is None:
            raise RuntimeError("Report generation requires model.use_decoder=true")
        image = model.encode_image(images)
        prefix = model.prefix_projector(image.image_base)
        decoder_batch = batch["decoder_batch"]
        generation_loss, _ = conditional_lm_loss(
            model.decoder,
            prefix,
            decoder_batch["input_ids"],
            decoder_batch["attention_mask"],
            int(decoder_batch["pad_token_id"]),
        )
        weight = float(cfg.get("loss", {}).get("generation_weight", 1.0))
        loss = weight * generation_loss
        return loss, {
            "loss": float(loss.detach()),
            "generation": float(generation_loss.detach()),
        }
    if task == "triage":
        logits = model.triage_logits(images)
        loss_cfg = cfg.get("loss", {})
        weights = loss_cfg.get("class_weights")
        class_weights = (
            torch.tensor(weights, device=logits.device, dtype=logits.dtype) if weights else None
        )
        loss = F.cross_entropy(
            logits,
            batch["labels"],
            weight=class_weights,
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        )
        accuracy = logits.argmax(dim=-1).eq(batch["labels"]).float().mean()
        return loss, {
            "loss": float(loss.detach()),
            "accuracy": float(accuracy.detach()),
            "__logits__": logits.detach(),
            "__targets__": batch["labels"].detach(),
        }
    if task == "cq500":
        logits = model.cq500_logits(images)
        loss_cfg = cfg.get("loss", {})
        criterion = AsymmetricLossMultiLabel(
            gamma_neg=float(loss_cfg.get("gamma_neg", 4.0)),
            gamma_pos=float(loss_cfg.get("gamma_pos", 1.0)),
            clip=float(loss_cfg.get("clip", 0.05)),
        )
        main = criterion(logits, batch["labels"])
        hierarchy = hierarchy_consistency_loss(logits, CQ500_LABELS)
        weight = float(loss_cfg.get("hierarchy_weight", 0.1))
        loss = main + weight * hierarchy
        return loss, {
            "loss": float(loss.detach()),
            "asymmetric": float(main.detach()),
            "hierarchy": float(hierarchy.detach()),
            "__logits__": logits.detach(),
            "__targets__": batch["labels"].detach(),
        }
    raise ValueError(f"Training is not defined for task={task!r}")
