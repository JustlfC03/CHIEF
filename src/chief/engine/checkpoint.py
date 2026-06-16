from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from chief import __version__
from chief.constants import CQ500_LABELS, TRIAGE_ID_TO_LABEL

CHECKPOINT_FORMAT_VERSION = 2
CANONICAL_TRIAGE_ORDER = [TRIAGE_ID_TO_LABEL[index] for index in sorted(TRIAGE_ID_TO_LABEL)]
LEGACY_TRIAGE_ORDER = ["negative", "positive", "non-emergency-positive"]


@dataclass
class CheckpointLoadResult:
    epoch: int
    global_step: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    metadata: dict[str, Any]
    legacy_migrated: bool = False
    ignored_keys: list[str] | None = None
    shape_mismatched_keys: list[str] | None = None


def _git_commit() -> str | None:
    value = os.environ.get("CHIEF_GIT_COMMIT")
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _release_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "chief_version": __version__,
        "git_commit": _git_commit(),
        "triage_class_order": CANONICAL_TRIAGE_ORDER,
        "cq500_label_order": list(CQ500_LABELS),
        "model_config": config.get("model", {}),
        "preprocessing_config": config.get("data", {}).get("preprocessing", {}),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, Any] | None = None,
    best_val: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format": "chief",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "config": config,
        "release_metadata": _release_metadata(config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": metrics or {},
        "best_val": float(best_val) if best_val is not None else None,
        "torch_version": torch.__version__,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    _atomic_torch_save(payload, Path(path))




def read_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a checkpoint once for architecture inspection and weight loading."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return torch.load(source, map_location=map_location, weights_only=False)


def checkpoint_model_config(checkpoint: Any) -> dict[str, Any] | None:
    """Return the exact saved model configuration for a v2 checkpoint.

    Historical research packages did not store a reliable architecture
    configuration, so callers must keep using an explicitly matching YAML for
    those packages.
    """
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "chief":
        return None
    release = checkpoint.get("release_metadata")
    if isinstance(release, dict) and isinstance(release.get("model_config"), dict):
        return dict(release["model_config"])
    config = checkpoint.get("config")
    if isinstance(config, dict) and isinstance(config.get("model"), dict):
        return dict(config["model"])
    return None

def _is_legacy_package(checkpoint: Any) -> bool:
    return (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("model"), dict)
        and checkpoint.get("format") != "chief"
        and any(
            key in checkpoint
            for key in (
                "image_to_gpt2_proj",
                "gpt2_model",
                "gen_hidden_proj",
                "classifier_head",
                "classifier_head_joint",
                "optim",
            )
        )
    )


def _map_prefixed(destination: dict[str, Tensor], source: Any, prefix: str) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if isinstance(value, Tensor):
            destination[f"{prefix}{key}"] = value


def _map_legacy_classifier_heads(
    state: dict[str, Tensor],
    checkpoint: dict[str, Any],
    expected: dict[str, Tensor],
    model: nn.Module,
) -> None:
    """Explicitly migrate only semantically compatible historical heads.

    Historical joint heads were trained with several incompatible feature
    compositions and cannot be identified reliably from tensor shapes alone.
    They are therefore never imported automatically. An explicitly image-only
    triage head may be migrated only into an image-only v2.0 probe; the CQ500
    head is migrated only when its shape is compatible.
    """

    def map_head(source_name: str, target_weight: str, target_bias: str, *, reorder: bool) -> bool:
        source = checkpoint.get(source_name)
        if not isinstance(source, dict):
            return False
        weight = source.get("weight")
        bias = source.get("bias")
        if not isinstance(weight, Tensor) or target_weight not in expected:
            return False
        if tuple(weight.shape) != tuple(expected[target_weight].shape):
            return False
        if reorder:
            order = torch.tensor([0, 2, 1], dtype=torch.long, device=weight.device)
            weight = weight.index_select(0, order)
            if isinstance(bias, Tensor) and bias.shape == (3,):
                bias = bias.index_select(0, order)
        state[target_weight] = weight
        if (
            isinstance(bias, Tensor)
            and target_bias in expected
            and tuple(bias.shape) == tuple(expected[target_bias].shape)
        ):
            state[target_bias] = bias
        return True

    if not bool(getattr(model, "triage_use_generation_features", False)):
        map_head(
            "classifier_head",
            "triage_probe.classifier.1.weight",
            "triage_probe.classifier.1.bias",
            reorder=True,
        )
    map_head(
        "classifier_head",
        "cq500_probe.classifier.1.weight",
        "cq500_probe.classifier.1.bias",
        reorder=False,
    )

def _upgrade_legacy_package(
    checkpoint: dict[str, Any],
    model: nn.Module,
    *,
    migrate_task_heads: bool = False,
) -> dict[str, Tensor]:
    """Migrate the original research-package layout into the v2.0 state layout."""
    state = {
        key: value
        for key, value in checkpoint["model"].items()
        if isinstance(key, str) and isinstance(value, Tensor)
    }
    _map_prefixed(state, checkpoint.get("image_to_gpt2_proj"), "prefix_projector.")
    _map_prefixed(state, checkpoint.get("gpt2_model"), "decoder.")
    _map_prefixed(state, checkpoint.get("gen_hidden_proj"), "generation_feature_proj.")

    expected = model.state_dict()
    if migrate_task_heads:
        _map_legacy_classifier_heads(state, checkpoint, expected, model)
    # Historical packages contain unused duplicate projections and decoder-only
    # tensors. Keep only keys whose name and shape match the cleaned model.
    return {
        key: value
        for key, value in state.items()
        if key in expected and tuple(value.shape) == tuple(expected[key].shape)
    }


def _extract_model_state(
    checkpoint: Any,
    model: nn.Module,
    *,
    migrate_legacy_task_heads: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Any], bool]:
    if _is_legacy_package(checkpoint):
        return (
            _upgrade_legacy_package(
                checkpoint, model, migrate_task_heads=migrate_legacy_task_heads
            ),
            checkpoint,
            True,
        )
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"], checkpoint, False
    if isinstance(checkpoint, dict) and all(isinstance(key, str) for key in checkpoint):
        if all(isinstance(value, Tensor) for value in checkpoint.values()):
            return checkpoint, {}, False
    raise ValueError("Checkpoint does not contain a recognizable model state dictionary")


def _validate_release_metadata(metadata: dict[str, Any]) -> None:
    release = metadata.get("release_metadata")
    if not isinstance(release, dict):
        return
    triage_order = release.get("triage_class_order")
    if triage_order is not None and list(triage_order) != CANONICAL_TRIAGE_ORDER:
        raise ValueError(
            "Checkpoint triage class order is incompatible: "
            f"{triage_order!r} != {CANONICAL_TRIAGE_ORDER!r}"
        )
    cq_order = release.get("cq500_label_order")
    if cq_order is not None and list(cq_order) != list(CQ500_LABELS):
        raise ValueError("Checkpoint CQ500 label order is incompatible with this release")


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    migrate_legacy_task_heads: bool = False,
    ignore_model_prefixes: Sequence[str] = (),
    allow_shape_mismatch: bool = False,
    checkpoint_payload: Any | None = None,
) -> CheckpointLoadResult:
    checkpoint = (
        read_checkpoint_payload(path, map_location=map_location)
        if checkpoint_payload is None
        else checkpoint_payload
    )
    state, metadata, legacy = _extract_model_state(
        checkpoint,
        model,
        migrate_legacy_task_heads=migrate_legacy_task_heads,
    )
    if not legacy:
        _validate_release_metadata(metadata)

    ignored_keys: list[str] = []
    shape_mismatched_keys: list[str] = []
    if ignore_model_prefixes or allow_shape_mismatch:
        expected = model.state_dict()
        filtered: dict[str, Tensor] = {}
        for key, value in state.items():
            if any(key == prefix or key.startswith(prefix + ".") for prefix in ignore_model_prefixes):
                ignored_keys.append(key)
                continue
            if key in expected and tuple(value.shape) != tuple(expected[key].shape):
                if allow_shape_mismatch:
                    shape_mismatched_keys.append(key)
                    continue
            filtered[key] = value
        state = filtered

    # Legacy packages and explicit shared-weight initialization are necessarily
    # partial. Resume paths remain strict by default.
    effective_strict = strict and not legacy and not ignore_model_prefixes and not allow_shape_mismatch
    incompatible = model.load_state_dict(state, strict=effective_strict)

    optimizer_state = metadata.get("optimizer", metadata.get("optim")) if isinstance(metadata, dict) else None
    if optimizer is not None and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except (ValueError, KeyError):
            if not legacy:
                raise
    if scheduler is not None and isinstance(metadata, dict) and "scheduler" in metadata:
        scheduler.load_state_dict(metadata["scheduler"])
    if scaler is not None and isinstance(metadata, dict) and "scaler" in metadata:
        scaler.load_state_dict(metadata["scaler"])
    return CheckpointLoadResult(
        epoch=int(metadata.get("epoch", -1)) if isinstance(metadata, dict) else -1,
        global_step=int(metadata.get("global_step", 0)) if isinstance(metadata, dict) else 0,
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        metadata=metadata if isinstance(metadata, dict) else {},
        legacy_migrated=legacy,
        ignored_keys=ignored_keys,
        shape_mismatched_keys=shape_mismatched_keys,
    )
