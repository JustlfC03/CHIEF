from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def _matches_prefix(parameter_name: str, prefixes: Sequence[str]) -> bool:
    return any(
        parameter_name == prefix or parameter_name.startswith(prefix + ".")
        for prefix in prefixes
    )


def _unfreeze_decoder_tail(model: nn.Module, last_n: int) -> int:
    """Unfreeze the final decoder blocks used by the original triage run."""
    if last_n <= 0:
        return 0
    decoder = getattr(model, "decoder", None)
    transformer = getattr(decoder, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        # The bundled tiny smoke decoder uses a GRU rather than GPT blocks.
        # Treat its recurrent core as the final decoder block.
        recurrent = getattr(decoder, "gru", None)
        if recurrent is None:
            raise ValueError("training.decoder_last_n requires a GPT-style decoder.transformer.h")
        matched = 0
        for parameter in recurrent.parameters():
            parameter.requires_grad = True
            matched += parameter.numel()
        lm_head = getattr(decoder, "lm_head", None)
        if lm_head is not None:
            for parameter in lm_head.parameters():
                parameter.requires_grad = True
                matched += parameter.numel()
        return matched
    matched = 0
    for block in list(blocks)[-last_n:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
            matched += parameter.numel()
    final_norm = getattr(transformer, "ln_f", None)
    if final_norm is not None:
        for parameter in final_norm.parameters():
            parameter.requires_grad = True
            matched += parameter.numel()
    lm_head = getattr(decoder, "lm_head", None)
    if lm_head is not None:
        for parameter in lm_head.parameters():
            parameter.requires_grad = True
            matched += parameter.numel()
    return matched


def set_trainable_modules(
    model: nn.Module,
    names: list[str] | None,
    *,
    decoder_last_n: int = 0,
) -> None:
    """Freeze all parameters except explicit modules and decoder tail blocks.

    With no restrictions, all parameters remain trainable. The decoder-tail
    option reproduces the conservative final-four-layer GPT-2 fine-tuning used
    by the original triage experiment without hard-coding layer indices.
    """
    if not names and decoder_last_n <= 0:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    matched = 0
    for parameter_name, parameter in model.named_parameters():
        if names and _matches_prefix(parameter_name, names):
            parameter.requires_grad = True
            matched += parameter.numel()
    matched += _unfreeze_decoder_tail(model, int(decoder_last_n))
    if matched == 0:
        raise ValueError(
            f"No parameters matched trainable_modules={names}, decoder_last_n={decoder_last_n}"
        )


def _parameter_groups(model: nn.Module, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build non-overlapping parameter groups from module-prefix rules."""
    group_specs = cfg.get("groups")
    if not group_specs:
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": float(cfg.get("lr", 1e-4)),
            }
        ]
    if not isinstance(group_specs, list):
        raise ValueError("optimizer.groups must be a list")

    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []
    for index, spec in enumerate(group_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"optimizer.groups[{index}] must be a mapping")
        prefixes = spec.get("modules", [])
        if not isinstance(prefixes, list) or not prefixes:
            raise ValueError(f"optimizer.groups[{index}].modules must be a non-empty list")
        parameters = [
            parameter
            for name, parameter in named
            if id(parameter) not in assigned and _matches_prefix(name, prefixes)
        ]
        if not parameters:
            raise ValueError(
                f"optimizer.groups[{index}] matched no trainable parameters: {prefixes}"
            )
        assigned.update(id(parameter) for parameter in parameters)
        group: dict[str, Any] = {
            "params": parameters,
            "lr": float(spec.get("lr", cfg.get("lr", 1e-4))),
        }
        if "weight_decay" in spec:
            group["weight_decay"] = float(spec["weight_decay"])
        groups.append(group)

    remaining = [parameter for _, parameter in named if id(parameter) not in assigned]
    if remaining:
        if bool(cfg.get("allow_unassigned", False)):
            groups.append({"params": remaining, "lr": float(cfg.get("lr", 1e-4))})
        else:
            remaining_names = [name for name, parameter in named if id(parameter) not in assigned]
            raise ValueError(
                "Trainable parameters were not assigned to optimizer.groups: "
                + ", ".join(remaining_names[:20])
            )
    return groups


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    groups = _parameter_groups(model, cfg)
    if not any(group["params"] for group in groups):
        raise ValueError("No trainable parameters")
    name = str(cfg.get("name", "adamw")).lower()
    kwargs = {
        "lr": float(cfg.get("lr", 1e-4)),
        "weight_decay": float(cfg.get("weight_decay", 1e-2)),
    }
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    if name == "adamw":
        return torch.optim.AdamW(groups, betas=betas, **kwargs)
    if name == "adam":
        return torch.optim.Adam(groups, betas=betas, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=float(cfg.get("momentum", 0.9)), **kwargs)
    raise ValueError(f"Unsupported optimizer {name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any] | None,
    total_steps: int,
):
    cfg = cfg or {"name": "none"}
    name = str(cfg.get("name", "none")).lower()
    if name == "none":
        return None
    warmup = int(cfg.get("warmup_steps", 0))
    min_factor = float(cfg.get("min_lr_factor", 0.0))

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        if name == "cosine":
            import math

            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "linear":
            return min_factor + (1.0 - min_factor) * (1.0 - progress)
        raise ValueError(f"Unsupported scheduler {name!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
