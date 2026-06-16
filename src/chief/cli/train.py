from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from chief.constants import CQ500_LABELS
from chief.engine.checkpoint import load_checkpoint
from chief.engine.optim import build_optimizer, build_scheduler, set_trainable_modules
from chief.engine.trainer import Trainer
from chief.utils import get_logger

from .common import base_parser, build_loaders, initialize, load_model

LOGGER = get_logger(__name__)


def _initialization_ignored_prefixes(task: str) -> tuple[str, ...]:
    """Task-local modules that must not be imported from a pretraining run."""
    prefixes = ["triage_probe", "cq500_probe", "generation_feature_proj"]
    if task == "cq500":
        # CQ500 uses the visual representation only.
        prefixes.extend(["decoder", "prefix_projector"])
    return tuple(prefixes)


def _initialize_cq500_bias_from_prior(model, dataset, *, eps: float = 1e-4) -> None:
    """Initialize the CQ500 output bias from training-set label prevalence."""
    frame = getattr(dataset, "frame", None)
    if frame is None:
        raise ValueError("CQ500 prior initialization requires a dataset with a manifest frame")
    missing = [label for label in CQ500_LABELS if label not in frame]
    if missing:
        raise ValueError(f"CQ500 training manifest is missing labels: {missing}")
    priors = frame[CQ500_LABELS].astype(float).mean(axis=0).to_numpy(dtype=np.float64)
    priors = np.clip(priors, eps, 1.0 - eps)
    bias = torch.as_tensor(
        np.log(priors / (1.0 - priors)),
        dtype=model.cq500_probe.classifier[-1].bias.dtype,
        device=model.cq500_probe.classifier[-1].bias.device,
    )
    with torch.no_grad():
        model.cq500_probe.classifier[-1].bias.copy_(bias)
    LOGGER.info("Initialized CQ500 classifier bias from %d training examinations", len(frame))


def main() -> None:
    parser = base_parser("Train CHIEF pretraining or a downstream probe")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Resume the same experiment, including optimizer/epoch")
    group.add_argument("--init-checkpoint", help="Initialize model weights for a new experiment")
    parser.add_argument(
        "--non-strict-load",
        action="store_true",
        help="Allow model key mismatch after reviewing the log",
    )
    args = parser.parse_args()
    cfg, device = initialize(args)
    task = str(cfg.get("task", "")).lower()
    if task not in {"pretrain", "report_generation", "triage", "cq500"}:
        parser.error("task must be pretrain, report_generation, triage, or cq500")
    model, text_tokenizer, decoder_tokenizer = load_model(cfg, device)
    train_cfg = cfg.get("training", {})
    set_trainable_modules(
        model,
        train_cfg.get("trainable_modules"),
        decoder_last_n=int(train_cfg.get("decoder_last_n", 0)),
    )
    train_loader, val_loader = build_loaders(cfg, text_tokenizer, decoder_tokenizer)
    optimizer = build_optimizer(model, cfg.get("optimizer", {}))
    updates_per_epoch = max(
        1,
        (len(train_loader) + int(cfg.get("training", {}).get("gradient_accumulation_steps", 1)) - 1)
        // int(cfg.get("training", {}).get("gradient_accumulation_steps", 1)),
    )
    total_steps = updates_per_epoch * int(cfg.get("training", {}).get("epochs", 1))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps)
    output_dir = Path(cfg.get("runtime", {}).get("output_dir", "outputs/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, ensure_ascii=False, indent=2)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=cfg,
        output_dir=output_dir,
    )
    if args.init_checkpoint:
        ignored_prefixes = _initialization_ignored_prefixes(task)
        result = load_checkpoint(
            args.init_checkpoint,
            model=trainer.model,
            strict=False,
            map_location=device,
            ignore_model_prefixes=ignored_prefixes,
            allow_shape_mismatch=True,
        )
        unexpected = list(result.unexpected_keys)
        disallowed_missing = [
            key
            for key in result.missing_keys
            if not any(key == prefix or key.startswith(prefix + ".") for prefix in ignored_prefixes)
        ]
        if (unexpected or disallowed_missing or result.shape_mismatched_keys) and not args.non_strict_load:
            raise RuntimeError(
                "Shared-weight initialization found an unexpected mismatch. "
                f"missing={disallowed_missing}, unexpected={unexpected}, "
                f"shape_mismatch={result.shape_mismatched_keys}. "
                "Review the model configuration or pass --non-strict-load deliberately."
            )
        if result.missing_keys or result.unexpected_keys or result.shape_mismatched_keys:
            LOGGER.warning(
                "Initialization summary: task-local skipped=%d missing=%s unexpected=%s shape_mismatch=%s",
                len(result.ignored_keys or []),
                disallowed_missing,
                unexpected,
                result.shape_mismatched_keys,
            )
    if task == "cq500" and not args.resume and bool(train_cfg.get("init_bias_from_prior", True)):
        _initialize_cq500_bias_from_prior(trainer.model, train_loader.dataset)
    if args.resume:
        trainer.resume(args.resume, strict=not args.non_strict_load)
    summary = trainer.fit()
    LOGGER.info("Training complete: %s", summary)


if __name__ == "__main__":
    main()
