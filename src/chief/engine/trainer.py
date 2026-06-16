from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from chief.evaluation.classification import classification_metrics
from chief.evaluation.multilabel import multilabel_metrics
from chief.tasks.steps import compute_task_loss, move_batch_to_device
from chief.utils.logging import get_logger

from .checkpoint import load_checkpoint, save_checkpoint

LOGGER = get_logger(__name__)


class Trainer:
    """Config-driven single-process trainer with AMP, accumulation and resume.

    Launch one process per experiment. Multi-GPU scaling can be added through
    an external launcher after verifying the private dataset sampler policy;
    the repository deliberately avoids silently duplicating patient samples.
    """

    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
        config: dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 1))
        self.accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.log_every = int(train_cfg.get("log_every", 10))
        self.save_every_epochs = int(train_cfg.get("save_every_epochs", 1))
        self.max_steps = int(train_cfg.get("max_steps", -1))
        self.max_validation_batches = int(train_cfg.get("max_validation_batches", -1))
        amp_requested = bool(train_cfg.get("amp", True))
        self.use_amp = amp_requested and device.type == "cuda"
        self.amp_dtype = torch.float16
        if str(train_cfg.get("amp_dtype", "float16")) == "bfloat16":
            self.amp_dtype = torch.bfloat16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        self.start_epoch = 0
        self.global_step = 0
        self.selection_metric = str(train_cfg.get("selection_metric", "loss"))
        self.selection_mode = str(train_cfg.get("selection_mode", "min")).lower()
        if self.selection_mode not in {"min", "max"}:
            raise ValueError("training.selection_mode must be 'min' or 'max'")
        self.best_val = math.inf if self.selection_mode == "min" else -math.inf

    def resume(self, checkpoint: str | Path, strict: bool = True) -> None:
        result = load_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            strict=strict,
            map_location=self.device,
        )
        self.start_epoch = result.epoch + 1
        self.global_step = result.global_step
        saved_metrics = result.metadata.get("metrics", {})
        saved_selection = saved_metrics.get("selection_metric")
        saved_mode = saved_metrics.get("selection_mode")
        if saved_selection is not None and str(saved_selection) != self.selection_metric:
            raise ValueError(
                f"Resume checkpoint selection_metric={saved_selection!r} does not match "
                f"configuration {self.selection_metric!r}"
            )
        if saved_mode is not None and str(saved_mode) != self.selection_mode:
            raise ValueError(
                f"Resume checkpoint selection_mode={saved_mode!r} does not match "
                f"configuration {self.selection_mode!r}"
            )
        if saved_selection is not None:
            saved_best = result.metadata.get("best_val", saved_metrics.get("selection_value"))
            if saved_best is not None:
                self.best_val = float(saved_best)
        elif self.selection_metric == "loss" and self.selection_mode == "min":
            # Backward compatibility with v1.3 loss-selected runs.
            saved_best = result.metadata.get("best_val", saved_metrics.get("val_loss"))
            if saved_best is not None:
                self.best_val = float(saved_best)
        else:
            LOGGER.warning(
                "Checkpoint predates explicit selection metadata; resetting best %s (%s) "
                "while preserving optimizer and epoch state",
                self.selection_metric,
                self.selection_mode,
            )
        LOGGER.info(
            "Resumed from epoch=%d step=%d best_%s=%s mode=%s legacy_migrated=%s",
            result.epoch,
            result.global_step,
            self.selection_metric,
            self.best_val,
            self.selection_mode,
            result.legacy_migrated,
        )

    @staticmethod
    def _average(metrics: list[dict[str, float]]) -> dict[str, float]:
        """Average batch metrics with examination-count weighting.

        DataLoader batches are not guaranteed to have equal size.  Weighting by
        the number of examinations keeps epoch-level accuracy and loss from
        over-weighting the final short batch.  ``__weight__`` is an internal
        field and is never exposed in logs or checkpoint metrics.
        """
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, float] = defaultdict(float)
        for row in metrics:
            weight = float(row.get("__weight__", 1.0))
            if not math.isfinite(weight) or weight <= 0:
                continue
            for key, value in row.items():
                if key == "__weight__":
                    continue
                if math.isfinite(value):
                    sums[key] += value * weight
                    counts[key] += weight
        return {key: sums[key] / counts[key] for key in sums if counts[key]}

    def _run_epoch(self, epoch: int, training: bool) -> dict[str, float]:
        loader = self.train_loader if training else self.val_loader
        if loader is None:
            return {}
        self.model.train(training)
        rows: list[dict[str, float]] = []
        validation_logits: list[torch.Tensor] = []
        validation_targets: list[torch.Tensor] = []
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(loader, desc=f"{'train' if training else 'val'} {epoch}", leave=False)
        for batch_index, batch in enumerate(iterator):
            batch = move_batch_to_device(batch, self.device)
            context = torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            )
            with torch.set_grad_enabled(training), context:
                loss, metrics = compute_task_loss(self.model, batch, self.config)
                scaled_loss = loss / self.accumulation_steps
            logits = metrics.pop("__logits__", None)
            targets = metrics.pop("__targets__", None)
            if not training and logits is not None and targets is not None:
                validation_logits.append(logits.float().cpu())
                validation_targets.append(targets.cpu())
            if training:
                self.scaler.scale(scaled_loss).backward()
                should_step = (
                    batch_index + 1
                ) % self.accumulation_steps == 0 or batch_index + 1 == len(loader)
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1
                metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
            rows.append({**metrics, "__weight__": float(batch["images"].shape[0])})
            iterator.set_postfix(loss=f"{metrics['loss']:.4f}")
            if training and self.global_step and self.global_step % self.log_every == 0:
                LOGGER.info("epoch=%d step=%d metrics=%s", epoch, self.global_step, metrics)
            if training and self.max_steps > 0 and self.global_step >= self.max_steps:
                break
            if (not training) and self.max_validation_batches > 0 and batch_index + 1 >= self.max_validation_batches:
                break
        epoch_metrics = self._average(rows)
        if not training and validation_logits:
            logits = torch.cat(validation_logits).numpy()
            targets = torch.cat(validation_targets).numpy()
            task = str(self.config.get("task", "")).lower()
            if task == "triage":
                probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
                predictions = probabilities.argmax(axis=-1)
                endpoint = classification_metrics(
                    targets, predictions, probabilities, labels=[0, 1, 2]
                )
                for key, value in endpoint.items():
                    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
                        epoch_metrics[key] = float(value)
            elif task == "cq500":
                probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
                endpoint = multilabel_metrics(targets, probabilities, label_names=None)
                for key, value in endpoint.items():
                    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
                        epoch_metrics[key] = float(value)
        return epoch_metrics

    def fit(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        started = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._run_epoch(epoch, training=True)
            val_metrics = self._run_epoch(epoch, training=False)
            row = {
                "epoch": epoch,
                "global_step": self.global_step,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            LOGGER.info("epoch=%d train=%s val=%s", epoch, train_metrics, val_metrics)
            source_metrics = val_metrics if val_metrics else train_metrics
            if self.selection_metric not in source_metrics:
                raise KeyError(
                    f"Selection metric {self.selection_metric!r} is absent from epoch metrics: "
                    f"{sorted(source_metrics)}"
                )
            selection_value = float(source_metrics[self.selection_metric])
            improved = (
                selection_value < self.best_val
                if self.selection_mode == "min"
                else selection_value > self.best_val
            )
            checkpoint_metrics = {
                "selection_metric": self.selection_metric,
                "selection_mode": self.selection_mode,
                "selection_value": selection_value,
                **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            }
            if improved:
                self.best_val = selection_value
                save_checkpoint(
                    self.output_dir / "best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    config=self.config,
                    epoch=epoch,
                    global_step=self.global_step,
                    metrics=checkpoint_metrics,
                    best_val=self.best_val,
                )
            if (epoch + 1) % self.save_every_epochs == 0:
                save_checkpoint(
                    self.output_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    config=self.config,
                    epoch=epoch,
                    global_step=self.global_step,
                    metrics=checkpoint_metrics,
                    best_val=self.best_val,
                )
            with (self.output_dir / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, ensure_ascii=False, indent=2)
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break
        summary = {
            "epochs_completed": len(history),
            "global_step": self.global_step,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "best_selection_value": self.best_val,
            "elapsed_seconds": time.time() - started,
        }
        with (self.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
