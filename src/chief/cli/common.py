from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from chief.config import load_config
from chief.data.collate import ChiefCollator
from chief.data.datasets import build_dataset
from chief.models.chief import build_model_and_tokenizers
from chief.utils import configure_logging, resolve_device, seed_everything


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted configuration key; may be repeated",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def initialize(args: argparse.Namespace) -> tuple[dict[str, Any], torch.device]:
    configure_logging(args.verbose)
    cfg = load_config(args.config, args.overrides)
    runtime = cfg.setdefault("runtime", {})
    seed_everything(int(runtime.get("seed", 42)), bool(runtime.get("deterministic", True)))
    device = resolve_device(str(runtime.get("device", "auto")))
    return cfg, device


def build_loaders(cfg: dict[str, Any], text_tokenizer: Any, decoder_tokenizer: Any | None):
    loader_cfg = cfg.get("loader", {})
    collator = ChiefCollator(
        text_tokenizer=text_tokenizer if cfg["task"] in {"pretrain", "retrieval"} else None,
        decoder_tokenizer=decoder_tokenizer
        if cfg["task"] in {"pretrain", "report_generation"}
        else None,
        max_text_length=int(loader_cfg.get("max_text_length", 256)),
        max_decoder_length=int(loader_cfg.get("max_decoder_length", 256)),
    )
    common = dict(
        batch_size=int(loader_cfg.get("batch_size", 1)),
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", torch.cuda.is_available())),
        collate_fn=collator,
        persistent_workers=bool(loader_cfg.get("persistent_workers", False))
        and int(loader_cfg.get("num_workers", 0)) > 0,
    )
    train_dataset = build_dataset(cfg, split=str(cfg.get("data", {}).get("train_split", "train")))
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=bool(loader_cfg.get("drop_last", False)), **common
    )
    val_loader = None
    val_split = cfg.get("data", {}).get("val_split")
    if val_split:
        try:
            val_dataset = build_dataset(cfg, split=str(val_split))
            val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
        except ValueError:
            if bool(cfg.get("data", {}).get("require_val_split", False)):
                raise
    return train_loader, val_loader


def load_model(cfg: dict[str, Any], device: torch.device):
    model, text_tokenizer, decoder_tokenizer = build_model_and_tokenizers(cfg)
    return model.to(device), text_tokenizer, decoder_tokenizer
