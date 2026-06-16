from __future__ import annotations

from typing import Any

import torch


class ChiefCollator:
    def __init__(
        self,
        text_tokenizer: Any | None = None,
        decoder_tokenizer: Any | None = None,
        max_text_length: int = 256,
        max_decoder_length: int = 256,
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_text_length = max_text_length
        self.max_decoder_length = max_decoder_length

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "sample_id": [item["sample_id"] for item in items],
            "image_path": [item["image_path"] for item in items],
            "images": torch.stack([item["image"] for item in items]),
        }
        if "report" in items[0]:
            reports = [item["report"] for item in items]
            batch["reports"] = reports
            if self.text_tokenizer is not None:
                batch["text_batch"] = self.text_tokenizer(
                    reports,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt",
                )
            if self.decoder_tokenizer is not None:
                decoder_batch = self.decoder_tokenizer(
                    reports,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_decoder_length,
                    return_tensors="pt",
                )
                decoder_batch["pad_token_id"] = int(self.decoder_tokenizer.pad_token_id)
                batch["decoder_batch"] = decoder_batch
        if "label" in items[0]:
            batch["labels"] = torch.stack([item["label"] for item in items])
        if "labels" in items[0]:
            batch["labels"] = torch.stack([item["labels"] for item in items])
        return batch
