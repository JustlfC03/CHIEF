from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class TextEncoderOutput:
    hidden_states: tuple[Tensor, ...]
    last_hidden_state: Tensor


class TinyTextEncoder(nn.Module):
    """Dependency-free encoder for tests and smoke runs, not for paper results."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 64, layers: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU()) for _ in range(layers)]
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **_: Any,
    ) -> TextEncoderOutput | tuple[Tensor, ...]:
        del attention_mask
        hidden = self.embeddings(input_ids)
        states = [hidden]
        for layer in self.encoder:
            hidden = hidden + layer(hidden)
            states.append(hidden)
        output = TextEncoderOutput(tuple(states), hidden)
        if return_dict:
            return output
        return (hidden, tuple(states)) if output_hidden_states else (hidden,)


class TinyTokenizer:
    """Character tokenizer with a Hugging Face-like call interface."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    vocab_size = 512
    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"

    def __init__(self, padding_side: str = "right", vocab_size: int = 512) -> None:
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
        if vocab_size <= 4:
            raise ValueError("vocab_size must exceed the four special tokens")
        self.padding_side = padding_side
        self.vocab_size = int(vocab_size)

    def _encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [4 + (ord(char) % (self.vocab_size - 4)) for char in text]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str = "pt",
        padding: str | bool = "longest",
        truncation: bool = True,
        max_length: int = 128,
        add_special_tokens: bool = True,
        **_: Any,
    ) -> dict[str, Tensor]:
        del return_tensors
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if truncation:
            encoded = [ids[:max_length] for ids in encoded]
        length = max(len(ids) for ids in encoded) if padding else None
        if padding == "max_length":
            length = max_length
        assert length is not None
        ids_out, masks = [], []
        for ids in encoded:
            ids = ids[:length]
            mask = [1] * len(ids)
            pad = length - len(ids)
            if self.padding_side == "left":
                ids_out.append([self.pad_token_id] * pad + ids)
                masks.append([0] * pad + mask)
            else:
                ids_out.append(ids + [self.pad_token_id] * pad)
                masks.append(mask + [0] * pad)
        return {
            "input_ids": torch.tensor(ids_out, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._encode(text, add_special_tokens=add_special_tokens)

    def batch_decode(
        self, sequences: Tensor | list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        if isinstance(sequences, Tensor):
            sequences = sequences.tolist()
        outputs = []
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for ids in sequences:
            chars = []
            for token_id in ids:
                if skip_special_tokens and token_id in special:
                    continue
                if token_id < 4:
                    continue
                chars.append(chr((token_id - 4) % 95 + 32))
            outputs.append("".join(chars))
        return outputs


def load_hf_text_encoder(
    model_name: str, cache_dir: str | None = None, local_files_only: bool = False
) -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("Install transformers from requirements.txt to load Hugging Face models") from exc
    return AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def load_hf_tokenizer(
    model_name: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    *,
    padding_side: str | None = None,
    do_lower_case: bool | None = None,
    use_fast: bool = True,
):
    """Load a Hugging Face tokenizer with explicit reproducibility settings."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install transformers from requirements.txt to load Hugging Face tokenizers"
        ) from exc
    kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "use_fast": bool(use_fast),
    }
    if padding_side is not None:
        kwargs["padding_side"] = padding_side
    if do_lower_case is not None:
        kwargs["do_lower_case"] = bool(do_lower_case)
    return AutoTokenizer.from_pretrained(model_name, **kwargs)
