from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PrefixProjector(nn.Module):
    def __init__(self, image_dim: int, decoder_hidden_dim: int, num_prefix_tokens: int = 1) -> None:
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.proj = nn.Linear(image_dim, decoder_hidden_dim * num_prefix_tokens)

    def forward(self, image_latent: Tensor) -> Tensor:
        batch = image_latent.shape[0]
        hidden = self.proj.out_features // self.num_prefix_tokens
        return self.proj(image_latent).reshape(batch, self.num_prefix_tokens, hidden)


@dataclass
class TinyCausalOutput:
    logits: Tensor
    hidden_states: tuple[Tensor, ...]


class TinyCausalLM(nn.Module):
    """Small decoder for smoke tests, not for scientific results."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 64) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size, hidden_size=hidden_size, n_embd=hidden_size
        )
        self.transformer = SimpleNamespace()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def resize_token_embeddings(self, vocab_size: int) -> None:
        old_weight = self.wte.weight.data
        hidden = old_weight.shape[1]
        self.wte = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        with torch.no_grad():
            count = min(vocab_size, old_weight.shape[0])
            self.wte.weight[:count].copy_(old_weight[:count])
        self.config.vocab_size = vocab_size

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **_: Any,
    ) -> TinyCausalOutput:
        del attention_mask, return_dict
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.wte(input_ids)
        hidden, _state = self.gru(inputs_embeds)
        logits = self.lm_head(hidden)
        states = (inputs_embeds, hidden) if output_hidden_states else (hidden,)
        return TinyCausalOutput(logits=logits, hidden_states=states)


def load_hf_decoder(model_name: str, cache_dir: str | None = None, local_files_only: bool = False):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install transformers from requirements.txt to load the report decoder") from exc
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def decoder_hidden_size(decoder: nn.Module) -> int:
    config = decoder.config
    for key in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    raise AttributeError("Could not infer decoder hidden size")


def conditional_lm_loss(
    decoder: nn.Module,
    prefix_embeddings: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    decoder_inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:].clone()
    token_embeddings = decoder.get_input_embeddings()(decoder_inputs)
    inputs_embeds = torch.cat((prefix_embeddings, token_embeddings), dim=1)
    prefix_mask = torch.ones(
        (input_ids.shape[0], prefix_embeddings.shape[1]),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    full_mask = torch.cat((prefix_mask, attention_mask[:, :-1]), dim=1)
    outputs = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    logits = outputs.logits[:, prefix_embeddings.shape[1] :, :]
    targets[attention_mask[:, 1:] == 0] = pad_token_id
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )
    return loss, outputs.hidden_states[-1]


@torch.no_grad()
def greedy_generate(
    decoder: nn.Module,
    prefix_embeddings: Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> Tensor:
    batch = prefix_embeddings.shape[0]
    generated = torch.full(
        (batch, 1), bos_token_id, dtype=torch.long, device=prefix_embeddings.device
    )
    finished = torch.zeros(batch, dtype=torch.bool, device=prefix_embeddings.device)
    for _ in range(max_new_tokens):
        token_embeddings = decoder.get_input_embeddings()(generated)
        inputs_embeds = torch.cat((prefix_embeddings, token_embeddings), dim=1)
        outputs = decoder(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        next_token = outputs.logits[:, -1].argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
        generated = torch.cat((generated, next_token[:, None]), dim=1)
        finished |= next_token.eq(eos_token_id)
        if bool(finished.all()):
            break
    return generated[:, 1:]



@torch.no_grad()
def top_k_sample_generate(
    decoder: nn.Module,
    prefix_embeddings: Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    *,
    temperature: float = 0.8,
    top_k: int = 50,
) -> Tensor:
    """Historical CHIEF decoding: temperature-scaled top-k sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    batch = prefix_embeddings.shape[0]
    generated = torch.full(
        (batch, 1), bos_token_id, dtype=torch.long, device=prefix_embeddings.device
    )
    finished = torch.zeros(batch, dtype=torch.bool, device=prefix_embeddings.device)
    for _ in range(max_new_tokens):
        token_embeddings = decoder.get_input_embeddings()(generated)
        inputs_embeds = torch.cat((prefix_embeddings, token_embeddings), dim=1)
        outputs = decoder(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        logits = outputs.logits[:, -1] / float(temperature)
        effective_k = min(int(top_k), logits.shape[-1])
        if effective_k > 0 and effective_k < logits.shape[-1]:
            boundary = torch.topk(logits, effective_k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < boundary, float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
        generated = torch.cat((generated, next_token[:, None]), dim=1)
        finished |= next_token.eq(eos_token_id)
        if bool(finished.all()):
            break
    return generated[:, 1:]

def image_conditioned_hidden_feature(
    decoder: nn.Module,
    prefix_embeddings: Tensor,
    bos_token_id: int,
) -> Tensor:
    """Extract an image-conditioned decoder feature from prefix plus BOS.

    The public triage interface supplies only the projected image prefix and a
    BOS token; reference-report tokens and labels are not model inputs.
    """
    batch = prefix_embeddings.shape[0]
    bos = torch.full(
        (batch, 1),
        int(bos_token_id),
        dtype=torch.long,
        device=prefix_embeddings.device,
    )
    bos_embedding = decoder.get_input_embeddings()(bos)
    inputs_embeds = torch.cat((prefix_embeddings, bos_embedding), dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    outputs = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1][:, -1]
