from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

DEFAULT_ABNORMALITY_TEMPLATES = (
    "是否存在{label}？请回答“是”或“否”：",
    "该头颅CT是否提示{label}？请回答是/否：",
    "影像上能否见到{label}？回答是或否：",
)


def _single_answer_token(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Next-token yes/no scoring requires {text!r} to map to exactly one token; "
            f"the configured tokenizer returned {len(ids)} tokens."
        )
    return int(ids[0])


def _next_token_logit(
    decoder,
    tokenizer,
    prefix: Tensor,
    prompts: list[str],
    yes_id: int,
    no_id: int,
) -> Tensor:
    tokenized = tokenizer(
        prompts,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(prefix.device)
    attention_mask = tokenized["attention_mask"].to(prefix.device)
    token_embeddings = decoder.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat((prefix, token_embeddings), dim=1)
    prefix_mask = torch.ones(
        (prefix.shape[0], prefix.shape[1]), device=prefix.device, dtype=attention_mask.dtype
    )
    outputs = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.cat((prefix_mask, attention_mask), dim=1),
        use_cache=False,
        return_dict=True,
    )
    # Select the final non-padding prompt token for either left- or right-padded
    # tokenizers. Using attention_mask.sum() alone is only correct for right padding.
    prompt_positions = torch.arange(
        attention_mask.shape[1], device=prefix.device
    ).unsqueeze(0)
    final_prompt_positions = (prompt_positions * attention_mask).amax(dim=1)
    positions = prefix.shape[1] + final_prompt_positions
    row = torch.arange(prefix.shape[0], device=prefix.device)
    logits = outputs.logits[row, positions]
    return logits[:, yes_id] - logits[:, no_id]


@torch.no_grad()
def score_abnormalities(
    model,
    images: Tensor,
    tokenizer: Any,
    labels: Sequence[str],
    *,
    templates: Sequence[str] = DEFAULT_ABNORMALITY_TEMPLATES,
    label_display_names: Mapping[str, str] | None = None,
    yes_text: str = "是",
    no_text: str = "否",
    temperature: float = 2.0,
    baseline_calibration: bool = True,
) -> Tensor:
    """Return calibrated probabilities shaped `[batch, labels]`.

    This implements the manuscript-level interface (template ensembling and
    baseline-prefix calibration). Threshold selection remains dataset-specific
    and must be fit on a validation cohort rather than the test set.
    """
    if model.decoder is None or model.prefix_projector is None:
        raise RuntimeError("Zero-shot scoring requires the image-conditioned decoder")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    yes_id = _single_answer_token(tokenizer, yes_text)
    no_id = _single_answer_token(tokenizer, no_text)
    image_latent = model.encode_image(images).image_base
    prefix = model.prefix_projector(image_latent)
    batch_size = images.shape[0]
    all_labels = []
    for label in labels:
        display = (label_display_names or {}).get(label, label)
        template_scores = []
        for template in templates:
            prompts = [template.format(label=display)] * batch_size
            score = _next_token_logit(model.decoder, tokenizer, prefix, prompts, yes_id, no_id)
            if baseline_calibration:
                # Calibrate against the same learned projector evaluated at a
                # zero visual latent. This preserves the projector bias and
                # matches the model used for the manuscript analysis.
                null_latent = torch.zeros_like(image_latent)
                null_prefix = model.prefix_projector(null_latent)
                baseline = _next_token_logit(
                    model.decoder, tokenizer, null_prefix, prompts, yes_id, no_id
                )
                score = score - baseline
            template_scores.append(score)
        all_labels.append(torch.stack(template_scores, dim=0).mean(dim=0))
    logits = torch.stack(all_labels, dim=1)
    return torch.sigmoid(logits / temperature)
