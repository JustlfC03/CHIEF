from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ctvit import CTViT
from .encoders import TinyTextEncoder, TinyTokenizer, load_hf_text_encoder, load_hf_tokenizer
from .generation import (
    PrefixProjector,
    TinyCausalLM,
    conditional_lm_loss,
    decoder_hidden_size,
    greedy_generate,
    load_hf_decoder,
    image_conditioned_hidden_feature,
    top_k_sample_generate,
)
from .losses import feature_decorrelation_loss, symmetric_info_nce
from .probes import MultiLabelProbe, TriageProbe


@dataclass
class ChiefLatents:
    text_base: Tensor | None = None
    image_base: Tensor | None = None
    text_contrastive: Tensor | None = None
    image_contrastive: Tensor | None = None
    visual_tokens: Tensor | None = None


@dataclass
class PretrainingOutput:
    loss: Tensor
    contrastive_loss: Tensor
    text_to_image_loss: Tensor
    image_to_text_loss: Tensor
    generation_loss: Tensor
    decorrelation_loss: Tensor
    logit_scale: Tensor
    latents: ChiefLatents


class ChiefModel(nn.Module):
    """Unified CHIEF model and downstream probe interfaces.

    The visual and text encoders first produce base semantic latents. Separate
    residual projection heads produce normalized contrastive latents for the
    bidirectional image-text alignment objective. Report generation and the
    downstream probes operate on the base visual latent.
    """

    def __init__(
        self,
        *,
        visual_transformer: nn.Module,
        text_transformer: nn.Module,
        dim_image: int,
        dim_text: int,
        dim_latent: int,
        image_pooling: str = "global_mean",
        text_pooling: str = "mean",
        last_n_text_layers: int = 4,
        text_layer_weights: list[float] | None = None,
        init_temperature: float = 0.15,
        max_logit_scale: float = 10.0,
        use_decorrelation: bool = True,
        decorrelation_weight: float = 0.01,
        decorrelation_offdiag_scale: float = 0.01,
        decoder: nn.Module | None = None,
        num_prefix_tokens: int = 1,
        triage_classes: int = 3,
        cq500_labels: int = 14,
        triage_use_generation_features: bool = False,
        decoder_bos_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.visual_transformer = visual_transformer
        self.text_transformer = text_transformer
        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent
        self.image_pooling = image_pooling
        self.text_pooling_type = text_pooling
        self.last_n_text_layers = last_n_text_layers
        self.text_layer_weights = text_layer_weights or [0.4, 0.3, 0.2, 0.1]

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias=False)
        self.text_feat_ln = nn.LayerNorm(dim_latent)
        self.image_feat_ln = nn.LayerNorm(dim_latent)
        self.text_contrastive_head = nn.Sequential(
            nn.Linear(dim_latent, dim_latent),
            nn.GELU(),
            nn.Linear(dim_latent, dim_latent),
        )
        self.image_contrastive_head = nn.Sequential(
            nn.Linear(dim_latent, dim_latent),
            nn.GELU(),
            nn.Linear(dim_latent, dim_latent),
        )
        self.contrastive_ln = nn.LayerNorm(dim_latent)

        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive")
        self.temperature = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.max_logit_scale = max_logit_scale
        self.use_decorrelation = use_decorrelation
        self.decorrelation_weight = decorrelation_weight
        self.decorrelation_offdiag_scale = decorrelation_offdiag_scale

        self.decoder = decoder
        self.num_prefix_tokens = num_prefix_tokens
        self.prefix_projector = (
            PrefixProjector(
                dim_latent,
                decoder_hidden_size(decoder),
                num_prefix_tokens=num_prefix_tokens,
            )
            if decoder is not None
            else None
        )

        self.triage_use_generation_features = bool(triage_use_generation_features)
        self.decoder_bos_token_id = decoder_bos_token_id
        if self.triage_use_generation_features and decoder is None:
            raise ValueError("triage_use_generation_features requires a configured decoder")
        self.generation_feature_proj = (
            nn.Linear(decoder_hidden_size(decoder), dim_latent)
            if self.triage_use_generation_features and decoder is not None
            else None
        )
        self.triage_probe = TriageProbe(
            dim_latent,
            triage_classes,
            generation_feature_dim=dim_latent if self.triage_use_generation_features else None,
        )
        self.cq500_probe = MultiLabelProbe(dim_latent, cq500_labels)

    @property
    def logit_scale(self) -> Tensor:
        return self.temperature.exp().clamp(max=self.max_logit_scale)

    def _pool_visual_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 5:
            raise ValueError(f"Expected visual tokens [B,T,H,W,D], got {tuple(tokens.shape)}")
        if self.image_pooling == "global_mean":
            pooled = tokens.mean(dim=(1, 2, 3))
        elif self.image_pooling == "global_max":
            pooled = tokens.amax(dim=(1, 2, 3))
        elif self.image_pooling == "temporal_mean_flatten":
            # Historical CHIEF aggregation: average over depth groups and keep
            # the 4x4 spatial grid before flattening (4*4*128 = 2048).
            pooled = tokens.mean(dim=1).flatten(start_dim=1)
        else:
            raise ValueError(f"Unknown image_pooling={self.image_pooling!r}")
        if pooled.shape[-1] != self.dim_image:
            raise RuntimeError(
                "Visual pooling produced dimension "
                f"{pooled.shape[-1]}, but model.dim_image={self.dim_image}."
            )
        return pooled

    def encode_image(self, images: Tensor, normalize_contrastive: bool = True) -> ChiefLatents:
        tokens = self.visual_transformer(images)
        image_features = self._pool_visual_tokens(tokens)
        image_base = self.image_feat_ln(self.to_visual_latent(image_features))
        projected = self.image_contrastive_head(image_base) + image_base
        image_contrastive = self.contrastive_ln(projected)
        if normalize_contrastive:
            image_contrastive = F.normalize(image_contrastive, dim=-1)
        return ChiefLatents(
            image_base=image_base,
            image_contrastive=image_contrastive,
            visual_tokens=tokens,
        )

    def _fuse_hidden_states(self, hidden_states: tuple[Tensor, ...]) -> Tensor:
        count = min(self.last_n_text_layers, len(hidden_states))
        selected = hidden_states[-count:][::-1]
        weights = torch.tensor(
            self.text_layer_weights[:count],
            device=selected[0].device,
            dtype=selected[0].dtype,
        )
        if weights.numel() != count or not torch.isfinite(weights).all():
            weights = torch.zeros(count, device=selected[0].device, dtype=selected[0].dtype)
        # The original CHIEF path treated [0.4, 0.3, 0.2, 0.1] as logits.
        weights = torch.softmax(weights, dim=0)
        stacked = torch.stack(selected, dim=0)
        return (stacked * weights[:, None, None, None]).sum(dim=0)

    def _pool_text(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        if self.text_pooling_type == "cls":
            return token_embeddings[:, 0]
        mask = attention_mask.bool().unsqueeze(-1)
        mean = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if self.text_pooling_type == "cls_mean":
            return 0.7 * token_embeddings[:, 0] + 0.3 * mean
        if self.text_pooling_type == "mean":
            return mean
        raise ValueError(f"Unknown text_pooling={self.text_pooling_type!r}")

    def encode_text(
        self,
        text_batch: dict[str, Tensor],
        normalize_contrastive: bool = True,
    ) -> ChiefLatents:
        outputs = self.text_transformer(
            input_ids=text_batch["input_ids"],
            attention_mask=text_batch["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        fused = self._fuse_hidden_states(tuple(outputs.hidden_states))
        text_features = self._pool_text(fused, text_batch["attention_mask"])
        text_base = self.text_feat_ln(self.to_text_latent(text_features))
        projected = self.text_contrastive_head(text_base) + text_base
        text_contrastive = self.contrastive_ln(projected)
        if normalize_contrastive:
            text_contrastive = F.normalize(text_contrastive, dim=-1)
        return ChiefLatents(text_base=text_base, text_contrastive=text_contrastive)

    def encode_pair(self, images: Tensor, text_batch: dict[str, Tensor]) -> ChiefLatents:
        image = self.encode_image(images)
        text = self.encode_text(text_batch)
        return ChiefLatents(
            text_base=text.text_base,
            image_base=image.image_base,
            text_contrastive=text.text_contrastive,
            image_contrastive=image.image_contrastive,
            visual_tokens=image.visual_tokens,
        )

    def forward_pretrain(
        self,
        images: Tensor,
        text_batch: dict[str, Tensor],
        decoder_batch: dict[str, Tensor] | None = None,
        generation_weight: float = 1.0,
    ) -> PretrainingOutput:
        latents = self.encode_pair(images, text_batch)
        assert latents.text_contrastive is not None and latents.image_contrastive is not None
        contrastive, text_to_image, image_to_text = symmetric_info_nce(
            latents.text_contrastive,
            latents.image_contrastive,
            self.logit_scale,
        )

        decorrelation = contrastive.new_zeros(())
        if self.use_decorrelation:
            # Match the original CHIEF pretraining path: geometry regularization
            # is applied to the normalized image/text contrastive representations.
            decorrelation = feature_decorrelation_loss(
                latents.text_contrastive,
                self.decorrelation_offdiag_scale,
            ) + feature_decorrelation_loss(
                latents.image_contrastive,
                self.decorrelation_offdiag_scale,
            )

        generation = contrastive.new_zeros(())
        if decoder_batch is not None:
            if self.decoder is None or self.prefix_projector is None:
                raise RuntimeError("decoder_batch was supplied but no decoder is configured")
            assert latents.image_base is not None
            prefix = self.prefix_projector(latents.image_base)
            generation, _ = conditional_lm_loss(
                self.decoder,
                prefix,
                decoder_batch["input_ids"],
                decoder_batch["attention_mask"],
                int(decoder_batch["pad_token_id"]),
            )

        total = contrastive + self.decorrelation_weight * decorrelation
        total = total + generation_weight * generation
        return PretrainingOutput(
            loss=total,
            contrastive_loss=contrastive,
            text_to_image_loss=text_to_image,
            image_to_text_loss=image_to_text,
            generation_loss=generation,
            decorrelation_loss=decorrelation,
            logit_scale=self.logit_scale.detach(),
            latents=latents,
        )

    def triage_logits(self, images: Tensor) -> Tensor:
        image = self.encode_image(images)
        assert image.image_base is not None
        generation_feature = None
        if self.triage_use_generation_features:
            if (
                self.decoder is None
                or self.prefix_projector is None
                or self.generation_feature_proj is None
                or self.decoder_bos_token_id is None
            ):
                raise RuntimeError("Generation-enhanced triage is incompletely configured")
            prefix = self.prefix_projector(image.image_base)
            hidden = image_conditioned_hidden_feature(
                self.decoder, prefix, self.decoder_bos_token_id
            )
            generation_feature = self.generation_feature_proj(hidden)
        return self.triage_probe(image.image_base, generation_feature)

    def cq500_logits(self, images: Tensor) -> Tensor:
        image = self.encode_image(images)
        assert image.image_base is not None
        return self.cq500_probe(image.image_base)

    @torch.no_grad()
    def generate_reports(
        self,
        images: Tensor,
        *,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int = 256,
        strategy: str = "top_k_sampling",
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Tensor:
        if self.decoder is None or self.prefix_projector is None:
            raise RuntimeError("No decoder is configured")
        image = self.encode_image(images)
        assert image.image_base is not None
        prefix = self.prefix_projector(image.image_base)
        if strategy == "greedy":
            return greedy_generate(
                self.decoder,
                prefix,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
            )
        if strategy == "top_k_sampling":
            return top_k_sample_generate(
                self.decoder,
                prefix,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        raise ValueError(f"Unsupported report generation strategy={strategy!r}")


def _hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for key in ("hidden_size", "d_model", "dim"):
        value = getattr(config, key, None) if config is not None else None
        if value is not None:
            return int(value)
    raise AttributeError("Could not infer text encoder hidden size")


def build_model_and_tokenizers(cfg: dict[str, Any]) -> tuple[ChiefModel, Any, Any | None]:
    model_cfg = cfg["model"]
    debug_tiny = bool(model_cfg.get("debug_tiny", False))
    visual = CTViT(**dict(model_cfg["visual"]))

    if debug_tiny:
        tiny_hidden = int(model_cfg.get("tiny_hidden_size", 64))
        text_encoder = TinyTextEncoder(
            vocab_size=int(model_cfg.get("tiny_vocab_size", 512)),
            hidden_size=tiny_hidden,
            layers=max(4, int(model_cfg.get("last_n_text_layers", 4))),
        )
        text_tokenizer = TinyTokenizer(
            padding_side=str(model_cfg.get("text_padding_side", "left")),
            vocab_size=int(model_cfg.get("tiny_vocab_size", 512)),
        )
        decoder = (
            TinyCausalLM(vocab_size=text_tokenizer.vocab_size, hidden_size=tiny_hidden)
            if model_cfg.get("use_decoder", True)
            else None
        )
        decoder_tokenizer = (
            TinyTokenizer(
                padding_side=str(model_cfg.get("decoder_padding_side", "right")),
                vocab_size=int(model_cfg.get("tiny_vocab_size", 512)),
            )
            if decoder is not None
            else None
        )
    else:
        cache_dir = cfg.get("runtime", {}).get("hf_cache_dir")
        local_only = bool(cfg.get("runtime", {}).get("local_files_only", False))
        text_name = model_cfg["text_encoder_name"]
        text_encoder = load_hf_text_encoder(text_name, cache_dir, local_only)
        text_tokenizer = load_hf_tokenizer(
            text_name,
            cache_dir,
            local_only,
            padding_side=str(model_cfg.get("text_padding_side", "left")),
            do_lower_case=bool(model_cfg.get("text_do_lower_case", False)),
            use_fast=bool(model_cfg.get("text_tokenizer_use_fast", False)),
        )
        # Match the original BERT tokenizer convention without adding tokens.
        if text_tokenizer.bos_token_id is None and text_tokenizer.cls_token_id is not None:
            text_tokenizer.bos_token = text_tokenizer.cls_token
        if text_tokenizer.eos_token_id is None and text_tokenizer.sep_token_id is not None:
            text_tokenizer.eos_token = text_tokenizer.sep_token

        decoder = None
        decoder_tokenizer = None
        if model_cfg.get("use_decoder", True):
            decoder_name = model_cfg["decoder_name"]
            decoder_tokenizer = load_hf_tokenizer(
                decoder_name,
                cache_dir,
                local_only,
                padding_side=str(model_cfg.get("decoder_padding_side", "right")),
                use_fast=bool(model_cfg.get("decoder_tokenizer_use_fast", True)),
            )
            if decoder_tokenizer.pad_token_id is None:
                decoder_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if decoder_tokenizer.eos_token_id is None:
                raise ValueError("Decoder tokenizer must define eos_token_id")
            if bool(model_cfg.get("decoder_bos_from_eos", True)):
                decoder_tokenizer.bos_token = decoder_tokenizer.eos_token
            elif decoder_tokenizer.bos_token_id is None:
                decoder_tokenizer.bos_token = decoder_tokenizer.eos_token
            decoder = load_hf_decoder(decoder_name, cache_dir, local_only)
            if len(decoder_tokenizer) != decoder.config.vocab_size:
                decoder.resize_token_embeddings(len(decoder_tokenizer))

    dim_text = int(model_cfg.get("dim_text", _hidden_size(text_encoder)))
    decoder_bos_token_id = (
        int(decoder_tokenizer.bos_token_id)
        if decoder_tokenizer is not None and decoder_tokenizer.bos_token_id is not None
        else None
    )
    model = ChiefModel(
        visual_transformer=visual,
        text_transformer=text_encoder,
        dim_image=int(model_cfg["dim_image"]),
        dim_text=dim_text,
        dim_latent=int(model_cfg["dim_latent"]),
        image_pooling=str(model_cfg.get("image_pooling", "global_mean")),
        text_pooling=str(model_cfg.get("text_pooling", "mean")),
        last_n_text_layers=int(model_cfg.get("last_n_text_layers", 4)),
        text_layer_weights=list(model_cfg.get("text_layer_weights", [0.4, 0.3, 0.2, 0.1])),
        init_temperature=float(model_cfg.get("init_temperature", 0.15)),
        max_logit_scale=float(model_cfg.get("max_logit_scale", 10.0)),
        use_decorrelation=bool(model_cfg.get("use_decorrelation", True)),
        decorrelation_weight=float(model_cfg.get("decorrelation_weight", 0.01)),
        decorrelation_offdiag_scale=float(model_cfg.get("decorrelation_offdiag_scale", 0.01)),
        decoder=decoder,
        num_prefix_tokens=int(model_cfg.get("num_prefix_tokens", 1)),
        triage_classes=int(model_cfg.get("triage_classes", 3)),
        cq500_labels=int(model_cfg.get("cq500_labels", 14)),
        triage_use_generation_features=bool(
            model_cfg.get("triage_use_generation_features", False)
        ),
        decoder_bos_token_id=decoder_bos_token_id,
    )
    return model, text_tokenizer, decoder_tokenizer
