from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def feature_decorrelation_loss(
    z: Tensor,
    offdiag_scale: float = 0.01,
) -> Tensor:
    """Barlow-style covariance decorrelation used by the CHIEF pretraining code.

    The diagonal term encourages unit feature variance and the off-diagonal
    term penalizes correlations between latent dimensions. The caller applies
    this to the image and text contrastive representations and controls the
    overall contribution through ``decorrelation_weight``.
    """
    if z.ndim != 2:
        raise ValueError(f"Expected [B,D], got {tuple(z.shape)}")
    centered = z - z.mean(dim=0, keepdim=True)
    if centered.shape[0] <= 1:
        return centered.new_zeros(())
    covariance = centered.t() @ centered / (centered.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    on_diagonal = (diagonal - 1.0).pow(2).sum()
    off_diagonal = covariance - torch.diag(diagonal)
    return on_diagonal + float(offdiag_scale) * off_diagonal.pow(2).sum()


def _distributed_concat_with_grad(tensor: Tensor) -> Tensor:
    """Gather equal-shaped batches across initialized DDP workers."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    try:
        from torch.distributed.nn.functional import all_gather
    except ImportError:
        return tensor
    gathered = all_gather(tensor)
    return torch.cat(list(gathered), dim=0)


def symmetric_info_nce(
    text_latents: Tensor,
    image_latents: Tensor,
    logit_scale: Tensor,
    *,
    gather_distributed: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    if text_latents.shape != image_latents.shape:
        raise ValueError(
            f"Paired latent shapes differ: {tuple(text_latents.shape)} vs {tuple(image_latents.shape)}"
        )
    local_batch = text_latents.shape[0]
    if gather_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        all_text = _distributed_concat_with_grad(text_latents)
        all_image = _distributed_concat_with_grad(image_latents)
        targets = torch.arange(local_batch, device=text_latents.device) + rank * local_batch
        logits_t2i = logit_scale * text_latents @ all_image.t()
        logits_i2t = logit_scale * image_latents @ all_text.t()
    else:
        targets = torch.arange(local_batch, device=text_latents.device)
        logits_t2i = logit_scale * text_latents @ image_latents.t()
        logits_i2t = logits_t2i.t()
    text_to_image = F.cross_entropy(logits_t2i, targets)
    image_to_text = F.cross_entropy(logits_i2t, targets)
    return 0.5 * (text_to_image + image_to_text), text_to_image, image_to_text


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(dtype=logits.dtype)
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        if self.clip > 0:
            negative = (negative + self.clip).clamp(max=1.0)
        loss = targets * torch.log(positive.clamp_min(self.eps))
        loss += (1.0 - targets) * torch.log(negative.clamp_min(self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            probability = positive * targets + negative * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss *= (1.0 - probability).pow(gamma)
        return -loss.mean()


def hierarchy_consistency_loss(
    logits: Tensor,
    label_names: Sequence[str],
) -> Tensor:
    """Soft parent-child constraints used by the CQ500 probe."""
    index = {name: i for i, name in enumerate(label_names)}
    probabilities = logits.sigmoid()
    penalties: list[Tensor] = []

    def parent_child(parent: str, children: Sequence[str]) -> None:
        if parent not in index:
            return
        child_indices = [index[name] for name in children if name in index]
        if not child_indices:
            return
        max_child = probabilities[:, child_indices].max(dim=1).values
        penalties.append(F.relu(max_child - probabilities[:, index[parent]]).mean())

    parent_child(
        "ICH",
        [
            "IPH",
            "IVH",
            "SDH",
            "EDH",
            "SAH",
            "ChronicBleed",
            "BleedLocation-Left",
            "BleedLocation-Right",
        ],
    )
    parent_child("Fracture", ["CalvarialFracture", "OtherFracture"])
    parent_child("MassEffect", ["MidlineShift"])
    return torch.stack(penalties).sum() if penalties else logits.new_zeros(())
