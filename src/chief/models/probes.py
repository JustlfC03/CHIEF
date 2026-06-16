from __future__ import annotations

import torch
from torch import Tensor, nn


class TriageProbe(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int = 3,
        dropout: float = 0.0,
        *,
        generation_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.uses_generation_features = generation_feature_dim is not None
        input_dim = latent_dim + (int(generation_feature_dim) if generation_feature_dim else 0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_classes))

    def forward(
        self,
        image_latent: Tensor,
        generation_feature: Tensor | None = None,
    ) -> Tensor:
        if self.uses_generation_features:
            if generation_feature is None:
                raise ValueError("generation_feature is required by this triage probe")
            image_latent = torch.cat((image_latent, generation_feature), dim=-1)
        return self.classifier(image_latent)


class MultiLabelProbe(nn.Module):
    def __init__(self, latent_dim: int, num_labels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_dim, num_labels))

    def forward(self, image_latent: Tensor) -> Tensor:
        return self.classifier(image_latent)
