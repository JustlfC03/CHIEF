from __future__ import annotations

from einops import rearrange
from torch import Tensor, nn

from .attention import ContinuousPositionBias, Transformer
from .vq import CompatibleVectorQuantizer


class CTViT(nn.Module):
    """Checkpoint-compatible factorized 3D CT transformer.

    The module preserves the public parameter names used by the original CHIEF
    implementation (``enc_spatial_transformer``, ``enc_temporal_transformer``,
    ``to_patch_emb`` and ``vq``), while removing device-specific and unused GAN
    code from the historical research repository.
    """

    def __init__(
        self,
        *,
        dim: int,
        image_size: int,
        patch_size: int,
        temporal_patch_size: int,
        spatial_depth: int,
        temporal_depth: int,
        codebook_size: int = 8192,
        use_vq: bool = True,
        dim_head: int = 64,
        heads: int = 8,
        channels: int = 1,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be positive")

        self.image_size = (int(image_size), int(image_size))
        self.patch_size = (int(patch_size), int(patch_size))
        self.temporal_patch_size = int(temporal_patch_size)
        self.dim = int(dim)
        self.channels = int(channels)
        self.use_vq = bool(use_vq)

        patch_h, patch_w = self.patch_size
        first_patch_dim = channels * patch_h * patch_w
        patch_dim = channels * temporal_patch_size * patch_h * patch_w
        # The first-frame branch is retained for legacy state-dict compatibility.
        self.to_patch_emb_first_frame = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(first_patch_dim),
            nn.Linear(first_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_emb = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        self.vq = CompatibleVectorQuantizer(dim, codebook_size) if self.use_vq else nn.Identity()

        # Retained because these tensors are present in the original checkpoint,
        # although CHIEF uses only the encoder pathway.
        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, first_patch_dim),
        )
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, patch_dim),
        )

    @property
    def patch_height_width(self) -> tuple[int, int]:
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def _patchify(self, volume: Tensor) -> Tensor:
        if volume.ndim != 5:
            raise ValueError(f"Expected [B,C,D,H,W], got {tuple(volume.shape)}")
        if tuple(volume.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Expected in-plane shape {self.image_size}, got {tuple(volume.shape[-2:])}"
            )
        if volume.shape[2] % self.temporal_patch_size:
            raise ValueError(
                f"Depth {volume.shape[2]} must be divisible by "
                f"temporal_patch_size={self.temporal_patch_size}"
            )
        ph, pw = self.patch_size
        patches = rearrange(
            volume,
            "b c (t pt) (h ph) (w pw) -> b t h w (c pt ph pw)",
            pt=self.temporal_patch_size,
            ph=ph,
            pw=pw,
        )
        # Identity at index 0 preserves historical state-dict indices 1/2/3.
        return self.to_patch_emb(patches)

    def encode(self, tokens: Tensor) -> Tensor:
        batch, depth_groups, height, width, _ = tokens.shape
        spatial = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        spatial_bias = self.spatial_rel_pos_bias(height, width, device=tokens.device)
        spatial = self.enc_spatial_transformer(
            spatial,
            attn_bias=spatial_bias,
            video_shape=(batch, depth_groups, height, width),
        )
        tokens = rearrange(
            spatial,
            "(b t) (h w) d -> b t h w d",
            b=batch,
            t=depth_groups,
            h=height,
            w=width,
        )
        temporal = rearrange(tokens, "b t h w d -> (b h w) t d")
        temporal = self.enc_temporal_transformer(
            temporal,
            video_shape=(batch, depth_groups, height, width),
        )
        return rearrange(
            temporal,
            "(b h w) t d -> b t h w d",
            b=batch,
            h=height,
            w=width,
        )

    def forward(
        self,
        volume: Tensor,
        *,
        return_encoded_tokens: bool = True,
        **_: object,
    ) -> Tensor:
        tokens = self.encode(self._patchify(volume))
        if self.use_vq:
            shape = tokens.shape
            flat = rearrange(tokens, "b t h w d -> b (t h w) d")
            flat, _indices, _commitment = self.vq(flat)
            tokens = flat.reshape(shape)
        if return_encoded_tokens:
            return tokens
        return tokens
