from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def exists(value: object) -> bool:
    return value is not None


class LegacyLayerNorm(nn.Module):
    """LayerNorm with CT-ViT-compatible parameter names (`gamma`, `beta`)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def feed_forward(dim: int, mult: float = 4.0, dropout: float = 0.0) -> nn.Sequential:
    inner_dim = int(mult * (2.0 / 3.0) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_context: int | None = None,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        num_null_kv: int = 0,
        norm_context: bool = True,
        dropout: float = 0.0,
        scale: float = 8.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = dim if dim_context is None else dim_context

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = LegacyLayerNorm(dim)
        self.context_norm = LegacyLayerNorm(dim_context) if norm_context else nn.Identity()
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        context: Tensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        batch = x.shape[0]
        if context is not None:
            context = self.context_norm(context)
        kv_input = x if context is None else context
        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v)]

        if self.num_null_kv:
            null_kv = self.null_kv.view(self.heads, self.num_null_kv, 2, -1)
            nk = null_kv[:, :, 0].unsqueeze(0).expand(batch, -1, -1, -1)
            nv = null_kv[:, :, 1].unsqueeze(0).expand(batch, -1, -1, -1)
            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

        q = F.normalize(q, dim=-1) * self.q_scale
        k = F.normalize(k, dim=-1) * self.k_scale
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if attn_bias is not None:
            if self.num_null_kv:
                attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
            sim = sim + attn_bias.to(device=sim.device, dtype=sim.dtype)

        if mask is not None:
            if self.num_null_kv:
                mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask.bool(), "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=sim.device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = self.attn_dropout(sim.softmax(dim=-1))
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class ContinuousPositionBias(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        num_dims: int = 2,
        layers: int = 2,
        log_dist: bool = True,
        cache_rel_pos: bool = False,
    ) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist
        self.cache_rel_pos = cache_rel_pos
        self.net = nn.ModuleList([nn.Sequential(nn.Linear(num_dims, dim), nn.LeakyReLU(0.1))])
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.1)))
        self.net.append(nn.Linear(dim, heads))
        self.register_buffer("rel_pos", None, persistent=False)

    def forward(self, *dimensions: int, device: torch.device | None = None) -> Tensor:
        device = device or next(self.parameters()).device
        needs_update = self.rel_pos is None or not self.cache_rel_pos
        if not needs_update and self.rel_pos is not None:
            expected = math.prod(dimensions)
            needs_update = self.rel_pos.shape[:2] != (expected, expected)
        if needs_update:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.rel_pos = rel_pos
        rel_pos = self.rel_pos.to(device=device, dtype=torch.float32)
        for layer in self.net:
            rel_pos = layer(rel_pos)
        return rearrange(rel_pos, "i j h -> h i j")


class Transformer(nn.Module):
    """CT-ViT transformer with state-dict-compatible module nesting."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int,
        dim_context: int | None = None,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: float = 4.0,
        attn_num_null_kv: int = 2,
        has_cross_attn: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _layer in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Identity(),
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            causal=causal,
                            dropout=attn_dropout,
                        ),
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            dim_context=dim_context,
                            heads=heads,
                            num_null_kv=attn_num_null_kv,
                            dropout=attn_dropout,
                        )
                        if has_cross_attn
                        else nn.Identity(),
                        feed_forward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )
        self.norm_out = LegacyLayerNorm(dim)
        self.has_cross_attn = has_cross_attn

    def forward(
        self,
        x: Tensor,
        video_shape: tuple[int, int, int, int] | None = None,
        attn_bias: Tensor | None = None,
        context: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_context_mask: Tensor | None = None,
    ) -> Tensor:
        del video_shape
        for _peg, self_attn, cross_attn, ff in self.layers:
            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x
            if self.has_cross_attn and context is not None:
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x
            x = ff(x) + x
        return self.norm_out(x)
