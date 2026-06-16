from __future__ import annotations

try:
    from vector_quantize_pytorch import VectorQuantize
except ImportError as exc:  # pragma: no cover - dependency error is user-facing
    raise ImportError(
        "CHIEF requires vector-quantize-pytorch==1.1.2 for checkpoint-compatible CTViT VQ. "
        "Install requirements.txt."
    ) from exc


class CompatibleVectorQuantizer(VectorQuantize):
    """Exact VQ module used by the original CHIEF CTViT encoder.

    The original implementation instantiated ``VectorQuantize`` from
    ``vector-quantize-pytorch==1.1.2`` with cosine similarity.  Keeping that
    pinned upstream implementation avoids approximating its EMA/codebook update
    behavior while preserving legacy state-dict keys such as
    ``vq._codebook.embed``.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        *,
        commitment_weight: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            codebook_size=codebook_size,
            use_cosine_sim=True,
            commitment_weight=commitment_weight,
            **kwargs,
        )
