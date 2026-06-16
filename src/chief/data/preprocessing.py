from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _interpolate_tensor(
    volume: Tensor,
    shape: tuple[int, int, int],
    mode: str,
    *,
    align_corners: bool,
) -> Tensor:
    kwargs: dict[str, object] = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = align_corners
    return F.interpolate(volume[None, None], size=shape, mode=mode, **kwargs)[0, 0]


def legacy_piecewise_map(volume: Tensor, a: float = 0.2, b: float = 0.7) -> Tensor:
    """Apply the piecewise intensity mapping used by the original CHIEF loader.

    The prepared NPY volume is first min-max normalized per examination and then
    mapped with breakpoints 0.433 and 0.45.  The function is intentionally kept
    separate from offline conversion so it is applied exactly once at model
    loading time.
    """
    minimum, maximum = volume.amin(), volume.amax()
    denominator = maximum - minimum
    if float(denominator.abs()) < 1e-8:
        return torch.zeros_like(volume)
    normalized = (volume - minimum) / denominator
    output = torch.zeros_like(normalized)
    low = normalized < 0.433
    middle = (normalized >= 0.433) & (normalized <= 0.45)
    high = normalized > 0.45
    output[low] = normalized[low] / 0.433 * a
    output[middle] = (normalized[middle] - 0.433) / (0.45 - 0.433) * (b - a) + a
    output[high] = (normalized[high] - 0.45) / (1.0 - 0.45) * (1.0 - b) + b
    return output


@dataclass(frozen=True)
class ModelInputTransform:
    """Transform a prepared CHIEF volume into the CTViT model input.

    The research pipeline stores examination-level NPY arrays at
    ``32 x 256 x 256``.  During loading, the historical piecewise map is applied
    and the volume is resized online to ``128 x 128 x 128`` before entering the
    visual encoder.  Keeping this transform separate from offline conversion
    prevents accidental double normalization or double piecewise mapping.
    """

    expected_input_shape: tuple[int, int, int] | None = (32, 256, 256)
    output_shape: tuple[int, int, int] = (128, 128, 128)
    scale: str = "legacy_piecewise"
    interpolation: str = "trilinear"
    align_corners: bool = True
    strict_input_shape: bool = True

    def __post_init__(self) -> None:
        if self.expected_input_shape is not None and (
            len(self.expected_input_shape) != 3 or any(v <= 0 for v in self.expected_input_shape)
        ):
            raise ValueError("expected_input_shape must contain three positive integers or be null")
        if len(self.output_shape) != 3 or any(v <= 0 for v in self.output_shape):
            raise ValueError("output_shape must contain three positive integers")
        if self.scale not in {"legacy_piecewise", "zero_one", "minus_one_one", "none"}:
            raise ValueError(f"Unsupported scale={self.scale!r}")

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "ModelInputTransform":
        cfg = cfg or {}
        # These keys belonged to the erroneous v1.3 mixed offline/online profile.
        forbidden = {
            "target_spacing_zyx",
            "hu_min",
            "hu_max",
            "trim_empty_slices",
            "empty_slice_hu_threshold",
            "empty_slice_min_fraction",
        }
        present = sorted(forbidden.intersection(cfg))
        if present:
            raise ValueError(
                "data.preprocessing is the online model-input transform in v2.0 and must not "
                f"contain offline conversion keys: {present}. Run preprocess.py separately."
            )
        expected_raw = cfg.get("expected_input_shape", (32, 256, 256))
        expected = (
            tuple(int(v) for v in expected_raw) if expected_raw is not None else None
        )
        output = tuple(int(v) for v in cfg.get("output_shape", (128, 128, 128)))
        return cls(
            expected_input_shape=expected,  # type: ignore[arg-type]
            output_shape=output,  # type: ignore[arg-type]
            scale=str(cfg.get("scale", "legacy_piecewise")),
            interpolation=str(cfg.get("interpolation", "trilinear")),
            align_corners=bool(cfg.get("align_corners", True)),
            strict_input_shape=bool(cfg.get("strict_input_shape", True)),
        )

    def __call__(
        self,
        volume: Tensor,
        spacing_zyx: tuple[float, float, float] | None = None,
    ) -> Tensor:
        del spacing_zyx  # Prepared NPY volumes have already completed offline resampling.
        if volume.ndim == 4 and volume.shape[0] == 1:
            volume = volume[0]
        if volume.ndim != 3:
            raise ValueError(f"Expected [D,H,W] or [1,D,H,W], got {tuple(volume.shape)}")
        if (
            self.expected_input_shape is not None
            and tuple(volume.shape) != self.expected_input_shape
            and self.strict_input_shape
        ):
            raise ValueError(
                f"Prepared volume shape {tuple(volume.shape)} does not match "
                f"expected_input_shape={self.expected_input_shape}. Convert raw scans with "
                "preprocess.py or set strict_input_shape=false only for a deliberate experiment."
            )

        volume = torch.nan_to_num(
            volume.to(dtype=torch.float32), nan=0.0, posinf=1.0, neginf=0.0
        )
        if self.scale == "legacy_piecewise":
            volume = legacy_piecewise_map(volume)
        elif self.scale == "zero_one":
            minimum, maximum = volume.amin(), volume.amax()
            denominator = maximum - minimum
            volume = (
                (volume - minimum) / denominator
                if float(denominator.abs()) >= 1e-8
                else torch.zeros_like(volume)
            )
        elif self.scale == "minus_one_one":
            minimum, maximum = volume.amin(), volume.amax()
            denominator = maximum - minimum
            volume = (
                2.0 * (volume - minimum) / denominator - 1.0
                if float(denominator.abs()) >= 1e-8
                else torch.zeros_like(volume)
            )

        if tuple(volume.shape) != self.output_shape:
            volume = _interpolate_tensor(
                volume,
                self.output_shape,
                self.interpolation,
                align_corners=self.align_corners,
            )
        return volume.unsqueeze(0).contiguous()


@dataclass(frozen=True)
class OfflineCTPreprocessor:
    """Reference conversion from a raw volume to a prepared CHIEF NPY array.

    ``profile='legacy_z2p5'`` combines the source geometry path (z-axis
    resampling to 2.5 mm, crop/pad to 64 slices, and resize to
    ``32 x 256 x 256``) with the fixed HU clipping described for the released
    pipeline, followed by min-max normalization. ``resize_only`` mirrors the
    simpler historical conversion used for already harmonized volumes. ``auto``
    selects ``legacy_z2p5`` when spacing metadata is available and otherwise
    selects ``resize_only``.

    This helper does not perform clinical series selection, de-identification,
    artefact review, head-coverage checks or report-image pairing.
    """

    output_shape: tuple[int, int, int] = (32, 256, 256)
    profile: str = "auto"
    target_z_spacing: float = 2.5
    intermediate_depth: int = 64
    hu_min: float | None = -1000.0
    hu_max: float | None = 1000.0
    normalize: str = "minmax"
    interpolation_order: int = 1
    anti_aliasing: bool = True

    def __post_init__(self) -> None:
        if len(self.output_shape) != 3 or any(v <= 0 for v in self.output_shape):
            raise ValueError("output_shape must contain three positive integers")
        if self.profile not in {"auto", "legacy_z2p5", "resize_only"}:
            raise ValueError(f"Unsupported offline profile={self.profile!r}")
        if self.target_z_spacing <= 0:
            raise ValueError("target_z_spacing must be positive")
        if self.intermediate_depth <= 0:
            raise ValueError("intermediate_depth must be positive")
        if (self.hu_min is None) != (self.hu_max is None):
            raise ValueError("hu_min and hu_max must both be set or both be null")
        if self.hu_min is not None and self.hu_max is not None:
            if not np.isfinite(self.hu_min) or not np.isfinite(self.hu_max):
                raise ValueError("HU window bounds must be finite")
            if self.hu_max <= self.hu_min:
                raise ValueError("hu_max must be greater than hu_min")
        if self.normalize not in {"minmax", "none"}:
            raise ValueError("normalize must be 'minmax' or 'none'")

    @staticmethod
    def _resize(array: np.ndarray, shape: tuple[int, int, int], order: int, anti_aliasing: bool) -> np.ndarray:
        try:
            from skimage.transform import resize
        except ImportError as exc:
            raise ImportError(
                "Offline preprocessing requires scikit-image. Install requirements.txt."
            ) from exc
        return resize(
            array,
            shape,
            order=order,
            anti_aliasing=anti_aliasing,
            preserve_range=True,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _crop_or_pad_depth(array: np.ndarray, target_depth: int) -> np.ndarray:
        depth = array.shape[0]
        if depth > target_depth:
            # Match the active original scripts: remove trailing all-zero slices
            # when present, then retain the last target_depth slices.
            if np.all(array[-1] == 0):
                non_empty = np.where(np.any(array != 0, axis=(1, 2)))[0]
                if non_empty.size:
                    array = array[: int(non_empty[-1]) + 1]
            if array.shape[0] > target_depth:
                array = array[-target_depth:]
        if array.shape[0] < target_depth:
            pad_total = target_depth - array.shape[0]
            before = pad_total // 2
            after = pad_total - before
            array = np.pad(array, ((before, after), (0, 0), (0, 0)), mode="constant")
        return array

    def __call__(
        self,
        volume: Tensor | np.ndarray,
        spacing_zyx: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        if isinstance(volume, Tensor):
            array = volume.detach().cpu().numpy()
        else:
            array = np.asarray(volume)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"Expected [D,H,W] or [1,D,H,W], got {array.shape}")
        array = np.nan_to_num(
            array.astype(np.float32, copy=False),
            nan=0.0,
            posinf=float(self.hu_max) if self.hu_max is not None else 0.0,
            neginf=float(self.hu_min) if self.hu_min is not None else 0.0,
        )
        if self.hu_min is not None and self.hu_max is not None:
            array = np.clip(array, self.hu_min, self.hu_max)

        profile = self.profile
        if profile == "auto":
            profile = "legacy_z2p5" if spacing_zyx is not None else "resize_only"
        if profile == "legacy_z2p5":
            if spacing_zyx is None:
                raise ValueError("legacy_z2p5 requires spacing metadata; use profile='resize_only'")
            new_depth = max(1, int(round(array.shape[0] * float(spacing_zyx[0]) / self.target_z_spacing)))
            if new_depth != array.shape[0]:
                array = self._resize(
                    array,
                    (new_depth, array.shape[1], array.shape[2]),
                    self.interpolation_order,
                    self.anti_aliasing,
                )
            array = self._crop_or_pad_depth(array, self.intermediate_depth)

        array = self._resize(
            array,
            self.output_shape,
            self.interpolation_order,
            self.anti_aliasing,
        )
        if self.normalize == "minmax":
            minimum = float(np.min(array))
            maximum = float(np.max(array))
            if maximum > minimum:
                array = (array - minimum) / (maximum - minimum)
            else:
                array = np.zeros_like(array)
        return np.ascontiguousarray(array, dtype=np.float32)


# Backward-compatible import name for v1.2/v1.3 callers.  In v2.0 this class
# is explicitly the online model-input transform, not the raw-scan converter.
CTPreprocessor = ModelInputTransform
