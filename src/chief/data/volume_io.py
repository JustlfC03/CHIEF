from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class VolumeLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class VolumeData:
    tensor: Tensor  # [D,H,W]
    spacing_zyx: tuple[float, float, float] | None = None
    source_format: str = "unknown"


def _from_numpy(array: np.ndarray) -> Tensor:
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise VolumeLoadError(f"Expected a 3D volume, got shape {array.shape}")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def _load_dicom_directory(path: Path) -> VolumeData:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise VolumeLoadError(
            "Reading DICOM directories requires SimpleITK. Install the medical I/O dependencies from requirements.txt."
        ) from exc
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or []
    if not series_ids:
        raise VolumeLoadError(f"No DICOM series found in {path}")
    if len(series_ids) > 1:
        raise VolumeLoadError(
            f"Found {len(series_ids)} DICOM series in {path}; provide one series directory."
        )
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_ids[0])
    if not file_names:
        raise VolumeLoadError(f"DICOM series {series_ids[0]!r} in {path} contains no readable files")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()
    # SimpleITK arrays are [Z,Y,X], while spacing is returned [X,Y,Z].
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    if len(spacing_xyz) != 3 or any(not np.isfinite(v) or v <= 0 for v in spacing_xyz):
        raise VolumeLoadError(f"Invalid DICOM spacing {spacing_xyz} in {path}")
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3 or any(size <= 0 for size in array.shape):
        raise VolumeLoadError(f"Invalid DICOM volume shape {array.shape} in {path}")
    return VolumeData(
        _from_numpy(array),
        spacing_zyx=(spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]),
        source_format="dicom",
    )


def load_volume_data(path: str | Path, npz_key: str | None = None) -> VolumeData:
    """Load a CT volume and available voxel spacing metadata.

    Supported inputs are NumPy, PyTorch, NIfTI and a directory containing one
    DICOM series. NIfTI is first reoriented to nibabel's closest canonical
    orientation and then converted from `[X,Y,Z]` to `[Z,Y,X]`.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        return _load_dicom_directory(source)

    name = source.name.lower()
    if name.endswith(".npy"):
        return VolumeData(_from_numpy(np.load(source, allow_pickle=False)), source_format="npy")
    if name.endswith(".npz"):
        with np.load(source, allow_pickle=False) as archive:
            key = npz_key or ("volume" if "volume" in archive.files else archive.files[0])
            if key not in archive.files:
                raise VolumeLoadError(
                    f"Key {key!r} not found in {source}; available={archive.files}"
                )
            spacing = None
            for spacing_key in ("spacing_zyx", "spacing"):
                if spacing_key in archive.files:
                    values = np.asarray(archive[spacing_key], dtype=float).reshape(-1)
                    if values.size == 3:
                        spacing = tuple(float(v) for v in values)
                    break
            return VolumeData(_from_numpy(archive[key]), spacing, "npz")
    if name.endswith(".pt") or name.endswith(".pth"):
        value = torch.load(source, map_location="cpu", weights_only=True)
        spacing = None
        if isinstance(value, dict):
            for spacing_key in ("spacing_zyx", "spacing"):
                if spacing_key in value:
                    spacing_values = value[spacing_key]
                    if isinstance(spacing_values, Tensor):
                        spacing_values = spacing_values.tolist()
                    spacing = tuple(float(v) for v in spacing_values)
                    break
            selected = None
            for key in (npz_key, "volume", "image", "ct"):
                if key and key in value:
                    selected = value[key]
                    break
            value = selected
        if not isinstance(value, Tensor):
            raise VolumeLoadError(f"PyTorch file {source} does not contain a tensor volume")
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            raise VolumeLoadError(f"Expected a 3D tensor, got {tuple(value.shape)}")
        return VolumeData(
            value.detach().to(dtype=torch.float32, device="cpu").contiguous(),
            spacing,
            "torch",
        )
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise VolumeLoadError(
                "Reading NIfTI requires nibabel. Install the medical I/O dependencies from requirements.txt."
            ) from exc
        image = nib.as_closest_canonical(nib.load(str(source)))
        array = np.asarray(image.get_fdata(dtype=np.float32))
        if array.ndim != 3:
            raise VolumeLoadError(f"Expected a 3D NIfTI, got shape {array.shape}")
        zooms_xyz = tuple(float(v) for v in image.header.get_zooms()[:3])
        return VolumeData(
            _from_numpy(np.transpose(array, (2, 1, 0))),
            spacing_zyx=(zooms_xyz[2], zooms_xyz[1], zooms_xyz[0]),
            source_format="nifti",
        )
    raise VolumeLoadError(f"Unsupported volume format: {source}")


def load_volume(path: str | Path, npz_key: str | None = None) -> Tensor:
    """Compatibility wrapper returning only the `[D,H,W]` tensor."""
    return load_volume_data(path, npz_key=npz_key).tensor
