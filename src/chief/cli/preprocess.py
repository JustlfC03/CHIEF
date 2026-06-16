from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chief.data.preprocessing import OfflineCTPreprocessor
from chief.data.volume_io import load_volume_data
from chief.utils import configure_logging


def _looks_prepared(array: np.ndarray, shape: tuple[int, int, int]) -> bool:
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if tuple(array.shape) != shape:
        return False
    if not np.isfinite(array).all():
        return False
    return float(array.min()) >= -1e-6 and float(array.max()) <= 1.0 + 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one raw CT volume or one-series DICOM directory to the prepared "
            "CHIEF NPY representation. The output is 32x256x256 by default; the "
            "online loader later maps it to 128x128x128 for CTViT."
        )
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--profile",
        choices=["auto", "legacy_z2p5", "resize_only"],
        default="auto",
        help=(
            "auto uses legacy_z2p5 when spacing metadata is available and resize_only "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--shape", type=int, nargs=3, default=(32, 256, 256), metavar=("D", "H", "W")
    )
    parser.add_argument("--target-z-spacing", type=float, default=2.5)
    parser.add_argument("--intermediate-depth", type=int, default=64)
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=1000.0)
    parser.add_argument(
        "--no-hu-clip",
        action="store_true",
        help="Disable the default fixed [-1000, 1000] HU window.",
    )
    parser.add_argument(
        "--normalization", choices=["minmax", "none"], default="minmax"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow reprocessing an input that already looks like a prepared CHIEF NPY.",
    )
    parser.add_argument(
        "--metadata-json",
        help="Optional path for a small conversion metadata sidecar.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    configure_logging(args.verbose)

    loaded = load_volume_data(args.input)
    output_shape = tuple(int(v) for v in args.shape)
    source_array = loaded.tensor.detach().cpu().numpy()
    if not args.force and loaded.source_format == "npy" and _looks_prepared(source_array, output_shape):
        raise SystemExit(
            "Input already looks like a prepared CHIEF NPY. Refusing to preprocess it again; "
            "use the file directly in the manifest or pass --force deliberately."
        )

    processor = OfflineCTPreprocessor(
        output_shape=output_shape,
        profile=args.profile,
        target_z_spacing=float(args.target_z_spacing),
        intermediate_depth=int(args.intermediate_depth),
        hu_min=None if args.no_hu_clip else float(args.hu_min),
        hu_max=None if args.no_hu_clip else float(args.hu_max),
        normalize=args.normalization,
    )
    processed = processor(loaded.tensor, loaded.spacing_zyx)
    output = Path(args.output)
    if output.suffix.lower() != ".npy":
        raise SystemExit("Output path must end with .npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, processed.astype(np.float32, copy=False))

    effective_profile = args.profile
    if effective_profile == "auto":
        effective_profile = "legacy_z2p5" if loaded.spacing_zyx is not None else "resize_only"
    metadata = {
        "input": str(Path(args.input).expanduser().resolve()),
        "output": str(output.resolve()),
        "source_format": loaded.source_format,
        "source_shape": list(loaded.tensor.shape),
        "source_spacing_zyx": list(loaded.spacing_zyx) if loaded.spacing_zyx else None,
        "profile": effective_profile,
        "prepared_shape": list(processed.shape),
        "hu_window": None if args.no_hu_clip else [float(args.hu_min), float(args.hu_max)],
        "normalization": args.normalization,
        "dtype": "float32",
    }
    if args.metadata_json:
        sidecar = Path(args.metadata_json)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"saved {output.resolve()} shape={processed.shape} dtype=float32 "
        f"range=[{processed.min():.6f}, {processed.max():.6f}] profile={effective_profile}"
    )


if __name__ == "__main__":
    main()
