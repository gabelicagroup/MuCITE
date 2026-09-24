"""Bake regular Cartesian 3D electrode masks from CSV point exports."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ...env.boundaries.mask3d import finalize_electrode_mask3d
from ..boundary_masks import save_electrode_mask3d_npz


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", header.strip().lower())


def _read_csv_columns(csv_path: Path) -> dict[str, np.ndarray]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        raw_fieldnames = next(reader, None)
        if raw_fieldnames is None:
            raise ValueError(f"CSV file is missing a header: {csv_path}")
        normalized_names: list[str] = []
        seen: dict[str, int] = {}
        for raw_name in raw_fieldnames:
            base_name = _normalize_header(raw_name)
            count = seen.get(base_name, 0) + 1
            seen[base_name] = count
            normalized_names.append(base_name if count == 1 else f"{base_name}{count}")
        buckets: dict[str, list[float]] = {key: [] for key in normalized_names}
        for row in reader:
            for index, normalized_name in enumerate(normalized_names):
                raw_value = row[index] if index < len(row) else ""
                if raw_value is None or raw_value.strip() == "":
                    buckets[normalized_name].append(np.nan)
                else:
                    buckets[normalized_name].append(float(raw_value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in buckets.items()}


def _pick_column(
    columns: dict[str, np.ndarray],
    aliases: Sequence[str],
    *,
    csv_path: Path,
    required: bool = True,
) -> Optional[np.ndarray]:
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    if not required:
        return None
    available = ", ".join(sorted(columns.keys()))
    alias_text = ", ".join(aliases)
    raise KeyError(f"Could not find any of [{alias_text}] in {csv_path}; available columns: {available}")


def _length_scale(unit: str) -> float:
    normalized = str(unit).strip().lower()
    if normalized == "m":
        return 1.0
    if normalized == "mm":
        return 1.0e-3
    raise ValueError("--length-unit must be 'm' or 'mm'.")


def _validate_uniform_axis(name: str, coords_m: np.ndarray) -> None:
    diffs = np.diff(coords_m)
    if not np.all(diffs > 0.0):
        raise ValueError(f"{name} coordinates must be strictly increasing.")
    step = float(np.median(diffs))
    tolerance = max(abs(step) * 1.0e-9, 1.0e-15)
    max_error = float(np.max(np.abs(diffs - step)))
    if max_error > tolerance:
        raise ValueError(f"{name} coordinates must be uniformly spaced; max spacing error={max_error:.6g} m.")


def _extract_global_coordinates(
    columns: dict[str, np.ndarray],
    *,
    csv_path: Path,
    coordinate_frame: str,
    length_unit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_x = _pick_column(columns, ("xm", "xmm", "x", "xcoordinate", "xcoord"), csv_path=csv_path)
    raw_y = _pick_column(columns, ("ym", "ymm", "y", "ycoordinate", "ycoord"), csv_path=csv_path)
    raw_z = _pick_column(columns, ("zm", "zmm", "z", "zcoordinate", "zcoord"), csv_path=csv_path)
    assert raw_x is not None and raw_y is not None and raw_z is not None
    scale = _length_scale(length_unit)
    x = np.asarray(raw_x, dtype=np.float64) * scale
    y = np.asarray(raw_y, dtype=np.float64) * scale
    z = np.asarray(raw_z, dtype=np.float64) * scale
    frame = str(coordinate_frame).strip().lower()
    if frame == "global":
        return x, y, z
    if frame == "simion":
        return y, z, x
    raise ValueError("--coordinate-frame must be 'global' or 'simion'.")


def _scatter_regular_values(
    values: np.ndarray,
    shape: tuple[int, int, int],
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    for ii, jj, kk, value in zip(i, j, k, values):
        out[ii, jj, kk] += float(value)
        counts[ii, jj, kk] += 1
    missing = counts == 0
    if np.any(missing):
        raise ValueError(f"Regular 3D mask grid is incomplete; missing {int(np.count_nonzero(missing))} nodes.")
    out = out / counts
    if np.issubdtype(dtype, np.integer):
        return np.rint(out).astype(dtype)
    return out.astype(dtype)


def bake_electrode_mask3d_csv(
    csv_path: Path,
    *,
    output_npz: Path,
    coordinate_frame: str = "global",
    length_unit: str = "mm",
) -> Path:
    columns = _read_csv_columns(csv_path)
    x_m, y_m, z_m = _extract_global_coordinates(
        columns,
        csv_path=csv_path,
        coordinate_frame=coordinate_frame,
        length_unit=length_unit,
    )
    metal_values = _pick_column(
        columns,
        ("iselectrode", "is_electrode", "electrode", "ismetal", "metal", "solid", "mask"),
        csv_path=csv_path,
    )
    electrode_id_values = _pick_column(
        columns,
        ("electrodeid", "electrode_id", "id", "potentialv", "potential", "voltage"),
        csv_path=csv_path,
        required=False,
    )
    assert metal_values is not None
    valid = np.isfinite(x_m) & np.isfinite(y_m) & np.isfinite(z_m) & np.isfinite(metal_values)
    if electrode_id_values is not None:
        valid &= np.isfinite(electrode_id_values)
    if not np.any(valid):
        raise ValueError(f"No valid 3D electrode mask rows remain after filtering: {csv_path}")

    x_coords = np.unique(x_m[valid])
    y_coords = np.unique(y_m[valid])
    z_coords = np.unique(z_m[valid])
    if x_coords.size < 2 or y_coords.size < 2 or z_coords.size < 2:
        raise ValueError(
            "3D electrode mask CSV must contain at least two unique nodes on each global axis "
            f"(got nx={x_coords.size}, ny={y_coords.size}, nz={z_coords.size})."
        )
    _validate_uniform_axis("x", x_coords)
    _validate_uniform_axis("y", y_coords)
    _validate_uniform_axis("z", z_coords)

    i = np.searchsorted(x_coords, x_m[valid])
    j = np.searchsorted(y_coords, y_m[valid])
    k = np.searchsorted(z_coords, z_m[valid])
    shape = (len(x_coords), len(y_coords), len(z_coords))
    metal = _scatter_regular_values(metal_values[valid], shape, i, j, k, dtype=np.float64) != 0.0
    if electrode_id_values is None:
        electrode_id = np.full(shape, -1, dtype=np.int32)
        electrode_id[metal] = 1
    else:
        electrode_id = _scatter_regular_values(electrode_id_values[valid], shape, i, j, k, dtype=np.int32)
        electrode_id = np.where(metal, electrode_id, -1).astype(np.int32, copy=False)

    mask = finalize_electrode_mask3d(
        x_coords_m=x_coords,
        y_coords_m=y_coords,
        z_coords_m=z_coords,
        metal_mask=metal,
        electrode_id=electrode_id,
        metadata={
            "source_path": str(csv_path),
            "source_format": "regular_cartesian_csv",
            "source_coordinate_frame": str(coordinate_frame),
            "source_length_unit": str(length_unit),
            "shape_order": "x_y_z",
        },
    )
    save_electrode_mask3d_npz(Path(output_npz), mask)
    return Path(output_npz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake a regular Cartesian 3D electrode mask from CSV.")
    parser.add_argument("--csv", required=True, help="Regular-grid CSV containing x/y/z and is_electrode columns.")
    parser.add_argument("--output-npz", default="outputs/electrode_mask_3d.npz", help="Output 3D electrode mask .npz path.")
    parser.add_argument(
        "--coordinate-frame",
        choices=("global", "simion"),
        default="global",
        help="Interpret CSV x/y/z as global Python coordinates or SIMION coordinates.",
    )
    parser.add_argument("--length-unit", choices=("m", "mm"), default="mm", help="Coordinate unit used by CSV x/y/z columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = bake_electrode_mask3d_csv(
        Path(args.csv),
        output_npz=Path(args.output_npz),
        coordinate_frame=str(args.coordinate_frame),
        length_unit=str(args.length_unit),
    )
    print(f"Saved 3D electrode mask: {output_path}")


if __name__ == "__main__":
    main()
