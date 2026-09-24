"""File adapters for Cartesian 3D electrode-mask models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ...env.boundaries.mask3d import (
    ElectrodeMask3D,
    finalize_electrode_mask3d,
)
from .metadata import json_ready


def _load_npz(path: Path) -> ElectrodeMask3D:
    with np.load(path, allow_pickle=True) as data:
        metadata = (
            json.loads(str(data["metadata_json"].item()))
            if "metadata_json" in data
            else {}
        )
        metadata["source_format"] = metadata.get(
            "source_format",
            "npz_3d_mask",
        )
        metadata["source_path"] = str(path)
        return finalize_electrode_mask3d(
            x_coords_m=np.asarray(data["x_coords_m"], dtype=np.float64),
            y_coords_m=np.asarray(data["y_coords_m"], dtype=np.float64),
            z_coords_m=np.asarray(data["z_coords_m"], dtype=np.float64),
            metal_mask=np.asarray(data["metal_mask"], dtype=bool),
            electrode_id=(
                np.asarray(data["electrode_id"], dtype=np.int32)
                if "electrode_id" in data
                else None
            ),
            surface_distance_m=(
                np.asarray(data["surface_distance_m"], dtype=np.float32)
                if "surface_distance_m" in data
                else None
            ),
            metadata=metadata,
        )


def _npy_groups(payload: object, path: Path) -> tuple[dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"3D electrode mask .npy must contain a dictionary: {path}"
        )
    grid = payload.get("grid3d", payload.get("grid", payload))
    mask = payload.get(
        "mask3d",
        payload.get("electrode_mask3d", payload.get("mask", payload)),
    )
    if not isinstance(grid, dict) or not isinstance(mask, dict):
        raise ValueError(
            "3D electrode mask dictionary must contain grid/mask "
            "dictionaries or top-level arrays."
        )
    return grid, mask


def _load_npy(path: Path) -> ElectrodeMask3D:
    grid, mask_group = _npy_groups(
        np.load(path, allow_pickle=True).item(),
        path,
    )
    metadata = {
        key: value
        for key, value in {**grid, **mask_group}.items()
        if not isinstance(value, np.ndarray)
    }
    metadata["source_format"] = metadata.get(
        "source_format",
        "npy_3d_mask",
    )
    metadata["source_path"] = str(path)
    return finalize_electrode_mask3d(
        x_coords_m=grid["x_coords_m"],
        y_coords_m=grid["y_coords_m"],
        z_coords_m=grid["z_coords_m"],
        metal_mask=mask_group["metal_mask"],
        electrode_id=mask_group.get("electrode_id"),
        surface_distance_m=mask_group.get("surface_distance_m"),
        metadata=metadata,
    )


def save_electrode_mask3d_npz(
    path: Path,
    mask: ElectrodeMask3D,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        x_coords_m=mask.x_coords_m,
        y_coords_m=mask.y_coords_m,
        z_coords_m=mask.z_coords_m,
        metal_mask=mask.metal_mask.astype(np.uint8),
        electrode_id=mask.electrode_id.astype(np.int32),
        surface_distance_m=mask.surface_distance_m.astype(np.float32),
        metadata_json=json.dumps(
            json_ready(mask.metadata),
            sort_keys=True,
        ),
    )


def load_electrode_mask3d(
    source_path: Optional[Path],
) -> Optional[ElectrodeMask3D]:
    if source_path is None:
        return None
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(
            f"3D electrode mask path does not exist: {path}"
        )
    if path.suffix.lower() == ".npz":
        return _load_npz(path)
    if path.suffix.lower() == ".npy":
        return _load_npy(path)
    raise ValueError(
        f"Unsupported 3D electrode mask format {path.suffix}; "
        "use .npy or .npz."
    )


__all__ = [
    "load_electrode_mask3d",
    "save_electrode_mask3d_npz",
]
