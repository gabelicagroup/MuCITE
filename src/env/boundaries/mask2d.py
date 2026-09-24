"""Axisymmetric electrode-mask model, validation, and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class ElectrodeMask:
    r_coords_m: np.ndarray
    z_coords_m: np.ndarray
    metal_mask: np.ndarray
    electrode_id: np.ndarray
    surface_distance_m: np.ndarray
    metadata: dict[str, Any]


def surface_distance_from_mask(
    metal_mask: np.ndarray,
    dr_m: float,
    dz_m: float,
) -> np.ndarray:
    metal = np.asarray(metal_mask, dtype=bool)
    surface = np.zeros_like(metal, dtype=bool)
    surface[1:, :] |= metal[1:, :] != metal[:-1, :]
    surface[:-1, :] |= metal[1:, :] != metal[:-1, :]
    surface[:, 1:] |= metal[:, 1:] != metal[:, :-1]
    surface[:, :-1] |= metal[:, 1:] != metal[:, :-1]

    if not np.any(surface):
        fill = -np.inf if np.all(metal) else np.inf
        return np.full(metal.shape, fill, dtype=np.float32)

    distance_m = distance_transform_edt(~surface, sampling=(dr_m, dz_m))
    signed_m = np.where(metal, -distance_m, distance_m)
    return signed_m.astype(np.float32, copy=False)


def _validated_mask_arrays(
    r_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    metal_mask: np.ndarray,
    electrode_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_coords = np.asarray(r_coords_m, dtype=float)
    z_coords = np.asarray(z_coords_m, dtype=float)
    metal = np.asarray(metal_mask, dtype=bool)
    ids = np.asarray(electrode_id, dtype=np.int32)
    if r_coords.ndim != 1 or z_coords.ndim != 1:
        raise ValueError("Electrode mask coordinates must be 1D arrays.")
    expected_shape = (len(r_coords), len(z_coords))
    if metal.shape != expected_shape:
        raise ValueError(f"Electrode mask shape={metal.shape}, expected={expected_shape}.")
    if ids.shape != expected_shape:
        raise ValueError("Electrode id array shape must match metal_mask.")
    return r_coords, z_coords, metal, ids


def _resolved_surface_distance(
    surface_distance_m: Optional[np.ndarray],
    metal_mask: np.ndarray,
    spacing_m: tuple[float, float],
    metadata: dict[str, Any],
) -> np.ndarray:
    if surface_distance_m is None:
        metadata["surface_distance_source"] = "python_edt_from_metal_mask"
        return surface_distance_from_mask(metal_mask, *spacing_m)
    distance = np.asarray(surface_distance_m, dtype=np.float32)
    if distance.shape != metal_mask.shape:
        raise ValueError("Surface distance array shape must match metal_mask.")
    return distance


def _update_mask_metadata(
    metadata: dict[str, Any],
    r_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    metal_mask: np.ndarray,
    electrode_id: np.ndarray,
    spacing_m: tuple[float, float],
) -> None:
    metadata.update({
        "loaded": True,
        "shape": [int(metal_mask.shape[0]), int(metal_mask.shape[1])],
        "r_min_m": float(r_coords_m[0]),
        "r_max_m": float(r_coords_m[-1]),
        "z_min_m": float(z_coords_m[0]),
        "z_max_m": float(z_coords_m[-1]),
        "dr_m": spacing_m[0],
        "dz_m": spacing_m[1],
        "metal_point_count": int(np.count_nonzero(metal_mask)),
        "metal_fraction": float(np.mean(metal_mask)),
        "electrode_ids_present": sorted(
            int(value) for value in np.unique(electrode_id[metal_mask])
            if int(value) >= 0
        ),
    })


def finalize_electrode_mask(
    *,
    r_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    metal_mask: np.ndarray,
    electrode_id: np.ndarray,
    surface_distance_m: Optional[np.ndarray],
    metadata: dict[str, Any],
) -> ElectrodeMask:
    r_coords_m, z_coords_m, metal_mask, electrode_id = (
        _validated_mask_arrays(
            r_coords_m, z_coords_m, metal_mask, electrode_id
        )
    )
    spacing_m = (
        float(np.median(np.diff(r_coords_m))),
        float(np.median(np.diff(z_coords_m))),
    )
    surface_distance_m = _resolved_surface_distance(
        surface_distance_m, metal_mask, spacing_m, metadata
    )
    _update_mask_metadata(
        metadata, r_coords_m, z_coords_m, metal_mask, electrode_id, spacing_m
    )
    return ElectrodeMask(
        r_coords_m=r_coords_m,
        z_coords_m=z_coords_m,
        metal_mask=metal_mask,
        electrode_id=electrode_id,
        surface_distance_m=surface_distance_m,
        metadata=metadata,
    )


def bilinear_sample(
    values: np.ndarray,
    r_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
) -> np.ndarray:
    r_clipped = np.clip(r_m, r_coords_m[0], r_coords_m[-1])
    z_clipped = np.clip(z_m, z_coords_m[0], z_coords_m[-1])

    ir = np.searchsorted(r_coords_m, r_clipped, side="right") - 1
    iz = np.searchsorted(z_coords_m, z_clipped, side="right") - 1
    ir = np.clip(ir, 0, len(r_coords_m) - 2)
    iz = np.clip(iz, 0, len(z_coords_m) - 2)

    r0 = r_coords_m[ir]
    r1 = r_coords_m[ir + 1]
    z0 = z_coords_m[iz]
    z1 = z_coords_m[iz + 1]
    wr = (r_clipped - r0) / (r1 - r0)
    wz = (z_clipped - z0) / (z1 - z0)

    v00 = values[ir, iz]
    v10 = values[ir + 1, iz]
    v01 = values[ir, iz + 1]
    v11 = values[ir + 1, iz + 1]
    return (1.0 - wr) * (1.0 - wz) * v00 + wr * (1.0 - wz) * v10 + (1.0 - wr) * wz * v01 + wr * wz * v11


def nearest_sample(
    values: np.ndarray,
    r_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    fill_value: Any,
) -> tuple[np.ndarray, np.ndarray]:
    inside = (
        (r_m >= r_coords_m[0])
        & (r_m <= r_coords_m[-1])
        & (z_m >= z_coords_m[0])
        & (z_m <= z_coords_m[-1])
    )
    result = np.full(r_m.shape, fill_value, dtype=values.dtype)
    if not np.any(inside):
        return result, inside

    r_inside = r_m[inside]
    z_inside = z_m[inside]
    ir_hi = np.searchsorted(r_coords_m, r_inside, side="left")
    iz_hi = np.searchsorted(z_coords_m, z_inside, side="left")
    ir_hi = np.clip(ir_hi, 1, len(r_coords_m) - 1)
    iz_hi = np.clip(iz_hi, 1, len(z_coords_m) - 1)
    ir_lo = ir_hi - 1
    iz_lo = iz_hi - 1
    choose_hi_r = np.abs(r_coords_m[ir_hi] - r_inside) < np.abs(r_inside - r_coords_m[ir_lo])
    choose_hi_z = np.abs(z_coords_m[iz_hi] - z_inside) < np.abs(z_inside - z_coords_m[iz_lo])
    ir = np.where(choose_hi_r, ir_hi, ir_lo)
    iz = np.where(choose_hi_z, iz_hi, iz_lo)
    result[inside] = values[ir, iz]
    return result, inside


def sample_electrode_mask(
    mask: ElectrodeMask,
    positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions_m = np.asarray(positions_m, dtype=float)
    r_m = np.hypot(positions_m[:, 0], positions_m[:, 1])
    z_m = positions_m[:, 2]
    metal, inside = nearest_sample(mask.metal_mask, mask.r_coords_m, mask.z_coords_m, r_m, z_m, False)
    electrode_id, _ = nearest_sample(mask.electrode_id, mask.r_coords_m, mask.z_coords_m, r_m, z_m, -1)
    surface_distance_m = bilinear_sample(mask.surface_distance_m, mask.r_coords_m, mask.z_coords_m, r_m, z_m)
    surface_distance_m = np.where(inside, surface_distance_m, np.nan)
    return metal.astype(bool), electrode_id.astype(np.int32), surface_distance_m, inside


__all__ = [
    "ElectrodeMask",
    "bilinear_sample",
    "finalize_electrode_mask",
    "nearest_sample",
    "sample_electrode_mask",
    "surface_distance_from_mask",
]
