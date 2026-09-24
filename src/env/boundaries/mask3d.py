"""Cartesian 3D electrode-mask model, validation, and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class ElectrodeMask3D:
    """Regular Cartesian electrode mask with shape ``(nx, ny, nz)``."""

    x_coords_m: np.ndarray
    y_coords_m: np.ndarray
    z_coords_m: np.ndarray
    metal_mask: np.ndarray
    electrode_id: np.ndarray
    surface_distance_m: np.ndarray
    metadata: dict[str, Any]


def _validate_axis(name: str, coords_m: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_m, dtype=np.float64)
    if coords.ndim != 1 or coords.size < 2:
        raise ValueError(f"{name} must be a 1D coordinate array with at least two nodes.")
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.all(np.diff(coords) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    diffs = np.diff(coords)
    step = float(np.median(diffs))
    tolerance = max(abs(step) * 1.0e-9, 1.0e-15)
    if np.max(np.abs(diffs - step)) > tolerance:
        raise ValueError(
            f"{name} must be uniformly spaced for Taichi 3D mask sampling; "
            f"max spacing error={np.max(np.abs(diffs - step)):.6g} m."
        )
    return np.ascontiguousarray(coords)


def surface_distance_from_mask3d(
    metal_mask: np.ndarray,
    dx_m: float,
    dy_m: float,
    dz_m: float,
) -> np.ndarray:
    """Return signed distance to the nearest metal/vacuum interface."""

    metal = np.asarray(metal_mask, dtype=bool)
    surface = np.zeros_like(metal, dtype=bool)
    surface[1:, :, :] |= metal[1:, :, :] != metal[:-1, :, :]
    surface[:-1, :, :] |= metal[1:, :, :] != metal[:-1, :, :]
    surface[:, 1:, :] |= metal[:, 1:, :] != metal[:, :-1, :]
    surface[:, :-1, :] |= metal[:, 1:, :] != metal[:, :-1, :]
    surface[:, :, 1:] |= metal[:, :, 1:] != metal[:, :, :-1]
    surface[:, :, :-1] |= metal[:, :, 1:] != metal[:, :, :-1]

    if not np.any(surface):
        fill = -np.inf if np.all(metal) else np.inf
        return np.full(metal.shape, fill, dtype=np.float32)

    distance_m = distance_transform_edt(~surface, sampling=(dx_m, dy_m, dz_m))
    signed_m = np.where(metal, -distance_m, distance_m)
    return signed_m.astype(np.float32, copy=False)


def _validated_mask3d_arrays(
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    metal_mask: np.ndarray,
    electrode_id: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_coords = _validate_axis("x_coords_m", x_coords_m)
    y_coords = _validate_axis("y_coords_m", y_coords_m)
    z_coords = _validate_axis("z_coords_m", z_coords_m)
    metal = np.ascontiguousarray(np.asarray(metal_mask, dtype=bool))
    expected_shape = (len(x_coords), len(y_coords), len(z_coords))
    if metal.shape != expected_shape:
        raise ValueError(
            f"3D electrode metal_mask shape={metal.shape}, expected={expected_shape}."
        )
    ids = _validated_electrode_ids(electrode_id, metal, expected_shape)
    return x_coords, y_coords, z_coords, metal, ids


def _validated_electrode_ids(
    electrode_id: Optional[np.ndarray],
    metal_mask: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    if electrode_id is None:
        ids = np.full(expected_shape, -1, dtype=np.int32)
        ids[metal_mask] = 1
        return ids
    ids = np.ascontiguousarray(np.asarray(electrode_id, dtype=np.int32))
    if ids.shape != expected_shape:
        raise ValueError(
            f"3D electrode_id shape={ids.shape}, expected={expected_shape}."
        )
    return ids


def _resolved_surface_distance3d(
    surface_distance_m: Optional[np.ndarray],
    metal_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    metadata: dict[str, Any],
) -> np.ndarray:
    if surface_distance_m is None:
        metadata["surface_distance_source"] = (
            "python_edt_from_metal_mask_3d"
        )
        return surface_distance_from_mask3d(metal_mask, *spacing_m)
    distance = np.ascontiguousarray(
        np.asarray(surface_distance_m, dtype=np.float32)
    )
    if distance.shape != metal_mask.shape:
        raise ValueError(
            "3D surface_distance_m shape="
            f"{distance.shape}, expected={metal_mask.shape}."
        )
    return distance


def _update_mask3d_metadata(
    metadata: dict[str, Any],
    coords_m: tuple[np.ndarray, np.ndarray, np.ndarray],
    metal_mask: np.ndarray,
    electrode_id: np.ndarray,
    spacing_m: tuple[float, float, float],
) -> None:
    x_coords, y_coords, z_coords = coords_m
    metadata.update({
        "loaded": True,
        "dimension": 3,
        "shape": [int(value) for value in metal_mask.shape],
        "x_min_m": float(x_coords[0]),
        "x_max_m": float(x_coords[-1]),
        "y_min_m": float(y_coords[0]),
        "y_max_m": float(y_coords[-1]),
        "z_min_m": float(z_coords[0]),
        "z_max_m": float(z_coords[-1]),
        "dx_m": spacing_m[0],
        "dy_m": spacing_m[1],
        "dz_m": spacing_m[2],
        "metal_point_count": int(np.count_nonzero(metal_mask)),
        "metal_fraction": float(np.mean(metal_mask)),
        "electrode_ids_present": sorted(
            int(value) for value in np.unique(electrode_id[metal_mask])
            if int(value) >= 0
        ),
    })


def finalize_electrode_mask3d(
    *,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    metal_mask: np.ndarray,
    electrode_id: Optional[np.ndarray] = None,
    surface_distance_m: Optional[np.ndarray] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ElectrodeMask3D:
    meta = {} if metadata is None else dict(metadata)
    x_coords, y_coords, z_coords, metal, ids = _validated_mask3d_arrays(
        x_coords_m, y_coords_m, z_coords_m, metal_mask, electrode_id
    )
    spacing_m = tuple(
        float(np.median(np.diff(coords)))
        for coords in (x_coords, y_coords, z_coords)
    )
    surface_distance = _resolved_surface_distance3d(
        surface_distance_m, metal, spacing_m, meta
    )
    _update_mask3d_metadata(
        meta, (x_coords, y_coords, z_coords), metal, ids, spacing_m
    )
    return ElectrodeMask3D(
        x_coords_m=x_coords,
        y_coords_m=y_coords,
        z_coords_m=z_coords,
        metal_mask=metal,
        electrode_id=ids,
        surface_distance_m=surface_distance,
        metadata=meta,
    )


def _trilinear_sample(
    values: np.ndarray,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    positions_m: np.ndarray,
) -> np.ndarray:
    x = np.clip(positions_m[:, 0], x_coords_m[0], x_coords_m[-1])
    y = np.clip(positions_m[:, 1], y_coords_m[0], y_coords_m[-1])
    z = np.clip(positions_m[:, 2], z_coords_m[0], z_coords_m[-1])
    dx = float(x_coords_m[1] - x_coords_m[0])
    dy = float(y_coords_m[1] - y_coords_m[0])
    dz = float(z_coords_m[1] - z_coords_m[0])
    fx = (x - x_coords_m[0]) / dx
    fy = (y - y_coords_m[0]) / dy
    fz = (z - z_coords_m[0]) / dz
    i0 = np.clip(np.floor(fx).astype(np.int64), 0, len(x_coords_m) - 2)
    j0 = np.clip(np.floor(fy).astype(np.int64), 0, len(y_coords_m) - 2)
    k0 = np.clip(np.floor(fz).astype(np.int64), 0, len(z_coords_m) - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    wx = fx - i0.astype(np.float64)
    wy = fy - j0.astype(np.float64)
    wz = fz - k0.astype(np.float64)
    c00 = (1.0 - wx) * values[i0, j0, k0] + wx * values[i1, j0, k0]
    c10 = (1.0 - wx) * values[i0, j1, k0] + wx * values[i1, j1, k0]
    c01 = (1.0 - wx) * values[i0, j0, k1] + wx * values[i1, j0, k1]
    c11 = (1.0 - wx) * values[i0, j1, k1] + wx * values[i1, j1, k1]
    c0 = (1.0 - wy) * c00 + wy * c10
    c1 = (1.0 - wy) * c01 + wy * c11
    return (1.0 - wz) * c0 + wz * c1


def sample_electrode_mask3d(
    mask: ElectrodeMask3D,
    positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_m must have shape (n, 3), received {positions.shape}.")
    inside = (
        (positions[:, 0] >= mask.x_coords_m[0])
        & (positions[:, 0] <= mask.x_coords_m[-1])
        & (positions[:, 1] >= mask.y_coords_m[0])
        & (positions[:, 1] <= mask.y_coords_m[-1])
        & (positions[:, 2] >= mask.z_coords_m[0])
        & (positions[:, 2] <= mask.z_coords_m[-1])
    )
    metal = np.zeros(positions.shape[0], dtype=bool)
    electrode_id = np.full(positions.shape[0], -1, dtype=np.int32)
    surface_distance_m = np.full(positions.shape[0], np.nan, dtype=np.float64)
    if not np.any(inside):
        return metal, electrode_id, surface_distance_m, inside

    local = positions[inside]
    dx = float(mask.x_coords_m[1] - mask.x_coords_m[0])
    dy = float(mask.y_coords_m[1] - mask.y_coords_m[0])
    dz = float(mask.z_coords_m[1] - mask.z_coords_m[0])
    ix = np.clip(np.floor((local[:, 0] - mask.x_coords_m[0]) / dx + 0.5).astype(np.int64), 0, len(mask.x_coords_m) - 1)
    iy = np.clip(np.floor((local[:, 1] - mask.y_coords_m[0]) / dy + 0.5).astype(np.int64), 0, len(mask.y_coords_m) - 1)
    iz = np.clip(np.floor((local[:, 2] - mask.z_coords_m[0]) / dz + 0.5).astype(np.int64), 0, len(mask.z_coords_m) - 1)
    metal[inside] = mask.metal_mask[ix, iy, iz]
    electrode_id[inside] = mask.electrode_id[ix, iy, iz]
    surface_distance_m[inside] = _trilinear_sample(
        mask.surface_distance_m,
        mask.x_coords_m,
        mask.y_coords_m,
        mask.z_coords_m,
        local,
    )
    return metal, electrode_id, surface_distance_m, inside


__all__ = [
    "ElectrodeMask3D",
    "finalize_electrode_mask3d",
    "sample_electrode_mask3d",
    "surface_distance_from_mask3d",
]
