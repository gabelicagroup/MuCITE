"""Regular Cartesian 3D electric-field loading and sampling utilities.

This module is intentionally separate from the existing axisymmetric baked field
schema. Multipole guides need true ``E(x, y, z)`` sampling, while the current gas
and PIC space-charge stack can continue to use the established 2D ``(r, z)``
mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class CartesianField3D:
    """Static DC/RF electric field on a regular global Cartesian grid.

    Arrays use shape ``(nx, ny, nz)`` and SI units. The RF arrays represent a
    reference peak-voltage field and are scaled by the runtime RF modulation
    before being added to the DC field.
    """

    x_coords_m: np.ndarray
    y_coords_m: np.ndarray
    z_coords_m: np.ndarray
    e_dc_x_v_per_m: np.ndarray
    e_dc_y_v_per_m: np.ndarray
    e_dc_z_v_per_m: np.ndarray
    e_rf_x_v_per_m: np.ndarray
    e_rf_y_v_per_m: np.ndarray
    e_rf_z_v_per_m: np.ndarray
    rf_reference_peak_voltage_v: float
    metadata: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.x_coords_m), len(self.y_coords_m), len(self.z_coords_m))

    @property
    def has_rf(self) -> bool:
        for values in (self.e_rf_x_v_per_m, self.e_rf_y_v_per_m, self.e_rf_z_v_per_m):
            if values.size > 0 and np.nanmax(np.abs(values)) > 0.0:
                return True
        return False

    def sample(self, position_m: np.ndarray, rf_modulation: float = 0.0) -> np.ndarray:
        """Sample ``E_dc + rf_modulation * E_rf_ref`` at one position."""

        position = np.asarray(position_m, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"position_m must have shape (3,), received {position.shape}.")
        return self.sample_many(position.reshape(1, 3), rf_modulation=rf_modulation)[0]

    def sample_many(self, positions_m: np.ndarray, rf_modulation: float = 0.0) -> np.ndarray:
        """Vectorized trilinear sampling for positions shaped ``(n, 3)``."""

        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions_m must have shape (n, 3), received {positions.shape}.")
        if positions.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)

        ex = _trilinear_many(self.e_dc_x_v_per_m, self.x_coords_m, self.y_coords_m, self.z_coords_m, positions)
        ey = _trilinear_many(self.e_dc_y_v_per_m, self.x_coords_m, self.y_coords_m, self.z_coords_m, positions)
        ez = _trilinear_many(self.e_dc_z_v_per_m, self.x_coords_m, self.y_coords_m, self.z_coords_m, positions)
        if rf_modulation != 0.0:
            ex = ex + float(rf_modulation) * _trilinear_many(
                self.e_rf_x_v_per_m,
                self.x_coords_m,
                self.y_coords_m,
                self.z_coords_m,
                positions,
            )
            ey = ey + float(rf_modulation) * _trilinear_many(
                self.e_rf_y_v_per_m,
                self.x_coords_m,
                self.y_coords_m,
                self.z_coords_m,
                positions,
            )
            ez = ez + float(rf_modulation) * _trilinear_many(
                self.e_rf_z_v_per_m,
                self.x_coords_m,
                self.y_coords_m,
                self.z_coords_m,
                positions,
            )
        return np.column_stack((ex, ey, ez))

    def metadata_summary(self) -> dict[str, Any]:
        """Return lightweight metadata suitable for reports."""

        return {
            "loaded": True,
            "nx": int(len(self.x_coords_m)),
            "ny": int(len(self.y_coords_m)),
            "nz": int(len(self.z_coords_m)),
            "x_min_m": float(self.x_coords_m[0]),
            "x_max_m": float(self.x_coords_m[-1]),
            "y_min_m": float(self.y_coords_m[0]),
            "y_max_m": float(self.y_coords_m[-1]),
            "z_min_m": float(self.z_coords_m[0]),
            "z_max_m": float(self.z_coords_m[-1]),
            "dx_m": float(np.median(np.diff(self.x_coords_m))),
            "dy_m": float(np.median(np.diff(self.y_coords_m))),
            "dz_m": float(np.median(np.diff(self.z_coords_m))),
            "has_rf": bool(self.has_rf),
            "rf_reference_peak_voltage_v": float(self.rf_reference_peak_voltage_v),
            **{
                key: value
                for key, value in self.metadata.items()
                if not isinstance(value, np.ndarray)
            },
        }


def _validate_axis(name: str, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a 1D coordinate array with at least two nodes.")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    diffs = np.diff(axis)
    step = float(np.median(diffs))
    tolerance = max(abs(step) * 1.0e-9, 1.0e-15)
    if np.max(np.abs(diffs - step)) > tolerance:
        raise ValueError(
            f"{name} must be uniformly spaced for the Taichi 3D field gather; "
            f"max spacing error={np.max(np.abs(diffs - step)):.6g} m."
        )
    return np.ascontiguousarray(axis)


def _require_array(
    group: dict[str, Any],
    name: str,
    expected_shape: tuple[int, int, int],
    *,
    required: bool,
) -> np.ndarray:
    if name not in group:
        if required:
            raise ValueError(f"3D electric field is missing electric3d.{name}.")
        return np.zeros(expected_shape, dtype=np.float64)
    values = np.ascontiguousarray(np.asarray(group[name], dtype=np.float64))
    if values.shape != expected_shape:
        raise ValueError(f"electric3d.{name} has shape {values.shape}, expected {expected_shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"electric3d.{name} contains non-finite values.")
    return values


def _trilinear_many(
    values: np.ndarray,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    positions_m: np.ndarray,
) -> np.ndarray:
    x = np.clip(positions_m[:, 0], float(x_coords_m[0]), float(x_coords_m[-1]))
    y = np.clip(positions_m[:, 1], float(y_coords_m[0]), float(y_coords_m[-1]))
    z = np.clip(positions_m[:, 2], float(z_coords_m[0]), float(z_coords_m[-1]))

    dx = float(x_coords_m[1] - x_coords_m[0])
    dy = float(y_coords_m[1] - y_coords_m[0])
    dz = float(z_coords_m[1] - z_coords_m[0])

    fx = (x - float(x_coords_m[0])) / dx
    fy = (y - float(y_coords_m[0])) / dy
    fz = (z - float(z_coords_m[0])) / dz

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


def _field_groups(
    baked: Any,
    field_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(baked, dict):
        raise ValueError(
            f"3D static field file must contain a dictionary: {field_path}"
        )
    if "grid3d" not in baked or "electric3d" not in baked:
        raise ValueError(
            "3D static field file must contain 'grid3d' and "
            f"'electric3d' groups: {field_path}"
        )
    grid, electric = baked["grid3d"], baked["electric3d"]
    if not isinstance(grid, dict) or not isinstance(electric, dict):
        raise ValueError(
            "3D static field 'grid3d' and 'electric3d' groups "
            "must be dictionaries."
        )
    return grid, electric


def _electric_components(
    electric: dict[str, Any],
    expected_shape: tuple[int, int, int],
) -> tuple[np.ndarray, ...]:
    names = (
        "e_dc_x_v_per_m",
        "e_dc_y_v_per_m",
        "e_dc_z_v_per_m",
        "e_rf_x_v_per_m",
        "e_rf_y_v_per_m",
        "e_rf_z_v_per_m",
    )
    return tuple(
        _require_array(
            electric, name, expected_shape, required=name.startswith("e_dc")
        )
        for name in names
    )


def load_baked_cartesian_field3d(path: Path) -> CartesianField3D:
    """Load a ``field_baker_3d.py`` exported Cartesian field file."""

    field_path = Path(path)
    if not field_path.exists():
        raise FileNotFoundError(f"3D static field file not found: {field_path}")

    grid, electric = _field_groups(
        np.load(field_path, allow_pickle=True).item(), field_path
    )
    x_coords_m = _validate_axis("grid3d.x_coords_m", grid["x_coords_m"])
    y_coords_m = _validate_axis("grid3d.y_coords_m", grid["y_coords_m"])
    z_coords_m = _validate_axis("grid3d.z_coords_m", grid["z_coords_m"])
    expected_shape = (len(x_coords_m), len(y_coords_m), len(z_coords_m))
    e_dc_x, e_dc_y, e_dc_z, e_rf_x, e_rf_y, e_rf_z = (
        _electric_components(electric, expected_shape)
    )
    rf_reference_peak_voltage_v = float(electric.get("rf_reference_peak_voltage_v", 1.0))
    if not np.isfinite(rf_reference_peak_voltage_v) or rf_reference_peak_voltage_v <= 0.0:
        raise ValueError(f"Invalid electric3d.rf_reference_peak_voltage_v: {rf_reference_peak_voltage_v}")

    metadata = {
        key: value
        for key, value in {**grid, **electric}.items()
        if not isinstance(value, np.ndarray)
    }
    metadata["source_path"] = str(field_path)
    return CartesianField3D(
        x_coords_m=x_coords_m,
        y_coords_m=y_coords_m,
        z_coords_m=z_coords_m,
        e_dc_x_v_per_m=e_dc_x,
        e_dc_y_v_per_m=e_dc_y,
        e_dc_z_v_per_m=e_dc_z,
        e_rf_x_v_per_m=e_rf_x,
        e_rf_y_v_per_m=e_rf_y,
        e_rf_z_v_per_m=e_rf_z,
        rf_reference_peak_voltage_v=rf_reference_peak_voltage_v,
        metadata=metadata,
    )
