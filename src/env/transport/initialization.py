"""Particle-pusher field and boundary-buffer initialization."""

from typing import Any, Optional

import numpy as np
import taichi as ti


def install_cartesian_field3d(owner: Any, source: Optional[object]) -> None:
    """Install optional Cartesian DC/RF overlay fields."""

    arrays = _cartesian_field_payload(owner, source)
    shape = (owner.field3d_nx, owner.field3d_ny, owner.field3d_nz)
    owner.E3d_dc_x = ti.field(dtype=ti.f64, shape=shape)
    owner.E3d_dc_y = ti.field(dtype=ti.f64, shape=shape)
    owner.E3d_dc_z = ti.field(dtype=ti.f64, shape=shape)
    owner.E3d_rf_x = ti.field(dtype=ti.f64, shape=shape)
    owner.E3d_rf_y = ti.field(dtype=ti.f64, shape=shape)
    owner.E3d_rf_z = ti.field(dtype=ti.f64, shape=shape)
    for field, values in zip(
        (
            owner.E3d_dc_x,
            owner.E3d_dc_y,
            owner.E3d_dc_z,
            owner.E3d_rf_x,
            owner.E3d_rf_y,
            owner.E3d_rf_z,
        ),
        arrays,
    ):
        field.from_numpy(values)


def _cartesian_field_payload(
    owner: Any,
    source: Optional[object],
) -> tuple[np.ndarray, ...]:
    owner.has_cartesian_field3d = source is not None
    if source is None:
        _set_cartesian_fallback_geometry(owner)
        zeros = np.zeros((2, 2, 2), dtype=np.float64)
        return (zeros, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())
    x_coords = np.asarray(source.x_coords_m, dtype=np.float64)
    y_coords = np.asarray(source.y_coords_m, dtype=np.float64)
    z_coords = np.asarray(source.z_coords_m, dtype=np.float64)
    owner.field3d_nx = int(len(x_coords))
    owner.field3d_ny = int(len(y_coords))
    owner.field3d_nz = int(len(z_coords))
    owner.field3d_x_min_m, owner.field3d_x_max_m = float(x_coords[0]), float(x_coords[-1])
    owner.field3d_y_min_m, owner.field3d_y_max_m = float(y_coords[0]), float(y_coords[-1])
    owner.field3d_z_min_m, owner.field3d_z_max_m = float(z_coords[0]), float(z_coords[-1])
    owner.field3d_dx_m = float(np.median(np.diff(x_coords)))
    owner.field3d_dy_m = float(np.median(np.diff(y_coords)))
    owner.field3d_dz_m = float(np.median(np.diff(z_coords)))
    return tuple(
        np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        for values in (
            source.e_dc_x_v_per_m,
            source.e_dc_y_v_per_m,
            source.e_dc_z_v_per_m,
            source.e_rf_x_v_per_m,
            source.e_rf_y_v_per_m,
            source.e_rf_z_v_per_m,
        )
    )


def _set_cartesian_fallback_geometry(owner: Any) -> None:
    owner.field3d_nx = owner.field3d_ny = owner.field3d_nz = 2
    owner.field3d_x_min_m = owner.field3d_y_min_m = owner.field3d_z_min_m = 0.0
    owner.field3d_x_max_m = owner.field3d_y_max_m = owner.field3d_z_max_m = 1.0
    owner.field3d_dx_m = owner.field3d_dy_m = owner.field3d_dz_m = 1.0


def install_electrode_mask3d(owner: Any, source: Optional[object]) -> None:
    """Install optional Cartesian electrode geometry buffers."""

    metal, electrode_id, distance = _electrode_mask3d_payload(owner, source)
    shape = (owner.mask3d_nx, owner.mask3d_ny, owner.mask3d_nz)
    owner.electrode3d_metal_mask = ti.field(dtype=ti.i32, shape=shape)
    owner.electrode3d_id = ti.field(dtype=ti.i32, shape=shape)
    owner.electrode3d_surface_distance = ti.field(dtype=ti.f64, shape=shape)
    owner.electrode3d_metal_mask.from_numpy(metal)
    owner.electrode3d_id.from_numpy(electrode_id)
    owner.electrode3d_surface_distance.from_numpy(distance)


def _electrode_mask3d_payload(
    owner: Any,
    source: Optional[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owner.has_electrode_mask3d = source is not None
    if source is None:
        _set_mask3d_fallback_geometry(owner)
        return (
            np.zeros((2, 2, 2), dtype=np.int32),
            np.full((2, 2, 2), -1, dtype=np.int32),
            np.full((2, 2, 2), 1.0e99, dtype=np.float64),
        )
    metal = np.asarray(source.metal_mask, dtype=np.int32)
    owner.mask3d_nx, owner.mask3d_ny, owner.mask3d_nz = map(int, metal.shape)
    owner.mask3d_x_min_m, owner.mask3d_x_max_m = float(source.x_coords_m[0]), float(source.x_coords_m[-1])
    owner.mask3d_y_min_m, owner.mask3d_y_max_m = float(source.y_coords_m[0]), float(source.y_coords_m[-1])
    owner.mask3d_z_min_m, owner.mask3d_z_max_m = float(source.z_coords_m[0]), float(source.z_coords_m[-1])
    owner.mask3d_dx_m = float(np.median(np.diff(source.x_coords_m)))
    owner.mask3d_dy_m = float(np.median(np.diff(source.y_coords_m)))
    owner.mask3d_dz_m = float(np.median(np.diff(source.z_coords_m)))
    return (
        metal,
        np.asarray(source.electrode_id, dtype=np.int32),
        np.asarray(source.surface_distance_m, dtype=np.float64),
    )


def _set_mask3d_fallback_geometry(owner: Any) -> None:
    owner.mask3d_nx = owner.mask3d_ny = owner.mask3d_nz = 2
    owner.mask3d_x_min_m = owner.mask3d_y_min_m = owner.mask3d_z_min_m = 0.0
    owner.mask3d_x_max_m = owner.mask3d_y_max_m = owner.mask3d_z_max_m = 1.0
    owner.mask3d_dx_m = owner.mask3d_dy_m = owner.mask3d_dz_m = 1.0


def install_electrode_mask2d(owner: Any, source: Optional[object]) -> None:
    """Install optional axisymmetric electrode geometry buffers."""

    metal, electrode_id, distance = _electrode_mask2d_payload(owner, source)
    shape = (owner.mask_nr, owner.mask_nz)
    owner.electrode_metal_mask = ti.field(dtype=ti.i32, shape=shape)
    owner.electrode_id = ti.field(dtype=ti.i32, shape=shape)
    owner.electrode_surface_distance = ti.field(dtype=ti.f64, shape=shape)
    owner.electrode_metal_mask.from_numpy(metal)
    owner.electrode_id.from_numpy(electrode_id)
    owner.electrode_surface_distance.from_numpy(distance)


def _electrode_mask2d_payload(
    owner: Any,
    source: Optional[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owner.has_electrode_mask = source is not None
    if source is None:
        _set_mask2d_fallback_geometry(owner)
        return (
            np.zeros((3, 3), dtype=np.int32),
            np.full((3, 3), -1, dtype=np.int32),
            np.full((3, 3), 1.0e99, dtype=np.float64),
        )
    metal = np.asarray(source.metal_mask, dtype=np.int32)
    owner.mask_nr, owner.mask_nz = map(int, metal.shape)
    owner.mask_r_min_m, owner.mask_r_max_m = float(source.r_coords_m[0]), float(source.r_coords_m[-1])
    owner.mask_z_min_m, owner.mask_z_max_m = float(source.z_coords_m[0]), float(source.z_coords_m[-1])
    owner.mask_dr_m = float(np.median(np.diff(source.r_coords_m)))
    owner.mask_dz_m = float(np.median(np.diff(source.z_coords_m)))
    return (
        metal,
        np.asarray(source.electrode_id, dtype=np.int32),
        np.asarray(source.surface_distance_m, dtype=np.float64),
    )


def _set_mask2d_fallback_geometry(owner: Any) -> None:
    owner.mask_nr = owner.mask_nz = 3
    owner.mask_r_min_m = owner.mask_z_min_m = 0.0
    owner.mask_r_max_m = owner.mask_z_max_m = 1.0
    owner.mask_dr_m = owner.mask_dz_m = 0.5


def configure_advection_cell_size(owner: Any) -> None:
    """Use the smallest active field/geometry spacing for the CFL limiter."""

    scales_m = [
        float(owner.static_grid.dr),
        float(owner.static_grid.dz),
        float(owner.pic_grid.dr),
        float(owner.pic_grid.dz),
    ]
    if owner.has_cartesian_field3d:
        scales_m.extend(
            [owner.field3d_dx_m, owner.field3d_dy_m, owner.field3d_dz_m]
        )
    if owner.has_electrode_mask:
        scales_m.extend([owner.mask_dr_m, owner.mask_dz_m])
    if owner.has_electrode_mask3d:
        scales_m.extend(
            [owner.mask3d_dx_m, owner.mask3d_dy_m, owner.mask3d_dz_m]
        )
    owner.advection_cell_size_m = min(
        value for value in scales_m if np.isfinite(value) and value > 0.0
    )
