"""Pure-electric-field trajectory benchmark against SIMION-style ion beams.

This entry point is intentionally independent from the main PIC/collision
simulation. It reads baked DC/RF fields, disables gas/collisions/IonSPA/space
charge, initializes a SIMION-like beam, and advances particles with RK4 under

    E(t) = E_dc + E_rf_basis * (Vpeak / Vref) * cos(2*pi*f*t + phase).

The coordinate convention follows the current project:
SIMION x -> global z, SIMION y -> global r. A SIMION +x launch is therefore a
Python +z launch.

If an electrode mask is available, PA grid indices follow the same mapping:
xi -> global z, yi -> global r. Electrode hit is classified from metal_mask,
while surface_distance is retained as a diagnostic quantity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from scipy.ndimage import distance_transform_edt

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ...config import PROJECT_ROOT
from ..artifacts.schema import read_patxt_mask_grid, validate_baked_grid_metadata


ELEMENTARY_CHARGE_C = 1.602176634e-19
AMU_KG = 1.66053906660e-27
DEFAULT_FIELD_PATH = Path("outputs") / "slens100x_bake_0p01_zmax65" / "baked_fields.npy"
DEFAULT_GEM_MASK_PATXT_PATH = Path("E_field") / "slens100x_gem.patxt"
DEFAULT_GEM_MASK_PATXT_PATHS = (
    DEFAULT_GEM_MASK_PATXT_PATH,
    Path("E_field") / "slens" / "slens100x_gem.patxt",
)
ACTIVE = 0
TRANSMITTED = 1
LOST_RADIAL = 2
LOST_UPSTREAM = 3
MAX_TIME = 4
ELECTRODE_HIT = 5
STATUS_NAMES = {
    ACTIVE: "active",
    TRANSMITTED: "transmitted",
    LOST_RADIAL: "lost_radial",
    LOST_UPSTREAM: "lost_upstream",
    MAX_TIME: "max_time",
    ELECTRODE_HIT: "electrode_hit",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    field_path: Path
    output_dir: Path
    n_ions: int = 1000
    mass_amu: float = 2000.0
    charge_state: int = 5
    kinetic_energy_ev: float = 10.0
    source_radius_m: float = 0.5e-3
    initial_z_m: float = 5.0e-3
    cone_half_angle_rad: float = math.radians(3.0)
    rf_peak_voltage_v: float = 50.0
    rf_frequency_hz: float = 6.5e5
    rf_phase_rad: float = 0.0
    dc_slens_offset_v: float = 10.0
    dc_exit_lens_v: float = 11.0
    detector_z_m: float = 70.0e-3
    dt_s: float = 2.0e-9
    t_max_s: float = 1.5e-4
    seed: int = 7
    sample_trajectories: int = 25
    record_every_steps: int = 25
    electrode_mask_path: Optional[Path] = None
    electrode_mask_cache_path: Optional[Path] = None
    electrode_mask_z_offset_m: Optional[float] = None
    initial_rays_simion_csv: Optional[Path] = None
    use_electrode_mask: bool = True


@dataclass(frozen=True)
class BakedField:
    r_coords_m: np.ndarray
    z_coords_m: np.ndarray
    e_dc_r_v_per_m: np.ndarray
    e_dc_z_v_per_m: np.ndarray
    e_rf_r_v_per_m: np.ndarray
    e_rf_z_v_per_m: np.ndarray
    rf_reference_peak_voltage_v: float
    metadata: Dict[str, object]


@dataclass(frozen=True)
class ElectrodeMask:
    r_coords_m: np.ndarray
    z_coords_m: np.ndarray
    metal_mask: np.ndarray
    electrode_id: np.ndarray
    surface_distance_m: np.ndarray
    metadata: Dict[str, object]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def load_baked_field(path: Path) -> BakedField:
    baked = np.load(path, allow_pickle=True).item()
    if not isinstance(baked, dict):
        raise TypeError(f"{path} does not contain a baked field dictionary.")

    grid = baked["grid"]
    validated_grid = validate_baked_grid_metadata(
        grid,
        source=str(path),
        minimum_nodes=2,
    )
    simion = baked["simion"]
    vref = float(simion.get("rf_reference_peak_voltage_v", 1.0))
    if not np.isfinite(vref) or vref <= 0.0:
        raise ValueError(f"Invalid RF reference peak voltage: {vref}")

    return BakedField(
        r_coords_m=validated_grid.r_coords_m,
        z_coords_m=validated_grid.z_coords_m,
        e_dc_r_v_per_m=np.asarray(simion["e_dc_r_v_per_m"], dtype=float),
        e_dc_z_v_per_m=np.asarray(simion["e_dc_z_v_per_m"], dtype=float),
        e_rf_r_v_per_m=np.asarray(simion["e_rf_r_v_per_m"], dtype=float),
        e_rf_z_v_per_m=np.asarray(simion["e_rf_z_v_per_m"], dtype=float),
        rf_reference_peak_voltage_v=vref,
        metadata={
            "grid_nr": int(grid["nr"]),
            "grid_nz": int(grid["nz"]),
            "r_max_m": float(grid["r_max_m"]),
            "z_max_m": float(grid["z_max_m"]),
            "dr_m": float(grid["dr_m"]),
            "dz_m": float(grid["dz_m"]),
            "rf_normalization": simion.get("rf_normalization", ""),
            "rf_reference_peak_voltage_v": vref,
            "dc_z_offset_m": float(simion.get("dc_z_offset_m", 0.0)),
            "rf_z_offset_m": float(simion.get("rf_z_offset_m", 0.0)),
            "dc_derivative_method": simion.get("dc_electric_field_derivative_method", ""),
            "rf_derivative_method": simion.get("rf_electric_field_derivative_method", ""),
        },
    )


def _resolve_default_electrode_mask_path() -> Optional[Path]:
    candidates = [
        *(PROJECT_ROOT / relative_path for relative_path in DEFAULT_GEM_MASK_PATXT_PATHS),
        *DEFAULT_GEM_MASK_PATXT_PATHS,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _surface_distance_from_mask(metal_mask: np.ndarray, dr_m: float, dz_m: float) -> np.ndarray:
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


def _finalize_electrode_mask(
    *,
    r_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    metal_mask: np.ndarray,
    electrode_id: np.ndarray,
    surface_distance_m: Optional[np.ndarray],
    metadata: Dict[str, object],
) -> ElectrodeMask:
    if r_coords_m.ndim != 1 or z_coords_m.ndim != 1:
        raise ValueError("Electrode mask coordinates must be 1D arrays.")
    if metal_mask.shape != (len(r_coords_m), len(z_coords_m)):
        raise ValueError(
            "Electrode metal_mask shape does not match coordinates: "
            f"shape={metal_mask.shape}, expected={(len(r_coords_m), len(z_coords_m))}"
        )
    if electrode_id.shape != metal_mask.shape:
        raise ValueError("Electrode id array shape must match metal_mask.")
    if surface_distance_m is None:
        dr_m = float(np.median(np.diff(r_coords_m)))
        dz_m = float(np.median(np.diff(z_coords_m)))
        surface_distance_m = _surface_distance_from_mask(metal_mask, dr_m, dz_m)
        metadata["surface_distance_source"] = "python_edt_from_metal_mask"
    elif surface_distance_m.shape != metal_mask.shape:
        raise ValueError("Surface distance array shape must match metal_mask.")

    metadata.update(
        {
            "loaded": True,
            "shape": [int(metal_mask.shape[0]), int(metal_mask.shape[1])],
            "r_min_m": float(r_coords_m[0]),
            "r_max_m": float(r_coords_m[-1]),
            "z_min_m": float(z_coords_m[0]),
            "z_max_m": float(z_coords_m[-1]),
            "dr_m": float(np.median(np.diff(r_coords_m))),
            "dz_m": float(np.median(np.diff(z_coords_m))),
            "metal_point_count": int(np.count_nonzero(metal_mask)),
            "metal_fraction": float(np.mean(metal_mask)),
            "electrode_ids_present": sorted(
                int(value) for value in np.unique(electrode_id[metal_mask]) if int(value) >= 0
            ),
        }
    )
    return ElectrodeMask(
        r_coords_m=np.asarray(r_coords_m, dtype=float),
        z_coords_m=np.asarray(z_coords_m, dtype=float),
        metal_mask=np.asarray(metal_mask, dtype=bool),
        electrode_id=np.asarray(electrode_id, dtype=np.int32),
        surface_distance_m=np.asarray(surface_distance_m, dtype=np.float32),
        metadata=metadata,
    )


def _load_electrode_mask_npz(path: Path) -> ElectrodeMask:
    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        metadata["source_format"] = metadata.get("source_format", "npz")
        metadata["cache_path"] = str(path)
        return _finalize_electrode_mask(
            r_coords_m=np.asarray(data["r_coords_m"], dtype=float),
            z_coords_m=np.asarray(data["z_coords_m"], dtype=float),
            metal_mask=np.asarray(data["metal_mask"], dtype=bool),
            electrode_id=np.asarray(data["electrode_id"], dtype=np.int32),
            surface_distance_m=np.asarray(data["surface_distance_m"], dtype=np.float32),
            metadata=metadata,
        )


def _save_electrode_mask_npz(path: Path, mask: ElectrodeMask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        r_coords_m=mask.r_coords_m,
        z_coords_m=mask.z_coords_m,
        metal_mask=mask.metal_mask.astype(np.uint8),
        electrode_id=mask.electrode_id.astype(np.int32),
        surface_distance_m=mask.surface_distance_m.astype(np.float32),
        metadata_json=json.dumps(_json_ready(mask.metadata), sort_keys=True),
    )


def _load_electrode_mask_patxt(path: Path, z_offset_m: float = 0.0, z_offset_source: str = "cli_or_default") -> ElectrodeMask:
    parsed = read_patxt_mask_grid(Path(path))
    header = parsed.header
    nx = header.nx
    ny = header.ny
    grids_per_mm = header.grids_per_mm
    pitch_m = 1.0e-3 / grids_per_mm
    z_coords_m = z_offset_m + np.arange(nx, dtype=float) * pitch_m
    r_coords_m = np.arange(ny, dtype=float) * pitch_m

    return _finalize_electrode_mask(
        r_coords_m=r_coords_m,
        z_coords_m=z_coords_m,
        metal_mask=parsed.metal_mask,
        electrode_id=parsed.electrode_id,
        surface_distance_m=None,
        metadata={
            "source_path": str(path),
            "source_format": "pa_text_mask",
            "coordinate_mapping": "PA text x index -> global z, y index -> global r",
            "z_offset_m": float(z_offset_m),
            "z_offset_source": z_offset_source,
            "pa_header_ng": grids_per_mm,
            "pa_header_nx": nx,
            "pa_header_ny": ny,
            "pa_header_nz": header.nz,
            "pa_header_fast_adjustable": header.fast_adjustable,
            "pa_point_count": parsed.point_count,
            "pa_expected_point_count": nx * ny,
            "pa_integrity_validated": True,
            "electrode_id_source": "rounded electrode potential in PA text electrode points",
        },
    )


def _default_mask_z_offset_m(field: BakedField) -> float:
    dc_offset = float(field.metadata.get("dc_z_offset_m", 0.0))
    rf_offset = float(field.metadata.get("rf_z_offset_m", 0.0))
    if abs(dc_offset - rf_offset) > 1.0e-12:
        raise ValueError(
            "Cannot infer electrode mask z offset because baked DC/RF offsets differ: "
            f"dc_z_offset_m={dc_offset}, rf_z_offset_m={rf_offset}"
        )
    return dc_offset


def load_electrode_mask(config: BenchmarkConfig, field: BakedField) -> Optional[ElectrodeMask]:
    if not config.use_electrode_mask:
        return None

    source_path = config.electrode_mask_path or _resolve_default_electrode_mask_path()
    if source_path is None:
        return None
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Electrode mask path does not exist: {source_path}")

    requested_z_offset_m = (
        _default_mask_z_offset_m(field)
        if config.electrode_mask_z_offset_m is None
        else float(config.electrode_mask_z_offset_m)
    )
    z_offset_source = "baked_field_metadata" if config.electrode_mask_z_offset_m is None else "cli"
    suffix = source_path.suffix.lower()
    cache_path = config.electrode_mask_cache_path
    if cache_path is not None and cache_path.exists():
        if cache_path.stat().st_mtime >= source_path.stat().st_mtime:
            cache = _load_electrode_mask_npz(cache_path)
            cached_z_offset_m = float(cache.metadata.get("z_offset_m", 0.0))
            cache_offset_matches = suffix != ".patxt" or abs(cached_z_offset_m - requested_z_offset_m) <= 1.0e-12
            if cache_offset_matches:
                cache.metadata["source_path"] = str(source_path)
                cache.metadata["loaded_from_cache"] = True
                return cache

    if suffix == ".npz":
        mask = _load_electrode_mask_npz(source_path)
    elif suffix == ".patxt":
        mask = _load_electrode_mask_patxt(
            source_path,
            z_offset_m=requested_z_offset_m,
            z_offset_source=z_offset_source,
        )
    else:
        raise ValueError(
            f"Unsupported electrode mask format {source_path.suffix}; use .patxt or .npz."
        )

    if cache_path is not None:
        _save_electrode_mask_npz(cache_path, mask)
        mask.metadata["cache_path"] = str(cache_path)
        mask.metadata["loaded_from_cache"] = False
    return mask


def bilinear_sample(values: np.ndarray, r_coords_m: np.ndarray, z_coords_m: np.ndarray, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
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
    fill_value,
) -> Tuple[np.ndarray, np.ndarray]:
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


def sample_electrode_mask(mask: ElectrodeMask, positions_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_m = np.hypot(positions_m[:, 0], positions_m[:, 1])
    z_m = positions_m[:, 2]
    metal, inside = nearest_sample(mask.metal_mask, mask.r_coords_m, mask.z_coords_m, r_m, z_m, False)
    electrode_id, _ = nearest_sample(mask.electrode_id, mask.r_coords_m, mask.z_coords_m, r_m, z_m, -1)
    surface_distance_m = bilinear_sample(mask.surface_distance_m, mask.r_coords_m, mask.z_coords_m, r_m, z_m)
    surface_distance_m = np.where(inside, surface_distance_m, np.nan)
    return metal.astype(bool), electrode_id.astype(np.int32), surface_distance_m, inside


def sample_electric_field(field: BakedField, positions_m: np.ndarray, time_s: float, config: BenchmarkConfig) -> np.ndarray:
    x_m = positions_m[:, 0]
    y_m = positions_m[:, 1]
    z_m = positions_m[:, 2]
    r_m = np.hypot(x_m, y_m)

    e_dc_r = bilinear_sample(field.e_dc_r_v_per_m, field.r_coords_m, field.z_coords_m, r_m, z_m)
    e_dc_z = bilinear_sample(field.e_dc_z_v_per_m, field.r_coords_m, field.z_coords_m, r_m, z_m)
    e_rf_r = bilinear_sample(field.e_rf_r_v_per_m, field.r_coords_m, field.z_coords_m, r_m, z_m)
    e_rf_z = bilinear_sample(field.e_rf_z_v_per_m, field.r_coords_m, field.z_coords_m, r_m, z_m)

    rf_factor = (
        config.rf_peak_voltage_v
        / field.rf_reference_peak_voltage_v
        * np.cos(2.0 * np.pi * config.rf_frequency_hz * time_s + config.rf_phase_rad)
    )
    e_r = e_dc_r + rf_factor * e_rf_r
    e_z = e_dc_z + rf_factor * e_rf_z

    e_xyz = np.zeros_like(positions_m, dtype=float)
    nonzero_r = r_m > 0.0
    e_xyz[nonzero_r, 0] = e_r[nonzero_r] * x_m[nonzero_r] / r_m[nonzero_r]
    e_xyz[nonzero_r, 1] = e_r[nonzero_r] * y_m[nonzero_r] / r_m[nonzero_r]
    e_xyz[:, 2] = e_z
    return e_xyz


SIMION_FLOAT_RE = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _simion_value(record: str, name: str, unit: str) -> float:
    match = re.search(rf"{re.escape(name)}\({SIMION_FLOAT_RE} {re.escape(unit)}\)", record)
    if not match:
        raise ValueError(f"SIMION record is missing {name}({unit}): {record[:160]}")
    return float(match.group(1))


def load_simion_initial_rays(path: Path, expected_count: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    starts = [match.start() for match in re.finditer(r"Begin Fly'm", text)]
    block = text[starts[-1] :] if starts else text

    created: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for chunk in re.split(r"\n\s*\n", block):
        if "Ion(" not in chunk or "Event(Ion Created)" not in chunk:
            continue
        record = " ".join(chunk.split())
        ion_match = re.search(r"Ion\((\d+)\)", record)
        if ion_match is None:
            continue
        ion_id = int(ion_match.group(1))
        x_mm = _simion_value(record, "X", "mm")
        y_mm = _simion_value(record, "Y", "mm")
        z_mm = _simion_value(record, "Z", "mm")
        vx_mm_per_us = _simion_value(record, "Vx", "mm/usec")
        vy_mm_per_us = _simion_value(record, "Vy", "mm/usec")
        vz_mm_per_us = _simion_value(record, "Vz", "mm/usec")

        # SIMION x is the current project axial coordinate; SIMION y/z are transverse.
        position_m = np.array([y_mm, z_mm, x_mm], dtype=float) * 1.0e-3
        velocity_m_per_s = np.array([vy_mm_per_us, vz_mm_per_us, vx_mm_per_us], dtype=float) * 1.0e3
        created[ion_id] = (position_m, velocity_m_per_s)

    if len(created) != expected_count:
        raise ValueError(
            f"Expected {expected_count} SIMION initial rays in {path}, found {len(created)}."
        )
    expected_ids = list(range(1, expected_count + 1))
    missing = [ion_id for ion_id in expected_ids if ion_id not in created]
    if missing:
        raise ValueError(f"SIMION initial rays are missing ion ids: {missing[:10]}")

    positions_m = np.vstack([created[ion_id][0] for ion_id in expected_ids])
    velocities_m_per_s = np.vstack([created[ion_id][1] for ion_id in expected_ids])
    radius_m = np.hypot(positions_m[:, 0], positions_m[:, 1])
    speed_m_per_s = np.linalg.norm(velocities_m_per_s, axis=1)
    polar_angle_rad = np.arctan2(np.hypot(velocities_m_per_s[:, 0], velocities_m_per_s[:, 1]), velocities_m_per_s[:, 2])
    checks = {
        "initial_source": str(path),
        "initial_source_format": "simion_csv_ion_created_records",
        "initial_radius_max_m": float(np.max(radius_m)),
        "initial_radius_mean_m": float(np.mean(radius_m)),
        "initial_z_m": float(np.mean(positions_m[:, 2])),
        "initial_z_min_m": float(np.min(positions_m[:, 2])),
        "initial_z_max_m": float(np.max(positions_m[:, 2])),
        "initial_polar_angle_max_deg": float(np.degrees(np.max(polar_angle_rad))),
        "initial_polar_angle_mean_deg": float(np.degrees(np.mean(polar_angle_rad))),
        "initial_vz_min_m_per_s": float(np.min(velocities_m_per_s[:, 2])),
        "initial_speed_m_per_s": float(np.mean(speed_m_per_s)),
        "initial_kinetic_energy_mean_ev": math.nan,
    }
    return positions_m, velocities_m_per_s, checks


def initialize_beam(config: BenchmarkConfig) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, float]]:
    mass_kg = config.mass_amu * AMU_KG
    charge_c = config.charge_state * ELEMENTARY_CHARGE_C
    if config.initial_rays_simion_csv is not None:
        positions_m, velocities_m_per_s, initial_checks = load_simion_initial_rays(
            config.initial_rays_simion_csv,
            config.n_ions,
        )
        initial_checks["initial_kinetic_energy_mean_ev"] = float(
            0.5 * mass_kg * np.mean(np.sum(velocities_m_per_s**2, axis=1)) / ELEMENTARY_CHARGE_C
        )
        return positions_m, velocities_m_per_s, mass_kg, charge_c, initial_checks

    rng = np.random.default_rng(config.seed)
    radius_m = config.source_radius_m * np.sqrt(rng.random(config.n_ions))
    azimuth = 2.0 * np.pi * rng.random(config.n_ions)
    positions_m = np.column_stack(
        (
            radius_m * np.cos(azimuth),
            radius_m * np.sin(azimuth),
            np.full(config.n_ions, config.initial_z_m, dtype=float),
        )
    )

    cos_theta = 1.0 - rng.random(config.n_ions) * (1.0 - np.cos(config.cone_half_angle_rad))
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    direction_azimuth = 2.0 * np.pi * rng.random(config.n_ions)
    directions = np.column_stack(
        (
            sin_theta * np.cos(direction_azimuth),
            sin_theta * np.sin(direction_azimuth),
            cos_theta,
        )
    )

    kinetic_energy_j = config.kinetic_energy_ev * ELEMENTARY_CHARGE_C
    speed_m_per_s = np.sqrt(2.0 * kinetic_energy_j / mass_kg)
    velocities_m_per_s = speed_m_per_s * directions

    polar_angle_rad = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    initial_checks = {
        "initial_radius_max_m": float(np.max(np.hypot(positions_m[:, 0], positions_m[:, 1]))),
        "initial_z_m": float(config.initial_z_m),
        "initial_polar_angle_max_deg": float(np.degrees(np.max(polar_angle_rad))),
        "initial_vz_min_m_per_s": float(np.min(velocities_m_per_s[:, 2])),
        "initial_speed_m_per_s": float(speed_m_per_s),
        "initial_kinetic_energy_mean_ev": float(0.5 * mass_kg * np.mean(np.sum(velocities_m_per_s**2, axis=1)) / ELEMENTARY_CHARGE_C),
    }
    return positions_m, velocities_m_per_s, mass_kg, charge_c, initial_checks


def rk4_step(
    field: BakedField,
    positions_m: np.ndarray,
    velocities_m_per_s: np.ndarray,
    time_s: float,
    dt_s: float,
    mass_kg: float,
    charge_c: float,
    config: BenchmarkConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    charge_over_mass = charge_c / mass_kg

    def acceleration(pos: np.ndarray, t: float) -> np.ndarray:
        return charge_over_mass * sample_electric_field(field, pos, t, config)

    k1_x = velocities_m_per_s
    k1_v = acceleration(positions_m, time_s)

    k2_x = velocities_m_per_s + 0.5 * dt_s * k1_v
    k2_v = acceleration(positions_m + 0.5 * dt_s * k1_x, time_s + 0.5 * dt_s)

    k3_x = velocities_m_per_s + 0.5 * dt_s * k2_v
    k3_v = acceleration(positions_m + 0.5 * dt_s * k2_x, time_s + 0.5 * dt_s)

    k4_x = velocities_m_per_s + dt_s * k3_v
    k4_v = acceleration(positions_m + dt_s * k3_x, time_s + dt_s)

    positions_next = positions_m + (dt_s / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    velocities_next = velocities_m_per_s + (dt_s / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return positions_next, velocities_next


def _record_state(
    rows: List[Dict[str, object]],
    particle_ids: np.ndarray,
    time_s: float,
    positions_m: np.ndarray,
    velocities_m_per_s: np.ndarray,
    status: np.ndarray,
    hit_electrode_id: np.ndarray,
    surface_distance_m: np.ndarray,
) -> None:
    r_m = np.hypot(positions_m[:, 0], positions_m[:, 1])
    for pid in particle_ids:
        rows.append(
            {
                "particle_id": int(pid),
                "t_s": float(time_s),
                "x_m": float(positions_m[pid, 0]),
                "y_m": float(positions_m[pid, 1]),
                "z_m": float(positions_m[pid, 2]),
                "r_m": float(r_m[pid]),
                "vx_m_per_s": float(velocities_m_per_s[pid, 0]),
                "vy_m_per_s": float(velocities_m_per_s[pid, 1]),
                "vz_m_per_s": float(velocities_m_per_s[pid, 2]),
                "status": STATUS_NAMES[int(status[pid])],
                "electrode_id": int(hit_electrode_id[pid]),
                "surface_distance_m": float(surface_distance_m[pid]),
            }
        )


def _record_envelope(
    rows: List[Dict[str, object]],
    time_s: float,
    positions_m: np.ndarray,
    status: np.ndarray,
) -> None:
    active = status == ACTIVE
    transmitted = status == TRANSMITTED
    lost_radial = status == LOST_RADIAL
    lost_upstream = status == LOST_UPSTREAM
    max_time = status == MAX_TIME
    electrode_hit = status == ELECTRODE_HIT
    r_m = np.hypot(positions_m[:, 0], positions_m[:, 1])
    z_m = positions_m[:, 2]

    if np.any(active):
        active_r = r_m[active]
        active_z = z_m[active]
        row = {
            "t_s": float(time_s),
            "active": int(np.count_nonzero(active)),
            "transmitted": int(np.count_nonzero(transmitted)),
            "lost_radial": int(np.count_nonzero(lost_radial)),
            "lost_upstream": int(np.count_nonzero(lost_upstream)),
            "electrode_hit": int(np.count_nonzero(electrode_hit)),
            "max_time": int(np.count_nonzero(max_time)),
            "z_mean_m": float(np.mean(active_z)),
            "z_p50_m": float(np.percentile(active_z, 50)),
            "r_mean_m": float(np.mean(active_r)),
            "r_p50_m": float(np.percentile(active_r, 50)),
            "r_p95_m": float(np.percentile(active_r, 95)),
            "r_max_m": float(np.max(active_r)),
        }
    else:
        row = {
            "t_s": float(time_s),
            "active": 0,
            "transmitted": int(np.count_nonzero(transmitted)),
            "lost_radial": int(np.count_nonzero(lost_radial)),
            "lost_upstream": int(np.count_nonzero(lost_upstream)),
            "electrode_hit": int(np.count_nonzero(electrode_hit)),
            "max_time": int(np.count_nonzero(max_time)),
            "z_mean_m": math.nan,
            "z_p50_m": math.nan,
            "r_mean_m": math.nan,
            "r_p50_m": math.nan,
            "r_p95_m": math.nan,
            "r_max_m": math.nan,
        }
    rows.append(row)


def _final_particle_rows(
    positions_m: np.ndarray,
    velocities_m_per_s: np.ndarray,
    status: np.ndarray,
    hit_electrode_id: np.ndarray,
    stop_time_s: np.ndarray,
    mass_kg: float,
) -> List[Dict[str, object]]:
    r_m = np.hypot(positions_m[:, 0], positions_m[:, 1])
    speed_m_per_s = np.linalg.norm(velocities_m_per_s, axis=1)
    kinetic_energy_ev = 0.5 * mass_kg * speed_m_per_s**2 / ELEMENTARY_CHARGE_C
    rows: List[Dict[str, object]] = []
    for pid in range(positions_m.shape[0]):
        rows.append(
            {
                "particle_id": int(pid),
                "stop_time_s": float(stop_time_s[pid]),
                "x_m": float(positions_m[pid, 0]),
                "y_m": float(positions_m[pid, 1]),
                "z_m": float(positions_m[pid, 2]),
                "r_m": float(r_m[pid]),
                "vx_m_per_s": float(velocities_m_per_s[pid, 0]),
                "vy_m_per_s": float(velocities_m_per_s[pid, 1]),
                "vz_m_per_s": float(velocities_m_per_s[pid, 2]),
                "speed_m_per_s": float(speed_m_per_s[pid]),
                "kinetic_energy_ev": float(kinetic_energy_ev[pid]),
                "status": STATUS_NAMES[int(status[pid])],
                "electrode_id": int(hit_electrode_id[pid]),
            }
        )
    return rows


def run_benchmark(config: BenchmarkConfig) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    field = load_baked_field(config.field_path)
    electrode_mask = load_electrode_mask(config, field)
    positions_m, velocities_m_per_s, mass_kg, charge_c, initial_checks = initialize_beam(config)
    status = np.full(config.n_ions, ACTIVE, dtype=np.int8)
    hit_electrode_id = np.full(config.n_ions, -1, dtype=np.int32)
    last_surface_distance_m = np.full(config.n_ions, np.nan, dtype=float)
    stop_time_s = np.full(config.n_ions, np.nan, dtype=float)
    representative_count = min(config.sample_trajectories, config.n_ions)
    representative_ids = np.linspace(0, config.n_ions - 1, representative_count, dtype=int)

    rf_period_s = 1.0 / config.rf_frequency_hz if config.rf_frequency_hz > 0.0 else math.inf
    max_speed_seen_m_per_s = float(np.max(np.linalg.norm(velocities_m_per_s, axis=1)))
    trajectory_rows: List[Dict[str, object]] = []
    envelope_rows: List[Dict[str, object]] = []
    time_s = 0.0
    step = 0
    z_min = float(field.z_coords_m[0])
    z_max = float(field.z_coords_m[-1])
    detector_z_m = float(config.detector_z_m)
    if detector_z_m > z_max + 1.0e-15:
        raise ValueError(f"Detector z={detector_z_m} m exceeds baked field z_max={z_max} m.")
    r_max = float(field.r_coords_m[-1])
    max_steps = int(math.ceil(config.t_max_s / config.dt_s))

    if electrode_mask is not None:
        metal, electrode_id, surface_distance_m, _inside = sample_electrode_mask(electrode_mask, positions_m)
        initial_hits = metal & (status == ACTIVE)
        status[initial_hits] = ELECTRODE_HIT
        hit_electrode_id[initial_hits] = electrode_id[initial_hits]
        stop_time_s[initial_hits] = 0.0
        last_surface_distance_m = surface_distance_m

    while step <= max_steps:
        if step % config.record_every_steps == 0 or step == 0:
            _record_state(
                trajectory_rows,
                representative_ids,
                time_s,
                positions_m,
                velocities_m_per_s,
                status,
                hit_electrode_id,
                last_surface_distance_m,
            )
            _record_envelope(envelope_rows, time_s, positions_m, status)

        active_ids = np.flatnonzero(status == ACTIVE)
        if active_ids.size == 0 or time_s >= config.t_max_s:
            break

        dt_s = min(config.dt_s, config.t_max_s - time_s)
        positions_next, velocities_next = rk4_step(
            field,
            positions_m[active_ids],
            velocities_m_per_s[active_ids],
            time_s,
            dt_s,
            mass_kg,
            charge_c,
            config,
        )
        positions_m[active_ids] = positions_next
        velocities_m_per_s[active_ids] = velocities_next
        if velocities_next.size:
            max_speed_seen_m_per_s = max(
                max_speed_seen_m_per_s,
                float(np.max(np.linalg.norm(velocities_next, axis=1))),
            )

        active_positions = positions_m[active_ids]
        active_r = np.hypot(active_positions[:, 0], active_positions[:, 1])
        active_z = active_positions[:, 2]
        electrode_hit = np.zeros(active_ids.shape, dtype=bool)
        if electrode_mask is not None:
            metal, electrode_id, surface_distance_m, _inside = sample_electrode_mask(electrode_mask, active_positions)
            electrode_hit = metal
            hit_ids = active_ids[electrode_hit]
            hit_electrode_id[hit_ids] = electrode_id[electrode_hit]
            last_surface_distance_m[active_ids] = surface_distance_m

        transmitted = (active_z >= detector_z_m) & ~electrode_hit
        lost_radial = (active_r > r_max) & ~electrode_hit & ~transmitted
        lost_upstream = (active_z < z_min) & ~electrode_hit & ~transmitted & ~lost_radial

        status[active_ids[electrode_hit]] = ELECTRODE_HIT
        status[active_ids[transmitted]] = TRANSMITTED
        status[active_ids[lost_radial]] = LOST_RADIAL
        status[active_ids[lost_upstream]] = LOST_UPSTREAM
        stopped = electrode_hit | transmitted | lost_radial | lost_upstream
        stop_time_s[active_ids[stopped]] = time_s + dt_s

        time_s += dt_s
        step += 1

    remaining_active = status == ACTIVE
    status[remaining_active] = MAX_TIME
    stop_time_s[remaining_active] = time_s
    _record_state(
        trajectory_rows,
        representative_ids,
        time_s,
        positions_m,
        velocities_m_per_s,
        status,
        hit_electrode_id,
        last_surface_distance_m,
    )
    _record_envelope(envelope_rows, time_s, positions_m, status)

    r_final_m = np.hypot(positions_m[:, 0], positions_m[:, 1])
    speed_final_m_per_s = np.linalg.norm(velocities_m_per_s, axis=1)
    kinetic_energy_final_ev = 0.5 * mass_kg * speed_final_m_per_s**2 / ELEMENTARY_CHARGE_C
    max_final_speed_times_dt_m = float(np.max(speed_final_m_per_s) * config.dt_s)
    max_seen_speed_times_dt_m = float(max_speed_seen_m_per_s * config.dt_s)
    min_grid_spacing_m = float(min(field.metadata["dr_m"], field.metadata["dz_m"]))
    final_counts = {name: int(np.count_nonzero(status == code)) for code, name in STATUS_NAMES.items()}
    hit_ids = hit_electrode_id[status == ELECTRODE_HIT]
    electrode_hit_counts_by_id = {
        str(int(value)): int(np.count_nonzero(hit_ids == value))
        for value in np.unique(hit_ids)
        if int(value) >= 0
    }
    final_statistics = {
        "transmitted_fraction": float(np.mean(status == TRANSMITTED)),
        "lost_fraction": float(np.mean((status == LOST_RADIAL) | (status == LOST_UPSTREAM) | (status == ELECTRODE_HIT))),
        "electrode_hit_fraction": float(np.mean(status == ELECTRODE_HIT)),
        "max_time_fraction": float(np.mean(status == MAX_TIME)),
        "tof_mean_us": float(np.mean(stop_time_s) * 1.0e6),
        "tof_p50_us": float(np.percentile(stop_time_s, 50) * 1.0e6),
        "tof_p95_us": float(np.percentile(stop_time_s, 95) * 1.0e6),
        "z_mean_m": float(np.mean(positions_m[:, 2])),
        "z_p50_m": float(np.percentile(positions_m[:, 2], 50)),
        "z_p95_m": float(np.percentile(positions_m[:, 2], 95)),
        "r_mean_m": float(np.mean(r_final_m)),
        "r_p50_m": float(np.percentile(r_final_m, 50)),
        "r_p95_m": float(np.percentile(r_final_m, 95)),
        "r_max_m": float(np.max(r_final_m)),
        "kinetic_energy_mean_ev": float(np.mean(kinetic_energy_final_ev)),
        "kinetic_energy_p50_ev": float(np.percentile(kinetic_energy_final_ev, 50)),
        "kinetic_energy_p95_ev": float(np.percentile(kinetic_energy_final_ev, 95)),
    }
    pass_items = {
        "electrode_mask_loaded_when_requested": (not config.use_electrode_mask) or electrode_mask is not None,
        "rf_basis_is_adjacent_plus1_minus1": field.metadata["rf_normalization"] == "adjacent_rf_electrodes_plus_1_v_minus_1_v",
        "rf_reference_is_1v": abs(field.rf_reference_peak_voltage_v - 1.0) <= 1.0e-12,
        "initial_radius_within_0p5mm": initial_checks["initial_radius_max_m"] <= config.source_radius_m * (1.0 + 1.0e-12),
        "initial_cone_within_config_deg": initial_checks["initial_polar_angle_max_deg"] <= math.degrees(config.cone_half_angle_rad) + 1.0e-12,
        "initial_ke_is_10ev": abs(initial_checks["initial_kinetic_energy_mean_ev"] - config.kinetic_energy_ev) <= 1.0e-6,
        "rk4_resolves_rf_period": (config.dt_s / rf_period_s if math.isfinite(rf_period_s) else 0.0) <= 5.0e-3,
        "rk4_step_below_grid_spacing": max_seen_speed_times_dt_m <= min_grid_spacing_m,
        "no_max_time_truncation": final_counts["max_time"] == 0,
        "no_active_particles_remaining": final_counts["active"] == 0,
    }
    pass_items["overall_pass"] = all(pass_items.values())

    summary = {
        "config": _json_ready(asdict(config)),
        "field_metadata": _json_ready(field.metadata),
        "physics": {
            "mass_kg": mass_kg,
            "charge_c": charge_c,
            "coordinate_mapping": "SIMION +x launch maps to Python +z launch",
            "dc_field_condition": {
                "slens_dc_offset_v": config.dc_slens_offset_v,
                "exit_lens_dc_v": config.dc_exit_lens_v,
                "runtime_dc_rescaling": "disabled; baked DC field is used directly",
            },
            "rf_field_equation": "E = E_dc + E_rf_basis*(rf_peak_voltage_v/rf_reference_peak_voltage_v)*cos(2*pi*rf_frequency_hz*t + rf_phase_rad)",
            "detector_z_m": detector_z_m,
            "ion_transfer_capillary_region_m": [0.0, 5.0e-3],
            "gas_flow": "disabled",
            "collisions": "disabled",
            "ion_spa": "disabled",
            "space_charge": "disabled",
            "electrode_collision_geometry": "enabled from electrode mask" if electrode_mask is not None else "not applied; no electrode mask was loaded",
        },
        "electrode_mask_metadata": _json_ready(electrode_mask.metadata if electrode_mask is not None else {"loaded": False}),
        "initial_checks": initial_checks,
        "time_step_checks": {
            "rf_period_s": rf_period_s,
            "dt_over_rf_period": config.dt_s / rf_period_s if math.isfinite(rf_period_s) else 0.0,
            "max_speed_seen_m_per_s": max_speed_seen_m_per_s,
            "max_seen_speed_times_dt_m": max_seen_speed_times_dt_m,
            "max_final_speed_times_dt_m": max_final_speed_times_dt_m,
            "grid_dr_m": field.metadata["dr_m"],
            "grid_dz_m": field.metadata["dz_m"],
        },
        "final_counts": final_counts,
        "electrode_hit_counts_by_id": electrode_hit_counts_by_id,
        "final_statistics": final_statistics,
        "pass_checks": pass_items,
    }
    final_particle_rows = _final_particle_rows(positions_m, velocities_m_per_s, status, hit_electrode_id, stop_time_s, mass_kg)
    return summary, trajectory_rows, envelope_rows, final_particle_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_beam_envelope(path: Path, envelope_rows: List[Dict[str, object]]) -> None:
    z_mean_mm = np.asarray([row["z_mean_m"] for row in envelope_rows], dtype=float) * 1.0e3
    r_p50_mm = np.asarray([row["r_p50_m"] for row in envelope_rows], dtype=float) * 1.0e3
    r_p95_mm = np.asarray([row["r_p95_m"] for row in envelope_rows], dtype=float) * 1.0e3
    r_max_mm = np.asarray([row["r_max_m"] for row in envelope_rows], dtype=float) * 1.0e3
    valid = np.isfinite(z_mean_mm) & np.isfinite(r_p95_mm)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(z_mean_mm[valid], r_p50_mm[valid], label="r p50", linewidth=1.5)
    ax.plot(z_mean_mm[valid], r_p95_mm[valid], label="r p95", linewidth=1.5)
    ax.plot(z_mean_mm[valid], r_max_mm[valid], label="r max", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("mean z [mm]")
    ax.set_ylabel("beam radius [mm]")
    ax.set_title("Beam Envelope")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _trajectory_arrays(rows: List[Dict[str, object]]) -> Dict[int, Dict[str, np.ndarray]]:
    grouped: Dict[int, Dict[str, List[float]]] = {}
    for row in rows:
        pid = int(row["particle_id"])
        bucket = grouped.setdefault(pid, {"t_s": [], "z_m": [], "r_m": []})
        bucket["t_s"].append(float(row["t_s"]))
        bucket["z_m"].append(float(row["z_m"]))
        bucket["r_m"].append(float(row["r_m"]))
    return {
        pid: {key: np.asarray(values, dtype=float) for key, values in bucket.items()}
        for pid, bucket in grouped.items()
    }


def plot_z_of_t(path: Path, trajectory_rows: List[Dict[str, object]]) -> None:
    grouped = _trajectory_arrays(trajectory_rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for pid, series in grouped.items():
        ax.plot(series["t_s"] * 1.0e6, series["z_m"] * 1.0e3, linewidth=0.8, alpha=0.75, label=str(pid))
    ax.set_xlabel("time [us]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Representative z(t)")
    ax.grid(True, alpha=0.25)
    if len(grouped) <= 12:
        ax.legend(title="particle", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_r_of_z(path: Path, trajectory_rows: List[Dict[str, object]]) -> None:
    grouped = _trajectory_arrays(trajectory_rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for pid, series in grouped.items():
        ax.plot(series["z_m"] * 1.0e3, series["r_m"] * 1.0e3, linewidth=0.8, alpha=0.75, label=str(pid))
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title("Representative r(z)")
    ax.grid(True, alpha=0.25)
    if len(grouped) <= 12:
        ax.legend(title="particle", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(
    config: BenchmarkConfig,
    summary: Dict[str, object],
    trajectory_rows: List[Dict[str, object]],
    envelope_rows: List[Dict[str, object]],
    final_particle_rows: List[Dict[str, object]],
) -> Dict[str, str]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"
    trajectory_path = output_dir / "representative_trajectories.csv"
    final_particle_path = output_dir / "final_particles.csv"
    envelope_path = output_dir / "beam_envelope.png"
    z_of_t_path = output_dir / "z_of_t.png"
    r_of_z_path = output_dir / "r_of_z.png"

    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    write_csv(trajectory_path, trajectory_rows)
    write_csv(final_particle_path, final_particle_rows)
    plot_beam_envelope(envelope_path, envelope_rows)
    plot_z_of_t(z_of_t_path, trajectory_rows)
    plot_r_of_z(r_of_z_path, trajectory_rows)

    return {
        "summary": str(summary_path),
        "trajectories": str(trajectory_path),
        "final_particles": str(final_particle_path),
        "beam_envelope": str(envelope_path),
        "z_of_t": str(z_of_t_path),
        "r_of_z": str(r_of_z_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pure-electric-field SIMION-style trajectory benchmark.")
    parser.add_argument("--field", default=str(DEFAULT_FIELD_PATH), help="Path to field_baker.py baked_fields.npy.")
    parser.add_argument("--output-dir", default="outputs/simion_field_benchmark", help="Benchmark output directory.")
    parser.add_argument("--n-ions", type=int, default=1000)
    parser.add_argument("--mass-amu", type=float, default=2000.0)
    parser.add_argument("--charge-state", type=int, default=5)
    parser.add_argument("--kinetic-energy-ev", type=float, default=10.0)
    parser.add_argument("--source-radius-mm", type=float, default=0.5)
    parser.add_argument(
        "--initial-z-mm",
        "--initial-ion-z-mm",
        "--initial-ions-position-z-mm",
        dest="initial_z_mm",
        type=float,
        default=5.0,
        help="Initial axial position in Python/global z [mm]. Use 5 mm for ions exiting the transfer capillary.",
    )
    parser.add_argument("--cone-half-angle-deg", type=float, default=3.0)
    parser.add_argument("--rf-peak-voltage", type=float, default=50.0)
    parser.add_argument("--rf-frequency", type=float, default=6.5e5)
    parser.add_argument("--rf-phase-deg", type=float, default=0.0)
    parser.add_argument("--dc-slens-offset-v", type=float, default=10.0, help="DC condition label only; baked DC field is not rescaled.")
    parser.add_argument("--dc-exit-lens-v", type=float, default=11.0, help="DC condition label only; baked DC field is not rescaled.")
    parser.add_argument("--detector-z-mm", type=float, default=70.0, help="Axial detector/transmission plane in Python/global z [mm].")
    parser.add_argument("--dt", type=float, default=2.0e-9, help="RK4 time step [s].")
    parser.add_argument("--t-max", type=float, default=1.5e-4, help="Maximum integration time [s].")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-trajectories", type=int, default=25)
    parser.add_argument("--record-every", type=int, default=25, help="Record representative trajectories every N steps.")
    parser.add_argument(
        "--electrode-mask",
        default="auto",
        help=(
            "Electrode mask path. Use 'auto' to resolve "
            "E_field/slens/slens100x_gem.patxt or the legacy root layout. "
            "Supported formats: .patxt, .npz."
        ),
    )
    parser.add_argument(
        "--electrode-mask-cache",
        default="",
        help="Optional .npz cache for parsed electrode mask arrays. Defaults to <output-dir>/electrode_mask_cache.npz.",
    )
    parser.add_argument(
        "--electrode-mask-z-offset-mm",
        type=float,
        default=None,
        help=(
            "Axial offset for .patxt electrode mask coordinates [mm]. "
            "Default follows dc_z_offset_m/rf_z_offset_m from the baked field."
        ),
    )
    parser.add_argument(
        "--initial-rays-simion-csv",
        default="",
        help=(
            "Optional SIMION CSV export. When set, Python uses Ion Created records "
            "as particle-by-particle initial positions and velocities."
        ),
    )
    parser.add_argument("--no-electrode-mask", action="store_true", help="Disable electrode hit classification.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    if args.n_ions <= 0:
        raise ValueError("--n-ions must be positive.")
    if args.mass_amu <= 0.0:
        raise ValueError("--mass-amu must be positive.")
    if args.charge_state <= 0:
        raise ValueError("--charge-state must be positive.")
    if args.kinetic_energy_ev <= 0.0:
        raise ValueError("--kinetic-energy-ev must be positive.")
    if args.source_radius_mm < 0.0:
        raise ValueError("--source-radius-mm must be non-negative.")
    if args.initial_z_mm < 0.0:
        raise ValueError("--initial-z-mm must be non-negative.")
    if args.detector_z_mm <= args.initial_z_mm:
        raise ValueError("--detector-z-mm must be greater than --initial-z-mm.")
    if args.cone_half_angle_deg < 0.0 or args.cone_half_angle_deg >= 90.0:
        raise ValueError("--cone-half-angle-deg must be in [0, 90).")
    if args.dt <= 0.0 or args.t_max <= 0.0:
        raise ValueError("--dt and --t-max must be positive.")
    if args.record_every <= 0:
        raise ValueError("--record-every must be positive.")

    electrode_mask_arg = str(args.electrode_mask).strip()
    electrode_mask_path = None if electrode_mask_arg == "" or electrode_mask_arg.lower() == "auto" else Path(electrode_mask_arg)
    electrode_mask_cache_path = (
        Path(args.electrode_mask_cache)
        if str(args.electrode_mask_cache).strip()
        else Path(args.output_dir) / "electrode_mask_cache.npz"
    )

    return BenchmarkConfig(
        field_path=Path(args.field),
        output_dir=Path(args.output_dir),
        n_ions=int(args.n_ions),
        mass_amu=float(args.mass_amu),
        charge_state=int(args.charge_state),
        kinetic_energy_ev=float(args.kinetic_energy_ev),
        source_radius_m=float(args.source_radius_mm) * 1.0e-3,
        initial_z_m=float(args.initial_z_mm) * 1.0e-3,
        cone_half_angle_rad=math.radians(float(args.cone_half_angle_deg)),
        rf_peak_voltage_v=float(args.rf_peak_voltage),
        rf_frequency_hz=float(args.rf_frequency),
        rf_phase_rad=math.radians(float(args.rf_phase_deg)),
        dc_slens_offset_v=float(args.dc_slens_offset_v),
        dc_exit_lens_v=float(args.dc_exit_lens_v),
        detector_z_m=float(args.detector_z_mm) * 1.0e-3,
        dt_s=float(args.dt),
        t_max_s=float(args.t_max),
        seed=int(args.seed),
        sample_trajectories=int(args.sample_trajectories),
        record_every_steps=int(args.record_every),
        electrode_mask_path=electrode_mask_path,
        electrode_mask_cache_path=electrode_mask_cache_path,
        electrode_mask_z_offset_m=(
            None
            if args.electrode_mask_z_offset_mm is None
            else float(args.electrode_mask_z_offset_mm) * 1.0e-3
        ),
        initial_rays_simion_csv=(
            Path(args.initial_rays_simion_csv)
            if str(args.initial_rays_simion_csv).strip()
            else None
        ),
        use_electrode_mask=not bool(args.no_electrode_mask),
    )


def main() -> None:
    config = build_config(parse_args())
    summary, trajectory_rows, envelope_rows, final_particle_rows = run_benchmark(config)
    output_paths = write_outputs(config, summary, trajectory_rows, envelope_rows, final_particle_rows)
    print("Saved SIMION pure-field benchmark outputs:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")
    counts = summary["final_counts"]
    stats = summary["final_statistics"]
    print(
        "Final: "
        f"transmitted={counts['transmitted']}, lost_radial={counts['lost_radial']}, "
        f"lost_upstream={counts['lost_upstream']}, electrode_hit={counts['electrode_hit']}, "
        f"max_time={counts['max_time']}, "
        f"transmitted_fraction={stats['transmitted_fraction']:.6g}"
    )


if __name__ == "__main__":
    main()
