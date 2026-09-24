"""Data-layer field-artifact loading and staged-field helpers.

This module owns file resolution, baked-grid validation, field-buffer loading,
and lightweight field metadata. Particle orchestration remains in
``simulation.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..artifacts.schema import validate_baked_grid_metadata
from ...config import PROJECT_ROOT, SimulationConfig


def resolve_static_field_path(field_path: Optional[Path]) -> Optional[Path]:
    """Resolve a baked field path, keeping repo-relative defaults robust."""

    if field_path is None:
        return None

    path = Path(field_path)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class FieldStage:
    """One external-field stage in a staged trap run."""

    name: str
    static_field_path: Path
    start_s: float
    end_s: float
    duration_s: float
    source_enabled: bool


def _resolve_schedule_relative_path(raw_path: Any, *, schedule_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    for candidate in (
        (schedule_dir / path).resolve(),
        (Path.cwd() / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
    ):
        if candidate.exists():
            return candidate
    return (schedule_dir / path).resolve()


def _stage_duration_s(raw_stage: dict[str, Any]) -> float:
    for key, scale in (("duration_s", 1.0), ("duration_ms", 1.0e-3), ("duration_us", 1.0e-6)):
        if key in raw_stage:
            duration_s = float(raw_stage[key]) * scale
            if duration_s <= 0.0 or not np.isfinite(duration_s):
                raise ValueError(
                    f"Stage duration must be a finite positive value, got {key}={raw_stage[key]!r}."
                )
            return duration_s
    raise ValueError("Each stage must define duration_s, duration_ms, or duration_us.")


def load_field_stage_schedule(schedule_path: Optional[Path]) -> list[FieldStage]:
    if schedule_path is None:
        return []
    resolved_schedule_path = resolve_static_field_path(schedule_path)
    if resolved_schedule_path is None or not resolved_schedule_path.exists():
        raise FileNotFoundError(f"Stage schedule JSON not found: {schedule_path}")
    with resolved_schedule_path.open("r", encoding="utf-8") as handle:
        raw_schedule = json.load(handle)
    raw_stages = raw_schedule.get("stages") if isinstance(raw_schedule, dict) else raw_schedule
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("Stage schedule must be a JSON object with a non-empty 'stages' list.")

    stages: list[FieldStage] = []
    elapsed_s = 0.0
    schedule_dir = resolved_schedule_path.parent
    for index, raw_stage in enumerate(raw_stages):
        if not isinstance(raw_stage, dict):
            raise ValueError(f"Stage #{index + 1} must be a JSON object.")
        field_value = raw_stage.get("static_field_path", raw_stage.get("static_field"))
        if not field_value:
            raise ValueError(f"Stage #{index + 1} is missing static_field_path.")
        static_field_path = _resolve_schedule_relative_path(field_value, schedule_dir=schedule_dir)
        if not static_field_path.exists():
            raise FileNotFoundError(f"Stage static field does not exist: {static_field_path}")
        duration_s = _stage_duration_s(raw_stage)
        name = str(raw_stage.get("name", f"stage_{index + 1}")).strip() or f"stage_{index + 1}"
        source_enabled = bool(raw_stage.get("source_enabled", index == 0))
        start_s = elapsed_s
        end_s = start_s + duration_s
        stages.append(
            FieldStage(
                name=name,
                static_field_path=static_field_path,
                start_s=start_s,
                end_s=end_s,
                duration_s=duration_s,
                source_enabled=source_enabled,
            )
        )
        elapsed_s = end_s
    return stages


def stage_schedule_summary(schedule_path: Optional[Path], stages: list[FieldStage]) -> dict[str, Any]:
    if not stages:
        return {"enabled": False}
    return {
        "enabled": True,
        "path": None if schedule_path is None else str(schedule_path),
        "total_time_s": float(stages[-1].end_s),
        "stages": [
            {
                "name": stage.name,
                "static_field_path": str(stage.static_field_path),
                "start_s": float(stage.start_s),
                "end_s": float(stage.end_s),
                "duration_s": float(stage.duration_s),
                "source_enabled": bool(stage.source_enabled),
            }
            for stage in stages
        ],
    }


def load_baked_field_file(field_path: Path) -> dict[str, Any]:
    """Load and minimally validate a ``field_baker.py`` exported npy file."""

    if not field_path.exists():
        raise FileNotFoundError(
            f"Static field file not found: {field_path}. "
            "Run src/field_baker.py first or pass --dummy-static-field for the analytic demo field."
        )

    baked = np.load(field_path, allow_pickle=True).item()
    if not isinstance(baked, dict):
        raise ValueError(f"Static field file must contain a dict exported by field_baker.py: {field_path}")

    for required_group in ("grid", "simion", "fluent"):
        if required_group not in baked or not isinstance(baked[required_group], dict):
            raise ValueError(f"Static field file is missing the '{required_group}' group: {field_path}")

    return baked


def config_with_baked_grid(config: SimulationConfig, baked_fields: dict[str, Any]) -> SimulationConfig:
    """Return a runtime config whose mesh extents match the baked field file."""

    grid_metadata = baked_fields["grid"]
    validated = validate_baked_grid_metadata(
        grid_metadata,
        source="runtime baked field",
        minimum_nodes=3,
    )
    if abs(validated.r_min_m) > 1.0e-15:
        raise ValueError(
            "The flowchart PIC runtime expects r_min_m = 0.0, "
            f"got {validated.r_min_m}."
        )
    if abs(validated.z_min_m) > 1.0e-15:
        raise ValueError(
            "The flowchart PIC runtime expects z_min_m = 0.0, "
            f"got {validated.z_min_m}."
        )

    capillary_exit_z_m = config.capillary_exit_z_m
    if capillary_exit_z_m is None and "capillary_exit_z_m" in grid_metadata:
        capillary_exit_z_m = float(grid_metadata["capillary_exit_z_m"])

    return replace(
        config,
        domain_radius_m=validated.r_max_m,
        domain_length_m=validated.z_max_m,
        grid_nr=validated.nr,
        grid_nz=validated.nz,
        capillary_exit_z_m=capillary_exit_z_m,
    )


def require_baked_array(
    baked_fields: dict[str, Any],
    group_name: str,
    field_name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Fetch a finite 2D field array from the baked field dictionary."""

    try:
        raw_array = baked_fields[group_name][field_name]
    except KeyError as exc:
        raise ValueError(f"Baked field file is missing '{group_name}.{field_name}'.") from exc

    array = np.ascontiguousarray(np.asarray(raw_array, dtype=np.float64))
    if array.shape != expected_shape:
        raise ValueError(
            f"Baked field '{group_name}.{field_name}' has shape {array.shape}, "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Baked field '{group_name}.{field_name}' contains non-finite values.")
    return array


def optional_baked_array(
    baked_fields: dict[str, Any],
    group_name: str,
    field_name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Fetch an optional baked array, returning zeros when older files omit it."""

    if field_name not in baked_fields[group_name]:
        return np.zeros(expected_shape, dtype=np.float64)
    return require_baked_array(baked_fields, group_name, field_name, expected_shape)


def load_baked_static_fields_into_grid(grid: Any, baked_fields: dict[str, Any]) -> None:
    """Copy baked DC/RF electric and gas fields into the Taichi grid buffers."""

    expected_shape = (grid.nr, grid.nz)

    # Runtime force/gas gathering does not consume the baked potentials. A
    # decoupled static-only grid therefore skips them, while unified/shared
    # compatibility grids retain the historical potential buffers.
    if grid.has_potential_fields:
        grid.phi_dc.from_numpy(require_baked_array(baked_fields, "simion", "phi_dc_v", expected_shape))
    grid.E_dc_r.from_numpy(require_baked_array(baked_fields, "simion", "e_dc_r_v_per_m", expected_shape))
    grid.E_dc_z.from_numpy(require_baked_array(baked_fields, "simion", "e_dc_z_v_per_m", expected_shape))
    if grid.has_potential_fields:
        grid.phi_rf.from_numpy(optional_baked_array(baked_fields, "simion", "phi_rf_v", expected_shape))
    grid.E_rf_r.from_numpy(optional_baked_array(baked_fields, "simion", "e_rf_r_v_per_m", expected_shape))
    grid.E_rf_z.from_numpy(optional_baked_array(baked_fields, "simion", "e_rf_z_v_per_m", expected_shape))
    grid.P_gas.from_numpy(require_baked_array(baked_fields, "fluent", "pressure_pa", expected_shape))
    grid.T_gas.from_numpy(require_baked_array(baked_fields, "fluent", "temperature_k", expected_shape))
    grid.v_gas_r.from_numpy(require_baked_array(baked_fields, "fluent", "v_r_m_per_s", expected_shape))
    grid.v_gas_z.from_numpy(require_baked_array(baked_fields, "fluent", "v_z_m_per_s", expected_shape))
    if grid.has_pic_fields:
        grid.clear_charge()
        grid.clear_space_charge_solution()


def validate_baked_grid_matches_runtime(grid: Any, baked_fields: dict[str, Any], field_path: Path) -> None:
    validated = validate_baked_grid_metadata(
        baked_fields["grid"],
        source=str(field_path),
        minimum_nodes=3,
    )
    expected = {
        "nr": int(grid.nr),
        "nz": int(grid.nz),
        "r_max_m": float(grid.r_max_m),
        "z_max_m": float(grid.z_max_m),
    }
    actual = {
        "nr": validated.nr,
        "nz": validated.nz,
        "r_max_m": validated.r_max_m,
        "z_max_m": validated.z_max_m,
    }
    if actual["nr"] != expected["nr"] or actual["nz"] != expected["nz"]:
        raise ValueError(
            f"Stage field grid shape mismatch for {field_path}: "
            f"got {actual['nr']}x{actual['nz']}, expected {expected['nr']}x{expected['nz']}."
        )
    if not np.isclose(actual["r_max_m"], expected["r_max_m"], rtol=0.0, atol=1.0e-15):
        raise ValueError(
            f"Stage field r_max_m mismatch for {field_path}: "
            f"got {actual['r_max_m']}, expected {expected['r_max_m']}."
        )
    if not np.isclose(actual["z_max_m"], expected["z_max_m"], rtol=0.0, atol=1.0e-15):
        raise ValueError(
            f"Stage field z_max_m mismatch for {field_path}: "
            f"got {actual['z_max_m']}, expected {expected['z_max_m']}."
        )


def load_baked_dc_fields_into_grid(grid: Any, baked_fields: dict[str, Any]) -> None:
    """Copy only baked DC potential and field arrays into an existing runtime grid."""

    expected_shape = (grid.nr, grid.nz)
    if grid.has_potential_fields:
        grid.phi_dc.from_numpy(require_baked_array(baked_fields, "simion", "phi_dc_v", expected_shape))
    grid.E_dc_r.from_numpy(require_baked_array(baked_fields, "simion", "e_dc_r_v_per_m", expected_shape))
    grid.E_dc_z.from_numpy(require_baked_array(baked_fields, "simion", "e_dc_z_v_per_m", expected_shape))


def baked_field_message(field_path: Path, baked_fields: dict[str, Any]) -> str:
    grid_metadata = baked_fields["grid"]
    return (
        "Loaded baked static DC/RF/gas fields from "
        f"{field_path} "
        f"(grid={int(grid_metadata['nr'])}x{int(grid_metadata['nz'])}, "
        f"r_max={float(grid_metadata['r_max_m']) * 1.0e3:.3f} mm, "
        f"z_max={float(grid_metadata['z_max_m']) * 1.0e3:.3f} mm)."
    )


def baked_metadata_summary(baked_fields: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return only lightweight metadata from a baked field dictionary."""

    if baked_fields is None:
        return {"loaded": False}

    summary: dict[str, Any] = {"loaded": True}
    for group_name in ("grid", "simion", "fluent"):
        group = baked_fields.get(group_name, {})
        if not isinstance(group, dict):
            continue
        summary[group_name] = {
            key: value
            for key, value in group.items()
            if not isinstance(value, np.ndarray)
        }
    return summary


def infer_rf_reference_peak_voltage_v(baked_fields: Optional[dict[str, Any]]) -> float:
    """Infer the peak voltage represented by the baked RF arrays."""

    if baked_fields is None:
        return 1.0

    simion_fields = baked_fields.get("simion", {})
    metadata_value = simion_fields.get("rf_reference_peak_voltage_v")
    if metadata_value is None:
        # Backward compatibility for baked files written before the Vpeak-only
        # convention was fixed. New files should use rf_reference_peak_voltage_v.
        metadata_value = simion_fields.get("rf_voltage_scale")
    if metadata_value is not None:
        reference_peak_v = abs(float(metadata_value))
        if np.isfinite(reference_peak_v) and reference_peak_v > 0.0:
            return reference_peak_v

    phi_rf_v = simion_fields.get("phi_rf_v")
    if phi_rf_v is not None:
        reference_peak_v = float(np.nanmax(np.abs(np.asarray(phi_rf_v, dtype=float))))
        if np.isfinite(reference_peak_v) and reference_peak_v > 0.0:
            return reference_peak_v

    return 1.0


def has_nonzero_baked_rf_field(baked_fields: Optional[dict[str, Any]]) -> bool:
    """Return true when a baked file contains a usable RF electric field."""

    if baked_fields is None:
        return False

    simion_fields = baked_fields.get("simion", {})
    for field_name in ("e_rf_r_v_per_m", "e_rf_z_v_per_m"):
        field_data = simion_fields.get(field_name)
        if field_data is None:
            continue
        field_array = np.asarray(field_data, dtype=float)
        if field_array.size > 0 and np.nanmax(np.abs(field_array)) > 0.0:
            return True
    return False


def cartesian_field3d_has_rf(field3d: Optional[Any]) -> bool:
    return bool(field3d is not None and field3d.has_rf)


def infer_cartesian_field3d_reference_peak_voltage_v(
    field3d: Optional[Any],
) -> Optional[float]:
    if field3d is None or not field3d.has_rf:
        return None
    reference_peak_v = abs(float(field3d.rf_reference_peak_voltage_v))
    if np.isfinite(reference_peak_v) and reference_peak_v > 0.0:
        return reference_peak_v
    return None


def rf_convention_message(
    *,
    enabled: bool,
    peak_v: float,
    reference_peak_v: float,
    frequency_hz: float,
    phase_rad: float,
    peak_to_reference_scale: float,
) -> str:
    if not enabled:
        return "RF modulation disabled."

    return (
        "RF convention: Vpeak is the runtime input; "
        "E_rf(t)=E_rf_ref*(Vpeak/Vref)*cos(2*pi*f*t+phase). "
        f"Vpeak={peak_v:.6g} V, "
        f"Vref={reference_peak_v:.6g} V, f={frequency_hz:.6g} Hz, "
        f"phase={phase_rad:.6g} rad, scale={peak_to_reference_scale:.6g}."
    )


def cartesian_field3d_message(
    field_path: Optional[Path],
    field3d: Optional[Any],
) -> str:
    if field_path is None or field3d is None:
        return "No Cartesian 3D electric overlay loaded."
    summary = field3d.metadata_summary()
    return (
        "Loaded Cartesian 3D electric overlay from "
        f"{field_path} "
        f"(grid={summary['nx']}x{summary['ny']}x{summary['nz']}, "
        f"x=[{summary['x_min_m'] * 1.0e3:.3f}, {summary['x_max_m'] * 1.0e3:.3f}] mm, "
        f"y=[{summary['y_min_m'] * 1.0e3:.3f}, {summary['y_max_m'] * 1.0e3:.3f}] mm, "
        f"z=[{summary['z_min_m'] * 1.0e3:.3f}, {summary['z_max_m'] * 1.0e3:.3f}] mm)."
    )


# Temporary compatibility names for tests and internal callers migrated from
# simulation.py. New code should import the public names above.
_resolve_static_field_path = resolve_static_field_path
_load_field_stage_schedule = load_field_stage_schedule
_stage_schedule_summary = stage_schedule_summary
_load_baked_field_file = load_baked_field_file
_config_with_baked_grid = config_with_baked_grid
_require_baked_array = require_baked_array
_optional_baked_array = optional_baked_array
_load_baked_static_fields_into_grid = load_baked_static_fields_into_grid
_validate_baked_grid_matches_runtime = validate_baked_grid_matches_runtime
_load_baked_dc_fields_into_grid = load_baked_dc_fields_into_grid
_baked_field_message = baked_field_message
_baked_metadata_summary = baked_metadata_summary
_infer_rf_reference_peak_voltage_v = infer_rf_reference_peak_voltage_v
_has_nonzero_baked_rf_field = has_nonzero_baked_rf_field
_cartesian_field3d_has_rf = cartesian_field3d_has_rf
_infer_cartesian_field3d_reference_peak_voltage_v = infer_cartesian_field3d_reference_peak_voltage_v
_rf_convention_message = rf_convention_message
_cartesian_field3d_message = cartesian_field3d_message
