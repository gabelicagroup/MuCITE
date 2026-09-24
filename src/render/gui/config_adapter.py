"""Adapt GUI model values into canonical simulation and ion contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import elementary_charge

from ...config import (
    ATOMIC_MASS_CONSTANT,
    PROJECT_ROOT,
    IonTemplate,
    SimulationConfig,
    make_ion_template,
    make_simulation_config,
)
from .models import AppConfig, BeamConfig, RuntimeConfig


def direction_vector(axis: str) -> np.ndarray:
    axis = str(axis).strip().lower()
    vectors = {
        "+x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "+y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "+z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }
    if axis not in vectors:
        raise ValueError(f"Unsupported direction axis: {axis!r}")
    return vectors[axis]


def build_ion_template(beam: BeamConfig) -> IonTemplate:
    mass_kg = float(beam.mass_amu) * ATOMIC_MASS_CONSTANT
    if mass_kg <= 0.0:
        raise ValueError("Beam mass must be positive.")
    if int(beam.charge_e) <= 0:
        raise ValueError("Charge state must be positive.")
    gas_mode = _gas_velocity_mode(beam) == "from-gas-field"
    if gas_mode or beam.kinetic_energy_ev is None or str(beam.kinetic_energy_ev).strip() == "":
        speed_m_per_s = 250.0
    else:
        kinetic_energy_ev = float(beam.kinetic_energy_ev)
        if kinetic_energy_ev <= 0.0:
            raise ValueError("Beam kinetic energy must be positive when provided.")
        kinetic_energy_j = kinetic_energy_ev * elementary_charge
        speed_m_per_s = np.sqrt(2.0 * kinetic_energy_j / mass_kg)
    return make_ion_template(
        name=str(beam.ion_name).strip() or "protein_ion",
        mass_kg=mass_kg,
        charge_state=int(beam.charge_e),
        collision_cross_section_m2=float(beam.collision_cross_section_m2),
        num_atoms=int(beam.num_atoms),
        heat_capacity_profile=str(beam.heat_capacity_profile),
        delta_h_kj_per_mol=float(beam.delta_h_kj_per_mol),
        delta_s_j_per_mol_k=float(beam.delta_s_j_per_mol_k),
        initial_internal_temperature_k=float(
            beam.initial_internal_temperature_k
        ),
        initial_position_m=np.array(
            [beam.initial_x_mm, beam.initial_y_mm, beam.initial_z_mm],
            dtype=np.float64,
        )
        * 1.0e-3,
        initial_velocity_m_per_s=speed_m_per_s * (
            direction_vector("+z") if gas_mode else direction_vector(beam.direction_axis)
        ),
    )


def _gas_velocity_mode(beam: BeamConfig) -> str:
    mode = str(beam.gas_velocity_init_mode).strip().lower()
    if mode == "off":
        mode = "static"
    if mode not in {"static", "from-gas-field"}:
        raise ValueError(f"Unsupported gas velocity initialization mode: {mode!r}")
    return mode


def _source_geometry(beam: BeamConfig) -> tuple[float | None, float | None]:
    source_radius_m = (
        float(beam.beam_radius_mm) * 1.0e-3
        if float(beam.beam_radius_mm) > 0.0
        else None
    )
    source_gaussian_sigma_m = (
        float(beam.source_gaussian_sigma_mm) * 1.0e-3
        if str(beam.source_profile) == "gaussian"
        and float(beam.source_gaussian_sigma_mm) > 0.0
        else None
    )
    return source_radius_m, source_gaussian_sigma_m


def _field_paths(
    app_config: AppConfig,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    runtime = app_config.runtime
    static_field_path = None
    if (
        not runtime.dummy_static_field
        and str(app_config.loaded_baked_field_path).strip()
    ):
        static_field_path = Path(app_config.loaded_baked_field_path)
    static_field_3d_path = (
        Path(runtime.static_field_3d_path)
        if str(runtime.static_field_3d_path).strip()
        else None
    )
    electrode_mask_path = _electrode_mask_path(runtime)
    electrode_mask_cache_path = (
        Path(runtime.electrode_mask_cache_path)
        if str(runtime.electrode_mask_cache_path).strip()
        else None
    )
    return (
        static_field_path,
        static_field_3d_path,
        electrode_mask_path,
        electrode_mask_cache_path,
    )


def _electrode_mask_path(runtime: RuntimeConfig) -> Path | None:
    electrode_mask_text = str(runtime.electrode_mask_path).strip()
    if not electrode_mask_text:
        return None
    if electrode_mask_text.lower() != "default":
        return Path(electrode_mask_text)
    if runtime.dummy_static_field:
        return None
    from ...data.boundary_masks import resolve_default_electrode_mask_path

    path = resolve_default_electrode_mask_path(PROJECT_ROOT)
    if path is None:
        raise FileNotFoundError(
            "Default electrode mask was requested but no "
            "slens100x_gem.patxt was found."
        )
    return path


def _source_birth_bounds(
    beam: BeamConfig,
) -> tuple[float, float | None]:
    z_min_m = max(float(beam.source_birth_velocity_z_min_mm), 0.0) * 1.0e-3
    z_max_m = (
        None
        if beam.source_birth_velocity_z_max_mm is None
        or str(beam.source_birth_velocity_z_max_mm).strip() == ""
        else max(float(beam.source_birth_velocity_z_max_mm), 0.0) * 1.0e-3
    )
    if z_max_m is not None and z_max_m < z_min_m:
        raise ValueError("Birth gas z max must be greater than or equal to z min.")
    return z_min_m, z_max_m


def _pic_grid_shape(runtime: RuntimeConfig) -> tuple[int, int]:
    pic_grid_nr = int(runtime.pic_grid_nr)
    pic_grid_nz = int(runtime.pic_grid_nz)
    if pic_grid_nr == 0 and pic_grid_nz == 0:
        return 0, 0
    if pic_grid_nr < 0 or pic_grid_nz < 0:
        raise ValueError("PIC grid nr and nz must be non-negative.")
    if pic_grid_nr == 0 or pic_grid_nz == 0:
        raise ValueError("PIC grid nr and nz must both be zero or both be positive.")
    if pic_grid_nr < 3 or pic_grid_nz < 3:
        raise ValueError("Explicit PIC grid node counts must both be at least 3.")
    return pic_grid_nr, pic_grid_nz


def _electrode_offset_m(runtime: RuntimeConfig) -> float | None:
    if (
        runtime.electrode_mask_z_offset_mm is None
        or str(runtime.electrode_mask_z_offset_mm).strip() == ""
    ):
        return None
    return float(runtime.electrode_mask_z_offset_mm) * 1.0e-3


def _beam_overrides(
    beam: BeamConfig,
    source_radius_m: float | None,
    source_gaussian_sigma_m: float | None,
    source_mode: str,
    z_min_m: float,
    z_max_m: float | None,
    macro_particle_weight: float,
) -> dict[str, Any]:
    gas_mode = _gas_velocity_mode(beam) == "from-gas-field"
    return {
        "ion_count": max(1, int(beam.particle_count)),
        "initial_position_jitter_m": max(float(beam.initial_position_jitter_mm), 0.0) * 1.0e-3,
        "initial_velocity_jitter_m_per_s": (
            0.0 if gas_mode else max(float(beam.velocity_jitter_m_per_s), 0.0)
        ),
        "source_radius_m": source_radius_m,
        "source_profile": str(beam.source_profile),
        "source_gaussian_sigma_m": source_gaussian_sigma_m,
        "source_mode": source_mode,
        "ion_current_a": max(float(beam.current_a), 0.0),
        "source_temperature_k": (
            None if gas_mode else _optional_positive(beam.source_temperature_k)
        ),
        "source_axial_velocity_m_per_s": (
            None if gas_mode else _optional_nonnegative(beam.source_axial_velocity_m_per_s)
        ),
        "source_velocity_from_gas_field": gas_mode,
        "source_birth_velocity_gas_path": (
            Path(beam.source_birth_velocity_gas_csv)
            if gas_mode and str(beam.source_birth_velocity_gas_csv).strip()
            else None
        ),
        "source_birth_velocity_z_min_m": z_min_m if gas_mode else 0.0,
        "source_birth_velocity_z_max_m": z_max_m if gas_mode else None,
        "source_birth_velocity_radius_m": _source_birth_radius_m(beam, gas_mode),
        "source_radial_velocity_scale": max(
            float(beam.source_radial_velocity_scale),
            0.0,
        ),
        "source_velocity_delta_m_per_s": (
            float(beam.source_velocity_delta_m_per_s) if gas_mode else 0.0
        ),
        "macro_particle_weight": macro_particle_weight,
        "cone_half_angle_rad": (
            0.0
            if gas_mode
            else np.radians(max(float(beam.cone_half_angle_deg), 0.0))
        ),
    }


def _optional_positive(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError("Source temperature must be positive when provided.")
    return parsed


def _optional_nonnegative(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError("Source axial velocity must be non-negative when provided.")
    return parsed


def _optional_mm_to_m(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value) * 1.0e-3


def _source_birth_radius_m(beam: BeamConfig, gas_mode: bool) -> float | None:
    if not gas_mode:
        return None
    return _optional_mm_to_m(beam.source_birth_velocity_radius_mm)


def _field_overrides(
    app_config: AppConfig,
    paths: tuple[Path | None, Path | None, Path | None, Path | None],
    pic_shape: tuple[int, int],
) -> dict[str, Any]:
    runtime = app_config.runtime
    static_field_path, static_field_3d_path, _, _ = paths
    pic_grid_nr, pic_grid_nz = pic_shape
    return {
        "total_time_s": max(float(runtime.total_time_s), 1.0e-12),
        "macro_time_step_s": max(float(runtime.macro_dt_s), 1.0e-12),
        "random_seed": int(runtime.random_seed),
        "static_field_path": static_field_path,
        "static_field_3d_path": static_field_3d_path,
        "rf_frequency_hz": max(float(runtime.rf_frequency_hz), 0.0),
        "rf_peak_voltage_v": max(float(runtime.rf_peak_voltage_v), 0.0),
        "rf_phase_rad": np.radians(float(runtime.rf_phase_deg)),
        "pic_poisson_backend": str(runtime.pic_poisson_backend),
        "pic_poisson_preconditioner": str(runtime.pic_poisson_preconditioner),
        "pic_poisson_warm_start": bool(runtime.pic_poisson_warm_start),
        "pic_amg_mode": str(runtime.pic_amg_mode),
        "pic_amg_solver": str(runtime.pic_amg_solver),
        "pic_amg_tolerance": max(float(runtime.pic_amg_tolerance), 1.0e-30),
        "pic_amg_max_iters": max(1, int(runtime.pic_amg_max_iters)),
        "pic_amg_fallback_backend": str(runtime.pic_amg_fallback_backend),
        "pic_grid_nr": pic_grid_nr,
        "pic_grid_nz": pic_grid_nz,
        "pic_space_charge_scale": float(runtime.pic_space_charge_scale),
        "detector_z_m": (
            float(runtime.detector_z_mm) * 1.0e-3
            if float(runtime.detector_z_mm) > 0.0
            else None
        ),
        "detector_radius_m": (
            float(runtime.detector_radius_mm) * 1.0e-3
            if float(runtime.detector_radius_mm) > 0.0
            else None
        ),
        "radial_limit_m": (
            float(runtime.radial_limit_mm) * 1.0e-3
            if float(runtime.radial_limit_mm) > 0.0
            else None
        ),
    }


def _source_runtime_overrides(
    beam: BeamConfig,
    runtime: RuntimeConfig,
) -> dict[str, Any]:
    return {
        "capillary_exit_z_m": (
            float(beam.capillary_exit_z_mm) * 1.0e-3
            if float(beam.capillary_exit_z_mm) >= 0.0
            else None
        ),
        "capillary_voltage_v": float(runtime.capillary_voltage_v),
        "capillary_prefill_length_m": (
            max(float(beam.capillary_prefill_length_mm), 0.0) * 1.0e-3
        ),
        "capillary_prefill_macro_particles": max(
            0,
            int(runtime.capillary_prefill_macro_particles),
        ),
        "macro_particles_per_injection": max(
            1,
            int(runtime.macro_particles_per_injection),
        ),
        "max_macro_particle_weight": max(
            float(runtime.max_macro_particle_weight),
            0.0,
        ),
        "stop_active_fraction_below": max(
            float(runtime.stop_active_fraction_below),
            0.0,
        ),
        "stop_stable_window_steps": max(
            0,
            int(runtime.stop_stable_window_steps),
        ),
        "stop_stable_fraction_tol": max(
            float(runtime.stop_stable_fraction_tol),
            0.0,
        ),
        "max_wall_time_s": max(float(runtime.max_wall_time_s), 0.0),
        "terminal_event_mode": str(runtime.terminal_event_mode),
        "max_terminal_event_rows": max(
            0,
            int(runtime.max_terminal_event_rows),
        ),
        "collision_batch_size": max(0, int(runtime.collision_batch_size)),
    }


def _collision_and_mask_overrides(
    runtime: RuntimeConfig,
    paths: tuple[Path | None, Path | None, Path | None, Path | None],
    electrode_mask_z_offset_m: float | None,
) -> dict[str, Any]:
    _, _, electrode_mask_path, electrode_mask_cache_path = paths
    return {
        "collision_physics_backend": str(
            runtime.collision_physics_backend
        ),
        "ionspa_backend": str(runtime.ionspa_backend),
        "collision_model": str(runtime.collision_mode),
        "fragmentation_mode": str(runtime.fragmentation_mode),
        "langevin_z_start_m": (
            max(float(runtime.langevin_z_start_mm), 0.0) * 1.0e-3
        ),
        "langevin_z_end_m": (
            max(float(runtime.langevin_z_end_mm), 0.0) * 1.0e-3
        ),
        "langevin_switch_probability": max(
            float(runtime.langevin_switch_prob),
            0.0,
        ),
        "langevin_max_dt_s": max(float(runtime.langevin_max_dt_s), 0.0),
        "electrode_mask_path": electrode_mask_path,
        "electrode_mask_cache_path": electrode_mask_cache_path,
        "electrode_mask_z_offset_m": electrode_mask_z_offset_m,
        "electrode_hit_distance_m": (
            max(float(runtime.electrode_hit_distance_mm), 0.0) * 1.0e-3
        ),
    }


def _iict_overrides(runtime: RuntimeConfig) -> dict[str, Any]:
    optional_paths = {
        "iict_parameter_config_path": runtime.iict_parameter_config_path,
        "iict_heat_capacity_csv_path": runtime.iict_heat_capacity_csv_path,
        "iict_pseudoatom_csv_path": runtime.iict_pseudoatom_csv_path,
    }
    overrides: dict[str, Any] = {
        name: Path(value) if str(value).strip() else None
        for name, value in optional_paths.items()
    }
    for name in (
        "iict_heat_capacity_model",
        "iict_num_atoms",
        "iict_constant_cv_j_per_k_per_ion",
        "iict_pseudoatom_model",
        "iict_pseudoatom_mass_da",
        "iict_pseudoatom_min_mass_da",
        "iict_pseudoatom_max_mass_da",
        "iict_fragmentation_model",
        "iict_delta_h_kj_per_mol",
        "iict_delta_s_j_per_mol_k",
    ):
        overrides[name] = getattr(runtime, name)
    return overrides


def build_simulation_config(app_config: AppConfig) -> SimulationConfig:
    beam = app_config.beam
    runtime = app_config.runtime
    source_radius_m, source_sigma_m = _source_geometry(beam)
    paths = _field_paths(app_config)
    macro_particle_weight = max(float(beam.macro_particle_weight), 0.0)
    source_mode = str(beam.source_mode).strip() or "packet"
    if _gas_velocity_mode(beam) == "from-gas-field":
        z_min_m, z_max_m = _source_birth_bounds(beam)
    else:
        z_min_m, z_max_m = 0.0, None
    pic_shape = _pic_grid_shape(runtime)
    electrode_offset_m = _electrode_offset_m(runtime)
    overrides = _beam_overrides(
        beam,
        source_radius_m,
        source_sigma_m,
        source_mode,
        z_min_m,
        z_max_m,
        macro_particle_weight,
    )
    overrides.update(_field_overrides(app_config, paths, pic_shape))
    overrides.update(_source_runtime_overrides(beam, runtime))
    overrides.update(
        _collision_and_mask_overrides(runtime, paths, electrode_offset_m)
    )
    overrides.update(_iict_overrides(runtime))
    return make_simulation_config(replace(SimulationConfig(), **overrides))
