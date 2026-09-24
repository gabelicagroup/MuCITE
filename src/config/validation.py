"""Normalization and validation for canonical configuration requests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .models import ExecutionConfig, IonTemplate, OutputConfig, SimulationConfig
from .iict_validation import validate_iict_request
from .tuning_validation import (
    validate_execution_tuning,
    validate_output_tuning,
    validate_simulation_tuning,
)

_SIMULATION_CONFIG_ENUMS: dict[str, frozenset[str]] = {
    "source_profile": frozenset({"uniform-disk", "gaussian"}),
    "source_mode": frozenset({"packet", "continuous-current"}),
    "pic_poisson_backend": frozenset(
        {
            "sor",
            "sparse_direct",
            "sparse_spsolve",
            "sparse_bicgstab",
            "sparse_cg",
            "amg",
            "pyamg",
        }
    ),
    "pic_poisson_preconditioner": frozenset({"none", "jacobi"}),
    "pic_amg_mode": frozenset(
        {"solve", "preconditioned_cg", "preconditioned_bicgstab"}
    ),
    "pic_amg_solver": frozenset({"ruge_stuben", "smoothed_aggregation"}),
    "pic_amg_fallback_backend": frozenset(
        {"sparse_bicgstab", "sparse_cg", "sparse_spsolve", "sparse_direct"}
    ),
    "terminal_event_mode": frozenset({"transport-only", "all", "none"}),
    "collision_physics_backend": frozenset({"iict-lite", "ionspa"}),
    "ionspa_backend": frozenset({"bundled", "approximate", "local"}),
    "collision_model": frozenset({"explicit", "hybrid-langevin"}),
    "fragmentation_mode": frozenset({"transport", "loss", "off"}),
    "gas_species": frozenset({"n2"}),
}

SIMULATION_CONFIG_PATH_FIELDS = (
    "source_birth_velocity_gas_path",
    "iict_parameter_config_path",
    "iict_heat_capacity_csv_path",
    "iict_pseudoatom_csv_path",
    "static_field_path",
    "stage_schedule_path",
    "static_field_3d_path",
    "electrode_mask_path",
    "electrode_mask_cache_path",
    "electrode_mask_3d_path",
)


def normalize_execution_config(config: ExecutionConfig) -> ExecutionConfig:
    """Canonicalize execution policy strings."""

    if not isinstance(config, ExecutionConfig):
        raise TypeError("config must be an ExecutionConfig instance.")
    return replace(config, backend=str(config.backend).strip().lower())


def validate_execution_config(config: ExecutionConfig) -> None:
    """Validate non-physical execution controls."""

    if not isinstance(config, ExecutionConfig):
        raise TypeError("config must be an ExecutionConfig instance.")
    if config.backend not in {"cpu", "taichi"}:
        raise ValueError("backend must be one of {cpu, taichi}.")
    validate_execution_tuning(config)


def normalize_output_config(config: OutputConfig) -> OutputConfig:
    """Canonicalize output format and optional paths."""

    if not isinstance(config, OutputConfig):
        raise TypeError("config must be an OutputConfig instance.")
    return replace(
        config,
        export_format=str(config.export_format).strip().lower(),
        export_dir=None if config.export_dir is None else Path(config.export_dir),
        report_dir=None if config.report_dir is None else Path(config.report_dir),
    )


def validate_output_config(config: OutputConfig) -> None:
    """Validate snapshot and report policy."""

    if not isinstance(config, OutputConfig):
        raise TypeError("config must be an OutputConfig instance.")
    if config.export_format not in {"npy", "h5"}:
        raise ValueError("export_format must be one of {h5, npy}.")
    if int(config.export_every) < 0:
        raise ValueError("export_every must be non-negative.")
    if int(config.trajectory_sample_count) < 0:
        raise ValueError("trajectory_sample_count must be non-negative.")
    if int(config.trajectory_record_every) <= 0:
        raise ValueError("trajectory_record_every must be positive.")
    if int(config.snapshot_plot_max_points) <= 0:
        raise ValueError("snapshot_plot_max_points must be positive.")
    validate_output_tuning(config)


def normalize_simulation_config(config: SimulationConfig) -> SimulationConfig:
    """Canonicalize strings and paths without resolving effective PIC nodes."""

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance.")
    updates: dict[str, Any] = {
        name: str(getattr(config, name)).strip().lower()
        for name in _SIMULATION_CONFIG_ENUMS
    }
    for name in SIMULATION_CONFIG_PATH_FIELDS:
        value = getattr(config, name)
        if value is not None:
            updates[name] = Path(value)
    for name in (
        "iict_heat_capacity_model",
        "iict_pseudoatom_model",
        "iict_fragmentation_model",
    ):
        value = getattr(config, name)
        if value is not None:
            updates[name] = str(value).strip().lower()
    updates["iict_cli_override_fields"] = tuple(
        str(name) for name in config.iict_cli_override_fields
    )
    return replace(config, **updates)


def _require_finite(value: float, field_name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    return numeric


def _require_positive(value: float, field_name: str) -> float:
    numeric = _require_finite(value, field_name)
    if numeric <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return numeric


def _require_nonnegative(value: float, field_name: str) -> float:
    numeric = _require_finite(value, field_name)
    if numeric < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return numeric


def _validate_optional_positive(
    value: Optional[float],
    field_name: str,
) -> None:
    if value is not None:
        _require_positive(value, field_name)


def _validate_optional_nonnegative(
    value: Optional[float],
    field_name: str,
) -> None:
    if value is not None:
        _require_nonnegative(value, field_name)


def _validate_enum_fields(config: SimulationConfig) -> None:
    for field_name, allowed_values in _SIMULATION_CONFIG_ENUMS.items():
        value = getattr(config, field_name)
        if value not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise ValueError(
                f"{field_name} must be one of {{{allowed}}}; got {value!r}."
            )


def _validate_core_ranges(config: SimulationConfig) -> None:
    if int(config.ion_count) <= 0:
        raise ValueError("ion_count must be positive.")
    _require_positive(config.total_time_s, "total_time_s")
    _require_positive(config.macro_time_step_s, "macro_time_step_s")
    _require_positive(config.domain_radius_m, "domain_radius_m")
    _require_positive(config.domain_length_m, "domain_length_m")
    _require_nonnegative(
        config.initial_position_jitter_m,
        "initial_position_jitter_m",
    )
    _require_nonnegative(
        config.initial_velocity_jitter_m_per_s,
        "initial_velocity_jitter_m_per_s",
    )


def _validate_mesh_request(config: SimulationConfig) -> None:
    if int(config.grid_nr) < 3 or int(config.grid_nz) < 3:
        raise ValueError("grid_nr and grid_nz must both be at least 3.")
    pic_nr = config.pic_grid_nr
    pic_nz = config.pic_grid_nz
    if (pic_nr is None) != (pic_nz is None):
        raise ValueError("pic_grid_nr and pic_grid_nz must be specified together.")
    if pic_nr is None:
        return
    requested = (int(pic_nr), int(pic_nz))
    if requested != (0, 0) and (requested[0] < 3 or requested[1] < 3):
        raise ValueError(
            "PIC grid node counts must both be 0 (share static) or both be at least 3."
        )


def _validate_rf(config: SimulationConfig) -> None:
    _require_nonnegative(config.rf_frequency_hz, "rf_frequency_hz")
    _require_nonnegative(config.rf_peak_voltage_v, "rf_peak_voltage_v")
    _require_finite(config.rf_phase_rad, "rf_phase_rad")
    _validate_optional_positive(
        config.rf_reference_peak_voltage_v,
        "rf_reference_peak_voltage_v",
    )


def _validate_source_profile(config: SimulationConfig) -> None:
    _validate_optional_nonnegative(config.source_radius_m, "source_radius_m")
    _validate_optional_positive(
        config.source_gaussian_sigma_m,
        "source_gaussian_sigma_m",
    )
    if config.source_profile != "gaussian":
        return
    if config.source_radius_m is None or config.source_radius_m <= 0.0:
        raise ValueError(
            "source_radius_m must be positive when source_profile is 'gaussian'."
        )
    if (
        config.source_gaussian_sigma_m is None
        or config.source_gaussian_sigma_m <= 0.0
    ):
        raise ValueError(
            "source_gaussian_sigma_m must be positive when "
            "source_profile is 'gaussian'."
        )


def _validate_source_gas(config: SimulationConfig) -> None:
    _require_nonnegative(config.ion_current_a, "ion_current_a")
    if config.source_mode == "continuous-current" and config.ion_current_a <= 0.0:
        raise ValueError(
            "ion_current_a must be positive when "
            "source_mode is 'continuous-current'."
        )
    _require_nonnegative(config.source_mach_number, "source_mach_number")
    _require_positive(config.source_gas_gamma, "source_gas_gamma")
    _require_positive(
        config.source_gas_molar_mass_kg_per_mol,
        "source_gas_molar_mass_kg_per_mol",
    )
    _validate_optional_positive(config.source_temperature_k, "source_temperature_k")
    _validate_optional_nonnegative(
        config.source_axial_velocity_m_per_s,
        "source_axial_velocity_m_per_s",
    )


def _validate_source_birth(config: SimulationConfig) -> None:
    _require_nonnegative(
        config.source_birth_velocity_z_min_m,
        "source_birth_velocity_z_min_m",
    )
    _validate_optional_nonnegative(
        config.source_birth_velocity_z_max_m,
        "source_birth_velocity_z_max_m",
    )
    if (
        config.source_birth_velocity_z_max_m is not None
        and config.source_birth_velocity_z_max_m
        < config.source_birth_velocity_z_min_m
    ):
        raise ValueError(
            "source_birth_velocity_z_max_m must be greater than or equal to "
            "source_birth_velocity_z_min_m."
        )
    _validate_optional_positive(
        config.source_birth_velocity_radius_m,
        "source_birth_velocity_radius_m",
    )
    _require_nonnegative(
        config.source_radial_velocity_scale,
        "source_radial_velocity_scale",
    )
    _require_finite(
        config.source_velocity_delta_m_per_s,
        "source_velocity_delta_m_per_s",
    )


def _validate_injection(config: SimulationConfig) -> None:
    _validate_optional_nonnegative(config.capillary_exit_z_m, "capillary_exit_z_m")
    _require_finite(config.capillary_voltage_v, "capillary_voltage_v")
    _require_nonnegative(
        config.capillary_prefill_length_m,
        "capillary_prefill_length_m",
    )
    if int(config.capillary_prefill_macro_particles) < 0:
        raise ValueError("capillary_prefill_macro_particles must be non-negative.")
    if int(config.macro_particles_per_injection) <= 0:
        raise ValueError("macro_particles_per_injection must be positive.")
    _require_nonnegative(
        config.max_macro_particle_weight,
        "max_macro_particle_weight",
    )
    _require_positive(config.macro_particle_weight, "macro_particle_weight")
    angle = _require_nonnegative(
        config.cone_half_angle_rad,
        "cone_half_angle_rad",
    )
    if angle >= 0.5 * np.pi:
        raise ValueError("cone_half_angle_rad must be in [0, pi/2).")


def _validate_poisson(config: SimulationConfig) -> None:
    _require_nonnegative(config.pic_space_charge_scale, "pic_space_charge_scale")
    sor_omega = _require_positive(config.sor_omega, "sor_omega")
    if sor_omega >= 2.0:
        raise ValueError("sor_omega must be in (0, 2).")
    if int(config.sor_max_iters) <= 0:
        raise ValueError("sor_max_iters must be positive.")
    _require_positive(config.sor_tolerance, "sor_tolerance")
    _require_positive(config.pic_amg_tolerance, "pic_amg_tolerance")
    if int(config.pic_amg_max_iters) <= 0:
        raise ValueError("pic_amg_max_iters must be positive.")


def _validate_termination(config: SimulationConfig) -> None:
    _validate_optional_positive(config.detector_z_m, "detector_z_m")
    _validate_optional_positive(config.detector_radius_m, "detector_radius_m")
    _validate_optional_positive(config.radial_limit_m, "radial_limit_m")
    stop_fraction = _require_nonnegative(
        config.stop_active_fraction_below,
        "stop_active_fraction_below",
    )
    if stop_fraction > 1.0:
        raise ValueError("stop_active_fraction_below must be in [0, 1].")
    if int(config.stop_stable_window_steps) < 0:
        raise ValueError("stop_stable_window_steps must be non-negative.")
    _require_nonnegative(
        config.stop_stable_fraction_tol,
        "stop_stable_fraction_tol",
    )
    _require_nonnegative(config.max_wall_time_s, "max_wall_time_s")
    if int(config.max_terminal_event_rows) < 0:
        raise ValueError("max_terminal_event_rows must be non-negative.")


def _validate_electrode_and_collision(config: SimulationConfig) -> None:
    _require_nonnegative(
        config.electrode_hit_distance_m,
        "electrode_hit_distance_m",
    )
    if config.electrode_mask_z_offset_m is not None:
        _require_finite(
            config.electrode_mask_z_offset_m,
            "electrode_mask_z_offset_m",
        )
    if int(config.collision_batch_size) < 0:
        raise ValueError("collision_batch_size must be non-negative.")
    validate_iict_request(config)
    _require_nonnegative(config.langevin_z_start_m, "langevin_z_start_m")
    _require_nonnegative(config.langevin_z_end_m, "langevin_z_end_m")
    if (
        config.collision_model == "hybrid-langevin"
        and config.langevin_z_end_m < config.langevin_z_start_m
    ):
        raise ValueError(
            "langevin_z_end_m must be greater than or equal to "
            "langevin_z_start_m for hybrid-langevin collisions."
        )
    switch = _require_nonnegative(
        config.langevin_switch_probability,
        "langevin_switch_probability",
    )
    if switch > 1.0:
        raise ValueError("langevin_switch_probability must be in [0, 1].")
    _require_nonnegative(config.langevin_max_dt_s, "langevin_max_dt_s")


def validate_simulation_config(config: SimulationConfig) -> None:
    """Validate stable invariants shared by CLI, GUI, JSON, and runtime paths."""

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance.")
    _validate_enum_fields(config)
    _validate_core_ranges(config)
    _validate_mesh_request(config)
    _validate_rf(config)
    _validate_source_profile(config)
    _validate_source_gas(config)
    _validate_source_birth(config)
    _validate_injection(config)
    _validate_poisson(config)
    _validate_termination(config)
    _validate_electrode_and_collision(config)
    validate_simulation_tuning(config)


def normalize_ion_template(template: IonTemplate) -> IonTemplate:
    """Return a canonical ion request with owned float64 state vectors."""

    if not isinstance(template, IonTemplate):
        raise TypeError("template must be an IonTemplate instance.")
    return replace(
        template,
        name=str(template.name).strip(),
        heat_capacity_profile=str(template.heat_capacity_profile).strip(),
        initial_position_m=np.array(
            template.initial_position_m,
            dtype=np.float64,
            copy=True,
        ),
        initial_velocity_m_per_s=np.array(
            template.initial_velocity_m_per_s,
            dtype=np.float64,
            copy=True,
        ),
    )


def validate_ion_template(template: IonTemplate) -> None:
    """Validate ion properties shared by all request adapters."""

    if not isinstance(template, IonTemplate):
        raise TypeError("template must be an IonTemplate instance.")
    if not template.name:
        raise ValueError("ion name must not be empty.")
    _require_positive(template.mass_kg, "mass_kg")
    if int(template.charge_state) <= 0:
        raise ValueError("charge_state must be positive.")
    _require_positive(
        template.collision_cross_section_m2,
        "collision_cross_section_m2",
    )
    if int(template.num_atoms) <= 2:
        raise ValueError("num_atoms must be greater than 2 for IonSPA heat capacity.")
    _require_positive(
        template.initial_internal_temperature_k,
        "initial_internal_temperature_k",
    )
    if not template.heat_capacity_profile:
        raise ValueError("heat_capacity_profile must not be empty.")
    _require_positive(template.delta_h_kj_per_mol, "delta_h_kj_per_mol")
    _require_finite(template.delta_s_j_per_mol_k, "delta_s_j_per_mol_k")
    for field_name in ("initial_position_m", "initial_velocity_m_per_s"):
        values = np.asarray(getattr(template, field_name), dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{field_name} must be a finite three-component vector."
            )
