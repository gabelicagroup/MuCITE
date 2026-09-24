"""Adapt parsed CLI values onto a canonical versioned configuration."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import warnings

import numpy as np
from scipy.constants import elementary_charge

from ...config.release_defaults import load_release_default

from ...config import (
    ATOMIC_MASS_CONSTANT,
    PROJECT_ROOT,
    ConfigDocument,
    ExecutionConfig,
    IonTemplate,
    OutputConfig,
    SimulationConfig,
    _nonnegative_float,
    _positive_float,
    load_config_document,
    make_config_document,
    make_execution_config,
    make_ion_template,
    make_output_config,
    make_simulation_config,
)


@dataclass(frozen=True)
class _OverrideContext:
    args: Namespace
    explicit_dests: frozenset[str]
    apply_all: bool

    def has(self, destination: str) -> bool:
        return self.apply_all or destination in self.explicit_dests


def _resolve_cli_electrode_mask_path(
    raw_value: Optional[str],
    *,
    no_electrode_mask: bool,
    dummy_static_field: bool,
) -> Optional[Path]:
    """Resolve the CLI electrode-mask policy without loading its payload."""

    if no_electrode_mask:
        if raw_value is not None:
            raise ValueError(
                "--no-electrode-mask cannot be combined with --electrode-mask."
            )
        return None
    if raw_value is None:
        if dummy_static_field:
            return None
        mask_text = "default"
    else:
        mask_text = str(raw_value).strip()
        if not mask_text:
            raise ValueError("--electrode-mask requires a path or 'default'.")
    if mask_text.lower() not in {"default", "auto"}:
        return Path(mask_text)
    from ...data.boundary_masks import resolve_default_electrode_mask_path

    resolved = resolve_default_electrode_mask_path(PROJECT_ROOT)
    if resolved is None:
        raise FileNotFoundError(
            "Default electrode mask was requested but no slens100x_gem.patxt "
            "was found. Pass --electrode-mask PATH or --no-electrode-mask."
        )
    return resolved


def _normalize_cli_pic_grid_shape(
    radial_nodes: Optional[int],
    axial_nodes: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    """Preserve default-shared, explicit-shared, and exact PIC requests."""

    if radial_nodes is None and axial_nodes is None:
        return None, None
    if radial_nodes is None or axial_nodes is None:
        raise ValueError("--pic-grid-nr and --pic-grid-nz must be set together.")
    radial_nodes = int(radial_nodes)
    axial_nodes = int(axial_nodes)
    if radial_nodes == 0 and axial_nodes == 0:
        return 0, 0
    if radial_nodes < 3 or axial_nodes < 3:
        raise ValueError(
            "--pic-grid-nr and --pic-grid-nz must both be 0, "
            "or both be at least 3."
        )
    return radial_nodes, axial_nodes


def _assign(
    output: dict[str, Any],
    context: _OverrideContext,
    destination: str,
    field_name: str,
    converter: Callable[[Any], Any] = lambda value: value,
) -> None:
    if context.has(destination):
        output[field_name] = converter(getattr(context.args, destination))


def _ion_scalar_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    mappings = (
        ("ion_name", "name", str),
        ("ion_mass_amu", "mass_kg", lambda value: float(value) * ATOMIC_MASS_CONSTANT),
        ("charge_state", "charge_state", int),
        ("collision_cross_section_m2", "collision_cross_section_m2", float),
        ("num_atoms", "num_atoms", int),
        ("heat_capacity_profile", "heat_capacity_profile", str),
        ("delta_h_kj_per_mol", "delta_h_kj_per_mol", float),
        ("delta_s_j_per_mol_k", "delta_s_j_per_mol_k", float),
        ("initial_internal_temperature_k", "initial_internal_temperature_k", float),
    )
    for destination, field_name, converter in mappings:
        _assign(output, context, destination, field_name, converter)
    return output


def _ion_position(
    base: IonTemplate,
    context: _OverrideContext,
) -> np.ndarray:
    position = np.array(base.initial_position_m, dtype=np.float64, copy=True)
    for index, destination in enumerate(
        ("initial_x_mm", "initial_y_mm", "initial_z_mm")
    ):
        if context.has(destination):
            position[index] = float(getattr(context.args, destination)) * 1.0e-3
    return position


def _ion_velocity(
    base: IonTemplate,
    final_mass_kg: float,
    context: _OverrideContext,
) -> np.ndarray:
    velocity = np.array(base.initial_velocity_m_per_s, dtype=np.float64, copy=True)
    kinetic_explicit = context.has("initial_kinetic_energy_ev")
    direction_explicit = context.has("initial_direction_axis")
    speed = float(np.linalg.norm(velocity))
    if kinetic_explicit and context.args.initial_kinetic_energy_ev is not None:
        energy_ev = _positive_float(
            context.args.initial_kinetic_energy_ev,
            "--initial-kinetic-energy-ev",
        )
        speed = float(np.sqrt(2.0 * energy_ev * elementary_charge / final_mass_kg))
    if not direction_explicit and not kinetic_explicit:
        return velocity
    direction = str(context.args.initial_direction_axis) if direction_explicit else ""
    if not direction:
        norm = float(np.linalg.norm(velocity))
        unit = velocity / norm if norm > 0.0 else np.array([0.0, 0.0, 1.0])
        return speed * unit
    vectors = {
        "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
    }
    return speed * np.asarray(vectors[direction], dtype=np.float64)


def _build_ion(base: IonTemplate, context: _OverrideContext) -> IonTemplate:
    overrides = _ion_scalar_overrides(context)
    final_mass = float(overrides.get("mass_kg", base.mass_kg))
    overrides["initial_position_m"] = _ion_position(base, context)
    overrides["initial_velocity_m_per_s"] = _ion_velocity(
        base,
        final_mass,
        context,
    )
    return make_ion_template(base, **overrides)


def _core_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    mappings = (
        ("ion_count", "ion_count", lambda value: max(1, int(value))),
        ("total_time", "total_time_s", lambda value: max(float(value), 1.0e-12)),
        ("macro_time_step", "macro_time_step_s", lambda value: max(float(value), 1.0e-12)),
        ("random_seed", "random_seed", int),
        ("macro_particle_weight", "macro_particle_weight", lambda value: _positive_float(value, "--macro-particle-weight")),
        ("initial_position_jitter_mm", "initial_position_jitter_m", lambda value: _nonnegative_float(value, "--initial-position-jitter-mm") * 1.0e-3),
        ("initial_velocity_jitter_m_per_s", "initial_velocity_jitter_m_per_s", lambda value: _nonnegative_float(value, "--initial-velocity-jitter-m-per-s")),
    )
    for destination, field_name, converter in mappings:
        _assign(output, context, destination, field_name, converter)
    return output


def _optional_scaled(
    value: Optional[float],
    name: str,
    *,
    nonnegative: bool = False,
) -> Optional[float]:
    if value is None:
        return None
    validator = _nonnegative_float if nonnegative else _positive_float
    return validator(value, name) * 1.0e-3


def _source_profile_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    _assign(output, context, "source_profile", "source_profile", str)
    _assign(output, context, "source_mode", "source_mode", str)
    _assign(output, context, "ion_current_a", "ion_current_a", lambda value: max(0.0, float(value)))
    _assign(output, context, "source_radius_mm", "source_radius_m", lambda value: _optional_scaled(value, "--source-radius-mm", nonnegative=True))
    _assign(output, context, "source_gaussian_sigma_mm", "source_gaussian_sigma_m", lambda value: _optional_scaled(value, "--source-gaussian-sigma-mm"))
    _assign(output, context, "source_mach_number", "source_mach_number", lambda value: max(0.0, float(value)))
    _assign(output, context, "source_gas_gamma", "source_gas_gamma", lambda value: _positive_float(value, "--source-gas-gamma"))
    _assign(output, context, "source_gas_molar_mass_kg_per_mol", "source_gas_molar_mass_kg_per_mol", lambda value: _positive_float(value, "--source-gas-molar-mass-kg-per-mol"))
    _assign(output, context, "source_temperature_k", "source_temperature_k", lambda value: None if value is None else _positive_float(value, "--source-temperature-k"))
    _assign(output, context, "source_axial_velocity_m_per_s", "source_axial_velocity_m_per_s", lambda value: None if value is None else _nonnegative_float(value, "--source-axial-velocity-m-per-s"))
    return output


def _source_birth_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    _assign(output, context, "source_velocity_from_gas_field", "source_velocity_from_gas_field", bool)
    _assign(output, context, "source_birth_velocity_gas_csv", "source_birth_velocity_gas_path", lambda value: Path(value) if str(value).strip() else None)
    _assign(output, context, "source_birth_velocity_z_min_mm", "source_birth_velocity_z_min_m", lambda value: _nonnegative_float(value, "--source-birth-velocity-z-min-mm") * 1.0e-3)
    _assign(output, context, "source_birth_velocity_z_max_mm", "source_birth_velocity_z_max_m", lambda value: _optional_scaled(value, "--source-birth-velocity-z-max-mm", nonnegative=True))
    _assign(output, context, "source_birth_velocity_radius_mm", "source_birth_velocity_radius_m", lambda value: _optional_scaled(value, "--source-birth-velocity-radius-mm"))
    _assign(output, context, "source_radial_velocity_scale", "source_radial_velocity_scale", lambda value: max(0.0, float(value)))
    _assign(output, context, "source_velocity_delta_m_per_s", "source_velocity_delta_m_per_s", float)
    return output


def _capillary_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    _assign(output, context, "capillary_exit_z_mm", "capillary_exit_z_m", lambda value: _optional_scaled(value, "--capillary-exit-z-mm", nonnegative=True))
    _assign(output, context, "capillary_voltage_v", "capillary_voltage_v", float)
    _assign(output, context, "capillary_prefill_length_mm", "capillary_prefill_length_m", lambda value: _nonnegative_float(value, "--capillary-prefill-length-mm") * 1.0e-3)
    _assign(output, context, "capillary_prefill_macro_particles", "capillary_prefill_macro_particles", lambda value: max(0, int(value)))
    _assign(output, context, "macro_particles_per_injection", "macro_particles_per_injection", lambda value: max(1, int(value)))
    _assign(output, context, "max_macro_particle_weight", "max_macro_particle_weight", lambda value: max(0.0, float(value)))
    _assign(output, context, "cone_half_angle_deg", "cone_half_angle_rad", lambda value: float(np.radians(_nonnegative_float(value, "--cone-half-angle-deg"))))
    return output


def _field_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    _assign(output, context, "rf_frequency", "rf_frequency_hz", lambda value: max(0.0, float(value)))
    _assign(output, context, "rf_peak_voltage", "rf_peak_voltage_v", lambda value: max(0.0, float(value)))
    _assign(output, context, "rf_phase_deg", "rf_phase_rad", lambda value: float(np.radians(value)))
    _assign(output, context, "rf_reference_peak_voltage", "rf_reference_peak_voltage_v")
    if context.has("dummy_static_field") and context.args.dummy_static_field:
        output["static_field_path"] = None
    elif context.has("static_field"):
        output["static_field_path"] = Path(context.args.static_field) if str(context.args.static_field).strip() else None
    _assign(output, context, "stage_schedule", "stage_schedule_path", lambda value: Path(value) if str(value).strip() else None)
    _assign(output, context, "static_field_3d", "static_field_3d_path", lambda value: Path(value) if str(value).strip() else None)
    return output


def _electrode_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    policy_selected = (
        context.has("electrode_mask")
        or context.has("no_electrode_mask")
        or context.has("dummy_static_field")
    )
    if policy_selected:
        output["electrode_mask_path"] = _resolve_cli_electrode_mask_path(
            context.args.electrode_mask,
            no_electrode_mask=bool(context.args.no_electrode_mask),
            dummy_static_field=bool(context.args.dummy_static_field),
        )
    _assign(output, context, "electrode_mask_cache", "electrode_mask_cache_path", lambda value: Path(value) if str(value).strip() else None)
    _assign(output, context, "electrode_mask_z_offset_mm", "electrode_mask_z_offset_m", lambda value: None if value is None else float(value) * 1.0e-3)
    _assign(output, context, "electrode_mask_3d", "electrode_mask_3d_path", lambda value: Path(value) if str(value).strip() else None)
    _assign(output, context, "electrode_hit_distance_mm", "electrode_hit_distance_m", lambda value: _nonnegative_float(value, "--electrode-hit-distance-mm") * 1.0e-3)
    return output


def _pic_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    mappings = (
        ("sor_omega", "sor_omega", lambda value: _positive_float(value, "--sor-omega")),
        ("sor_max_iters", "sor_max_iters", lambda value: max(1, int(value))),
        ("sor_tolerance", "sor_tolerance", lambda value: _positive_float(value, "--sor-tolerance")),
        ("pic_poisson_backend", "pic_poisson_backend", str),
        ("pic_poisson_preconditioner", "pic_poisson_preconditioner", str),
        ("pic_poisson_warm_start", "pic_poisson_warm_start", bool),
        ("pic_amg_mode", "pic_amg_mode", str),
        ("pic_amg_solver", "pic_amg_solver", str),
        ("pic_amg_tolerance", "pic_amg_tolerance", lambda value: _positive_float(value, "--pic-amg-tolerance")),
        ("pic_amg_max_iters", "pic_amg_max_iters", lambda value: max(1, int(value))),
        ("pic_amg_fallback_backend", "pic_amg_fallback_backend", str),
        ("pic_space_charge_scale", "pic_space_charge_scale", lambda value: _nonnegative_float(value, "--pic-space-charge-scale")),
    )
    for destination, field_name, converter in mappings:
        _assign(output, context, destination, field_name, converter)
    if context.has("pic_grid_nr") or context.has("pic_grid_nz"):
        output["pic_grid_nr"], output["pic_grid_nz"] = _normalize_cli_pic_grid_shape(
            context.args.pic_grid_nr,
            context.args.pic_grid_nz,
        )
    return output


def _boundary_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    _assign(output, context, "detector_z_mm", "detector_z_m", lambda value: _optional_scaled(value, "--detector-z-mm"))
    _assign(output, context, "detector_radius_mm", "detector_radius_m", lambda value: _optional_scaled(value, "--detector-radius-mm"))
    _assign(output, context, "radial_limit_mm", "radial_limit_m", lambda value: _optional_scaled(value, "--radial-limit-mm"))
    _assign(output, context, "stop_active_fraction_below", "stop_active_fraction_below", lambda value: max(0.0, min(float(value), 1.0)))
    _assign(output, context, "stop_stable_window", "stop_stable_window_steps", lambda value: max(0, int(value)))
    _assign(output, context, "stop_stable_fraction_tol", "stop_stable_fraction_tol", lambda value: max(0.0, float(value)))
    _assign(output, context, "max_wall_time_s", "max_wall_time_s", lambda value: max(0.0, float(value)))
    _assign(output, context, "terminal_event_mode", "terminal_event_mode", str)
    _assign(output, context, "max_terminal_event_rows", "max_terminal_event_rows", lambda value: max(0, int(value)))
    return output


def _collision_overrides(context: _OverrideContext) -> dict[str, Any]:
    output: dict[str, Any] = {}
    _assign(output, context, "collision_batch_size", "collision_batch_size", lambda value: max(0, int(value)))
    _assign(output, context, "legacy_collision_gather", "collision_preselection_enabled", lambda value: not bool(value))
    _assign(output, context, "collision_physics_backend", "collision_physics_backend", str)
    _assign(output, context, "ionspa_backend", "ionspa_backend", str)
    iict_mappings = (
        ("iict_parameter_config", "iict_parameter_config_path", lambda value: Path(value) if str(value).strip() else None),
        ("iict_heat_capacity_model", "iict_heat_capacity_model", str),
        ("iict_num_atoms", "iict_num_atoms", int),
        ("iict_constant_cv_j_per_k_per_ion", "iict_constant_cv_j_per_k_per_ion", float),
        ("iict_heat_capacity_csv", "iict_heat_capacity_csv_path", lambda value: Path(value) if str(value).strip() else None),
        ("iict_pseudoatom_model", "iict_pseudoatom_model", str),
        ("iict_pseudoatom_mass_da", "iict_pseudoatom_mass_da", float),
        ("iict_pseudoatom_csv", "iict_pseudoatom_csv_path", lambda value: Path(value) if str(value).strip() else None),
        ("iict_pseudoatom_min_mass_da", "iict_pseudoatom_min_mass_da", float),
        ("iict_pseudoatom_max_mass_da", "iict_pseudoatom_max_mass_da", float),
        ("iict_fragmentation_model", "iict_fragmentation_model", str),
        ("iict_delta_h_kj_per_mol", "iict_delta_h_kj_per_mol", float),
        ("iict_delta_s_j_per_mol_k", "iict_delta_s_j_per_mol_k", float),
    )
    cli_source_fields: list[str] = []
    for destination, field_name, converter in iict_mappings:
        _assign(output, context, destination, field_name, converter)
        if (
            context.has(destination)
            and field_name != "iict_parameter_config_path"
        ):
            cli_source_fields.append(field_name)
    output["iict_cli_override_fields"] = tuple(cli_source_fields)
    _assign(output, context, "collision_model", "collision_model", str)
    _assign(output, context, "fragmentation_mode", "fragmentation_mode", str)
    _assign(output, context, "langevin_z_start_mm", "langevin_z_start_m", lambda value: _nonnegative_float(value, "--langevin-z-start-mm") * 1.0e-3)
    _assign(output, context, "langevin_z_end_mm", "langevin_z_end_m", lambda value: _nonnegative_float(value, "--langevin-z-end-mm") * 1.0e-3)
    _assign(output, context, "langevin_switch_prob", "langevin_switch_probability", lambda value: max(0.0, float(value)))
    _assign(output, context, "langevin_max_dt_s", "langevin_max_dt_s", lambda value: max(0.0, float(value)))
    return output


def _build_simulation(
    base: SimulationConfig,
    context: _OverrideContext,
) -> SimulationConfig:
    overrides: dict[str, Any] = {}
    for builder in (
        _core_overrides, _source_profile_overrides, _source_birth_overrides,
        _capillary_overrides, _field_overrides, _electrode_overrides,
        _pic_overrides, _boundary_overrides, _collision_overrides,
    ):
        overrides.update(builder(context))
    return make_simulation_config(base, **overrides)


def _build_execution(
    base: ExecutionConfig,
    context: _OverrideContext,
) -> ExecutionConfig:
    overrides: dict[str, Any] = {}
    for name in ("backend", "gui", "live_window", "no_window", "progress"):
        _assign(overrides, context, name, name)
    return make_execution_config(base, **overrides)


def _build_output(
    base: OutputConfig,
    context: _OverrideContext,
) -> OutputConfig:
    overrides: dict[str, Any] = {}
    _assign(overrides, context, "export_every", "export_every", lambda value: max(0, int(value)))
    _assign(overrides, context, "export_dir", "export_dir", lambda value: Path(value) if str(value).strip() else None)
    _assign(overrides, context, "export_format", "export_format", str)
    _assign(overrides, context, "trajectory_sample_count", "trajectory_sample_count", lambda value: max(0, int(value)))
    _assign(overrides, context, "trajectory_record_every", "trajectory_record_every", lambda value: max(1, int(value)))
    _assign(overrides, context, "snapshot_plot_max_points", "snapshot_plot_max_points", lambda value: max(1, int(value)))
    _assign(overrides, context, "no_snapshot_plots", "snapshot_plots_enabled", lambda value: not bool(value))
    _assign(overrides, context, "report", "report", bool)
    _assign(overrides, context, "report_dir", "report_dir", lambda value: Path(value) if str(value).strip() else None)
    return make_output_config(base, **overrides)


def resolve_cli_document(
    args: Namespace,
    explicit_dests: frozenset[str],
) -> ConfigDocument:
    """Merge only explicit CLI values over a strict JSON document."""

    config_path = str(args.config).strip()
    base = (
        load_config_document(Path(config_path))
        if config_path
        else load_release_default()
    )
    context = _OverrideContext(
        args=args,
        explicit_dests=explicit_dests,
        apply_all=False,
    )
    if "ionspa_backend" in explicit_dests:
        if (
            "collision_physics_backend" in explicit_dests
            and args.collision_physics_backend == "iict-lite"
        ):
            raise ValueError(
                "--ionspa-backend cannot be combined with "
                "--collision-physics-backend iict-lite."
            )
        warnings.warn(
            "--ionspa-backend is deprecated; use "
            "--collision-physics-backend ionspa and configure a lawful "
            "IonSPA provider explicitly.",
            FutureWarning,
            stacklevel=2,
        )
        context_args = vars(args).copy()
        context_args["collision_physics_backend"] = "ionspa"
        args = Namespace(**context_args)
        context = _OverrideContext(
            args=args,
            explicit_dests=explicit_dests | {"collision_physics_backend"},
            apply_all=False,
        )
    return make_config_document(
        _build_simulation(base.simulation, context),
        _build_ion(base.ion, context),
        _build_execution(base.execution, context),
        _build_output(base.output, context),
    )
