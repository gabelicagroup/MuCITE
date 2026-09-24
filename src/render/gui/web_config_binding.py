"""Mapping between the browser form payload and GUI configuration DTOs."""

from __future__ import annotations

from typing import Any

from .models import AppConfig


BEAM_FIELDS = (
    "ion_name",
    "mass_amu",
    "charge_e",
    "particle_count",
    "source_mode",
    "current_a",
    "kinetic_energy_ev",
    "initial_internal_temperature_k",
    "num_atoms",
    "heat_capacity_profile",
    "delta_h_kj_per_mol",
    "delta_s_j_per_mol_k",
    "beam_radius_mm",
    "initial_z_mm",
    "cone_half_angle_deg",
    "direction_axis",
    "source_profile",
    "source_gaussian_sigma_mm",
    "initial_position_jitter_mm",
    "velocity_jitter_m_per_s",
    "collision_cross_section_m2",
    "gas_velocity_init_mode",
    "source_temperature_k",
    "source_axial_velocity_m_per_s",
    "source_birth_velocity_gas_csv",
    "source_birth_velocity_z_min_mm",
    "source_birth_velocity_z_max_mm",
    "source_birth_velocity_radius_mm",
    "source_radial_velocity_scale",
    "source_velocity_delta_m_per_s",
    "capillary_exit_z_mm",
    "capillary_prefill_length_mm",
)

FIELD_BAKE_FIELDS = (
    "simion_dc_path",
    "simion_rf_path",
    "gas_field_mode",
    "fluent_path",
    "z_max_mm",
    "r_max_mm",
    "dz_mm",
    "dr_mm",
    "pa_grids_per_mm",
    "offset_z_simion_mm",
    "offset_z_fluent_mm",
    "background_pressure_pa",
    "background_temperature_k",
    "capillary_exit_z_mm",
    "fluent_capillary_total_length_mm",
    "fluent_capillary_radius_mm",
    "fluent_capillary_external_start_z_mm",
    "output_npy",
)

RUNTIME_FIELDS = (
    "backend",
    "total_time_s",
    "macro_dt_s",
    "random_seed",
    "snapshot_every",
    "pic_space_charge_scale",
    "collision_physics_backend",
    "iict_parameter_config_path",
    "iict_heat_capacity_model",
    "iict_num_atoms",
    "iict_constant_cv_j_per_k_per_ion",
    "iict_heat_capacity_csv_path",
    "iict_pseudoatom_model",
    "iict_pseudoatom_mass_da",
    "iict_pseudoatom_csv_path",
    "iict_pseudoatom_min_mass_da",
    "iict_pseudoatom_max_mass_da",
    "iict_fragmentation_model",
    "iict_delta_h_kj_per_mol",
    "iict_delta_s_j_per_mol_k",
    "rf_peak_voltage_v",
    "rf_frequency_hz",
    "rf_phase_deg",
    "pic_grid_nr",
    "pic_grid_nz",
    "pic_poisson_backend",
    "pic_poisson_preconditioner",
    "pic_poisson_warm_start",
    "pic_amg_mode",
    "pic_amg_solver",
    "pic_amg_tolerance",
    "pic_amg_max_iters",
    "pic_amg_fallback_backend",
    "detector_z_mm",
    "detector_radius_mm",
    "radial_limit_mm",
    "capillary_voltage_v",
    "capillary_prefill_macro_particles",
    "macro_particles_per_injection",
    "max_macro_particle_weight",
    "max_wall_time_s",
    "max_terminal_event_rows",
    "collision_batch_size",
    "langevin_z_start_mm",
    "langevin_z_end_mm",
    "langevin_switch_prob",
    "langevin_max_dt_s",
    "electrode_mask_z_offset_mm",
    "electrode_hit_distance_mm",
    "ionspa_backend",
    "collision_mode",
    "terminal_event_mode",
    "electrode_mask_path",
    "electrode_mask_cache_path",
    "fragmentation_mode",
)

IICT_OPTIONAL_RUNTIME_FIELDS = frozenset({
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
})

OUTPUT_FIELDS = (
    "trajectory_sample_count",
    "trajectory_record_every",
    "snapshot_plot_max_points",
    "terminal_time_bin_ms",
    "save_snapshots",
    "save_h5",
    "save_figures",
    "generate_report_after_run",
)


def _apply_group(
    target: Any,
    payload: Any,
    fields: tuple[str, ...],
    optional_empty_fields: frozenset[str] = frozenset(),
) -> None:
    if not isinstance(payload, dict):
        return
    for name in fields:
        if name not in payload:
            continue
        value = payload[name]
        if name in optional_empty_fields and value == "":
            value = None
        setattr(target, name, value)


def apply_web_payload(config: AppConfig, payload: dict[str, Any]) -> None:
    """Apply only fields represented by the browser form."""

    if not isinstance(payload, dict):
        return
    for name in ("session_name", "output_dir", "loaded_baked_field_path"):
        if name in payload:
            setattr(config, name, str(payload[name]))
    _apply_group(
        config.beam,
        payload.get("beam"),
        BEAM_FIELDS,
        frozenset({
            "kinetic_energy_ev",
            "source_temperature_k",
            "source_axial_velocity_m_per_s",
            "source_birth_velocity_z_max_mm",
            "source_birth_velocity_radius_mm",
        }),
    )
    _apply_group(
        config.field_bake,
        payload.get("field_bake"),
        FIELD_BAKE_FIELDS,
    )
    _apply_group(
        config.runtime,
        payload.get("runtime"),
        RUNTIME_FIELDS,
        IICT_OPTIONAL_RUNTIME_FIELDS
        | frozenset({"electrode_mask_z_offset_mm"}),
    )
    _apply_group(config.output, payload.get("output"), OUTPUT_FIELDS)


__all__ = ["apply_web_payload"]
