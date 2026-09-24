"""Report-facing DTO mappings for ion templates and simulation config."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.constants import elementary_charge

from ...config import ATOMIC_MASS_CONSTANT, IonTemplate, SimulationConfig


def _template_summary(template: IonTemplate) -> dict[str, Any]:
    speed_m_per_s = float(np.linalg.norm(template.initial_velocity_m_per_s))
    return {
        "name": template.name,
        "mass_amu": float(template.mass_kg / ATOMIC_MASS_CONSTANT),
        "mass_kg": float(template.mass_kg),
        "charge_state": int(template.charge_state),
        "charge_c": float(template.charge_state * elementary_charge),
        "collision_cross_section_m2": float(template.collision_cross_section_m2),
        "collision_cross_section_nm2": float(template.collision_cross_section_m2 * 1.0e18),
        "num_atoms": int(template.num_atoms),
        "initial_internal_temperature_k": float(template.initial_internal_temperature_k),
        "initial_position_m": np.asarray(template.initial_position_m, dtype=float),
        "initial_velocity_m_per_s": np.asarray(template.initial_velocity_m_per_s, dtype=float),
        "initial_speed_m_per_s": speed_m_per_s,
        "initial_kinetic_energy_ev": float(
            0.5 * template.mass_kg * speed_m_per_s**2 / elementary_charge
        ),
        "heat_capacity_profile": template.heat_capacity_profile,
        "delta_h_kj_per_mol": float(template.delta_h_kj_per_mol),
        "delta_s_j_per_mol_k": float(template.delta_s_j_per_mol_k),
    }


def _base_config_summary(config: SimulationConfig) -> dict[str, Any]:
    return {
        "ion_count": int(config.ion_count),
        "total_time_s": float(config.total_time_s),
        "macro_time_step_s": float(config.macro_time_step_s),
        "random_seed": int(config.random_seed),
        "domain_radius_m": float(config.domain_radius_m),
        "domain_length_m": float(config.domain_length_m),
        "grid_nr": int(config.grid_nr),
        "grid_nz": int(config.grid_nz),
        "pic_grid_nr": None if config.pic_grid_nr is None else int(config.pic_grid_nr),
        "pic_grid_nz": None if config.pic_grid_nz is None else int(config.pic_grid_nz),
        "pic_space_charge_scale": float(config.pic_space_charge_scale),
        "rf_frequency_hz": float(config.rf_frequency_hz),
        "rf_peak_voltage_v": float(config.rf_peak_voltage_v),
        "rf_phase_rad": float(config.rf_phase_rad),
        "rf_reference_peak_voltage_v": (
            None
            if config.rf_reference_peak_voltage_v is None
            else float(config.rf_reference_peak_voltage_v)
        ),
        "initial_position_jitter_m": float(config.initial_position_jitter_m),
        "initial_velocity_jitter_m_per_s": float(
            config.initial_velocity_jitter_m_per_s
        ),
        "source_radius_m": (
            None if config.source_radius_m is None else float(config.source_radius_m)
        ),
        "source_profile": str(config.source_profile),
    }


def _source_config_summary(config: SimulationConfig) -> dict[str, Any]:
    return {
        "source_gaussian_sigma_m": (
            None
            if config.source_gaussian_sigma_m is None
            else float(config.source_gaussian_sigma_m)
        ),
        "source_mode": config.source_mode,
        "collision_operator": str(config.collision_model),
        "ion_current_a": float(config.ion_current_a),
        "source_mach_number": float(config.source_mach_number),
        "source_gas_gamma": float(config.source_gas_gamma),
        "source_gas_molar_mass_kg_per_mol": float(
            config.source_gas_molar_mass_kg_per_mol
        ),
        "source_temperature_k": (
            None
            if config.source_temperature_k is None
            else float(config.source_temperature_k)
        ),
        "source_axial_velocity_m_per_s": (
            None
            if config.source_axial_velocity_m_per_s is None
            else float(config.source_axial_velocity_m_per_s)
        ),
        "source_velocity_from_gas_field": bool(config.source_velocity_from_gas_field),
        "source_birth_velocity_gas_path": config.source_birth_velocity_gas_path,
        "source_birth_velocity_z_min_m": float(config.source_birth_velocity_z_min_m),
        "source_birth_velocity_z_max_m": (
            None
            if config.source_birth_velocity_z_max_m is None
            else float(config.source_birth_velocity_z_max_m)
        ),
        "source_birth_velocity_radius_m": (
            None
            if config.source_birth_velocity_radius_m is None
            else float(config.source_birth_velocity_radius_m)
        ),
        "source_radial_velocity_scale": float(config.source_radial_velocity_scale),
        "source_velocity_delta_m_per_s": float(config.source_velocity_delta_m_per_s),
    }


def _injection_config_summary(config: SimulationConfig) -> dict[str, Any]:
    return {
        "capillary_exit_z_m": (
            None if config.capillary_exit_z_m is None else float(config.capillary_exit_z_m)
        ),
        "capillary_voltage_v": float(config.capillary_voltage_v),
        "capillary_prefill_length_m": float(config.capillary_prefill_length_m),
        "capillary_prefill_macro_particles": int(
            config.capillary_prefill_macro_particles
        ),
        "macro_particles_per_injection": int(config.macro_particles_per_injection),
        "max_macro_particle_weight": float(config.max_macro_particle_weight),
        "cone_half_angle_rad": float(config.cone_half_angle_rad),
        "macro_particle_weight": float(config.macro_particle_weight),
    }


def _solver_config_summary(config: SimulationConfig) -> dict[str, Any]:
    return {
        "sor_omega": float(config.sor_omega),
        "sor_max_iters": int(config.sor_max_iters),
        "sor_tolerance": float(config.sor_tolerance),
        "pic_poisson_backend": str(config.pic_poisson_backend),
        "pic_poisson_preconditioner": str(config.pic_poisson_preconditioner),
        "pic_poisson_warm_start": bool(config.pic_poisson_warm_start),
        "pic_amg_mode": str(config.pic_amg_mode),
        "pic_amg_solver": str(config.pic_amg_solver),
        "pic_amg_tolerance": float(config.pic_amg_tolerance),
        "pic_amg_max_iters": int(config.pic_amg_max_iters),
        "pic_amg_fallback_backend": str(config.pic_amg_fallback_backend),
        "static_field_path": config.static_field_path,
        "stage_schedule_path": config.stage_schedule_path,
    }


def _runtime_config_summary(config: SimulationConfig) -> dict[str, Any]:
    return {
        "static_field_3d_path": config.static_field_3d_path,
        "detector_z_m": (
            None if config.detector_z_m is None else float(config.detector_z_m)
        ),
        "detector_radius_m": (
            None if config.detector_radius_m is None else float(config.detector_radius_m)
        ),
        "radial_limit_m": (
            None if config.radial_limit_m is None else float(config.radial_limit_m)
        ),
        "stop_active_fraction_below": float(config.stop_active_fraction_below),
        "stop_stable_window_steps": int(config.stop_stable_window_steps),
        "stop_stable_fraction_tol": float(config.stop_stable_fraction_tol),
        "max_wall_time_s": float(config.max_wall_time_s),
        "terminal_event_mode": str(config.terminal_event_mode),
        "max_terminal_event_rows": int(config.max_terminal_event_rows),
        "collision_batch_size": int(config.collision_batch_size),
        "collision_preselection_enabled": bool(
            config.collision_preselection_enabled
        ),
        "collision_physics_backend": str(config.collision_physics_backend),
        "ionspa_backend": str(config.ionspa_backend),
        "collision_model": str(config.collision_model),
        "fragmentation_mode": str(config.fragmentation_mode),
        "langevin_z_start_m": float(config.langevin_z_start_m),
        "langevin_z_end_m": float(config.langevin_z_end_m),
        "langevin_switch_probability": float(config.langevin_switch_probability),
        "langevin_max_dt_s": float(config.langevin_max_dt_s),
        "electrode_mask_path": config.electrode_mask_path,
        "electrode_mask_cache_path": config.electrode_mask_cache_path,
        "electrode_mask_z_offset_m": (
            None
            if config.electrode_mask_z_offset_m is None
            else float(config.electrode_mask_z_offset_m)
        ),
        "electrode_mask_3d_path": config.electrode_mask_3d_path,
        "electrode_hit_distance_m": float(config.electrode_hit_distance_m),
    }


def _iict_config_summary(config: SimulationConfig) -> dict[str, Any]:
    return {
        "iict_parameter_config_path": config.iict_parameter_config_path,
        "iict_heat_capacity_model": config.iict_heat_capacity_model,
        "iict_num_atoms": config.iict_num_atoms,
        "iict_constant_cv_j_per_k_per_ion": (
            config.iict_constant_cv_j_per_k_per_ion
        ),
        "iict_heat_capacity_csv_path": config.iict_heat_capacity_csv_path,
        "iict_pseudoatom_model": config.iict_pseudoatom_model,
        "iict_pseudoatom_mass_da": config.iict_pseudoatom_mass_da,
        "iict_pseudoatom_csv_path": config.iict_pseudoatom_csv_path,
        "iict_pseudoatom_min_mass_da": config.iict_pseudoatom_min_mass_da,
        "iict_pseudoatom_max_mass_da": config.iict_pseudoatom_max_mass_da,
        "iict_fragmentation_model": config.iict_fragmentation_model,
        "iict_delta_h_kj_per_mol": config.iict_delta_h_kj_per_mol,
        "iict_delta_s_j_per_mol_k": config.iict_delta_s_j_per_mol_k,
    }


def _config_summary(config: SimulationConfig) -> dict[str, Any]:
    summary = _base_config_summary(config)
    summary.update(_source_config_summary(config))
    summary.update(_injection_config_summary(config))
    summary.update(_solver_config_summary(config))
    summary.update(_runtime_config_summary(config))
    summary.update(_iict_config_summary(config))
    return summary
