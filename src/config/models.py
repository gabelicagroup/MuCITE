"""Immutable user-request models for ions and coupled simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import ATOMIC_MASS_CONSTANT, N2_MOLAR_MASS_KG_PER_MOL


@dataclass(frozen=True)
class ExecutionConfig:
    """Process/UI controls that do not change the physical request."""

    backend: str = "cpu"
    gui: bool = False
    live_window: bool = False
    no_window: bool = False
    progress: bool = False
    progress_interval_s: float = 0.25


@dataclass(frozen=True)
class OutputConfig:
    """Snapshot and report policy kept outside the physical request."""

    export_every: int = 0
    export_dir: Optional[Path] = None
    export_format: str = "npy"
    trajectory_sample_count: int = 200
    trajectory_record_every: int = 1
    snapshot_plot_max_points: int = 200_000
    snapshot_plots_enabled: bool = True
    dt_quantile_reservoir_size: int = 8_192
    macro_history_max_rows: int = 4_096
    terminal_event_sample_rows: int = 50_000
    terminal_event_sample_seed: int = 0
    report_plot_max_points: int = 200_000
    trajectory_plot_max_tracks: int = 50
    report: bool = False
    report_dir: Optional[Path] = None


@dataclass(frozen=True)
class IonTemplate:
    """Static ion properties used to initialize a population."""

    name: str = "protein_ion"
    mass_kg: float = 5000.0 * ATOMIC_MASS_CONSTANT
    charge_state: int = 10
    collision_cross_section_m2: float = 12.0e-18
    num_atoms: int = 1200
    initial_internal_temperature_k: float = 300.0
    heat_capacity_profile: str = "peptide"
    delta_h_kj_per_mol: float = 80.0
    delta_s_j_per_mol_k: float = -120.0
    initial_position_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    initial_velocity_m_per_s: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 250.0], dtype=float)
    )


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level control parameters for the macro/micro stepping scheme.

    ``pic_grid_nr/pic_grid_nz`` deliberately preserve three request forms:
    ``None/None`` means default shared static/PIC storage, ``0/0`` explicitly
    requests shared storage, and a positive pair requests exact PIC node counts.
    Runtime field loading resolves either shared form only after the static mesh
    dimensions are known.

    ``rf_peak_voltage_v`` is a single-phase peak voltage (Vpeak), not Vpp.
    """

    ion_count: int = 32
    total_time_s: float = 8.0e-5
    macro_time_step_s: float = 5.0e-8
    collision_safety_factor: float = 0.1
    rf_safety_factor: float = 0.05
    acceleration_safety_factor: float = 0.2
    spatial_tolerance_m: float = 5.0e-5
    min_time_step_s: float = 1.0e-12
    max_time_step_s: float = 5.0e-8
    advection_cfl_fraction: float = 0.5
    random_seed: int = 7
    domain_radius_m: float = 8.0e-3
    domain_length_m: float = 25.0e-3
    rf_frequency_hz: float = 1.0e6
    rf_peak_voltage_v: float = 50.0
    rf_phase_rad: float = 0.0
    rf_reference_peak_voltage_v: Optional[float] = None
    initial_position_jitter_m: float = 2.0e-4
    initial_velocity_jitter_m_per_s: float = 25.0
    source_radius_m: Optional[float] = None
    source_profile: str = "uniform-disk"
    source_gaussian_sigma_m: Optional[float] = None
    source_gaussian_max_attempts: int = 1_000
    source_gaussian_min_batch: int = 1_024
    source_gaussian_oversample_factor: int = 2
    source_mode: str = "packet"
    ion_current_a: float = 0.0
    source_mach_number: float = 1.0
    source_gas_gamma: float = 1.4
    source_gas_molar_mass_kg_per_mol: float = N2_MOLAR_MASS_KG_PER_MOL
    source_temperature_k: Optional[float] = None
    source_axial_velocity_m_per_s: Optional[float] = None
    source_velocity_from_gas_field: bool = False
    source_birth_velocity_gas_path: Optional[Path] = None
    source_birth_velocity_z_min_m: float = 0.0
    source_birth_velocity_z_max_m: Optional[float] = None
    source_birth_velocity_radius_m: Optional[float] = None
    source_reference_velocity_sample_count: int = 101
    source_radial_velocity_scale: float = 1.0
    source_velocity_delta_m_per_s: float = 0.0
    capillary_exit_z_m: Optional[float] = None
    capillary_voltage_v: float = 0.0
    capillary_prefill_length_m: float = 0.0
    capillary_prefill_macro_particles: int = 0
    macro_particles_per_injection: int = 5
    max_macro_particle_weight: float = 0.0
    cone_half_angle_rad: float = 0.0
    grid_nr: int = 96
    grid_nz: int = 256
    pic_grid_nr: Optional[int] = None
    pic_grid_nz: Optional[int] = None
    pic_space_charge_scale: float = 1.0
    macro_particle_weight: float = 1.0
    sor_omega: float = 1.7
    sor_max_iters: int = 200
    sor_tolerance: float = 1.0e-6
    pic_poisson_backend: str = "sor"
    pic_poisson_preconditioner: str = "none"
    pic_poisson_warm_start: bool = True
    pic_amg_mode: str = "solve"
    pic_amg_solver: str = "ruge_stuben"
    pic_amg_tolerance: float = 1.0e-3
    pic_amg_max_iters: int = 50
    pic_amg_fallback_backend: str = "sparse_bicgstab"
    static_field_path: Optional[Path] = None
    stage_schedule_path: Optional[Path] = None
    static_field_3d_path: Optional[Path] = None
    detector_z_m: Optional[float] = None
    detector_radius_m: Optional[float] = None
    radial_limit_m: Optional[float] = None
    stop_active_fraction_below: float = 0.0
    stop_stable_window_steps: int = 0
    stop_stable_fraction_tol: float = 0.0
    max_wall_time_s: float = 0.0
    electrode_mask_path: Optional[Path] = None
    electrode_mask_cache_path: Optional[Path] = None
    electrode_mask_z_offset_m: Optional[float] = None
    electrode_mask_3d_path: Optional[Path] = None
    electrode_hit_distance_m: float = 0.0
    electrode_sweep_spacing_fraction: float = 0.25
    electrode_hit_bisection_iterations: int = 24
    terminal_event_mode: str = "transport-only"
    max_terminal_event_rows: int = 1_000_000
    collision_batch_size: int = 20_000
    collision_preselection_enabled: bool = True
    collision_physics_backend: str = "ionspa"
    ionspa_backend: str = "bundled"
    iict_parameter_config_path: Optional[Path] = None
    iict_heat_capacity_model: Optional[str] = None
    iict_num_atoms: Optional[int] = None
    iict_constant_cv_j_per_k_per_ion: Optional[float] = None
    iict_heat_capacity_csv_path: Optional[Path] = None
    iict_pseudoatom_model: Optional[str] = None
    iict_pseudoatom_mass_da: Optional[float] = None
    iict_pseudoatom_csv_path: Optional[Path] = None
    iict_pseudoatom_min_mass_da: Optional[float] = None
    iict_pseudoatom_max_mass_da: Optional[float] = None
    iict_fragmentation_model: Optional[str] = None
    iict_delta_h_kj_per_mol: Optional[float] = None
    iict_delta_s_j_per_mol_k: Optional[float] = None
    iict_cli_override_fields: tuple[str, ...] = ()
    collision_model: str = "explicit"
    fragmentation_mode: str = "transport"
    gas_species: str = "n2"
    gas_gamma: float = 1.4
    gas_molar_mass_kg_per_mol: float = N2_MOLAR_MASS_KG_PER_MOL
    langevin_z_start_m: float = 4.5e-3
    langevin_z_end_m: float = 6.0e-3
    langevin_switch_probability: float = 0.05
    langevin_max_dt_s: float = 1.0e-9
