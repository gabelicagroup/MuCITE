"""Presentation configuration containers for the MuCITE GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ...config.release_defaults import parameter_example_path


DEFAULT_GUI_STATIC_FIELD_PATH = ""
DEFAULT_GUI_BIRTH_GAS_CSV = ""
DEFAULT_GUI_ELECTRODE_MASK_CACHE = ""

DEFAULT_BEAM_MASS_AMU = 2000.0


def default_project_name() -> str:
    return "mucite_project_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def default_output_dir() -> str:
    return str(Path("outputs") / "mucite_project")


@dataclass
class BeamConfig:
    """Beam and ion-source inputs shown in the beam setup dialog."""

    ion_name: str = "demonstration_ion"
    mass_amu: float = DEFAULT_BEAM_MASS_AMU
    charge_e: int = 5
    particle_count: int = 8
    source_mode: str = "packet"
    macro_particle_weight: float = 1.0
    current_a: float = 1.0e-9
    kinetic_energy_ev: Optional[float] = None
    initial_internal_temperature_k: float = 300.0
    collision_cross_section_m2: float = 2.0e-18
    num_atoms: int = 72
    heat_capacity_profile: str = "peptide"
    delta_h_kj_per_mol: float = 80.0
    delta_s_j_per_mol_k: float = -120.0
    beam_radius_mm: float = 0.15
    initial_x_mm: float = 0.0
    initial_y_mm: float = 0.0
    initial_z_mm: float = 4.5
    initial_position_jitter_mm: float = 0.0
    velocity_jitter_m_per_s: float = 0.0
    direction_axis: str = "+z"
    cone_half_angle_deg: float = 0.0
    source_profile: str = "gaussian"
    source_gaussian_sigma_mm: float = 0.05
    gas_velocity_init_mode: str = "static"
    source_temperature_k: Optional[float] = 300.0
    source_axial_velocity_m_per_s: Optional[float] = 250.0
    source_birth_velocity_gas_csv: str = DEFAULT_GUI_BIRTH_GAS_CSV
    source_birth_velocity_z_min_mm: float = 0.0
    source_birth_velocity_z_max_mm: Optional[float] = None
    source_birth_velocity_radius_mm: Optional[float] = None
    source_radial_velocity_scale: float = 1.0
    source_velocity_delta_m_per_s: float = 0.0
    capillary_exit_z_mm: float = 4.5
    capillary_prefill_length_mm: float = 0.0
    notes: str = ""


@dataclass
class FieldBakeConfig:
    """Offline field-baking inputs exposed by the field baker dialog."""

    simion_dc_path: str = str(Path("E_field") / "slens" / "slens100x_dc.patxt")
    simion_rf_path: str = str(Path("E_field") / "slens" / "slens100x_rf.patxt")
    gas_field_mode: str = "static"
    fluent_path: str = ""
    coordinate_mapping: str = "simion_x_to_z_y_to_r"
    offset_z_simion_mm: float = 0.0
    offset_z_fluent_mm: float = -95.0
    pa_grids_per_mm: float = 100.0
    z_min_mm: float = 0.0
    z_max_mm: float = 65.0
    r_min_mm: float = 0.0
    r_max_mm: float = 20.0
    dz_mm: float = 0.1
    dr_mm: float = 0.1
    dc_voltage_scale: float = 1.0
    background_pressure_pa: float = 200.0
    background_temperature_k: float = 300.0
    capillary_exit_z_mm: float = 6.0
    fluent_capillary_total_length_mm: float = 101.0
    fluent_capillary_radius_mm: float = 0.25
    fluent_capillary_external_start_z_mm: float = 4.5
    phi_key_for_plot: str = "phi_dc_v"
    output_npy: str = str(Path("outputs") / "mucite_project" / "baked_fields.npy")
    output_plot_dir: str = str(Path("outputs") / "mucite_project" / "field_baker_plots")


@dataclass
class RuntimeConfig:
    """Main simulation controls exposed in the runtime panel."""

    backend: str = "taichi"
    total_time_s: float = 1.0e-9
    macro_dt_s: float = 5.0e-8
    random_seed: int = 7
    snapshot_every: int = 1000
    pic_space_charge_scale: float = 1.0
    collision_physics_backend: str = "iict-lite"
    ionspa_backend: str = "local"
    iict_parameter_config_path: str = field(
        default_factory=lambda: str(parameter_example_path())
    )
    iict_heat_capacity_model: Optional[str] = None
    iict_num_atoms: Optional[int] = None
    iict_constant_cv_j_per_k_per_ion: Optional[float] = None
    iict_heat_capacity_csv_path: str = ""
    iict_pseudoatom_model: Optional[str] = None
    iict_pseudoatom_mass_da: Optional[float] = None
    iict_pseudoatom_csv_path: str = ""
    iict_pseudoatom_min_mass_da: Optional[float] = None
    iict_pseudoatom_max_mass_da: Optional[float] = None
    iict_fragmentation_model: Optional[str] = None
    iict_delta_h_kj_per_mol: Optional[float] = None
    iict_delta_s_j_per_mol_k: Optional[float] = None
    collision_mode: str = "explicit"
    fragmentation_mode: str = "off"
    rf_frequency_hz: float = 6.5e5
    rf_peak_voltage_v: float = 100.0
    rf_phase_deg: float = -90.0
    pic_grid_nr: int = 0
    pic_grid_nz: int = 0
    pic_poisson_backend: str = "amg"
    pic_poisson_preconditioner: str = "none"
    pic_poisson_warm_start: bool = True
    pic_amg_mode: str = "preconditioned_cg"
    pic_amg_solver: str = "ruge_stuben"
    pic_amg_tolerance: float = 1.0e-6
    pic_amg_max_iters: int = 150
    pic_amg_fallback_backend: str = "sparse_direct"
    detector_z_mm: float = 65.0
    detector_radius_mm: float = 20.0
    radial_limit_mm: float = 20.0
    capillary_voltage_v: float = 0.0
    capillary_prefill_macro_particles: int = 0
    macro_particles_per_injection: int = 1
    max_macro_particle_weight: float = 500.0
    stop_active_fraction_below: float = 0.0
    stop_stable_window_steps: int = 0
    stop_stable_fraction_tol: float = 0.0
    max_wall_time_s: float = 604800.0
    terminal_event_mode: str = "transport-only"
    max_terminal_event_rows: int = 0
    collision_batch_size: int = 20_000
    langevin_z_start_mm: float = 4.5
    langevin_z_end_mm: float = 6.0
    langevin_switch_prob: float = 0.05
    langevin_max_dt_s: float = 1.0e-9
    electrode_mask_path: str = ""
    electrode_mask_cache_path: str = str(DEFAULT_GUI_ELECTRODE_MASK_CACHE)
    electrode_mask_z_offset_mm: Optional[float] = None
    electrode_hit_distance_mm: float = 0.01
    dummy_static_field: bool = True
    static_field_3d_path: str = ""


@dataclass
class OutputConfig:
    """Output policy for snapshots, reports, and plots."""

    save_snapshots: bool = True
    save_csv: bool = True
    save_h5: bool = False
    save_figures: bool = True
    trajectory_sample_count: int = 200
    trajectory_record_every: int = 1
    snapshot_plot_max_points: int = 200000
    terminal_time_bin_ms: float = 0.2
    report_format: str = "markdown"
    generate_report_after_run: bool = True


@dataclass
class SessionState:
    """Small runtime state snapshot used by status labels."""

    session_id: str = field(default_factory=default_project_name)
    current_status: str = "CONFIGURED"
    current_macro_step: int = 0
    current_micro_step: int = 0
    current_time_s: float = 0.0
    alive_count: int = 0
    collision_count: int = 0
    mean_energy_ev: float = 0.0
    mean_temperature_k: float = 0.0
    latest_snapshot_path: str = ""
    loaded_baked_field_path: str = str(DEFAULT_GUI_STATIC_FIELD_PATH)
    last_warning: str = ""


@dataclass
class AppConfig:
    """Top-level GUI project configuration saved to JSON."""

    session_name: str = field(default_factory=default_project_name)
    output_dir: str = field(default_factory=default_output_dir)
    loaded_baked_field_path: str = str(DEFAULT_GUI_STATIC_FIELD_PATH)
    beam: BeamConfig = field(default_factory=BeamConfig)
    field_bake: FieldBakeConfig = field(default_factory=FieldBakeConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    state: SessionState = field(default_factory=SessionState)
