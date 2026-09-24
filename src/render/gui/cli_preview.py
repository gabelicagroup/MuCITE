"""Stable CLI argument preview generated from GUI configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import AppConfig


@dataclass
class _ArgumentBuilder:
    args: list[str] = field(default_factory=list)

    def add(self, *items: object) -> None:
        self.args.extend(str(item) for item in items)

    def add_optional(self, flag: str, value: object) -> None:
        if value is not None and str(value).strip() != "":
            self.add(flag, value)


def _add_launch_arguments(builder: _ArgumentBuilder, app_config: AppConfig) -> None:
    beam = app_config.beam
    runtime = app_config.runtime
    builder.add(
        "-m",
        "src",
        "--backend",
        runtime.backend,
        "--no-window",
        "--static-field",
        app_config.loaded_baked_field_path,
        "--source-mode",
        beam.source_mode,
        "--ion-count",
        int(beam.particle_count),
        "--ion-current-a",
        beam.current_a,
        "--random-seed",
        int(runtime.random_seed),
        "--collision-model",
        runtime.collision_mode,
        "--total-time",
        runtime.total_time_s,
        "--source-profile",
        beam.source_profile,
        "--initial-x-mm",
        beam.initial_x_mm,
        "--initial-y-mm",
        beam.initial_y_mm,
        "--initial-z-mm",
        beam.initial_z_mm,
    )
    _add_ion_arguments(builder, app_config)


def _add_collision_physics_arguments(
    builder: _ArgumentBuilder,
    app_config: AppConfig,
) -> None:
    runtime = app_config.runtime
    backend = str(runtime.collision_physics_backend)
    builder.add("--collision-physics-backend", backend)
    if backend == "ionspa":
        if runtime.ionspa_backend != "bundled":
            builder.add("--ionspa-backend", runtime.ionspa_backend)
        return
    optional_arguments = (
        ("--iict-parameter-config", runtime.iict_parameter_config_path),
        ("--iict-heat-capacity-model", runtime.iict_heat_capacity_model),
        ("--iict-num-atoms", runtime.iict_num_atoms),
        (
            "--iict-constant-cv-j-per-k-per-ion",
            runtime.iict_constant_cv_j_per_k_per_ion,
        ),
        ("--iict-heat-capacity-csv", runtime.iict_heat_capacity_csv_path),
        ("--iict-pseudoatom-model", runtime.iict_pseudoatom_model),
        ("--iict-pseudoatom-mass-da", runtime.iict_pseudoatom_mass_da),
        ("--iict-pseudoatom-csv", runtime.iict_pseudoatom_csv_path),
        (
            "--iict-pseudoatom-min-mass-da",
            runtime.iict_pseudoatom_min_mass_da,
        ),
        (
            "--iict-pseudoatom-max-mass-da",
            runtime.iict_pseudoatom_max_mass_da,
        ),
        ("--iict-fragmentation-model", runtime.iict_fragmentation_model),
        ("--iict-delta-h-kj-per-mol", runtime.iict_delta_h_kj_per_mol),
        ("--iict-delta-s-j-per-mol-k", runtime.iict_delta_s_j_per_mol_k),
    )
    for flag, value in optional_arguments:
        builder.add_optional(flag, value)


def _add_ion_arguments(builder: _ArgumentBuilder, app_config: AppConfig) -> None:
    beam = app_config.beam
    runtime = app_config.runtime
    builder.add(
        "--initial-internal-temperature-k",
        beam.initial_internal_temperature_k,
        "--ion-name",
        beam.ion_name,
        "--num-atoms",
        int(beam.num_atoms),
        "--heat-capacity-profile",
        beam.heat_capacity_profile,
        "--delta-h-kj-per-mol",
        beam.delta_h_kj_per_mol,
        "--delta-s-j-per-mol-k",
        beam.delta_s_j_per_mol_k,
        "--capillary-exit-z-mm",
        beam.capillary_exit_z_mm,
        "--capillary-prefill-length-mm",
        beam.capillary_prefill_length_mm,
        "--capillary-prefill-macro-particles",
        int(runtime.capillary_prefill_macro_particles),
    )
    if str(beam.gas_velocity_init_mode).strip().lower() not in {
        "from-gas-field",
    }:
        builder.args.append(f"--initial-direction-axis={beam.direction_axis}")


def _add_source_arguments(builder: _ArgumentBuilder, app_config: AppConfig) -> None:
    beam = app_config.beam
    gas_mode = str(beam.gas_velocity_init_mode).strip().lower() == "from-gas-field"
    if float(beam.beam_radius_mm) > 0.0:
        builder.add("--source-radius-mm", beam.beam_radius_mm)
    if str(beam.source_profile).strip().lower() == "gaussian":
        builder.add(
            "--source-gaussian-sigma-mm",
            beam.source_gaussian_sigma_mm,
        )
    if gas_mode:
        builder.args.append("--source-velocity-from-gas-field")
        builder.add(
            "--source-birth-velocity-gas-csv",
            beam.source_birth_velocity_gas_csv,
            "--source-birth-velocity-z-min-mm",
            beam.source_birth_velocity_z_min_mm,
        )
        builder.add_optional(
            "--source-birth-velocity-z-max-mm",
            beam.source_birth_velocity_z_max_mm,
        )
        builder.add_optional(
            "--source-birth-velocity-radius-mm",
            beam.source_birth_velocity_radius_mm,
        )
    else:
        builder.add_optional("--initial-kinetic-energy-ev", beam.kinetic_energy_ev)
        builder.add_optional("--source-temperature-k", beam.source_temperature_k)
        builder.add_optional(
            "--source-axial-velocity-m-per-s",
            beam.source_axial_velocity_m_per_s,
        )


def _add_beam_transport_arguments(
    builder: _ArgumentBuilder,
    app_config: AppConfig,
) -> None:
    beam = app_config.beam
    runtime = app_config.runtime
    gas_mode = str(beam.gas_velocity_init_mode).strip().lower() == "from-gas-field"
    builder.add(
        "--source-radial-velocity-scale",
        beam.source_radial_velocity_scale,
        "--source-velocity-delta-m-per-s",
        beam.source_velocity_delta_m_per_s if gas_mode else 0.0,
        "--capillary-voltage-v",
        runtime.capillary_voltage_v,
        "--macro-particles-per-injection",
        int(runtime.macro_particles_per_injection),
        "--max-macro-particle-weight",
        runtime.max_macro_particle_weight,
        "--macro-particle-weight",
        beam.macro_particle_weight,
        "--ion-mass-amu",
        beam.mass_amu,
        "--charge-state",
        int(beam.charge_e),
        "--collision-cross-section-m2",
        beam.collision_cross_section_m2,
        "--initial-position-jitter-mm",
        beam.initial_position_jitter_mm,
        "--initial-velocity-jitter-m-per-s",
        0.0 if gas_mode else beam.velocity_jitter_m_per_s,
        "--cone-half-angle-deg",
        0.0 if gas_mode else beam.cone_half_angle_deg,
    )


def _add_solver_arguments(builder: _ArgumentBuilder, app_config: AppConfig) -> None:
    runtime = app_config.runtime
    builder.add(
        "--rf-peak-voltage",
        runtime.rf_peak_voltage_v,
        "--rf-frequency",
        runtime.rf_frequency_hz,
        "--rf-phase-deg",
        runtime.rf_phase_deg,
        "--macro-time-step",
        runtime.macro_dt_s,
        "--pic-grid-nr",
        int(runtime.pic_grid_nr),
        "--pic-grid-nz",
        int(runtime.pic_grid_nz),
        "--pic-space-charge-scale",
        runtime.pic_space_charge_scale,
        "--pic-poisson-backend",
        runtime.pic_poisson_backend,
        "--pic-poisson-preconditioner",
        runtime.pic_poisson_preconditioner,
        "--pic-amg-mode",
        runtime.pic_amg_mode,
        "--pic-amg-solver",
        runtime.pic_amg_solver,
        "--pic-amg-tolerance",
        runtime.pic_amg_tolerance,
        "--pic-amg-max-iters",
        int(runtime.pic_amg_max_iters),
        "--pic-amg-fallback-backend",
        runtime.pic_amg_fallback_backend,
        "--stop-active-fraction-below",
        runtime.stop_active_fraction_below,
        "--stop-stable-window",
        int(runtime.stop_stable_window_steps),
        "--stop-stable-fraction-tol",
        runtime.stop_stable_fraction_tol,
    )
    builder.args.append(
        "--pic-poisson-warm-start"
        if runtime.pic_poisson_warm_start
        else "--no-pic-poisson-warm-start"
    )


def _add_domain_arguments(builder: _ArgumentBuilder, app_config: AppConfig) -> None:
    runtime = app_config.runtime
    if float(runtime.detector_z_mm) > 0.0:
        builder.add("--detector-z-mm", runtime.detector_z_mm)
    if float(runtime.detector_radius_mm) > 0.0:
        builder.add("--detector-radius-mm", runtime.detector_radius_mm)
    if float(runtime.radial_limit_mm) > 0.0:
        builder.add("--radial-limit-mm", runtime.radial_limit_mm)
    builder.add_optional("--static-field-3d", runtime.static_field_3d_path)
    preview_mask_text = str(runtime.electrode_mask_path).strip()
    if not preview_mask_text or (
        runtime.dummy_static_field
        and preview_mask_text.lower() == "default"
    ):
        builder.args.append("--no-electrode-mask")
    else:
        builder.add("--electrode-mask", runtime.electrode_mask_path)
    if runtime.dummy_static_field:
        builder.args.append("--dummy-static-field")
    builder.add_optional(
        "--electrode-mask-cache",
        runtime.electrode_mask_cache_path,
    )
    builder.add_optional(
        "--electrode-mask-z-offset-mm",
        runtime.electrode_mask_z_offset_mm,
    )


def _add_runtime_policy_arguments(
    builder: _ArgumentBuilder,
    app_config: AppConfig,
) -> None:
    runtime = app_config.runtime
    builder.add(
        "--electrode-hit-distance-mm",
        runtime.electrode_hit_distance_mm,
        "--terminal-event-mode",
        runtime.terminal_event_mode,
        "--max-terminal-event-rows",
        int(runtime.max_terminal_event_rows),
        "--collision-batch-size",
        int(runtime.collision_batch_size),
        "--fragmentation-mode",
        runtime.fragmentation_mode,
        "--langevin-z-start-mm",
        runtime.langevin_z_start_mm,
        "--langevin-z-end-mm",
        runtime.langevin_z_end_mm,
        "--langevin-switch-prob",
        runtime.langevin_switch_prob,
        "--langevin-max-dt-s",
        runtime.langevin_max_dt_s,
        "--max-wall-time-s",
        runtime.max_wall_time_s,
    )


def _add_output_arguments(
    builder: _ArgumentBuilder,
    app_config: AppConfig,
    export_every: int,
) -> None:
    output = app_config.output
    builder.add(
        "--export-every",
        export_every,
        "--trajectory-sample-count",
        int(output.trajectory_sample_count),
        "--trajectory-record-every",
        int(output.trajectory_record_every),
        "--snapshot-plot-max-points",
        int(output.snapshot_plot_max_points),
        "--export-dir",
        app_config.output_dir,
        "--report-dir",
        app_config.output_dir,
    )
    if output.save_h5:
        builder.add("--export-format", "h5")
    if not output.save_figures:
        builder.args.append("--no-snapshot-plots")
    if output.generate_report_after_run:
        builder.args.append("--report")


def gui_run_arguments(app_config: AppConfig) -> list[str]:
    """Return a CLI-style argument preview for the GUI simulation config."""

    output = app_config.output
    export_every = (
        int(app_config.runtime.snapshot_every)
        if output.save_snapshots
        else 0
    )
    builder = _ArgumentBuilder()
    _add_launch_arguments(builder, app_config)
    _add_collision_physics_arguments(builder, app_config)
    _add_source_arguments(builder, app_config)
    _add_beam_transport_arguments(builder, app_config)
    _add_solver_arguments(builder, app_config)
    _add_domain_arguments(builder, app_config)
    _add_runtime_policy_arguments(builder, app_config)
    _add_output_arguments(builder, app_config, export_every)
    return builder.args
