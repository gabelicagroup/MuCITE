"""Configuration synchronization and text state for the Tk main window."""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path

from .iict_dialog import iict_parameter_summary
from .models import OutputConfig, RuntimeConfig


MAX_CONSOLE_LINES = 2_000


class _WindowStateMixin:
    def _append_console(self, tab_name: str, line: str) -> None:
        text = self.console_tabs.get(tab_name, self.console_tabs["Logs"])
        text.configure(state=tk.NORMAL)
        text.insert(tk.END, line.rstrip() + "\n")
        last_line = int(str(text.index("end-1c")).split(".", 1)[0])
        overflow = last_line - MAX_CONSOLE_LINES
        if overflow > 0:
            text.delete("1.0", f"{overflow + 1}.0")
        text.see(tk.END)
        text.configure(state=tk.DISABLED)

    def _set_status(self, status: str) -> None:
        self.status_var.set(status)
        self.config.state.current_status = status

    def _apply_config_to_vars(self) -> None:
        runtime = self.config.runtime
        output = self.config.output
        self._apply_project_vars(runtime)
        self._apply_runtime_vars(runtime)
        self._apply_output_vars(output)
        self._set_status(self.config.state.current_status)
        self._update_summaries()

    def _apply_project_vars(self, runtime: RuntimeConfig) -> None:
        self.session_name_var.set(self.config.session_name)
        self.output_dir_var.set(self.config.output_dir)
        self.static_field_var.set(self.config.loaded_baked_field_path)
        self.backend_var.set(runtime.backend)
        self.dummy_static_field_var.set(runtime.dummy_static_field)

    def _apply_runtime_vars(self, runtime: RuntimeConfig) -> None:
        self._apply_runtime_core_vars(runtime)
        self._apply_runtime_policy_vars(runtime)

    def _apply_runtime_core_vars(self, runtime: RuntimeConfig) -> None:
        self.total_time_var.set(str(runtime.total_time_s))
        self.macro_dt_var.set(str(runtime.macro_dt_s))
        self.random_seed_var.set(str(runtime.random_seed))
        self.snapshot_every_var.set(str(runtime.snapshot_every))
        self.pic_space_charge_scale_var.set(str(runtime.pic_space_charge_scale))
        self.collision_mode_var.set(runtime.collision_mode)
        self.rf_frequency_var.set(str(runtime.rf_frequency_hz))
        self.rf_peak_voltage_var.set(str(runtime.rf_peak_voltage_v))
        self.rf_phase_var.set(str(runtime.rf_phase_deg))
        self.detector_z_var.set(str(runtime.detector_z_mm))
        self.detector_radius_var.set(str(runtime.detector_radius_mm))
        self.radial_limit_var.set(str(runtime.radial_limit_mm))
        self.capillary_voltage_var.set(str(runtime.capillary_voltage_v))
        self.capillary_prefill_macro_particles_var.set(
            str(runtime.capillary_prefill_macro_particles)
        )
        self.macro_particles_per_injection_var.set(
            str(runtime.macro_particles_per_injection)
        )
        self.max_macro_particle_weight_var.set(
            str(runtime.max_macro_particle_weight)
        )

    def _apply_runtime_policy_vars(self, runtime: RuntimeConfig) -> None:
        self.max_wall_time_var.set(str(runtime.max_wall_time_s))
        self.terminal_event_mode_var.set(runtime.terminal_event_mode)
        self.max_terminal_event_rows_var.set(
            str(runtime.max_terminal_event_rows)
        )

    def _apply_output_vars(self, output: OutputConfig) -> None:
        self.save_snapshots_var.set(output.save_snapshots)
        self.save_h5_var.set(output.save_h5)
        self.save_figures_var.set(output.save_figures)
        self.trajectory_sample_count_var.set(
            str(output.trajectory_sample_count)
        )
        self.trajectory_record_every_var.set(
            str(output.trajectory_record_every)
        )
        self.snapshot_plot_max_points_var.set(
            str(output.snapshot_plot_max_points)
        )
        self.terminal_time_bin_var.set(str(output.terminal_time_bin_ms))
        self.report_after_run_var.set(output.generate_report_after_run)

    def _sync_config_from_vars(self) -> None:
        self._sync_project_from_vars()
        self._sync_runtime_from_vars(self.config.runtime)
        self._sync_output_from_vars(self.config.output)
        self.config.state.loaded_baked_field_path = (
            self.config.loaded_baked_field_path
        )
        self._update_summaries()

    def _sync_project_from_vars(self) -> None:
        self.config.session_name = (
            self.session_name_var.get().strip() or self.config.session_name
        )
        self.config.output_dir = (
            self.output_dir_var.get().strip() or self.config.output_dir
        )
        self.config.loaded_baked_field_path = self.static_field_var.get().strip()

    def _sync_runtime_from_vars(self, runtime: RuntimeConfig) -> None:
        runtime.backend = self.backend_var.get().strip() or "cpu"
        runtime.dummy_static_field = bool(self.dummy_static_field_var.get())
        runtime.total_time_s = float(self.total_time_var.get())
        runtime.macro_dt_s = float(self.macro_dt_var.get())
        runtime.random_seed = int(float(self.random_seed_var.get()))
        runtime.snapshot_every = int(float(self.snapshot_every_var.get()))
        runtime.pic_space_charge_scale = float(
            self.pic_space_charge_scale_var.get()
        )
        if (
            not math.isfinite(runtime.pic_space_charge_scale)
            or runtime.pic_space_charge_scale < 0.0
        ):
            raise ValueError("PIC space-charge scale must be finite and non-negative.")
        runtime.collision_mode = (
            self.collision_mode_var.get().strip() or "explicit"
        )
        runtime.rf_frequency_hz = float(self.rf_frequency_var.get())
        runtime.rf_peak_voltage_v = float(self.rf_peak_voltage_var.get())
        runtime.rf_phase_deg = float(self.rf_phase_var.get())
        runtime.detector_z_mm = float(self.detector_z_var.get())
        runtime.detector_radius_mm = float(self.detector_radius_var.get())
        runtime.radial_limit_mm = float(self.radial_limit_var.get())
        runtime.capillary_voltage_v = float(self.capillary_voltage_var.get())
        runtime.capillary_prefill_macro_particles = int(
            float(self.capillary_prefill_macro_particles_var.get())
        )
        runtime.macro_particles_per_injection = int(
            float(self.macro_particles_per_injection_var.get())
        )
        runtime.max_macro_particle_weight = float(
            self.max_macro_particle_weight_var.get()
        )
        self._sync_runtime_policy(runtime)

    def _sync_runtime_policy(self, runtime: RuntimeConfig) -> None:
        runtime.max_wall_time_s = float(self.max_wall_time_var.get())
        runtime.terminal_event_mode = (
            self.terminal_event_mode_var.get().strip() or "transport-only"
        )
        runtime.max_terminal_event_rows = int(
            float(self.max_terminal_event_rows_var.get())
        )

    def _sync_output_from_vars(self, output: OutputConfig) -> None:
        output.save_snapshots = bool(self.save_snapshots_var.get())
        output.save_h5 = bool(self.save_h5_var.get())
        output.save_figures = bool(self.save_figures_var.get())
        output.trajectory_sample_count = int(
            float(self.trajectory_sample_count_var.get())
        )
        output.trajectory_record_every = int(
            float(self.trajectory_record_every_var.get())
        )
        output.snapshot_plot_max_points = int(
            float(self.snapshot_plot_max_points_var.get())
        )
        output.terminal_time_bin_ms = float(self.terminal_time_bin_var.get())
        if output.terminal_time_bin_ms <= 0.0:
            raise ValueError("Terminal current time bin must be positive.")
        output.generate_report_after_run = bool(
            self.report_after_run_var.get()
        )

    def _update_summaries(self) -> None:
        self._update_beam_summary()
        self._update_field_summary()
        self._update_runtime_dialog_summaries()

    def _update_beam_summary(self) -> None:
        beam = self.config.beam
        gas_mode = beam.gas_velocity_init_mode == "from-gas-field"
        velocity = "gas field" if gas_mode else (
            "fallback 250 m/s" if beam.kinetic_energy_ev is None
            else f"{float(beam.kinetic_energy_ev):.6g} eV"
        )
        self.beam_summary_var.set(
            "ion={}, mode={}, current={:.6g} A, mass={:.6g} amu, charge=+{}, "
            "particles={}, velocity={}".format(
                beam.ion_name,
                beam.source_mode,
                beam.current_a,
                beam.mass_amu,
                beam.charge_e,
                beam.particle_count,
                velocity,
            )
        )

    def _update_field_summary(self) -> None:
        field = self.config.field_bake
        self.field_summary_var.set(
            "dc={}, rf={}, gas={}, grid z=[{:.6g}, {:.6g}] mm "
            "r=[{:.6g}, {:.6g}] mm d=({:.6g}, {:.6g}) mm; "
            "Fluent offset={:.6g} mm; "
            "capillary exit/L/R/external-start={:.6g}/{:.6g}/{:.6g}/{:.6g} mm".format(
                Path(field.simion_dc_path).name if field.simion_dc_path else "-",
                Path(field.simion_rf_path).name if field.simion_rf_path else "-",
                field.gas_field_mode,
                field.z_min_mm,
                field.z_max_mm,
                field.r_min_mm,
                field.r_max_mm,
                field.dz_mm,
                field.dr_mm,
                field.offset_z_fluent_mm,
                field.capillary_exit_z_mm,
                field.fluent_capillary_total_length_mm,
                field.fluent_capillary_radius_mm,
                field.fluent_capillary_external_start_z_mm,
            )
        )

    def _update_runtime_dialog_summaries(self) -> None:
        runtime = self.config.runtime
        grid = (
            "shared static grid"
            if (runtime.pic_grid_nr, runtime.pic_grid_nz) == (0, 0)
            else f"decoupled {runtime.pic_grid_nr} x {runtime.pic_grid_nz}"
        )
        self.pic_summary_var.set(
            f"scale={runtime.pic_space_charge_scale:.6g}; {grid}; "
            f"solver={runtime.pic_poisson_backend}"
        )
        if runtime.collision_physics_backend == "ionspa":
            physics = f"ionspa/{runtime.ionspa_backend}"
        else:
            physics = f"iict-lite ({iict_parameter_summary(runtime)})"
        self.collision_summary_var.set(
            f"{runtime.collision_mode}; physics={physics}; "
            f"products={runtime.fragmentation_mode}"
        )
