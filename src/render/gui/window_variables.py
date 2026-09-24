"""Tk variable construction for the main GUI window."""

from __future__ import annotations

import tkinter as tk


class _WindowVariablesMixin:
    def _build_vars(self) -> None:
        self._build_project_vars()
        self._build_runtime_vars()
        self._build_output_vars()
        self._build_summary_vars()

    def _build_project_vars(self) -> None:
        self.session_name_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.static_field_var = tk.StringVar()
        self.status_var = tk.StringVar(value="CONFIGURED")
        self.backend_var = tk.StringVar(value="cpu")
        self.dummy_static_field_var = tk.BooleanVar(value=False)

    def _build_runtime_vars(self) -> None:
        self.total_time_var = tk.StringVar()
        self.macro_dt_var = tk.StringVar()
        self.random_seed_var = tk.StringVar()
        self.snapshot_every_var = tk.StringVar()
        self.pic_space_charge_scale_var = tk.StringVar()
        self.collision_mode_var = tk.StringVar()
        self.pic_summary_var = tk.StringVar()
        self.collision_summary_var = tk.StringVar()
        self.rf_frequency_var = tk.StringVar()
        self.rf_peak_voltage_var = tk.StringVar()
        self.rf_phase_var = tk.StringVar()
        self.detector_z_var = tk.StringVar()
        self.detector_radius_var = tk.StringVar()
        self.radial_limit_var = tk.StringVar()
        self.capillary_voltage_var = tk.StringVar()
        self.capillary_prefill_macro_particles_var = tk.StringVar()
        self.macro_particles_per_injection_var = tk.StringVar()
        self.max_macro_particle_weight_var = tk.StringVar()
        self.max_wall_time_var = tk.StringVar()
        self.terminal_event_mode_var = tk.StringVar()
        self.max_terminal_event_rows_var = tk.StringVar()

    def _build_output_vars(self) -> None:
        self.save_snapshots_var = tk.BooleanVar(value=True)
        self.save_h5_var = tk.BooleanVar(value=False)
        self.save_figures_var = tk.BooleanVar(value=True)
        self.trajectory_sample_count_var = tk.StringVar()
        self.trajectory_record_every_var = tk.StringVar()
        self.snapshot_plot_max_points_var = tk.StringVar()
        self.terminal_time_bin_var = tk.StringVar(value="0.2")
        self.report_after_run_var = tk.BooleanVar(value=True)
        self.auto_refresh_var = tk.BooleanVar(value=True)

    def _build_summary_vars(self) -> None:
        self.macro_var = tk.StringVar(value="0")
        self.time_var = tk.StringVar(value="0.000 ms")
        self.alive_var = tk.StringVar(value="-")
        self.collisions_var = tk.StringVar(value="0")
        self.exit_current_var = tk.StringVar(value="-")
        self.exit_ions_var = tk.StringVar(value="-")
        self.beam_summary_var = tk.StringVar()
        self.field_summary_var = tk.StringVar()
