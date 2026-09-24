"""Compact runtime controls with focused PIC and collision dialogs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable


class _RuntimeLayoutMixin:
    def _build_runtime_group(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Runtime Setup", padding=8)
        group.pack(fill=tk.X, pady=(0, 8), padx=4)
        self._build_runtime_core(group)
        self._build_runtime_geometry(group)
        self._build_runtime_policy(group)
        self._build_runtime_output(group)

    def _build_runtime_core(self, group: ttk.Frame) -> None:
        rows = (
            ("Total time [s]", self.total_time_var),
            ("Macro step [s]", self.macro_dt_var),
            ("Random seed", self.random_seed_var),
            ("Snapshot/export every [macro]", self.snapshot_every_var),
            ("RF frequency [Hz]", self.rf_frequency_var),
            ("RF Vpeak [V]", self.rf_peak_voltage_var),
            ("RF phase [deg]", self.rf_phase_var),
        )
        for row, (label, variable) in enumerate(rows):
            self._add_runtime_entry(group, label, variable, row)
        self._add_runtime_entry(
            group,
            "PIC space-charge scale (0 = off)",
            self.pic_space_charge_scale_var,
            7,
        )
        self._add_runtime_settings(
            group,
            "PIC grid / Poisson",
            8,
            self._open_pic_dialog,
            self.pic_summary_var,
        )
        self._add_runtime_selector(
            group,
            "Collisions",
            self.collision_mode_var,
            ("explicit", "hybrid-langevin"),
            10,
            self._on_collision_mode_selected,
            self._open_collision_dialog,
            self.collision_summary_var,
        )

    def _build_runtime_geometry(self, group: ttk.Frame) -> None:
        rows = (
            ("Detector z [mm]", self.detector_z_var),
            ("Detector radius [mm]", self.detector_radius_var),
            ("Radial limit [mm]", self.radial_limit_var),
            ("Capillary voltage [V]", self.capillary_voltage_var),
            ("Prefill weighted ion packs", self.capillary_prefill_macro_particles_var),
            ("Weighted packs per injection", self.macro_particles_per_injection_var),
            ("Max ions per weighted pack", self.max_macro_particle_weight_var),
            ("Max wall time [s]", self.max_wall_time_var),
        )
        for index, (label, variable) in enumerate(rows, start=12):
            self._add_runtime_entry(group, label, variable, index)

    def _build_runtime_policy(self, group: ttk.Frame) -> None:
        self._add_runtime_combo(
            group,
            "Terminal events",
            self.terminal_event_mode_var,
            ("transport-only", "all", "none"),
            20,
        )
        self._add_runtime_entry(
            group,
            "Max terminal rows (0 = exact/unlimited)",
            self.max_terminal_event_rows_var,
            21,
        )

    def _build_runtime_output(self, group: ttk.Frame) -> None:
        rows = (
            ("Trajectory sample count", self.trajectory_sample_count_var),
            ("Trajectory record every", self.trajectory_record_every_var),
            ("Snapshot plot max points", self.snapshot_plot_max_points_var),
            ("Terminal current bin [ms]", self.terminal_time_bin_var),
        )
        for index, (label, variable) in enumerate(rows, start=22):
            self._add_runtime_entry(group, label, variable, index)
        checks = (
            ("Use dummy static field", self.dummy_static_field_var),
            ("Save snapshots", self.save_snapshots_var),
            ("Save snapshots as H5", self.save_h5_var),
            ("Save snapshot plots", self.save_figures_var),
            ("Generate report after run", self.report_after_run_var),
        )
        for index, (label, variable) in enumerate(checks, start=26):
            self._add_runtime_check(group, label, variable, index)

    def _add_runtime_settings(
        self,
        parent: ttk.Frame,
        label: str,
        row: int,
        open_dialog: Callable[[], None],
        summary: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Button(parent, text="Settings...", command=open_dialog).grid(
            row=row, column=1, sticky="ew", padx=(8, 0), pady=3
        )
        ttk.Label(parent, textvariable=summary, wraplength=190).grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        parent.columnconfigure(1, weight=1)

    def _add_runtime_selector(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        row: int,
        selected: Callable[[object], None],
        open_dialog: Callable[[], None],
        summary: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
        combo = ttk.Combobox(
            holder, textvariable=variable, values=values, state="readonly", width=14
        )
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        combo.bind("<<ComboboxSelected>>", selected)
        ttk.Button(holder, text="Settings...", command=open_dialog).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Label(parent, textvariable=summary, wraplength=190).grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        parent.columnconfigure(1, weight=1)

    def _add_runtime_entry(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var, width=18).grid(
            row=row, column=1, sticky="ew", padx=(8, 0), pady=3
        )
        parent.columnconfigure(1, weight=1)

    def _add_runtime_combo(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        values: tuple[str, ...],
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(
            parent, textvariable=var, values=values, state="readonly", width=16
        ).grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _add_runtime_check(
        parent: ttk.Frame,
        text: str,
        variable: tk.Variable,
        row: int,
    ) -> None:
        ttk.Checkbutton(parent, text=text, variable=variable).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=3
        )
