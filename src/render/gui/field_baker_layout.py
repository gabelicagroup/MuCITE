"""Tk widget construction for the field-baker window."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any


def build_field_baker_layout(ui: Any) -> None:
    """Build the field-baker controls without owning application behavior."""
    root_frame = ttk.Frame(ui.root, padding=12)
    root_frame.pack(fill=tk.BOTH, expand=True)
    left = ttk.Frame(root_frame)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
    right = ttk.Frame(root_frame)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    _build_control_panel(ui, left)
    _build_preview_panel(ui, right)


def _build_control_panel(ui: Any, parent: ttk.Frame) -> None:
    ttk.Label(
        parent,
        text="Field Baker Control Panel",
        font=("Segoe UI", 15, "bold"),
    ).pack(anchor=tk.W, pady=(0, 10))
    _build_file_groups(ui, parent)
    _build_parameter_group(ui, parent)
    _build_grid_group(ui, parent)
    _build_preview_options(ui, parent)
    _build_notes(parent)
    _build_run_controls(ui, parent)
    _build_log(ui, parent)


def _build_file_groups(ui: Any, parent: ttk.Frame) -> None:
    files_group = ttk.LabelFrame(parent, text="Input Files", padding=10)
    files_group.pack(fill=tk.X, pady=(0, 10))
    ui._add_file_row(files_group, "SIMION DC File", ui.simion_dc_var, 0)
    ui._add_file_row(files_group, "SIMION RF File", ui.simion_rf_var, 1)
    ui._add_file_row(files_group, "Fluent CSV", ui.fluent_var, 2)

    outputs_group = ttk.LabelFrame(parent, text="Output Paths", padding=10)
    outputs_group.pack(fill=tk.X, pady=(0, 10))
    ui._add_file_row(
        outputs_group,
        "Output NPY",
        ui.output_npy_var,
        0,
        save_mode=True,
    )
    ui._add_directory_row(outputs_group, "Plot Dir", ui.output_plot_dir_var, 1)


def _build_parameter_group(ui: Any, parent: ttk.Frame) -> None:
    group = ttk.LabelFrame(
        parent,
        text="Alignment / Background Parameters",
        padding=10,
    )
    group.pack(fill=tk.X, pady=(0, 10))
    rows = (
        ("SIMION z offset [mm]", ui.offset_simion_var),
        ("Fluent z offset [mm]", ui.offset_fluent_var),
        ("Capillary total length [mm]", ui.fluent_capillary_total_length_var),
        ("Capillary radius [mm]", ui.fluent_capillary_radius_var),
        ("External gas start z [mm]", ui.fluent_capillary_external_start_z_var),
        ("Background P [Pa]", ui.background_pressure_var),
        ("Background T [K]", ui.background_temperature_var),
        ("SIMION PA grids/mm", ui.simion_pa_grids_var),
        ("DC voltage scale", ui.simion_dc_voltage_scale_var),
        ("Clip Fluent after z [mm]", ui.clip_fluent_after_z_var),
    )
    for row, (label, variable) in enumerate(rows):
        ui._add_entry_row(group, label, variable, row)


def _build_grid_group(ui: Any, parent: ttk.Frame) -> None:
    group = ttk.LabelFrame(parent, text="Global Grid Parameters", padding=10)
    group.pack(fill=tk.X, pady=(0, 10))
    rows = (
        ("z min [mm]", ui.z_min_var),
        ("z max [mm]", ui.z_max_var),
        ("r min [mm]", ui.r_min_var),
        ("r max [mm]", ui.r_max_var),
        ("dz [mm]", ui.dz_var),
        ("dr [mm]", ui.dr_var),
        ("capillary exit z [mm]", ui.capillary_exit_z_var),
    )
    for row, (label, variable) in enumerate(rows):
        ui._add_entry_row(group, label, variable, row)


def _build_preview_options(ui: Any, parent: ttk.Frame) -> None:
    group = ttk.LabelFrame(parent, text="Preview Options", padding=10)
    group.pack(fill=tk.X, pady=(0, 10))
    ttk.Label(group, text="Potential field for heatmap").grid(
        row=0,
        column=0,
        sticky="w",
        pady=4,
    )
    phi_combo = ttk.Combobox(
        group,
        textvariable=ui.phi_key_var,
        values=("phi_dc_v", "phi_rf_v"),
        state="readonly",
    )
    phi_combo.grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=4)
    group.columnconfigure(1, weight=1)
    ttk.Button(
        group,
        text="Reset Grid Defaults",
        command=ui._reset_grid_defaults,
    ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))


def _build_notes(parent: ttk.Frame) -> None:
    group = ttk.LabelFrame(parent, text="Notes", padding=10)
    group.pack(fill=tk.X, pady=(0, 10))
    text = (
        "Global grid:\n"
        "default z = [0, 50] mm, r = [0, 10] mm, dz = dr = 0.1 mm\n\n"
        "Capillary exit is written into baked metadata and shown in plots.\n"
        "Fluent z offset is applied directly; capillary length does not replace it.\n"
        "The 0.25 mm radius and external start z=4.5 mm define the external-gas mask.\n"
        "Use the offsets to align SIMION / Fluent exports to this global frame.\n"
        "Refined SIMION PA text files are used directly; Python-side solving is disabled.\n"
        "PATXT header ng defines grids/mm by default (current S-lens: 100); x->z and y->r.\n"
        "RF PA text must be normalized with adjacent RF electrodes at +1 V and -1 V; Vref = 1 V.\n"
        "Set 'Clip Fluent after z' when the CFD outlet/end plane is inside the global field.\n"
        "If you load slens_rf.patxt / slens_dc.patxt, z_max >= 65 mm and r_max >= 20 mm are recommended.\n"
        "If the baked field is clipped near the boundaries, expand z/r max or refine dz/dr."
    )
    ttk.Label(group, text=text, justify=tk.LEFT).pack(anchor=tk.W)


def _build_run_controls(ui: Any, parent: ttk.Frame) -> None:
    button_row = ttk.Frame(parent)
    button_row.pack(fill=tk.X, pady=(4, 8))
    ui.run_button = ttk.Button(
        button_row,
        text="Bake And Preview",
        command=ui._start_bake,
    )
    ui.run_button.pack(side=tk.LEFT)
    ttk.Button(
        button_row,
        text="Quit",
        command=ui.root.destroy,
    ).pack(side=tk.LEFT, padx=(8, 0))

    status_frame = ttk.Frame(parent)
    status_frame.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(status_frame, text="Status:", font=("Segoe UI", 10, "bold")).pack(
        side=tk.LEFT
    )
    ttk.Label(
        status_frame,
        textvariable=ui.status_var,
        wraplength=360,
    ).pack(side=tk.LEFT, padx=(8, 0))


def _build_log(ui: Any, parent: ttk.Frame) -> None:
    group = ttk.LabelFrame(parent, text="Log", padding=8)
    group.pack(fill=tk.BOTH, expand=True)
    ui.log = scrolledtext.ScrolledText(
        group,
        width=48,
        height=20,
        wrap=tk.WORD,
        font=("Consolas", 9),
    )
    ui.log.pack(fill=tk.BOTH, expand=True)
    ui.log.configure(state=tk.DISABLED)


def _build_preview_panel(ui: Any, parent: ttk.Frame) -> None:
    ttk.Label(
        parent,
        text="Sanity Preview",
        font=("Segoe UI", 15, "bold"),
    ).pack(anchor=tk.W, pady=(0, 10))
    ui.preview_notebook = ttk.Notebook(parent)
    ui.preview_notebook.pack(fill=tk.BOTH, expand=True)

    ui.heatmap_tab = ttk.Frame(ui.preview_notebook)
    ui.centerline_tab = ttk.Frame(ui.preview_notebook)
    ui.preview_notebook.add(ui.heatmap_tab, text="Heatmaps")
    ui.preview_notebook.add(ui.centerline_tab, text="Centerline Profiles")

    default_text = "Run the baker to generate preview images."
    ui.heatmap_label = ttk.Label(ui.heatmap_tab, text=default_text)
    ui.heatmap_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    ui.centerline_label = ttk.Label(ui.centerline_tab, text=default_text)
    ui.centerline_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
