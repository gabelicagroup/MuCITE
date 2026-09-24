"""Field, gas, and electrode-mask configuration dialog."""

from __future__ import annotations

import copy
import math
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from .dialogs import _DialogBase
from .models import FieldBakeConfig, RuntimeConfig


FieldApplyCallback = Callable[[FieldBakeConfig, RuntimeConfig], None]


FIELD_NUMERIC_LABELS = {
    "offset_z_simion_mm": "SIMION z offset",
    "offset_z_fluent_mm": "Fluent z offset",
    "pa_grids_per_mm": "PA grids/mm",
    "z_min_mm": "z min",
    "z_max_mm": "z max",
    "r_min_mm": "r min",
    "r_max_mm": "r max",
    "dz_mm": "dz",
    "dr_mm": "dr",
    "dc_voltage_scale": "DC voltage scale",
    "background_pressure_pa": "Gas pressure",
    "background_temperature_k": "Gas temperature",
    "capillary_exit_z_mm": "Gas alignment exit z",
    "fluent_capillary_total_length_mm": "Fluent capillary total length",
    "fluent_capillary_radius_mm": "Fluent capillary radius",
    "fluent_capillary_external_start_z_mm": "Fluent external gas start z",
}


def _normalized_file_suffix(value: str, suffix: str) -> str:
    """Return a non-empty path with exactly the requested lowercase suffix."""

    text = str(value).strip()
    if not text:
        return ""
    return str(Path(text).with_suffix(suffix))


def _save_suffix(key: str, *, save: bool) -> str | None:
    if not save:
        return None
    if key == "output_npy":
        return ".npy"
    if key == "electrode_mask_cache_path":
        return ".npz"
    return None


def _validate_field_config(config: FieldBakeConfig) -> None:
    for field_name, label in FIELD_NUMERIC_LABELS.items():
        if not math.isfinite(float(getattr(config, field_name))):
            raise ValueError(f"{label} must be finite.")
    if config.gas_field_mode == "import" and not str(config.fluent_path).strip():
        raise ValueError("Import gas mode requires a Fluent gas-field path.")
    if config.z_max_mm <= config.z_min_mm or config.r_max_mm <= config.r_min_mm:
        raise ValueError("Grid maxima must be greater than grid minima.")
    if config.dz_mm <= 0.0 or config.dr_mm <= 0.0:
        raise ValueError("dz and dr must be positive.")
    if config.pa_grids_per_mm <= 0.0:
        raise ValueError("PA grids/mm must be positive.")
    if config.background_pressure_pa < 0.0 or config.background_temperature_k <= 0.0:
        raise ValueError("Gas pressure must be non-negative and temperature positive.")
    if config.fluent_capillary_total_length_mm <= 0.0:
        raise ValueError("Fluent capillary total length must be positive.")
    if config.fluent_capillary_radius_mm <= 0.0:
        raise ValueError("Fluent capillary radius must be positive.")
    if config.fluent_capillary_external_start_z_mm >= config.capillary_exit_z_mm:
        raise ValueError("Fluent external gas start z must be smaller than capillary exit z.")


def _validate_mask_config(runtime: RuntimeConfig) -> None:
    offset_mm = runtime.electrode_mask_z_offset_mm
    if offset_mm is not None and not math.isfinite(float(offset_mm)):
        raise ValueError("Mask z offset must be finite when provided.")
    distance_mm = float(runtime.electrode_hit_distance_mm)
    if not math.isfinite(distance_mm):
        raise ValueError("Electrode hit distance must be finite.")
    if distance_mm < 0.0:
        raise ValueError("Electrode hit distance cannot be negative.")


def _initial_gas_mode(config: FieldBakeConfig) -> str:
    """Resolve new and legacy GUI configurations to an explicit gas mode."""
    raw_mode = str(getattr(config, "gas_field_mode", "")).strip().lower()
    if not raw_mode:
        raw_mode = "import" if str(config.fluent_path).strip() else "static"
    return raw_mode if raw_mode in {"static", "import"} else "static"


def _build_import_gas_group(
    dialog: FieldBakerDialog,
    parent: ttk.Frame,
) -> ttk.LabelFrame:
    group = ttk.LabelFrame(parent, text="Imported Gas Field", padding=8)
    dialog._add_file(group, 0, "Fluent CSV", "fluent_path", dialog.config.fluent_path)
    dialog._add_entry(
        group,
        1,
        "Fluent z offset [mm]",
        "offset_z_fluent_mm",
        dialog.config.offset_z_fluent_mm,
    )
    rows = (
        ("Capillary total length [mm]", "fluent_capillary_total_length_mm"),
        ("Capillary radius [mm]", "fluent_capillary_radius_mm"),
        ("External gas start z [mm]", "fluent_capillary_external_start_z_mm"),
    )
    for row, (label, key) in enumerate(rows, start=2):
        dialog._add_entry(group, row, label, key, getattr(dialog.config, key))
    dialog._add_shared_entry(group, 5, "Outside-domain P [Pa]", "background_pressure_pa")
    dialog._add_shared_entry(group, 6, "Outside-domain T [K]", "background_temperature_k")
    ttk.Label(
        group,
        text="Fluent z offset is used directly; capillary total length does not modify it.",
        wraplength=470,
        justify=tk.LEFT,
    ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))
    return group


class FieldBakerDialog(_DialogBase):
    """Edit field baking plus the visually related 2D mask settings."""

    def __init__(
        self,
        parent: tk.Misc,
        config: FieldBakeConfig,
        runtime_config: RuntimeConfig,
        on_apply: FieldApplyCallback,
    ) -> None:
        super().__init__(parent, "Field Baker")
        self.config = copy.deepcopy(config)
        self.runtime_config = copy.deepcopy(runtime_config)
        self.on_apply = on_apply
        self.static_gas_group: ttk.LabelFrame
        self.import_gas_group: ttk.LabelFrame
        self._build()

    def _build(self) -> None:
        self._build_field_files()
        self._build_gas_controls()
        self._build_grid_controls()
        self._build_mask_controls()
        self._buttons(self._apply)
        self._sync_gas_mode()

    def _build_field_files(self) -> None:
        group = ttk.LabelFrame(self.body, text="Electric Field Files", padding=10)
        group.pack(fill=tk.X)
        self._add_file(group, 0, "SIMION DC", "simion_dc_path", self.config.simion_dc_path)
        self._add_file(group, 1, "SIMION RF", "simion_rf_path", self.config.simion_rf_path)
        self._add_file(group, 2, "Output NPY", "output_npy", self.config.output_npy, save=True)
        self._add_directory(group, 3, "Output plot dir", "output_plot_dir", self.config.output_plot_dir)

    def _build_gas_controls(self) -> None:
        mode_group = ttk.LabelFrame(self.body, text="Gas Field", padding=10)
        mode_group.pack(fill=tk.X, pady=(10, 0))
        self._add_combo(
            mode_group,
            0,
            "Gas field mode",
            "gas_field_mode",
            _initial_gas_mode(self.config),
            ("static", "import"),
        )
        mode_var = self.vars["gas_field_mode"]
        mode_var.trace_add("write", lambda *_args: self._sync_gas_mode())
        self._make_background_vars()

        self.static_gas_group = ttk.LabelFrame(mode_group, text="Static Gas", padding=8)
        self._add_shared_entry(self.static_gas_group, 0, "Pressure [Pa]", "background_pressure_pa")
        self._add_shared_entry(self.static_gas_group, 1, "Temperature [K]", "background_temperature_k")
        ttk.Label(
            self.static_gas_group,
            text="The whole gas grid is uniform; axial and radial gas velocity are 0 m/s.",
            wraplength=470,
            justify=tk.LEFT,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.import_gas_group = _build_import_gas_group(self, mode_group)
        for gas_group in (self.static_gas_group, self.import_gas_group):
            gas_group.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="ew",
                pady=(8, 0),
            )

    def _make_background_vars(self) -> None:
        self.vars["background_pressure_pa"] = tk.StringVar(
            value=str(self.config.background_pressure_pa)
        )
        self.vars["background_temperature_k"] = tk.StringVar(
            value=str(self.config.background_temperature_k)
        )

    def _add_shared_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.vars[key], width=18).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=(10, 0),
            pady=3,
        )
        parent.columnconfigure(1, weight=1)

    def _sync_gas_mode(self) -> None:
        if not hasattr(self, "static_gas_group") or not hasattr(self, "import_gas_group"):
            return
        self.static_gas_group.grid_remove()
        self.import_gas_group.grid_remove()
        selected = str(self.vars["gas_field_mode"].get()).strip().lower()
        target = self.import_gas_group if selected == "import" else self.static_gas_group
        target.grid()

    def _build_grid_controls(self) -> None:
        group = ttk.LabelFrame(self.body, text="Grid / Alignment", padding=10)
        group.pack(fill=tk.X, pady=(10, 0))
        rows = (
            ("SIMION z offset [mm]", "offset_z_simion_mm", self.config.offset_z_simion_mm),
            ("PA grids/mm", "pa_grids_per_mm", self.config.pa_grids_per_mm),
            ("z min [mm]", "z_min_mm", self.config.z_min_mm),
            ("z max [mm]", "z_max_mm", self.config.z_max_mm),
            ("r min [mm]", "r_min_mm", self.config.r_min_mm),
            ("r max [mm]", "r_max_mm", self.config.r_max_mm),
            ("dz [mm]", "dz_mm", self.config.dz_mm),
            ("dr [mm]", "dr_mm", self.config.dr_mm),
            ("DC voltage scale", "dc_voltage_scale", self.config.dc_voltage_scale),
            ("Gas alignment exit z [mm]", "capillary_exit_z_mm", self.config.capillary_exit_z_mm),
        )
        for row, (label, key, value) in enumerate(rows):
            self._add_entry(group, row, label, key, value)
        self._add_combo(
            group,
            len(rows),
            "Plot field",
            "phi_key_for_plot",
            self.config.phi_key_for_plot,
            ("phi_dc_v", "phi_rf_v"),
        )

    def _build_mask_controls(self) -> None:
        group = ttk.LabelFrame(self.body, text="2D Electrode Mask", padding=10)
        group.pack(fill=tk.X, pady=(10, 0))
        runtime = self.runtime_config
        self._add_file(group, 0, "Mask source", "electrode_mask_path", runtime.electrode_mask_path, mask=True)
        self._add_file(group, 1, "Mask cache", "electrode_mask_cache_path", runtime.electrode_mask_cache_path, save=True, mask=True)
        offset = runtime.electrode_mask_z_offset_mm
        self._add_entry(
            group,
            2,
            "Mask z offset [mm] (blank = field metadata)",
            "electrode_mask_z_offset_mm",
            "" if offset is None else offset,
        )
        self._add_entry(
            group,
            3,
            "Electrode hit distance [mm]",
            "electrode_hit_distance_mm",
            runtime.electrode_hit_distance_mm,
        )

    def _add_file(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        value: object,
        *,
        save: bool = False,
        mask: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        var = tk.StringVar(value=str(value))
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=48).grid(
            row=row, column=1, sticky="ew", padx=(10, 8), pady=3
        )
        command = lambda: self._browse_file(
            var,
            key=key,
            save=save,
            mask=mask,
        )
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, pady=3)
        parent.columnconfigure(1, weight=1)

    def _add_directory(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        value: object,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        var = tk.StringVar(value=str(value))
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=48).grid(
            row=row, column=1, sticky="ew", padx=(10, 8), pady=3
        )
        ttk.Button(parent, text="Browse", command=lambda: self._browse_directory(var)).grid(
            row=row, column=2, pady=3
        )
        parent.columnconfigure(1, weight=1)

    def _browse_file(
        self,
        var: tk.StringVar,
        *,
        key: str,
        save: bool,
        mask: bool,
    ) -> None:
        required_suffix = _save_suffix(key, save=save)
        if required_suffix == ".npy":
            filetypes = [("NumPy field", "*.npy"), ("All files", "*.*")]
        elif required_suffix == ".npz":
            filetypes = [("Electrode mask cache", "*.npz"), ("All files", "*.*")]
        elif mask:
            filetypes = [
                ("Electrode mask", ("*.patxt", "*.npz")),
                ("All files", "*.*"),
            ]
        else:
            filetypes = [
                ("Field data", ("*.patxt", "*.csv", "*.txt", "*.npy")),
                ("All files", "*.*"),
            ]
        if save:
            path = filedialog.asksaveasfilename(
                parent=self.top,
                filetypes=filetypes,
                defaultextension=required_suffix,
            )
        else:
            path = filedialog.askopenfilename(parent=self.top, filetypes=filetypes)
        if path:
            var.set(
                path
                if required_suffix is None
                else _normalized_file_suffix(path, required_suffix)
            )

    def _browse_directory(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(parent=self.top, title="Select directory")
        if path:
            var.set(path)

    def _field_values(self) -> dict[str, object]:
        return {
            "gas_field_mode": self._get_str("gas_field_mode"),
            "simion_dc_path": self._get_str("simion_dc_path"),
            "simion_rf_path": self._get_str("simion_rf_path"),
            "fluent_path": self._get_str("fluent_path"),
            "offset_z_simion_mm": self._get_float("offset_z_simion_mm"),
            "offset_z_fluent_mm": self._get_float("offset_z_fluent_mm"),
            "pa_grids_per_mm": self._get_float("pa_grids_per_mm"),
            "z_min_mm": self._get_float("z_min_mm"),
            "z_max_mm": self._get_float("z_max_mm"),
            "r_min_mm": self._get_float("r_min_mm"),
            "r_max_mm": self._get_float("r_max_mm"),
            "dz_mm": self._get_float("dz_mm"),
            "dr_mm": self._get_float("dr_mm"),
            "dc_voltage_scale": self._get_float("dc_voltage_scale"),
            "background_pressure_pa": self._get_float("background_pressure_pa"),
            "background_temperature_k": self._get_float("background_temperature_k"),
            "capillary_exit_z_mm": self._get_float("capillary_exit_z_mm"),
            "fluent_capillary_total_length_mm": self._get_float(
                "fluent_capillary_total_length_mm"
            ),
            "fluent_capillary_radius_mm": self._get_float(
                "fluent_capillary_radius_mm"
            ),
            "fluent_capillary_external_start_z_mm": self._get_float(
                "fluent_capillary_external_start_z_mm"
            ),
            "phi_key_for_plot": self._get_str("phi_key_for_plot"),
            "output_npy": _normalized_file_suffix(
                self._get_str("output_npy"),
                ".npy",
            ),
            "output_plot_dir": self._get_str("output_plot_dir"),
        }

    def _apply(self) -> None:
        try:
            config = replace(self.config, **self._field_values())
            _validate_field_config(config)
            runtime = replace(
                self.runtime_config,
                electrode_mask_path=self._get_str("electrode_mask_path"),
                electrode_mask_cache_path=_normalized_file_suffix(
                    self._get_str("electrode_mask_cache_path"),
                    ".npz",
                ),
                electrode_mask_z_offset_mm=self._get_optional_float("electrode_mask_z_offset_mm"),
                electrode_hit_distance_mm=self._get_float("electrode_hit_distance_mm"),
            )
            _validate_mask_config(runtime)
        except Exception as exc:
            messagebox.showerror("Invalid Field Config", str(exc), parent=self.top)
            return
        self.on_apply(config, runtime)
        self.top.destroy()


__all__ = ["FieldBakerDialog"]
