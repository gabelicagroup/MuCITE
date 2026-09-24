"""Beam-source configuration dialog with mode-aware velocity controls."""

from __future__ import annotations

import copy
import math
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from .models import BeamConfig


HEAT_CAPACITY_PROFILES = (
    "peptide",
    "peptide90",
    "lipid",
    "oligonucleotide",
    "drug",
    "sugar",
    "tunemix",
    "tunemix322",
    "tunemix622",
    "tunemix922",
    "tunemix1222",
    "tunemix1522",
    "tunemix2122",
)


def normalize_gas_velocity_mode(mode: object) -> str:
    """Return the canonical GUI velocity-source mode.

    ``off`` is accepted as the schema-v1 spelling of ``static`` so an old
    project can be opened and saved with the presentation model.
    """

    value = str(mode).strip().lower()
    if value == "off":
        return "static"
    if value in {"static", "from-gas-field"}:
        return value
    raise ValueError("Gas velocity mode must be 'static' or 'from-gas-field'.")


def beam_velocity_visibility(mode: object) -> dict[str, bool]:
    """Describe which mutually exclusive velocity editor should be visible."""

    canonical = normalize_gas_velocity_mode(mode)
    return {
        "static_motion": canonical == "static",
        "gas_field": canonical == "from-gas-field",
    }


class _BeamDialogBase:
    """Provide the scrollable modal shell and compact form-row builders."""

    def __init__(
        self,
        parent: tk.Misc,
        config: BeamConfig,
        on_apply: Callable[[BeamConfig], None],
    ) -> None:
        self.config = copy.deepcopy(config)
        self.on_apply = on_apply
        self.vars: dict[str, tk.Variable] = {}
        self.top = tk.Toplevel(parent)
        self.top.title("Beam Setup")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.geometry("760x800")
        self.top.minsize(660, 560)
        self._build_shell()
        self._build_sections()
        self._build_buttons()
        self._sync_velocity_frames()

    def _build_shell(self) -> None:
        outer = ttk.Frame(self.top, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            outer,
            orient=tk.VERTICAL,
            command=self.canvas.yview,
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.body = ttk.Frame(self.canvas, padding=(0, 0, 8, 0))
        self._body_window = self.canvas.create_window(
            (0, 0),
            window=self.body,
            anchor="nw",
        )
        self.body.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._fit_body_width)
        self.button_row = ttk.Frame(outer)
        self.button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _update_scroll_region(self, _event: tk.Event[tk.Misc]) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _fit_body_width(self, event: tk.Event[tk.Misc]) -> None:
        self.canvas.itemconfigure(self._body_window, width=event.width)

    def _add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        value: object,
        *,
        optional: bool = False,
    ) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        shown = "" if optional and value is None else str(value)
        variable = tk.StringVar(value=shown)
        self.vars[key] = variable
        ttk.Entry(row, textvariable=variable, width=24).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0))
        return variable

    def _add_combo(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        value: object,
        values: tuple[str, ...],
    ) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label).pack(side=tk.LEFT)
        variable = tk.StringVar(value=str(value))
        self.vars[key] = variable
        ttk.Combobox(
            row,
            textvariable=variable,
            values=values,
            state="readonly",
            width=22,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0))
        return variable

    def _add_gas_file(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Birth gas CSV (blank = baked field)").pack(side=tk.LEFT)
        variable = tk.StringVar(value=str(self.config.source_birth_velocity_gas_csv))
        self.vars["source_birth_velocity_gas_csv"] = variable
        ttk.Button(
            row,
            text="Browse",
            command=lambda: self._browse_gas_file(variable),
        ).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Entry(row, textvariable=variable, width=32).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0))

    def _browse_gas_file(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            parent=self.top,
            title="Select raw gas velocity field",
            filetypes=[("CSV / text", "*.csv *.txt"), ("All files", "*.*")],
        )
        if selected:
            variable.set(str(Path(selected)))


class BeamSetupDialog(_BeamDialogBase):
    """Edit ion, source-geometry, and velocity-source presentation values."""

    def _build_sections(self) -> None:
        self._build_ion_properties()
        self._build_source_geometry()
        self._build_velocity_source()

    def _build_ion_properties(self) -> None:
        group = ttk.LabelFrame(self.body, text="Ion properties", padding=10)
        group.pack(fill=tk.X)
        self._add_entry(group, "Ion name", "ion_name", self.config.ion_name)
        self._add_entry(group, "Mass [amu]", "mass_amu", self.config.mass_amu)
        self._add_entry(group, "Charge state [e]", "charge_e", self.config.charge_e)
        self._add_entry(group, "Number of atoms", "num_atoms", self.config.num_atoms)
        self._add_entry(group, "Collision CCS [m2]", "collision_cross_section_m2", self.config.collision_cross_section_m2)
        self._add_combo(
            group,
            "Heat-capacity profile",
            "heat_capacity_profile",
            self.config.heat_capacity_profile,
            HEAT_CAPACITY_PROFILES,
        )
        self._add_entry(group, "Delta H [kJ/mol]", "delta_h_kj_per_mol", self.config.delta_h_kj_per_mol)
        self._add_entry(group, "Delta S [J/mol/K]", "delta_s_j_per_mol_k", self.config.delta_s_j_per_mol_k)
        self._add_entry(
            group,
            "Ion internal T [K] (used by the collision backend)",
            "initial_internal_temperature_k",
            self.config.initial_internal_temperature_k,
        )

    def _build_source_geometry(self) -> None:
        group = ttk.LabelFrame(self.body, text="Source geometry", padding=10)
        group.pack(fill=tk.X, pady=(10, 0))
        self._add_combo(group, "Source mode", "source_mode", self.config.source_mode, ("packet", "continuous-current"))
        self._add_entry(group, "Weighted ion pack slots", "particle_count", self.config.particle_count)
        self._add_entry(group, "Ions per weighted pack", "macro_particle_weight", self.config.macro_particle_weight)
        self._add_entry(group, "Current [A] (0 = packet)", "current_a", self.config.current_a)
        self._add_entry(group, "Beam radius [mm]", "beam_radius_mm", self.config.beam_radius_mm)
        self._add_entry(group, "Initial x [mm]", "initial_x_mm", self.config.initial_x_mm)
        self._add_entry(group, "Initial y [mm]", "initial_y_mm", self.config.initial_y_mm)
        self._add_entry(group, "Initial z [mm]", "initial_z_mm", self.config.initial_z_mm)
        self._add_entry(group, "Position jitter [mm]", "initial_position_jitter_mm", self.config.initial_position_jitter_mm)
        self._add_combo(group, "Transverse profile", "source_profile", self.config.source_profile, ("uniform-disk", "gaussian"))
        self._add_entry(group, "Gaussian sigma [mm]", "source_gaussian_sigma_mm", self.config.source_gaussian_sigma_mm)
        self._add_entry(group, "Capillary exit z [mm]", "capillary_exit_z_mm", self.config.capillary_exit_z_mm)
        self._add_entry(group, "Capillary prefill length [mm]", "capillary_prefill_length_mm", self.config.capillary_prefill_length_mm)

    def _build_velocity_source(self) -> None:
        group = ttk.LabelFrame(self.body, text="Velocity source", padding=10)
        group.pack(fill=tk.X, pady=(10, 0))
        mode = normalize_gas_velocity_mode(self.config.gas_velocity_init_mode)
        variable = self._add_combo(
            group,
            "Gas velocity",
            "gas_velocity_init_mode",
            mode,
            ("static", "from-gas-field"),
        )
        variable.trace_add("write", lambda *_args: self._sync_velocity_frames())
        self._build_static_motion_fields(group)
        self._build_gas_field_inputs(group)
        self._add_entry(group, "Notes", "notes", self.config.notes)

    def _build_static_motion_fields(self, group: ttk.Frame) -> None:
        self.static_motion_frame = ttk.LabelFrame(group, text="Static motion", padding=8)
        self._add_entry(self.static_motion_frame, "Initial KE [eV]", "kinetic_energy_ev", self.config.kinetic_energy_ev, optional=True)
        self._add_entry(self.static_motion_frame, "Source T [K]", "source_temperature_k", self.config.source_temperature_k, optional=True)
        self._add_entry(
            self.static_motion_frame,
            "Source axial velocity [m/s]",
            "source_axial_velocity_m_per_s",
            self.config.source_axial_velocity_m_per_s,
            optional=True,
        )
        self._add_combo(
            self.static_motion_frame,
            "Direction axis",
            "direction_axis",
            self.config.direction_axis,
            ("+z", "-z", "+x", "-x", "+y", "-y"),
        )
        self._add_entry(self.static_motion_frame, "Cone half-angle [deg]", "cone_half_angle_deg", self.config.cone_half_angle_deg)
        self._add_entry(self.static_motion_frame, "Velocity jitter [m/s]", "velocity_jitter_m_per_s", self.config.velocity_jitter_m_per_s)

    def _build_gas_field_inputs(self, group: ttk.Frame) -> None:
        self.gas_field_frame = ttk.LabelFrame(group, text="Gas field", padding=8)
        self._add_gas_file(self.gas_field_frame)
        self._add_entry(self.gas_field_frame, "Birth gas z min [mm]", "source_birth_velocity_z_min_mm", self.config.source_birth_velocity_z_min_mm)
        self._add_entry(
            self.gas_field_frame,
            "Birth gas z max [mm] (blank = automatic)",
            "source_birth_velocity_z_max_mm",
            self.config.source_birth_velocity_z_max_mm,
            optional=True,
        )
        self._add_entry(
            self.gas_field_frame,
            "Birth gas radius [mm] (blank = automatic)",
            "source_birth_velocity_radius_mm",
            self.config.source_birth_velocity_radius_mm,
            optional=True,
        )
        self._add_entry(self.gas_field_frame, "Radial velocity scale", "source_radial_velocity_scale", self.config.source_radial_velocity_scale)
        self._add_entry(self.gas_field_frame, "Axial velocity delta [m/s]", "source_velocity_delta_m_per_s", self.config.source_velocity_delta_m_per_s)

    def _sync_velocity_frames(self) -> None:
        visibility = beam_velocity_visibility(self.vars["gas_velocity_init_mode"].get())
        self.static_motion_frame.pack_forget()
        self.gas_field_frame.pack_forget()
        if visibility["static_motion"]:
            self.static_motion_frame.pack(fill=tk.X, pady=(8, 0))
        if visibility["gas_field"]:
            self.gas_field_frame.pack(fill=tk.X, pady=(8, 0))

    def _build_buttons(self) -> None:
        ttk.Button(self.button_row, text="Apply", command=self._apply).pack(side=tk.RIGHT)
        ttk.Button(self.button_row, text="Cancel", command=self.top.destroy).pack(side=tk.RIGHT, padx=(0, 8))

    def _text(self, key: str) -> str:
        return str(self.vars[key].get()).strip()

    def _float(self, key: str) -> float:
        return float(self._text(key))

    def _optional_float(self, key: str) -> Optional[float]:
        value = self._text(key)
        return None if not value else float(value)

    def _int(self, key: str) -> int:
        return int(float(self._text(key)))

    def _ion_updates(self) -> dict[str, object]:
        return {
            "ion_name": self._text("ion_name"),
            "mass_amu": self._float("mass_amu"),
            "charge_e": self._int("charge_e"),
            "num_atoms": self._int("num_atoms"),
            "collision_cross_section_m2": self._float("collision_cross_section_m2"),
            "heat_capacity_profile": self._text("heat_capacity_profile"),
            "delta_h_kj_per_mol": self._float("delta_h_kj_per_mol"),
            "delta_s_j_per_mol_k": self._float("delta_s_j_per_mol_k"),
            "initial_internal_temperature_k": self._float("initial_internal_temperature_k"),
        }

    def _geometry_updates(self) -> dict[str, object]:
        return {
            "source_mode": self._text("source_mode"),
            "particle_count": self._int("particle_count"),
            "macro_particle_weight": self._float("macro_particle_weight"),
            "current_a": self._float("current_a"),
            "beam_radius_mm": self._float("beam_radius_mm"),
            "initial_x_mm": self._float("initial_x_mm"),
            "initial_y_mm": self._float("initial_y_mm"),
            "initial_z_mm": self._float("initial_z_mm"),
            "initial_position_jitter_mm": self._float("initial_position_jitter_mm"),
            "source_profile": self._text("source_profile"),
            "source_gaussian_sigma_mm": self._float("source_gaussian_sigma_mm"),
            "capillary_exit_z_mm": self._float("capillary_exit_z_mm"),
            "capillary_prefill_length_mm": self._float("capillary_prefill_length_mm"),
        }

    def _velocity_updates(self) -> dict[str, object]:
        mode = normalize_gas_velocity_mode(
            self._text("gas_velocity_init_mode")
        )
        updates: dict[str, object] = {
            "gas_velocity_init_mode": mode,
            "notes": self._text("notes"),
        }
        if mode == "static":
            updates.update(self._static_velocity_updates())
        else:
            updates.update(self._gas_field_velocity_updates())
        return updates

    def _static_velocity_updates(self) -> dict[str, object]:
        return {
            "kinetic_energy_ev": self._optional_float("kinetic_energy_ev"),
            "source_temperature_k": self._optional_float("source_temperature_k"),
            "source_axial_velocity_m_per_s": self._optional_float(
                "source_axial_velocity_m_per_s"
            ),
            "direction_axis": self._text("direction_axis"),
            "cone_half_angle_deg": self._float("cone_half_angle_deg"),
            "velocity_jitter_m_per_s": self._float(
                "velocity_jitter_m_per_s"
            ),
        }

    def _gas_field_velocity_updates(self) -> dict[str, object]:
        return {
            "source_birth_velocity_gas_csv": self._text(
                "source_birth_velocity_gas_csv"
            ),
            "source_birth_velocity_z_min_mm": self._float(
                "source_birth_velocity_z_min_mm"
            ),
            "source_birth_velocity_z_max_mm": self._optional_float(
                "source_birth_velocity_z_max_mm"
            ),
            "source_birth_velocity_radius_mm": self._optional_float(
                "source_birth_velocity_radius_mm"
            ),
            "source_radial_velocity_scale": self._float(
                "source_radial_velocity_scale"
            ),
            "source_velocity_delta_m_per_s": self._float(
                "source_velocity_delta_m_per_s"
            ),
        }

    @staticmethod
    def _validate(updates: dict[str, object]) -> None:
        positive = (
            "mass_amu",
            "particle_count",
            "macro_particle_weight",
            "collision_cross_section_m2",
            "delta_h_kj_per_mol",
            "initial_internal_temperature_k",
        )
        if any(float(updates[name]) <= 0.0 for name in positive):
            raise ValueError("Mass, count, weight, CCS, Delta H, and ion internal T must be positive.")
        if int(updates["charge_e"]) <= 0:
            raise ValueError("Charge state must be positive.")
        if int(updates["num_atoms"]) <= 2:
            raise ValueError("Number of atoms must be greater than 2.")
        if not str(updates["ion_name"]).strip() or not str(updates["heat_capacity_profile"]).strip():
            raise ValueError("Ion name and heat-capacity profile must not be empty.")
        nonnegative = (
            "current_a",
            "beam_radius_mm",
            "initial_position_jitter_mm",
            "source_gaussian_sigma_mm",
            "capillary_exit_z_mm",
            "capillary_prefill_length_mm",
        )
        if any(float(updates[name]) < 0.0 for name in nonnegative):
            raise ValueError("Source lengths, current, radii, and jitter must be non-negative.")
        mode = normalize_gas_velocity_mode(updates["gas_velocity_init_mode"])
        if mode == "static":
            BeamSetupDialog._validate_static_velocity(updates)
        else:
            BeamSetupDialog._validate_gas_field_velocity(updates)
        if not math.isfinite(float(updates["delta_s_j_per_mol_k"])):
            raise ValueError("Delta S must be finite.")

    @staticmethod
    def _validate_static_velocity(updates: dict[str, object]) -> None:
        if float(updates["velocity_jitter_m_per_s"]) < 0.0:
            raise ValueError("Velocity jitter must be non-negative.")
        angle = float(updates["cone_half_angle_deg"])
        if not 0.0 <= angle < 90.0:
            raise ValueError("Cone half-angle must be in [0, 90).")
        optional_positive = ("kinetic_energy_ev", "source_temperature_k")
        if any(updates[name] is not None and float(updates[name]) <= 0.0 for name in optional_positive):
            raise ValueError("Optional kinetic energy and source temperature must be positive.")
        axial = updates["source_axial_velocity_m_per_s"]
        if axial is not None and float(axial) < 0.0:
            raise ValueError("Source axial velocity must be non-negative.")

    @staticmethod
    def _validate_gas_field_velocity(updates: dict[str, object]) -> None:
        for name in (
            "source_birth_velocity_z_min_mm",
            "source_radial_velocity_scale",
        ):
            if float(updates[name]) < 0.0:
                raise ValueError("Gas-field z min and radial scale must be non-negative.")
        z_max = updates["source_birth_velocity_z_max_mm"]
        if z_max is not None and float(z_max) < float(updates["source_birth_velocity_z_min_mm"]):
            raise ValueError("Birth gas z max must be greater than or equal to z min.")
        radius = updates["source_birth_velocity_radius_mm"]
        if radius is not None and float(radius) <= 0.0:
            raise ValueError("Birth gas radius must be positive when provided.")
        if not math.isfinite(float(updates["source_velocity_delta_m_per_s"])):
            raise ValueError("Velocity delta must be finite.")

    def _apply(self) -> None:
        try:
            updates = self._ion_updates()
            updates.update(self._geometry_updates())
            updates.update(self._velocity_updates())
            self._validate(updates)
            config = replace(self.config, **updates)
        except Exception as exc:
            messagebox.showerror("Invalid Beam Config", str(exc), parent=self.top)
            return
        self.on_apply(config)
        self.top.destroy()


__all__ = [
    "BeamSetupDialog",
    "HEAT_CAPACITY_PROFILES",
    "beam_velocity_visibility",
    "normalize_gas_velocity_mode",
]
