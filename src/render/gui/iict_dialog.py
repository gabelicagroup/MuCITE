"""Editable JSON and direct overrides for the independent IICT backend."""

from __future__ import annotations

import copy
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from .iict_config_validation import validate_iict_runtime_config
from .models import RuntimeConfig


def iict_parameter_summary(config: RuntimeConfig) -> str:
    """Return a compact source/model summary without loading either backend."""

    source = (
        Path(config.iict_parameter_config_path).name
        if config.iict_parameter_config_path.strip()
        else "direct GUI values"
    )
    heat = config.iict_heat_capacity_model or "JSON"
    pseudoatom = config.iict_pseudoatom_model or "JSON"
    fragmentation = config.iict_fragmentation_model or "JSON"
    return (
        f"source={source}; heat={heat}; pseudo-atom={pseudoatom}; "
        f"rate={fragmentation}"
    )


class IictSettingsDialog:
    """Edit iict-lite parameter source and optional GUI overrides."""

    def __init__(
        self,
        parent: tk.Misc,
        config: RuntimeConfig,
        on_apply: Callable[[RuntimeConfig], None],
    ) -> None:
        self.config = copy.deepcopy(config)
        self.on_apply = on_apply
        self.vars: dict[str, tk.StringVar] = {}
        self.top = tk.Toplevel(parent)
        self.top.title("iict-lite Parameters")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.geometry("720x570")
        self.top.minsize(620, 480)
        self._build()

    def _build(self) -> None:
        body = ttk.Frame(self.top, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            body,
            text=(
                "Blank override fields are read from the schema-v1 parameter "
                "JSON. Without JSON, choose complete models below."
            ),
            wraplength=670,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 8))
        self._build_parameter_source(body)
        notebook = ttk.Notebook(body)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self._build_heat_tab(notebook)
        self._build_pseudoatom_tab(notebook)
        self._build_fragmentation_tab(notebook)
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=self._apply).pack(side=tk.RIGHT)
        ttk.Button(
            buttons,
            text="Cancel",
            command=self.top.destroy,
        ).pack(side=tk.RIGHT, padx=(0, 8))

    def _build_parameter_source(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Parameter document", padding=10)
        group.pack(fill=tk.X)
        variable = tk.StringVar(value=self.config.iict_parameter_config_path)
        self.vars["iict_parameter_config_path"] = variable
        ttk.Label(group, text="Schema-v1 JSON (optional)").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Entry(group, textvariable=variable).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(10, 6),
        )
        ttk.Button(
            group,
            text="Browse...",
            command=lambda: self._browse_file(
                variable,
                "Select iict-lite parameter JSON",
                (("JSON", "*.json"), ("All files", "*.*")),
            ),
        ).grid(row=0, column=2)
        group.columnconfigure(1, weight=1)

    def _build_heat_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Heat Capacity")
        self._add_combo(
            tab,
            0,
            "Model (blank = JSON)",
            "iict_heat_capacity_model",
            self.config.iict_heat_capacity_model,
            ("", "classical", "constant_cv", "tabulated"),
        )
        self._add_optional_entry(
            tab, 1, "Atom count (blank = Beam)", "iict_num_atoms"
        )
        self._add_optional_entry(
            tab,
            2,
            "Constant Cv [J/K/ion]",
            "iict_constant_cv_j_per_k_per_ion",
        )
        self._add_path_entry(
            tab,
            3,
            "T,U or T,Cv CSV",
            "iict_heat_capacity_csv_path",
        )

    def _build_pseudoatom_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Pseudo-atom Mass")
        self._add_combo(
            tab,
            0,
            "Model (blank = JSON)",
            "iict_pseudoatom_model",
            self.config.iict_pseudoatom_model,
            ("", "constant", "tabulated"),
        )
        self._add_optional_entry(
            tab, 1, "Constant mass [Da]", "iict_pseudoatom_mass_da"
        )
        self._add_path_entry(
            tab,
            2,
            "T, relative speed, mass CSV",
            "iict_pseudoatom_csv_path",
        )
        self._add_optional_entry(
            tab, 3, "Minimum allowed mass [Da]", "iict_pseudoatom_min_mass_da"
        )
        self._add_optional_entry(
            tab, 4, "Maximum allowed mass [Da]", "iict_pseudoatom_max_mass_da"
        )

    def _build_fragmentation_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Fragmentation Rate")
        self._add_combo(
            tab,
            0,
            "Rate model (blank = JSON)",
            "iict_fragmentation_model",
            self.config.iict_fragmentation_model,
            ("", "none", "eyring"),
        )
        self._add_optional_entry(
            tab, 1, "Eyring delta H [kJ/mol]", "iict_delta_h_kj_per_mol"
        )
        self._add_optional_entry(
            tab, 2, "Eyring delta S [J/mol/K]", "iict_delta_s_j_per_mol_k"
        )
        ttk.Label(
            tab,
            text=(
                "This selects the molecular rate model. Fragment product "
                "handling (transport/loss/off) remains a separate runtime setting."
            ),
            wraplength=560,
            justify=tk.LEFT,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _add_combo(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        value: object,
        values: tuple[str, ...],
    ) -> None:
        shown = "" if value is None else str(value)
        variable = tk.StringVar(value=shown)
        self.vars[key] = variable
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)
        parent.columnconfigure(1, weight=1)

    def _add_optional_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
    ) -> None:
        value = getattr(self.config, key)
        variable = tk.StringVar(value="" if value is None else str(value))
        self.vars[key] = variable
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            columnspan=2,
            sticky="ew",
            padx=(10, 0),
            pady=5,
        )

    def _add_path_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
    ) -> None:
        variable = tk.StringVar(value=str(getattr(self.config, key)))
        self.vars[key] = variable
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=(10, 6),
            pady=5,
        )
        ttk.Button(
            parent,
            text="Browse...",
            command=lambda: self._browse_file(
                variable,
                "Select iict-lite table",
                (("CSV", "*.csv"), ("All files", "*.*")),
            ),
        ).grid(row=row, column=2, pady=5)
        parent.columnconfigure(1, weight=1)

    def _browse_file(
        self,
        variable: tk.StringVar,
        title: str,
        filetypes: tuple[tuple[str, str], ...],
    ) -> None:
        path = filedialog.askopenfilename(
            parent=self.top,
            title=title,
            filetypes=filetypes,
        )
        if path:
            variable.set(path)

    def _apply(self) -> None:
        try:
            self._read_config()
            validate_iict_runtime_config(self.config)
        except Exception as exc:
            messagebox.showerror(
                "Invalid iict-lite Parameters",
                str(exc),
                parent=self.top,
            )
            return
        self.on_apply(self.config)
        self.top.destroy()

    def _read_config(self) -> None:
        text = lambda key: self.vars[key].get().strip()
        optional_float = lambda key: None if not text(key) else float(text(key))
        optional_int = lambda key: None if not text(key) else int(float(text(key)))
        optional_model = lambda key: text(key) or None
        self.config.iict_parameter_config_path = text(
            "iict_parameter_config_path"
        )
        self.config.iict_heat_capacity_model = optional_model(
            "iict_heat_capacity_model"
        )
        self.config.iict_num_atoms = optional_int("iict_num_atoms")
        self.config.iict_constant_cv_j_per_k_per_ion = optional_float(
            "iict_constant_cv_j_per_k_per_ion"
        )
        self.config.iict_heat_capacity_csv_path = text(
            "iict_heat_capacity_csv_path"
        )
        self.config.iict_pseudoatom_model = optional_model(
            "iict_pseudoatom_model"
        )
        self.config.iict_pseudoatom_mass_da = optional_float(
            "iict_pseudoatom_mass_da"
        )
        self.config.iict_pseudoatom_csv_path = text(
            "iict_pseudoatom_csv_path"
        )
        self.config.iict_pseudoatom_min_mass_da = optional_float(
            "iict_pseudoatom_min_mass_da"
        )
        self.config.iict_pseudoatom_max_mass_da = optional_float(
            "iict_pseudoatom_max_mass_da"
        )
        self.config.iict_fragmentation_model = optional_model(
            "iict_fragmentation_model"
        )
        self.config.iict_delta_h_kj_per_mol = optional_float(
            "iict_delta_h_kj_per_mol"
        )
        self.config.iict_delta_s_j_per_mol_k = optional_float(
            "iict_delta_s_j_per_mol_k"
        )


__all__ = ["IictSettingsDialog", "iict_parameter_summary"]
