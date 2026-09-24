"""Focused PIC and collision configuration dialogs."""

from __future__ import annotations

import copy
import math
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from .iict_config_validation import validate_iict_runtime_config
from .iict_dialog import IictSettingsDialog, iict_parameter_summary
from .models import RuntimeConfig


def pic_settings_visibility(backend: object) -> dict[str, bool]:
    solver = str(backend).strip().lower()
    return {"amg": solver == "amg"}


def collision_settings_visibility(mode: object) -> dict[str, bool]:
    canonical = str(mode).strip().lower()
    if canonical not in {"explicit", "hybrid-langevin"}:
        raise ValueError("Collision mode must be 'explicit' or 'hybrid-langevin'.")
    return {"hybrid": canonical == "hybrid-langevin"}


def collision_backend_visibility(backend: object) -> dict[str, bool]:
    canonical = str(backend).strip().lower()
    if canonical not in {"ionspa", "iict-lite"}:
        raise ValueError("Collision physics backend must be 'ionspa' or 'iict-lite'.")
    return {
        "ionspa": canonical == "ionspa",
        "iict": canonical == "iict-lite",
    }


def _entry(
    parent: ttk.Frame,
    variables: dict[str, tk.Variable],
    row: int,
    label: str,
    key: str,
    value: object,
) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
    variable = tk.StringVar(value=str(value))
    variables[key] = variable
    ttk.Entry(parent, textvariable=variable).grid(
        row=row, column=1, sticky="ew", padx=(8, 0), pady=3
    )
    parent.columnconfigure(1, weight=1)


def _combo(
    parent: ttk.Frame,
    variables: dict[str, tk.Variable],
    row: int,
    label: str,
    key: str,
    value: object,
    values: tuple[str, ...],
) -> ttk.Combobox:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
    variable = tk.StringVar(value=str(value))
    variables[key] = variable
    widget = ttk.Combobox(
        parent, textvariable=variable, values=values, state="readonly"
    )
    widget.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
    parent.columnconfigure(1, weight=1)
    return widget


def _dialog_buttons(
    parent: ttk.Frame,
    apply_command: Callable[[], None],
    cancel_command: Callable[[], None],
) -> None:
    holder = ttk.Frame(parent)
    holder.pack(fill=tk.X, pady=(12, 0))
    ttk.Button(holder, text="Apply", command=apply_command).pack(side=tk.RIGHT)
    ttk.Button(holder, text="Cancel", command=cancel_command).pack(
        side=tk.RIGHT, padx=(0, 8)
    )


class PicSettingsDialog:
    """Edit PIC scale, grid shape, and modern Poisson/AMG controls."""

    def __init__(
        self,
        parent: tk.Misc,
        config: RuntimeConfig,
        on_apply: Callable[[RuntimeConfig], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.on_apply = on_apply
        self.on_cancel = on_cancel
        self.vars: dict[str, tk.Variable] = {}
        self.top = tk.Toplevel(parent)
        self.top.title("PIC / Poisson Settings")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)
        self.body = ttk.Frame(self.top, padding=12)
        self.body.pack(fill=tk.BOTH, expand=True)
        self._build()

    def _build(self) -> None:
        mode = ttk.LabelFrame(self.body, text="PIC Field", padding=10)
        mode.pack(fill=tk.X)
        _entry(
            mode,
            self.vars,
            0,
            "Space-charge scale (0 = off)",
            "pic_space_charge_scale",
            self.config.pic_space_charge_scale,
        )
        ttk.Label(
            mode,
            text="Use nr=nz=0 for the shared static grid; use a pair >= 3 for a decoupled grid.",
            wraplength=460,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self._build_grid_frame()
        self._build_solver_frame()
        self._update_visibility()
        _dialog_buttons(self.body, self._apply, self._cancel)

    def _build_grid_frame(self) -> None:
        self.grid_frame = ttk.LabelFrame(
            self.body, text="PIC Grid", padding=10
        )
        self.grid_frame.pack(fill=tk.X, pady=(10, 0))
        _entry(
            self.grid_frame, self.vars, 0, "Radial nodes (nr)", "pic_grid_nr", self.config.pic_grid_nr
        )
        _entry(
            self.grid_frame, self.vars, 1, "Axial nodes (nz)", "pic_grid_nz", self.config.pic_grid_nz
        )

    def _build_solver_frame(self) -> None:
        solver = ttk.LabelFrame(self.body, text="Poisson Solver", padding=10)
        solver.pack(fill=tk.X, pady=(10, 0))
        backend = _combo(
            solver,
            self.vars,
            0,
            "Backend",
            "pic_poisson_backend",
            self.config.pic_poisson_backend,
            ("amg", "sparse_direct", "sparse_cg", "sparse_bicgstab"),
        )
        _combo(
            solver,
            self.vars,
            1,
            "Sparse preconditioner",
            "pic_poisson_preconditioner",
            self.config.pic_poisson_preconditioner,
            ("none", "jacobi"),
        )
        self.vars["pic_poisson_warm_start"] = tk.BooleanVar(
            value=self.config.pic_poisson_warm_start
        )
        ttk.Checkbutton(
            solver,
            text="Reuse previous potential (warm start)",
            variable=self.vars["pic_poisson_warm_start"],
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        backend.bind("<<ComboboxSelected>>", self._update_visibility)
        self._build_amg_frame()

    def _build_amg_frame(self) -> None:
        self.amg_frame = ttk.LabelFrame(self.body, text="AMG", padding=10)
        self.amg_frame.pack(fill=tk.X, pady=(10, 0))
        rows = (
            ("Mode", "pic_amg_mode", self.config.pic_amg_mode, ("solve", "preconditioned_cg", "preconditioned_bicgstab")),
            ("Hierarchy", "pic_amg_solver", self.config.pic_amg_solver, ("ruge_stuben", "smoothed_aggregation")),
            ("Fallback", "pic_amg_fallback_backend", self.config.pic_amg_fallback_backend, ("sparse_direct", "sparse_cg", "sparse_bicgstab", "sparse_spsolve")),
        )
        for row, (label, key, value, choices) in enumerate(rows):
            _combo(self.amg_frame, self.vars, row, label, key, value, choices)
        _entry(self.amg_frame, self.vars, 3, "Relative tolerance", "pic_amg_tolerance", self.config.pic_amg_tolerance)
        _entry(self.amg_frame, self.vars, 4, "Maximum iterations", "pic_amg_max_iters", self.config.pic_amg_max_iters)

    def _update_visibility(self, _event: object = None) -> None:
        visibility = pic_settings_visibility(
            self.vars["pic_poisson_backend"].get(),
        )
        if visibility["amg"]:
            self.amg_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.amg_frame.pack_forget()

    def _apply(self) -> None:
        try:
            self._read_config()
            nodes = (self.config.pic_grid_nr, self.config.pic_grid_nz)
            if nodes != (0, 0) and not all(value >= 3 for value in nodes):
                raise ValueError("PIC grid nodes must be 0/0 (shared) or both >= 3.")
            if (
                not math.isfinite(self.config.pic_space_charge_scale)
                or self.config.pic_space_charge_scale < 0.0
            ):
                raise ValueError("PIC space-charge scale must be finite and non-negative.")
            if self.config.pic_amg_tolerance <= 0.0:
                raise ValueError("AMG tolerance must be positive.")
            if self.config.pic_amg_max_iters <= 0:
                raise ValueError("AMG maximum iterations must be positive.")
        except Exception as exc:
            messagebox.showerror("Invalid PIC Settings", str(exc), parent=self.top)
            return
        self.on_apply(self.config)
        self.top.destroy()

    def _cancel(self) -> None:
        if self.on_cancel is not None:
            self.on_cancel()
        self.top.destroy()

    def _read_config(self) -> None:
        get = lambda key: str(self.vars[key].get()).strip()
        self.config.pic_space_charge_scale = float(get("pic_space_charge_scale"))
        self.config.pic_grid_nr = int(float(get("pic_grid_nr")))
        self.config.pic_grid_nz = int(float(get("pic_grid_nz")))
        self.config.pic_poisson_backend = get("pic_poisson_backend")
        self.config.pic_poisson_preconditioner = get("pic_poisson_preconditioner")
        self.config.pic_poisson_warm_start = bool(self.vars["pic_poisson_warm_start"].get())
        self.config.pic_amg_mode = get("pic_amg_mode")
        self.config.pic_amg_solver = get("pic_amg_solver")
        self.config.pic_amg_tolerance = float(get("pic_amg_tolerance"))
        self.config.pic_amg_max_iters = int(float(get("pic_amg_max_iters")))
        self.config.pic_amg_fallback_backend = get("pic_amg_fallback_backend")


class CollisionSettingsDialog:
    """Edit explicit or hybrid collision settings with mode-specific controls."""

    def __init__(
        self,
        parent: tk.Misc,
        config: RuntimeConfig,
        on_apply: Callable[[RuntimeConfig], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.on_apply = on_apply
        self.on_cancel = on_cancel
        self.vars: dict[str, tk.Variable] = {}
        self.iict_summary = tk.StringVar(
            master=parent, value=iict_parameter_summary(self.config)
        )
        self.top = tk.Toplevel(parent)
        self.top.title("Collision Settings")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)
        self.body = ttk.Frame(self.top, padding=12)
        self.body.pack(fill=tk.BOTH, expand=True)
        self._build()

    def _build(self) -> None:
        general = ttk.LabelFrame(self.body, text="Collision Model", padding=10)
        general.pack(fill=tk.X)
        mode = _combo(
            general, self.vars, 0, "Mode", "collision_mode",
            self.config.collision_mode, ("explicit", "hybrid-langevin")
        )
        backend = _combo(
            general,
            self.vars,
            1,
            "Single-collision physics",
            "collision_physics_backend",
            self.config.collision_physics_backend,
            ("ionspa", "iict-lite"),
        )
        self._build_backend_frames(general)
        _combo(
            general, self.vars, 3, "Fragment product handling", "fragmentation_mode",
            self.config.fragmentation_mode, ("transport", "loss", "off")
        )
        _entry(
            general, self.vars, 4, "Batch size", "collision_batch_size",
            self.config.collision_batch_size
        )
        mode.bind("<<ComboboxSelected>>", self._update_visibility)
        backend.bind("<<ComboboxSelected>>", self._update_visibility)
        self._build_hybrid_frame()
        self._update_visibility()
        _dialog_buttons(self.body, self._apply, self._cancel)

    def _build_backend_frames(self, parent: ttk.Frame) -> None:
        self.ionspa_frame = ttk.Frame(parent)
        self.ionspa_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(3, 0),
        )
        _combo(
            self.ionspa_frame,
            self.vars,
            0,
            "IonSPA provider",
            "ionspa_backend",
            self.config.ionspa_backend,
            ("local", "approximate"),
        )
        self.iict_frame = ttk.Frame(parent)
        self.iict_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(3, 0),
        )
        ttk.Button(
            self.iict_frame,
            text="iict-lite parameters...",
            command=self._open_iict_settings,
        ).pack(side=tk.LEFT)
        ttk.Label(
            self.iict_frame,
            textvariable=self.iict_summary,
            wraplength=330,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    def _build_hybrid_frame(self) -> None:
        self.hybrid_frame = ttk.LabelFrame(
            self.body, text="Hybrid / Langevin", padding=10
        )
        self.hybrid_frame.pack(fill=tk.X, pady=(10, 0))
        rows = (
            ("z start [mm]", "langevin_z_start_mm", self.config.langevin_z_start_mm),
            ("z end [mm]", "langevin_z_end_mm", self.config.langevin_z_end_mm),
            ("Switch probability", "langevin_switch_prob", self.config.langevin_switch_prob),
            ("Maximum dt [s]", "langevin_max_dt_s", self.config.langevin_max_dt_s),
        )
        for row, (label, key, value) in enumerate(rows):
            _entry(self.hybrid_frame, self.vars, row, label, key, value)

    def _update_visibility(self, _event: object = None) -> None:
        visibility = collision_settings_visibility(self.vars["collision_mode"].get())
        if visibility["hybrid"]:
            self.hybrid_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.hybrid_frame.pack_forget()
        backend = collision_backend_visibility(
            self.vars["collision_physics_backend"].get()
        )
        if backend["ionspa"]:
            self.ionspa_frame.grid()
            self.iict_frame.grid_remove()
        else:
            self.iict_frame.grid()
            self.ionspa_frame.grid_remove()

    def _open_iict_settings(self) -> None:
        self.config.collision_physics_backend = str(
            self.vars["collision_physics_backend"].get()
        ).strip()
        IictSettingsDialog(
            self.top,
            self.config,
            self._apply_iict_settings,
        )

    def _apply_iict_settings(self, config: RuntimeConfig) -> None:
        self.config = config
        self.iict_summary.set(iict_parameter_summary(config))

    def _apply(self) -> None:
        try:
            self._read_config()
            validate_iict_runtime_config(self.config)
            if self.config.collision_batch_size < 0:
                raise ValueError("Collision batch size cannot be negative.")
            if self.config.collision_mode == "hybrid-langevin":
                if self.config.langevin_z_end_mm < self.config.langevin_z_start_mm:
                    raise ValueError("Hybrid z end must be >= z start.")
                if not 0.0 <= self.config.langevin_switch_prob <= 1.0:
                    raise ValueError("Switch probability must be in [0, 1].")
                if self.config.langevin_max_dt_s <= 0.0:
                    raise ValueError("Hybrid maximum dt must be positive.")
        except Exception as exc:
            messagebox.showerror("Invalid Collision Settings", str(exc), parent=self.top)
            return
        self.on_apply(self.config)
        self.top.destroy()

    def _cancel(self) -> None:
        if self.on_cancel is not None:
            self.on_cancel()
        self.top.destroy()

    def _read_config(self) -> None:
        get = lambda key: str(self.vars[key].get()).strip()
        self.config.collision_mode = get("collision_mode")
        self.config.collision_physics_backend = get(
            "collision_physics_backend"
        )
        self.config.ionspa_backend = get("ionspa_backend")
        self.config.fragmentation_mode = get("fragmentation_mode")
        self.config.collision_batch_size = int(float(get("collision_batch_size")))
        self.config.langevin_z_start_mm = float(get("langevin_z_start_mm"))
        self.config.langevin_z_end_mm = float(get("langevin_z_end_mm"))
        self.config.langevin_switch_prob = float(get("langevin_switch_prob"))
        self.config.langevin_max_dt_s = float(get("langevin_max_dt_s"))
