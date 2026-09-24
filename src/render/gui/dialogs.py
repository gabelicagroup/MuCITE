"""Shared Tk dialog primitives and lazy compatibility exports."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional


class _DialogBase:
    """Small modal shell shared by focused presentation dialogs."""

    def __init__(self, parent: tk.Misc, title: str) -> None:
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(0, weight=1)
        self.body = ttk.Frame(self.top, padding=12)
        self.body.grid(row=0, column=0, sticky="nsew")
        self.vars: dict[str, tk.Variable] = {}

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        value: object,
        width: int = 18,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        variable = tk.StringVar(value=str(value))
        self.vars[key] = variable
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row, column=1, sticky="ew", padx=(10, 0), pady=3
        )
        parent.columnconfigure(1, weight=1)

    def _add_combo(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        key: str,
        value: object,
        values: tuple[str, ...],
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        variable = tk.StringVar(value=str(value))
        self.vars[key] = variable
        ttk.Combobox(
            parent, textvariable=variable, values=values, state="readonly"
        ).grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        parent.columnconfigure(1, weight=1)

    def _buttons(self, command: Callable[[], None]) -> None:
        holder = ttk.Frame(self.body)
        holder.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(holder, text="Apply", command=command).pack(side=tk.RIGHT)
        ttk.Button(holder, text="Cancel", command=self.top.destroy).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

    def _get_float(self, key: str) -> float:
        return float(str(self.vars[key].get()).strip())

    def _get_optional_float(self, key: str) -> Optional[float]:
        text = str(self.vars[key].get()).strip()
        return None if text == "" else float(text)

    def _get_int(self, key: str) -> int:
        return int(float(str(self.vars[key].get()).strip()))

    def _get_str(self, key: str) -> str:
        return str(self.vars[key].get()).strip()


def __getattr__(name: str) -> Any:
    """Keep historical dialog imports without retaining duplicate classes."""

    if name == "BeamSetupDialog":
        from .beam_dialog import BeamSetupDialog

        return BeamSetupDialog
    if name == "FieldBakerDialog":
        from .field_dialog import FieldBakerDialog

        return FieldBakerDialog
    raise AttributeError(name)


__all__ = ["BeamSetupDialog", "FieldBakerDialog"]
