"""Terminal-current table layout kept separate from the main window shell."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


TERMINAL_COLUMNS = (
    "interval",
    "active",
    "exit_current",
    "exit_ions",
    "exit_macro",
    "electrode",
    "other_loss",
)

TERMINAL_HEADINGS = (
    ("interval", "time [ms]", 120),
    ("active", "active weighted packs", 145),
    ("exit_current", "exit [nA]", 90),
    ("exit_ions", "exit real ions", 110),
    ("exit_macro", "exit weighted packs", 140),
    ("electrode", "electrode packs", 110),
    ("other_loss", "other loss packs", 110),
)


class _WindowTerminalLayoutMixin:
    def _build_terminal_console(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=4)
        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(controls, text="Time bin [ms]").pack(side=tk.LEFT)
        selector = ttk.Combobox(
            controls,
            textvariable=self.terminal_time_bin_var,
            values=("0.2", "1.0"),
            state="readonly",
            width=8,
        )
        selector.pack(side=tk.LEFT, padx=(6, 0))
        selector.bind("<<ComboboxSelected>>", self._refresh_terminal_report)
        self.terminal_tree = ttk.Treeview(
            frame, columns=TERMINAL_COLUMNS, show="headings", height=7
        )
        for key, label, width in TERMINAL_HEADINGS:
            self.terminal_tree.heading(key, text=label)
            self.terminal_tree.column(key, width=width, anchor=tk.E)
        self.terminal_tree.pack(fill=tk.BOTH, expand=True)
        notebook.add(frame, text="Terminal Report")
