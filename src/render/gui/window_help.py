"""Scrollable Tk help windows for the MuCITE workbench."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk

from .help_content import (
    about_text,
    parameter_reference_text,
    simulation_workflow_text,
)


class _WindowHelpMixin:
    """Open non-modal reference windows owned by the main Tk window."""

    def _show_parameter_reference(self) -> None:
        self._open_help_window(
            "MuCITE Parameter Reference",
            parameter_reference_text(),
        )

    def _show_workflow_help(self) -> None:
        self._open_help_window(
            "MuCITE Simulation Workflow",
            simulation_workflow_text(),
        )

    def _show_about(self) -> None:
        self._open_help_window("About MuCITE", about_text(), width=76, height=28)

    def _open_help_window(
        self,
        title: str,
        content: str,
        *,
        width: int = 104,
        height: int = 38,
    ) -> None:
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("920x680")
        window.minsize(640, 420)
        window.transient(self.root)
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        viewer = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            width=width,
            height=height,
            padx=10,
            pady=10,
            font=("TkDefaultFont", 10),
        )
        viewer.pack(fill=tk.BOTH, expand=True)
        viewer.insert("1.0", content)
        viewer.configure(state=tk.DISABLED)
        ttk.Button(frame, text="Close", command=window.destroy).pack(
            anchor=tk.E,
            pady=(10, 0),
        )
        window.bind("<Escape>", lambda _event: window.destroy())
        viewer.focus_set()


__all__ = ["_WindowHelpMixin"]
