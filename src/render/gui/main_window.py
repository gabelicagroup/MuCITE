"""Compatibility facade and composition root for the Tk GUI window."""

from __future__ import annotations

# Preserve the historical public import surface of this module.
import copy
import csv
import os
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config_io import load_app_config, save_app_config
from .beam_dialog import BeamSetupDialog
from .field_dialog import FieldBakerDialog
from .models import AppConfig, FieldBakeConfig
from .plotting import (
    make_beam_preview_figure,
    make_diagnostic_image_figure,
    make_empty_figure,
    make_field_preview_figure,
    make_snapshot_figure,
    make_summary_figure,
)
from .window_actions import (
    _WindowSessionActionsMixin,
    _WindowWorkerActionsMixin,
)
from .window_layout import PLOT_TABS, ScrollableFrame, _WindowLayoutMixin
from .window_help import _WindowHelpMixin
from .window_results import _WindowMessageMixin, _WindowSnapshotMixin
from .window_runtime_layout import _RuntimeLayoutMixin
from .window_state import _WindowStateMixin
from .window_terminal_layout import _WindowTerminalLayoutMixin
from .window_terminal_results import _WindowTerminalResultsMixin
from .window_variables import _WindowVariablesMixin
from .workers import (
    WorkerHandle,
    run_beam_smoke_task,
    run_field_bake_task,
    run_field_diagnostics_task,
    run_report_task,
    run_simulation_task,
    start_worker,
)


class MainWindow(
    _WindowVariablesMixin,
    _WindowTerminalLayoutMixin,
    _WindowLayoutMixin,
    _WindowHelpMixin,
    _RuntimeLayoutMixin,
    _WindowStateMixin,
    _WindowSessionActionsMixin,
    _WindowWorkerActionsMixin,
    _WindowMessageMixin,
    _WindowTerminalResultsMixin,
    _WindowSnapshotMixin,
):
    """Main graphical workflow window."""

    def __init__(self) -> None:
        self.config = AppConfig()
        self.message_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.current_worker: Optional[WorkerHandle] = None
        self.beam_validated = False
        self.field_diagnostics_seen = False
        self.latest_snapshot: Optional[np.ndarray] = None
        self.snapshot_revision = 0
        self.snapshot_rendered_revision: dict[str, int] = {}
        self.snapshot_render_pending = False
        self.snapshot_records: list[dict[str, str]] = []
        self.last_progress_macro = -1
        self.last_progress_log_bin = -1
        self.last_template: Any = None
        self.last_sim_config: Any = None
        self.last_report_snapshot: Any = None
        self.last_result: Any = None
        self.last_terminal_rows: list[dict[str, Any]] = []
        self.root = tk.Tk()
        self.root.title("MuCITE Ion Simulation Workbench")
        self.root.geometry("1440x900")
        self.root.minsize(1120, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.figures: dict[str, Figure] = {}
        self.canvases: dict[str, FigureCanvasTkAgg] = {}
        self.plot_frames: dict[str, ttk.Frame] = {}
        self.console_tabs: dict[str, scrolledtext.ScrolledText] = {}
        self._build_vars()
        self._build_layout()
        self._apply_config_to_vars()
        self._initialize_plot_tabs()
        self.root.after(100, self._poll_messages)

    def run(self) -> None:
        self.root.mainloop()


def launch_main_window() -> None:
    MainWindow().run()
