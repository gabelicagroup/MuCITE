"""User actions and worker dispatch for the Tk main window."""

from __future__ import annotations

import copy
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from .config_io import load_app_config, save_app_config
from .beam_dialog import BeamSetupDialog
from .field_dialog import FieldBakerDialog
from .models import AppConfig, FieldBakeConfig, RuntimeConfig
from .runtime_dialogs import CollisionSettingsDialog, PicSettingsDialog
from .workers import (
    run_beam_smoke_task,
    run_field_bake_task,
    run_field_diagnostics_task,
    run_report_task,
    run_simulation_task,
    start_worker,
)


class _WindowSessionActionsMixin:
    def _new_session(self) -> None:
        self.config = AppConfig()
        self.beam_validated = False
        self.field_diagnostics_seen = False
        self.latest_snapshot = None
        self.snapshot_records = []
        self.snapshot_listbox.delete(0, tk.END)
        self._apply_config_to_vars()
        self._initialize_plot_tabs()
        self._append_console("Logs", "New GUI project created.")

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Project",
            filetypes=[("MuCITE project", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.config = load_app_config(Path(path))
        except Exception as exc:
            messagebox.showerror("Load Project Failed", str(exc), parent=self.root)
            return
        self.beam_validated = False
        self.field_diagnostics_seen = False
        self._apply_config_to_vars()
        self._append_console("Logs", f"Loaded project: {path}")

    def _save_config(self) -> None:
        try:
            self._sync_config_from_vars()
        except Exception as exc:
            messagebox.showerror("Invalid Project", str(exc), parent=self.root)
            return
        default_path = (
            Path(self.config.output_dir) / f"{self.config.session_name}.json"
        )
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Project",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            defaultextension=".json",
            filetypes=[("MuCITE project", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_app_config(Path(path), self.config)
        except Exception as exc:
            messagebox.showerror("Save Project Failed", str(exc), parent=self.root)
            return
        self._append_console("Logs", f"Saved project: {path}")

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(
            parent=self.root,
            title="Select output directory",
        )
        if path:
            self.output_dir_var.set(path)
            self.config.output_dir = path
            self.config.field_bake.output_npy = str(
                Path(path) / "baked_fields.npy"
            )
            self.config.field_bake.output_plot_dir = str(
                Path(path) / "field_baker_plots"
            )
            self._update_summaries()

    def _browse_static_field(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select baked field",
            filetypes=[
                ("NumPy baked field", "*.npy"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.static_field_var.set(path)
            self.config.loaded_baked_field_path = path
            self.config.state.loaded_baked_field_path = path
            self._set_status("FIELDS_BAKED")
            self._show_field_diagnostics()

    def _open_beam_dialog(self) -> None:
        BeamSetupDialog(
            self.root,
            self.config.beam,
            self._apply_beam_config,
        )

    def _apply_beam_config(self, config: Any) -> None:
        self.config.beam = config
        self.beam_validated = False
        self._update_summaries()
        self._set_status("CONFIGURED")
        self._append_console("Logs", "Beam configuration updated.")

    def _open_field_dialog(self) -> None:
        try:
            self._sync_config_from_vars()
        except Exception as exc:
            messagebox.showerror("Invalid Config", str(exc), parent=self.root)
            return
        FieldBakerDialog(
            self.root,
            self.config.field_bake,
            self.config.runtime,
            self._apply_field_config,
        )

    def _apply_field_config(
        self,
        config: FieldBakeConfig,
        runtime: RuntimeConfig,
    ) -> None:
        self.config.field_bake = config
        self.config.runtime = runtime
        self.field_diagnostics_seen = False
        self._apply_config_to_vars()
        self._update_summaries()
        self._append_console("Logs", "Field bake configuration updated.")

    def _open_pic_dialog(self) -> None:
        if not self._prepare_runtime_dialog():
            return
        PicSettingsDialog(
            self.root,
            self.config.runtime,
            self._apply_runtime_dialog_config,
        )

    def _open_collision_dialog(self) -> None:
        previous_mode = self.config.runtime.collision_mode
        if not self._prepare_runtime_dialog():
            self.collision_mode_var.set(previous_mode)
            return
        CollisionSettingsDialog(
            self.root,
            self.config.runtime,
            self._apply_runtime_dialog_config,
            lambda: self._restore_runtime_mode(
                "collision_mode",
                self.collision_mode_var,
                previous_mode,
            ),
        )

    def _on_collision_mode_selected(self, _event: object = None) -> None:
        self._open_collision_dialog()

    def _prepare_runtime_dialog(self) -> bool:
        try:
            self._sync_config_from_vars()
        except Exception as exc:
            messagebox.showerror("Invalid Runtime Config", str(exc), parent=self.root)
            return False
        return True

    def _apply_runtime_dialog_config(self, runtime: RuntimeConfig) -> None:
        self.config.runtime = runtime
        self._apply_config_to_vars()
        self._append_console("Logs", "Runtime model settings updated.")

    def _restore_runtime_mode(
        self,
        attribute: str,
        variable: tk.Variable,
        value: str,
    ) -> None:
        setattr(self.config.runtime, attribute, value)
        variable.set(value)
        self._update_summaries()

    def _export_current_figure(self) -> None:
        try:
            self._sync_config_from_vars()
            selected = self.plot_notebook.tab(
                self.plot_notebook.select(),
                "text",
            )
            figure = self.figures.get(str(selected))
            if figure is None:
                raise ValueError("No figure is active.")
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = str(selected).lower().replace(" ", "_")
            path = output_dir / f"{safe_name}_gui_export.png"
            figure.savefig(path, dpi=160)
            self._append_console("Logs", f"Exported figure: {path}")
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc), parent=self.root)

    def _open_output_dir(self) -> None:
        try:
            self._sync_config_from_vars()
            path = Path(self.config.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                self._append_console("Logs", f"Output directory: {path}")
        except Exception as exc:
            messagebox.showerror(
                "Open Directory Failed",
                str(exc),
                parent=self.root,
            )

    def _on_close(self) -> None:
        if getattr(self, "_close_pending", False):
            return
        worker = self.current_worker
        if worker is not None and worker.thread.is_alive():
            proceed = messagebox.askyesno(
                "Task Running",
                "A worker is still running. Stop and close?",
                parent=self.root,
            )
            if not proceed:
                return
            worker.stop_event.set()
            self._append_console(
                "Warnings",
                f"Stop requested for {worker.kind}; waiting for cleanup.",
            )
        self._close_pending = True
        self._closing_worker = worker
        self._finish_close_when_worker_stops()

    def _finish_close_when_worker_stops(self) -> None:
        worker = getattr(self, "_closing_worker", None)
        if worker is not None and worker.thread.is_alive():
            self.root.after(50, self._finish_close_when_worker_stops)
            return
        if worker is not None:
            worker.thread.join()
            if self.current_worker is worker:
                self.current_worker = None
        self._closing_worker = None
        self.root.destroy()


class _WindowWorkerActionsMixin:
    def _current_task_alive(self) -> bool:
        return self.current_worker is not None

    def _start_worker(self, kind: str, target: Any, *args: Any) -> None:
        if self._current_task_alive():
            messagebox.showwarning(
                "Task Running",
                "Wait for the current task to finish or press Stop.",
                parent=self.root,
            )
            return
        self.current_worker = start_worker(
            self.message_queue,
            kind,
            target,
            *args,
        )

    def _start_beam_smoke(self) -> None:
        try:
            self._sync_config_from_vars()
        except Exception as exc:
            messagebox.showerror("Invalid Config", str(exc), parent=self.root)
            return
        self._set_status("CONFIGURED")
        self._start_worker(
            "beam_smoke",
            run_beam_smoke_task,
            copy.deepcopy(self.config),
        )

    def _start_field_bake(self) -> None:
        try:
            self._sync_config_from_vars()
        except Exception as exc:
            messagebox.showerror("Invalid Config", str(exc), parent=self.root)
            return
        self._set_status("FIELDS_BAKING")
        self._start_worker(
            "field_bake",
            run_field_bake_task,
            copy.deepcopy(self.config),
        )

    def _show_field_diagnostics(self) -> None:
        try:
            self._sync_config_from_vars()
        except Exception as exc:
            messagebox.showerror(
                "Field Diagnostics Failed",
                str(exc),
                parent=self.root,
            )
            return
        self._set_status("FIELD_DIAGNOSTICS")
        self._start_worker(
            "field_diagnostics",
            run_field_diagnostics_task,
            copy.deepcopy(self.config),
        )

    def _start_simulation(self) -> None:
        try:
            self._sync_config_from_vars()
        except Exception as exc:
            messagebox.showerror(
                "Invalid Runtime Config",
                str(exc),
                parent=self.root,
            )
            return
        if not self.config.runtime.dummy_static_field:
            field_path = Path(self.config.loaded_baked_field_path)
            if not field_path.exists():
                messagebox.showerror(
                    "Missing Baked Field",
                    f"Baked field does not exist:\n{field_path}",
                    parent=self.root,
                )
                return
        if not self.beam_validated:
            proceed = messagebox.askyesno(
                "Beam Smoke Test Not Run",
                "Beam smoke test has not passed in this GUI project. "
                "Run simulation anyway?",
                parent=self.root,
            )
            if not proceed:
                return
        self.latest_snapshot = None
        self.last_progress_macro = -1
        self.last_progress_log_bin = -1
        self.exit_current_var.set("-")
        self.exit_ions_var.set("-")
        self._set_status("RUNNING")
        self._append_console("Logs", "Starting simulation worker.")
        self._start_worker(
            "simulation",
            run_simulation_task,
            copy.deepcopy(self.config),
        )

    def _stop_current_task(self) -> None:
        if self.current_worker is None:
            self._append_console("Logs", "No active worker to stop.")
            return
        self.current_worker.stop_event.set()
        self._append_console(
            "Warnings",
            f"Stop requested for {self.current_worker.kind}.",
        )

    def _start_report(self) -> None:
        if self.last_report_snapshot is None:
            messagebox.showwarning(
                "No Completed Run",
                "Run a simulation before generating a report.",
                parent=self.root,
            )
            return
        try:
            self._sync_config_from_vars()
        except Exception as exc:
            messagebox.showerror("Invalid Config", str(exc), parent=self.root)
            return
        self._start_worker(
            "report",
            run_report_task,
            copy.deepcopy(self.config),
            self.last_report_snapshot,
        )
