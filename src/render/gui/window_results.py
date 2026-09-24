"""Worker-message routing and result visualization for the Tk window."""

from __future__ import annotations

import csv
import queue
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox
from typing import Any

import numpy as np

from .plotting import (
    make_beam_preview_figure,
    make_diagnostic_image_figure,
    make_snapshot_figure,
    make_summary_figure,
)


SNAPSHOT_PLOT_TABS = frozenset(
    {"Snapshot", "XY", "RZ", "Phase Space", "Temperature", "Energy", "Loss Map"}
)
MAX_LIVE_SNAPSHOT_RECORDS = 5_000


class _WindowMessageMixin:
    def _poll_messages(self) -> None:
        while True:
            try:
                message = self.message_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_message(message)
        self.root.after(100, self._poll_messages)

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = str(message.get("type", ""))
        kind = str(message.get("kind", ""))
        if message_type == "task_started":
            self._append_console("Logs", f"Task started: {kind}")
        elif message_type == "task_done":
            if (
                self.current_worker is not None
                and self.current_worker.kind == kind
            ):
                self.current_worker = None
        elif message_type == "task_stopped":
            self._set_status("STOPPED")
            self._append_console("Warnings", f"Task stopped: {kind}")
        elif message_type == "task_failed":
            self._set_status("FAILED")
            error = str(message.get("error", ""))
            self._append_console("Warnings", error)
            messagebox.showerror("Task Failed", error, parent=self.root)
        elif message_type == "log":
            channel = str(message.get("channel", "Logs")).title()
            if channel not in self.console_tabs:
                channel = "Logs"
            self._append_console(channel, str(message.get("text", "")))
        elif message_type == "beam_smoke_finished":
            self._handle_beam_smoke_finished(message)
        elif message_type == "field_bake_finished":
            self._handle_field_bake_finished(message)
        elif message_type == "field_diagnostics_finished":
            self._handle_field_diagnostics_finished(message)
        elif message_type == "simulation_progress":
            self._handle_simulation_progress(message)
        elif message_type == "snapshot_ready":
            self._handle_snapshot_ready(message)
        elif message_type == "simulation_finished":
            self._handle_simulation_finished(message)
        elif message_type == "report_finished":
            paths = message.get("report_paths", {})
            self._append_console(
                "Run Summary",
                f"Report written: {paths}",
            )

    def _handle_beam_smoke_finished(
        self,
        message: dict[str, Any],
    ) -> None:
        sample = message["sample"]
        stats = message["stats"]
        self.beam_validated = True
        self._set_status("BEAM_VALIDATED")
        self._set_figure("Snapshot", make_beam_preview_figure(sample))
        self._append_console(
            "Validation",
            "Beam smoke test passed: sample_count={sample_count}, "
            "mean KE={energy_ev_mean:.6g} eV, "
            "p95 KE={energy_ev_p95:.6g} eV, "
            "p95 angle={angle_deg_p95:.6g} deg".format(**stats),
        )
        self._append_console(
            "Logs",
            "Beam smoke outputs: "
            f"{stats['sample_path']}, {stats['preview_path']}",
        )

    def _handle_field_bake_finished(
        self,
        message: dict[str, Any],
    ) -> None:
        result = dict(message.get("result", {}))
        output_npy = str(
            result.get("output_npy", self.config.field_bake.output_npy)
        )
        self.config.field_bake.output_npy = output_npy
        self.config.loaded_baked_field_path = output_npy
        self.static_field_var.set(output_npy)
        self._append_console(
            "Validation",
            f"Field bake completed: {output_npy}",
        )
        for path in result.get("plot_paths", []):
            self._append_console("Logs", f"Diagnostic plot: {path}")
        self._show_field_diagnostics()

    def _handle_field_diagnostics_finished(
        self,
        message: dict[str, Any],
    ) -> None:
        path = Path(str(message["path"]))
        self._set_figure("Snapshot", make_diagnostic_image_figure(path))
        revisions = getattr(self, "snapshot_rendered_revision", {})
        revisions["Snapshot"] = getattr(self, "snapshot_revision", 0)
        self.snapshot_rendered_revision = revisions
        snapshot_frame = self.plot_frames.get("Snapshot")
        if snapshot_frame is not None:
            self.plot_notebook.select(snapshot_frame)
        self.field_diagnostics_seen = True
        self._set_status("READY_TO_RUN" if self.beam_validated else "FIELDS_BAKED")
        self._append_console("Validation", f"Global (r,z) field diagnostic: {path}")
        summary_path = str(message.get("summary_path", ""))
        if summary_path:
            self._append_console("Logs", f"Alignment summary: {summary_path}")

    def _handle_simulation_progress(
        self,
        message: dict[str, Any],
    ) -> None:
        progress = dict(message.get("progress", {}))
        macro_step = int(progress.get("macro_step_index", 0))
        self.macro_var.set(str(macro_step))
        self.time_var.set(
            f"{float(progress.get('time_s', 0.0)) * 1.0e3:.6g} ms"
        )
        self.alive_var.set(str(int(progress.get("ion_count", 0))))
        self.collisions_var.set(
            str(int(progress.get("collision_count", 0)))
        )
        bin_ms = max(float(self.config.output.terminal_time_bin_ms), 1.0e-9)
        progress_bin = int(float(progress.get("time_s", 0.0)) * 1.0e3 / bin_ms)
        if progress_bin != getattr(self, "last_progress_log_bin", -1):
            self._append_console(
                "Logs",
                "t={:.3f} ms | active weighted ion packs={} | cumulative collisions={}".format(
                    float(progress.get("time_s", 0.0)) * 1.0e3,
                    int(progress.get("ion_count", 0)),
                    int(progress.get("collision_count", 0)),
                ),
            )
            self.last_progress_log_bin = progress_bin
        self.last_progress_macro = macro_step

    def _handle_snapshot_ready(self, message: dict[str, Any]) -> None:
        snapshot = np.asarray(message["snapshot"], dtype=np.float64)
        self.latest_snapshot = snapshot
        storage_ref = str(message.get("storage_ref", ""))
        label = (
            f"macro {message.get('macro_step', '?')}  "
            f"t={float(message.get('time_s', 0.0)) * 1.0e3:.6g} ms"
        )
        self.snapshot_records.append(
            {"label": label, "storage_ref": storage_ref}
        )
        self.snapshot_listbox.insert(tk.END, label)
        overflow = len(self.snapshot_records) - MAX_LIVE_SNAPSHOT_RECORDS
        if overflow > 0:
            del self.snapshot_records[:overflow]
            self.snapshot_listbox.delete(0, overflow - 1)
        self.config.state.latest_snapshot_path = storage_ref
        self.snapshot_revision = getattr(self, "snapshot_revision", 0) + 1
        if self.auto_refresh_var.get():
            self._schedule_snapshot_render()

    def _handle_simulation_finished(
        self,
        message: dict[str, Any],
    ) -> None:
        self.last_template = message.get("template")
        self.last_sim_config = message.get("config")
        self.last_report_snapshot = message.get("report_snapshot")
        self.last_result = message.get("result")
        result = self.last_result
        stopped = (
            result is not None
            and str(getattr(result, "termination_reason", "")) == "cancelled"
        )
        self._set_status("STOPPED" if stopped else "FINISHED")
        if result is not None:
            self._set_figure(
                "Snapshot",
                make_summary_figure(result.macro_history),
            )
            self._append_console(
                "Run Summary",
                "Simulation {}: backend={}, survivors={}, "
                "collisions={}, termination={}, export_dir={}".format(
                    "stopped" if stopped else "finished",
                    result.particle_backend,
                    len(result.survivors),
                    result.collision_count,
                    result.termination_reason,
                    result.export_directory,
                ),
            )
        report_paths = message.get("report_paths", {})
        self._update_terminal_report(
            self.last_report_snapshot,
            report_paths,
            message.get("terminal_bins_by_ms", {}),
        )
        report_path = report_paths.get("report") if report_paths else None
        if report_path:
            self._append_console(
                "Run Summary",
                f"Detailed report: {report_path}",
            )
        self._load_snapshot_index(silent=True)


class _WindowSnapshotMixin:
    def _selected_plot_tab(self) -> str:
        try:
            return str(self.plot_notebook.tab(self.plot_notebook.select(), "text"))
        except Exception:
            return "Snapshot"

    def _schedule_snapshot_render(self) -> None:
        if getattr(self, "snapshot_render_pending", False):
            return
        self.snapshot_render_pending = True
        try:
            self.root.after(50, self._render_latest_snapshot)
        except (AttributeError, tk.TclError):
            self._render_latest_snapshot()

    def _render_latest_snapshot(self) -> None:
        self.snapshot_render_pending = False
        if self.latest_snapshot is None:
            return
        tab = self._selected_plot_tab()
        if tab not in SNAPSHOT_PLOT_TABS:
            return
        revisions = getattr(self, "snapshot_rendered_revision", {})
        if revisions.get(tab) == getattr(self, "snapshot_revision", 0):
            return
        self._display_snapshot(self.latest_snapshot, tab)
        revisions[tab] = getattr(self, "snapshot_revision", 0)
        self.snapshot_rendered_revision = revisions

    def _display_snapshot(
        self,
        snapshot: np.ndarray,
        tab_name: str | None = None,
    ) -> None:
        tab = tab_name or self._selected_plot_tab()
        if tab not in SNAPSHOT_PLOT_TABS:
            tab = "Snapshot"
        max_points = int(self.config.output.snapshot_plot_max_points)
        self._set_figure(
            tab,
            make_snapshot_figure(snapshot, tab, max_points=max_points),
        )

    def _on_plot_tab_changed(self, _event: tk.Event) -> None:
        if self._selected_plot_tab() in SNAPSHOT_PLOT_TABS:
            self._schedule_snapshot_render()

    def _load_snapshot_index(self, *, silent: bool = False) -> None:
        try:
            self._sync_config_from_vars()
            index_path = Path(self.config.output_dir) / "snapshots_index.csv"
            if not index_path.exists():
                if not silent:
                    messagebox.showwarning(
                        "No Snapshot Index",
                        f"No snapshot index found:\n{index_path}",
                        parent=self.root,
                    )
                return
            with index_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(
                    deque(
                        csv.DictReader(handle),
                        maxlen=MAX_LIVE_SNAPSHOT_RECORDS,
                    )
                )
            self._replace_snapshot_records(rows)
            if not silent:
                self._append_console(
                    "Logs",
                    f"Loaded {len(rows)} snapshot records from {index_path}",
                )
        except Exception as exc:
            self._append_console(
                "Warnings",
                f"Failed to load snapshot index: {exc}",
            )
            if not silent:
                messagebox.showerror(
                    "Snapshot Load Failed",
                    str(exc),
                    parent=self.root,
                )

    def _replace_snapshot_records(self, rows: list[dict[str, str]]) -> None:
        self.snapshot_records = []
        self.snapshot_listbox.delete(0, tk.END)
        for row in rows:
            label = (
                f"macro {row.get('macro_step', '?')}  "
                f"t={float(row.get('time_s', 0.0)) * 1.0e3:.6g} ms  "
                f"n={row.get('particle_count', '?')}"
            )
            self.snapshot_records.append(
                {
                    "label": label,
                    "storage_ref": row.get("storage_ref", ""),
                }
            )
            self.snapshot_listbox.insert(tk.END, label)

    def _on_snapshot_selected(self, _event: tk.Event) -> None:
        selection = self.snapshot_listbox.curselection()
        if not selection:
            return
        self._load_snapshot_at_index(int(selection[0]))

    def _step_snapshot(self, delta: int) -> None:
        if not self.snapshot_records:
            self._load_snapshot_index(silent=True)
        if not self.snapshot_records:
            return
        selection = self.snapshot_listbox.curselection()
        index = int(selection[0]) if selection else 0
        index = max(
            0,
            min(len(self.snapshot_records) - 1, index + delta),
        )
        self.snapshot_listbox.selection_clear(0, tk.END)
        self.snapshot_listbox.selection_set(index)
        self.snapshot_listbox.see(index)
        self._load_snapshot_at_index(index)

    def _load_snapshot_at_index(self, index: int) -> None:
        try:
            record = self.snapshot_records[index]
            storage_ref = record["storage_ref"]
            output_dir = Path(self.config.output_dir)
            snapshot = self._read_snapshot(output_dir, storage_ref)
        except Exception as exc:
            self._append_console(
                "Warnings",
                f"Failed to load snapshot: {exc}",
            )
            messagebox.showerror(
                "Snapshot Load Failed",
                str(exc),
                parent=self.root,
            )
            return
        self.latest_snapshot = snapshot
        self.snapshot_revision = getattr(self, "snapshot_revision", 0) + 1
        self._schedule_snapshot_render()

    @staticmethod
    def _read_snapshot(output_dir: Path, storage_ref: str) -> np.ndarray:
        if storage_ref.endswith(".npy"):
            path = output_dir / storage_ref
            if not path.exists():
                path = Path(storage_ref)
            return np.asarray(np.load(path), dtype=np.float64)
        if storage_ref.startswith("snapshots/"):
            import h5py  # type: ignore

            with h5py.File(output_dir / "snapshots.h5", "r") as handle:
                return np.asarray(
                    handle[storage_ref][...],
                    dtype=np.float64,
                )
        raise ValueError(
            f"Unsupported snapshot storage reference: {storage_ref}"
        )

    def _plot_last_summary(self) -> None:
        if self.last_result is None:
            messagebox.showwarning(
                "No Run",
                "No completed run is available.",
                parent=self.root,
            )
            return
        self._set_figure(
            "Snapshot",
            make_summary_figure(self.last_result.macro_history),
        )
