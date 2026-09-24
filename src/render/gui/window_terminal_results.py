"""Terminal-event source selection and Tk result presentation."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np

from .terminal_view import (
    MAX_TERMINAL_UI_BINS,
    aggregate_terminal_rows,
    iter_terminal_event_csv,
    make_terminal_current_figure,
)


class _WindowTerminalResultsMixin:
    def _terminal_source(
        self,
        snapshot: Any,
        report_paths: dict[str, Any],
    ) -> tuple[Path | None, tuple[Any, ...]]:
        candidates = []
        if report_paths.get("terminal_events"):
            candidates.append(Path(str(report_paths["terminal_events"])))
        for key in ("summary", "report"):
            if report_paths.get(key):
                candidates.append(
                    Path(str(report_paths[key])).parent / "terminal_events.csv"
                )
        stream_path = getattr(snapshot, "terminal_stream_path", None)
        if stream_path is not None:
            candidates.append(Path(stream_path))
        for path in candidates:
            if path.is_file():
                return path, ()
        return None, tuple(getattr(snapshot, "terminal_event_rows", ()))

    def _update_terminal_report(
        self,
        snapshot: Any,
        report_paths: dict[str, Any],
        terminal_bins_by_ms: dict[str, Any] | None = None,
    ) -> None:
        if snapshot is None:
            return
        self.terminal_event_path, self.last_terminal_rows = self._terminal_source(
            snapshot, dict(report_paths or {})
        )
        report_result = getattr(snapshot, "result", None)
        history = getattr(report_result, "macro_history", None)
        if history is None:
            result = getattr(self, "last_result", None)
            history = getattr(result, "macro_history", ())
        self.last_terminal_history = tuple(history)
        template = getattr(snapshot, "template", None)
        self.last_terminal_charge_state = int(
            getattr(template, "charge_state", self.config.beam.charge_e)
        )
        self.terminal_bins_by_ms = dict(terminal_bins_by_ms or {})
        self._warn_incomplete_terminal_data(snapshot)
        self._refresh_terminal_report()

    def _warn_incomplete_terminal_data(self, snapshot: Any) -> None:
        runtime = dict(getattr(snapshot, "terminal_event_runtime", {}))
        if runtime.get("max_rows_reached", False):
            self._append_console(
                "Warnings",
                "Terminal-event row limit was reached; later current bins are incomplete.",
            )
        if self.terminal_event_path is None and not runtime.get(
            "sample_is_complete", True
        ):
            self._append_console(
                "Warnings",
                "Terminal current uses a retained event sample; bins are approximate.",
            )

    def _refresh_terminal_report(self, _event: object = None) -> None:
        if not hasattr(self, "last_terminal_history"):
            return
        bin_ms = float(self.terminal_time_bin_var.get())
        self.config.output.terminal_time_bin_ms = bin_ms
        bins = self.terminal_bins_by_ms.get(f"{bin_ms:g}")
        if bins is None:
            bins = self._aggregate_terminal_fallback(bin_ms)
        self.last_terminal_bins = bins
        if bins:
            self.exit_current_var.set(f"{bins[-1]['z_exit_current_na']:.6g} nA")
            self.exit_ions_var.set(f"{bins[-1]['z_exit_real_ions']:.6g}")
        self._populate_terminal_tree(bins)
        self._set_figure(
            "Terminal Current",
            make_terminal_current_figure(
                bins, source_current_a=float(self.config.beam.current_a)
            ),
        )

    def _aggregate_terminal_fallback(
        self,
        bin_ms: float,
    ) -> list[dict[str, float]]:
        path = getattr(self, "terminal_event_path", None)
        rows = (
            iter_terminal_event_csv(path)
            if path is not None
            else self.last_terminal_rows
        )
        return aggregate_terminal_rows(
            rows,
            bin_ms=bin_ms,
            charge_state=self.last_terminal_charge_state,
            macro_history=self.last_terminal_history,
            max_bins=MAX_TERMINAL_UI_BINS,
        )

    def _populate_terminal_tree(self, bins: list[dict[str, float]]) -> None:
        tree = getattr(self, "terminal_tree", None)
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        for row in bins[-2000:]:
            active = row["active_macro"]
            values = (
                f"{row['start_ms']:.3f}-{row['end_ms']:.3f}",
                "-" if not np.isfinite(active) else f"{active:.0f}",
                f"{row['z_exit_current_na']:.6g}",
                f"{row['z_exit_real_ions']:.6g}",
                f"{row['z_exit_macro']:.0f}",
                f"{row['electrode_hit_macro']:.0f}",
                f"{row['loss_macro'] - row['electrode_hit_macro']:.0f}",
            )
            tree.insert("", tk.END, values=values)
