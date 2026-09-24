"""Mutable browser workbench state, independent of HTTP routing."""

from __future__ import annotations

import queue
import math
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .config_io import (
    load_app_config,
    save_app_config,
)
from .models import AppConfig
from .web_config_binding import apply_web_payload
from .web_messages import handle_web_message, render_terminal_report
from .workers import WorkerHandle, start_worker


def _json_terminal_bins(rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {
            key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in row.items()
        }
        for row in rows[-1000:]
    ]


class WebGuiState:
    """State and task lifecycle for one browser GUI project."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.config = AppConfig()
        self.message_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.current_worker: Optional[WorkerHandle] = None
        self.logs: list[str] = ["Web GUI backend started."]
        self.warnings: list[str] = []
        self.validation: list[str] = []
        self.run_summary: list[str] = []
        self.status = "CONFIGURED"
        self.latest_image_path = ""
        self.progress_line = ""
        self.beam_validated = False
        self.last_template: Any = None
        self.last_sim_config: Any = None
        self.last_report_snapshot: Any = None
        self.last_result: Any = None
        self.last_report_paths: dict[str, Any] = {}
        self.terminal_bins: list[dict[str, float]] = []
        self.terminal_bins_by_ms: dict[str, list[dict[str, float]]] = {}
        self.last_progress_macro = -1
        self.last_progress_log_bin = -1

    def append(self, bucket: str, text: str) -> None:
        target = getattr(self, bucket)
        target.append(str(text))
        if len(target) > 300:
            del target[: len(target) - 300]

    def is_busy(self) -> bool:
        return self.current_worker is not None

    def apply_payload(self, payload: dict[str, Any]) -> None:
        old_bin_ms = float(self.config.output.terminal_time_bin_ms)
        apply_web_payload(self.config, payload)
        if (
            self.last_report_snapshot is not None
            and float(self.config.output.terminal_time_bin_ms) != old_bin_ms
        ):
            render_terminal_report(
                self, self.last_report_snapshot, self.last_report_paths
            )

    def start_task(self, kind: str, target: Any, *args: Any) -> None:
        if self.is_busy():
            active_kind = (
                self.current_worker.kind if self.current_worker else ""
            )
            self.append(
                "warnings", f"Task already running: {active_kind}"
            )
            return
        self.current_worker = start_worker(
            self.message_queue, kind, target, *args
        )

    def stop(self) -> None:
        worker = self.current_worker
        if worker is None:
            self.append("logs", "No active task to stop.")
            return
        worker.stop_event.set()
        self.append("warnings", f"Stop requested for {worker.kind}.")

    def shutdown(self) -> None:
        """Stop and join the active worker before the process exits."""

        with self.lock:
            worker = self.current_worker
            if worker is not None:
                worker.stop_event.set()
                self.append(
                    "warnings",
                    f"Shutdown requested; waiting for {worker.kind} cleanup.",
                )
        if worker is None:
            return
        worker.thread.join()
        with self.lock:
            self.drain_events()
            if self.current_worker is worker:
                self.current_worker = None

    def save_config(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.config.session_name}.json"
        save_app_config(path, self.config)
        self.append("logs", f"Saved project: {path}")
        return path

    def load_config(self, path: Path) -> None:
        self.config = load_app_config(path)
        self.status = "CONFIGURED"
        self.latest_image_path = ""
        self.progress_line = ""
        self.beam_validated = False
        self.last_template = None
        self.last_sim_config = None
        self.last_report_snapshot = None
        self.last_result = None
        self.last_report_paths = {}
        self.terminal_bins = []
        self.terminal_bins_by_ms = {}
        self.last_progress_macro = -1
        self.last_progress_log_bin = -1
        self.append("logs", f"Loaded project: {path}")

    def drain_events(self) -> None:
        latest_snapshot: dict[str, Any] | None = None
        while True:
            try:
                message = self.message_queue.get_nowait()
            except queue.Empty:
                break
            if str(message.get("type", "")) == "snapshot_ready":
                latest_snapshot = message
                continue
            if str(message.get("type", "")) == "simulation_finished":
                latest_snapshot = None
            try:
                self.handle_message(message)
            except Exception:
                self.append("warnings", traceback.format_exc())
        if latest_snapshot is not None:
            try:
                self.handle_message(latest_snapshot)
            except Exception:
                self.append("warnings", traceback.format_exc())

    def handle_message(self, message: dict[str, Any]) -> None:
        handle_web_message(self, message)

    def snapshot(self) -> dict[str, Any]:
        self.drain_events()
        beam = self.config.beam
        config_values = asdict(self.config)
        config_values.pop("state", None)
        image_url = (
            "/artifact?path=" + self.latest_image_path
            if self.latest_image_path else ""
        )
        return {
            "config": config_values,
            "status": self.status,
            "latest_image_url": image_url,
            "logs": self.logs,
            "warnings": self.warnings,
            "validation": self.validation,
            "run_summary": self.run_summary,
            "terminal_bins": _json_terminal_bins(self.terminal_bins),
            "progress_line": self.progress_line,
            "beam_summary": (
                f"mass={beam.mass_amu} amu, charge=+{beam.charge_e}, "
                f"particles={beam.particle_count}, "
                f"radius={beam.beam_radius_mm} mm"
            ),
        }


__all__ = ["WebGuiState"]
