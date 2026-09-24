"""Tkinter presentation adapter for live simulation output."""

from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any

from ..config import SimulationProgress, SimulationResult


class RuntimeOutputWindow:
    """Simple live status window for long-running simulations."""

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation
        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.finished = False
        self.close_requested = False
        self.result: SimulationResult | None = None
        self.error: Exception | None = None
        self.error_traceback = ""
        self.worker: threading.Thread | None = None

        self.root = tk.Tk()
        self.root.title("Simu_IonSource Runtime Monitor")
        self.root.geometry("860x620")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.backend_var = tk.StringVar(value=simulation.particle_backend)
        self.time_var = tk.StringVar(value="0.000e+00 s")
        self.macro_var = tk.StringVar(value="0")
        self.micro_var = tk.StringVar(value="0")
        self.ion_count_var = tk.StringVar(value="0")
        self.collision_var = tk.StringVar(value="0")
        self.temperature_var = tk.StringVar(value="0.00 K")
        self.position_var = tk.StringVar(value="0.000e+00 m")
        self.status_var = tk.StringVar(value="Preparing simulation...")

        self._build_layout()

    def _build_layout(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="Simu_IonSource Live Output", font=("Segoe UI", 16, "bold"))
        title.pack(anchor=tk.W)

        summary = ttk.Frame(frame, padding=(0, 10, 0, 10))
        summary.pack(fill=tk.X)

        self._add_summary_row(summary, "Backend", self.backend_var, 0, 0)
        self._add_summary_row(summary, "Macro Step", self.macro_var, 0, 1)
        self._add_summary_row(summary, "Micro Step", self.micro_var, 1, 0)
        self._add_summary_row(summary, "Sim Time", self.time_var, 1, 1)
        self._add_summary_row(summary, "Ion Count", self.ion_count_var, 2, 0)
        self._add_summary_row(summary, "Collisions", self.collision_var, 2, 1)
        self._add_summary_row(summary, "Mean T", self.temperature_var, 3, 0)
        self._add_summary_row(summary, "Mean Z", self.position_var, 3, 1)

        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(status_frame, text="Status:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(8, 0))

        self.log = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=24, font=("Consolas", 10))
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.insert(tk.END, f"{self.simulation.backend_message}\n")
        self.log.configure(state=tk.DISABLED)

    def _add_summary_row(self, parent: ttk.Frame, label: str, value_var: tk.StringVar, row: int, column: int) -> None:
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=column, sticky="w", padx=(0, 24), pady=4)
        ttk.Label(holder, text=f"{label}:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(holder, textvariable=value_var).pack(side=tk.LEFT, padx=(8, 0))

    def _append_log(self, line: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _progress_callback(self, progress: SimulationProgress) -> None:
        self.message_queue.put(("progress", progress))

    def _worker(self) -> None:
        try:
            result = self.simulation.run(progress_callback=self._progress_callback)
            self.message_queue.put(("result", result))
        except Exception as exc:
            self.message_queue.put(("error", (exc, traceback.format_exc())))

    def _handle_progress(self, progress: SimulationProgress) -> None:
        self.backend_var.set(progress.particle_backend)
        self.time_var.set(f"{progress.time_s:.3e} s")
        self.macro_var.set(str(progress.macro_step_index))
        self.micro_var.set(str(progress.micro_step_index))
        self.ion_count_var.set(str(progress.ion_count))
        self.collision_var.set(str(progress.collision_count))
        self.temperature_var.set(f"{progress.mean_internal_temperature_k:.2f} K")
        self.position_var.set(f"{progress.mean_axial_position_m:.3e} m")
        self.status_var.set("Simulation running...")
        self._append_log(
            "macro={:4d} micro={:7d} backend={} t={:.3e} s ions={} collisions={} mean_T={:.2f} K mean_z={:.3e} m".format(
                progress.macro_step_index,
                progress.micro_step_index,
                progress.particle_backend,
                progress.time_s,
                progress.ion_count,
                progress.collision_count,
                progress.mean_internal_temperature_k,
                progress.mean_axial_position_m,
            )
        )

    def _handle_result(self, result: SimulationResult) -> None:
        self.result = result
        self.finished = True
        if result.termination_reason == "cancelled":
            self.status_var.set("Simulation cancelled.")
        else:
            self.status_var.set("Simulation completed.")
        self._append_log(
            "Finished. "
            f"reason={result.termination_reason or 'completed'} "
            f"backend={result.particle_backend} "
            f"survivors={len(result.survivors)} "
            f"collisions={result.collision_count}"
        )

    def _handle_error(self, payload: object) -> None:
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], Exception):
            self.error = payload[0]
            self.error_traceback = str(payload[1])
        else:
            self.error = RuntimeError(str(payload))
            self.error_traceback = str(payload)
        self.finished = True
        self.status_var.set("Simulation failed.")
        self._append_log("Error:")
        self._append_log(self.error_traceback)

    def _drain_queue(self) -> None:
        while True:
            try:
                message_type, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break

            if message_type == "progress":
                self._handle_progress(payload)  # type: ignore[arg-type]
            elif message_type == "result":
                self._handle_result(payload)  # type: ignore[arg-type]
            elif message_type == "error":
                self._handle_error(payload)

    def _poll_queue(self) -> None:
        self._drain_queue()
        if self.finished and self.close_requested:
            self.root.quit()
            return
        if not self.finished:
            self.root.after(100, self._poll_queue)

    def _on_close(self) -> None:
        if self.finished:
            self.root.quit()
            return
        if self.close_requested:
            return
        self.close_requested = True
        self.status_var.set("Cancellation requested; waiting for a safe checkpoint...")
        self._append_log("Cancellation requested. Waiting for simulation cleanup.")
        self.simulation.request_cancel()

    def run(self) -> SimulationResult:
        self.status_var.set("Starting worker thread...")
        self.worker = threading.Thread(
            target=self._worker,
            name="simu-runtime-worker",
            daemon=False,
        )
        self.worker.start()
        self.root.after(100, self._poll_queue)
        try:
            self.root.mainloop()
        finally:
            if self.worker.is_alive():
                self.simulation.request_cancel()
            self.worker.join()
            self._drain_queue()
            try:
                self.root.destroy()
            except tk.TclError:
                pass

        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError("Runtime window closed without a simulation result.")
        return self.result


def launch_runtime_window(simulation: Any) -> SimulationResult:
    """Launch the Tkinter runtime monitor for a simulation."""

    return RuntimeOutputWindow(simulation).run()
