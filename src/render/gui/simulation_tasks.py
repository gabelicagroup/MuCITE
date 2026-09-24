"""Simulation, snapshot, and report background tasks for the GUI."""

from __future__ import annotations

import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ...data.logger import DataLogger
from ...data.report import (
    ReportSnapshot,
    _write_simulation_report,
    capture_report_snapshot,
)
from ..cli.runner import build_demo_simulation
from .cli_preview import gui_run_arguments
from .config_adapter import build_ion_template, build_simulation_config
from .models import AppConfig
from .terminal_view import (
    MAX_TERMINAL_UI_BINS,
    aggregate_terminal_resolutions,
    iter_terminal_event_csv,
)
from .worker_lifecycle import TaskStopped, _post


def _build_data_logger(
    app_config: AppConfig,
    output_dir: Path,
) -> Optional[DataLogger]:
    if (
        not app_config.output.save_snapshots
        or int(app_config.runtime.snapshot_every) <= 0
    ):
        return None
    file_format = "h5" if app_config.output.save_h5 else "npy"
    return DataLogger(
        output_dir,
        export_every_macro_steps=max(
            1,
            int(app_config.runtime.snapshot_every),
        ),
        file_format=file_format,
        representative_trajectory_count=max(
            0,
            int(app_config.output.trajectory_sample_count),
        ),
        trajectory_record_every_snapshots=max(
            1,
            int(app_config.output.trajectory_record_every),
        ),
        make_plots=bool(app_config.output.save_figures),
        plot_max_points=max(
            1,
            int(app_config.output.snapshot_plot_max_points),
        ),
    )


def _emit_latest_snapshot(
    message_queue: "queue.Queue[dict[str, Any]]",
    data_logger: Optional[DataLogger],
    emitted_refs: set[str],
) -> None:
    if data_logger is None or not data_logger.records:
        return
    record = data_logger.records[-1]
    if record.storage_ref in emitted_refs:
        return
    emitted_refs.clear()
    emitted_refs.add(record.storage_ref)
    if data_logger.file_format == "npy":
        path = data_logger.output_dir / record.storage_ref
        if path.exists():
            snapshot = np.asarray(np.load(path), dtype=np.float64)
            if snapshot.shape[0] > 50000:
                offsets = np.linspace(
                    0,
                    snapshot.shape[0] - 1,
                    50000,
                    dtype=np.int64,
                )
                snapshot = snapshot[offsets]
            _post(
                message_queue,
                "snapshot_ready",
                snapshot=snapshot,
                storage_ref=str(path),
                macro_step=record.macro_step,
                time_s=record.time_s,
            )


def _post_simulation_start(
    message_queue: "queue.Queue[dict[str, Any]]",
    simulation: Any,
    replayable_gui_argv: list[str],
) -> None:
    if simulation.backend_message:
        _post(
            message_queue,
            "log",
            channel="logs",
            text=simulation.backend_message,
        )
    _post(
        message_queue,
        "log",
        channel="logs",
        text="CLI preview: python -m src " + " ".join(replayable_gui_argv),
    )


def _progress_callback(
    message_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    data_logger: Optional[DataLogger],
    emitted_refs: set[str],
    simulation: Any,
) -> Callable[[Any], None]:
    def callback(progress: Any) -> None:
        if stop_event.is_set():
            simulation.request_cancel()
            return
        _post(
            message_queue,
            "simulation_progress",
            progress=asdict(progress),
        )
        _emit_latest_snapshot(message_queue, data_logger, emitted_refs)

    return callback


def _start_cancellation_relay(
    stop_event: threading.Event,
    simulation: Any,
) -> tuple[threading.Event, threading.Thread]:
    relay_shutdown = threading.Event()

    def relay() -> None:
        while not relay_shutdown.is_set():
            if stop_event.wait(timeout=0.01):
                simulation.request_cancel()
                return

    thread = threading.Thread(
        target=relay,
        name="mucite-cancel-relay",
        daemon=True,
    )
    thread.start()
    return relay_shutdown, thread


def _run_and_capture(
    message_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    data_logger: Optional[DataLogger],
    simulation: Any,
    replayable_gui_argv: list[str],
    config: Any,
    template: Any,
) -> tuple[Any, ReportSnapshot]:
    emitted_refs: set[str] = set()
    callback = _progress_callback(
        message_queue,
        stop_event,
        data_logger,
        emitted_refs,
        simulation,
    )
    result = simulation.run(progress_callback=callback)
    _emit_latest_snapshot(message_queue, data_logger, emitted_refs)
    snapshot = capture_report_snapshot(
        argv=replayable_gui_argv,
        requested_config=config,
        template=template,
        simulation=simulation,
        result=result,
    )
    return result, snapshot


def _requested_report_paths(
    app_config: AppConfig,
    output_dir: Path,
    report_snapshot: ReportSnapshot,
) -> dict[str, Path]:
    if not app_config.output.generate_report_after_run:
        return {}
    return _write_simulation_report(
        report_dir=output_dir,
        snapshot=report_snapshot,
    )


def _configure_terminal_stream(
    simulation: Any,
    output_dir: Path,
    app_config: AppConfig,
) -> None:
    """Keep exact terminal timing on disk while bounding GUI report memory."""

    recorder = getattr(simulation, "terminal_event_recorder", None)
    open_stream = getattr(recorder, "open_csv_stream", None)
    if not callable(open_stream):
        return
    row_limit = max(0, int(app_config.runtime.max_terminal_event_rows))
    sample_size = 50_000 if row_limit == 0 else min(50_000, row_limit)
    open_stream(
        output_dir / "terminal_events.csv",
        stream_only=True,
        sample_size=sample_size,
    )


def _build_terminal_bins(
    report_snapshot: Any,
    output_dir: Path,
) -> dict[str, list[dict[str, float]]]:
    template = getattr(report_snapshot, "template", None)
    result = getattr(report_snapshot, "result", None)
    if template is None or result is None:
        return {}
    candidates = (
        getattr(report_snapshot, "terminal_stream_path", None),
        output_dir / "terminal_events.csv",
    )
    path = next(
        (Path(value) for value in candidates if value is not None and Path(value).is_file()),
        None,
    )
    rows = (
        iter_terminal_event_csv(path)
        if path is not None
        else tuple(getattr(report_snapshot, "terminal_event_rows", ()))
    )
    grouped = aggregate_terminal_resolutions(
        rows,
        bin_sizes_ms=(0.2, 1.0),
        charge_state=int(template.charge_state),
        macro_history=tuple(result.macro_history),
        max_bins=MAX_TERMINAL_UI_BINS,
    )
    return {f"{bin_ms:g}": bins for bin_ms, bins in grouped.items()}


def _post_simulation_finished(
    message_queue: "queue.Queue[dict[str, Any]]",
    app_config: AppConfig,
    output_dir: Path,
    config: Any,
    template: Any,
    result: Any,
    report_snapshot: ReportSnapshot,
) -> None:
    report_paths = _requested_report_paths(
        app_config,
        output_dir,
        report_snapshot,
    )
    terminal_bins_by_ms = _build_terminal_bins(report_snapshot, output_dir)
    _post(
        message_queue,
        "simulation_finished",
        config=config,
        template=template,
        result=result,
        report_snapshot=report_snapshot,
        terminal_bins_by_ms=terminal_bins_by_ms,
        report_paths={
            key: str(value) for key, value in report_paths.items()
        },
    )


def _execute_simulation_task(
    message_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    app_config: AppConfig,
) -> None:
    template = build_ion_template(app_config.beam)
    config = build_simulation_config(app_config)
    output_dir = Path(app_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_logger = _build_data_logger(app_config, output_dir)
    simulation = build_demo_simulation(
        config=config,
        template=template,
        particle_backend=str(app_config.runtime.backend),
        data_logger=data_logger,
    )
    _configure_terminal_stream(simulation, output_dir, app_config)
    replayable_gui_argv = gui_run_arguments(app_config)[2:]
    _post_simulation_start(
        message_queue,
        simulation,
        replayable_gui_argv,
    )
    relay_shutdown, relay_thread = _start_cancellation_relay(
        stop_event,
        simulation,
    )
    try:
        result, report_snapshot = _run_and_capture(
            message_queue,
            stop_event,
            data_logger,
            simulation,
            replayable_gui_argv,
            config,
            template,
        )
    finally:
        relay_shutdown.set()
        relay_thread.join()
    _post_simulation_finished(
        message_queue,
        app_config,
        output_dir,
        config=config,
        template=template,
        result=result,
        report_snapshot=report_snapshot,
    )


def run_simulation_task(
    message_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    app_config: AppConfig,
) -> None:
    try:
        _execute_simulation_task(message_queue, stop_event, app_config)
    finally:
        from ...core import release_pic_taichi_runtime

        release_pic_taichi_runtime()


def run_report_task(
    message_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    app_config: AppConfig,
    report_snapshot: ReportSnapshot,
) -> None:
    if stop_event.is_set():
        raise TaskStopped()
    paths = _write_simulation_report(
        report_dir=Path(app_config.output_dir),
        snapshot=report_snapshot,
    )
    _post(
        message_queue,
        "report_finished",
        report_paths={key: str(value) for key, value in paths.items()},
    )
