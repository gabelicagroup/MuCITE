"""Worker-message reducers for the browser presentation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .plotting import (
    make_snapshot_figure,
    make_summary_figure,
)
from .terminal_view import (
    MAX_TERMINAL_UI_BINS,
    aggregate_terminal_rows,
    iter_terminal_event_csv,
    make_terminal_current_figure,
)

if TYPE_CHECKING:
    from .web_state import WebGuiState


MessageHandler = Callable[["WebGuiState", dict[str, Any]], None]


def _task_lifecycle(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    message_type = str(message.get("type", ""))
    kind = str(message.get("kind", ""))
    if message_type == "task_started":
        state.status = kind.upper()
        state.append("logs", f"Task started: {kind}")
    elif message_type == "task_done":
        worker = state.current_worker
        if worker is not None and worker.kind == kind:
            state.current_worker = None
    elif message_type == "task_stopped":
        state.status = "STOPPED"
        state.append("warnings", f"Task stopped: {kind}")
    elif message_type == "task_failed":
        state.status = "FAILED"
        state.append("warnings", str(message.get("error", "")))
    elif message_type == "log":
        state.append("logs", str(message.get("text", "")))


def _beam_smoke_finished(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    stats = dict(message.get("stats", {}))
    state.beam_validated = True
    state.status = "BEAM_VALIDATED"
    state.latest_image_path = str(stats.get("preview_path", ""))
    state.append(
        "validation",
        "Beam smoke passed: sample_count={sample_count}, "
        "mean KE={energy_ev_mean:.6g} eV, "
        "p95 angle={angle_deg_p95:.6g} deg".format(**stats),
    )


def _field_bake_finished(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    result = dict(message.get("result", {}))
    output_npy = str(
        result.get("output_npy", state.config.field_bake.output_npy)
    )
    state.config.field_bake.output_npy = output_npy
    state.config.loaded_baked_field_path = output_npy
    state.status = "FIELDS_BAKED"
    state.append("validation", f"Field bake completed: {output_npy}")


def _field_diagnostics_finished(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    path = str(message.get("path", ""))
    state.latest_image_path = path
    state.status = "READY_TO_RUN" if state.beam_validated else "FIELDS_BAKED"
    state.append("validation", f"Field diagnostic: {path}")


def _simulation_progress(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    progress = dict(message.get("progress", {}))
    macro_step = int(progress.get("macro_step_index", 0))
    time_ms = float(progress.get("time_s", 0.0)) * 1.0e3
    state.progress_line = (
        f"t={time_ms:.3f} ms | active weighted ion packs={int(progress.get('ion_count', 0))} "
        f"| cumulative collisions={int(progress.get('collision_count', 0))}"
    )
    bin_ms = max(float(state.config.output.terminal_time_bin_ms), 1.0e-9)
    progress_bin = int(time_ms / bin_ms)
    if progress_bin != state.last_progress_log_bin:
        state.append("logs", state.progress_line)
        state.last_progress_log_bin = progress_bin
    state.last_progress_macro = macro_step


def _snapshot_ready(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    snapshot = np.asarray(message.get("snapshot"), dtype=np.float64)
    output_dir = Path(state.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "latest_snapshot_gui.png"
    make_snapshot_figure(
        snapshot,
        "Snapshot",
        max_points=int(state.config.output.snapshot_plot_max_points),
    ).savefig(path, dpi=140)
    state.latest_image_path = str(path)


def _remember_simulation_result(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    state.last_template = message.get("template")
    state.last_sim_config = message.get("config")
    state.last_report_snapshot = message.get("report_snapshot")
    state.last_result = message.get("result")
    reason = str(getattr(state.last_result, "termination_reason", ""))
    state.status = "STOPPED" if reason == "cancelled" else "FINISHED"


def _render_simulation_summary(state: "WebGuiState") -> None:
    result = state.last_result
    if result is None:
        return
    output_dir = Path(state.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_summary_gui.png"
    make_summary_figure(result.macro_history).savefig(path, dpi=140)
    state.latest_image_path = str(path)
    state.append(
        "run_summary",
        "{}: backend={}, survivors={}, collisions={}, "
        "termination={}".format(
            "Stopped" if result.termination_reason == "cancelled" else "Finished",
            result.particle_backend,
            len(result.survivors),
            result.collision_count,
            result.termination_reason,
        ),
    )


def _simulation_finished(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    _remember_simulation_result(state, message)
    _render_simulation_summary(state)
    report_paths = dict(message.get("report_paths", {}))
    state.last_report_paths = report_paths
    state.terminal_bins_by_ms = dict(message.get("terminal_bins_by_ms", {}))
    render_terminal_report(state, state.last_report_snapshot, report_paths)
    if report_paths:
        state.append("run_summary", f"Report paths: {report_paths}")


def _terminal_source(snapshot: Any, report_paths: dict[str, Any]) -> tuple[Any, bool]:
    candidates: list[Path] = []
    runtime = dict(getattr(snapshot, "terminal_event_runtime", {}))
    if report_paths.get("terminal_events"):
        candidates.append(Path(str(report_paths["terminal_events"])))
    for key in ("summary", "report"):
        if report_paths.get(key):
            candidates.append(Path(str(report_paths[key])).parent / "terminal_events.csv")
    stream_path = getattr(snapshot, "terminal_stream_path", None)
    if stream_path is not None:
        candidates.append(Path(stream_path))
    for path in candidates:
        if path.is_file():
            return iter_terminal_event_csv(path), not bool(
                runtime.get("max_rows_reached", False)
            )
    return tuple(getattr(snapshot, "terminal_event_rows", ())), bool(
        runtime.get("sample_is_complete", True)
    )


def render_terminal_report(
    state: "WebGuiState",
    snapshot: Any,
    report_paths: dict[str, Any],
) -> None:
    """Rebuild the browser terminal view at its selected time resolution."""

    if snapshot is None:
        return
    bin_ms = float(state.config.output.terminal_time_bin_ms)
    bins = state.terminal_bins_by_ms.get(f"{bin_ms:g}")
    runtime = dict(getattr(snapshot, "terminal_event_runtime", {}))
    complete = not bool(runtime.get("max_rows_reached", False))
    if bins is None:
        rows, complete = _terminal_source(snapshot, report_paths)
        bins = aggregate_terminal_rows(
            rows,
            bin_ms=bin_ms,
            charge_state=int(snapshot.template.charge_state),
            macro_history=tuple(snapshot.result.macro_history),
            max_bins=MAX_TERMINAL_UI_BINS,
        )
    state.terminal_bins = bins[-2000:]
    output_dir = Path(state.config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "terminal_current_gui.png"
    make_terminal_current_figure(
        bins, source_current_a=float(state.config.beam.current_a)
    ).savefig(path, dpi=140)
    state.latest_image_path = str(path)
    exit_ions = sum(row["z_exit_real_ions"] for row in bins)
    state.append(
        "run_summary",
        f"Terminal: {len(bins)} bins x {bin_ms:g} ms; "
        f"exit real ions={exit_ions:.6g}",
    )
    if not complete:
        state.append(
            "warnings",
            "Terminal bins are incomplete because sampling or the row limit was active.",
        )


def _report_finished(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    state.append(
        "run_summary",
        f"Report paths: {message.get('report_paths', {})}",
    )


_HANDLERS: dict[str, MessageHandler] = {
    "task_started": _task_lifecycle,
    "task_done": _task_lifecycle,
    "task_stopped": _task_lifecycle,
    "task_failed": _task_lifecycle,
    "log": _task_lifecycle,
    "beam_smoke_finished": _beam_smoke_finished,
    "field_bake_finished": _field_bake_finished,
    "field_diagnostics_finished": _field_diagnostics_finished,
    "simulation_progress": _simulation_progress,
    "snapshot_ready": _snapshot_ready,
    "simulation_finished": _simulation_finished,
    "report_finished": _report_finished,
}


def handle_web_message(
    state: "WebGuiState",
    message: dict[str, Any],
) -> None:
    handler = _HANDLERS.get(str(message.get("type", "")))
    if handler is not None:
        handler(state, message)


__all__ = ["handle_web_message", "render_terminal_report"]
