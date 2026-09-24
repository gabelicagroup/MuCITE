"""Stable JSON summary construction from immutable report snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import PARTICLE_STATUS_NAMES, SimulationConfig
from . import schema as _schema
from .snapshot import ReportSnapshot


def _report_config_payload(
    requested_config: SimulationConfig,
    effective_config: SimulationConfig,
) -> dict[str, Any]:
    """Return backward-compatible requested/effective config summaries."""

    requested = _schema._json_safe(
        _schema._config_summary(requested_config)
    )
    effective = _schema._json_safe(
        _schema._config_summary(effective_config)
    )
    differences = {
        key: {
            "requested": requested.get(key),
            "effective": effective.get(key),
        }
        for key in sorted(set(requested) | set(effective))
        if requested.get(key) != effective.get(key)
    }
    return {
        "config": effective,
        "requested_config": requested,
        "effective_config": effective,
        "config_differences": differences,
    }


def _macro_history_payload(result: Any) -> tuple[dict[str, Any], int, dict[str, Any]]:
    summary = dict(getattr(result, "macro_history_summary", {}) or {})
    macro_step_count = int(
        summary.get("macro_step_count", len(result.macro_history))
    )
    final_macro = dict(
        summary.get(
            "last",
            result.macro_history[-1] if result.macro_history else {},
        )
        or {}
    )
    if not summary:
        summary = {
            "macro_step_count": macro_step_count,
            "sample_row_count": int(len(result.macro_history)),
            "max_sample_rows": int(len(result.macro_history)),
            "sample_is_complete": True,
            "sampling": "complete",
            "first": result.macro_history[0] if result.macro_history else None,
            "last": final_macro or None,
            "numeric_fields": {},
        }
    return summary, macro_step_count, final_macro


def _final_particle_diagnostics(
    snapshot: ReportSnapshot,
    final_particle_path: Path,
) -> dict[str, Any]:
    config = snapshot.effective_config
    continuous_note = (
        "In continuous-current mode these counts are slot end-states at stop "
        "time, not cumulative transport."
        if config.source_mode == "continuous-current"
        else ""
    )
    note = (
        "electrode_hit is populated only when --electrode-mask or "
        "--electrode-mask-3d is enabled. "
        + continuous_note
    ).strip()
    return {
        "path": final_particle_path,
        "rows": int(len(snapshot.final_particle_rows)),
        "status_names": PARTICLE_STATUS_NAMES,
        "note": note,
    }


def _collision_statistics(result: Any) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in result.collision_statistics.items()
    }


def _result_payload(
    snapshot: ReportSnapshot,
    final_particle_path: Path,
    terminal_event_path: Path,
    diagnostic_plot_paths: dict[str, Path | None],
    macro_payload: tuple[dict[str, Any], int, dict[str, Any]],
) -> dict[str, Any]:
    result = snapshot.result
    macro_summary, macro_count, final_macro = macro_payload
    return {
        "particle_backend": result.particle_backend,
        "particle_backend_requested": result.particle_backend_requested,
        "particle_backend_effective": result.particle_backend_effective,
        "survivors": int(len(result.survivors)),
        "collision_count": int(result.collision_count),
        "collision_statistics": _collision_statistics(result),
        "termination_reason": result.termination_reason,
        "numerical_failure": _schema._json_safe(result.numerical_failure),
        "export_directory": result.export_directory,
        "macro_step_count": macro_count,
        "micro_step_count": int(result.micro_step_count),
        "final_macro": final_macro,
        "macro_history": result.macro_history,
        "macro_history_summary": macro_summary,
        "stage_history": result.stage_history,
        "survivor_statistics": _schema._survivor_summary(result.survivors),
        "final_status_counts": snapshot.final_status_counts,
        "final_particle_transport": (
            _schema._final_particle_transport_summary(
                snapshot.final_particle_rows
            )
        ),
        "final_particle_diagnostics": _final_particle_diagnostics(
            snapshot,
            final_particle_path,
        ),
        "terminal_event_diagnostics": {
            "path": terminal_event_path,
            **snapshot.terminal_event_runtime,
        },
        "diagnostic_plots": {
            name: path
            for name, path in diagnostic_plot_paths.items()
            if path is not None
        },
        "runtime_field_statistics": snapshot.runtime_field_statistics,
    }


def build_summary(
    snapshot: ReportSnapshot,
    final_particle_path: Path,
    terminal_event_path: Path,
    diagnostic_plot_paths: dict[str, Path | None],
) -> dict[str, Any]:
    """Build the established ``simulation_summary.json`` object."""

    config_payload = _report_config_payload(
        snapshot.requested_config,
        snapshot.effective_config,
    )
    return {
        "command": ["python", "-m", "src", *list(snapshot.argv)],
        "template": _schema._template_summary(snapshot.template),
        **config_payload,
        "field_metadata": snapshot.field_metadata,
        "rf_runtime": snapshot.rf_runtime,
        "collision_physics": snapshot.ionspa,
        "ionspa": snapshot.ionspa,
        "electrode_mask": snapshot.electrode_mask,
        "electrode_mask_3d": snapshot.electrode_mask_3d,
        "grid_runtime": snapshot.grid_runtime,
        "poisson_solver": snapshot.poisson_solver,
        "source_runtime": snapshot.source_runtime,
        "result": _result_payload(
            snapshot,
            final_particle_path,
            terminal_event_path,
            diagnostic_plot_paths,
            _macro_history_payload(snapshot.result),
        ),
        "performance": _schema._performance_summary(snapshot.result),
    }
