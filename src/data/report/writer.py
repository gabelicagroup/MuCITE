"""Orchestrate report artifacts from an immutable runtime snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from ...config import SimulationProgress
from . import schema as _schema
from .csv_tables import (
    materialize_terminal_event_csv,
    write_final_particle_csv,
)
from .markdown import build_markdown
from .plots import (
    write_final_active_spatial_plot,
    write_z_exit_xy_plot,
)
from .snapshot import ReportSnapshot
from .summary_builder import _report_config_payload, build_summary


__all__ = [
    "_report_config_payload",
    "_stdout_progress",
    "_write_simulation_report",
]


def _stdout_progress(progress: SimulationProgress) -> None:
    print(
        "macro={:4d} micro={:7d} backend={} t={:.3e} s ions={} "
        "collisions={} mean_T={:.2f} K mean_z={:.3e} m".format(
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


def _write_diagnostic_plots(
    report_dir: Path,
    snapshot: ReportSnapshot,
    max_points: int,
) -> dict[str, Path | None]:
    return {
        "final_active_rz": write_final_active_spatial_plot(
            report_dir / "ion_spatial_distribution_rz.png",
            snapshot.final_particle_rows,
            snapshot.effective_config,
            max_points=max_points,
        ),
        "z_exit_xy": write_z_exit_xy_plot(
            report_dir / "z_exit_xy_distribution.png",
            snapshot.terminal_event_rows,
            snapshot.effective_config,
            max_points=max_points,
        ),
    }


def _write_simulation_report(
    *,
    report_dir: Path,
    snapshot: ReportSnapshot,
    report_plot_max_points: int = 200_000,
) -> dict[str, Path]:
    """Write all report artifacts from the immutable snapshot."""

    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": report_dir / "simulation_summary.json",
        "report": report_dir / "simulation_report.md",
        "final_particles": report_dir / "final_particles.csv",
        "terminal_events": report_dir / "terminal_events.csv",
    }
    write_final_particle_csv(
        paths["final_particles"],
        snapshot.final_particle_rows,
    )
    materialize_terminal_event_csv(
        paths["terminal_events"],
        snapshot.terminal_event_rows,
        snapshot.terminal_stream_path,
    )
    diagnostic_plots = _write_diagnostic_plots(
        report_dir,
        snapshot,
        int(report_plot_max_points),
    )
    summary = build_summary(
        snapshot,
        paths["final_particles"],
        paths["terminal_events"],
        diagnostic_plots,
    )
    paths["summary"].write_text(
        json.dumps(_schema._json_safe(summary), indent=2),
        encoding="utf-8",
    )
    paths["report"].write_text(
        build_markdown(summary, snapshot, paths),
        encoding="utf-8",
    )
    return {"summary": paths["summary"], "report": paths["report"]}
