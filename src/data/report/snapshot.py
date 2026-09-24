"""Immutable, lightweight report input captured at the runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import schema as _schema
from ...config import IonTemplate, SimulationConfig, SimulationResult


class FrozenDict(dict):
    """A JSON-compatible dictionary that rejects mutation after construction."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("report snapshot mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return FrozenDict({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ReportIonState:
    """Small immutable survivor record used only by report statistics."""

    position_m: tuple[float, float, float]
    velocity_m_per_s: tuple[float, float, float]
    mass_kg: float
    internal_temperature_k: float


@dataclass(frozen=True)
class ReportResult:
    """Report-facing projection of ``SimulationResult`` without final slot arrays."""

    survivors: tuple[ReportIonState, ...]
    macro_history: tuple[FrozenDict, ...]
    macro_history_summary: FrozenDict
    collision_count: int
    particle_backend: str
    particle_backend_requested: str
    particle_backend_effective: str
    export_directory: str | None
    final_time_s: float
    micro_step_count: int
    run_wall_time_s: float
    run_started_at: str
    run_ended_at: str
    termination_reason: str
    performance_breakdown: FrozenDict
    dt_statistics: FrozenDict
    dt_limiter_counts: FrozenDict
    dt_raw_limiter_counts: FrozenDict
    collision_statistics: FrozenDict
    stage_history: tuple[FrozenDict, ...]
    numerical_failure: FrozenDict


@dataclass(frozen=True)
class ReportSnapshot:
    """Complete report payload with no reference to a live simulation or field grid."""

    argv: tuple[str, ...]
    requested_config: SimulationConfig
    effective_config: SimulationConfig
    template: IonTemplate
    result: ReportResult
    field_metadata: FrozenDict
    rf_runtime: FrozenDict
    ionspa: FrozenDict
    electrode_mask: FrozenDict
    electrode_mask_3d: FrozenDict
    grid_runtime: FrozenDict
    poisson_solver: FrozenDict
    source_runtime: FrozenDict
    runtime_field_statistics: FrozenDict
    final_particle_rows: tuple[FrozenDict, ...]
    final_status_counts: FrozenDict
    terminal_event_rows: tuple[FrozenDict, ...]
    terminal_event_runtime: FrozenDict
    terminal_stream_path: Path | None


def _vector3(values: Iterable[float]) -> tuple[float, float, float]:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"Expected a three-component vector, got shape {vector.shape}.")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _capture_result(
    result: SimulationResult,
    *,
    particle_backend_requested: str,
    particle_backend_effective: str,
) -> ReportResult:
    survivors = tuple(
        ReportIonState(
            position_m=_vector3(ion.position_m),
            velocity_m_per_s=_vector3(ion.velocity_m_per_s),
            mass_kg=float(ion.mass_kg),
            internal_temperature_k=float(ion.internal_temperature_k),
        )
        for ion in result.survivors
    )
    return ReportResult(
        survivors=survivors,
        macro_history=tuple(_freeze(row) for row in result.macro_history),
        macro_history_summary=_freeze(dict(result.macro_history_summary or {})),
        collision_count=int(result.collision_count),
        particle_backend=str(result.particle_backend),
        particle_backend_requested=str(result.particle_backend_requested or particle_backend_requested),
        particle_backend_effective=str(result.particle_backend_effective or particle_backend_effective),
        export_directory=None if result.export_directory is None else str(result.export_directory),
        final_time_s=float(result.final_time_s),
        micro_step_count=int(result.micro_step_count),
        run_wall_time_s=float(result.run_wall_time_s),
        run_started_at=str(result.run_started_at),
        run_ended_at=str(result.run_ended_at),
        termination_reason=str(result.termination_reason),
        performance_breakdown=_freeze(dict(result.performance_breakdown)),
        dt_statistics=_freeze(dict(result.dt_statistics)),
        dt_limiter_counts=_freeze(dict(result.dt_limiter_counts)),
        dt_raw_limiter_counts=_freeze(dict(result.dt_raw_limiter_counts)),
        collision_statistics=_freeze(dict(result.collision_statistics)),
        stage_history=tuple(_freeze(row) for row in result.stage_history),
        numerical_failure=_freeze(dict(result.numerical_failure)),
    )


def _ionspa_runtime_info(adapter: Any) -> dict[str, Any]:
    runtime_info = getattr(adapter, "runtime_info", None)
    if callable(runtime_info):
        runtime_info = runtime_info()
    if isinstance(runtime_info, Mapping):
        return dict(runtime_info)
    return {
        "loaded": getattr(adapter, "ionspa", None) is not None,
        "import_error": (
            None
            if getattr(adapter, "import_error", None) is None
            else repr(getattr(adapter, "import_error"))
        ),
    }


def _capture_terminal_runtime(
    terminal_recorder: Any,
    effective_config: SimulationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terminal_recorder.flush()
    rows = terminal_recorder.report_rows()
    distribution = _schema._terminal_event_summary(
        rows,
        total_row_count=terminal_recorder.recorded_row_count,
        exact_by_status=terminal_recorder.recorded_summary_by_status(),
        distribution_sample_is_complete=terminal_recorder.sample_is_complete,
    )
    runtime = {
        "rows": int(terminal_recorder.recorded_row_count),
        "sample_rows": int(len(rows)),
        "streamed_rows": int(terminal_recorder.streamed_row_count),
        "stream_only": bool(terminal_recorder.stream_only),
        "sample_size": int(terminal_recorder.sample_size),
        "sample_is_complete": bool(terminal_recorder.sample_is_complete),
        "distribution_statistics": distribution["distribution_statistics"],
        "summary_by_status": distribution["by_status"],
        "tof_reference": distribution["tof_reference"],
        "recording_mode": effective_config.terminal_event_mode,
        "max_rows": int(effective_config.max_terminal_event_rows),
        "max_rows_reached": bool(terminal_recorder.max_rows_reached),
        "omitted_rows_by_status": dict(terminal_recorder.omitted_rows_by_status),
        "omitted_real_ions_by_status": dict(
            terminal_recorder.omitted_real_ions_by_status
        ),
        "omitted_parent_real_ions_by_status": dict(
            terminal_recorder.omitted_parent_real_ions_by_status
        ),
        "omitted_fragment_real_ions_by_status": dict(
            terminal_recorder.omitted_fragment_real_ions_by_status
        ),
        "recorder": terminal_recorder.diagnostics(),
    }
    return rows, runtime


def _capture_grid_runtime(simulation: Any) -> dict[str, Any]:
    return {
        "static_grid_nr": int(simulation.static_grid.nr),
        "static_grid_nz": int(simulation.static_grid.nz),
        "pic_grid_nr": int(simulation.pic_grid.nr),
        "pic_grid_nz": int(simulation.pic_grid.nz),
        "pic_grid_decoupled": bool(simulation.pic_grid_is_decoupled),
        "pic_grid_resolution_source": str(
            simulation.pic_grid_resolution_source
        ),
        "pic_space_charge_scale": float(simulation.pic_space_charge_scale),
        "r_max_m": float(simulation.static_grid.r_max_m),
        "z_max_m": float(simulation.static_grid.z_max_m),
        "pic_dr_m": float(simulation.pic_grid.dr),
        "pic_dz_m": float(simulation.pic_grid.dz),
        "storage_estimate": simulation.grid_storage_estimate,
    }


def _capture_template(template: IonTemplate) -> IonTemplate:
    return replace(
        template,
        initial_position_m=_freeze(template.initial_position_m),
        initial_velocity_m_per_s=_freeze(template.initial_velocity_m_per_s),
    )


def _capture_rf_runtime(simulation: Any) -> FrozenDict:
    return _freeze(
        {
            "enabled": bool(simulation.rf_enabled),
            "reference_peak_voltage_v": float(
                simulation.rf_reference_peak_voltage_v
            ),
            "peak_to_reference_scale": float(
                simulation.rf_peak_to_reference_scale
            ),
            "angular_frequency_rad_s": float(
                simulation.rf_angular_frequency_rad_s
            ),
        }
    )


def capture_report_snapshot(
    *,
    argv: list[str],
    requested_config: SimulationConfig,
    template: IonTemplate,
    simulation: Any,
    result: SimulationResult,
) -> ReportSnapshot:
    """Capture all report inputs while the completed runtime is still available."""

    effective_config = simulation.config
    final_particle_rows, final_status_counts = _schema._final_particle_table(
        simulation,
        result,
    )
    terminal_recorder = simulation.terminal_event_recorder
    terminal_event_rows, terminal_event_runtime = _capture_terminal_runtime(
        terminal_recorder,
        effective_config,
    )
    return ReportSnapshot(
        argv=tuple(str(value) for value in argv),
        requested_config=requested_config,
        effective_config=effective_config,
        template=_capture_template(template),
        result=_capture_result(
            result,
            particle_backend_requested=str(simulation.particle_backend_requested),
            particle_backend_effective=str(simulation.particle_backend_effective),
        ),
        field_metadata=_freeze(simulation.baked_field_metadata),
        rf_runtime=_capture_rf_runtime(simulation),
        ionspa=_freeze(_ionspa_runtime_info(simulation.adapter)),
        electrode_mask=_freeze(simulation.electrode_mask_metadata),
        electrode_mask_3d=_freeze(simulation.electrode_mask_3d_metadata),
        grid_runtime=_freeze(_capture_grid_runtime(simulation)),
        poisson_solver=_freeze(simulation.pic_solver.poisson_solver.get_diagnostics()),
        source_runtime=_freeze(_schema._source_runtime_summary(simulation, result)),
        runtime_field_statistics=_freeze(_schema._runtime_field_summary(simulation)),
        final_particle_rows=tuple(_freeze(row) for row in final_particle_rows),
        final_status_counts=_freeze(final_status_counts),
        terminal_event_rows=tuple(_freeze(row) for row in terminal_event_rows),
        terminal_event_runtime=_freeze(terminal_event_runtime),
        terminal_stream_path=(
            None if terminal_recorder.stream_path is None else Path(terminal_recorder.stream_path)
        ),
    )
