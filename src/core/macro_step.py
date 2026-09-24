"""Macro-step scheduling and bounded diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import PARTICLE_ACTIVE
from .events import MacroStepCompleted
from .ports import ProgressObserver, SimulationRuntime
from .run_state import RunState


@dataclass(frozen=True)
class MacroStepPlan:
    """One macro interval and its frozen space-charge solution."""

    end_s: float
    poisson_result: Any = None
    skipped: bool = False


def _wall_limit_reached(runtime: SimulationRuntime, run: RunState) -> bool:
    max_wall_time_s = float(runtime.config.max_wall_time_s)
    if max_wall_time_s <= 0.0:
        return False
    return time.monotonic() - run.run_wall_start >= max_wall_time_s


def prepare_macro_step(
    runtime: SimulationRuntime,
    run: RunState,
) -> MacroStepPlan | None:
    """Apply stage control and freeze one macro-step PIC field."""

    if runtime.is_cancel_requested():
        run.termination.finish("cancelled")
        return None
    runtime._apply_stage_for_time(run.clock.time_s)
    active_count = len(runtime._active_indices(run.particle_state))
    if active_count == 0 and not runtime._continuous_source_enabled():
        run.termination.finish("all_particles_terminal")
        return None
    if _wall_limit_reached(runtime, run):
        run.termination.finish("max_wall_time")
        return None
    run.clock.macro_step_index += 1
    macro_end_s = min(
        run.clock.time_s + float(runtime.config.macro_time_step_s),
        float(runtime.config.total_time_s),
        float(runtime._next_stage_boundary_s(run.clock.time_s)),
    )
    if macro_end_s <= run.clock.time_s:
        runtime._apply_stage_for_time(run.clock.time_s + 1.0e-15)
        return MacroStepPlan(end_s=macro_end_s, skipped=True)
    started_at = run.performance.start()
    poisson_result = runtime._update_space_charge_fields()
    run.performance.finish("time_pic_poisson", started_at)
    return MacroStepPlan(end_s=macro_end_s, poisson_result=poisson_result)


def _terminal_count(runtime: SimulationRuntime) -> int:
    if runtime.config.source_mode == "continuous-current":
        return int(sum(runtime._source_terminal_macro_particles_by_code.values()))
    return int(np.count_nonzero(runtime._particle_status_codes != PARTICLE_ACTIVE))


def _history_row(
    runtime: SimulationRuntime,
    progress: Any,
    poisson_result: Any,
    terminal_count: int,
    sce_external_ratio_p95: float,
) -> dict[str, Any]:
    disabled = poisson_result is None
    return {
        "time_s": progress.time_s,
        "stage_index": float(runtime.current_stage_index),
        "stage_name": runtime._current_stage_name(),
        "stage_source_enabled": float(bool(runtime._current_stage_source_enabled)),
        "ion_count": float(progress.ion_count),
        "mean_internal_temperature_k": progress.mean_internal_temperature_k,
        "mean_axial_position_m": progress.mean_axial_position_m,
        "collision_count": float(progress.collision_count),
        "sor_iters": float(0 if disabled else poisson_result.iterations),
        "poisson_backend": "disabled" if disabled else poisson_result.backend_name,
        "poisson_iterations": float(0 if disabled else poisson_result.iterations),
        "poisson_converged": float(
            1 if disabled or poisson_result.converged else 0
        ),
        "poisson_residual_initial": (
            float("nan") if disabled else float(poisson_result.residual_initial)
        ),
        "poisson_residual_final": (
            float("nan") if disabled else float(poisson_result.residual_final)
        ),
        "poisson_relative_residual": (
            float("nan") if disabled else float(poisson_result.relative_residual)
        ),
        "poisson_runtime_ms": (
            float("nan") if disabled else float(poisson_result.runtime_ms)
        ),
        "terminal_count": float(terminal_count),
        "sce_external_ratio_p95": sce_external_ratio_p95,
    }


def _record_macro_outputs(
    runtime: SimulationRuntime,
    run: RunState,
    progress: Any,
    observer: ProgressObserver | None,
) -> None:
    logger = runtime.data_logger
    if logger is not None and logger.should_log(run.clock.macro_step_index):
        logger.log_snapshot(
            run.clock.macro_step_index,
            run.clock.time_s,
            runtime._build_particle_snapshot(run.particle_state),
        )
    if observer is not None:
        observer(progress)
    runtime.event_bus.publish(
        MacroStepCompleted(
            time_s=float(run.clock.time_s),
            macro_step_index=int(run.clock.macro_step_index),
            micro_step_index=int(run.clock.micro_step_index),
            active_particle_count=int(progress.ion_count),
            collision_count=int(run.collision_count),
        )
    )


def _apply_stable_window_stop(
    runtime: SimulationRuntime,
    run: RunState,
    terminal_count: int,
) -> None:
    stable_window = int(runtime.config.stop_stable_window_steps)
    if runtime._continuous_source_enabled() or stable_window <= 0:
        return
    if terminal_count <= 0 or len(run.terminal_count_history) <= stable_window:
        return
    old_count = run.terminal_count_history[-stable_window - 1]
    terminal_delta = terminal_count - old_count
    tolerated_delta = int(
        np.ceil(
            float(runtime.config.ion_count)
            * max(float(runtime.config.stop_stable_fraction_tol), 0.0)
        )
    )
    if terminal_delta <= tolerated_delta:
        run.termination.finish("terminal_status_stable")


def finalize_macro_step(
    runtime: SimulationRuntime,
    run: RunState,
    poisson_result: Any,
    observer: ProgressObserver | None,
) -> None:
    """Synchronize one macro boundary, publish, and apply stable stopping."""

    started_at = run.performance.start()
    runtime._download_motion_state_into(run.particle_state)
    run.performance.finish("time_download_state", started_at)
    progress = runtime._build_progress_snapshot(
        run.particle_state,
        run.clock.time_s,
        run.clock.macro_step_index,
        run.clock.micro_step_index,
        run.collision_count,
    )
    ratio_p95 = runtime._space_charge_external_ratio_p95(
        run.particle_state,
        run.clock.time_s,
    )
    terminal_count = _terminal_count(runtime)
    run.terminal_count_history.append(terminal_count)
    run.macro_tracker.record(
        _history_row(
            runtime,
            progress,
            poisson_result,
            terminal_count,
            ratio_p95,
        )
    )
    _record_macro_outputs(runtime, run, progress, observer)
    if not run.termination.finished:
        _apply_stable_window_stop(runtime, run, terminal_count)
