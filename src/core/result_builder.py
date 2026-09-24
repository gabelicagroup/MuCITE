"""Build the stable result contract at the control/output boundary."""

from __future__ import annotations

import time
from datetime import datetime

from ..config import SimulationResult
from .events import RunFinished
from .ports import SimulationRuntime
from .run_state import RunState


def _collision_statistics(runtime: SimulationRuntime) -> dict[str, float]:
    return {
        "collision_macro_events": float(runtime._collision_macro_events),
        "collision_real_ions_represented": float(
            runtime._collision_real_ions_represented
        ),
        "fragmented_real_ions_represented": float(
            runtime._fragmented_real_ions_represented
        ),
        "hybrid_langevin_macro_events": float(
            runtime._hybrid_langevin_macro_events
        ),
        "hybrid_langevin_real_ions_represented": float(
            runtime._hybrid_langevin_real_ions_represented
        ),
        "hybrid_langevin_estimated_binary_collisions": float(
            runtime._hybrid_langevin_estimated_binary_collisions
        ),
    }


def build_simulation_result(
    runtime: SimulationRuntime,
    run: RunState,
) -> SimulationResult:
    """Finalize timings, publish ``RunFinished``, and return the public DTO."""

    wall_time_s = float(time.monotonic() - run.run_wall_start)
    ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
    result = SimulationResult(
        survivors=runtime._collect_survivors(run.particle_state),
        macro_history=run.macro_tracker.sample_rows(),
        macro_history_summary=run.macro_tracker.summary(),
        collision_count=run.collision_count,
        particle_backend=runtime.particle_backend,
        particle_backend_requested=runtime.particle_backend_requested,
        particle_backend_effective=runtime.particle_backend_effective,
        export_directory=run.export_directory,
        final_state=run.particle_state,
        final_time_s=float(run.clock.time_s),
        micro_step_count=int(run.clock.micro_step_index),
        run_wall_time_s=wall_time_s,
        run_started_at=run.run_started_at,
        run_ended_at=ended_at,
        termination_reason=run.termination.reason,
        performance_breakdown=run.performance.finalize(wall_time_s),
        dt_statistics=run.dt_tracker.summary(),
        dt_limiter_counts=run.dt_tracker.limiter_counts,
        dt_raw_limiter_counts=run.dt_tracker.raw_limiter_counts,
        stage_history=list(runtime.stage_history),
        numerical_failure=run.termination.numerical_failure,
        collision_statistics=_collision_statistics(runtime),
    )
    runtime.event_bus.publish(
        RunFinished(
            ended_at=ended_at,
            final_time_s=float(run.clock.time_s),
            termination_reason=run.termination.reason,
        )
    )
    return result
