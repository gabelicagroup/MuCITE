"""Macro/micro event loop for the coupled simulation runtime."""

from __future__ import annotations

import time
from contextlib import nullcontext
from datetime import datetime

from ..config import SimulationResult
from .events import RunStarted
from .macro_step import finalize_macro_step, prepare_macro_step
from .micro_step import advance_micro_steps
from .ports import ProgressObserver, SimulationRuntime
from .policy import EnginePolicy
from .result_builder import build_simulation_result
from .run_state import RunState


class SimulationEngine:
    """Control time advancement through a model/runtime port."""

    def __init__(
        self,
        runtime: SimulationRuntime,
        policy: EnginePolicy | None = None,
    ) -> None:
        self.runtime = runtime
        self.policy = policy or EnginePolicy()

    def run(
        self,
        progress_observer: ProgressObserver | None = None,
    ) -> SimulationResult:
        """Run until a configured terminal condition or cancellation."""

        guard = getattr(self.runtime, "run_guard", nullcontext())
        with guard:
            return self._run_guarded(progress_observer)

    def _run_guarded(
        self,
        progress_observer: ProgressObserver | None,
    ) -> SimulationResult:
        run = self._start_run()
        try:
            self._initialize(run, progress_observer)
            self._advance(run, progress_observer)
            self._log_final_snapshot(run)
            self._resolve_termination(run)
            return build_simulation_result(self.runtime, run)
        finally:
            self._close_sinks()

    def _start_run(self) -> RunState:
        wall_start = time.monotonic()
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.runtime.event_bus.publish(RunStarted(started_at=started_at))
        return RunState(
            particle_state={},
            stable_window_steps=int(
                self.runtime.config.stop_stable_window_steps
            ),
            random_seed=int(self.runtime.config.random_seed),
            run_wall_start=wall_start,
            run_started_at=started_at,
            progress_interval_s=float(self.policy.progress_interval_s),
            dt_quantile_reservoir_size=int(
                self.policy.dt_quantile_reservoir_size
            ),
            macro_history_max_rows=int(
                self.policy.macro_history_max_rows
            ),
        )

    def _initialize(
        self,
        run: RunState,
        observer: ProgressObserver | None,
    ) -> None:
        self.runtime.initialize_ions()
        started_at = run.performance.start()
        run.particle_state = self.runtime._download_cloud_state()
        run.performance.finish("time_download_state", started_at)
        started_at = run.performance.start()
        self.runtime._taichi_runtime_apply_terminal_boundary_events(
            run.particle_state,
            run.clock.time_s,
        )
        run.performance.finish("time_terminal_events", started_at)
        logger = self.runtime.data_logger
        if logger is not None:
            logger.log_snapshot(
                0,
                run.clock.time_s,
                self.runtime._build_particle_snapshot(run.particle_state),
            )
            run.export_directory = str(logger.output_dir)
        if observer is not None:
            observer(
                self.runtime._build_progress_snapshot(
                    run.particle_state,
                    run.clock.time_s,
                    run.clock.macro_step_index,
                    run.clock.micro_step_index,
                    run.collision_count,
                )
            )
        if self.runtime.is_cancel_requested():
            run.termination.finish("cancelled")

    def _should_continue(self, run: RunState) -> bool:
        if run.termination.finished:
            return False
        if run.clock.time_s >= float(self.runtime.config.total_time_s):
            return False
        active = len(self.runtime._active_indices(run.particle_state)) > 0
        return active or self.runtime._continuous_source_enabled()

    def _advance(
        self,
        run: RunState,
        observer: ProgressObserver | None,
    ) -> None:
        while self._should_continue(run):
            plan = prepare_macro_step(self.runtime, run)
            if plan is None:
                break
            if plan.skipped:
                continue
            advance_micro_steps(self.runtime, run, plan.end_s, observer)
            finalize_macro_step(
                self.runtime,
                run,
                plan.poisson_result,
                observer,
            )

    def _log_final_snapshot(self, run: RunState) -> None:
        logger = self.runtime.data_logger
        if logger is None:
            return
        if logger.last_logged_macro_step == run.clock.macro_step_index:
            return
        logger.log_snapshot(
            run.clock.macro_step_index,
            run.clock.time_s,
            self.runtime._build_particle_snapshot(run.particle_state),
        )

    def _resolve_termination(self, run: RunState) -> None:
        if run.termination.finished:
            return
        if self.runtime.is_cancel_requested():
            run.termination.finish("cancelled")
        elif run.clock.time_s >= float(self.runtime.config.total_time_s):
            run.termination.finish("max_total_time")
        else:
            run.termination.finish("all_particles_terminal")

    def _close_sinks(self) -> None:
        try:
            if self.runtime.data_logger is not None:
                self.runtime.data_logger.close()
        finally:
            self.runtime.terminal_event_recorder.close()
