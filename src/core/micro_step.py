"""Micro-step scheduling for the coupled particle runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann

from ..config import DT_LIMITER_NAMES
from .timestep import (
    InvalidTimeStepError,
    select_active_timestep,
)
from .ports import ProgressObserver, SimulationRuntime
from .run_state import RunState


@dataclass(frozen=True)
class PreparedTimeStep:
    """Selected global step and cached per-particle collision rates."""

    dt_s: float
    cached_collision_rates_hz: np.ndarray


def _runtime_stop_requested(runtime: SimulationRuntime, run: RunState) -> bool:
    if runtime.is_cancel_requested():
        run.termination.finish("cancelled")
        return True
    max_wall_time_s = float(runtime.config.max_wall_time_s)
    if max_wall_time_s > 0.0:
        elapsed_s = time.monotonic() - run.run_wall_start
        if elapsed_s >= max_wall_time_s:
            run.termination.finish("max_wall_time")
            return True
    return False


def _advance_idle_source(
    runtime: SimulationRuntime,
    run: RunState,
    macro_end_s: float,
) -> bool:
    remaining_s = max(float(macro_end_s - run.clock.time_s), 0.0)
    idle_dt_s = min(
        remaining_s,
        max(float(runtime.integrator.max_dt_s), runtime.integrator.min_dt_s),
    )
    if idle_dt_s <= 0.0:
        return False
    new_time_s = run.clock.time_s + idle_dt_s
    started_at = run.performance.start()
    runtime._advance_capillary_source(
        run.particle_state,
        idle_dt_s,
        new_time_s,
    )
    run.performance.finish("time_source_injection", started_at)
    started_at = run.performance.start()
    runtime._inject_continuous_source(
        run.particle_state,
        new_time_s,
        idle_dt_s,
    )
    run.performance.finish("time_source_injection", started_at)
    run.clock.time_s = new_time_s
    run.clock.micro_step_index += 1
    return True


def _prepare_candidate_fields(
    runtime: SimulationRuntime,
    run: RunState,
    macro_end_s: float,
) -> None:
    started_at = run.performance.start()
    runtime.particle_pusher.prepare_dt_candidates(
        runtime.cloud,
        float(run.clock.time_s),
        float(macro_end_s),
        float(runtime.rf_angular_frequency_rad_s),
        float(runtime.config.rf_phase_rad),
        float(runtime.rf_peak_to_reference_scale),
        float(runtime.template.collision_cross_section_m2),
        float(Boltzmann),
        float(runtime.integrator.collision_safety_factor),
        float(runtime.integrator.rf_frequency_hz if runtime.rf_enabled else 0.0),
        float(runtime.integrator.rf_safety_factor),
        float(runtime.integrator.acceleration_safety_factor),
        float(runtime.integrator.spatial_tolerance_m),
        float(runtime.integrator.min_dt_s),
        float(runtime.integrator.max_dt_s),
        int(runtime.collision_model_code),
        float(runtime.config.langevin_z_start_m),
        float(runtime.config.langevin_z_end_m),
        float(runtime.config.langevin_switch_probability),
        float(runtime.config.langevin_max_dt_s),
    )
    run.performance.finish("time_sample_fields", started_at)


def _numerical_failure(
    runtime: SimulationRuntime,
    run: RunState,
    error: InvalidTimeStepError,
    limiter_codes: np.ndarray,
    raw_limiter_codes: np.ndarray,
) -> dict[str, object]:
    failed_indices = np.asarray(error.particle_indices, dtype=np.int32)
    positions, velocities = runtime._download_selected_motion_state(failed_indices)
    failure: dict[str, object] = {
        "error": str(error),
        "particle_indices": failed_indices.tolist(),
        "dt_candidates_s": np.asarray(error.candidates_s, dtype=np.float64).tolist(),
        "positions_m": positions.tolist(),
        "velocities_m_per_s": velocities.tolist(),
        "dt_limiter_codes": limiter_codes[failed_indices].tolist(),
        "dt_raw_limiter_codes": raw_limiter_codes[failed_indices].tolist(),
    }
    try:
        samples = runtime._sample_local_state_batch(positions, run.clock.time_s)
        failure["local_external_field_v_per_m"] = (
            samples.external_field_v_per_m.tolist()
        )
        failure["local_sce_field_v_per_m"] = samples.sce_field_v_per_m.tolist()
        failure["local_total_field_v_per_m"] = samples.total_field_v_per_m.tolist()
    except Exception as diagnostic_error:
        failure["local_field_diagnostic_error"] = repr(diagnostic_error)
    return failure


def _prepare_timestep(
    runtime: SimulationRuntime,
    run: RunState,
    macro_end_s: float,
) -> PreparedTimeStep | None:
    _prepare_candidate_fields(runtime, run, macro_end_s)
    started_at = run.performance.start()
    candidates, limiter_codes, raw_limiter_codes, cached_rates = (
        runtime.particle_pusher.download_dt_diagnostics(runtime.cloud)
    )
    try:
        selected = select_active_timestep(
            candidates,
            run.particle_state["active"],
        )
    except InvalidTimeStepError as error:
        run.termination.numerical_failure = _numerical_failure(
            runtime,
            run,
            error,
            limiter_codes,
            raw_limiter_codes,
        )
        run.termination.finish("numerical_failure")
        run.performance.finish("time_dt_estimate", started_at)
        return None
    particle_index = int(selected.particle_index)
    dt_s = float(selected.dt_s)
    limiter = DT_LIMITER_NAMES.get(int(limiter_codes[particle_index]), "macro_end")
    raw_limiter = DT_LIMITER_NAMES.get(
        int(raw_limiter_codes[particle_index]),
        "macro_end",
    )
    run.performance.finish("time_dt_estimate", started_at)
    run.dt_tracker.update(dt_s, limiter=limiter, raw_limiter=raw_limiter)
    return PreparedTimeStep(
        dt_s=dt_s,
        cached_collision_rates_hz=np.asarray(cached_rates, dtype=np.float64),
    )


def _apply_collision_and_push(
    runtime: SimulationRuntime,
    run: RunState,
    prepared: PreparedTimeStep,
) -> None:
    state = run.particle_state
    started_at = run.performance.start()
    update = runtime._taichi_runtime_apply_collisions_before_push(
        state,
        prepared.dt_s,
        run.clock.time_s,
        prepared.cached_collision_rates_hz,
    )
    run.collision_count += int(update.collision_hits)
    run.performance.finish("time_collision_loop", started_at)
    if update.updated_indices.size > 0:
        started_at = run.performance.start()
        runtime._upload_active_state_updates(state, update.updated_indices)
        run.performance.finish("time_upload_state", started_at)
    if update.deactivated_indices.size > 0:
        started_at = run.performance.start()
        runtime._upload_deactivated_indices(update.deactivated_indices)
        run.performance.finish("time_upload_state", started_at)
    started_at = run.performance.start()
    runtime.particle_pusher.push_rk4(
        runtime.cloud,
        float(run.clock.time_s),
        float(prepared.dt_s),
        float(runtime.rf_angular_frequency_rad_s),
        float(runtime.config.rf_phase_rad),
        float(runtime.rf_peak_to_reference_scale),
    )
    run.performance.finish("time_rk4_push", started_at)


def _apply_terminal_and_source(
    runtime: SimulationRuntime,
    run: RunState,
    dt_s: float,
) -> None:
    old_time_s = run.clock.time_s
    new_time_s = old_time_s + dt_s
    run.clock.micro_step_index += 1
    started_at = run.performance.start()
    runtime._taichi_runtime_apply_terminal_boundary_events(
        run.particle_state,
        new_time_s,
        step_start_time_s=old_time_s,
        step_dt_s=dt_s,
    )
    run.performance.finish("time_terminal_events", started_at)
    started_at = run.performance.start()
    runtime._advance_capillary_source(run.particle_state, dt_s, new_time_s)
    run.performance.finish("time_source_injection", started_at)
    started_at = run.performance.start()
    runtime._inject_continuous_source(run.particle_state, new_time_s, dt_s)
    run.performance.finish("time_source_injection", started_at)
    run.clock.time_s = new_time_s


def _apply_particle_stop_conditions(
    runtime: SimulationRuntime,
    run: RunState,
) -> None:
    active_count = len(runtime._active_indices(run.particle_state))
    if active_count == 0 and not runtime._continuous_source_enabled():
        run.termination.finish("all_particles_terminal")
        return
    threshold = float(runtime.config.stop_active_fraction_below)
    if not runtime._continuous_source_enabled() and threshold > 0.0:
        if active_count / float(runtime.config.ion_count) <= threshold:
            run.termination.finish("active_fraction_below_threshold")


def _maybe_emit_progress(
    runtime: SimulationRuntime,
    run: RunState,
    observer: ProgressObserver | None,
) -> None:
    if observer is None:
        return
    now = time.monotonic()
    if now - run.last_progress_wall_time < run.progress_interval_s:
        return
    started_at = run.performance.start()
    runtime._download_motion_state_into(run.particle_state)
    run.performance.finish("time_download_state", started_at)
    observer(
        runtime._build_progress_snapshot(
            run.particle_state,
            run.clock.time_s,
            run.clock.macro_step_index,
            run.clock.micro_step_index,
            run.collision_count,
        )
    )
    run.last_progress_wall_time = now


def advance_micro_steps(
    runtime: SimulationRuntime,
    run: RunState,
    macro_end_s: float,
    observer: ProgressObserver | None,
) -> None:
    """Advance particles while the macro-step space-charge field is frozen."""

    while run.clock.time_s < macro_end_s:
        if _runtime_stop_requested(runtime, run):
            break
        if len(runtime._active_indices(run.particle_state)) == 0:
            if not runtime._continuous_source_enabled():
                run.termination.finish("all_particles_terminal")
                break
            if not _advance_idle_source(runtime, run, macro_end_s):
                break
            continue
        prepared = _prepare_timestep(runtime, run, macro_end_s)
        if prepared is None:
            break
        _apply_collision_and_push(runtime, run, prepared)
        _apply_terminal_and_source(runtime, run, prepared.dt_s)
        _apply_particle_stop_conditions(runtime, run)
        if run.termination.finished:
            break
        _maybe_emit_progress(runtime, run, observer)
