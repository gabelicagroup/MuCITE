"""Physical-time dissociation, independent of binary collision selection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config import PARTICLE_FRAGMENTED, SLOT_ACTIVE
from ..boundaries.events import TerminalEvents
from .dissociation_cache import cached_fragmentation_rates

DISSOCIATION_SCHEDULE_VERSION = "post-collision-alive-time-v1"
LOSS_WEIGHT_UNDERFLOW_LIMIT = float(np.finfo(np.float64).tiny)


def survival_update(
    parent_weight: np.ndarray, rate_s_inverse: np.ndarray, alive_dt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return remaining parents, dissociated weight and dimensionless hazard."""
    parent, rate, dt = np.broadcast_arrays(
        np.asarray(parent_weight, dtype=np.float64),
        np.asarray(rate_s_inverse, dtype=np.float64),
        np.asarray(alive_dt_s, dtype=np.float64),
    )
    if (
        not all(np.all(np.isfinite(a)) for a in (parent, rate, dt))
        or any(np.any(a < 0) for a in (parent, rate, dt))
    ):
        raise FloatingPointError("Invalid parent weight, dissociation rate or exposure time.")
    hazard = rate * dt
    if not np.all(np.isfinite(hazard)):
        raise FloatingPointError("Non-finite dissociation hazard increment.")
    # Do not use parent*(1-p): p rounds to 1 long before exp(-H) underflows.
    remaining = parent * np.exp(-hazard)
    # Preserve zero exposure exactly, including on backends that flush subnormals.
    remaining = np.where(hazard == 0.0, parent, remaining)
    return remaining, parent * -np.expm1(-hazard), hazard


def _loss_motion(runtime: Any, indices: np.ndarray, events: TerminalEvents):
    positions, velocities = runtime._download_selected_motion_state(indices)
    _, offsets, event_offsets = np.intersect1d(
        indices, events.indices, return_indices=True,
    )
    positions[offsets] = events.positions_m[event_offsets]
    velocities[offsets] = events.velocities_m_per_s[event_offsets]
    return positions, velocities


def _record_loss(
    runtime: Any, state: dict[str, np.ndarray], indices: np.ndarray,
    lost: np.ndarray, surviving: np.ndarray, event_times_s: np.ndarray,
    events: TerminalEvents,
) -> None:
    """Record integrated loss at the exposure endpoint, with unique event IDs."""
    mask = lost > 0
    selected = indices[mask]
    if selected.size == 0:
        return
    if str(runtime.config.terminal_event_mode).lower() != "all":
        summary = dict(parent_real_ions_remaining=0.0,
                       fragmented_real_ions_represented=float(np.sum(lost[mask])))
        runtime._accumulate_terminal_accounting(
            PARTICLE_FRAGMENTED, int(selected.size), float(np.sum(lost[mask])),
            **summary,
        )
        runtime.terminal_event_recorder.omit_summary(
            PARTICLE_FRAGMENTED, int(selected.size), float(np.sum(lost[mask])),
            **summary,
        )
        return
    positions, velocities = _loss_motion(runtime, selected, events)
    track_ids = runtime._particle_track_id[selected].copy()
    partial = surviving[mask] > 0.0
    track_ids[partial] = runtime._next_track_ids(int(np.count_nonzero(partial)))
    runtime._record_terminal_events(
        selected, np.full(selected.size, PARTICLE_FRAGMENTED, dtype=np.int16),
        event_time_s=event_times_s[mask], positions_m=positions,
        velocities_m_per_s=velocities, track_ids=track_ids,
        temperatures_k=state["temperature"][selected],
        represented_real_ions=lost[mask],
    )


def _exposure_times(
    runtime: Any, indices: np.ndarray, start_s: float, dt_s: float,
    events: TerminalEvents,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(start_s) or not np.isfinite(dt_s) or dt_s < 0.0:
        raise FloatingPointError("Invalid dissociation step time.")
    end_times = np.full(indices.size, start_s + dt_s, dtype=np.float64)
    if events.count:
        _, offsets, event_offsets = np.intersect1d(
            indices, events.indices, return_indices=True,
        )
        terminal_times = events.event_times_s[event_offsets]
        tolerance = 16 * np.finfo(float).eps * max(abs(start_s), dt_s, 1e-30)
        if (np.any(~np.isfinite(terminal_times))
                or np.any(terminal_times < start_s - tolerance)
                or np.any(terminal_times > start_s + dt_s + tolerance)):
            raise FloatingPointError("Localized terminal time is outside the micro step.")
        end_times[offsets] = np.clip(terminal_times, start_s, start_s + dt_s)
    birth_times = runtime._particle_birth_time_s[indices]
    begin_times = np.maximum(start_s, np.where(np.isfinite(birth_times), birth_times, start_s))
    return np.maximum(end_times - begin_times, 0.0), end_times


def _check_transport_survival(runtime: Any, indices: np.ndarray) -> None:
    weights = runtime._particle_weight[indices]
    valid = weights > 0.0
    actual = runtime._particle_parent_weight[indices][valid] / weights[valid]
    expected = np.exp(-runtime._particle_dissociation_hazard[indices][valid])
    error = np.abs(actual - expected)
    diagnostics = runtime.dissociation_runtime_info
    diagnostics["transport_survival_checks"] += int(np.count_nonzero(valid))
    diagnostics["transport_survival_max_abs_error"] = max(
        diagnostics["transport_survival_max_abs_error"], float(np.max(error, initial=0)),
    )
    normal = expected >= np.finfo(float).tiny
    relative = error[normal] / expected[normal]
    diagnostics["transport_survival_max_relative_error"] = max(
        diagnostics["transport_survival_max_relative_error"],
        float(np.max(relative, initial=0)),
    )


def _apply_loss_weights(
    runtime: Any, state: dict[str, np.ndarray], indices: np.ndarray,
    surviving: np.ndarray, end_times: np.ndarray, events: TerminalEvents,
) -> TerminalEvents:
    state["weight"][indices] = surviving
    runtime._particle_weight[indices] = surviving
    runtime.cloud.update_weights(indices, surviving, int(indices.size))
    depleted = indices[surviving == 0.0]
    if depleted.size:
        positions, velocities = _loss_motion(runtime, depleted, events)
        state["positions"][depleted] = positions
        state["velocities"][depleted] = velocities
        runtime._particle_status_codes[depleted] = PARTICLE_FRAGMENTED
        runtime._particle_tof_s[depleted] = runtime._particle_tof_from_event_time(
            depleted, end_times[surviving == 0.0],
        )
        for slot in depleted:
            runtime._release_free_slot(state, int(slot))
        runtime._upload_deactivated_indices(depleted)
        events = events.take(np.flatnonzero(~np.isin(events.indices, depleted)))
    return events


def _transport_indices(runtime: Any, state: dict[str, np.ndarray]) -> np.ndarray:
    active = (runtime._active_indices(state) if hasattr(runtime, "_active_indices")
              else np.flatnonzero(state["active"] != 0))
    return active[state["phase"][active] == SLOT_ACTIVE].astype(np.int32, copy=False)


def check_final_transport_survival(runtime: Any, state: dict[str, np.ndarray]) -> None:
    """Always audit remaining transport parents before constructing a result."""
    if str(runtime.config.fragmentation_mode).lower() == "transport":
        _check_transport_survival(runtime, _transport_indices(runtime, state))


def _scheduled_survival_check(
    runtime: Any, indices: np.ndarray, events: TerminalEvents,
) -> None:
    runtime._dissociation_step_count += 1
    every = int(getattr(runtime.config, "dissociation_check_every", 100))
    if every > 0 and runtime._dissociation_step_count % every == 0:
        _check_transport_survival(runtime, indices)
    elif events.count:
        # Terminal slots must be checked before release even between audits.
        _check_transport_survival(runtime, np.intersect1d(indices, events.indices))


def _commit_parent_decay(
    runtime: Any, indices: np.ndarray, parents: np.ndarray, surviving: np.ndarray,
    lost: np.ndarray, hazard: np.ndarray, mode: str,
) -> None:
    if mode == "loss":
        # Subnormal products can stop decaying through rounding. Account for
        # the entire residual before freeing the slot; do not prune at W < 1.
        depleted = (parents > 0) & (hazard > 0) & (surviving < LOSS_WEIGHT_UNDERFLOW_LIMIT)
        surviving[depleted] = 0.0
        lost[depleted] = parents[depleted]
        runtime.dissociation_runtime_info["loss_underflow_depletions"] += int(
            np.count_nonzero(depleted)
        )
    runtime._particle_dissociation_hazard[indices] += hazard
    runtime._particle_parent_weight[indices] = surviving
    runtime._fragmented_real_ions_represented += float(np.sum(lost))


def advance_dissociation(
    runtime: Any, state: dict[str, np.ndarray], *,
    step_start_time_s: float, step_dt_s: float, events: TerminalEvents,
) -> TerminalEvents:
    """Accumulate alive time once, using post-collision temperature for the step."""
    mode = str(runtime.config.fragmentation_mode).lower()
    if mode == "off":
        return events
    if mode not in {"transport", "loss"}:
        raise ValueError(f"Unsupported fragmentation mode: {mode}")
    indices = _transport_indices(runtime, state)
    if indices.size == 0:
        return events
    exposure, end_times = _exposure_times(
        runtime, indices, step_start_time_s, step_dt_s, events,
    )
    rates = cached_fragmentation_rates(runtime, state, indices)
    parents = runtime._particle_parent_weight[indices]
    rates = np.broadcast_to(np.asarray(rates, dtype=np.float64), parents.shape)
    surviving, lost, hazard = survival_update(parents, rates, exposure)
    _commit_parent_decay(runtime, indices, parents, surviving, lost, hazard, mode)
    if mode == "transport":
        _scheduled_survival_check(runtime, indices, events)
        return events
    _record_loss(runtime, state, indices, lost, surviving, end_times, events)
    return _apply_loss_weights(runtime, state, indices, surviving, end_times, events)
