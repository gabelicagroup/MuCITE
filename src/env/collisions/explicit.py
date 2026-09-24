"""Explicit IonSPA collision and fragmentation updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...config import PARTICLE_FRAGMENTED, SLOT_FREE, CollisionBatchUpdate


@dataclass(frozen=True)
class _FragmentationUpdate:
    hits: int
    represented_real_ions: float
    deactivated_indices: np.ndarray


def _fragmentation_mode(runtime: Any) -> str:
    mode = str(runtime.config.fragmentation_mode).lower()
    if mode not in {"transport", "loss", "off"}:
        raise ValueError(
            f"Unsupported fragmentation_mode: {runtime.config.fragmentation_mode!r}"
        )
    return mode


def _transport_fragmentation(
    runtime: Any,
    hit_indices: np.ndarray,
    probability: np.ndarray,
) -> _FragmentationUpdate:
    parent_before = np.asarray(
        runtime._particle_parent_weight[hit_indices],
        dtype=np.float64,
    )
    fragment_weights = parent_before * probability
    parent_after = np.maximum(parent_before - fragment_weights, 0.0)
    fragment_mask = fragment_weights > 0.0
    hits = int(np.count_nonzero(fragment_mask))
    represented = 0.0
    if hits > 0:
        represented = float(np.sum(fragment_weights[fragment_mask]))
        runtime._particle_parent_weight[hit_indices] = parent_after
    return _FragmentationUpdate(
        hits,
        represented,
        np.zeros(0, dtype=np.int32),
    )


def _record_partial_fragment_events(
    runtime: Any,
    fragment_indices: np.ndarray,
    full_fragment_mask: np.ndarray,
    fragment_positions: np.ndarray,
    fragment_velocities: np.ndarray,
    fragment_temperatures: np.ndarray,
    fragment_weights: np.ndarray,
    time_s: float,
) -> None:
    partial_indices = fragment_indices[~full_fragment_mask]
    if partial_indices.size == 0:
        return
    partial_track_ids = np.arange(
        runtime._next_track_id,
        runtime._next_track_id + int(partial_indices.size),
        dtype=np.int64,
    )
    runtime._next_track_id += int(partial_indices.size)
    runtime._record_terminal_events(
        partial_indices,
        np.full(partial_indices.shape, PARTICLE_FRAGMENTED, dtype=np.int16),
        event_time_s=float(time_s),
        positions_m=fragment_positions[~full_fragment_mask],
        velocities_m_per_s=fragment_velocities[~full_fragment_mask],
        temperatures_k=fragment_temperatures[~full_fragment_mask],
        represented_real_ions=fragment_weights[~full_fragment_mask],
        track_ids=partial_track_ids,
    )


def _record_full_fragment_events(
    runtime: Any,
    fragment_indices: np.ndarray,
    full_fragment_mask: np.ndarray,
    fragment_positions: np.ndarray,
    fragment_velocities: np.ndarray,
    fragment_temperatures: np.ndarray,
    fragment_weights: np.ndarray,
    time_s: float,
) -> None:
    full_indices = fragment_indices[full_fragment_mask]
    if full_indices.size == 0:
        return
    runtime._record_terminal_events(
        full_indices,
        np.full(full_indices.shape, PARTICLE_FRAGMENTED, dtype=np.int16),
        event_time_s=float(time_s),
        positions_m=fragment_positions[full_fragment_mask],
        velocities_m_per_s=fragment_velocities[full_fragment_mask],
        temperatures_k=fragment_temperatures[full_fragment_mask],
        represented_real_ions=fragment_weights[full_fragment_mask],
    )


def _record_fragment_events(
    runtime: Any,
    hit_positions: np.ndarray,
    velocity_after: np.ndarray,
    temperature_after: np.ndarray,
    fragment_mask: np.ndarray,
    fragment_indices: np.ndarray,
    fragment_weights: np.ndarray,
    full_fragment_mask: np.ndarray,
    time_s: float,
) -> None:
    fragment_positions = np.asarray(hit_positions, dtype=np.float64)[fragment_mask]
    fragment_velocities = velocity_after[fragment_mask]
    fragment_temperatures = temperature_after[fragment_mask]
    _record_partial_fragment_events(
        runtime,
        fragment_indices,
        full_fragment_mask,
        fragment_positions,
        fragment_velocities,
        fragment_temperatures,
        fragment_weights,
        time_s,
    )
    _record_full_fragment_events(
        runtime,
        fragment_indices,
        full_fragment_mask,
        fragment_positions,
        fragment_velocities,
        fragment_temperatures,
        fragment_weights,
        time_s,
    )


def _record_omitted_fragments(
    runtime: Any,
    fragmentation_hits: int,
    represented_real_ions: float,
) -> None:
    runtime._accumulate_terminal_accounting(
        PARTICLE_FRAGMENTED,
        fragmentation_hits,
        represented_real_ions,
        parent_real_ions_remaining=0.0,
        fragmented_real_ions_represented=represented_real_ions,
    )
    runtime.terminal_event_recorder.omit_summary(
        PARTICLE_FRAGMENTED,
        fragmentation_hits,
        represented_real_ions,
        parent_real_ions_remaining=0.0,
        fragmented_real_ions_represented=represented_real_ions,
    )


def _deactivate_full_fragments(
    runtime: Any,
    state: dict[str, np.ndarray],
    full_fragment_indices: np.ndarray,
    time_s: float,
) -> np.ndarray:
    if full_fragment_indices.size == 0:
        return np.zeros(0, dtype=np.int32)
    runtime._particle_status_codes[full_fragment_indices] = PARTICLE_FRAGMENTED
    runtime._particle_tof_s[full_fragment_indices] = (
        runtime._particle_tof_from_event_time(full_fragment_indices, time_s)
    )
    runtime._particle_phase_codes[full_fragment_indices] = SLOT_FREE
    deactivated_indices: set[int] = set()
    for slot_index in full_fragment_indices.tolist():
        runtime._release_free_slot(state, int(slot_index))
        deactivated_indices.add(int(slot_index))
    return np.asarray(sorted(deactivated_indices), dtype=np.int32)


def _apply_loss_fragmentation(
    runtime: Any,
    state: dict[str, np.ndarray],
    hit_indices: np.ndarray,
    hit_positions: np.ndarray,
    velocity_after: np.ndarray,
    temperature_after: np.ndarray,
    probability: np.ndarray,
    time_s: float,
) -> _FragmentationUpdate:
    weights_before = np.asarray(state["weight"][hit_indices], dtype=np.float64)
    fragment_weights = weights_before * probability
    survivor_weights = weights_before - fragment_weights
    fragment_mask = fragment_weights > 0.0
    hits = int(np.count_nonzero(fragment_mask))
    if hits == 0:
        return _FragmentationUpdate(0, 0.0, np.zeros(0, dtype=np.int32))
    represented = float(np.sum(fragment_weights[fragment_mask]))
    fragment_indices = hit_indices[fragment_mask]
    selected_weights = fragment_weights[fragment_mask]
    selected_survivors = survivor_weights[fragment_mask]
    full_mask = selected_survivors <= 0.0
    if str(runtime.config.terminal_event_mode).lower() == "all":
        _record_fragment_events(
            runtime, hit_positions, velocity_after, temperature_after,
            fragment_mask, fragment_indices, selected_weights, full_mask, time_s,
        )
    else:
        _record_omitted_fragments(runtime, hits, represented)
    deactivated = _deactivate_full_fragments(
        runtime,
        state,
        fragment_indices[full_mask],
        time_s,
    )
    survivor_weights = np.maximum(survivor_weights, 0.0)
    state["weight"][hit_indices] = survivor_weights
    runtime._particle_weight[hit_indices] = survivor_weights
    runtime._particle_parent_weight[hit_indices] = survivor_weights
    return _FragmentationUpdate(hits, represented, deactivated)


def _apply_fragmentation(
    runtime: Any,
    state: dict[str, np.ndarray],
    hit_indices: np.ndarray,
    hit_positions: np.ndarray,
    velocity_after: np.ndarray,
    temperature_after: np.ndarray,
    probability: np.ndarray,
    time_s: float,
) -> _FragmentationUpdate:
    mode = _fragmentation_mode(runtime)
    if mode == "transport":
        return _transport_fragmentation(runtime, hit_indices, probability)
    if mode == "loss":
        return _apply_loss_fragmentation(
            runtime, state, hit_indices, hit_positions, velocity_after,
            temperature_after, probability, time_s,
        )
    return _FragmentationUpdate(0, 0.0, np.zeros(0, dtype=np.int32))


def _apply_hit_physics(
    runtime: Any,
    state: dict[str, np.ndarray],
    hit_indices: np.ndarray,
    gas_velocities: np.ndarray,
    gas_temperatures: np.ndarray,
    gas_number_density: np.ndarray,
    gas_mass_density: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = runtime.adapter.apply_collision_batch(
        velocities_m_per_s=state["velocities"][hit_indices],
        masses_kg=state["mass"][hit_indices],
        internal_temperatures_k=state["temperature"][hit_indices],
        gas_velocities_m_per_s=np.asarray(gas_velocities, dtype=np.float64),
        gas_temperatures_k=np.asarray(gas_temperatures, dtype=np.float64),
        gas_number_density_m3=np.asarray(gas_number_density, dtype=np.float64),
        gas_mass_density_kg_per_m3=np.asarray(gas_mass_density, dtype=np.float64),
        template=runtime.template,
        gas_name=str(getattr(runtime.config, "gas_species", "n2")),
        dt_s=dt_s,
        rng=runtime.rng,
    )
    velocity_after, temperature_after, probability = result
    state["velocities"][hit_indices] = velocity_after
    state["temperature"][hit_indices] = temperature_after
    runtime._particle_collision_counts[hit_indices] += 1
    return velocity_after, temperature_after, probability


def _build_explicit_update(
    hit_indices: np.ndarray,
    collision_real_ions: float,
    fragmentation: _FragmentationUpdate,
) -> CollisionBatchUpdate:
    updated_indices = np.setdiff1d(
        hit_indices,
        fragmentation.deactivated_indices,
        assume_unique=True,
    ).astype(np.int32, copy=False)
    return CollisionBatchUpdate(
        collision_hits=int(hit_indices.size),
        fragmentation_hits=fragmentation.hits,
        collision_real_ions_represented=collision_real_ions,
        fragmented_real_ions_represented=float(
            fragmentation.represented_real_ions
        ),
        updated_indices=updated_indices,
        deactivated_indices=fragmentation.deactivated_indices,
    )


class ExplicitCollisionMixin:
    """Apply IonSPA and fragmentation to selected binary hits."""

    def _apply_explicit_hits(
        self,
        state: dict[str, np.ndarray],
        hit_indices: np.ndarray,
        hit_positions: np.ndarray,
        gas_velocities: np.ndarray,
        gas_temperatures: np.ndarray,
        gas_number_density: np.ndarray,
        gas_mass_density: np.ndarray,
        dt_s: float,
        time_s: float,
    ) -> CollisionBatchUpdate:
        runtime = self.runtime
        hit_indices = np.asarray(hit_indices, dtype=np.int32)
        if hit_indices.size == 0:
            return runtime._empty_collision_update()
        velocity_after, temperature_after, probability = _apply_hit_physics(
            runtime, state, hit_indices, gas_velocities, gas_temperatures,
            gas_number_density, gas_mass_density, dt_s,
        )
        probability = np.clip(
            np.asarray(probability, dtype=np.float64),
            0.0,
            1.0,
        )
        collision_real_ions = float(
            np.sum(np.asarray(state["weight"][hit_indices], dtype=np.float64))
        )
        fragmentation = _apply_fragmentation(
            runtime, state, hit_indices, hit_positions, velocity_after,
            temperature_after, probability, time_s,
        )
        runtime._collision_real_ions_represented += collision_real_ions
        runtime._fragmented_real_ions_represented += (
            fragmentation.represented_real_ions
        )
        runtime._collision_macro_events += int(hit_indices.size)
        return _build_explicit_update(
            hit_indices,
            collision_real_ions,
            fragmentation,
        )


__all__ = ["ExplicitCollisionMixin"]
