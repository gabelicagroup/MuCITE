"""Cached-rate collision preselection for the Taichi runtime."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...config import CollisionBatchUpdate
from .probability import collision_probability_from_lambda_dt


def _cached_rates(
    runtime: Any,
    cached_collision_rates_hz: Optional[np.ndarray],
) -> np.ndarray:
    if cached_collision_rates_hz is None:
        return np.asarray(
            runtime.cloud.collision_rate_hz.to_numpy(),
            dtype=np.float64,
        )
    rates_hz = np.asarray(cached_collision_rates_hz, dtype=np.float64)
    if rates_hz.shape != (runtime.cloud.n_particles,):
        raise ValueError(
            "cached_collision_rates_hz must have one value per particle slot."
        )
    return rates_hz


def _batch_size(runtime: Any, active_count: int) -> int:
    configured = int(runtime.config.collision_batch_size)
    if configured <= 0 or configured >= active_count:
        return int(active_count)
    return configured


def _download_chunk(
    runtime: Any,
    state: dict[str, np.ndarray],
    chunk_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions_m, velocities_m_per_s = (
        runtime._download_selected_motion_state(chunk_indices)
    )
    state["positions"][chunk_indices] = positions_m
    state["velocities"][chunk_indices] = velocities_m_per_s
    return positions_m, velocities_m_per_s


def _chunk_lambda_dt(
    cached_rates_hz: np.ndarray,
    chunk_indices: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    lambda_dt = np.asarray(
        cached_rates_hz[chunk_indices] * float(dt_s),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(lambda_dt)) or np.any(lambda_dt < 0.0):
        raise RuntimeError(
            "Cached collision rates must be finite and non-negative "
            "after a valid dt candidate selection."
        )
    return lambda_dt


def _hybrid_mask(
    runtime: Any,
    positions_m: np.ndarray,
    lambda_dt: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(lambda_dt.size, dtype=bool)
    if str(runtime.config.collision_model).lower() == "hybrid-langevin":
        mask = (
            (positions_m[:, 2] >= float(runtime.config.langevin_z_start_m))
            & (positions_m[:, 2] <= float(runtime.config.langevin_z_end_m))
            & (lambda_dt >= float(runtime.config.langevin_switch_probability))
        )
    return mask


def _apply_preselected_hybrid(
    runtime: Any,
    state: dict[str, np.ndarray],
    chunk_indices: np.ndarray,
    positions_m: np.ndarray,
    velocities_m_per_s: np.ndarray,
    lambda_dt: np.ndarray,
    hybrid_mask: np.ndarray,
    time_s: float,
) -> Optional[CollisionBatchUpdate]:
    if not np.any(hybrid_mask):
        return None
    hybrid_indices = np.asarray(chunk_indices[hybrid_mask], dtype=np.int32)
    hybrid_samples = runtime._sample_local_state_batch(
        positions_m[hybrid_mask],
        time_s,
    )
    return runtime.collision_operator._apply_hybrid_selected(
        state,
        hybrid_indices,
        velocities_m_per_s[hybrid_mask],
        hybrid_samples.gas_velocity_m_per_s,
        hybrid_samples.temperature_k,
        lambda_dt[hybrid_mask],
    )


def _apply_preselected_explicit(
    runtime: Any,
    state: dict[str, np.ndarray],
    chunk_indices: np.ndarray,
    positions_m: np.ndarray,
    lambda_dt: np.ndarray,
    hybrid_mask: np.ndarray,
    dt_s: float,
    time_s: float,
) -> Optional[CollisionBatchUpdate]:
    explicit_offsets = np.flatnonzero(~hybrid_mask)
    random_values = runtime.rng.random(int(explicit_offsets.size))
    probability = collision_probability_from_lambda_dt(
        lambda_dt[explicit_offsets]
    )
    hit_mask = random_values < probability
    if not np.any(hit_mask):
        return None
    hit_offsets = explicit_offsets[hit_mask]
    hit_indices = np.asarray(chunk_indices[hit_offsets], dtype=np.int32)
    hit_positions = np.asarray(positions_m[hit_offsets], dtype=np.float64)
    hit_samples = runtime._sample_local_state_batch(
        hit_positions,
        time_s,
    )
    return runtime.collision_operator._apply_explicit_hits(
        state, hit_indices, hit_positions,
        hit_samples.gas_velocity_m_per_s, hit_samples.temperature_k,
        hit_samples.number_density_m3, hit_samples.mass_density_kg_per_m3,
        dt_s, time_s,
    )


def _preselected_chunk_updates(
    runtime: Any,
    state: dict[str, np.ndarray],
    chunk_indices: np.ndarray,
    cached_rates_hz: np.ndarray,
    dt_s: float,
    time_s: float,
) -> list[CollisionBatchUpdate]:
    positions_m, velocities_m_per_s = _download_chunk(
        runtime,
        state,
        chunk_indices,
    )
    lambda_dt = _chunk_lambda_dt(
        cached_rates_hz,
        chunk_indices,
        dt_s,
    )
    hybrid_mask = _hybrid_mask(runtime, positions_m, lambda_dt)
    updates: list[CollisionBatchUpdate] = []
    hybrid_update = _apply_preselected_hybrid(
        runtime, state, chunk_indices, positions_m, velocities_m_per_s,
        lambda_dt, hybrid_mask, time_s,
    )
    if hybrid_update is not None:
        updates.append(hybrid_update)
    explicit_update = _apply_preselected_explicit(
        runtime, state, chunk_indices, positions_m, lambda_dt, hybrid_mask,
        dt_s, time_s,
    )
    if explicit_update is not None:
        updates.append(explicit_update)
    return updates


class CollisionPreselectionMixin:
    """Select cached-rate collision candidates in stable chunk order."""

    def _taichi_runtime_apply_collisions_before_push_preselected(
        self,
        state: dict[str, np.ndarray],
        dt_s: float,
        time_s: float,
        *,
        cached_collision_rates_hz: Optional[np.ndarray] = None,
    ) -> CollisionBatchUpdate:
        active_indices = self._active_indices(state)
        if active_indices.size == 0:
            return self._empty_collision_update()
        rates_hz = _cached_rates(self, cached_collision_rates_hz)
        batch_size = _batch_size(self, int(active_indices.size))
        updates: list[CollisionBatchUpdate] = []
        for start in range(0, active_indices.size, batch_size):
            chunk_indices = np.asarray(
                active_indices[start : start + batch_size],
                dtype=np.int32,
            )
            updates.extend(
                _preselected_chunk_updates(
                    self,
                    state,
                    chunk_indices,
                    rates_hz,
                    dt_s,
                    time_s,
                )
            )
        if not updates:
            return self._empty_collision_update()
        return self.collision_operator._combine_updates(*updates)


__all__ = ["CollisionPreselectionMixin"]
