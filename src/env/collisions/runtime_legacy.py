"""Legacy full-gather collision runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config import CollisionBatchUpdate


def _apply_legacy_chunk(
    runtime: Any,
    state: dict[str, np.ndarray],
    chunk_indices: np.ndarray,
    dt_s: float,
    time_s: float,
) -> CollisionBatchUpdate:
    positions_m, velocities_m_per_s = (
        runtime._download_selected_motion_state(chunk_indices)
    )
    state["positions"][chunk_indices] = positions_m
    state["velocities"][chunk_indices] = velocities_m_per_s
    samples = runtime._sample_local_state_batch(positions_m, time_s)
    return runtime._apply_collision_updates(
        state,
        chunk_indices,
        positions_m,
        samples,
        dt_s,
        time_s,
    )


class CollisionLegacyRuntimeMixin:
    """Retain the fixed-seed full CPU gather path."""

    def _taichi_runtime_apply_collisions_before_push_legacy(
        self,
        state: dict[str, np.ndarray],
        dt_s: float,
        time_s: float,
    ) -> CollisionBatchUpdate:
        active_indices = self._active_indices(state)
        if active_indices.size == 0:
            return self._empty_collision_update()
        batch_size = int(self.config.collision_batch_size)
        if batch_size <= 0 or batch_size >= active_indices.size:
            return _apply_legacy_chunk(
                self,
                state,
                active_indices,
                dt_s,
                time_s,
            )
        updates: list[CollisionBatchUpdate] = []
        for start in range(0, active_indices.size, batch_size):
            chunk_indices = np.asarray(
                active_indices[start : start + batch_size],
                dtype=np.int32,
            )
            updates.append(
                _apply_legacy_chunk(
                    self,
                    state,
                    chunk_indices,
                    dt_s,
                    time_s,
                )
            )
        return self.collision_operator._combine_updates(*updates)


__all__ = ["CollisionLegacyRuntimeMixin"]
