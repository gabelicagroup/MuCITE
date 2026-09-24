"""Shared collision-runtime state and dispatch methods."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...config import CollisionBatchUpdate, LocalStateBatch


class CollisionStateMixin:
    """Collision counters, empty updates, and runtime dispatch."""

    def _reset_collision_statistics(self) -> None:
        self._collision_real_ions_represented = 0.0
        self._fragmented_real_ions_represented = 0.0
        self._collision_macro_events = 0
        self._hybrid_langevin_macro_events = 0
        self._hybrid_langevin_real_ions_represented = 0.0
        self._hybrid_langevin_estimated_binary_collisions = 0.0

    def _apply_collision_updates(
        self,
        state: dict[str, np.ndarray],
        active_indices: np.ndarray,
        positions_m: np.ndarray,
        samples: LocalStateBatch,
        dt_s: float,
        time_s: float,
    ) -> CollisionBatchUpdate:
        return self.collision_operator.apply_updates(
            state,
            active_indices,
            positions_m,
            samples,
            dt_s,
            time_s,
        )

    def _empty_collision_update(self) -> CollisionBatchUpdate:
        empty_indices = np.zeros(0, dtype=np.int32)
        return CollisionBatchUpdate(
            collision_hits=0,
            fragmentation_hits=0,
            collision_real_ions_represented=0.0,
            fragmented_real_ions_represented=0.0,
            updated_indices=empty_indices,
            deactivated_indices=empty_indices,
        )

    def _taichi_runtime_apply_collisions_before_push(
        self,
        state: dict[str, np.ndarray],
        dt_s: float,
        time_s: float,
        cached_collision_rates_hz: Optional[np.ndarray] = None,
    ) -> CollisionBatchUpdate:
        if not self.config.collision_preselection_enabled:
            return self._taichi_runtime_apply_collisions_before_push_legacy(
                state,
                dt_s,
                time_s,
            )
        return self._taichi_runtime_apply_collisions_before_push_preselected(
            state,
            dt_s,
            time_s,
            cached_collision_rates_hz=cached_collision_rates_hz,
        )


__all__ = ["CollisionStateMixin"]
