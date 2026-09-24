"""Public whole-macro collision operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config import CollisionBatchUpdate, LocalStateBatch
from .explicit import ExplicitCollisionMixin
from .hybrid import HybridCollisionMixin
from .selection import apply_collision_updates
from .update_helpers import CollisionUpdateMixin


class CollisionOperator(
    ExplicitCollisionMixin,
    HybridCollisionMixin,
    CollisionUpdateMixin,
):
    """Apply explicit or hybrid whole-macro collision updates."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def apply_updates(
        self,
        state: dict[str, np.ndarray],
        active_indices: np.ndarray,
        positions_m: np.ndarray,
        samples: LocalStateBatch,
        dt_s: float,
        time_s: float,
    ) -> CollisionBatchUpdate:
        return apply_collision_updates(
            self,
            state,
            active_indices,
            positions_m,
            samples,
            dt_s,
            time_s,
        )


__all__ = ["CollisionOperator"]
