"""Sparse collision-update aggregation."""

from __future__ import annotations

import numpy as np

from ...config import CollisionBatchUpdate


def _nonempty_indices(
    updates: tuple[CollisionBatchUpdate, ...],
    attribute: str,
) -> list[np.ndarray]:
    return [
        np.asarray(getattr(update, attribute), dtype=np.int32)
        for update in updates
        if getattr(update, attribute).size
    ]


def _merge_indices(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.int32)
    return np.unique(np.concatenate(chunks)).astype(np.int32, copy=False)


def combine_collision_updates(
    *updates: CollisionBatchUpdate,
) -> CollisionBatchUpdate:
    """Combine sparse updates while keeping deactivated slots authoritative."""

    updated_indices = _merge_indices(_nonempty_indices(updates, "updated_indices"))
    deactivated_indices = _merge_indices(
        _nonempty_indices(updates, "deactivated_indices")
    )
    if updated_indices.size and deactivated_indices.size:
        updated_indices = np.setdiff1d(
            updated_indices,
            deactivated_indices,
            assume_unique=True,
        ).astype(np.int32, copy=False)
    return CollisionBatchUpdate(
        collision_hits=sum(int(update.collision_hits) for update in updates),
        fragmentation_hits=sum(int(update.fragmentation_hits) for update in updates),
        collision_real_ions_represented=sum(
            float(update.collision_real_ions_represented) for update in updates
        ),
        fragmented_real_ions_represented=sum(
            float(update.fragmented_real_ions_represented) for update in updates
        ),
        updated_indices=updated_indices,
        deactivated_indices=deactivated_indices,
    )


class CollisionUpdateMixin:
    """Compatibility methods for sparse update aggregation."""

    @staticmethod
    def _combine_updates(
        *updates: CollisionBatchUpdate,
    ) -> CollisionBatchUpdate:
        return combine_collision_updates(*updates)


__all__ = ["CollisionUpdateMixin", "combine_collision_updates"]
