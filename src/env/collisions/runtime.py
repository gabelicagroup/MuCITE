"""Composed collision behavior for the main simulation runtime."""

from .runtime_common import CollisionStateMixin
from .runtime_legacy import CollisionLegacyRuntimeMixin
from .runtime_preselection import CollisionPreselectionMixin


class CollisionRuntimeMixin(
    CollisionPreselectionMixin,
    CollisionLegacyRuntimeMixin,
    CollisionStateMixin,
):
    """Aggregate collision runtime behavior without duplicating methods."""


__all__ = ["CollisionRuntimeMixin"]
