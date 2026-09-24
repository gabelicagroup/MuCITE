"""Canonical collision operators and pluggable physics backends."""

from .constants import (
    EXPECTED_BUNDLED_VERSION,
    PSEUDOATOM_MASS_CEILING_DA,
    PSEUDOATOM_MASS_FLOOR_DA,
    SUPPORTED_IONSPA_BACKENDS,
)
from .models import ApproximateIonModel, FallbackIonModel, IonSpaLoadError
from .operator import CollisionOperator
from .probability import collision_probability_from_lambda_dt
from .protocol import CollisionPhysicsBackend
from .runtime import CollisionRuntimeMixin
from .factory import build_collision_physics_backend


def __getattr__(name: str):
    """Load the legacy IonSPA adapter only when callers explicitly request it."""

    if name == "IonSpaCollisionAdapter":
        from .adapter import IonSpaCollisionAdapter

        return IonSpaCollisionAdapter
    if name == "IonSpaBackendAdapter":
        from .ionspa_compat import IonSpaBackendAdapter

        return IonSpaBackendAdapter
    raise AttributeError(name)

__all__ = [
    "ApproximateIonModel",
    "CollisionOperator",
    "CollisionPhysicsBackend",
    "CollisionRuntimeMixin",
    "EXPECTED_BUNDLED_VERSION",
    "FallbackIonModel",
    "IonSpaCollisionAdapter",
    "IonSpaBackendAdapter",
    "IonSpaLoadError",
    "PSEUDOATOM_MASS_CEILING_DA",
    "PSEUDOATOM_MASS_FLOOR_DA",
    "SUPPORTED_IONSPA_BACKENDS",
    "build_collision_physics_backend",
    "collision_probability_from_lambda_dt",
]
