"""Lazy construction of collision-physics backends."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from ...config import IonTemplate, PROJECT_ROOT, SimulationConfig
from .protocol import CollisionPhysicsBackend


def build_collision_physics_backend(
    *,
    config: SimulationConfig,
    template: IonTemplate,
    project_root: Path | None = None,
) -> CollisionPhysicsBackend:
    """Build only the explicitly selected backend.

    Keeping imports inside the selected branch guarantees that ``iict-lite``
    neither imports nor probes the optional IonSPA provider.
    """

    selected = str(config.collision_physics_backend).strip().lower()
    root = Path(project_root or PROJECT_ROOT)
    if selected == "iict-lite":
        module = import_module(".iict_lite.backend", package=__package__)
        return module.IictLiteBackend(config, template, project_root=root)
    if selected == "ionspa":
        module = import_module(".ionspa_compat", package=__package__)
        return module.IonSpaBackendAdapter(config, template, project_root=root)
    raise ValueError(
        "collision_physics_backend must be one of {iict-lite, ionspa}; "
        f"got {selected!r}."
    )


__all__ = ["build_collision_physics_backend"]
