"""Particle transport strategies and composed pusher."""

from importlib import import_module
from typing import Any

__all__ = ["AdaptiveRK4Integrator", "ParticlePusher"]

_EXPORTS = {
    "AdaptiveRK4Integrator": (
        ".adaptive_integrator",
        "AdaptiveRK4Integrator",
    ),
    "ParticlePusher": (".pusher", "ParticlePusher"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
