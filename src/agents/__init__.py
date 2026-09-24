"""Particle/entity state for the model layer."""

from importlib import import_module
from typing import Any

__all__ = [
    "ParticleRuntimeState",
    "IonCloud3D",
    "IonState",
    "TerminalAccounting",
    "TerminalEventBatch",
]

_EXPORTS = {
    "ParticleRuntimeState": (".slot_state", "ParticleRuntimeState"),
    "IonCloud3D": (".taichi_cloud", "IonCloud3D"),
    "IonState": (".ion", "IonState"),
    "TerminalAccounting": (".events", "TerminalAccounting"),
    "TerminalEventBatch": (".events", "TerminalEventBatch"),
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
