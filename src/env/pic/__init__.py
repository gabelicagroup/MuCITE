"""Axisymmetric particle-in-cell operators."""

from importlib import import_module
from typing import Any

__all__ = ["PICSolver", "PoissonConvergenceError"]

_EXPORTS = {
    "PICSolver": (".operator", "PICSolver"),
    "PoissonConvergenceError": (
        ".operator",
        "PoissonConvergenceError",
    ),
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
