"""Electric, gas, and space-charge field models."""

from importlib import import_module
from typing import Any

__all__ = [
    "ALL_2D_FIELD_NAMES",
    "PIC_CACHE_FIELD_NAMES",
    "PIC_FIELD_NAMES",
    "STATIC_CACHE_FIELD_NAMES",
    "STATIC_POTENTIAL_FIELD_NAMES",
    "STATIC_RUNTIME_FIELD_NAMES",
    "UnifiedGrid2D",
    "RuntimeFieldSampler",
    "CartesianField3D",
    "load_baked_cartesian_field3d",
    "estimate_grid_storage_bytes",
    "estimate_runtime_grid_resources",
    "resolve_pic_grid_shape",
]

_AXISYMMETRIC_EXPORTS = {
    "ALL_2D_FIELD_NAMES",
    "PIC_CACHE_FIELD_NAMES",
    "PIC_FIELD_NAMES",
    "STATIC_CACHE_FIELD_NAMES",
    "STATIC_POTENTIAL_FIELD_NAMES",
    "STATIC_RUNTIME_FIELD_NAMES",
    "UnifiedGrid2D",
    "estimate_grid_storage_bytes",
    "estimate_runtime_grid_resources",
    "resolve_pic_grid_shape",
}
_EXPORTS = {
    **{
        name: (".axisymmetric_grid", name)
        for name in _AXISYMMETRIC_EXPORTS
    },
    "RuntimeFieldSampler": (".local_sampler", "RuntimeFieldSampler"),
    "CartesianField3D": (".cartesian3d", "CartesianField3D"),
    "load_baked_cartesian_field3d": (
        ".cartesian3d",
        "load_baked_cartesian_field3d",
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
