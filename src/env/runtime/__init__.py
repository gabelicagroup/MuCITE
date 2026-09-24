"""Non-collision model-runtime behavior and bootstrap composition."""

from .cloud_io import CloudStateIOMixin
from .fields import FieldRuntimeMixin
from .projection import ResultProjectionMixin
from .stages import StageRuntimeMixin


class ModelRuntimeMixin(
    StageRuntimeMixin,
    CloudStateIOMixin,
    FieldRuntimeMixin,
    ResultProjectionMixin,
):
    """Composed non-collision behavior for the simulation runtime."""


__all__ = ["ModelRuntimeMixin"]
