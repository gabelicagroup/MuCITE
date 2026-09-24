"""Argument-group registration helpers."""

from .boundary import add_boundary_arguments
from .collision import add_collision_arguments
from .field import add_field_arguments
from .ion import add_ion_arguments
from .output import add_output_arguments
from .pic import add_pic_arguments
from .runtime import add_runtime_arguments
from .source import add_source_arguments

__all__ = [
    "add_boundary_arguments",
    "add_collision_arguments",
    "add_field_arguments",
    "add_ion_arguments",
    "add_output_arguments",
    "add_pic_arguments",
    "add_runtime_arguments",
    "add_source_arguments",
]
