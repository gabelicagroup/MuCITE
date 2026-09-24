"""Compatibility facade for strict retained/generated artifact schemas."""

from .grid_metadata import (
    BakedGridMetadata,
    _required_grid_value,
    _validate_coordinate_axis,
    validate_baked_grid_metadata,
)
from .patxt import (
    PatxtHeader,
    PatxtMaskGrid,
    PatxtPotentialGrid,
    _first_missing_point,
    _header_float,
    _header_int,
    _parse_patxt_header,
    _read_patxt_2d,
    read_patxt_mask_grid,
    read_patxt_potential_grid,
)

__all__ = [
    "BakedGridMetadata",
    "PatxtHeader",
    "PatxtMaskGrid",
    "PatxtPotentialGrid",
    "read_patxt_mask_grid",
    "read_patxt_potential_grid",
    "validate_baked_grid_metadata",
]
