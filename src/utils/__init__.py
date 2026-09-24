"""Pure reusable mathematics with no framework-layer dependency."""

from .geometry import (
    interpolate_segment_state,
    locate_first_segment_hit,
    segment_cylinder_exit_fraction,
    segment_plane_crossing_fraction,
)

__all__ = [
    "interpolate_segment_state",
    "locate_first_segment_hit",
    "segment_cylinder_exit_fraction",
    "segment_plane_crossing_fraction",
]
