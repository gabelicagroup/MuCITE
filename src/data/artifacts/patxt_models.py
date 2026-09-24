"""Data transfer objects used by the strict PATXT parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PatxtHeader:
    raw: dict[str, str]
    mode: int
    symmetry: str
    field_type: str
    data_format: str
    fast_adjustable: int
    mirror_x: int
    mirror_y: int
    mirror_z: int
    nx: int
    ny: int
    nz: int
    grids_per_mm: float
    max_voltage_v: float


@dataclass(frozen=True)
class PatxtPotentialGrid:
    header: PatxtHeader
    potential_v: np.ndarray
    electrode_mask: np.ndarray
    point_count: int


@dataclass(frozen=True)
class PatxtMaskGrid:
    header: PatxtHeader
    metal_mask: np.ndarray
    electrode_id: np.ndarray
    point_count: int


@dataclass
class _PatxtParseState:
    raw_header: dict[str, str] = field(default_factory=dict)
    header: Optional[PatxtHeader] = None
    section: str = "before_header"
    saw_begin_header: bool = False
    saw_end_header: bool = False
    saw_begin_points: bool = False
    saw_end_points: bool = False
    point_count: int = 0
    potential_v: Optional[np.ndarray] = None
    electrode_mask: Optional[np.ndarray] = None
    metal_mask: Optional[np.ndarray] = None
    electrode_id: Optional[np.ndarray] = None


__all__ = [
    "PatxtHeader",
    "PatxtMaskGrid",
    "PatxtPotentialGrid",
]
