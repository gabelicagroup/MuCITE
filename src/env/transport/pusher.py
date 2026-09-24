"""Composed 3D particle transport operator."""

from typing import Optional

import numpy as np
import taichi as ti

from ..fields.axisymmetric_grid import UnifiedGrid2D
from .field_gather import FieldGatherMixin
from .initialization import (
    configure_advection_cell_size,
    install_cartesian_field3d,
    install_electrode_mask2d,
    install_electrode_mask3d,
)
from .rk4 import RK4TransportMixin
from .terminal_classifier import TerminalClassifierMixin
from .time_step import TimeStepMixin


@ti.data_oriented
class ParticlePusher(
    FieldGatherMixin,
    TimeStepMixin,
    TerminalClassifierMixin,
    RK4TransportMixin,
):
    """3D gather, timestep, terminal, and RK4 strategies over shared fields."""

    def __init__(
        self,
        grid: UnifiedGrid2D,
        *,
        pic_grid: Optional[UnifiedGrid2D] = None,
        drag_frequency_hz: float = 0.0,
        advection_cfl_fraction: float = 0.5,
        collision_rate_model_code: int = 0,
        gas_molecule_mass_kg: float = 1.0,
        electrode_mask: Optional[object] = None,
        electrode_mask3d: Optional[object] = None,
        cartesian_field3d: Optional[object] = None,
    ) -> None:
        self.grid = grid
        self.static_grid = grid
        self.pic_grid = pic_grid if pic_grid is not None else grid
        self.drag_frequency_hz = float(drag_frequency_hz)
        self.advection_cfl_fraction = float(advection_cfl_fraction)
        self.collision_rate_model_code = int(collision_rate_model_code)
        self.gas_molecule_mass_kg = float(gas_molecule_mass_kg)
        if self.collision_rate_model_code not in {0, 1}:
            raise ValueError("collision_rate_model_code must be 0 or 1.")
        if (
            not np.isfinite(self.gas_molecule_mass_kg)
            or self.gas_molecule_mass_kg <= 0.0
        ):
            raise ValueError("gas_molecule_mass_kg must be finite and positive.")
        if (
            not np.isfinite(self.advection_cfl_fraction)
            or not 0.0 < self.advection_cfl_fraction <= 1.0
        ):
            raise ValueError("advection_cfl_fraction must satisfy 0 < C <= 1.")
        install_cartesian_field3d(self, cartesian_field3d)
        install_electrode_mask3d(self, electrode_mask3d)
        install_electrode_mask2d(self, electrode_mask)
        configure_advection_cell_size(self)
