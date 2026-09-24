"""Particle entity state independent of transport and presentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IonState:
    """State vector for one simulated ion."""

    position_m: np.ndarray
    velocity_m_per_s: np.ndarray
    mass_kg: float
    charge_c: float
    collision_cross_section_m2: float
    internal_temperature_k: float
    species_name: str = "ion"
    alive: bool = True


__all__ = ["IonState"]
