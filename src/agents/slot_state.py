"""Particle slot metadata storage for the coupled runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import elementary_charge

from ..config import (
    PARTICLE_ACTIVE,
    SLOT_ACTIVE,
    SLOT_FREE,
    IonTemplate,
)


@dataclass
class ParticleRuntimeState:
    """Python-side metadata arrays that mirror the Taichi particle slot pool."""

    collision_counts: np.ndarray
    status_codes: np.ndarray
    tof_s: np.ndarray
    electrode_id: np.ndarray
    surface_distance_m: np.ndarray
    charge_c: np.ndarray
    mass_kg: np.ndarray
    weight: np.ndarray
    parent_weight: np.ndarray
    track_id: np.ndarray
    birth_time_s: np.ndarray
    phase_codes: np.ndarray
    ever_used: np.ndarray
    next_track_id: int = 0

    @classmethod
    def allocate(
        cls,
        count: int,
        template: IonTemplate,
        *,
        default_weight: float,
        zero_weight: bool = False,
    ) -> "ParticleRuntimeState":
        """Allocate a free slot pool with runtime metadata arrays."""

        count = int(count)
        weight = np.zeros(count, dtype=np.float64) if zero_weight else np.full(count, default_weight, dtype=np.float64)
        parent_weight = weight.copy()
        return cls(
            collision_counts=np.zeros(count, dtype=np.int64),
            status_codes=np.full(count, PARTICLE_ACTIVE, dtype=np.int16),
            tof_s=np.full(count, np.nan, dtype=np.float64),
            electrode_id=np.full(count, -1, dtype=np.int32),
            surface_distance_m=np.full(count, np.nan, dtype=np.float64),
            charge_c=np.full(count, template.charge_state * elementary_charge, dtype=np.float64),
            mass_kg=np.full(count, template.mass_kg, dtype=np.float64),
            weight=weight,
            parent_weight=parent_weight,
            track_id=np.full(count, -1, dtype=np.int64),
            birth_time_s=np.full(count, np.nan, dtype=np.float64),
            phase_codes=np.full(count, SLOT_FREE, dtype=np.int16),
            ever_used=np.zeros(count, dtype=bool),
            next_track_id=0,
        )

    @classmethod
    def packet(cls, count: int, template: IonTemplate, *, macro_weight: float) -> "ParticleRuntimeState":
        """Allocate metadata for packet mode, where every slot starts active."""

        state = cls.allocate(count, template, default_weight=macro_weight)
        state.track_id = np.arange(int(count), dtype=np.int64)
        state.birth_time_s = np.zeros(int(count), dtype=np.float64)
        state.phase_codes = np.full(int(count), SLOT_ACTIVE, dtype=np.int16)
        state.ever_used = np.ones(int(count), dtype=bool)
        state.next_track_id = int(count)
        return state

    @classmethod
    def continuous(cls, count: int, template: IonTemplate) -> "ParticleRuntimeState":
        """Allocate metadata for continuous-current mode with all slots initially free."""

        return cls.allocate(count, template, default_weight=0.0, zero_weight=True)

    def bind_to(self, owner: Any) -> None:
        """Expose arrays through the existing runtime attribute names.

        The aliases keep the refactor behavior-preserving while moving array
        allocation and ownership into a dedicated state object.
        """

        owner.particle_state = self
        owner._particle_collision_counts = self.collision_counts
        owner._particle_status_codes = self.status_codes
        owner._particle_tof_s = self.tof_s
        owner._particle_electrode_id = self.electrode_id
        owner._particle_surface_distance_m = self.surface_distance_m
        owner._particle_charge_c = self.charge_c
        owner._particle_mass_kg = self.mass_kg
        owner._particle_weight = self.weight
        owner._particle_parent_weight = self.parent_weight
        owner._particle_track_id = self.track_id
        owner._particle_birth_time_s = self.birth_time_s
        owner._particle_phase_codes = self.phase_codes
        owner._particle_ever_used = self.ever_used
        owner._next_track_id = int(self.next_track_id)
