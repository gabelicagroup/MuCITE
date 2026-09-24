"""Backend-neutral collision-physics contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...agents.ion import IonState
from ...config import CollisionOutcome, IonTemplate
from ..gas.layered import GasState


@runtime_checkable
class CollisionPhysicsBackend(Protocol):
    """Thermodynamic and single-event collision API used by the runtime."""

    def energy_from_temperature(
        self,
        temperature_k: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return ion internal energy in joules per ion."""

    def temperature_from_energy(
        self,
        internal_energy_j: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return ion internal temperature in kelvin."""

    def pseudoatom_mass(
        self,
        temperature_k: float | np.ndarray,
        relative_speed_m_per_s: float | np.ndarray,
        gas_name: str,
    ) -> float | np.ndarray:
        """Return pseudo-atom mass in kilograms."""

    def apply_collision(
        self,
        ion: IonState,
        template: IonTemplate,
        gas_state: GasState,
        *,
        gas_name: str,
        dt_s: float,
        rng: np.random.Generator,
    ) -> CollisionOutcome:
        """Apply one already-triggered collision event."""

    def apply_collision_batch(
        self,
        *,
        velocities_m_per_s: np.ndarray,
        masses_kg: np.ndarray,
        internal_temperatures_k: np.ndarray,
        gas_velocities_m_per_s: np.ndarray,
        gas_temperatures_k: np.ndarray,
        gas_number_density_m3: np.ndarray,
        gas_mass_density_kg_per_m3: np.ndarray,
        template: IonTemplate,
        gas_name: str,
        dt_s: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the scalar physical core to a vectorized event batch."""

    def fragmentation_rate(
        self,
        temperature_k: float | np.ndarray,
        ion_parameters: IonTemplate,
    ) -> float | np.ndarray:
        """Return the unimolecular fragmentation rate in inverse seconds."""

    @property
    def runtime_info(self) -> dict[str, Any]:
        """Return JSON-safe provenance and effective-parameter metadata."""


__all__ = ["CollisionPhysicsBackend"]
