"""Independent paper-driven IICT collision backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ....agents.ion import IonState
from ....config import CollisionOutcome, IonTemplate, PROJECT_ROOT, SimulationConfig
from ...gas.layered import GasState
from .parameters import ResolvedIictParameters, resolve_iict_parameters
from .physics import apply_iict_collision_core


_DOI = "10.1016/j.ijms.2024.117290"


def _finite_positive(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values.")
    return array


def _gas_molecule_mass_kg(
    number_density_m3: np.ndarray,
    mass_density_kg_per_m3: np.ndarray,
) -> np.ndarray:
    number_density = _finite_positive(number_density_m3, "gas_number_density_m3")
    mass_density = _finite_positive(
        mass_density_kg_per_m3,
        "gas_mass_density_kg_per_m3",
    )
    if number_density.shape != mass_density.shape:
        raise ValueError("Gas number-density and mass-density shapes must match.")
    mass = mass_density / number_density
    return _finite_positive(mass, "gas_molecule_mass_kg")


def _validated_dt(dt_s: float) -> float:
    value = float(dt_s)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("dt_s must be finite and non-negative.")
    return value


class IictLiteBackend:
    """IICT implementation using only user parameters and Prell (2024)."""

    def __init__(
        self,
        config: SimulationConfig,
        template: IonTemplate,
        *,
        project_root: Path | None = None,
    ) -> None:
        self.template = template
        self.project_root = Path(project_root or PROJECT_ROOT)
        self.parameters = resolve_iict_parameters(
            config,
            template,
            project_root=self.project_root,
        )

    def energy_from_temperature(
        self,
        temperature_k: float | np.ndarray,
    ) -> float | np.ndarray:
        return self.parameters.heat_capacity.energy_from_temperature(temperature_k)

    def temperature_from_energy(
        self,
        internal_energy_j: float | np.ndarray,
    ) -> float | np.ndarray:
        return self.parameters.heat_capacity.temperature_from_energy(
            internal_energy_j
        )

    def pseudoatom_mass(
        self,
        temperature_k: float | np.ndarray,
        relative_speed_m_per_s: float | np.ndarray,
        gas_name: str,
    ) -> float | np.ndarray:
        mass = self.parameters.pseudoatom_mass.mass_kg(
            temperature_k,
            relative_speed_m_per_s,
            gas_name,
        )
        values = _finite_positive(mass, "pseudoatom_mass_kg")
        return float(values) if values.ndim == 0 else values

    def fragmentation_rate(
        self,
        temperature_k: float | np.ndarray,
        ion_parameters: IonTemplate,
    ) -> float | np.ndarray:
        del ion_parameters
        return self.parameters.fragmentation.rate_s_inverse(temperature_k)

    def _updated_internal_state(
        self,
        temperatures_k: np.ndarray,
        transferred_energy_j: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        internal_before = np.asarray(
            self.energy_from_temperature(temperatures_k),
            dtype=np.float64,
        )
        internal_after = internal_before + transferred_energy_j
        if not np.all(np.isfinite(internal_after)) or np.any(internal_after <= 0.0):
            minimum = float(np.min(internal_after))
            raise ValueError(
                "IICT collision produced non-positive internal energy "
                f"({minimum:.6g} J); parameters/state are outside model bounds."
            )
        temperature_after = np.asarray(
            self.temperature_from_energy(internal_after),
            dtype=np.float64,
        )
        probability = np.asarray(
            self.parameters.fragmentation.probability(
                temperature_after,
                _validated_dt(dt_s),
            ),
            dtype=np.float64,
        )
        return temperature_after, probability

    def _apply_batch(
        self,
        *,
        velocities_m_per_s: np.ndarray,
        masses_kg: np.ndarray,
        internal_temperatures_k: np.ndarray,
        gas_velocities_m_per_s: np.ndarray,
        gas_temperatures_k: np.ndarray,
        gas_number_density_m3: np.ndarray,
        gas_mass_density_kg_per_m3: np.ndarray,
        gas_name: str,
        dt_s: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        temperatures = _finite_positive(
            internal_temperatures_k,
            "internal_temperatures_k",
        )
        velocities = np.asarray(velocities_m_per_s, dtype=np.float64)
        gas_velocities = np.asarray(gas_velocities_m_per_s, dtype=np.float64)
        relative_speed = np.linalg.norm(velocities - gas_velocities, axis=1)
        pseudoatom_mass = np.asarray(
            self.pseudoatom_mass(temperatures, relative_speed, gas_name),
            dtype=np.float64,
        )
        gas_mass = _gas_molecule_mass_kg(
            gas_number_density_m3,
            gas_mass_density_kg_per_m3,
        )
        core = apply_iict_collision_core(
            ion_velocity_m_per_s=velocities,
            ion_mass_kg=np.asarray(masses_kg, dtype=np.float64),
            ion_temperature_k=temperatures,
            gas_bulk_velocity_m_per_s=gas_velocities,
            gas_temperature_k=np.asarray(gas_temperatures_k, dtype=np.float64),
            gas_mass_kg=gas_mass,
            pseudoatom_mass_kg=pseudoatom_mass,
            rng=rng,
        )
        temperature_after, probability = self._updated_internal_state(
            temperatures,
            core.transferred_internal_energy_j,
            dt_s,
        )
        return (
            core.ion_velocity_after_m_per_s,
            temperature_after,
            probability,
            core.transferred_internal_energy_j,
        )

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
        velocity, temperature, probability, transferred = self._apply_batch(
            velocities_m_per_s=np.asarray(ion.velocity_m_per_s)[None, :],
            masses_kg=np.asarray([ion.mass_kg]),
            internal_temperatures_k=np.asarray([ion.internal_temperature_k]),
            gas_velocities_m_per_s=np.asarray(gas_state.velocity_m_per_s)[None, :],
            gas_temperatures_k=np.asarray([gas_state.temperature_k]),
            gas_number_density_m3=np.asarray([gas_state.number_density_m3]),
            gas_mass_density_kg_per_m3=np.asarray(
                [gas_state.mass_density_kg_per_m3]
            ),
            gas_name=gas_name,
            dt_s=dt_s,
            rng=rng,
        )
        del template
        return CollisionOutcome(
            velocity_m_per_s=velocity[0],
            internal_temperature_k=float(temperature[0]),
            transferred_internal_energy_j=float(transferred[0]),
            fragmentation_probability=float(probability[0]),
        )

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
        velocities = np.asarray(velocities_m_per_s, dtype=np.float64)
        if velocities.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return np.zeros((0, 3), dtype=np.float64), empty, empty
        del template
        result = self._apply_batch(
            velocities_m_per_s=velocities,
            masses_kg=masses_kg,
            internal_temperatures_k=internal_temperatures_k,
            gas_velocities_m_per_s=gas_velocities_m_per_s,
            gas_temperatures_k=gas_temperatures_k,
            gas_number_density_m3=gas_number_density_m3,
            gas_mass_density_kg_per_m3=gas_mass_density_kg_per_m3,
            gas_name=gas_name,
            dt_s=dt_s,
            rng=rng,
        )
        return result[0], result[1], result[2]

    @property
    def runtime_info(self) -> dict[str, Any]:
        parameters: ResolvedIictParameters = self.parameters
        return {
            "collision_physics_backend": "iict-lite",
            "requested_backend": "iict-lite",
            "effective_backend": "iict-lite",
            "loaded": True,
            "implementation": f"{type(self).__module__}.{type(self).__name__}",
            "version": 1,
            "origin": str(Path(__file__).resolve()),
            "ionspa_loaded": False,
            "ionspa_accessed": False,
            "theory_doi": _DOI,
            "equations": ["38-43", "47", "52", "54-59", "60-61"],
            "parameter_config_path": parameters.parameter_config_path,
            "parameter_config_sha256": parameters.parameter_config_sha256,
            "effective_parameters": parameters.effective_parameters,
            "parameter_sources": parameters.parameter_sources,
            "equivalence_claim": (
                "Independent paper-driven approximation; not numerically "
                "equivalent to IonSPA."
            ),
        }


__all__ = ["IictLiteBackend"]
