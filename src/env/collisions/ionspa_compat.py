"""Optional IonSPA compatibility adapter implementing the common protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...agents.ion import IonState
from ...config import CollisionOutcome, IonTemplate, PROJECT_ROOT, SimulationConfig
from ..gas.layered import GasState
from .adapter import IonSpaCollisionAdapter


class IonSpaBackendAdapter:
    """Adapt a licensed local or retained legacy IonSPA provider."""

    def __init__(
        self,
        config: SimulationConfig,
        template: IonTemplate,
        *,
        project_root: Path | None = None,
    ) -> None:
        self.template = template
        self.project_root = Path(project_root or PROJECT_ROOT)
        self._delegate = IonSpaCollisionAdapter(
            project_root=self.project_root,
            backend=config.ionspa_backend,
        )

    @property
    def ionspa(self) -> Any:
        return self._delegate.ionspa

    @property
    def import_error(self) -> Exception | None:
        return self._delegate.import_error

    def energy_from_temperature(
        self,
        temperature_k: float | np.ndarray,
    ) -> float | np.ndarray:
        return self._delegate._build_ion_model(self.template).UfromT(temperature_k)

    def temperature_from_energy(
        self,
        internal_energy_j: float | np.ndarray,
    ) -> float | np.ndarray:
        return self._delegate._build_ion_model(self.template).TfromU(
            internal_energy_j
        )

    def pseudoatom_mass(
        self,
        temperature_k: float | np.ndarray,
        relative_speed_m_per_s: float | np.ndarray,
        gas_name: str,
    ) -> float | np.ndarray:
        temperatures, speeds = np.broadcast_arrays(
            np.asarray(temperature_k, dtype=np.float64),
            np.asarray(relative_speed_m_per_s, dtype=np.float64),
        )
        result = self._delegate._pseudoatom_mass_kg_many(
            gas_name,
            temperatures,
            speeds,
        )
        return float(result) if np.asarray(result).ndim == 0 else result

    def fragmentation_rate(
        self,
        temperature_k: float | np.ndarray,
        ion_parameters: IonTemplate,
    ) -> float | np.ndarray:
        temperatures = np.asarray(temperature_k, dtype=np.float64)
        rate = self._delegate._fracloss_many(
            1.0,
            temperatures,
            ion_parameters.delta_h_kj_per_mol,
            ion_parameters.delta_s_j_per_mol_k,
        )
        return float(rate) if np.asarray(rate).ndim == 0 else rate

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
        return self._delegate.apply_collision(
            ion,
            template,
            gas_state,
            gas_name=gas_name,
            dt_s=dt_s,
            rng=rng,
        )

    def apply_collision_batch(
        self,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._delegate.apply_collision_batch(**kwargs)

    @property
    def runtime_info(self) -> dict[str, Any]:
        info = dict(self._delegate.runtime_info)
        info.update(
            {
                "collision_physics_backend": "ionspa",
                "ionspa_loaded": self.ionspa is not None,
                "license_responsibility": (
                    "The user must hold a valid license for the selected "
                    "IonSPA provider."
                ),
            }
        )
        return info


__all__ = ["IonSpaBackendAdapter"]
