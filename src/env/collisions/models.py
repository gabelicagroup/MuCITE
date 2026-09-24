"""IonSPA adapter model types."""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann

from ...config import IonTemplate


class IonSpaLoadError(RuntimeError):
    """Raised when the requested canonical IonSPA backend cannot be loaded."""


class ApproximateIonModel:
    """Deliberately selected approximation of IonSPA thermodynamics."""

    def __init__(self, template: IonTemplate) -> None:
        self.dof = max(3 * template.num_atoms - 6, 1)
        self.dH = template.delta_h_kj_per_mol
        self.dS = template.delta_s_j_per_mol_k

    def UfromT(self, temperature_k: float) -> float:
        return self.dof * Boltzmann * temperature_k

    def TfromU(self, internal_energy_j: float) -> float:
        return np.maximum(internal_energy_j, 0.0) / (self.dof * Boltzmann)


FallbackIonModel = ApproximateIonModel

__all__ = [
    "ApproximateIonModel",
    "FallbackIonModel",
    "IonSpaLoadError",
]
