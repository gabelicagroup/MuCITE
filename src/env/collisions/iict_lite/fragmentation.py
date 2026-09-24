"""Eyring kinetics from Prell 2024 Eq. 60-61.

Reference DOI: 10.1016/j.ijms.2024.117290.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.constants import Boltzmann, Planck, gas_constant


Numeric = float | np.ndarray


class FragmentationModel(Protocol):
    """Time-integrated fragmentation model independent of collision impulse."""

    def rate_s_inverse(self, temperature_k: Numeric) -> Numeric:
        """Return ``k(T)`` in inverse seconds."""

    def probability(self, temperature_k: Numeric, dt_s: float) -> Numeric:
        """Return ``1 - exp(-k(T) * dt)``."""

    @property
    def metadata(self) -> dict[str, object]:
        """Return JSON-safe effective parameters."""


def _temperatures(temperature_k: Numeric) -> np.ndarray:
    values = np.asarray(temperature_k, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Fragmentation temperature_k must be finite and positive.")
    return values


def _dt(dt_s: float) -> float:
    value = float(dt_s)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Fragmentation dt_s must be finite and non-negative.")
    return value


def _scalar_or_array(reference: Numeric, values: np.ndarray) -> Numeric:
    return float(values) if np.asarray(reference).ndim == 0 else values


@dataclass(frozen=True)
class NoFragmentationModel:
    """Explicitly disabled fragmentation."""

    def rate_s_inverse(self, temperature_k: Numeric) -> Numeric:
        values = _temperatures(temperature_k)
        result = np.zeros(values.shape, dtype=np.float64)
        return _scalar_or_array(temperature_k, result)

    def probability(self, temperature_k: Numeric, dt_s: float) -> Numeric:
        _dt(dt_s)
        return self.rate_s_inverse(temperature_k)

    @property
    def metadata(self) -> dict[str, object]:
        return {"type": "none"}


@dataclass(frozen=True)
class EyringFragmentationModel:
    """Eyring rate from Prell (2024), Eq. 60-61."""

    delta_h_kj_per_mol: float
    delta_s_j_per_mol_k: float

    def __post_init__(self) -> None:
        delta_h = float(self.delta_h_kj_per_mol)
        delta_s = float(self.delta_s_j_per_mol_k)
        if not np.isfinite(delta_h) or delta_h <= 0.0:
            raise ValueError("Eyring delta_h_kj_per_mol must be positive.")
        if not np.isfinite(delta_s):
            raise ValueError("Eyring delta_s_j_per_mol_k must be finite.")

    def rate_s_inverse(self, temperature_k: Numeric) -> Numeric:
        temperatures = _temperatures(temperature_k)
        delta_h_j_per_mol = float(self.delta_h_kj_per_mol) * 1000.0
        log_rate = (
            np.log(Boltzmann / Planck)
            + np.log(temperatures)
            + float(self.delta_s_j_per_mol_k) / gas_constant
            - delta_h_j_per_mol / (gas_constant * temperatures)
        )
        rate = np.exp(np.minimum(log_rate, np.log(np.finfo(np.float64).max)))
        return _scalar_or_array(temperature_k, rate)

    def probability(self, temperature_k: Numeric, dt_s: float) -> Numeric:
        interval = _dt(dt_s)
        rate = np.asarray(self.rate_s_inverse(temperature_k), dtype=np.float64)
        with np.errstate(over="ignore", invalid="raise"):
            probability = -np.expm1(-rate * interval)
        probability = np.clip(probability, 0.0, 1.0)
        return _scalar_or_array(temperature_k, probability)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "type": "eyring",
            "delta_h_kj_per_mol": float(self.delta_h_kj_per_mol),
            "delta_s_j_per_mol_k": float(self.delta_s_j_per_mol_k),
            "probability": "1 - exp(-k(T) * dt)",
            "equations": "Prell 2024 Eq. 60-61",
        }


__all__ = [
    "EyringFragmentationModel",
    "FragmentationModel",
    "NoFragmentationModel",
]
