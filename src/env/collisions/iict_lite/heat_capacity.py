"""Independent heat-capacity models for the paper-driven IICT backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.constants import Boltzmann


Numeric = float | np.ndarray


class HeatCapacityModel(Protocol):
    """Invertible per-ion internal-energy model."""

    def energy_from_temperature(self, temperature_k: Numeric) -> Numeric:
        """Return ``U(T)`` in joules per ion."""

    def temperature_from_energy(self, internal_energy_j: Numeric) -> Numeric:
        """Return the inverse ``T(U)`` in kelvin."""

    @property
    def metadata(self) -> dict[str, object]:
        """Return JSON-safe effective parameters."""


def _validated_values(
    values: Numeric,
    *,
    name: str,
    nonnegative: bool = True,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def _scalar_or_array(reference: Numeric, values: np.ndarray) -> Numeric:
    return float(values) if np.asarray(reference).ndim == 0 else values


@dataclass(frozen=True)
class ClassicalHeatCapacityModel:
    """Classical vibrational model ``U = dof * kB * T``."""

    num_atoms: int

    def __post_init__(self) -> None:
        if int(self.num_atoms) <= 2:
            raise ValueError("classical num_atoms must be greater than 2.")

    @property
    def degrees_of_freedom(self) -> int:
        return 3 * int(self.num_atoms) - 6

    def energy_from_temperature(self, temperature_k: Numeric) -> Numeric:
        values = _validated_values(temperature_k, name="temperature_k")
        result = self.degrees_of_freedom * Boltzmann * values
        return _scalar_or_array(temperature_k, result)

    def temperature_from_energy(self, internal_energy_j: Numeric) -> Numeric:
        values = _validated_values(internal_energy_j, name="internal_energy_j")
        result = values / (self.degrees_of_freedom * Boltzmann)
        return _scalar_or_array(internal_energy_j, result)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "type": "classical",
            "num_atoms": int(self.num_atoms),
            "degrees_of_freedom": self.degrees_of_freedom,
            "formula": "U = (3*num_atoms - 6) * kB * T",
        }


@dataclass(frozen=True)
class ConstantCvHeatCapacityModel:
    """Constant per-ion heat capacity with the zero ``U(0)=0``."""

    cv_j_per_k_per_ion: float

    def __post_init__(self) -> None:
        value = float(self.cv_j_per_k_per_ion)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("cv_j_per_k_per_ion must be finite and positive.")

    def energy_from_temperature(self, temperature_k: Numeric) -> Numeric:
        values = _validated_values(temperature_k, name="temperature_k")
        result = float(self.cv_j_per_k_per_ion) * values
        return _scalar_or_array(temperature_k, result)

    def temperature_from_energy(self, internal_energy_j: Numeric) -> Numeric:
        values = _validated_values(internal_energy_j, name="internal_energy_j")
        result = values / float(self.cv_j_per_k_per_ion)
        return _scalar_or_array(internal_energy_j, result)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "type": "constant_cv",
            "cv_j_per_k_per_ion": float(self.cv_j_per_k_per_ion),
        }


def _linear_interp_extrap(
    values: np.ndarray,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.interp(flat, x_nodes, y_nodes)
    low = flat < x_nodes[0]
    high = flat > x_nodes[-1]
    slope_low = (y_nodes[1] - y_nodes[0]) / (x_nodes[1] - x_nodes[0])
    slope_high = (y_nodes[-1] - y_nodes[-2]) / (
        x_nodes[-1] - x_nodes[-2]
    )
    result[low] = y_nodes[0] + slope_low * (flat[low] - x_nodes[0])
    result[high] = y_nodes[-1] + slope_high * (
        flat[high] - x_nodes[-1]
    )
    return result.reshape(values.shape)


class TabulatedHeatCapacityModel:
    """Monotone piecewise-linear ``U(T)`` and inverse ``T(U)``."""

    def __init__(
        self,
        temperature_k: np.ndarray,
        internal_energy_j: np.ndarray,
        *,
        source_path: str,
        source_quantity: str,
    ) -> None:
        temperatures = _validated_values(
            temperature_k,
            name="tabulated temperature_k",
        )
        energies = _validated_values(
            internal_energy_j,
            name="tabulated internal_energy_j",
        )
        if temperatures.ndim != 1 or energies.shape != temperatures.shape:
            raise ValueError("Tabulated T and U columns must be matching 1D arrays.")
        if temperatures.size < 2:
            raise ValueError("Tabulated heat capacity requires at least two rows.")
        if np.any(np.diff(temperatures) <= 0.0):
            raise ValueError("Tabulated temperatures must be strictly increasing.")
        if np.any(np.diff(energies) <= 0.0):
            raise ValueError("Tabulated internal energies must be strictly increasing.")
        self._temperature_k = temperatures
        self._internal_energy_j = energies
        self._source_path = str(source_path)
        self._source_quantity = str(source_quantity)

    def energy_from_temperature(self, temperature_k: Numeric) -> Numeric:
        values = _validated_values(temperature_k, name="temperature_k")
        result = _linear_interp_extrap(
            values,
            self._temperature_k,
            self._internal_energy_j,
        )
        if np.any(result < 0.0) or not np.all(np.isfinite(result)):
            raise ValueError("Tabulated U(T) extrapolation produced invalid energy.")
        return _scalar_or_array(temperature_k, result)

    def temperature_from_energy(self, internal_energy_j: Numeric) -> Numeric:
        values = _validated_values(internal_energy_j, name="internal_energy_j")
        result = _linear_interp_extrap(
            values,
            self._internal_energy_j,
            self._temperature_k,
        )
        if np.any(result < 0.0) or not np.all(np.isfinite(result)):
            raise ValueError("Tabulated T(U) extrapolation produced invalid temperature.")
        return _scalar_or_array(internal_energy_j, result)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "type": "tabulated",
            "source_path": self._source_path,
            "source_quantity": self._source_quantity,
            "rows": int(self._temperature_k.size),
            "temperature_min_k": float(self._temperature_k[0]),
            "temperature_max_k": float(self._temperature_k[-1]),
        }


def tabulated_energy_model(
    temperature_k: np.ndarray,
    internal_energy_j: np.ndarray,
    *,
    source_path: str,
) -> TabulatedHeatCapacityModel:
    """Build a table while enforcing the physical zero ``U(0)=0``."""

    temperatures = np.asarray(temperature_k, dtype=np.float64)
    energies = np.asarray(internal_energy_j, dtype=np.float64)
    if temperatures.size and temperatures[0] > 0.0:
        temperatures = np.concatenate(([0.0], temperatures))
        energies = np.concatenate(([0.0], energies))
    elif temperatures.size and energies[0] != 0.0:
        raise ValueError("A tabulated U row at T=0 K must have U=0 J/ion.")
    return TabulatedHeatCapacityModel(
        temperatures,
        energies,
        source_path=source_path,
        source_quantity="U[J/ion]",
    )


def tabulated_cv_model(
    temperature_k: np.ndarray,
    cv_j_per_k_per_ion: np.ndarray,
    *,
    source_path: str,
) -> TabulatedHeatCapacityModel:
    """Integrate a positive ``Cv(T)`` table into monotone ``U(T)`` nodes."""

    temperatures = _validated_values(
        temperature_k,
        name="tabulated temperature_k",
    )
    cv = _validated_values(
        cv_j_per_k_per_ion,
        name="tabulated cv_j_per_k_per_ion",
    )
    if np.any(cv <= 0.0):
        raise ValueError("Tabulated heat capacity values must be positive.")
    if temperatures.size and temperatures[0] > 0.0:
        temperatures = np.concatenate(([0.0], temperatures))
        cv = np.concatenate(([cv[0]], cv))
    increments = 0.5 * (cv[1:] + cv[:-1]) * np.diff(temperatures)
    energies = np.concatenate(([0.0], np.cumsum(increments)))
    return TabulatedHeatCapacityModel(
        temperatures,
        energies,
        source_path=source_path,
        source_quantity="Cv[J/K/ion]",
    )


__all__ = [
    "ClassicalHeatCapacityModel",
    "ConstantCvHeatCapacityModel",
    "HeatCapacityModel",
    "TabulatedHeatCapacityModel",
    "tabulated_cv_model",
    "tabulated_energy_model",
]
