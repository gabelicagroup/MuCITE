"""User-supplied pseudo-atom mass models for independent IICT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ....config import ATOMIC_MASS_CONSTANT


Numeric = float | np.ndarray


class PseudoAtomMassModel(Protocol):
    """Return an effective pseudo-atom mass without bundled lookup data."""

    def mass_kg(
        self,
        temperature_k: Numeric,
        relative_speed_m_per_s: Numeric,
        gas_name: str,
    ) -> Numeric:
        """Return mass in kilograms for broadcast-compatible inputs."""

    @property
    def metadata(self) -> dict[str, object]:
        """Return JSON-safe effective parameters."""


def _validate_bounds(min_mass_da: float, max_mass_da: float) -> None:
    values = np.asarray([min_mass_da, max_mass_da], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Pseudo-atom mass bounds must be finite and positive.")
    if float(min_mass_da) >= float(max_mass_da):
        raise ValueError("Pseudo-atom min_mass_da must be below max_mass_da.")


def _validate_gas_name(expected: str | None, actual: str) -> None:
    if expected is None:
        return
    if str(actual).strip().lower() != expected:
        raise ValueError(
            f"Pseudo-atom table is for gas {expected!r}, not {actual!r}."
        )


def _validated_inputs(
    temperature_k: Numeric,
    relative_speed_m_per_s: Numeric,
) -> tuple[np.ndarray, np.ndarray]:
    temperatures, speeds = np.broadcast_arrays(
        np.asarray(temperature_k, dtype=np.float64),
        np.asarray(relative_speed_m_per_s, dtype=np.float64),
    )
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures < 0.0):
        raise ValueError("Pseudo-atom temperatures must be finite and non-negative.")
    if not np.all(np.isfinite(speeds)) or np.any(speeds < 0.0):
        raise ValueError("Relative speeds must be finite and non-negative.")
    return temperatures, speeds


def _scalar_or_array(values: np.ndarray) -> Numeric:
    return float(values) if values.ndim == 0 else values


@dataclass(frozen=True)
class ConstantPseudoAtomMassModel:
    """One explicit pseudo-atom mass, never inferred from gas identity."""

    mass_da: float
    min_mass_da: float
    max_mass_da: float
    gas_name: str | None = None

    def __post_init__(self) -> None:
        _validate_bounds(self.min_mass_da, self.max_mass_da)
        value = float(self.mass_da)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("pseudoatom mass_da must be finite and positive.")
        if value < self.min_mass_da or value > self.max_mass_da:
            raise ValueError("pseudoatom mass_da is outside configured bounds.")
        if self.gas_name is not None:
            object.__setattr__(self, "gas_name", self.gas_name.strip().lower())

    def mass_kg(
        self,
        temperature_k: Numeric,
        relative_speed_m_per_s: Numeric,
        gas_name: str,
    ) -> Numeric:
        _validate_gas_name(self.gas_name, gas_name)
        temperatures, _ = _validated_inputs(
            temperature_k,
            relative_speed_m_per_s,
        )
        result = np.full(temperatures.shape, self.mass_da * ATOMIC_MASS_CONSTANT)
        return _scalar_or_array(result)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "type": "constant",
            "mass_da": float(self.mass_da),
            "min_mass_da": float(self.min_mass_da),
            "max_mass_da": float(self.max_mass_da),
            "gas_name": self.gas_name,
        }


class TabulatedPseudoAtomMassModel:
    """Boundary-clamped bilinear interpolation over a complete T-speed grid."""

    def __init__(
        self,
        temperature_k: np.ndarray,
        relative_speed_m_per_s: np.ndarray,
        mass_da: np.ndarray,
        *,
        min_mass_da: float,
        max_mass_da: float,
        source_path: str,
        gas_name: str | None = None,
    ) -> None:
        _validate_bounds(min_mass_da, max_mass_da)
        temperatures, speeds, grid = _rectangular_mass_grid(
            temperature_k,
            relative_speed_m_per_s,
            mass_da,
        )
        if np.any(grid < min_mass_da) or np.any(grid > max_mass_da):
            raise ValueError("Pseudo-atom table contains mass outside configured bounds.")
        self._temperature_k = temperatures
        self._relative_speed_m_per_s = speeds
        self._mass_da = grid
        self._interpolator = RegularGridInterpolator(
            (temperatures, speeds),
            grid,
            bounds_error=True,
        )
        self._min_mass_da = float(min_mass_da)
        self._max_mass_da = float(max_mass_da)
        self._source_path = str(source_path)
        self._gas_name = None if gas_name is None else gas_name.strip().lower()

    def mass_kg(
        self,
        temperature_k: Numeric,
        relative_speed_m_per_s: Numeric,
        gas_name: str,
    ) -> Numeric:
        _validate_gas_name(self._gas_name, gas_name)
        temperatures, speeds = _validated_inputs(
            temperature_k,
            relative_speed_m_per_s,
        )
        clipped_t = np.clip(
            temperatures,
            self._temperature_k[0],
            self._temperature_k[-1],
        )
        clipped_v = np.clip(
            speeds,
            self._relative_speed_m_per_s[0],
            self._relative_speed_m_per_s[-1],
        )
        points = np.column_stack((clipped_t.reshape(-1), clipped_v.reshape(-1)))
        mass_da = self._interpolator(points).reshape(temperatures.shape)
        mass_da = np.clip(mass_da, self._min_mass_da, self._max_mass_da)
        result = mass_da * ATOMIC_MASS_CONSTANT
        return _scalar_or_array(result)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "type": "tabulated",
            "source_path": self._source_path,
            "gas_name": self._gas_name,
            "temperature_nodes": int(self._temperature_k.size),
            "relative_speed_nodes": int(self._relative_speed_m_per_s.size),
            "min_mass_da": self._min_mass_da,
            "max_mass_da": self._max_mass_da,
            "out_of_range_policy": "clamp_to_table_boundary",
        }


def _rectangular_mass_grid(
    temperature_k: np.ndarray,
    relative_speed_m_per_s: np.ndarray,
    mass_da: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperatures = np.asarray(temperature_k, dtype=np.float64)
    speeds = np.asarray(relative_speed_m_per_s, dtype=np.float64)
    masses = np.asarray(mass_da, dtype=np.float64)
    if not (temperatures.ndim == speeds.ndim == masses.ndim == 1):
        raise ValueError("Pseudo-atom CSV columns must be one-dimensional.")
    if not (temperatures.size == speeds.size == masses.size):
        raise ValueError("Pseudo-atom CSV columns must have equal row counts.")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures < 0.0):
        raise ValueError("Pseudo-atom table temperatures must be finite and non-negative.")
    if not np.all(np.isfinite(speeds)) or np.any(speeds < 0.0):
        raise ValueError("Pseudo-atom table speeds must be finite and non-negative.")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("Pseudo-atom table masses must be finite and positive.")
    return _assemble_grid(temperatures, speeds, masses)


def _assemble_grid(
    temperatures: np.ndarray,
    speeds: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_nodes = np.unique(temperatures)
    v_nodes = np.unique(speeds)
    if t_nodes.size < 2 or v_nodes.size < 2:
        raise ValueError("Pseudo-atom table requires at least a 2 x 2 grid.")
    if temperatures.size != t_nodes.size * v_nodes.size:
        raise ValueError("Pseudo-atom table must contain one complete rectangular grid.")
    grid = np.full((t_nodes.size, v_nodes.size), np.nan, dtype=np.float64)
    for temperature, speed, mass in zip(temperatures, speeds, masses):
        i = int(np.searchsorted(t_nodes, temperature))
        j = int(np.searchsorted(v_nodes, speed))
        if np.isfinite(grid[i, j]):
            raise ValueError("Pseudo-atom table contains duplicate T-speed rows.")
        grid[i, j] = mass
    if not np.all(np.isfinite(grid)):
        raise ValueError("Pseudo-atom table is missing one or more grid points.")
    return t_nodes, v_nodes, grid


__all__ = [
    "ConstantPseudoAtomMassModel",
    "PseudoAtomMassModel",
    "TabulatedPseudoAtomMassModel",
]
