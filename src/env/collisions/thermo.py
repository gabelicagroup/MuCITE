"""IonSPA fragmentation and pseudo-atom thermodynamic helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.constants import Boltzmann

from ...config import ATOMIC_MASS_CONSTANT
from ..gas.layered import GasState
from .constants import PSEUDOATOM_MASS_CEILING_DA, PSEUDOATOM_MASS_FLOOR_DA


class IonSpaThermoMixin:
    """Thermodynamic helpers shared by scalar and vector collision paths."""

    @staticmethod
    def _sanitize_pseudoatom_mass_da(pseudoatom_mass_da: Any) -> Any:
        values = np.asarray(pseudoatom_mass_da, dtype=np.float64)
        sanitized = np.nan_to_num(
            values,
            nan=PSEUDOATOM_MASS_FLOOR_DA,
            posinf=PSEUDOATOM_MASS_CEILING_DA,
            neginf=PSEUDOATOM_MASS_FLOOR_DA,
        )
        sanitized = np.clip(
            sanitized,
            PSEUDOATOM_MASS_FLOOR_DA,
            PSEUDOATOM_MASS_CEILING_DA,
        )
        if sanitized.ndim == 0:
            return float(sanitized)
        return sanitized

    def _fracloss(
        self,
        dt_s: float,
        temperature_k: float,
        delta_h_kj_per_mol: float,
        delta_s_j_per_mol_k: float,
    ) -> float:
        if self.ionspa is not None:
            return float(
                self.ionspa.fracloss(
                    dt_s, temperature_k, delta_h_kj_per_mol,
                    delta_s_j_per_mol_k,
                )
            )
        delta_h_j_per_mol = delta_h_kj_per_mol * 1000.0
        exponent = (
            delta_s_j_per_mol_k / 8.31446261815324
            - delta_h_j_per_mol / (8.31446261815324 * temperature_k)
        )
        return float(
            (Boltzmann / 6.62607015e-34)
            * dt_s
            * temperature_k
            * np.exp(exponent)
        )

    def _fracloss_many(
        self,
        dt_s: float,
        temperature_k: np.ndarray,
        delta_h_kj_per_mol: float,
        delta_s_j_per_mol_k: float,
    ) -> np.ndarray:
        temperatures = np.asarray(temperature_k, dtype=np.float64)
        if self.ionspa is not None:
            return np.asarray(
                self.ionspa.fracloss(
                    dt_s, temperatures, delta_h_kj_per_mol,
                    delta_s_j_per_mol_k,
                ),
                dtype=np.float64,
            )
        delta_h_j_per_mol = delta_h_kj_per_mol * 1000.0
        exponent = (
            delta_s_j_per_mol_k / 8.31446261815324
            - delta_h_j_per_mol / (8.31446261815324 * temperatures)
        )
        return np.asarray(
            (Boltzmann / 6.62607015e-34)
            * dt_s
            * temperatures
            * np.exp(exponent),
            dtype=np.float64,
        )

    def _pseudoatom_mass_kg(
        self,
        gas_state: GasState,
        gas_name: str,
        ion_temperature_k: float,
        relative_speed_m_per_s: float,
    ) -> float:
        if self.ionspa is None:
            return 6.0 * ATOMIC_MASS_CONSTANT
        gas_key = gas_name.lower()
        cell = self._cell_cache.get(gas_key)
        if cell is None:
            cell = self.ionspa.cellclass(
                gas=gas_key,
                T=gas_state.temperature_k,
                pressure=gas_state.pressure_pa * 1.0e-5,
                L=1.0,
            )
            self._cell_cache[gas_key] = cell
        mass_da = cell.pseudoatom_mass(
            ion_temperature_k,
            relative_speed_m_per_s,
        )
        return self._sanitize_pseudoatom_mass_da(mass_da) * ATOMIC_MASS_CONSTANT

    def _pseudoatom_mass_kg_many(
        self,
        gas_name: str,
        ion_temperature_k: np.ndarray,
        relative_speed_m_per_s: np.ndarray,
    ) -> np.ndarray:
        temperatures = np.asarray(ion_temperature_k, dtype=np.float64)
        relative_speed = np.asarray(relative_speed_m_per_s, dtype=np.float64)
        if self.ionspa is None:
            return np.full(
                temperatures.shape,
                6.0 * ATOMIC_MASS_CONSTANT,
                dtype=np.float64,
            )
        gas_key = gas_name.lower()
        cell = self._cell_cache.get(gas_key)
        if cell is None:
            cell = self.ionspa.cellclass(
                gas=gas_key,
                T=300.0,
                pressure=1.0e-3,
                L=1.0,
            )
            self._cell_cache[gas_key] = cell
        mass_da = cell.pseudoatom_mass(temperatures, relative_speed)
        mass_da = self._sanitize_pseudoatom_mass_da(mass_da)
        return np.asarray(mass_da, dtype=np.float64) * ATOMIC_MASS_CONSTANT


__all__ = ["IonSpaThermoMixin"]
