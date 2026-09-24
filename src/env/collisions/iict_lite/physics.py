"""Paper-derived IICT impulse and statistical validation formulas.

Equations refer to J. S. Prell, Int. J. Mass Spectrom. 504 (2024) 117290,
https://doi.org/10.1016/j.ijms.2024.117290.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann
from scipy.special import erf


@dataclass(frozen=True)
class IictCollisionCoreResult:
    """Vectorized state and accounting values for already-triggered events."""

    ion_velocity_after_m_per_s: np.ndarray
    transferred_internal_energy_j: np.ndarray
    pseudoatom_velocity_lab_before_m_per_s: np.ndarray
    pseudoatom_velocity_lab_after_m_per_s: np.ndarray
    gas_velocity_lab_before_m_per_s: np.ndarray
    gas_velocity_lab_after_m_per_s: np.ndarray
    pseudoatom_velocity_ion_before_m_per_s: np.ndarray
    pseudoatom_velocity_ion_after_m_per_s: np.ndarray


def _matrix3(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _vector(values: np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_masses(
    ion_mass_kg: np.ndarray,
    gas_mass_kg: np.ndarray,
    pseudoatom_mass_kg: np.ndarray,
) -> None:
    if np.any(ion_mass_kg <= 0.0) or np.any(gas_mass_kg <= 0.0):
        raise ValueError("Ion and gas masses must be positive.")
    if np.any(pseudoatom_mass_kg <= 0.0):
        raise ValueError("Pseudo-atom masses must be positive.")
    if np.any(pseudoatom_mass_kg > ion_mass_kg):
        raise ValueError("Pseudo-atom mass must not exceed the ion mass.")


def _validated_core_arrays(
    ion_velocity_m_per_s: np.ndarray,
    ion_mass_kg: np.ndarray,
    ion_temperature_k: np.ndarray,
    gas_bulk_velocity_m_per_s: np.ndarray,
    gas_temperature_k: np.ndarray,
    gas_mass_kg: np.ndarray,
    pseudoatom_mass_kg: np.ndarray,
) -> tuple[np.ndarray, ...]:
    ion_velocity = _matrix3(ion_velocity_m_per_s, "ion_velocity_m_per_s")
    count = ion_velocity.shape[0]
    gas_velocity = _matrix3(
        gas_bulk_velocity_m_per_s,
        "gas_bulk_velocity_m_per_s",
    )
    if gas_velocity.shape[0] != count:
        raise ValueError("Ion and gas velocity batches must have equal lengths.")
    arrays = (
        _vector(ion_mass_kg, count, "ion_mass_kg"),
        _vector(ion_temperature_k, count, "ion_temperature_k"),
        _vector(gas_temperature_k, count, "gas_temperature_k"),
        _vector(gas_mass_kg, count, "gas_mass_kg"),
        _vector(pseudoatom_mass_kg, count, "pseudoatom_mass_kg"),
    )
    if np.any(arrays[1] <= 0.0) or np.any(arrays[2] <= 0.0):
        raise ValueError("Ion and gas temperatures must be positive.")
    _validate_masses(arrays[0], arrays[3], arrays[4])
    return (ion_velocity, gas_velocity, *arrays)


def _sample_thermal_velocities(
    count: int,
    ion_temperature_k: np.ndarray,
    gas_temperature_k: np.ndarray,
    gas_mass_kg: np.ndarray,
    pseudoatom_mass_kg: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    draws = rng.normal(0.0, 1.0, size=(count, 2, 3))
    gas_thermal = draws[:, 0, :] * np.sqrt(
        Boltzmann * gas_temperature_k / gas_mass_kg
    )[:, None]
    atom_thermal = draws[:, 1, :] * np.sqrt(
        Boltzmann * ion_temperature_k / pseudoatom_mass_kg
    )[:, None]
    return gas_thermal, atom_thermal


def _elastic_pair_after(
    atom_velocity_before: np.ndarray,
    gas_velocity_before: np.ndarray,
    atom_mass_kg: np.ndarray,
    gas_mass_kg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Prell 2024 Eq. 38-39 to the pseudo-atom/gas pair."""

    denominator = atom_mass_kg + gas_mass_kg
    atom_after = (
        (atom_mass_kg - gas_mass_kg)[:, None] * atom_velocity_before
        + (2.0 * gas_mass_kg)[:, None] * gas_velocity_before
    ) / denominator[:, None]
    gas_after = (
        (2.0 * atom_mass_kg)[:, None] * atom_velocity_before
        + (gas_mass_kg - atom_mass_kg)[:, None] * gas_velocity_before
    ) / denominator[:, None]
    return atom_after, gas_after


def _recombine_ion_velocity(
    ion_velocity_before: np.ndarray,
    atom_velocity_after: np.ndarray,
    ion_mass_kg: np.ndarray,
    atom_mass_kg: np.ndarray,
) -> np.ndarray:
    """Recombine pseudo-atom momentum with the rest of the ion, Eq. 40-41."""

    return (
        (ion_mass_kg - atom_mass_kg)[:, None] * ion_velocity_before
        + atom_mass_kg[:, None] * atom_velocity_after
    ) / ion_mass_kg[:, None]


def _core_result(
    ion_velocity_before: np.ndarray,
    ion_velocity_after: np.ndarray,
    atom_velocity_before: np.ndarray,
    atom_velocity_after: np.ndarray,
    gas_velocity_before: np.ndarray,
    gas_velocity_after: np.ndarray,
    atom_ion_velocity_before: np.ndarray,
    atom_mass_kg: np.ndarray,
) -> IictCollisionCoreResult:
    """Account for internal energy with Prell 2024 Eq. 43."""

    atom_ion_after = atom_velocity_after - ion_velocity_before
    delta_u = 0.5 * atom_mass_kg * (
        np.einsum("ij,ij->i", atom_ion_after, atom_ion_after)
        - np.einsum(
            "ij,ij->i",
            atom_ion_velocity_before,
            atom_ion_velocity_before,
        )
    )
    if (
        not np.all(np.isfinite(ion_velocity_after))
        or not np.all(np.isfinite(delta_u))
    ):
        raise FloatingPointError("IICT collision produced non-finite state.")
    return IictCollisionCoreResult(
        ion_velocity_after,
        delta_u,
        atom_velocity_before,
        atom_velocity_after,
        gas_velocity_before,
        gas_velocity_after,
        atom_ion_velocity_before,
        atom_ion_after,
    )


def apply_iict_collision_core(
    *,
    ion_velocity_m_per_s: np.ndarray,
    ion_mass_kg: np.ndarray,
    ion_temperature_k: np.ndarray,
    gas_bulk_velocity_m_per_s: np.ndarray,
    gas_temperature_k: np.ndarray,
    gas_mass_kg: np.ndarray,
    pseudoatom_mass_kg: np.ndarray,
    rng: np.random.Generator,
) -> IictCollisionCoreResult:
    """Apply one head-on IICT impulse per row using Eq. 38-43."""

    arrays = _validated_core_arrays(
        ion_velocity_m_per_s, ion_mass_kg, ion_temperature_k,
        gas_bulk_velocity_m_per_s, gas_temperature_k, gas_mass_kg,
        pseudoatom_mass_kg,
    )
    ion_v, gas_bulk_v, ion_m, ion_t, gas_t, gas_m, atom_m = arrays
    count = ion_v.shape[0]
    gas_thermal, atom_ion_before = _sample_thermal_velocities(
        count, ion_t, gas_t, gas_m, atom_m, rng,
    )
    gas_lab_before = gas_bulk_v + gas_thermal
    atom_lab_before = ion_v + atom_ion_before
    atom_lab_after, gas_lab_after = _elastic_pair_after(
        atom_lab_before,
        gas_lab_before,
        atom_m,
        gas_m,
    )
    ion_v_after = _recombine_ion_velocity(
        ion_v,
        atom_lab_after,
        ion_m,
        atom_m,
    )
    return _core_result(
        ion_v,
        ion_v_after,
        atom_lab_before,
        atom_lab_after,
        gas_lab_before,
        gas_lab_after,
        atom_ion_before,
        atom_m,
    )


def iict_transfer_efficiency(
    pseudoatom_mass_kg: float | np.ndarray,
    gas_mass_kg: float | np.ndarray,
) -> float | np.ndarray:
    """Return ``chi_ag = 4 ma mg / (ma + mg)^2``."""

    atom_m, gas_m = np.broadcast_arrays(
        np.asarray(pseudoatom_mass_kg, dtype=np.float64),
        np.asarray(gas_mass_kg, dtype=np.float64),
    )
    if np.any(atom_m <= 0.0) or np.any(gas_m <= 0.0):
        raise ValueError("Transfer-efficiency masses must be positive.")
    result = 4.0 * atom_m * gas_m / (atom_m + gas_m) ** 2
    return float(result) if result.ndim == 0 else result


def expected_internal_energy_change_j(
    *,
    ion_temperature_k: float | np.ndarray,
    gas_temperature_k: float | np.ndarray,
    relative_ion_kinetic_energy_j: float | np.ndarray,
    ion_mass_kg: float | np.ndarray,
    gas_mass_kg: float | np.ndarray,
    pseudoatom_mass_kg: float | np.ndarray,
) -> float | np.ndarray:
    """Return the ensemble mean ``<Delta U>`` from Eq. 47."""

    ti, tg, ke, mi, mg, ma = np.broadcast_arrays(
        ion_temperature_k,
        gas_temperature_k,
        relative_ion_kinetic_energy_j,
        ion_mass_kg,
        gas_mass_kg,
        pseudoatom_mass_kg,
    )
    if np.any(mi <= 0.0) or np.any(mg <= 0.0) or np.any(ma <= 0.0):
        raise ValueError("Eq. 47 masses must be positive.")
    chi = np.asarray(iict_transfer_efficiency(ma, mg))
    result = 1.5 * chi * Boltzmann * (tg - ti) + chi * mg / mi * ke
    return float(result) if result.ndim == 0 else result


def expected_kinetic_energy_change_j(
    *,
    ion_temperature_k: float | np.ndarray,
    gas_temperature_k: float | np.ndarray,
    relative_ion_kinetic_energy_j: float | np.ndarray,
    ion_mass_kg: float | np.ndarray,
    gas_mass_kg: float | np.ndarray,
    pseudoatom_mass_kg: float | np.ndarray,
) -> float | np.ndarray:
    """Return the ensemble mean ion ``<Delta KE>`` from Eq. 42."""

    ti, tg, ke, mi, mg, ma = np.broadcast_arrays(
        ion_temperature_k,
        gas_temperature_k,
        relative_ion_kinetic_energy_j,
        ion_mass_kg,
        gas_mass_kg,
        pseudoatom_mass_kg,
    )
    if np.any(mi <= 0.0) or np.any(mg <= 0.0) or np.any(ma <= 0.0):
        raise ValueError("Eq. 42 masses must be positive.")
    chi = np.asarray(iict_transfer_efficiency(ma, mg))
    damping = -(chi / mi) * (ma + mg - mg * ma / mi) * ke
    ion_heating = (ma / mi) * (1.0 - chi) * 1.5 * Boltzmann * ti
    gas_heating = (ma / mi) * chi * 1.5 * Boltzmann * tg
    result = damping + ion_heating + gas_heating
    return float(result) if result.ndim == 0 else result


def steady_drift_temperature_k(
    gas_temperature_k: float | np.ndarray,
    gas_mass_kg: float | np.ndarray,
    drift_speed_m_per_s: float | np.ndarray,
) -> float | np.ndarray:
    """Return the two-temperature steady state from Eq. 51-52."""

    tg, mg, speed = np.broadcast_arrays(
        gas_temperature_k,
        gas_mass_kg,
        drift_speed_m_per_s,
    )
    if np.any(tg <= 0.0) or np.any(mg <= 0.0) or np.any(speed < 0.0):
        raise ValueError("Eq. 52 inputs must have physical positive values.")
    result = tg + mg * speed**2 / (3.0 * Boltzmann)
    return float(result) if result.ndim == 0 else result


def mean_relative_speed_m_per_s(
    ion_speed_m_per_s: float | np.ndarray,
    gas_temperature_k: float | np.ndarray,
    gas_mass_kg: float | np.ndarray,
) -> float | np.ndarray:
    """Return the drifting Maxwellian mean relative speed, Eq. 55-56."""

    speed, temperature, mass = np.broadcast_arrays(
        ion_speed_m_per_s,
        gas_temperature_k,
        gas_mass_kg,
    )
    if np.any(speed < 0.0) or np.any(temperature <= 0.0) or np.any(mass <= 0.0):
        raise ValueError("Eq. 55 inputs must have physical positive values.")
    thermal_scale = np.sqrt(2.0 * Boltzmann * temperature / mass)
    s_value = speed / thermal_scale
    safe_s = np.maximum(s_value, 1.0e-8)
    bracket = (
        (safe_s + 0.5 / safe_s) * (np.sqrt(np.pi) / 2.0) * erf(safe_s)
        + 0.5 * np.exp(-(safe_s**2))
    )
    bracket = np.where(s_value < 1.0e-8, 1.0, bracket)
    result = np.sqrt(8.0 * Boltzmann * temperature / (np.pi * mass)) * bracket
    return float(result) if result.ndim == 0 else result


def collision_frequency_s_inverse(
    *,
    ion_speed_m_per_s: float | np.ndarray,
    gas_temperature_k: float | np.ndarray,
    gas_mass_kg: float | np.ndarray,
    gas_number_density_m3: float | np.ndarray,
    collision_cross_section_m2: float | np.ndarray,
) -> float | np.ndarray:
    """Return ``N_g * sigma_ig * <v_rel>`` from Eq. 54-56."""

    speed, density, cross_section = np.broadcast_arrays(
        ion_speed_m_per_s,
        gas_number_density_m3,
        collision_cross_section_m2,
    )
    if np.any(density < 0.0) or np.any(cross_section < 0.0):
        raise ValueError("Gas density and collision cross section must be non-negative.")
    mean_relative = np.asarray(
        mean_relative_speed_m_per_s(speed, gas_temperature_k, gas_mass_kg)
    )
    result = density * cross_section * mean_relative
    return float(result) if result.ndim == 0 else result


def mean_free_path_m(
    ion_speed_m_per_s: float | np.ndarray,
    collision_frequency_s_inverse_value: float | np.ndarray,
) -> float | np.ndarray:
    """Return Eq. 54 mean distance, ``lambda = <v_i> / frequency``."""

    speed, frequency = np.broadcast_arrays(
        ion_speed_m_per_s,
        collision_frequency_s_inverse_value,
    )
    if (
        not np.all(np.isfinite(speed))
        or not np.all(np.isfinite(frequency))
        or np.any(speed < 0.0)
        or np.any(frequency < 0.0)
    ):
        raise ValueError(
            "Speed and collision frequency must be finite and non-negative."
        )
    result = np.full(speed.shape, np.inf, dtype=np.float64)
    np.divide(speed, frequency, out=result, where=frequency > 0.0)
    return float(result) if result.ndim == 0 else result


def mean_collision_waiting_time_s(
    collision_frequency_s_inverse_value: float | np.ndarray,
) -> float | np.ndarray:
    """Return Eq. 58 mean waiting time, ``tau = 1 / frequency``."""

    frequency = np.asarray(
        collision_frequency_s_inverse_value,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("Collision frequency must be finite and non-negative.")
    result = np.full(frequency.shape, np.inf, dtype=np.float64)
    np.divide(1.0, frequency, out=result, where=frequency > 0.0)
    return float(result) if result.ndim == 0 else result


__all__ = [
    "IictCollisionCoreResult",
    "apply_iict_collision_core",
    "collision_frequency_s_inverse",
    "expected_internal_energy_change_j",
    "expected_kinetic_energy_change_j",
    "iict_transfer_efficiency",
    "mean_collision_waiting_time_s",
    "mean_free_path_m",
    "mean_relative_speed_m_per_s",
    "steady_drift_temperature_k",
]
