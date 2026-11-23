"""Neutral Maxwellian collision rates shared by all collision backends."""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann
from scipy.special import erf

COLLISION_RATE_MODEL = "neutral-maxwellian-mean-relative-speed-v1"


def mean_relative_speed_m_per_s(
    ion_speed_m_per_s: float | np.ndarray,
    gas_temperature_k: float | np.ndarray,
    gas_mass_kg: float | np.ndarray,
) -> float | np.ndarray:
    """Average relative speed for a fixed ion velocity and Maxwellian neutrals.

    ``ion_speed_m_per_s`` is the drift relative to the gas bulk velocity,
    not a thermally averaged ion speed. The cold-gas limit is the drift.
    """
    speed, temperature, mass = np.broadcast_arrays(
        np.asarray(ion_speed_m_per_s, dtype=np.float64),
        np.asarray(gas_temperature_k, dtype=np.float64),
        np.asarray(gas_mass_kg, dtype=np.float64),
    )
    if (
        not all(np.all(np.isfinite(a)) for a in (speed, temperature, mass))
        or np.any(speed < 0) or np.any(temperature < 0) or np.any(mass <= 0)
    ):
        raise ValueError("Speed/temperature must be finite non-negative; mass positive.")
    a = np.sqrt(2.0 * Boltzmann * temperature / mass)
    s = np.divide(speed, a, out=np.zeros_like(speed), where=a > 0)
    u = np.minimum(s, 0.25) ** 2
    # Series avoids the 1/s term and amplified erf error near zero drift.
    series = 1.0 + u * (1 / 3 + u * (-1 / 30 + u * (
        1 / 210 + u * (-1 / 1512 + u / 11880))))
    safe_s = np.maximum(s, 0.25)
    regular = (
        (safe_s + 0.5 / safe_s) * np.sqrt(np.pi) / 2 * erf(safe_s)
        + 0.5 * np.exp(-safe_s**2)
    )
    result = np.where(a > 0, 2 * a / np.sqrt(np.pi)
                      * np.where(s < 0.25, series, regular), speed)
    return float(result) if result.ndim == 0 else result


def collision_frequency_s_inverse(
    *, ion_speed_m_per_s: float | np.ndarray,
    gas_temperature_k: float | np.ndarray,
    gas_mass_kg: float | np.ndarray,
    gas_number_density_m3: float | np.ndarray,
    collision_cross_section_m2: float | np.ndarray,
) -> float | np.ndarray:
    """Return n*sigma*<g>, independent of macro-particle weight."""
    density, cross_section = np.broadcast_arrays(
        np.asarray(gas_number_density_m3, dtype=np.float64),
        np.asarray(collision_cross_section_m2, dtype=np.float64),
    )
    if (
        not np.all(np.isfinite(density)) or not np.all(np.isfinite(cross_section))
        or np.any(density < 0) or np.any(cross_section < 0)
    ):
        raise ValueError("Density and cross section must be finite non-negative.")
    result = density * cross_section * mean_relative_speed_m_per_s(
        ion_speed_m_per_s, gas_temperature_k, gas_mass_kg,
    )
    return float(result) if np.ndim(result) == 0 else result
