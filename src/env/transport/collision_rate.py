"""Taichi-compatible collision-rate formulas for the runtime scheduler."""

import taichi as ti


@ti.func
def _erf_approx_positive(value: ti.f64) -> ti.f64:
    """Approximate erf(x) for x >= 0 with maximum error about 1.5e-7."""

    t_value = 1.0 / (1.0 + 0.3275911 * value)
    polynomial = (
        (
            (
                (1.061405429 * t_value - 1.453152027) * t_value
                + 1.421413741
            )
            * t_value
            - 0.284496736
        )
        * t_value
        + 0.254829592
    )
    return 1.0 - polynomial * t_value * ti.exp(-(value * value))


@ti.func
def iict_mean_relative_speed(
    bulk_relative_speed_m_per_s: ti.f64,
    gas_temperature_k: ti.f64,
    gas_molecule_mass_kg: ti.f64,
    boltzmann_j_per_k: ti.f64,
) -> ti.f64:
    """Prell 2024 Eq. 55-56; DOI 10.1016/j.ijms.2024.117290."""

    thermal_scale = ti.sqrt(
        2.0 * boltzmann_j_per_k * gas_temperature_k
        / gas_molecule_mass_kg
    )
    s_value = bulk_relative_speed_m_per_s / thermal_scale
    bracket = 1.0
    if s_value >= 1.0e-6:
        erf_value = _erf_approx_positive(s_value)
        bracket = (
            (s_value + 0.5 / s_value)
            * 0.886226925452758
            * erf_value
            + 0.5 * ti.exp(-(s_value * s_value))
        )
    mean_thermal = ti.sqrt(
        8.0 * boltzmann_j_per_k * gas_temperature_k
        / (3.141592653589793 * gas_molecule_mass_kg)
    )
    return mean_thermal * bracket


__all__ = ["iict_mean_relative_speed"]
