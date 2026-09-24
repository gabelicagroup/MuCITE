from __future__ import annotations

import math

import pytest

from src.data.diagnostics.pic_convergence import (
    EPSILON_0,
    plasma_frequency_rad_per_s,
    run_macro_step_convergence,
    run_manufactured_poisson_convergence,
)


def test_plasma_frequency_uses_number_density_charge_and_mass() -> None:
    omega = plasma_frequency_rad_per_s(
        2.5e14,
        charge_c=2.0e-19,
        mass_kg=3.0e-25,
    )
    assert omega == pytest.approx(math.sqrt(2.5e14 * (2.0e-19) ** 2 / (EPSILON_0 * 3.0e-25)))

    with pytest.raises(ValueError, match="number_density"):
        plasma_frequency_rad_per_s(0.0, charge_c=1.0, mass_kg=1.0)
    with pytest.raises(ValueError, match="charge_c"):
        plasma_frequency_rad_per_s(1.0, charge_c=0.0, mass_kg=1.0)


def test_manufactured_poisson_solution_is_second_order_on_nested_grids() -> None:
    result = run_manufactured_poisson_convergence(
        grid_shapes=((9, 25), (17, 49), (33, 97)),
    )

    errors = [row["relative_l2_phi_error"] for row in result["rows"]]
    orders = [row["observed_order"] for row in result["rows"][1:]]
    assert result["passed"] is True
    assert errors[0] > errors[1] > errors[2]
    assert min(orders) >= 1.8
    assert all(row["poisson_converged"] for row in result["rows"])


def test_frozen_field_macro_error_decreases_and_default_phase_passes() -> None:
    result = run_macro_step_convergence(
        number_density_m3=1.0e14,
        ion_mass_amu=100.0,
        charge_state=1,
        macro_dt_values_s=(2.0e-7, 1.0e-7, 5.0e-8, 2.5e-8),
        reference_macro_dt_s=5.0e-8,
    )

    errors = [row["phase_space_error"] for row in result["rows"]]
    orders = [row["observed_order"] for row in result["rows"][1:]]
    assert result["passed"] is True
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert min(orders) >= 0.8
    assert result["reference_macro_phase_rad"] < result["max_macro_phase_rad"]
