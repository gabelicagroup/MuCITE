"""Independent neutral-thermal rate checks and CPU/Taichi parity."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti
from scipy.constants import Avogadro, Boltzmann
from scipy.integrate import quad

from src.config import SimulationConfig
from src.core.micro_step import _validate_collision_step
from src.core.simulation import _ensure_pic_taichi_initialized
from src.core.timestep import InvalidTimeStepError
from src.env.gas.kinetics import collision_frequency_s_inverse, mean_relative_speed_m_per_s
from src.env.transport.collision_rate import iict_mean_relative_speed
from src.env.collisions.probability import collision_probability_from_lambda_dt


pytestmark = pytest.mark.physics
GAS_MASS_KG = 0.0280134 / Avogadro


@ti.kernel
def _thermal_speed_kernel(
    drift: ti.types.ndarray(dtype=ti.f64, ndim=1),
    temperature: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(drift.shape[0]):
        result[i] = iict_mean_relative_speed(drift[i], temperature[i], GAS_MASS_KG, Boltzmann)


def test_zero_drift_limit_and_cold_gas_limit():
    expected = np.sqrt(8 * Boltzmann * 300 / (np.pi * GAS_MASS_KG))
    assert mean_relative_speed_m_per_s(0, 300, GAS_MASS_KG) == pytest.approx(expected)
    assert mean_relative_speed_m_per_s(500, 0, GAS_MASS_KG) == 500
    assert mean_relative_speed_m_per_s(1e7, 300, GAS_MASS_KG) == pytest.approx(1e7, rel=1e-8)
    rate = collision_frequency_s_inverse(
        ion_speed_m_per_s=0, gas_temperature_k=300, gas_mass_kg=GAS_MASS_KG,
        gas_number_density_m3=200 / (Boltzmann * 300), collision_cross_section_m2=1.62e-18)
    assert rate == pytest.approx(3.72482e7, rel=2e-6)


@pytest.mark.parametrize("drift", [0.01, 100, 500, 2000, 100000])
def test_mean_relative_speed_matches_noncentral_speed_density_integral(drift):
    a = np.sqrt(2 * Boltzmann * 300 / GAS_MASS_KG)
    def integrand(g):
        return g**2 / (np.sqrt(np.pi) * a * drift) * (
            np.exp(-((g - drift) / a)**2) - np.exp(-((g + drift) / a)**2))
    expected, _ = quad(integrand, max(0, drift - 12*a), drift + 12*a,
                       epsabs=1e-7, epsrel=1e-10)
    assert mean_relative_speed_m_per_s(drift, 300, GAS_MASS_KG) == pytest.approx(expected, rel=1e-9)


def test_cpu_taichi_relative_speed_parity_including_series_transition():
    _ensure_pic_taichi_initialized("cpu")
    s = np.r_[0, np.geomspace(1e-12, 1e4, 200), .249999, .25, .250001]
    temperatures = np.repeat([0, 1, 300, 473, 2000], len(s)).astype(float)
    drift = np.tile(s, 5) * np.sqrt(2 * Boltzmann * np.maximum(temperatures, 1) / GAS_MASS_KG)
    actual = np.zeros_like(drift)
    _thermal_speed_kernel(drift, temperatures, actual)
    expected = mean_relative_speed_m_per_s(drift, temperatures, GAS_MASS_KG)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)


def test_zero_density_and_cross_section_give_zero_rate():
    actual = collision_frequency_s_inverse(
        ion_speed_m_per_s=0, gas_temperature_k=300, gas_mass_kg=GAS_MASS_KG,
        gas_number_density_m3=[0, 1e20], collision_cross_section_m2=[1e-18, 0])
    np.testing.assert_array_equal(actual, 0)
    for invalid in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            mean_relative_speed_m_per_s(invalid, 300, GAS_MASS_KG)


def test_tiny_collision_probabilities_remain_nonzero():
    x = np.array([1e-30, 1e-20, 1e-12])
    p = collision_probability_from_lambda_dt(x)
    np.testing.assert_allclose(p, x, rtol=1e-12)
    assert np.all(p > 0)


def test_actual_global_step_is_checked_after_reduction_and_hybrid_gate():
    runtime = SimpleNamespace(
        integrator=SimpleNamespace(collision_safety_factor=.1),
        collision_model_code=0,
        config=SimulationConfig(langevin_z_start_m=0, langevin_z_end_m=1),
        _download_selected_motion_state=lambda idx: (np.zeros((idx.size, 3)), None),
    )
    run = SimpleNamespace(particle_state=dict(active=np.array([1, 0])))
    rates = np.array([1e9, np.inf])
    _validate_collision_step(runtime, run, 1e-11, rates)
    with pytest.raises(InvalidTimeStepError, match="collision-rate"):
        _validate_collision_step(runtime, run, 1e-9, rates)
    runtime.collision_model_code = 1
    _validate_collision_step(runtime, run, 1e-9, rates)
