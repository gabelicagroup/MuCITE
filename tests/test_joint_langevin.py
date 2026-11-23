"""Joint IICT moments, guarded diffusion and event fallback physics contracts."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.constants import Avogadro, Boltzmann, atomic_mass
from scipy.linalg import expm

from src.config import IonTemplate, SimulationConfig, make_simulation_config
from src.core.micro_step import _coordinate_collision_step, _validate_collision_step
from src.core.timestep import InvalidTimeStepError
from src.env.collisions.iict_lite.physics import apply_iict_collision_core
from src.env.collisions.factory import build_collision_physics_backend
from src.env.collisions.joint_moments import gaussian_joint_step, iict_jump_moments
from src.env.collisions.joint_step import JointIictCollisionStep

pytestmark = pytest.mark.physics
MG = .0280134 / Avogadro


def _moments(count, *, mass_da=5000):
    w = np.tile([100., -200., 300.], (count, 1))
    mass = np.full(count, mass_da * atomic_mass)
    ti, tg, ma = np.full(count, 800.), np.full(count, 300.), np.full(count, 30*atomic_mass)
    return w, mass, ti, tg, ma, iict_jump_moments(w, mass, ti, tg, MG, ma)


def test_joint_jump_moments_match_existing_iict_core_by_exact_gaussian_quadrature():
    nodes, weights = hermgauss(3)
    combinations = np.array(list(product(range(3), repeat=6)))
    normal = np.sqrt(2) * nodes[combinations]
    probability = np.prod(weights[combinations] / np.sqrt(np.pi), axis=1)
    w, mass, ti, tg, ma, moments = _moments(len(normal), mass_da=556.27)
    rng = SimpleNamespace(normal=lambda *_, **__: normal.reshape(-1, 2, 3))
    core = apply_iict_collision_core(
        ion_velocity_m_per_s=w, ion_mass_kg=mass, ion_temperature_k=ti,
        gas_bulk_velocity_m_per_s=np.zeros_like(w), gas_temperature_k=tg,
        gas_mass_kg=np.full(len(w), MG), pseudoatom_mass_kg=ma, rng=rng)
    scale = np.array([100., 100., 100., 1e-20])
    jump = np.column_stack((core.ion_velocity_after_m_per_s - w,
                            core.transferred_internal_energy_j)) / scale
    expected_mean = np.r_[moments.mean_velocity[0], moments.mean_energy_j[0]] / scale
    expected_second = moments.raw_second_moment()[0] / np.outer(scale, scale)
    np.testing.assert_allclose(probability @ jump, expected_mean, rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(np.einsum("n,ni,nj->ij", probability, jump, jump),
                               expected_second, rtol=2e-13, atol=1e-13)


def test_frozen_joint_diffusion_matches_linear_sde_mean_and_covariance():
    count, lam = 100_000, 8.
    w, _, _, _, _, moments = _moments(count)
    energy = np.full(count, 1e-17)
    v, u = gaussian_joint_step(w, energy, moments, np.full(count, lam), np.random.default_rng(915))
    scale = np.array([100., 100., 100., 1e-19])
    observations = np.column_stack((v, u - energy)) / scale
    c = moments.momentum_fraction[0]
    drift = np.diag([-c, -c, -c, 0.])
    q = moments.raw_second_moment()[0] / np.outer(scale, scale)
    block = np.block([[drift, q], [np.zeros((4, 4)), -drift.T]])
    propagation = expm(lam * block)
    covariance = propagation[:4, 4:] @ expm(lam * drift.T)
    expected_mean = np.r_[w[0] * np.exp(-c * lam), lam * moments.mean_energy_j[0]] / scale
    standard_errors = np.sqrt(np.diag(covariance) / count)
    assert np.all(np.abs(observations.mean(axis=0) - expected_mean) < 6 * standard_errors)
    normalization = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    np.testing.assert_allclose(np.cov(observations, rowvar=False) / normalization,
                               covariance / normalization, rtol=0, atol=.02)


class _TransparentBackend:
    internal_energy_bounds_j = (0., float("inf"))

    def energy_from_temperature(self, t):
        return np.asarray(t) * 1000 * Boltzmann

    def temperature_from_energy(self, u):
        return np.asarray(u) / (1000 * Boltzmann)

    def pseudoatom_mass(self, t, w, gas):
        return np.full_like(t, 30 * atomic_mass)

    def apply_collision_state_batch(self, **kwargs):
        return kwargs["velocities_m_per_s"], kwargs["internal_temperatures_k"]


def test_sparse_fallback_preserves_no_collision_probability_and_actual_counts():
    count, duration, density = 50_000, 1e-9, 1e20
    template = IonTemplate(collision_cross_section_m2=1.7e-18)
    result = JointIictCollisionStep(
        backend=_TransparentBackend(), template=template,
        config=replace(SimulationConfig(), langevin_min_collisions=1e6),
        velocities_m_per_s=np.zeros((count, 3)), masses_kg=np.full(count, template.mass_kg),
        temperatures_k=np.full(count, 800.), gas_velocities_m_per_s=np.zeros((count, 3)),
        gas_temperatures_k=np.full(count, 300.), gas_number_density_m3=np.full(count, density),
        dt_s=duration, rng=np.random.default_rng(33),
    ).run()
    lam = density * 1.7e-18 * np.sqrt(8 * Boltzmann * 300 / (np.pi * MG)) * duration
    p0 = np.exp(-lam)
    assert abs(np.mean(result.binary_counts == 0) - p0) < 6*np.sqrt(p0*(1-p0)/count)
    assert abs(np.mean(result.binary_counts) - lam) < 6*np.sqrt(lam/count)
    np.testing.assert_array_equal(result.gaussian_steps, 0)
    np.testing.assert_array_equal(result.velocities_m_per_s, 0)


def test_zero_density_uses_no_events_or_noise():
    rng = np.random.default_rng(4)
    config, backend = SimulationConfig(), _TransparentBackend()
    result = JointIictCollisionStep(
        backend=backend, template=IonTemplate(), config=config,
        velocities_m_per_s=np.zeros((2, 3)), masses_kg=np.full(2, 5000*atomic_mass),
        temperatures_k=np.full(2, 800.), gas_velocities_m_per_s=np.zeros((2, 3)),
        gas_temperatures_k=np.full(2, 300.), gas_number_density_m3=np.zeros(2),
        dt_s=1., rng=rng,
    ).run()
    np.testing.assert_array_equal(result.binary_counts, 0)
    np.testing.assert_array_equal(rng.random(4), np.random.default_rng(4).random(4))


def _public_iict_config(num_atoms):
    return replace(SimulationConfig(), collision_physics_backend="iict-lite",
                   iict_heat_capacity_model="classical", iict_pseudoatom_model="constant",
                   iict_fragmentation_model="none", iict_num_atoms=num_atoms,
                   iict_pseudoatom_mass_da=30.0)


def test_dense_joint_branch_updates_internal_energy_for_iict_lite():
    count = 128
    config = _public_iict_config(1200)
    template = IonTemplate(mass_kg=5000*atomic_mass, num_atoms=1200)
    backend = build_collision_physics_backend(config=config, template=template)
    result = JointIictCollisionStep(
        backend=backend, template=template, config=config,
        velocities_m_per_s=np.zeros((count, 3)), masses_kg=np.full(count, template.mass_kg),
        temperatures_k=np.full(count, 800.), gas_velocities_m_per_s=np.zeros((count, 3)),
        gas_temperatures_k=np.full(count, 300.), gas_number_density_m3=np.full(count, 1e22),
        dt_s=3e-7, rng=np.random.default_rng(22),
    ).run()
    assert np.sum(result.gaussian_steps) > 0
    assert np.mean(result.temperatures_k) < 800
    assert np.all(np.isfinite(result.temperatures_k))
    assert np.all(result.temperatures_k > 0)
    assert np.var(result.velocities_m_per_s[:, 2]) > 0


def test_light_ion_guard_uses_actual_binary_energy_updates():
    config = _public_iict_config(77)
    template = IonTemplate(mass_kg=556.27*atomic_mass, num_atoms=77)
    backend = build_collision_physics_backend(config=config, template=template)
    result = JointIictCollisionStep(
        backend=backend, template=template, config=config,
        velocities_m_per_s=np.zeros((128, 3)), masses_kg=np.full(128, template.mass_kg),
        temperatures_k=np.full(128, 800.), gas_velocities_m_per_s=np.zeros((128, 3)),
        gas_temperatures_k=np.full(128, 300.), gas_number_density_m3=np.full(128, 1e22),
        dt_s=3e-7, rng=np.random.default_rng(22),
    ).run()
    assert np.sum(result.gaussian_steps) == 0
    assert np.sum(result.binary_counts) > 0
    assert result.guard_fallbacks > 0
    assert np.mean(result.temperatures_k) < 800


def test_global_hybrid_gate_is_recoordinated_before_rejecting_a_step():
    config = SimulationConfig(collision_model="hybrid-langevin", langevin_switch_probability=.5,
                              langevin_z_start_m=0, langevin_z_end_m=.01)
    runtime = SimpleNamespace(config=config, collision_model_code=1,
        integrator=SimpleNamespace(collision_safety_factor=.1, min_dt_s=1e-12),
        _download_selected_motion_state=lambda i: (np.zeros((len(i), 3)), None))
    run = SimpleNamespace(particle_state=dict(active=np.ones(2)))
    rates = np.array([1e9, 1e10])
    selected = _coordinate_collision_step(runtime, run, .2e-9, rates)
    assert selected.dt_s == pytest.approx(.1e-9)
    _validate_collision_step(runtime, run, selected.dt_s, rates)
    runtime.integrator.min_dt_s = .15e-9
    with pytest.raises(InvalidTimeStepError, match="Minimum timestep"):
        _coordinate_collision_step(runtime, run, .2e-9, rates)


def test_final_branch_respects_hybrid_ceiling_even_if_local_gate_was_explicit():
    from src.env.collisions.runtime_preselection import _hybrid_mask
    config = SimulationConfig(collision_model="hybrid-langevin", langevin_switch_probability=.05,
                              langevin_z_start_m=0, langevin_z_end_m=.01, langevin_max_dt_s=1e-9)
    runtime = SimpleNamespace(config=config, collision_model_code=1,
        integrator=SimpleNamespace(collision_safety_factor=.1, min_dt_s=1e-12),
        _download_selected_motion_state=lambda i: (np.zeros((len(i), 3)), None))
    run = SimpleNamespace(particle_state=dict(active=np.ones(1)))
    rate = np.array([3.7e7])
    # At 1 ns the local gated probability is .037, but the old explicit
    # candidate .1/nu re-enters hybrid above the configured 1-ns ceiling.
    selected = _coordinate_collision_step(runtime, run, .1/rate[0], rate)
    assert selected is None  # The explicit bound is valid; no needless clipping.
    actual_dt = .1/rate[0]
    mask = _hybrid_mask(runtime, np.zeros((1, 3)), rate*actual_dt, actual_dt)
    np.testing.assert_array_equal(mask, False)
    _validate_collision_step(runtime, run, actual_dt, rate)
    runtime.integrator.min_dt_s = 2e-9
    assert _coordinate_collision_step(runtime, run, actual_dt, rate) is None


@pytest.mark.parametrize("field,value", [("langevin_min_collisions", 0),
    ("langevin_max_relaxation", .3), ("langevin_max_energy_fraction", float("nan"))])
def test_joint_controls_reject_invalid_requests(field, value):
    with pytest.raises(ValueError):
        make_simulation_config(replace(SimulationConfig(), **{field: value}))
