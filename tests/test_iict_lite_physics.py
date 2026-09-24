"""Equation-level and runtime validation for the paper-driven IICT core."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import Boltzmann, atomic_mass

from src.agents.taichi_cloud import IonCloud3D
from src.config import IonTemplate, SimulationConfig
from src.core.simulation import FlowchartPicSimulation, _ensure_pic_taichi_initialized
from src.data.report.snapshot import capture_report_snapshot
from src.data.report.summary_builder import build_summary
from src.env.collisions import (
    CollisionPhysicsBackend,
    IonSpaCollisionAdapter,
    build_collision_physics_backend,
)
from src.env.collisions.iict_lite import (
    apply_iict_collision_core,
    collision_frequency_s_inverse,
    expected_internal_energy_change_j,
    expected_kinetic_energy_change_j,
    mean_collision_waiting_time_s,
    mean_free_path_m,
    mean_relative_speed_m_per_s,
    steady_drift_temperature_k,
)
from src.env.fields.axisymmetric_grid import UnifiedGrid2D
from src.env.transport import ParticlePusher


def _core_inputs(count: int) -> dict[str, np.ndarray]:
    return {
        "ion_velocity_m_per_s": np.tile([100.0, -20.0, 700.0], (count, 1)),
        "ion_mass_kg": np.full(count, 5000.0 * atomic_mass),
        "ion_temperature_k": np.full(count, 600.0),
        "gas_bulk_velocity_m_per_s": np.zeros((count, 3)),
        "gas_temperature_k": np.full(count, 300.0),
        "gas_mass_kg": np.full(count, 28.0 * atomic_mass),
        "pseudoatom_mass_kg": np.full(count, 30.0 * atomic_mass),
    }


def _lite_config(**overrides: object) -> SimulationConfig:
    values = {
        "collision_physics_backend": "iict-lite",
        "iict_heat_capacity_model": "classical",
        "iict_num_atoms": 72,
        "iict_pseudoatom_model": "constant",
        "iict_pseudoatom_mass_da": 30.0,
        "iict_fragmentation_model": "none",
    }
    values.update(overrides)
    return SimulationConfig(**values)


def _batch_arguments(
    template: IonTemplate,
    *,
    count: int,
    temperature_k: float,
    speed_m_per_s: float,
) -> dict[str, object]:
    number_density = np.full(count, 2.0e20)
    return {
        "velocities_m_per_s": np.tile([0.0, 0.0, speed_m_per_s], (count, 1)),
        "masses_kg": np.full(count, template.mass_kg),
        "internal_temperatures_k": np.full(count, temperature_k),
        "gas_velocities_m_per_s": np.zeros((count, 3)),
        "gas_temperatures_k": np.full(count, 300.0),
        "gas_number_density_m3": number_density,
        "gas_mass_density_kg_per_m3": number_density * 28.0 * atomic_mass,
        "template": template,
        "gas_name": "n2",
        "dt_s": 1.0e-9,
    }


def test_single_collision_momentum_and_energy_accounting() -> None:
    inputs = _core_inputs(1)
    result = apply_iict_collision_core(
        **inputs,
        rng=np.random.default_rng(91),
    )
    atom_mass = inputs["pseudoatom_mass_kg"][0]
    gas_mass = inputs["gas_mass_kg"][0]
    ion_mass = inputs["ion_mass_kg"][0]
    pair_momentum_before = (
        atom_mass * result.pseudoatom_velocity_lab_before_m_per_s[0]
        + gas_mass * result.gas_velocity_lab_before_m_per_s[0]
    )
    pair_momentum_after = (
        atom_mass * result.pseudoatom_velocity_lab_after_m_per_s[0]
        + gas_mass * result.gas_velocity_lab_after_m_per_s[0]
    )
    np.testing.assert_allclose(
        pair_momentum_after,
        pair_momentum_before,
        rtol=2.0e-15,
        atol=1.0e-35,
    )
    energy_before = 0.5 * atom_mass * np.dot(
        result.pseudoatom_velocity_lab_before_m_per_s[0],
        result.pseudoatom_velocity_lab_before_m_per_s[0],
    ) + 0.5 * gas_mass * np.dot(
        result.gas_velocity_lab_before_m_per_s[0],
        result.gas_velocity_lab_before_m_per_s[0],
    )
    energy_after = 0.5 * atom_mass * np.dot(
        result.pseudoatom_velocity_lab_after_m_per_s[0],
        result.pseudoatom_velocity_lab_after_m_per_s[0],
    ) + 0.5 * gas_mass * np.dot(
        result.gas_velocity_lab_after_m_per_s[0],
        result.gas_velocity_lab_after_m_per_s[0],
    )
    assert energy_after == pytest.approx(energy_before, rel=3.0e-15)
    expected_ion_momentum = (
        (ion_mass - atom_mass) * inputs["ion_velocity_m_per_s"][0]
        + atom_mass * result.pseudoatom_velocity_lab_after_m_per_s[0]
    )
    np.testing.assert_allclose(
        ion_mass * result.ion_velocity_after_m_per_s[0],
        expected_ion_momentum,
        rtol=2.0e-15,
        atol=1.0e-35,
    )
    expected_delta_u = 0.5 * atom_mass * (
        np.dot(
            result.pseudoatom_velocity_ion_after_m_per_s[0],
            result.pseudoatom_velocity_ion_after_m_per_s[0],
        )
        - np.dot(
            result.pseudoatom_velocity_ion_before_m_per_s[0],
            result.pseudoatom_velocity_ion_before_m_per_s[0],
        )
    )
    assert result.transferred_internal_energy_j[0] == pytest.approx(
        expected_delta_u,
        rel=2.0e-15,
    )


@pytest.mark.physics
def test_monte_carlo_means_match_equations_42_and_47() -> None:
    count = 200_000
    inputs = _core_inputs(count)
    result = apply_iict_collision_core(
        **inputs,
        rng=np.random.default_rng(2024),
    )
    velocity = inputs["ion_velocity_m_per_s"][0]
    ion_mass = inputs["ion_mass_kg"][0]
    relative_ke = 0.5 * ion_mass * np.dot(velocity, velocity)
    parameters = {
        "ion_temperature_k": 600.0,
        "gas_temperature_k": 300.0,
        "relative_ion_kinetic_energy_j": relative_ke,
        "ion_mass_kg": ion_mass,
        "gas_mass_kg": 28.0 * atomic_mass,
        "pseudoatom_mass_kg": 30.0 * atomic_mass,
    }
    expected_u = expected_internal_energy_change_j(**parameters)
    actual_u = float(np.mean(result.transferred_internal_energy_j))
    assert actual_u == pytest.approx(expected_u, rel=0.03)
    ke_before = 0.5 * ion_mass * np.sum(
        inputs["ion_velocity_m_per_s"] ** 2,
        axis=1,
    )
    ke_after = 0.5 * ion_mass * np.sum(
        result.ion_velocity_after_m_per_s**2,
        axis=1,
    )
    expected_ke = expected_kinetic_energy_change_j(**parameters)
    assert float(np.mean(ke_after - ke_before)) == pytest.approx(
        expected_ke,
        rel=0.03,
    )


@pytest.mark.physics
def test_steady_drift_temperature_matches_equation_52() -> None:
    count = 200_000
    gas_temperature = 300.0
    gas_mass = 28.0 * atomic_mass
    drift_speed = 700.0
    ion_temperature = steady_drift_temperature_k(
        gas_temperature,
        gas_mass,
        drift_speed,
    )
    expected = gas_temperature + gas_mass * drift_speed**2 / (
        3.0 * Boltzmann
    )
    assert ion_temperature == pytest.approx(expected, rel=2.0e-15)
    inputs = _core_inputs(count)
    inputs["ion_velocity_m_per_s"][:] = [0.0, 0.0, drift_speed]
    inputs["ion_temperature_k"][:] = ion_temperature
    delta_u = apply_iict_collision_core(
        **inputs,
        rng=np.random.default_rng(4),
    ).transferred_internal_energy_j
    standard_error = float(np.std(delta_u) / np.sqrt(count))
    assert abs(float(np.mean(delta_u))) < 4.0 * standard_error


@pytest.mark.physics
def test_stationary_gas_long_run_reaches_thermal_limit() -> None:
    template = IonTemplate(mass_kg=5000.0 * atomic_mass, num_atoms=72)
    backend = build_collision_physics_backend(
        config=_lite_config(iict_pseudoatom_mass_da=84.0),
        template=template,
    )
    arguments = _batch_arguments(
        template,
        count=3000,
        temperature_k=900.0,
        speed_m_per_s=1600.0,
    )
    velocities = arguments.pop("velocities_m_per_s")
    temperatures = arguments.pop("internal_temperatures_k")
    rng = np.random.default_rng(5)
    for _ in range(1800):
        velocities, temperatures, _ = backend.apply_collision_batch(
            velocities_m_per_s=velocities,
            internal_temperatures_k=temperatures,
            rng=rng,
            **arguments,
        )
    mean_ke = 0.5 * template.mass_kg * np.mean(
        np.sum(velocities**2, axis=1)
    )
    assert mean_ke == pytest.approx(1.5 * Boltzmann * 300.0, rel=0.08)
    assert float(np.mean(temperatures)) == pytest.approx(300.0, rel=0.03)


def test_equations_54_to_58_and_zero_drift_thermal_rate() -> None:
    gas_mass = 28.0 * atomic_mass
    thermal_mean = np.sqrt(8.0 * Boltzmann * 300.0 / (np.pi * gas_mass))
    assert mean_relative_speed_m_per_s(0.0, 300.0, gas_mass) == pytest.approx(
        thermal_mean
    )
    frequency = collision_frequency_s_inverse(
        ion_speed_m_per_s=0.0,
        gas_temperature_k=300.0,
        gas_mass_kg=gas_mass,
        gas_number_density_m3=2.0e20,
        collision_cross_section_m2=1.0e-18,
    )
    assert frequency == pytest.approx(2.0e20 * 1.0e-18 * thermal_mean)
    assert mean_collision_waiting_time_s(frequency) == pytest.approx(
        1.0 / frequency
    )
    assert mean_free_path_m(0.0, frequency) == 0.0


def _initialize_stationary_cloud(cloud: IonCloud3D) -> None:
    zeros = np.zeros(1, dtype=np.float64)
    cloud.x.from_numpy(zeros)
    cloud.y.from_numpy(zeros)
    cloud.z.from_numpy(np.full(1, 5.0e-4))
    cloud.vx.from_numpy(zeros)
    cloud.vy.from_numpy(zeros)
    cloud.vz.from_numpy(zeros)
    cloud.q.from_numpy(np.ones(1))
    cloud.m.from_numpy(np.full(1, 5000.0 * atomic_mass))
    cloud.active.from_numpy(np.ones(1, dtype=np.int32))


@pytest.mark.physics
def test_iict_runtime_scheduler_uses_thermal_relative_speed() -> None:
    _ensure_pic_taichi_initialized("cpu")
    grid = UnifiedGrid2D(3, 3, 1.0e-3, 0.0, 1.0e-3)
    grid.bake_zero_static_fields(
        background_pressure_pa=100.0,
        background_temperature_k=300.0,
    )
    gas_mass = 28.0 * atomic_mass
    pusher = ParticlePusher(
        grid,
        collision_rate_model_code=1,
        gas_molecule_mass_kg=gas_mass,
    )
    cloud = IonCloud3D(1)
    _initialize_stationary_cloud(cloud)
    pusher.prepare_dt_candidates(
        cloud, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-18,
        Boltzmann, 0.1, 0.0, 0.05, 0.2, 1.0e-3, 1.0e-12,
        1.0, 0, 0.0, 0.0, 0.05, 1.0e-9,
    )
    rate = float(cloud.collision_rate_hz.to_numpy()[0])
    number_density = 100.0 / (Boltzmann * 300.0)
    expected = collision_frequency_s_inverse(
        ion_speed_m_per_s=0.0,
        gas_temperature_k=300.0,
        gas_mass_kg=gas_mass,
        gas_number_density_m3=number_density,
        collision_cross_section_m2=1.0e-18,
    )
    assert rate == pytest.approx(expected, rel=3.0e-7)


def test_legacy_ionspa_adapter_remains_compatible() -> None:
    template = IonTemplate()
    config = SimulationConfig(
        collision_physics_backend="ionspa",
        ionspa_backend="approximate",
    )
    with pytest.warns(RuntimeWarning):
        common = build_collision_physics_backend(config=config, template=template)
    with pytest.warns(RuntimeWarning):
        legacy = IonSpaCollisionAdapter(backend="approximate")
    assert isinstance(common, CollisionPhysicsBackend)
    arguments = _batch_arguments(
        template,
        count=2,
        temperature_k=350.0,
        speed_m_per_s=200.0,
    )
    common_output = common.apply_collision_batch(
        **arguments,
        rng=np.random.default_rng(700),
    )
    legacy_output = legacy.apply_collision_batch(
        **arguments,
        rng=np.random.default_rng(700),
    )
    for actual, expected in zip(common_output, legacy_output):
        np.testing.assert_array_equal(actual, expected)
    assert common.runtime_info["collision_physics_backend"] == "ionspa"


@pytest.mark.physics
def test_small_iict_simulation_smoke_and_report_metadata(
    tmp_path: Path,
) -> None:
    config = replace(
        _lite_config(),
        ion_count=2,
        total_time_s=1.0e-9,
        macro_time_step_s=1.0e-9,
        domain_radius_m=2.0e-3,
        domain_length_m=5.0e-3,
        grid_nr=5,
        grid_nz=7,
        pic_space_charge_scale=0.0,
        rf_frequency_hz=0.0,
        rf_peak_voltage_v=0.0,
        initial_position_jitter_m=0.0,
        initial_velocity_jitter_m_per_s=0.0,
    )
    template = IonTemplate(
        mass_kg=5000.0 * atomic_mass,
        num_atoms=72,
        charge_state=1,
        collision_cross_section_m2=1.0e-30,
        initial_position_m=np.array([0.0, 0.0, 1.0e-3]),
        initial_velocity_m_per_s=np.zeros(3),
    )
    simulation = FlowchartPicSimulation(template, config, particle_backend="cpu")
    simulation.replace_static_fields(
        lambda grid: grid.bake_zero_static_fields(
            background_pressure_pa=0.0,
            background_temperature_k=300.0,
        )
    )
    result = simulation.run()
    assert result.final_time_s == pytest.approx(1.0e-9)
    assert simulation.adapter.runtime_info["effective_backend"] == "iict-lite"
    assert result.collision_count == 0
    snapshot = capture_report_snapshot(
        argv=["--collision-physics-backend", "iict-lite"],
        requested_config=config,
        template=template,
        simulation=simulation,
        result=result,
    )
    summary = build_summary(
        snapshot,
        tmp_path / "final_particles.csv",
        tmp_path / "terminal_events.csv",
        {},
    )
    physics = summary["collision_physics"]
    assert physics["effective_backend"] == "iict-lite"
    assert physics["parameter_sources"]["pseudoatom_mass.mass_da"] == (
        "simulation_config"
    )
    assert "Independent paper-driven approximation" in (
        physics["equivalence_claim"]
    )


def test_invalid_core_state_fails_closed() -> None:
    inputs = _core_inputs(1)
    inputs["ion_mass_kg"][0] = 0.0
    with pytest.raises(ValueError, match="masses must be positive"):
        apply_iict_collision_core(**inputs, rng=np.random.default_rng(1))
    inputs = _core_inputs(1)
    inputs["ion_velocity_m_per_s"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        apply_iict_collision_core(**inputs, rng=np.random.default_rng(1))
