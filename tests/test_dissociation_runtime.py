"""End-to-end checks of scheduling, localized exits and continuous sources."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import elementary_charge

from src.config import IonTemplate, SimulationConfig
from src.core.simulation import FlowchartPicSimulation
from src.data.logger import DataLogger


pytestmark = pytest.mark.physics


def _simulation(mode="transport", initial_temperature_k=800, **overrides):
    config = SimulationConfig(
        ion_count=4, total_time_s=4e-6, macro_time_step_s=2e-6,
        max_time_step_s=2e-6, domain_radius_m=.002, domain_length_m=.01,
        grid_nr=5, grid_nz=7, pic_space_charge_scale=0,
        pic_poisson_backend="sparse_direct", detector_z_m=.00125,
        rf_peak_voltage_v=0, rf_frequency_hz=0,
        initial_position_jitter_m=0, initial_velocity_jitter_m_per_s=0,
        macro_particle_weight=1000, fragmentation_mode=mode,
        terminal_event_mode="all", collision_batch_size=0,
        collision_physics_backend="iict-lite",
        iict_heat_capacity_model="classical", iict_num_atoms=77,
        iict_pseudoatom_model="constant", iict_pseudoatom_mass_da=30,
        iict_fragmentation_model="eyring", iict_delta_h_kj_per_mol=111.7,
        iict_delta_s_j_per_mol_k=-38.1,
    )
    config = replace(config, **overrides)
    template = IonTemplate(
        name="leucine_enkephalin_MH", charge_state=1, num_atoms=77,
        mass_kg=556.27 * 1.66053906892e-27, collision_cross_section_m2=1.62e-18,
        initial_internal_temperature_k=initial_temperature_k, delta_h_kj_per_mol=111.7,
        delta_s_j_per_mol_k=-38.1, initial_position_m=np.array([0, 0, .001]),
        initial_velocity_m_per_s=np.array([0, 0, 250.0]),
    )
    sim = FlowchartPicSimulation(template, config, particle_backend="cpu")
    for name in ("E_dc_r", "E_dc_z", "E_rf_r", "E_rf_z", "P_gas", "v_gas_r"):
        getattr(sim.static_grid, name).fill(0)
    sim.static_grid.T_gas.fill(300)
    sim.static_grid.v_gas_z.fill(250)
    sim.field_sampler.refresh_all()
    return sim


@pytest.mark.parametrize("mode", ["transport", "loss", "off"])
def test_exit_weight_uses_localized_exit_time_and_is_conserved(mode):
    sim = _simulation(mode)
    result = sim.run()
    assert result.termination_reason == "all_particles_terminal"
    assert result.collision_count == 0
    rows = sim.terminal_event_recorder.rows
    exits = [r for r in rows if r["status"] == "z_exit"]
    assert len(exits) == 4
    rate = sim.adapter.fragmentation_rate(800, sim.template)
    expected_hazard = 0 if mode == "off" else rate * 1e-6
    for row in exits:
        assert row["event_time_s"] == pytest.approx(1e-6)
        assert row["dissociation_hazard"] == pytest.approx(expected_hazard)
        assert row["parent_real_ions_remaining"] == pytest.approx(1000 * np.exp(-expected_hazard))
    assert sum(r["represented_real_ions"] for r in rows) == pytest.approx(4000)


def test_transport_hazard_does_not_change_motion_collisions_or_rng():
    config = dict(ion_count=16, total_time_s=20e-9, macro_time_step_s=10e-9,
                  max_time_step_s=2e-9, detector_z_m=.009)
    sims = [_simulation(mode, **config) for mode in ("transport", "off")]
    results = []
    rate_calls = []
    original_rate = sims[0].adapter.fragmentation_rate
    def counted_rate(temperatures, template):
        rate_calls.append(len(temperatures))
        return original_rate(temperatures, template)
    sims[0].adapter.fragmentation_rate = counted_rate
    for sim in sims:
        sim.static_grid.P_gas.fill(200)
        sim.field_sampler.refresh_all()
        results.append(sim.run())
    assert results[0].collision_count == results[1].collision_count > 0
    assert results[0].micro_step_count == results[1].micro_step_count
    assert 0 < len(rate_calls) <= results[0].micro_step_count
    diagnostics = sims[0].dissociation_runtime_info
    assert diagnostics["dissociation_rate_cache_hits"] > 0
    assert (diagnostics["dissociation_rate_evaluated_particles"]
            + diagnostics["dissociation_rate_cache_hits"]
            == 16 * results[0].micro_step_count)
    assert sum(rate_calls) == diagnostics["dissociation_rate_evaluated_particles"]
    for field in results[0].final_state:
        if field != "dissociation_hazard":
            np.testing.assert_array_equal(results[0].final_state[field], results[1].final_state[field])
    np.testing.assert_array_equal(sims[0].rng.random(8), sims[1].rng.random(8))
    assert np.all(sims[0]._particle_parent_weight < sims[1]._particle_parent_weight)


def test_continuous_injection_and_slot_reuse_start_exposure_at_birth():
    sim = _simulation(
        source_mode="continuous-current", total_time_s=10e-6,
        ion_current_a=elementary_charge * 1000 / 2e-6,
        source_axial_velocity_m_per_s=250, max_macro_particle_weight=1000,
        macro_particles_per_injection=1, capillary_exit_z_m=.001,
        capillary_prefill_length_m=0,
    )
    result = sim.run()
    assert result.termination_reason != "numerical_failure"
    rows = sim.terminal_event_recorder.rows
    assert len(rows) >= 3
    assert len({r["track_id"] for r in rows}) == len(rows)
    expected_hazard = sim.adapter.fragmentation_rate(800, sim.template) * 1e-6
    for row in rows:
        assert row["dissociation_hazard"] == pytest.approx(expected_hazard)
        assert row["tof_s"] == pytest.approx(1e-6)
    assert sim._next_track_id > sim.config.ion_count


def test_legacy_snapshot_columns_are_padded_without_changing_them(tmp_path):
    logger = DataLogger(tmp_path, make_plots=False, representative_trajectory_count=0)
    original = np.arange(36, dtype=float).reshape(2, 18)
    logger.log_snapshot(1, 1e-6, original)
    logger.close()
    data = np.load(tmp_path / "snapshot_macro_00001.npy")
    np.testing.assert_array_equal(data[:, :18], original)
    assert np.all(np.isnan(data[:, 18]))


@pytest.mark.parametrize("terminal_mode", ["all", "transport-only"])
def test_depleted_final_state_keeps_last_pushed_position(terminal_mode):
    sim = _simulation("loss", initial_temperature_k=2000, ion_count=1,
                      detector_z_m=.009, total_time_s=1e-5, macro_time_step_s=1e-5,
                      max_time_step_s=1e-5, terminal_event_mode=terminal_mode)
    result = sim.run()
    assert result.termination_reason == "all_particles_terminal"
    assert result.final_state["active"][0] == 0
    expected_z = .001 + 250 * sim._particle_tof_s[0]
    assert result.final_state["positions"][0, 2] == pytest.approx(expected_z)
    if terminal_mode == "all":
        row = sim.terminal_event_recorder.rows[-1]
        assert row["z_m"] == pytest.approx(expected_z)
        assert row["event_time_s"] == pytest.approx(sim._particle_tof_s[0])


def test_joint_hybrid_updates_temperature_before_the_single_hazard_exposure():
    duration = 20e-9
    sim = _simulation("transport", ion_count=64, total_time_s=duration,
        macro_time_step_s=duration, max_time_step_s=duration, detector_z_m=.009,
        collision_model="hybrid-langevin", langevin_z_start_m=0,
        langevin_z_end_m=.01, langevin_max_dt_s=duration)
    sim.static_grid.P_gas.fill(200)
    sim.field_sampler.refresh_all()
    result = sim.run()
    assert result.micro_step_count == 1
    assert sim.dissociation_runtime_info["hybrid_joint_binary_events"] > 0
    assert sim.dissociation_runtime_info["hybrid_joint_gaussian_steps"] == 0
    assert np.any(result.final_state["temperature"] != 800)
    expected = sim.adapter.fragmentation_rate(result.final_state["temperature"], sim.template) * duration
    np.testing.assert_allclose(sim._particle_dissociation_hazard, expected)
    np.testing.assert_array_equal(result.final_state["weight"], 1000)
