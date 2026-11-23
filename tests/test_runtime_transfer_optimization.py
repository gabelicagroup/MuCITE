"""Sparse transfers and cache invalidation must preserve physical state."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.agents.slot_state import ParticleRuntimeState
from src.config import IonTemplate, SimulationConfig, SLOT_ACTIVE
from src.config.validation import validate_simulation_config
from src.core.timestep import InvalidTimeStepError, select_indexed_timestep
from src.env.boundaries.events import TerminalEvents
from src.env.boundaries.runtime import TerminalBoundaryRuntimeMixin
from src.env.collisions.dissociation import advance_dissociation, check_final_transport_survival
from src.env.collisions.runtime_common import CollisionStateMixin
from src.env.runtime.cloud_io import CloudStateIOMixin
from src.env.runtime.step_cache import ParticleStepCache


def test_zero_terminal_count_does_not_touch_event_arrays_or_motion():
    def forbidden(*args):
        raise AssertionError("Empty event path must not download particle arrays")
    owner = SimpleNamespace(cloud=SimpleNamespace(terminal_count={None: 0}),
                            _download_selected_motion_state=forbidden)
    events = TerminalBoundaryRuntimeMixin._download_terminal_endpoint_events(owner, 1.0)
    assert events.count == 0


def test_compact_timestep_ties_and_invalid_values_keep_original_slot_ids():
    indices = np.array([2, 500, 119999], dtype=np.int32)
    selected = select_indexed_timestep(np.array([1e-9, 1e-9, 2e-9]), indices)
    assert selected.dt_s == 1e-9 and selected.particle_index == 2
    with pytest.raises(InvalidTimeStepError) as caught:
        select_indexed_timestep(np.array([1e-9, np.nan, 0.0]), indices)
    np.testing.assert_array_equal(caught.value.particle_indices, [500, 119999])


def test_motion_cache_reads_missing_rows_and_invalidates_at_motion_boundaries():
    state = {"active": np.array([0, 1, 0, 1, 1], np.int32)}
    cache = ParticleStepCache(state)
    calls = []
    stage = [1.0]
    def gather(indices):
        calls.append(indices.copy())
        values = np.broadcast_to(indices[:, None] + stage[0], (indices.size, 3)).copy()
        return values, -values
    first, _ = cache.read(np.array([3], np.int32), gather)
    first[:] = -999
    second, _ = cache.read(np.array([4, 3, 1], np.int32), gather)
    np.testing.assert_array_equal(second[:, 0], [5, 4, 2])
    np.testing.assert_array_equal(calls[1], [4, 1])
    cache.invalidate_motion()
    stage[0] = 10
    final, _ = cache.read(np.array([3], np.int32), gather)
    assert final[0, 0] == 13 and len(calls) == 3


def test_microstep_cache_is_discarded_after_release_and_on_scope_exception():
    owner = CloudStateIOMixin()
    owner._particle_weight = np.ones(3)
    owner._particle_parent_weight = np.ones(3)
    owner._particle_phase_codes = np.full(3, SLOT_ACTIVE)
    state = {"active": np.ones(3, np.int32), "phase": np.full(3, SLOT_ACTIVE),
             "weight": np.ones(3)}
    with pytest.raises(RuntimeError), owner._micro_step_scope(state):
        first = owner._active_indices(state)
        assert first is owner._active_indices(state)
        owner._release_free_slot(state, 1)
        np.testing.assert_array_equal(owner._active_indices(state), [0, 2])
        raise RuntimeError("scope cleanup")
    assert owner._particle_step_cache is None


def test_motion_cache_allocation_is_lazy_and_stages_change_eligibility():
    owner = CloudStateIOMixin()
    owner.config = SimpleNamespace(fragmentation_mode="transport")
    state = {"active": np.ones(3, np.int32)}
    with owner._micro_step_scope(state):
        cache = owner._particle_step_cache
        assert not cache.motion_enabled and cache.positions_m is None
        owner._enable_motion_cache()
        assert cache.motion_enabled
        owner._invalidate_motion_cache(after_push=True)
        assert not cache.motion_enabled
        owner.electrode_mask = object()
        owner._invalidate_motion_cache(after_push=True)
        assert cache.motion_enabled


def test_full_population_cache_reuses_partial_reads_without_exposing_storage():
    cache = ParticleStepCache({"active": np.ones(4, np.int32)})
    calls = []
    def gather(indices):
        calls.append(indices.copy())
        rows = np.repeat(indices[:, None], 3, axis=1).astype(float)
        return rows, -rows
    cache.read(np.array([1, 3], np.int32), gather)
    positions, velocities = cache.read(cache.indices.copy(), gather)
    np.testing.assert_array_equal(calls[-1], [0, 2])
    positions[:] = -999
    velocities[:] = -999
    final, _ = cache.read(cache.indices, gather)
    np.testing.assert_array_equal(final[:, 0], [0, 1, 2, 3])
    assert len(calls) == 2


def _decay_runtime(every=100):
    template = IonTemplate()
    calls = []
    def rate(temperature, parameters):
        calls.append(np.asarray(temperature).copy())
        return np.asarray(temperature) * parameters.delta_h_kj_per_mol
    owner = SimpleNamespace(template=template,
        config=SimpleNamespace(fragmentation_mode="transport", dissociation_check_every=every),
        adapter=SimpleNamespace(fragmentation_rate=rate))
    ParticleRuntimeState.packet(3, template, macro_weight=1000).bind_to(owner)
    CollisionStateMixin._reset_collision_statistics(owner)
    state = {"active": np.ones(3, np.int32), "phase": np.full(3, SLOT_ACTIVE),
             "temperature": np.full(3, 800.0)}
    return owner, state, calls


def _decay_step(owner, state, step, events=None):
    advance_dissociation(owner, state, step_start_time_s=step * 1e-9,
                         step_dt_s=1e-9, events=events or TerminalEvents.empty())


def test_decay_rate_cache_refreshes_temperature_identity_template_and_provider():
    owner, state, calls = _decay_runtime()
    _decay_step(owner, state, 0)
    _decay_step(owner, state, 1)
    assert [len(call) for call in calls] == [3]
    state["temperature"][1] = 900
    _decay_step(owner, state, 2)
    owner._particle_track_id[2] += 10
    _decay_step(owner, state, 3)
    assert [len(call) for call in calls] == [3, 1, 1]
    owner.template = replace(owner.template, delta_h_kj_per_mol=120)
    _decay_step(owner, state, 4)
    assert len(calls[-1]) == 3
    owner.adapter.fragmentation_rate = lambda temperature, _: np.full_like(temperature, 7)
    before = owner._particle_dissociation_hazard.copy()
    _decay_step(owner, state, 5)
    np.testing.assert_allclose(owner._particle_dissociation_hazard - before, 7e-9)


@pytest.mark.parametrize("every", [0, 1, 100])
def test_audit_frequency_does_not_skip_decay_or_terminal_and_final_checks(every):
    owner, state, calls = _decay_runtime(every)
    for step in range(3):
        _decay_step(owner, state, step)
    before = owner.dissociation_runtime_info["transport_survival_checks"]
    event = TerminalEvents(indices=np.array([1], np.int32), statuses=np.array([1], np.int16),
        positions_m=np.zeros((1, 3)), velocities_m_per_s=np.zeros((1, 3)),
        event_times_s=np.array([3.5e-9]), electrode_ids=np.array([-1], np.int32),
        surface_distances_m=np.array([np.nan]))
    _decay_step(owner, state, 3, event)
    assert owner.dissociation_runtime_info["transport_survival_checks"] > before
    state["active"][1] = 0
    before = owner.dissociation_runtime_info["transport_survival_checks"]
    check_final_transport_survival(owner, state)
    assert owner.dissociation_runtime_info["transport_survival_checks"] == before + 2
    expected = 800 * owner.template.delta_h_kj_per_mol * np.array([4, 3.5, 4]) * 1e-9
    np.testing.assert_allclose(owner._particle_dissociation_hazard, expected)
    np.testing.assert_allclose(owner._particle_parent_weight, 1000 * np.exp(-expected))
    assert len(calls) == 1


@pytest.mark.parametrize("invalid", [-1, 0.5, True])
def test_audit_frequency_rejects_noninteger_or_negative_values(invalid):
    with pytest.raises(ValueError, match="dissociation_check_every"):
        validate_simulation_config(SimulationConfig(dissociation_check_every=invalid))


def test_audit_frequency_round_trips_through_cli_and_json(tmp_path):
    from src.config import dump_config_document, load_config_document
    from src.render.cli.parser import parse_cli
    from src.render.cli.config_adapter import resolve_cli_document
    parsed = parse_cli(["--dissociation-check-every", "0"])
    document = resolve_cli_document(parsed.namespace, parsed.explicit_dests)
    assert document.simulation.dissociation_check_every == 0
    path = tmp_path / "request.json"
    dump_config_document(path, document)
    assert load_config_document(path).simulation.dissociation_check_every == 0


@pytest.mark.physics
def test_device_gathers_use_only_selected_slots_and_terminal_prefix():
    import taichi as ti
    from src.agents.taichi_cloud import IonCloud3D
    from src.core.simulation import _ensure_pic_taichi_initialized
    from src.env.transport.time_step import TimeStepMixin
    _ensure_pic_taichi_initialized("cpu")
    @ti.data_oriented
    class Downloader(TimeStepMixin):
        pass
    cloud = IonCloud3D(1000)
    cloud.dt_candidate.from_numpy(np.arange(1000, dtype=np.float64) + 1)
    indices = np.array([999, 3, 42], np.int32)
    compact = Downloader().download_dt_diagnostics(cloud, indices)
    np.testing.assert_array_equal(compact[0], [1000, 4, 43])
    assert all(row.shape == (3,) for row in compact)
    cloud.terminal_count[None] = 2
    for j, slot in enumerate([999, 3]):
        cloud.terminal_indices[j] = slot
        cloud.terminal_status[j] = j + 1
        cloud.terminal_electrode_id[j] = 20 + j
        cloud.terminal_surface_distance[j] = j * 1e-6
        cloud.z[slot] = slot * 1e-6
    owner = CloudStateIOMixin()
    owner.cloud = cloud
    events = TerminalBoundaryRuntimeMixin._download_terminal_endpoint_events(owner, 4e-6)
    np.testing.assert_array_equal(events.indices, [999, 3])
    np.testing.assert_array_equal(events.statuses, [1, 2])
    np.testing.assert_array_equal(events.electrode_ids, [20, 21])
    np.testing.assert_array_equal(events.positions_m[:, 2], np.array([999, 3]) * 1e-6)
