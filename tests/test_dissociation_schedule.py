"""Physical-time dissociation and terminal-weight conservation contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.agents.slot_state import ParticleRuntimeState
from src.config import (
    IonTemplate, PARTICLE_FRAGMENTED, PARTICLE_Z_EXIT,
    SLOT_ACTIVE, SLOT_CAPILLARY, SLOT_FREE,
)
from src.env.boundaries.events import TerminalEvents
from src.env.collisions.dissociation import advance_dissociation, survival_update
from src.env.collisions.iict_lite.fragmentation import EyringFragmentationModel
from src.env.collisions.runtime_common import CollisionStateMixin
from src.env.runtime.cloud_io import CloudStateIOMixin
from src.env.sources.slot_runtime import SourceSlotRuntimeMixin


pytestmark = pytest.mark.physics


def _runtime(count=3, mode="transport", terminal_mode="all"):
    template = IonTemplate(delta_h_kj_per_mol=111.7, delta_s_j_per_mol_k=-38.1)
    model = EyringFragmentationModel(111.7, -38.1)
    runtime = SimpleNamespace(
        template=template,
        config=SimpleNamespace(fragmentation_mode=mode, terminal_event_mode=terminal_mode),
        adapter=SimpleNamespace(fragmentation_rate=lambda t, _: model.rate_s_inverse(t)),
        rng=np.random.default_rng(7), rows=[], omitted=[], weight_uploads=[],
        deactivated=[], summaries=[],
    )
    ParticleRuntimeState.packet(count, template, macro_weight=1000).bind_to(runtime)
    CollisionStateMixin._reset_collision_statistics(runtime)
    state = dict(active=np.ones(count, np.int32), phase=np.full(count, SLOT_ACTIVE),
                 weight=np.full(count, 1000.0), temperature=np.full(count, 800.0),
                 positions=np.zeros((count, 3)), velocities=np.zeros((count, 3)))
    runtime._record_terminal_events = lambda *a, **k: runtime.rows.append((a, k))
    runtime._accumulate_terminal_accounting = lambda *a, **k: runtime.summaries.append((a, k))
    runtime.terminal_event_recorder = SimpleNamespace(
        omit_summary=lambda *a, **k: runtime.omitted.append((a, k)))
    runtime._download_selected_motion_state = lambda idx: (
        state["positions"][idx].copy(), state["velocities"][idx].copy())
    runtime._particle_tof_from_event_time = lambda idx, t: np.asarray(t)
    runtime._release_free_slot = lambda s, i: CloudStateIOMixin._release_free_slot(runtime, s, i)
    runtime._upload_deactivated_indices = lambda idx: runtime.deactivated.extend(idx)
    runtime._next_track_ids = lambda n: SourceSlotRuntimeMixin._next_track_ids(runtime, n)
    runtime.cloud = SimpleNamespace(update_weights=lambda i, w, n: runtime.weight_uploads.append(w.copy()))
    return runtime, state, model


def _events(indices, times):
    indices = np.asarray(indices, dtype=np.int32)
    n = indices.size
    return TerminalEvents(
        indices=indices, statuses=np.full(n, PARTICLE_Z_EXIT, np.int16),
        positions_m=np.full((n, 3), 0.01), velocities_m_per_s=np.full((n, 3), 100),
        event_times_s=np.asarray(times, dtype=float),
        electrode_ids=np.full(n, -1, np.int32), surface_distances_m=np.full(n, np.nan),
    )


@pytest.mark.parametrize("steps", [1, 10, 200, 4000])
def test_hot_ion_dissociates_without_any_collision_and_is_dt_independent(steps):
    runtime, state, model = _runtime(count=1)
    duration_s = 200e-6
    for j in range(steps):
        advance_dissociation(runtime, state, step_start_time_s=j * duration_s / steps,
                             step_dt_s=duration_s / steps, events=TerminalEvents.empty())
    expected_hazard = model.rate_s_inverse(800.0) * duration_s
    assert runtime._particle_parent_weight[0] / 1000 == pytest.approx(
        np.exp(-expected_hazard), rel=1e-10)
    assert runtime._particle_dissociation_hazard[0] == pytest.approx(expected_hazard, rel=1e-10)
    np.testing.assert_array_equal(runtime._particle_collision_counts, 0)
    np.testing.assert_array_equal(state["temperature"], 800.0)
    np.testing.assert_array_equal(state["weight"], 1000.0)
    np.testing.assert_array_equal(runtime.rng.random(8), np.random.default_rng(7).random(8))


def test_birth_and_terminal_times_bound_exposure_and_buffer_is_excluded():
    runtime, state, model = _runtime(4)
    runtime._particle_birth_time_s[:] = [0, 1.2e-6, 0, 0]
    state["phase"][2] = SLOT_CAPILLARY
    state["active"][3] = 0
    events = _events([0, 1], [1.5e-6, 1.75e-6])
    result = advance_dissociation(runtime, state, step_start_time_s=1e-6,
                                  step_dt_s=1e-6, events=events)
    np.testing.assert_allclose(runtime._particle_dissociation_hazard,
                               model.rate_s_inverse(800) * np.array([.5e-6, .55e-6, 0, 0]))
    assert result is events
    assert not runtime.rows


def test_piecewise_heating_and_cooling_accumulate_the_full_history():
    runtime, state, model = _runtime(1)
    expected = 0.0
    start = 0.0
    for temperature, dt in [(500, 80e-6), (1500, 1e-6), (500, 100e-6)]:
        state["temperature"][:] = temperature
        advance_dissociation(runtime, state, step_start_time_s=start,
                             step_dt_s=dt, events=TerminalEvents.empty())
        expected += model.rate_s_inverse(temperature) * dt
        start += dt
    assert runtime._particle_parent_weight[0] / 1000 == pytest.approx(np.exp(-expected))


@pytest.mark.parametrize("terminal_mode", ["all", "transport-only"])
def test_loss_conserves_weight_and_does_not_count_exit_twice(terminal_mode):
    runtime, state, model = _runtime(1, mode="loss", terminal_mode=terminal_mode)
    end = 1e-6
    events = _events([0], [end])
    result = advance_dissociation(runtime, state, step_start_time_s=0,
                                  step_dt_s=2e-6, events=events)
    remaining = 1000 * np.exp(-model.rate_s_inverse(800) * end)
    assert state["weight"][0] == pytest.approx(remaining)
    assert runtime._particle_parent_weight[0] == pytest.approx(remaining)
    assert runtime._fragmented_real_ions_represented + remaining == pytest.approx(1000)
    assert result.count == 1
    assert not runtime.deactivated
    if terminal_mode == "all":
        _, payload = runtime.rows[0]
        assert payload["event_time_s"][0] == end
        assert payload["track_ids"][0] != runtime._particle_track_id[0]
        np.testing.assert_array_equal(payload["positions_m"], events.positions_m)
        assert payload["represented_real_ions"][0] + remaining == pytest.approx(1000)
    else:
        assert len(runtime.omitted) == len(runtime.summaries) == 1


def test_underflowed_loss_deactivates_and_removes_geometric_exit():
    runtime, state, _ = _runtime(1, mode="loss")
    runtime.adapter.fragmentation_rate = lambda t, _: np.full_like(t, 1e9)
    result = advance_dissociation(runtime, state, step_start_time_s=0,
                                  step_dt_s=2e-6, events=_events([0], [1e-6]))
    assert result.count == 0
    assert state["weight"][0] == state["active"][0] == 0
    assert state["phase"][0] == SLOT_FREE
    assert runtime._particle_status_codes[0] == PARTICLE_FRAGMENTED
    assert runtime.deactivated == [0]
    assert runtime.rows[0][1]["track_ids"][0] == 0
    assert runtime._fragmented_real_ions_represented == 1000


def test_off_does_not_evaluate_rates_or_mutate_state():
    runtime, state, _ = _runtime(1, mode="off")
    def forbidden(*_):
        raise AssertionError("rate evaluated in off mode")
    runtime.adapter.fragmentation_rate = forbidden
    advance_dissociation(runtime, state, step_start_time_s=0, step_dt_s=1,
                         events=TerminalEvents.empty())
    assert runtime._particle_parent_weight[0] == 1000
    assert runtime._particle_dissociation_hazard[0] == 0


def test_survival_preserves_tiny_remaining_weight_and_rejects_invalid_input():
    surviving, lost, hazard = survival_update(np.array([1000.0]), np.array([50.0]), np.array([1.0]))
    assert surviving[0] > 0.0
    assert surviving[0] == 1000 * np.exp(-50)
    assert lost[0] == 1000
    assert hazard[0] == 50
    for bad in [np.nan, np.inf, -1.0]:
        with pytest.raises(FloatingPointError):
            survival_update(np.array([1000]), np.array([bad]), np.array([1.0]))


def test_reused_and_activated_slots_reset_hazard():
    runtime, _, _ = _runtime(1)
    runtime._particle_dissociation_hazard[:] = 10
    SourceSlotRuntimeMixin._mark_slots_active(runtime, np.array([0]), 2.0)
    assert runtime._particle_dissociation_hazard[0] == 0
    assert runtime._particle_birth_time_s[0] == 2.0


def test_terminal_time_outside_step_is_rejected():
    runtime, state, _ = _runtime(1)
    with pytest.raises(FloatingPointError, match="outside"):
        advance_dissociation(runtime, state, step_start_time_s=0,
                             step_dt_s=1e-6, events=_events([0], [2e-6]))


@pytest.mark.parametrize("terminal_mode", ["all", "transport-only"])
def test_repeated_loss_releases_subnormal_residual_once(terminal_mode):
    runtime, state, _ = _runtime(1, mode="loss", terminal_mode=terminal_mode)
    runtime.adapter.fragmentation_rate = lambda t, _: np.full_like(t, 1e7)
    for i in range(8000):
        advance_dissociation(runtime, state, step_start_time_s=i * 1e-8,
                             step_dt_s=1e-8, events=TerminalEvents.empty())
    assert state["weight"][0] == state["active"][0] == 0
    assert runtime._particle_parent_weight[0] == 0
    assert runtime.deactivated == [0]
    assert runtime.dissociation_runtime_info["loss_underflow_depletions"] == 1
    assert runtime._fragmented_real_ions_represented == pytest.approx(1000, rel=1e-12)


@pytest.mark.parametrize("weight", [.1, 1e-20, 1e-250])
def test_small_normal_loss_weights_are_retained(weight):
    runtime, state, _ = _runtime(1, mode="loss")
    state["weight"][:] = runtime._particle_weight[:] = weight
    runtime._particle_parent_weight[:] = weight
    runtime.adapter.fragmentation_rate = lambda t, _: np.full_like(t, .1)
    advance_dissociation(runtime, state, step_start_time_s=0,
                         step_dt_s=1, events=TerminalEvents.empty())
    assert state["weight"][0] == weight * np.exp(-.1)
    assert state["active"][0] == 1
    assert not runtime.deactivated


def test_subnormal_loss_cleanup_accounts_for_residual_and_requires_exposure():
    runtime, state, _ = _runtime(1, mode="loss")
    original = np.finfo(np.float64).tiny
    state["weight"][:] = runtime._particle_weight[:] = original
    runtime._particle_parent_weight[:] = original
    runtime.adapter.fragmentation_rate = lambda t, _: np.full_like(t, .1)
    advance_dissociation(runtime, state, step_start_time_s=0,
                         step_dt_s=0, events=TerminalEvents.empty())
    assert state["weight"][0] == original
    assert not runtime.deactivated
    advance_dissociation(runtime, state, step_start_time_s=0,
                         step_dt_s=1, events=TerminalEvents.empty())
    assert runtime.rows[0][1]["represented_real_ions"][0] == original
    assert runtime._fragmented_real_ions_represented == original
    assert runtime.deactivated == [0]


def test_zero_exposure_preserves_subnormal_weight_bits():
    parents = np.array([1e-310, 1e-320, 1000.])
    for rates, times in [(np.ones(3), np.zeros(3)), (np.zeros(3), np.ones(3))]:
        remaining, lost, hazard = survival_update(parents, rates, times)
        np.testing.assert_array_equal(remaining.view(np.uint64), parents.view(np.uint64))
        np.testing.assert_array_equal(lost, 0)
        np.testing.assert_array_equal(hazard, 0)


@pytest.mark.parametrize("terminal_mode", ["all", "transport-only"])
@pytest.mark.parametrize("localized", [False, True])
def test_depleted_slot_keeps_endpoint_motion(terminal_mode, localized):
    runtime, state, _ = _runtime(1, mode="loss", terminal_mode=terminal_mode)
    runtime.adapter.fragmentation_rate = lambda t, _: np.full_like(t, 1e9)
    positions = np.array([[0., 0., .002]])
    velocities = np.array([[1., 2., 300.]])
    runtime._download_selected_motion_state = lambda i: (positions[i].copy(), velocities[i].copy())
    events = _events([0], [1e-6]) if localized else TerminalEvents.empty()
    advance_dissociation(runtime, state, step_start_time_s=0,
                         step_dt_s=2e-6, events=events)
    np.testing.assert_array_equal(state["positions"], events.positions_m if localized else positions)
    np.testing.assert_array_equal(state["velocities"], events.velocities_m_per_s if localized else velocities)
    assert runtime.deactivated == [0]


def test_transport_retains_total_weight_after_complete_precursor_dissociation():
    runtime, state, _ = _runtime(1, mode="transport")
    runtime.adapter.fragmentation_rate = lambda t, _: np.full_like(t, 1e9)
    advance_dissociation(runtime, state, step_start_time_s=0,
                         step_dt_s=2e-6, events=TerminalEvents.empty())
    assert runtime._particle_parent_weight[0] == 0
    assert state["weight"][0] == runtime._particle_weight[0] == 1000
    assert state["active"][0] == 1
    assert not runtime.deactivated
