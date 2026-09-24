"""Architecture and behavior contracts for environment boundary policies."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import (
    PARTICLE_ELECTRODE_HIT,
    PARTICLE_Z_EXIT,
    SimulationConfig,
)
from src.env.boundaries import BoundaryRuntimeMixin
from src.env.boundaries import electrodes as electrode_module
from src.env.boundaries.accounting import TerminalAccountingMixin
from src.env.boundaries.electrodes import ElectrodeSamplingMixin
from src.env.boundaries.events import TerminalEvents
from src.env.boundaries.localization import TerminalLocalizationMixin
from src.env.boundaries.runtime import TerminalBoundaryRuntimeMixin
from src.core.simulation import FlowchartPicSimulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DIR = PROJECT_ROOT / "src" / "env" / "boundaries"
TARGET_METHODS = {
    "_particle_tof_from_event_time",
    "_accumulate_terminal_accounting",
    "_record_terminal_events",
    "_sample_electrode_events",
    "_sample_electrode_event",
    "_electrode_segment_sample_spacing_m",
    "_find_swept_electrode_hits",
    "_localize_terminal_segments",
    "_taichi_runtime_apply_terminal_boundary_events",
    "_classify_domain_exit",
}


def test_simulation_inherits_boundary_methods_without_local_duplicates() -> None:
    tree = ast.parse(
        (PROJECT_ROOT / "src" / "core" / "simulation.py").read_text(
            encoding="utf-8"
        )
    )
    simulation_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FlowchartPicSimulation"
    )
    local_methods = {
        node.name
        for node in simulation_class.body
        if isinstance(node, ast.FunctionDef)
    }
    assert local_methods.isdisjoint(TARGET_METHODS)
    assert issubclass(FlowchartPicSimulation, BoundaryRuntimeMixin)


def test_inherited_boundary_methods_preserve_unbound_function_identity() -> None:
    expected_owners = {
        "_particle_tof_from_event_time": TerminalAccountingMixin,
        "_accumulate_terminal_accounting": TerminalAccountingMixin,
        "_record_terminal_events": TerminalAccountingMixin,
        "_sample_electrode_events": ElectrodeSamplingMixin,
        "_sample_electrode_event": ElectrodeSamplingMixin,
        "_electrode_segment_sample_spacing_m": ElectrodeSamplingMixin,
        "_find_swept_electrode_hits": ElectrodeSamplingMixin,
        "_localize_terminal_segments": TerminalLocalizationMixin,
        "_taichi_runtime_apply_terminal_boundary_events": (
            TerminalBoundaryRuntimeMixin
        ),
        "_classify_domain_exit": TerminalBoundaryRuntimeMixin,
    }
    for method_name, owner in expected_owners.items():
        assert getattr(FlowchartPicSimulation, method_name) is getattr(
            owner,
            method_name,
        )


def test_boundary_modules_meet_layer_and_granularity_contracts() -> None:
    forbidden_parts = {"simulation", "data", "render", "report", "reporters"}
    for path in BOUNDARY_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert len(source.splitlines()) <= 500, path
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(
            set(module.split(".")) & forbidden_parts
            for module in imports
        ), path
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.end_lineno - node.lineno + 1 <= 300, (
                    path,
                    node.name,
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 50, (
                    path,
                    node.name,
                )


def test_particle_tof_keeps_birth_time_and_nonnegative_semantics() -> None:
    owner = SimpleNamespace(
        _particle_birth_time_s=np.array([np.nan, 1.0, 4.0])
    )
    tof_s = FlowchartPicSimulation._particle_tof_from_event_time(
        owner,
        np.array([0, 1, 2], dtype=np.int32),
        np.array([3.0, 3.0, 3.0]),
    )
    np.testing.assert_array_equal(tof_s, [3.0, 2.0, 0.0])


def test_electrode_sampling_keeps_3d_mask_priority(monkeypatch) -> None:
    monkeypatch.setattr(
        electrode_module,
        "sample_electrode_mask3d",
        lambda _mask, _positions: (
            np.array([True, False]),
            np.array([7, 7], dtype=np.int32),
            np.array([-0.1, 1.0]),
            np.array([True, True]),
        ),
    )
    monkeypatch.setattr(
        electrode_module,
        "sample_electrode_mask",
        lambda _mask, _positions: (
            np.array([True, True]),
            np.array([2, 2], dtype=np.int32),
            np.array([-0.2, -0.2]),
            np.array([True, True]),
        ),
    )
    owner = SimpleNamespace(
        config=SimulationConfig(electrode_hit_distance_m=0.0),
        electrode_mask3d=object(),
        electrode_mask=object(),
    )
    hits, ids, distances_m = FlowchartPicSimulation._sample_electrode_events(
        owner,
        np.zeros((2, 3)),
    )
    np.testing.assert_array_equal(hits, [True, True])
    np.testing.assert_array_equal(ids, [7, 2])
    np.testing.assert_array_equal(distances_m, [-0.1, -0.2])


def test_equal_fraction_prefers_electrode_over_detector() -> None:
    owner = SimpleNamespace(
        config=SimulationConfig(
            detector_z_m=0.5,
            detector_radius_m=2.0,
            radial_limit_m=2.0,
            domain_radius_m=2.0,
            domain_length_m=2.0,
        ),
        _electrode_segment_sample_spacing_m=lambda: 0.1,
        _sample_electrode_event=lambda position: (
            float(position[2]) >= 0.5,
            11,
            -1.0e-6,
        ),
    )
    localized = FlowchartPicSimulation._localize_terminal_segments(
        owner,
        start_positions_m=np.array([[0.0, 0.0, 0.0]]),
        end_positions_m=np.array([[0.0, 0.0, 1.0]]),
        start_velocities_m_per_s=np.zeros((1, 3)),
        end_velocities_m_per_s=np.zeros((1, 3)),
        endpoint_status=np.array([PARTICLE_Z_EXIT], dtype=np.int16),
        endpoint_electrode_id=np.array([-1], dtype=np.int32),
        endpoint_surface_distance_m=np.array([np.nan]),
        step_start_time_s=2.0,
        step_dt_s=0.5,
    )
    assert localized[0][0] == PARTICLE_ELECTRODE_HIT
    assert localized[1][0, 2] == pytest.approx(0.5, abs=1.0e-8)
    assert localized[3][0] == pytest.approx(2.25, abs=1.0e-8)
    assert localized[4][0] == 11


def test_terminal_event_order_is_time_then_particle_index() -> None:
    events = TerminalEvents(
        indices=np.array([5, 3, 2], dtype=np.int32),
        statuses=np.array([1, 2, 3], dtype=np.int16),
        electrode_ids=np.full(3, -1, dtype=np.int32),
        surface_distances_m=np.full(3, np.nan),
        positions_m=np.zeros((3, 3)),
        velocities_m_per_s=np.zeros((3, 3)),
        event_times_s=np.array([1.0, 0.5, 0.5]),
    ).ordered()
    np.testing.assert_array_equal(events.indices, [2, 3, 5])
    np.testing.assert_array_equal(events.statuses, [3, 2, 1])
