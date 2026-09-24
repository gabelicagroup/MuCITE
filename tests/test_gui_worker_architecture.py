"""Architecture and behavior contracts for split GUI worker adapters."""

from __future__ import annotations

import ast
import hashlib
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core import simulation as core_simulation
from src.render.cli import runner as cli_runner
from src.render.gui import (
    beam_tasks,
    cli_preview,
    config_adapter,
    field_tasks,
    simulation_tasks,
    worker_lifecycle,
    workers,
)
from src.render.gui.models import AppConfig, BeamConfig


WORKER_MODULES = (
    "workers.py",
    "worker_lifecycle.py",
    "config_adapter.py",
    "cli_preview.py",
    "beam_tasks.py",
    "field_tasks.py",
    "simulation_tasks.py",
)

BEAM_PREVIEW_HASHES = {
    "positions_m": "d45a5510e430f83288122c9062914f412d5573bee5fdc960c3efe3f7ef0876a5",
    "velocities_m_per_s": "95ce83286ac03264fc0054fdc70c27018a917106f72adc8a59603778b99088ac",
    "kinetic_energy_ev": "4edc8607a34ef21973d61cef4a44fad62a381819efcd177aeeec5ea0fec2e3c2",
    "angle_deg": "9637ec95c3a7fd582e471bba902a4e1417d349c042328475e898234195cbf9d5",
}


def _gui_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "render" / "gui"


def _definition_spans(tree: ast.AST) -> list[tuple[str, int, str]]:
    return [
        (
            node.name,
            int(node.end_lineno - node.lineno + 1),
            type(node).__name__,
        )
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _import_module(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    return "" if node.module is None else node.module


def test_worker_modules_obey_granularity_and_avoid_main_cycle() -> None:
    for filename in WORKER_MODULES:
        source = (_gui_root() / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert len(source.splitlines()) <= 500, filename
        for name, span, kind in _definition_spans(tree):
            limit = 300 if kind == "ClassDef" else 50
            assert span <= limit, f"{filename}:{name} spans {span} lines"
        imports = [
            _import_module(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not any(
            imported == "main" or imported.endswith(".main")
            for imported in imports
        ), (filename, imports)


def test_workers_is_thin_facade_and_offline_imports_remain_lazy() -> None:
    facade_tree = ast.parse(
        (_gui_root() / "workers.py").read_text(encoding="utf-8")
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in facade_tree.body
    )
    for filename in ("config_adapter.py", "field_tasks.py"):
        tree = ast.parse(
            (_gui_root() / filename).read_text(encoding="utf-8")
        )
        top_level_imports = [
            _import_module(node)
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not any(
            "physics" in imported or "tools" in imported
            for imported in top_level_imports
        ), (filename, top_level_imports)


def test_worker_package_exports_preserve_canonical_object_identity() -> None:
    expected = {
        "TaskStopped": worker_lifecycle.TaskStopped,
        "WorkerHandle": worker_lifecycle.WorkerHandle,
        "_post": worker_lifecycle._post,
        "start_worker": worker_lifecycle.start_worker,
        "direction_vector": config_adapter.direction_vector,
        "build_ion_template": config_adapter.build_ion_template,
        "build_simulation_config": config_adapter.build_simulation_config,
        "gui_run_arguments": cli_preview.gui_run_arguments,
        "sample_beam_phase_space": beam_tasks.sample_beam_phase_space,
        "run_beam_smoke_task": beam_tasks.run_beam_smoke_task,
        "run_field_bake_task": field_tasks.run_field_bake_task,
        "_emit_latest_snapshot": simulation_tasks._emit_latest_snapshot,
        "run_simulation_task": simulation_tasks.run_simulation_task,
        "run_report_task": simulation_tasks.run_report_task,
        "build_demo_simulation": cli_runner.build_demo_simulation,
    }
    for name, canonical in expected.items():
        assert getattr(workers, name) is canonical


def test_workers_preserves_public_and_private_import_surface() -> None:
    expected_names = {
        "queue",
        "threading",
        "traceback",
        "asdict",
        "dataclass",
        "replace",
        "Path",
        "Any",
        "Callable",
        "Optional",
        "np",
        "elementary_charge",
        "ATOMIC_MASS_CONSTANT",
        "PROJECT_ROOT",
        "IonTemplate",
        "SimulationConfig",
        "make_ion_template",
        "make_simulation_config",
        "build_demo_simulation",
        "ReportSnapshot",
        "_write_simulation_report",
        "capture_report_snapshot",
        "DataLogger",
        "sample_source_xy_offsets",
        "AppConfig",
        "BeamConfig",
        "TaskStopped",
        "WorkerHandle",
        "_post",
        "start_worker",
        "direction_vector",
        "build_ion_template",
        "build_simulation_config",
        "gui_run_arguments",
        "sample_beam_phase_space",
        "run_beam_smoke_task",
        "run_field_bake_task",
        "_emit_latest_snapshot",
        "run_simulation_task",
        "run_report_task",
    }
    assert expected_names <= set(vars(workers))


def test_cli_preview_preserves_complete_default_argument_order() -> None:
    app_config = AppConfig()
    app_config.loaded_baked_field_path = "fields/main.npy"
    app_config.beam.source_birth_velocity_gas_csv = "gas/source.csv"
    app_config.runtime.electrode_mask_cache_path = "cache/mask.npz"
    app_config.output_dir = "out/session"
    arguments = workers.gui_run_arguments(app_config)
    assert arguments[:2] == ["-m", "src"]
    assert arguments[arguments.index("--ion-name") + 1] == "demonstration_ion"
    assert arguments[arguments.index("--heat-capacity-profile") + 1] == "peptide"
    assert arguments[arguments.index("--pic-poisson-backend") + 1] == "amg"
    assert "--pic-amg-tolerance" in arguments
    assert "--pic-poisson-warm-start" in arguments
    assert not any(argument.startswith("--sor-") for argument in arguments)


def test_seeded_beam_preview_preserves_random_phase_space() -> None:
    beam = BeamConfig(
        particle_count=5,
        beam_radius_mm=0.2,
        source_profile="gaussian",
        source_gaussian_sigma_mm=0.05,
        initial_position_jitter_mm=0.01,
        direction_axis="-y",
        cone_half_angle_deg=12.0,
        velocity_jitter_m_per_s=3.0,
        kinetic_energy_ev=8.0,
    )
    sample = workers.sample_beam_phase_space(beam)
    hashes = {
        key: hashlib.sha256(
            np.ascontiguousarray(value).view(np.uint8)
        ).hexdigest()
        for key, value in sample.items()
    }
    assert hashes == BEAM_PREVIEW_HASHES


def test_worker_lifecycle_preserves_message_order() -> None:
    messages: queue.Queue[dict[str, object]] = queue.Queue()

    def target(
        message_queue: queue.Queue[dict[str, object]],
        _stop_event: threading.Event,
        value: int,
    ) -> None:
        worker_lifecycle._post(message_queue, "payload", value=value)

    handle = workers.start_worker(messages, "example", target, 7)
    handle.thread.join(timeout=5.0)
    assert not handle.thread.is_alive()
    received = [messages.get_nowait() for _ in range(messages.qsize())]
    assert [message["type"] for message in received] == [
        "task_started",
        "payload",
        "task_done",
    ]
    assert received[0]["kind"] == "example"
    assert received[1]["value"] == 7


@dataclass
class _Progress:
    macro_step: int = 3
    time_s: float = 2.0e-6


class _Simulation:
    backend_message = "backend selected"

    def run(self, *, progress_callback: object) -> object:
        progress_callback(_Progress())  # type: ignore[operator]
        return self.result


class _CancellableSimulation:
    backend_message = ""

    def __init__(self) -> None:
        self.cancel_requested = threading.Event()
        self.request_count = 0
        self.result = SimpleNamespace(termination_reason="cancelled")

    def request_cancel(self) -> None:
        self.request_count += 1
        self.cancel_requested.set()

    def run(self, *, progress_callback: object) -> object:
        if not self.cancel_requested.wait(timeout=2.0):
            raise TimeoutError("GUI cancellation was not relayed")
        progress_callback(_Progress())  # type: ignore[operator]
        return self.result


def test_simulation_task_preserves_capture_report_and_message_protocol(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    app_config = AppConfig()
    app_config.output_dir = str(tmp_path)
    app_config.output.save_snapshots = False
    app_config.output.generate_report_after_run = True
    template = object()
    config = object()
    result = object()
    report_snapshot = object()
    simulation = _Simulation()
    simulation.result = result
    calls: dict[str, object] = {}

    monkeypatch.setattr(simulation_tasks, "build_ion_template", lambda _beam: template)
    monkeypatch.setattr(simulation_tasks, "build_simulation_config", lambda _app: config)
    monkeypatch.setattr(
        simulation_tasks,
        "gui_run_arguments",
        lambda _app: ["-m", "src", "--ion-count", "1"],
    )

    def build_demo(**kwargs: object) -> _Simulation:
        calls["build"] = kwargs
        return simulation

    def capture(**kwargs: object) -> object:
        calls["capture"] = kwargs
        return report_snapshot

    def write_report(**kwargs: object) -> dict[str, Path]:
        calls["report"] = kwargs
        return {"summary": tmp_path / "simulation_summary.json"}

    monkeypatch.setattr(simulation_tasks, "build_demo_simulation", build_demo)
    monkeypatch.setattr(simulation_tasks, "capture_report_snapshot", capture)
    monkeypatch.setattr(simulation_tasks, "_write_simulation_report", write_report)
    messages: queue.Queue[dict[str, object]] = queue.Queue()
    workers.run_simulation_task(messages, threading.Event(), app_config)
    received = [messages.get_nowait() for _ in range(messages.qsize())]

    assert [message["type"] for message in received] == [
        "log",
        "log",
        "simulation_progress",
        "simulation_finished",
    ]
    assert received[0]["text"] == "backend selected"
    assert received[1]["text"] == "CLI preview: python -m src --ion-count 1"
    assert received[2]["progress"] == {"macro_step": 3, "time_s": 2.0e-6}
    assert calls["build"]["data_logger"] is None
    assert calls["capture"]["argv"] == ["--ion-count", "1"]
    assert calls["capture"]["simulation"] is simulation
    assert calls["capture"]["result"] is result
    assert calls["report"]["snapshot"] is report_snapshot
    assert received[3]["report_paths"] == {
        "summary": str(tmp_path / "simulation_summary.json")
    }


def _patch_task_inputs(
    monkeypatch: object,
    simulation: object,
    captured: list[dict[str, object]],
) -> None:
    monkeypatch.setattr(simulation_tasks, "build_ion_template", lambda _beam: object())
    monkeypatch.setattr(simulation_tasks, "build_simulation_config", lambda _app: object())
    monkeypatch.setattr(
        simulation_tasks,
        "build_demo_simulation",
        lambda **_kwargs: simulation,
    )
    monkeypatch.setattr(
        simulation_tasks,
        "gui_run_arguments",
        lambda _app: ["-m", "src"],
    )
    monkeypatch.setattr(
        simulation_tasks,
        "capture_report_snapshot",
        lambda **kwargs: captured.append(kwargs) or object(),
    )


def test_simulation_stop_is_cooperative_and_cancelled_result_is_captured(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    app_config = AppConfig()
    app_config.output_dir = str(tmp_path)
    app_config.output.save_snapshots = False
    app_config.output.generate_report_after_run = False
    simulation = _CancellableSimulation()
    captured: list[dict[str, object]] = []
    releases: list[bool] = []
    _patch_task_inputs(monkeypatch, simulation, captured)
    monkeypatch.setattr(
        core_simulation,
        "release_pic_taichi_runtime",
        lambda: releases.append(True),
    )
    stop_event = threading.Event()
    stop_event.set()

    messages: queue.Queue[dict[str, object]] = queue.Queue()
    simulation_tasks.run_simulation_task(messages, stop_event, app_config)

    assert simulation.request_count >= 2
    assert captured[0]["result"].termination_reason == "cancelled"
    assert releases == [True]
    assert not any(
        thread.name == "mucite-cancel-relay" for thread in threading.enumerate()
    )


class _FailingSimulation:
    backend_message = ""

    def request_cancel(self) -> None:
        pass

    def run(self, *, progress_callback: object) -> object:
        del progress_callback
        raise RuntimeError("simulation failed")


def test_simulation_runtime_cleanup_runs_after_failure(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    app_config = AppConfig()
    app_config.output_dir = str(tmp_path)
    app_config.output.save_snapshots = False
    captured: list[dict[str, object]] = []
    releases: list[bool] = []
    _patch_task_inputs(monkeypatch, _FailingSimulation(), captured)
    monkeypatch.setattr(
        core_simulation,
        "release_pic_taichi_runtime",
        lambda: releases.append(True),
    )

    with pytest.raises(RuntimeError, match="simulation failed"):
        simulation_tasks.run_simulation_task(
            queue.Queue(),
            threading.Event(),
            app_config,
        )

    assert releases == [True]
    assert not any(
        thread.name == "mucite-cancel-relay" for thread in threading.enumerate()
    )
