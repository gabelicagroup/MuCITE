"""Architecture contracts for source-model extraction."""

from __future__ import annotations

import ast
from pathlib import Path

from src.env.sources.continuous_runtime import ContinuousSourceRuntimeMixin
from src.env.sources.packet_runtime import PacketSourceRuntimeMixin
from src.env.sources.slot_runtime import SourceSlotRuntimeMixin
from src.core.simulation import FlowchartPicSimulation


SOURCE_RUNTIME_FILES = (
    Path("src/env/sources/access.py"),
    Path("src/env/sources/continuous_runtime.py"),
    Path("src/env/sources/packet_runtime.py"),
    Path("src/env/sources/runtime.py"),
    Path("src/env/sources/slot_runtime.py"),
)


def test_source_runtime_methods_are_inherited_from_model_layer() -> None:
    assert (
        FlowchartPicSimulation.initialize_ions
        is PacketSourceRuntimeMixin.initialize_ions
    )
    assert (
        FlowchartPicSimulation._inject_continuous_source
        is ContinuousSourceRuntimeMixin._inject_continuous_source
    )
    assert (
        FlowchartPicSimulation._populate_capillary_slots
        is SourceSlotRuntimeMixin._populate_capillary_slots
    )


def test_source_runtime_obeys_size_budgets() -> None:
    for path in SOURCE_RUNTIME_FILES:
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 500, path
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.end_lineno is not None
                assert node.end_lineno - node.lineno + 1 <= 300, (
                    path,
                    node.name,
                )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno is not None
                assert node.end_lineno - node.lineno + 1 <= 50, (
                    path,
                    node.name,
                )


def test_source_model_does_not_import_control_or_output_layers() -> None:
    forbidden = (".core", ".data", ".render", ".simulation")
    for path in SOURCE_RUNTIME_FILES:
        source = path.read_text(encoding="utf-8")
        assert not any(name in source for name in forbidden), path
