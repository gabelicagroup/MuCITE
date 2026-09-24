"""Granularity and dependency contracts for report artifact writers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import src.data.report as report_package
from src.data.report.csv_tables import materialize_terminal_event_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "src" / "data" / "report"
SPLIT_MODULES = (
    "csv_tables.py",
    "plots.py",
    "summary_builder.py",
    "markdown.py",
    "writer.py",
)


def test_report_writer_modules_obey_granularity_limits() -> None:
    violations: list[str] = []
    for name in SPLIT_MODULES:
        path = REPORT_DIR / name
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        if len(source.splitlines()) > 500:
            violations.append(f"{name}: file>{500}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.end_lineno - node.lineno + 1 > 300:
                    violations.append(f"{name}:{node.name}: class>300")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.end_lineno - node.lineno + 1 > 50:
                    violations.append(f"{name}:{node.name}: function>50")
    assert violations == []


def test_report_writer_modules_do_not_import_render_or_engine() -> None:
    forbidden_parts = {"render", "gui"}
    violations: list[str] = []
    for name in SPLIT_MODULES:
        tree = ast.parse(
            (REPORT_DIR / name).read_text(encoding="utf-8-sig")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                parts = set(module.split("."))
                if parts & forbidden_parts or module.endswith(
                    ("core.engine", "simulation")
                ):
                    violations.append(f"{name}: {module}")
    assert violations == []


def test_report_package_exports_public_writer_apis() -> None:
    for name in (
        "_stdout_progress",
        "_report_config_payload",
        "_write_simulation_report",
    ):
        assert callable(getattr(report_package, name))


def test_terminal_stream_is_copied_byte_for_byte(tmp_path: Path) -> None:
    stream_path = tmp_path / "runtime" / "terminal_events.csv"
    stream_path.parent.mkdir()
    original = b"event_id,status\r\n0,z_exit\r\n"
    stream_path.write_bytes(original)
    destination = tmp_path / "report" / "terminal_events.csv"
    destination.parent.mkdir()

    materialize_terminal_event_csv(destination, [], stream_path)

    assert destination.read_bytes() == original


def test_missing_terminal_stream_fails_closed(tmp_path: Path) -> None:
    destination = tmp_path / "terminal_events.csv"
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="stream is missing"):
        materialize_terminal_event_csv(destination, [], missing)
