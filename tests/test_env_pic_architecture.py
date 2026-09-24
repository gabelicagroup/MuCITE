"""Architecture contracts for the environment-owned PIC implementation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.env.fields.axisymmetric_grid import resolve_pic_grid_shape
from src.env.pic.poisson import (
    AxisymmetricAMGPoissonSolver,
    AxisymmetricSparsePoissonSolver,
    create_poisson_solver,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODULES = (
    PROJECT_ROOT / "src" / "env" / "fields" / "axisymmetric_grid.py",
    PROJECT_ROOT / "src" / "env" / "pic" / "operator.py",
    *(PROJECT_ROOT / "src" / "env" / "pic" / "poisson").glob("*.py"),
)
@pytest.mark.parametrize(
    ("requested_nr", "requested_nz", "expected"),
    [
        (None, None, (17, 31, "default_shared_static")),
        (0, 0, (17, 31, "shared_static")),
        (17, 31, (17, 31, "explicit")),
        (9, 15, (9, 15, "explicit")),
    ],
)
def test_pic_grid_resolution_semantics_are_preserved(
    requested_nr: int | None,
    requested_nz: int | None,
    expected: tuple[int, int, str],
) -> None:
    actual = resolve_pic_grid_shape(
        static_nr=17,
        static_nz=31,
        r_max_m=1.0,
        z_min_m=0.0,
        z_max_m=2.0,
        requested_nr=requested_nr,
        requested_nz=requested_nz,
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("backend", "expected_type", "expected_name"),
    [
        ("sparse_direct", AxisymmetricSparsePoissonSolver, "sparse_direct"),
        ("sparse_spsolve", AxisymmetricSparsePoissonSolver, "sparse_spsolve"),
        ("sparse_bicgstab", AxisymmetricSparsePoissonSolver, "sparse_bicgstab"),
        ("sparse_cg", AxisymmetricSparsePoissonSolver, "sparse_cg"),
        ("amg", AxisymmetricAMGPoissonSolver, "amg"),
        ("pyamg", AxisymmetricAMGPoissonSolver, "amg"),
        ("sparse_amg", AxisymmetricAMGPoissonSolver, "amg"),
    ],
)
def test_poisson_factory_preserves_backend_aliases(
    backend: str,
    expected_type: type,
    expected_name: str,
) -> None:
    solver = create_poisson_solver(backend)
    assert type(solver) is expected_type
    assert solver.backend_name == expected_name


def test_pic_production_modules_meet_file_and_class_size_limits() -> None:
    for path in PRODUCTION_MODULES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert len(source.splitlines()) <= 500, path
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.end_lineno is not None
                assert node.end_lineno - node.lineno + 1 <= 300, (
                    path,
                    node.name,
                )
