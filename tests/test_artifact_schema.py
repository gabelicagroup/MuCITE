"""Structure and behavior contracts for artifact schema modules."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.data.artifacts import schema
from src.data.artifacts import grid_metadata, patxt, patxt_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PACKAGE = PROJECT_ROOT / "src" / "data" / "artifacts"


def _patxt_lines() -> list[str]:
    return [
        "begin_potential_array",
        "begin_header",
        "mode 0",
        "symmetry cylindrical",
        "fast_adjustable 1",
        "nx 3",
        "ny 2",
        "nz 1",
        "ng 100",
        "end_header",
        "begin_points",
        "0 0 0 0 0",
        "1 0 0 1 7.6",
        "2 0 0 0 0",
        "0 1 0 0 0",
        "1 1 0 0 0",
        "2 1 0 1 3",
        "end_points",
        "end_potential_array",
    ]


def _write_patxt(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _grid_metadata() -> dict[str, object]:
    return {
        "nr": 3,
        "nz": 4,
        "r_min_m": 0.0,
        "r_max_m": 2.0e-3,
        "z_min_m": 0.0,
        "z_max_m": 6.0e-3,
        "dr_m": 1.0e-3,
        "dz_m": 2.0e-3,
        "r_coords_m": np.array([0.0, 1.0e-3, 2.0e-3]),
        "z_coords_m": np.array([0.0, 2.0e-3, 4.0e-3, 6.0e-3]),
    }


def test_artifact_modules_meet_phase3_size_limits() -> None:
    for source_path in ARTIFACT_PACKAGE.glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 500, source_path.name
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            first_line = min(
                [node.lineno, *(item.lineno for item in node.decorator_list)]
            )
            line_count = node.end_lineno - first_line + 1
            limit = 300 if isinstance(node, ast.ClassDef) else 50
            assert line_count <= limit, f"{source_path.name}:{node.name}={line_count}"


def test_package_schema_preserves_public_and_private_parser_identity() -> None:
    assert schema.PatxtHeader is patxt.PatxtHeader is patxt_models.PatxtHeader
    assert schema.PatxtMaskGrid is patxt_models.PatxtMaskGrid
    assert schema.PatxtPotentialGrid is patxt_models.PatxtPotentialGrid
    assert schema.read_patxt_mask_grid is patxt.read_patxt_mask_grid
    assert schema.read_patxt_potential_grid is patxt.read_patxt_potential_grid
    assert schema._read_patxt_2d is patxt._read_patxt_2d
    assert schema._parse_patxt_header is patxt._parse_patxt_header
    assert schema._validate_coordinate_axis is grid_metadata._validate_coordinate_axis


def test_valid_patxt_payloads_preserve_values_and_types(tmp_path: Path) -> None:
    path = tmp_path / "valid.patxt"
    _write_patxt(path, _patxt_lines())

    potential = schema.read_patxt_potential_grid(path)
    mask = schema.read_patxt_mask_grid(path)

    assert isinstance(potential, schema.PatxtPotentialGrid)
    assert isinstance(mask, schema.PatxtMaskGrid)
    assert potential.header.nx == 3
    assert potential.header.ny == 2
    assert potential.point_count == mask.point_count == 6
    np.testing.assert_array_equal(potential.potential_v, [[0.0, 7.6, 0.0], [0.0, 0.0, 3.0]])
    np.testing.assert_array_equal(potential.electrode_mask, mask.metal_mask)
    np.testing.assert_array_equal(mask.electrode_id, [[-1, 8, -1], [-1, -1, 3]])


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda lines: [*lines[:16], "0 0 0 0 0", *lines[17:]],
            "Duplicate PATXT point (x=0, y=0, z=0) at line 17 in {path}.",
        ),
        (
            lambda lines: [*lines[:15], "1 1 0 broken 0", *lines[16:]],
            "Malformed numeric PATXT point row at line 16 in {path}: "
            "'1 1 0 broken 0'.",
        ),
        (
            lambda lines: [*lines[:16], *lines[17:]],
            "PATXT point grid is incomplete in {path}: parsed 5 unique points, "
            "expected 6, missing 1; first missing point is (x=2, y=1, z=0).",
        ),
        (
            lambda lines: lines[:17],
            "PATXT points section is truncated in {path}: missing end_points "
            "after 6 of 6 points.",
        ),
    ],
)
def test_patxt_strict_errors_remain_exact(
    tmp_path: Path,
    mutate: object,
    expected: str,
) -> None:
    path = tmp_path / "invalid.patxt"
    lines = mutate(_patxt_lines())  # type: ignore[operator]
    _write_patxt(path, lines)

    with pytest.raises(ValueError) as exc_info:
        schema.read_patxt_potential_grid(path)

    assert str(exc_info.value) == expected.format(path=path)


def test_mask_reader_preserves_int32_range_error(tmp_path: Path) -> None:
    path = tmp_path / "overflow.patxt"
    lines = _patxt_lines()
    lines[16] = "2 1 0 1 2147483648"
    _write_patxt(path, lines)

    with pytest.raises(ValueError) as exc_info:
        schema.read_patxt_mask_grid(path)

    assert str(exc_info.value) == (
        "PATXT electrode id is outside int32 range at "
        f"line 17 in {path}: 2147483648."
    )


def test_baked_grid_metadata_preserves_values_and_contiguous_axes() -> None:
    validated = schema.validate_baked_grid_metadata(
        _grid_metadata(),
        source="unit-test",
    )

    assert isinstance(validated, schema.BakedGridMetadata)
    assert validated.nr == 3
    assert validated.nz == 4
    assert validated.r_coords_m.flags.c_contiguous
    assert validated.z_coords_m.flags.c_contiguous


def test_baked_grid_missing_scalar_error_remains_exact() -> None:
    metadata = _grid_metadata()
    del metadata["nr"]

    with pytest.raises(ValueError) as exc_info:
        schema.validate_baked_grid_metadata(metadata, source="unit-test")

    assert str(exc_info.value) == (
        "Baked grid scalar metadata is invalid in unit-test: "
        "Baked grid metadata is missing required key 'nr' in unit-test."
    )


def test_baked_grid_nonuniform_axis_error_remains_exact() -> None:
    metadata = _grid_metadata()
    metadata["z_coords_m"] = np.array([0.0, 1.0e-3, 4.0e-3, 6.0e-3])

    with pytest.raises(ValueError) as exc_info:
        schema.validate_baked_grid_metadata(metadata, source="unit-test")

    assert str(exc_info.value) == (
        "Baked grid z_coords_m is not uniformly spaced in unit-test."
    )
