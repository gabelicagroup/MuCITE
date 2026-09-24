"""Small synthetic contracts for the validated field/trajectory conventions.

These tests intentionally use tiny arrays and text artifacts.  They lock the
coordinate, unit, RF, interpolation, integration, mask, and baked-schema
behavior without regenerating the retained one-gigabyte field artifact.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.benchmarks.simion_field_benchmark import (
    BakedField,
    BenchmarkConfig,
    _load_electrode_mask_patxt as _load_benchmark_electrode_mask_patxt,
    load_simion_initial_rays,
    rk4_step,
    sample_electric_field,
)
from src.config import PROJECT_ROOT, SimulationConfig
from src.render.cli.config_adapter import _resolve_cli_electrode_mask_path
from src.data.boundary_masks.mask2d_io import (
    _load_electrode_mask_patxt,
    resolve_default_electrode_mask_path,
)
from src.env.pic.poisson import compute_axisymmetric_e_field
from src.core.simulation import _config_with_baked_grid, _load_baked_field_file
from src.data.tools.field_baker import (
    FieldBaker,
    GlobalGrid,
    _read_simion_patxt,
)


pytestmark = pytest.mark.physics


def _benchmark_config(**overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "field_path": Path("synthetic-field.npy"),
        "output_dir": Path("synthetic-output"),
        "rf_peak_voltage_v": 4.0,
        "rf_frequency_hz": 0.0,
        "rf_phase_rad": 0.0,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def _constant_field(
    *,
    e_dc_r_v_per_m: float = 0.0,
    e_dc_z_v_per_m: float = 0.0,
    e_rf_r_v_per_m: float = 0.0,
    e_rf_z_v_per_m: float = 0.0,
    rf_reference_peak_voltage_v: float = 1.0,
    extent_m: float = 1.0,
) -> BakedField:
    coords_m = np.array([0.0, extent_m], dtype=float)

    def full(value: float) -> np.ndarray:
        return np.full((2, 2), value, dtype=float)

    return BakedField(
        r_coords_m=coords_m,
        z_coords_m=coords_m,
        e_dc_r_v_per_m=full(e_dc_r_v_per_m),
        e_dc_z_v_per_m=full(e_dc_z_v_per_m),
        e_rf_r_v_per_m=full(e_rf_r_v_per_m),
        e_rf_z_v_per_m=full(e_rf_z_v_per_m),
        rf_reference_peak_voltage_v=rf_reference_peak_voltage_v,
        metadata={"source": "synthetic-test"},
    )


def _patxt(*, include_last_point: bool = True) -> str:
    points = [
        "0 0 0 0 0",
        "1 0 0 1 7.6",
        "2 0 0 0 0",
        "0 1 0 0 0",
        "1 1 0 0 0",
    ]
    if include_last_point:
        points.append("2 1 0 1 3")
    return "\n".join(
        [
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
            *points,
            "end_points",
            "",
        ]
    )


def test_simion_initial_ray_mapping_and_units(tmp_path: Path) -> None:
    source = tmp_path / "rays.csv"
    source.write_text(
        "\n".join(
            [
                "Begin Fly'm",
                "",
                (
                    "Ion(1) Event(Ion Created) "
                    "X(5 mm) Y(1 mm) Z(-2 mm) "
                    "Vx(3 mm/usec) Vy(4 mm/usec) Vz(-5 mm/usec)"
                ),
                "",
                (
                    "Ion(2) Event(Ion Created) "
                    "X(6 mm) Y(-1.5 mm) Z(2.5 mm) "
                    "Vx(7 mm/usec) Vy(-8 mm/usec) Vz(9 mm/usec)"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    positions_m, velocities_m_per_s, checks = load_simion_initial_rays(source, 2)

    np.testing.assert_allclose(
        positions_m,
        np.array(
            [
                [1.0e-3, -2.0e-3, 5.0e-3],
                [-1.5e-3, 2.5e-3, 6.0e-3],
            ]
        ),
    )
    np.testing.assert_allclose(
        velocities_m_per_s,
        np.array(
            [
                [4.0e3, -5.0e3, 3.0e3],
                [-8.0e3, 9.0e3, 7.0e3],
            ]
        ),
    )
    assert checks["initial_z_min_m"] == pytest.approx(5.0e-3)
    assert checks["initial_z_max_m"] == pytest.approx(6.0e-3)


def test_rf_scaling_and_axisymmetric_cartesian_projection() -> None:
    field = _constant_field(
        e_dc_r_v_per_m=2.0,
        e_dc_z_v_per_m=3.0,
        e_rf_r_v_per_m=1.0,
        e_rf_z_v_per_m=-1.0,
        rf_reference_peak_voltage_v=2.0,
        extent_m=1.0e-3,
    )
    positions_m = np.array(
        [
            [0.3e-3, 0.4e-3, 0.5e-3],
            [0.0, 0.0, 0.5e-3],
        ]
    )

    sampled = sample_electric_field(
        field,
        positions_m,
        time_s=0.0,
        config=_benchmark_config(),
    )

    # RF factor = (4 Vpeak / 2 Vref) * cos(0) = 2.
    # Therefore E_r = 2 + 2*1 = 4 V/m and E_z = 3 + 2*(-1) = 1 V/m.
    np.testing.assert_allclose(
        sampled,
        np.array(
            [
                [2.4, 3.2, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        atol=1.0e-14,
    )


def test_rk4_constant_field_matches_analytic_motion() -> None:
    field = _constant_field(e_dc_z_v_per_m=5.0)
    positions_m = np.array([[0.0, 0.0, 0.25]], dtype=float)
    velocities_m_per_s = np.array([[0.0, 0.0, 2.0]], dtype=float)
    dt_s = 1.0e-3

    next_positions_m, next_velocities_m_per_s = rk4_step(
        field,
        positions_m,
        velocities_m_per_s,
        time_s=0.0,
        dt_s=dt_s,
        mass_kg=2.0,
        charge_c=4.0,
        config=_benchmark_config(rf_peak_voltage_v=0.0),
    )

    acceleration_m_per_s2 = 4.0 / 2.0 * 5.0
    expected_z_m = 0.25 + 2.0 * dt_s + 0.5 * acceleration_m_per_s2 * dt_s**2
    expected_vz_m_per_s = 2.0 + acceleration_m_per_s2 * dt_s
    np.testing.assert_allclose(next_positions_m, [[0.0, 0.0, expected_z_m]])
    np.testing.assert_allclose(next_velocities_m_per_s, [[0.0, 0.0, expected_vz_m_per_s]])


def test_axisymmetric_field_derivative_enforces_zero_radial_axis() -> None:
    grid = SimpleNamespace(nr=4, nz=5, dr=0.1, dz=0.2)
    r_m = np.arange(grid.nr, dtype=float)[:, None] * grid.dr
    z_m = np.arange(grid.nz, dtype=float)[None, :] * grid.dz
    phi_v = 3.0 * r_m - 2.0 * z_m

    e_r_v_per_m, e_z_v_per_m = compute_axisymmetric_e_field(phi_v, grid)

    np.testing.assert_allclose(e_r_v_per_m[0, :], 0.0)
    np.testing.assert_allclose(e_r_v_per_m[1:, :], -3.0)
    np.testing.assert_allclose(e_z_v_per_m, 2.0)


def test_small_patxt_preserves_grid_density_mapping_and_electrodes(tmp_path: Path) -> None:
    path = tmp_path / "mask.patxt"
    path.write_text(_patxt(), encoding="utf-8")

    mask = _load_electrode_mask_patxt(
        path,
        z_offset_m=5.0e-3,
        z_offset_source="synthetic-test",
    )

    assert mask.metal_mask.shape == (2, 3)
    np.testing.assert_allclose(mask.r_coords_m, [0.0, 10.0e-6])
    np.testing.assert_allclose(mask.z_coords_m, [5.0e-3, 5.01e-3, 5.02e-3])
    np.testing.assert_array_equal(
        mask.metal_mask,
        np.array([[False, True, False], [False, False, True]]),
    )
    assert mask.electrode_id[0, 1] == 8
    assert mask.electrode_id[1, 2] == 3
    assert mask.metadata["coordinate_mapping"] == (
        "PA text x index -> global z, y index -> global r"
    )
    assert mask.metadata["z_offset_m"] == pytest.approx(5.0e-3)


def test_small_baked_artifact_requires_stable_top_level_groups(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.npy"
    payload = {
        "grid": {"nr": 3, "nz": 3},
        "simion": {"phi_dc_v": np.zeros((3, 3))},
        "fluent": {"pressure_pa": np.ones((3, 3))},
    }
    np.save(valid_path, payload, allow_pickle=True)

    loaded = _load_baked_field_file(valid_path)
    assert set(loaded) == {"grid", "simion", "fluent"}

    invalid_path = tmp_path / "missing-fluent.npy"
    np.save(invalid_path, {"grid": {}, "simion": {}}, allow_pickle=True)
    with pytest.raises(ValueError, match="'fluent' group"):
        _load_baked_field_file(invalid_path)


def test_runtime_config_adopts_baked_grid_extent_and_resolution() -> None:
    baked_fields = {
        "grid": {
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
            "capillary_exit_z_m": 1.0e-3,
        }
    }

    updated = _config_with_baked_grid(SimulationConfig(), baked_fields)

    assert updated.grid_nr == 3
    assert updated.grid_nz == 4
    assert updated.domain_radius_m == pytest.approx(2.0e-3)
    assert updated.domain_length_m == pytest.approx(6.0e-3)
    assert updated.capillary_exit_z_m == pytest.approx(1.0e-3)


def test_truncated_patxt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "truncated.patxt"
    path.write_text(_patxt(include_last_point=False), encoding="utf-8")

    with pytest.raises(ValueError, match="point|complete|truncated"):
        _load_electrode_mask_patxt(path)


def test_inconsistent_baked_grid_coordinates_are_rejected() -> None:
    baked_fields = {
        "grid": {
            "nr": 3,
            "nz": 3,
            "r_min_m": 0.0,
            "r_max_m": 2.0e-3,
            "z_min_m": 0.0,
            "z_max_m": 2.0e-3,
            "dr_m": 1.0e-3,
            "dz_m": 1.0e-3,
            "r_coords_m": np.array([0.0, 1.0e-3, 3.0e-3]),
            "z_coords_m": np.array([0.0, 1.0e-3, 2.0e-3]),
        }
    }

    with pytest.raises(ValueError, match="coordinate|r_coords|metadata"):
        _config_with_baked_grid(SimulationConfig(), baked_fields)


@pytest.mark.parametrize(
    ("case_name", "content"),
    [
        pytest.param(
            "truncated",
            _patxt(include_last_point=False),
            id="truncated",
        ),
        pytest.param(
            "missing-end-points",
            _patxt().replace("end_points\n", ""),
            id="missing-end-points",
        ),
        pytest.param(
            "duplicate",
            _patxt().replace(
                "end_points",
                "2 1 0 1 3\n2 1 0 1 3\nend_points",
            ),
            id="duplicate",
        ),
        pytest.param(
            "malformed-row",
            _patxt().replace("1 1 0 0 0", "1 1 broken"),
            id="malformed-row",
        ),
    ],
)
@pytest.mark.parametrize(
    "loader",
    [
        pytest.param(_read_simion_patxt, id="field-baker"),
        pytest.param(_load_electrode_mask_patxt, id="runtime-mask"),
        pytest.param(
            _load_benchmark_electrode_mask_patxt,
            id="benchmark-mask",
        ),
    ],
)
def test_all_patxt_consumers_reject_corrupt_point_grids(
    tmp_path: Path,
    case_name: str,
    content: str,
    loader: object,
) -> None:
    path = tmp_path / f"{case_name}.patxt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="PATXT|point|truncated|incomplete|Duplicate|Malformed"):
        loader(path)  # type: ignore[operator]


def test_field_baker_infers_patxt_grid_density_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "potential.patxt"
    potential_patxt = (
        _patxt()
        .replace("ny 2", "ny 3")
        .replace(
            "end_points",
            "0 2 0 0 0\n1 2 0 0 0\n2 2 0 0 0\nend_points",
        )
    )
    path.write_text(potential_patxt, encoding="utf-8")
    baker = FieldBaker(
        GlobalGrid(
            z_min_m=0.0,
            z_max_m=20.0e-6,
            r_min_m=0.0,
            r_max_m=20.0e-6,
            dz_m=10.0e-6,
            dr_m=10.0e-6,
        )
    )

    baked = baker.bake_simion_field(
        path,
        0.0,
        label="dc",
        pa_effective_grids_per_mm=None,
        voltage_scale=1.0,
    )

    assert baked["dc_pa_header_ng"] == pytest.approx(100.0)
    assert baked["dc_pa_effective_grids_per_mm"] == pytest.approx(100.0)
    assert baked["dc_pa_requested_grids_per_mm"] is None
    assert baked["dc_pa_grid_density_source"] == "patxt_header_ng"
    assert baked["dc_pa_integrity_validated"] is True

    with pytest.raises(ValueError, match="disagrees with PATXT header ng"):
        baker.bake_simion_field(
            path,
            0.0,
            label="dc",
            pa_effective_grids_per_mm=10.0,
            voltage_scale=1.0,
        )


def test_global_grid_rejects_non_integral_extent() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        GlobalGrid(
            z_min_m=0.0,
            z_max_m=2.5e-3,
            r_min_m=0.0,
            r_max_m=2.0e-3,
            dz_m=1.0e-3,
            dr_m=1.0e-3,
        )


def test_default_mask_resolver_supports_nested_workspace_layout(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "E_field" / "slens" / "slens100x_gem.patxt"
    nested.parent.mkdir(parents=True)
    nested.write_text(_patxt(), encoding="utf-8")

    assert resolve_default_electrode_mask_path(tmp_path) == nested


def test_dummy_field_does_not_implicitly_resolve_large_default_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        _resolve_cli_electrode_mask_path(
            None,
            no_electrode_mask=False,
            dummy_static_field=True,
        )
        is None
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_cli_electrode_mask_path(
            "explicit-mask.patxt",
            no_electrode_mask=True,
            dummy_static_field=False,
        )

    # A synthetic file exercises the default lookup without a private dataset.
    from src.render.cli import config_adapter

    nested = tmp_path / "E_field" / "slens" / "slens100x_gem.patxt"
    nested.parent.mkdir(parents=True)
    nested.write_text(_patxt(), encoding="utf-8")
    monkeypatch.setattr(config_adapter, "PROJECT_ROOT", tmp_path)
    resolved_default = resolve_default_electrode_mask_path(tmp_path)
    assert resolved_default is not None
    assert resolved_default.name == "slens100x_gem.patxt"
    assert _resolve_cli_electrode_mask_path(
        None,
        no_electrode_mask=False,
        dummy_static_field=False,
    ) == resolved_default
