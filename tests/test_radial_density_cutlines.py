"""Radial profiles retain weighted density and reject electrode-filled slabs."""
from argparse import Namespace
import json

import numpy as np
import pytest

from src.data.diagnostics.plot_collision_density_comparison import (
    annular_volumes, cutline_bounds, load_radial_profiles, parse_args, validate_open_slabs,
)


def test_cut_bounds_are_centered_slabs_in_meters():
    np.testing.assert_allclose(cutline_bounds([8, 10, 12], .6, 5, 13),
                               [[.0077, .0083], [.0097, .0103], [.0117, .0123]])


@pytest.mark.parametrize("centers,width", [([8, 8], .6), ([np.nan], .6), ([8], 0), ([8], np.inf), ([5], .6)])
def test_invalid_or_outside_cut_slabs_rejected(centers, width):
    with pytest.raises(ValueError):
        cutline_bounds(centers, width, 5, 13)


def test_mask_check_requires_open_and_covered_annuli():
    mask = Namespace(r_coords_m=np.array([0, .5, 1.]), z_coords_m=np.array([0, 1., 2.]),
                     metal_mask=np.zeros((3, 3), dtype=bool))
    validate_open_slabs(mask, 0, 1, np.array([[0., 1.]]))
    mask.metal_mask[2, 1] = True
    with pytest.raises(ValueError, match="intersects electrodes"):
        validate_open_slabs(mask, 0, 1, np.array([[0., 1.]]))
    with pytest.raises(ValueError, match="covered"):
        validate_open_slabs(mask, 0, 2, np.array([[0., 1.]]))


def test_radial_profiles_use_annular_volume_and_time_exposure(tmp_path):
    columns = ["r_m", "z_m", "status_code", "represented_real_ions"]
    (tmp_path / "metadata.json").write_text(json.dumps({"columns": columns}))
    for step, weight in ((0, 10.), (1, 30.), (3, 50.)):
        np.save(tmp_path / f"snapshot_macro_{step:05d}.npy",
                np.array([[.2, .2, 0, weight], [1.2, 1.2, 0, 2*weight],
                          [.2, .2, 1, 10000], [.2, 4., 0, 10000]]))
    args = Namespace(start_ms=0., end_ms=3., snapshot_stride=1)
    bounds, r = np.array([[0., 1.], [1., 2.]]), np.array([0., 1., 2.])
    density, audit = load_radial_profiles(tmp_path, {"config":{"macro_time_step_s":.001}}, args, r, bounds)
    expected = 10/6 + 30/2 + 50/3
    counts = np.array([[expected, 0], [0, 2*expected]])
    volumes = np.array([annular_volumes(r, b)[:, 0] for b in bounds])
    np.testing.assert_allclose(density*volumes, counts)
    np.testing.assert_allclose(audit["mean_slab_represented_ions"], [expected, 2*expected])
    assert density[0, 1] == 0
    assert density[1, 0] == 0


def test_original_layout_remains_default():
    args = parse_args(["--case-dir", "case", "--output", "test.png"])
    assert args.layout == "density-map"
    args = parse_args(["--case-dir", "case", "--output", "test.png", "--layout", "radial-cutlines"])
    assert args.cut_z_mm == [8., 10., 12.]


def test_duplicate_final_export_is_not_counted_twice(tmp_path):
    columns = ["r_m", "z_m", "status_code", "represented_real_ions"]
    (tmp_path / "metadata.json").write_text(json.dumps({"columns": columns}))
    (tmp_path / "snapshots_index.csv").write_text(
        "macro_step,time_s,particle_count,storage_ref\n"
        "0,0,1,snapshot_macro_00000.npy\n"
        "1,.001,1,snapshot_macro_00001.npy\n"
        "3,.003,1,snapshot_macro_00003.npy\n"
        "4,.003,1,snapshot_macro_00004.npy\n")
    for step, weight in ((0, 10.), (1, 30.), (3, 50.), (4, 70.)):
        np.save(tmp_path / f"snapshot_macro_{step:05d}.npy", np.array([[.2, .2, 0, weight]]))
    args = Namespace(start_ms=0., end_ms=3., snapshot_stride=1)
    density, audit = load_radial_profiles(tmp_path, {"config":{"macro_time_step_s":.001}},
                                          args, np.array([0., 1.]), np.array([[0., 1.]]))
    assert len(audit["snapshots"]) == 3
    assert density[0, 0]*np.pi == pytest.approx(10/6+30/2+70/3)
