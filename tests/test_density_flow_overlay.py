"""The overlay uses shared absolute contours without changing density integrals."""
from argparse import Namespace
import json

import numpy as np
import pytest

from src.data.diagnostics.plot_collision_density_comparison import (
    annular_volumes, conservative_density_smoothing, contour_levels, load_density, make_overlay_plot, parse_args,
    suppress_repeated_streamline_loops,
)


@pytest.mark.parametrize("levels", [[0, 1], [2, 1], [1, 1], [np.inf], [np.nan], []])
def test_invalid_contour_levels_rejected(levels):
    with pytest.raises(ValueError):
        contour_levels(levels)


def test_overlay_cli_keeps_absolute_contour_levels():
    args = parse_args(["--case-dir", "case", "--output", "test.png", "--layout", "density-overlay",
                       "--ion-contour-levels", "3e11", "1e12", "3e12"])
    assert args.ion_contour_levels == [3e11, 1e12, 3e12]


def test_density_streamlines_cli_allows_display_smoothing():
    args = parse_args(["--case-dir", "case", "--output", "test.png", "--layout", "density-streamlines",
                       "--density-smoothing-mm", ".2"])
    assert args.layout == "density-streamlines"
    assert args.density_smoothing_mm == .2
    assert args.font_family == "Arial"
    assert args.font_size == 14.


def test_time_averaged_cells_cli_keeps_smoothing_disabled():
    args = parse_args(["--case-dir", "case", "--output", "test.png", "--layout", "density-streamlines",
                       "--density-rendering", "cells", "--density-smoothing-mm", "0"])
    assert args.density_rendering == "cells"
    assert args.density_smoothing_mm == 0.


def test_display_filter_removes_only_compact_repeated_streamline_loops():
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import FancyArrowPatch

    theta = np.linspace(0, 12*np.pi, 600)
    repeated_loop = np.column_stack((11.5+.2*np.cos(theta), 4+.15*np.sin(theta)))
    simple_loop = repeated_loop[:100]
    throughflow = np.array([[8., 1.], [9., 1.5], [10., 2.]])
    segments = [throughflow, repeated_loop, simple_loop]
    fig, ax = plt.subplots()
    lines = LineCollection(segments)
    ax.add_collection(lines)
    patches = [FancyArrowPatch(segment[0], segment[1]) for segment in segments]
    for patch in patches:
        ax.add_patch(patch)
    streams = Namespace(lines=lines, arrows=PatchCollection(patches))
    removed = suppress_repeated_streamline_loops(streams, patches)
    assert len(removed) == 1
    assert len(ax.patches) == 2
    assert len(streams.arrows.get_paths()) == 2
    np.testing.assert_array_equal(lines.get_segments()[0], throughflow)
    np.testing.assert_array_equal(lines.get_segments()[1], simple_loop)
    np.testing.assert_array_equal(segments[1], repeated_loop)
    plt.close(fig)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_density_font_size_rejected(value):
    with pytest.raises(SystemExit):
        parse_args(["--case-dir", "case", "--output", "test.png", "--font-size", value])


def test_density_map_ignores_duplicate_final_export_time(tmp_path):
    columns = ["r_m", "z_m", "status_code", "represented_real_ions"]
    (tmp_path / "metadata.json").write_text(json.dumps({"columns": columns}))
    (tmp_path / "snapshots_index.csv").write_text(
        "macro_step,time_s,particle_count,storage_ref\n"
        "0,0,1,snapshot_macro_00000.npy\n"
        "1,.001,1,snapshot_macro_00001.npy\n"
        "3,.003,1,snapshot_macro_00003.npy\n"
        "4,.003,1,snapshot_macro_00004.npy\n")
    for step, weight in ((0, 10), (1, 30), (3, 50), (4, 70)):
        np.save(tmp_path / f"snapshot_macro_{step:05d}.npy", np.array([[.2, .2, 0, weight]]))
    density, detail = load_density(tmp_path, {"config":{"macro_time_step_s":.001}},
                                   Namespace(start_ms=0., end_ms=3., snapshot_stride=1),
                                   np.array([0., 1.]), np.array([0., 1.]))
    assert len(detail["snapshots"]) == 3
    assert density[0, 0]*np.pi == pytest.approx(10/6+30/2+70/3)


@pytest.mark.parametrize("layout, rendering", [("density-overlay", "contours"),
                                              ("density-streamlines", "contours"),
                                              ("density-streamlines", "cells")])
def test_overlay_writes_original_grids_and_common_levels(tmp_path, monkeypatch, layout, rendering):
    from matplotlib.collections import QuadMesh
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    figure_text, text_styles, colorbars, meshes = [], [], [], []
    savefig = Figure.savefig

    def capture_text(figure, *args, **kwargs):
        figure_text.extend(text.get_text() for text in figure.findobj(Text))
        text_styles.extend((text.get_fontfamily(), text.get_fontsize()) for text in figure.findobj(Text))
        colorbars.extend(ax._colorbar for ax in figure.axes if hasattr(ax, "_colorbar"))
        meshes.extend(collection.get_array() for collection in figure.axes[0].collections if isinstance(collection, QuadMesh))
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_text)
    z, r = np.linspace(5, 13, 51), np.linspace(0, 8, 41)
    zz, rr = np.meshgrid(z, r)
    re, ze = np.linspace(0, .008, 21), np.linspace(.005, .013, 26)
    zc, rc = np.meshgrid((ze[:-1]+ze[1:])*.5, (re[:-1]+re[1:])*.5)
    grids, groups, details = {}, {}, []
    for rf in (0., 100.):
        for mode in ("explicit", "hybrid-langevin"):
            key = (rf, mode)
            grids[key] = 8e12*np.exp(-((rc-.0015)/.0008)**2-((zc-.009)/.003)**2)
            report = {"config":{"langevin_z_start_m":.006, "langevin_z_end_m":.007}}
            groups[key] = (tmp_path, report)
            details.append({"rf_Vpeak":rf, "collision_model":mode, "snapshots":[{}, {}],
                            "mean_local_represented_ions":float(np.sum(grids[key]*annular_volumes(re, ze)))})
    args = Namespace(output=str(tmp_path / "overlay.png"), ion_contour_levels=[3e11, 1e12, 3e12],
                     gas_vmin=-120., gas_vmax=720., z_min_mm=5., z_max_mm=13.,
                     r_min_mm=0., r_max_mm=8., start_ms=.4, end_ms=.8, dpi=72, context_note="RF0 is not stationary.",
                     layout=layout, density_rendering=rendering, density_min=1e10, density_max=1e13)
    make_overlay_plot(args, groups, {"config":{"capillary_exit_z_m":.006}}, tmp_path / "field.npy",
                       z, r, np.zeros(zz.shape, dtype=bool), 50*np.sin(zz)*np.exp(-rr/3), np.ones(zz.shape)*300,
                       tmp_path / "mask.npz", grids, details, re, ze)
    meta = json.loads((tmp_path / "overlay.summary.json").read_text())
    assert meta["layout"] == layout
    if layout == "density-streamlines":
        assert meta["density_color_limits_m3"] == [1e10, 1e13]
        assert meta["ion_contour_levels_m3"] is None
        assert meta["gas_color_limits_m_per_s"] is None
        assert meta["model_labels_visible"] is False
        assert meta["font_family"] == "Arial"
        assert meta["tick_font_size_pt"] == 14.
        assert meta["title_and_axis_label_font_size_pt"] == 16.
        assert all(family == ["Arial"] for family, _ in text_styles)
        assert any(size == 16. for _, size in text_styles)
        assert colorbars[0].extendrect is True
        assert meta["density_rendering"] == rendering
        assert any("Time-averaged ion number density" in text for text in figure_text)
        if rendering == "cells":
            assert len(meshes) == 1
            np.testing.assert_array_equal(meshes[0].data, grids[(0., "explicit")])
        assert not any("hybrid" in text.lower() or "explicit" in text.lower() for text in figure_text)
        assert [panel["panel"] for panel in meta["panel_mapping"]] == ["a", "b", "c", "d"]
    else:
        assert meta["ion_contour_levels_m3"] == [3e11, 1e12, 3e12]
    assert meta["smoothing"] == "none"
    with np.load(tmp_path / "overlay.density.npz") as output:
        np.testing.assert_array_equal(output["density_rf100_explicit_m3"], grids[(100., "explicit")])
    assert (tmp_path / "overlay.png").stat().st_size > 1000


def test_display_smoothing_preserves_weight_and_respects_metal_barriers():
    r, z = np.array([0., .1, .3, .6]), np.array([0., .2, .4, .6])
    values = np.zeros((3, 3))
    values[0, 0] = 1000.
    metal = np.zeros(values.shape, dtype=bool)
    metal[:, 1] = True
    result = conservative_density_smoothing(values, r, z, metal, .2)
    volume = annular_volumes(r, z)
    assert np.sum(result*volume) == pytest.approx(np.sum(values*volume), rel=1e-12)
    assert (result >= 0).all()
    assert not result[:, 2].any()
    np.testing.assert_array_equal(result[:, 1], values[:, 1])
    np.testing.assert_array_equal(values, np.array([[1000., 0, 0], [0, 0, 0], [0, 0, 0]]))


def test_display_smoothing_keeps_constant_density_and_zero_scale():
    r, z = np.linspace(0, .01, 6), np.linspace(0, .02, 8)
    values = np.full((5, 7), 1e12)
    metal = np.zeros(values.shape, dtype=bool)
    np.testing.assert_allclose(conservative_density_smoothing(values, r, z, metal, .003), values)
    np.testing.assert_array_equal(conservative_density_smoothing(values, r, z, metal, 0), values)
