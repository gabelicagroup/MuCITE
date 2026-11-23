from argparse import Namespace

import numpy as np
import pytest

from src.data.diagnostics.plot_production_field_alignment import (
    direction_samples, display_norms, panel_statistics, parse_args, render_figure, select_field_input,
    draw_electric_field_lines,
)
from src.data.diagnostics.plot_field_flow_ion_overlay import electric_components


def test_field_override_requires_explicit_authorization(tmp_path):
    recorded, other = tmp_path/"recorded.npy", tmp_path/"other.npy"
    np.save(recorded, np.zeros(1))
    np.save(other, np.ones(1))
    assert select_field_input(recorded) == (recorded, recorded)
    with pytest.raises(ValueError, match="allow-field-override"):
        select_field_input(recorded, other)
    assert select_field_input(recorded, other, allow_override=True) == (recorded, other)


def test_bounded_color_mapping_does_not_change_near_zero_fields():
    norms = display_norms(1., 1e3, 1e6, 900.)
    dc, rf = np.array([0., 1e-9, 1e-8]), np.array([0., 10., 999., 1e4, 1e6])
    original_dc, original_rf = dc.copy(), rf.copy()
    assert np.max(norms["dc"](dc)) < 1e-7
    np.testing.assert_array_equal(norms["rf"](rf)[:3], [0., 0., 0.])
    np.testing.assert_array_equal(dc, original_dc)
    np.testing.assert_array_equal(rf, original_rf)
    assert norms["rf"].vmin == norms["total"].vmin
    assert norms["rf"].vmax == norms["total"].vmax


def test_dc_log_range_preserves_original_values():
    norms = display_norms(1., 1e3, 1e6, 900., dc_min_v_per_m=1e-3)
    original = np.array([0., 1e-9, 1e-4, 1e-3, 1.])
    values = original.copy()
    np.testing.assert_array_equal(norms["dc"](values), [0., 0., 0., 0., 1.])
    np.testing.assert_array_equal(values, original)
    assert norms["dc"].vmin == 1e-3
    assert norms["dc"].vmax == 1.


@pytest.mark.parametrize("minimum", [-1., 1., np.nan, np.inf])
def test_invalid_dc_minimum_is_rejected(minimum):
    with pytest.raises(ValueError):
        display_norms(1., 1e3, 1e6, 900., dc_min_v_per_m=minimum)


@pytest.mark.parametrize("limits", [(0, 1e3, 1e6, 900), (1, 1e6, 1e3, 900), (1, 0, 1e6, 900), (1, 1e3, np.inf, 900)])
def test_bad_display_limits_are_rejected(limits):
    with pytest.raises(ValueError):
        display_norms(*limits)


def test_arrows_exclude_metal_and_small_fields():
    axis = np.arange(3.)
    er, ez = np.ones((3, 3))*2000, np.zeros((3, 3))
    metal = np.zeros((3, 3), bool)
    metal[1, 1] = True
    er[0, 0] = 1e-8
    x, y, dx, dy = direction_samples(axis, axis, er, ez, metal, 1000, .5)
    assert len(x) == 7
    assert not any((x == 1) & (y == 1))
    assert not any((x == 0) & (y == 0))
    np.testing.assert_array_equal(dx, 0)
    np.testing.assert_array_equal(dy, 1)


def test_statistics_exclude_metal_and_report_clipped_values():
    values = np.array([[1., 1e5], [0., 1e7]])
    metal = np.array([[False, True], [False, False]])
    norm = display_norms(1., 1e3, 1e6, 900)["rf"]
    stats = panel_statistics(values, metal, norm)
    assert stats["median"] == 1.
    assert stats["fraction_below_display_minimum"] == pytest.approx(2/3)
    assert stats["fraction_above_display_maximum"] == pytest.approx(1/3)


def test_cli_phase_is_explicit_and_range_is_narrow():
    args = parse_args(["--case-dir", "case", "--output", "figure.png"])
    assert args.rf_phase_deg == 0
    assert args.electric_min_v_per_m == 1e3
    assert args.electric_max_v_per_m == 1e6


def test_gas_electric_overlay_cli():
    args = parse_args(["--case-dir", "case", "--output", "figure.png", "--combined-panel", "gas-electric",
                       "--z-min-mm", "3.5", "--z-max-mm", "13"])
    assert args.combined_panel == "gas-electric"
    assert args.z_max_mm == 13.
    assert args.electric_overlay == "arrows"


def test_field_lines_omit_zero_fields_without_changing_arrays():
    import matplotlib.pyplot as plt

    z, r = np.arange(4.), np.arange(4.)
    er, ez, metal = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4), bool)
    fig, ax = plt.subplots()
    assert draw_electric_field_lines(ax, z, r, er, ez, metal, 1e3, .7) == (0, 0)
    np.testing.assert_array_equal(er, 0.)
    np.testing.assert_array_equal(ez, 0.)
    plt.close(fig)


def test_field_line_inputs_use_electric_components_and_exclude_metal(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    z, r = np.linspace(0, 3, 16), np.linspace(0, 3, 16)
    er, ez = np.zeros((16, 16)), np.full((16, 16), 2000.)
    metal = np.zeros((16, 16), bool)
    metal[7:9, 7:9] = True
    ez[1, 1] = 1.
    er_before, ez_before = er.copy(), ez.copy()
    fig, ax = plt.subplots()
    original = ax.streamplot
    captured = {}

    def capture(x, y, u, v, **kwargs):
        captured["u"], captured["v"] = u, v
        return original(x, y, u, v, **kwargs)

    monkeypatch.setattr(ax, "streamplot", capture)
    paths, arrows = draw_electric_field_lines(ax, z, r, er, ez, metal, 1e3, .7)
    excluded = metal | (np.hypot(er, ez) < 1e3)
    np.testing.assert_array_equal(np.ma.getmaskarray(captured["u"]), excluded)
    np.testing.assert_array_equal(np.ma.getmaskarray(captured["v"]), excluded)
    np.testing.assert_array_equal(captured["u"].data, ez)
    np.testing.assert_array_equal(captured["v"].data, er)
    np.testing.assert_array_equal(er, er_before)
    np.testing.assert_array_equal(ez, ez_before)
    assert paths > 0 and arrows > 0
    plt.close(fig)


@pytest.mark.parametrize("phase, expected", [(0., (103., -46.)), (90., (3., 4.)), (180., (-97., 54.))])
def test_instantaneous_field_uses_peak_scaling_and_selected_cosine_phase(phase, expected):
    simion = {"e_dc_r_v_per_m":np.full((2, 2), 3.), "e_dc_z_v_per_m":np.full((2, 2), 4.),
              "e_rf_r_v_per_m":np.full((2, 2), 2.), "e_rf_z_v_per_m":np.full((2, 2), -1.)}
    original = {key: value.copy() for key, value in simion.items()}
    er, ez = electric_components(simion, np.s_[:, :], "total", 100., 2., phase)
    np.testing.assert_allclose(er, expected[0], atol=1e-12)
    np.testing.assert_allclose(ez, expected[1], atol=1e-12)
    for key in simion:
        np.testing.assert_array_equal(simion[key], original[key])


@pytest.mark.parametrize("dc_minimum, dc_cmap, exit_window, combined, overlay", [(0., "Greys", False, "electric", "arrows"),
    (1e-3, "magma", False, "electric", "arrows"), (1., "magma", True, "electric", "arrows"),
    (1e-3, "magma", False, "gas-electric", "arrows"), (1e-3, "magma", False, "gas-electric", "lines")])
def test_render_four_panels_in_arial_without_changing_fields(tmp_path, monkeypatch, dc_minimum, dc_cmap, exit_window, combined, overlay):
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from matplotlib.text import Text

    captured = {}
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        captured["titles"] = [ax.get_title(loc="left") for ax in figure.axes[:4]]
        captured["text"] = [(text.get_text(), text.get_fontfamily()) for text in figure.findobj(Text)]
        captured["bars"] = [ax._colorbar for ax in figure.axes if hasattr(ax, "_colorbar")]
        captured["legends"] = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
        captured["quivers"] = list(figure.findobj(Quiver))
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    z = np.linspace(52., 62., 41) if exit_window else np.linspace(3.5, 13.5, 41)
    r = np.linspace(0, 10, 31)
    shape = (len(r), len(z))
    metal = np.zeros(shape, bool)
    metal[15:, 10:12] = True
    gas, dc, rf = np.full(shape, 800.), np.full(shape, 1e-8), np.full(shape, 2e4)
    originals = [a.copy() for a in (gas, dc, rf)]
    args = Namespace(dc_max_v_per_m=1e5 if exit_window else 1., electric_min_v_per_m=1e3, electric_max_v_per_m=1e6,
                     dc_min_v_per_m=dc_minimum, dc_cmap=dc_cmap,
                     gas_max_m_per_s=900., font_family="Arial", font_size=14., dpi=72,
                     rf_phase_deg=0., arrow_spacing_mm=.6, z_min_mm=float(z[0]), z_max_mm=float(z[-1]),
                     r_min_mm=0., r_max_mm=10., combined_panel=combined, electric_overlay=overlay)
    config = {"rf_peak_voltage_v":100., "rf_frequency_hz":650000., "capillary_exit_z_m":.006}
    output = tmp_path/"figure.png"
    stats, count = render_figure(args, config, z, r, metal, gas, dc, rf, np.zeros(shape), rf, output)
    for expected, actual in zip(originals, (gas, dc, rf)):
        np.testing.assert_array_equal(actual, expected)
    assert len(captured["bars"]) == 4
    assert all(bar.extendrect for bar in captured["bars"])
    assert all(family == ["Arial"] for _, family in captured["text"])
    if combined == "gas-electric":
        assert any(f"Gas speed + DC/RF {overlay}" in title for title in captured["titles"])
        assert stats["gas_electric"]["background_quantity"] == "gas speed"
        assert stats["gas_electric"]["units"] == "m/s"
        assert stats["gas_electric"]["display_minimum"] == stats["gas"]["display_minimum"]
        assert stats["gas_electric"]["display_maximum"] == stats["gas"]["display_maximum"]
        if overlay == "lines":
            assert stats["gas_electric"]["field_line_path_count"] > 0
            assert any("DC + RF electric field lines" in text for text in captured["legends"])
            assert not captured["quivers"]
        else:
            assert stats["gas_electric"]["direction_arrow_count"] == count
            assert stats["gas_electric"]["direction_arrow_length_mm"] == .28
            assert any("fixed-length arrows" in text for text in captured["legends"])
    else:
        assert any("Instantaneous DC + RF" in title for title in captured["titles"])
    if overlay == "arrows":
        assert len(captured["quivers"]) == 1
        quiver = captured["quivers"][0]
        np.testing.assert_allclose(quiver.U, .28)
        np.testing.assert_allclose(quiver.V, 0.)
        assert len(quiver.U) == count
        assert quiver.scale == 1.
    assert any("no space charge" in text for text, _ in captured["text"])
    assert stats["dc"]["maximum"] == 1e-8
    assert stats["dc"]["display_minimum"] == dc_minimum
    assert stats["dc"]["colormap"] == dc_cmap
    if exit_window:
        assert not captured["legends"]
    else:
        assert any("Capillary exit" in text for text in captured["legends"])
    assert count > 0
    assert output.stat().st_size > 1000
