from argparse import Namespace

import numpy as np
import pytest

from src.data.diagnostics.plot_spatial_survival_temperature import (
    comparison_cases, finalize_moments, matching_physics, parse_args, render_group, snapshot_moments,
)


COLUMNS = ["r_m", "z_m", "teff_k", "represented_real_ions", "parent_real_ions_remaining", "status_code"]


def test_weighted_temperature_and_survival_use_ratios_of_weighted_sums():
    data = np.array([[.0001, .0062, 300., 1., 1., 0.], [.0001, .0062, 900., 3., 0., 0.]])
    moments = snapshot_moments(data, COLUMNS, [0., 1.], [6., 7., 8.])
    temperature, survival = finalize_moments(moments)
    assert temperature[0, 0] == 750.
    assert survival[0] == 25.
    assert np.isnan(temperature[0, 1]) and np.isnan(survival[1])


def test_radius_crop_does_not_change_axial_survival():
    data = np.array([[.0001, .0062, 300., 1., 1., 0.], [.002, .0062, 900., 3., 0., 0.]])
    moments = snapshot_moments(data, COLUMNS, [0., 1.], [6., 7.])
    temperature, survival = finalize_moments(moments)
    assert temperature[0, 0] == 300.
    assert survival[0] == 25.
    assert moments["cell_weight"].sum() == 1.
    assert moments["axial_weight"].sum() == 4.


def test_time_exposure_is_applied_before_dividing_moments():
    a = snapshot_moments(np.array([[.0001, .0062, 300., 1., 1., 0.]]), COLUMNS, [0., 1.], [6., 7.])
    b = snapshot_moments(np.array([[.0001, .0062, 900., 3., 0., 0.]]), COLUMNS, [0., 1.], [6., 7.])
    combined = {key: .25*a[key]+.75*b[key] for key in a}
    temperature, survival = finalize_moments(combined)
    assert temperature[0, 0] == 840.
    assert survival[0] == 10.


def test_inactive_packs_are_excluded_and_input_is_not_mutated():
    data = np.array([[.0001, .0062, 300., 1., 1., 0.], [.0001, .0062, 900., 10., 0., 1.]])
    original = data.copy()
    moments = snapshot_moments(data, COLUMNS, [0., 1.], [6., 7.])
    temperature, survival = finalize_moments(moments)
    assert temperature[0, 0] == 300. and survival[0] == 100.
    np.testing.assert_array_equal(original, data)


@pytest.mark.parametrize("column,value", [("parent_real_ions_remaining", 2.), ("teff_k", np.nan), ("r_m", -.001)])
def test_invalid_active_state_is_rejected(column, value):
    data = np.array([[.0001, .0062, 300., 1., 1., 0.]])
    data[0, COLUMNS.index(column)] = value
    with pytest.raises(ValueError, match="Invalid active"):
        snapshot_moments(data, COLUMNS, [0., 1.], [6., 7.])


def test_physics_matching_allows_rf_and_field_only():
    config = {"rf_peak_voltage_v": 80., "static_field_path": "DC0", "fragmentation_mode": "transport", "collision_model": "hybrid-langevin"}
    report = {"template": {"num_atoms": 78}, "config": config}
    other = {"template": report["template"].copy(), "config": dict(config, rf_peak_voltage_v=100., static_field_path="DC40")}
    matching_physics([report, other])
    other["config"]["collision_model"] = "explicit"
    with pytest.raises(ValueError, match="collision_model"):
        matching_physics([report, other])


def test_defaults_keep_requested_window_and_common_temperature_scale():
    args = parse_args(["--case", "80", "0", "case", "--output-dir", "out"])
    assert (args.z_min_mm, args.z_max_mm) == (5., 65.)
    assert (args.start_ms, args.end_ms) == (.4, 1.)
    assert (args.temperature_min_k, args.temperature_max_k) == (0., 2000.)
    assert args.layout == "grouped"


def test_comparison_panel_order_is_dc_rows_rf_columns():
    cases = [{"rf": rf, "dc": dc} for rf, dc in [(100., 100.), (80., 0.), (100., 0.), (80., 100.)]]
    ordered, rfs, dcs = comparison_cases(cases)
    assert rfs == [80., 100.] and dcs == [0., 100.]
    assert [(case["rf"], case["dc"]) for case in ordered] == [(80., 0.), (100., 0.), (80., 100.), (100., 100.)]
    with pytest.raises(ValueError, match="every RF/DC"):
        comparison_cases(cases[:-1])


def test_comparison_cli_accepts_new_temperature_range():
    args = parse_args(["--case", "80", "0", "case", "--output-dir", "out", "--layout", "comparison",
                       "--temperature-min-k", "100", "--temperature-max-k", "1200"])
    assert args.layout == "comparison"
    assert (args.temperature_min_k, args.temperature_max_k) == (100., 1200.)


def test_three_row_render_has_separate_survival_and_radial_axes(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    captured = {}
    save = Figure.savefig

    def capture(figure, *args, **kwargs):
        captured["axes"] = [(ax.get_ylabel(), ax.get_ylim()) for ax in figure.axes]
        captured["labels"] = [t.get_text() for ax in figure.axes for t in ax.texts]
        captured["fonts"] = [ax.yaxis.label.get_fontfamily() for ax in figure.axes]
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    z_edges, r_edges = np.array([5., 25., 45., 65.]), np.array([0., 2.5, 5.])
    temperature = np.array([[300., 600., 1000.], [np.nan, 800., 1200.]])
    cases = [{"rf": 80., "dc": dc, "temperature": temperature.copy(), "survival": np.array([100., 80., 20.]),
              "report": {"config": {"capillary_exit_z_m": .006, "detector_z_m": .061}},
              "statistics": {"exit_survival_percent": 15.}} for dc in (0., 40., 100.)]
    args = Namespace(temperature_min_k=0., temperature_max_k=2000., dpi=72)
    out = tmp_path/"plot.png"
    render_group(cases, r_edges, z_edges, (z_edges, r_edges, np.zeros((3, 4), bool)), args, out)
    assert len(captured["axes"]) == 7
    assert sum(label == "Survival yield / %" for label, _ in captured["axes"]) == 3
    assert sum(label == "r / mm" for label, _ in captured["axes"]) == 3
    assert all(limits == (-3., 103.) for label, limits in captured["axes"] if label == "Survival yield / %")
    assert all(limits == (0., 5.) for label, limits in captured["axes"] if label == "r / mm")
    assert all(font == ["Arial"] for font in captured["fonts"])
    np.testing.assert_array_equal(cases[0]["temperature"], temperature)
    assert out.stat().st_size > 1000


def test_comparison_render_shares_color_limits_and_keeps_extreme_values(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    captured = {}
    save = Figure.savefig

    def capture(figure, *args, **kwargs):
        captured["axes"] = figure.axes.copy()
        captured["bars"] = [ax._colorbar for ax in figure.axes if hasattr(ax, "_colorbar")]
        captured["panel_titles"] = [ax.get_title(loc="left") for ax in figure.axes[:4]]
        captured["spacing"] = (figure.subplotpars.wspace, figure.subplotpars.hspace)
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    z_edges, r_edges = np.array([5., 25., 45., 65.]), np.array([0., 2.5, 5.])
    temperature = np.array([[50., 600., 1700.], [np.nan, 800., 1200.]])
    cases = [{"rf": rf, "dc": dc, "temperature": temperature.copy(), "survival": np.array([100., 80., 20.]),
              "report": {"config": {"capillary_exit_z_m": .006, "detector_z_m": .061}},
              "statistics": {"exit_survival_percent": 15.}} for rf in (100., 80.) for dc in (100., 0.)]
    args = Namespace(temperature_min_k=100., temperature_max_k=1200., dpi=72, layout="comparison")
    out = tmp_path/"comparison.png"
    render_group(cases, r_edges, z_edges, (z_edges, r_edges, np.zeros((3, 4), bool)), args, out)
    assert len(captured["axes"]) == 9
    assert captured["spacing"] == (.06, .17)
    assert captured["axes"][-1].get_position().x0 == pytest.approx(.885)
    assert captured["axes"][0].get_yticklabels()
    assert not captured["axes"][1].get_yticklabels()
    assert not captured["axes"][4].get_yticklabels()
    assert captured["axes"][5].get_yticklabels()
    assert len(captured["bars"]) == 1
    bar = captured["bars"][0]
    assert (bar.norm.vmin, bar.norm.vmax) == (100., 1200.)
    assert bar.extend == "both" and bar.extendrect
    assert [title.split(" = ")[-1] for title in captured["panel_titles"]] == ["0 V", "0 V", "100 V", "100 V"]
    heat_axes = [ax for ax in captured["axes"] if ax.collections]
    for ax in heat_axes:
        if ax is not bar.ax:
            assert ax.collections[0].norm is bar.norm
    for case in cases:
        np.testing.assert_array_equal(case["temperature"], temperature)
    assert out.stat().st_size > 1000
