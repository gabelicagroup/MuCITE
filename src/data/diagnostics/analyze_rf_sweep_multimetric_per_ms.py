from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ELEMENTARY_CHARGE_C = 1.602176634e-19
CASE_PATTERN = re.compile(r"RF(?P<rf>\d+)V")
EVENT_COLUMNS = [
    "status",
    "event_time_s",
    "r_m",
    "ke_ev",
    "internal_temperature_k",
    "represented_real_ions",
    "parent_real_ions_remaining",
    "fragmented_real_ions_represented",
]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return math.nan
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    threshold = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(values[min(max(index, 0), values.size - 1)])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return math.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def distribution_stats(events: pd.DataFrame) -> dict[str, float]:
    if events.empty:
        return {
            "ke_ev_mean_weighted": math.nan,
            "ke_ev_p05_weighted": math.nan,
            "ke_ev_p50_weighted": math.nan,
            "ke_ev_p95_weighted": math.nan,
            "temperature_k_mean_weighted": math.nan,
            "temperature_k_p05_weighted": math.nan,
            "temperature_k_p50_weighted": math.nan,
            "temperature_k_p95_weighted": math.nan,
            "exit_beam_radius_mm_p50_weighted": math.nan,
            "exit_beam_radius_mm_p95_weighted": math.nan,
        }
    weights = events["represented_real_ions"].to_numpy(dtype=np.float64)
    radius_mm = 1.0e3 * events["r_m"].to_numpy(dtype=np.float64)
    ke = events["ke_ev"].to_numpy(dtype=np.float64)
    temperature = events["internal_temperature_k"].to_numpy(dtype=np.float64)
    return {
        "ke_ev_mean_weighted": weighted_mean(ke, weights),
        "ke_ev_p05_weighted": weighted_quantile(ke, weights, 0.05),
        "ke_ev_p50_weighted": weighted_quantile(ke, weights, 0.50),
        "ke_ev_p95_weighted": weighted_quantile(ke, weights, 0.95),
        "temperature_k_mean_weighted": weighted_mean(temperature, weights),
        "temperature_k_p05_weighted": weighted_quantile(temperature, weights, 0.05),
        "temperature_k_p50_weighted": weighted_quantile(temperature, weights, 0.50),
        "temperature_k_p95_weighted": weighted_quantile(temperature, weights, 0.95),
        "exit_beam_radius_mm_p50_weighted": weighted_quantile(radius_mm, weights, 0.50),
        "exit_beam_radius_mm_p95_weighted": weighted_quantile(radius_mm, weights, 0.95),
    }


def current_na(weight: float, duration_s: float) -> float:
    if duration_s <= 0.0:
        return math.nan
    return float(weight) * ELEMENTARY_CHARGE_C / duration_s * 1.0e9


def terminal_outcome_stats(
    events: pd.DataFrame,
    duration_s: float,
    source_current_na: float,
) -> dict[str, float]:
    """Return terminal outcome currents and input-normalized fractions."""
    outcomes: dict[str, float] = {}
    total_weight = 0.0
    for status in ("z_exit", "electrode_hit", "domain_out", "radial_out"):
        weight = float(events.loc[events["status"].eq(status), "represented_real_ions"].sum())
        current = current_na(weight, duration_s)
        outcomes[f"stable_{status}_weight"] = weight
        outcomes[f"stable_{status}_current_nA"] = current
        outcomes[f"stable_{status}_percent_of_input"] = 100.0 * current / source_current_na
        total_weight += weight
    outcomes["stable_terminal_accounted_percent_of_input"] = (
        100.0 * current_na(total_weight, duration_s) / source_current_na
    )
    return outcomes


def load_case(case_dir: Path, bin_width_ms: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    match = CASE_PATTERN.search(case_dir.name)
    if not match:
        raise ValueError(f"Cannot parse RF amplitude from {case_dir.name}")
    rf_vpeak = int(match.group("rf"))

    with (case_dir / "simulation_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    requested_time_s = float(summary["config"]["total_time_s"])
    simulated_time_s = float(summary["performance"]["simulated_time_s"])
    source_current_a = float(summary["source_runtime"]["ion_current_a"])
    source_current_na = source_current_a * 1.0e9
    termination_reason = str(summary["result"]["termination_reason"])
    completed = simulated_time_s >= requested_time_s - 1.0e-12

    events = pd.read_csv(case_dir / "terminal_events.csv", usecols=EVENT_COLUMNS)
    exits = events.loc[events["status"].eq("z_exit")].copy()
    exits = exits.loc[np.isfinite(exits["event_time_s"])].copy()

    requested_time_ms = requested_time_s * 1.0e3
    bin_count = int(math.ceil(requested_time_ms / bin_width_ms - 1.0e-12))
    rows: list[dict[str, object]] = []
    for bin_index in range(bin_count):
        start_ms = bin_index * bin_width_ms
        end_ms = min((bin_index + 1) * bin_width_ms, requested_time_ms)
        duration_ms = end_ms - start_ms
        start_s = start_ms * 1.0e-3
        end_s = end_ms * 1.0e-3
        if bin_index == bin_count - 1:
            mask = exits["event_time_s"].ge(start_s) & exits["event_time_s"].le(end_s + 1.0e-15)
        else:
            mask = exits["event_time_s"].ge(start_s) & exits["event_time_s"].lt(end_s)
        subset = exits.loc[mask]

        total_weight = float(subset["represented_real_ions"].sum())
        parent_weight = float(subset["parent_real_ions_remaining"].sum())
        fragment_weight = float(subset["fragmented_real_ions_represented"].sum())
        duration_s = duration_ms * 1.0e-3
        total_current_na = current_na(total_weight, duration_s)
        parent_current_na = current_na(parent_weight, duration_s)
        fragment_current_na = current_na(fragment_weight, duration_s)
        bin_state = "complete" if simulated_time_s >= end_s - 1.0e-12 else "partial_or_missing"

        row: dict[str, object] = {
            "case": case_dir.name,
            "rf_vpeak": rf_vpeak,
            "requested_total_time_ms": requested_time_ms,
            "simulated_time_ms": simulated_time_s * 1.0e3,
            "completed_requested_time": completed,
            "termination_reason": termination_reason,
            "source_current_nA": source_current_na,
            "ms_bin": bin_index + 1,
            "time_start_ms": start_ms,
            "time_end_ms": end_ms,
            "bin_duration_ms": duration_ms,
            "bin_data_state": bin_state,
            "z_exit_macro_events_this_ms": int(len(subset)),
            "z_exit_weight_this_ms": total_weight,
            "z_exit_parent_weight_this_ms": parent_weight,
            "z_exit_fragment_weight_this_ms": fragment_weight,
            "ion_current_nA_this_ms": total_current_na,
            "parent_current_nA_this_ms": parent_current_na,
            "fragment_current_nA_this_ms": fragment_current_na,
            "transmission_percent_this_ms": 100.0 * total_current_na / source_current_na,
            "intact_transmission_percent_this_ms": 100.0 * parent_current_na / source_current_na,
            "fragment_transmission_percent_this_ms": 100.0 * fragment_current_na / source_current_na,
            "fragmentation_percent_of_exit_this_ms": (
                100.0 * fragment_weight / total_weight if total_weight > 0.0 else math.nan
            ),
        }
        row.update(distribution_stats(subset))
        rows.append(row)

    full_duration_s = max(simulated_time_s, 1.0e-30)
    overall_total_weight = float(exits["represented_real_ions"].sum())
    overall_parent_weight = float(exits["parent_real_ions_remaining"].sum())
    overall_fragment_weight = float(exits["fragmented_real_ions_represented"].sum())
    overall_current_na = current_na(overall_total_weight, full_duration_s)

    stable_start_s = min(1.0e-3, simulated_time_s)
    stable = exits.loc[exits["event_time_s"].ge(stable_start_s)]
    stable_all = events.loc[
        events["event_time_s"].ge(stable_start_s) & events["event_time_s"].le(simulated_time_s + 1.0e-15)
    ]
    stable_duration_s = simulated_time_s - stable_start_s
    stable_total_weight = float(stable["represented_real_ions"].sum())
    stable_parent_weight = float(stable["parent_real_ions_remaining"].sum())
    stable_fragment_weight = float(stable["fragmented_real_ions_represented"].sum())
    stable_current_na = current_na(stable_total_weight, stable_duration_s)
    stable_parent_current_na = current_na(stable_parent_weight, stable_duration_s)
    stable_fragment_current_na = current_na(stable_fragment_weight, stable_duration_s)
    stable_stats = distribution_stats(stable)

    repeated = {
        "overall_ion_current_nA_0_to_end": overall_current_na,
        "overall_transmission_percent_0_to_end": 100.0 * overall_current_na / source_current_na,
        "overall_fragmentation_percent_of_exit": (
            100.0 * overall_fragment_weight / overall_total_weight if overall_total_weight > 0.0 else math.nan
        ),
        "stable_window_ms": f"{stable_start_s * 1.0e3:g}-{simulated_time_s * 1.0e3:g}",
        "stable_ion_current_nA": stable_current_na,
        "stable_parent_current_nA": stable_parent_current_na,
        "stable_fragment_current_nA": stable_fragment_current_na,
        "stable_transmission_percent": 100.0 * stable_current_na / source_current_na,
        "stable_intact_transmission_percent": 100.0 * stable_parent_current_na / source_current_na,
        "stable_fragment_transmission_percent": 100.0 * stable_fragment_current_na / source_current_na,
        "stable_fragmentation_percent_of_exit": (
            100.0 * stable_fragment_weight / stable_total_weight if stable_total_weight > 0.0 else math.nan
        ),
        **{f"stable_{key}": value for key, value in stable_stats.items()},
        **terminal_outcome_stats(stable_all, stable_duration_s, source_current_na),
    }
    for row in rows:
        row.update(repeated)

    case_summary = {
        "case": case_dir.name,
        "rf_vpeak": rf_vpeak,
        "source_current_nA": source_current_na,
        "completed_requested_time": completed,
        **repeated,
    }
    return rows, case_summary


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    return ImageFont.truetype(str(path), size=size)


def _rgb(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4)) + (alpha,)


def _nice_linear_ticks(maximum: float, count: int = 5) -> np.ndarray:
    if not math.isfinite(maximum) or maximum <= 0.0:
        return np.linspace(0.0, 1.0, count)
    raw_step = maximum / (count - 1)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    step = next(candidate for candidate in (1.0, 2.0, 2.5, 5.0, 10.0) if candidate >= normalized) * magnitude
    top = math.ceil(maximum / step) * step
    return np.arange(0.0, top + step * 0.5, step)


def _draw_panel(
    image: Image.Image,
    rectangle: tuple[int, int, int, int],
    panel_label: str,
    x_values: np.ndarray,
    ylabel: str,
    lines: list[tuple[str, np.ndarray, str, str]],
    y_ticks: np.ndarray,
    y_limits: tuple[float, float],
    log_y: bool = False,
    band: tuple[np.ndarray, np.ndarray, str] | None = None,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = rectangle
    plot_left, plot_top = left + 190, top + 85
    plot_right, plot_bottom = right - 45, bottom - 150
    tick_font = _font(26)
    label_font = _font(33)
    legend_font = _font(25)
    panel_font = _font(34, bold=True)

    x_min, x_max = 0.0, 200.0
    y_min, y_max = y_limits

    def x_pixel(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def y_pixel(value: float) -> float:
        if log_y:
            value = max(value, y_min)
            fraction = (math.log10(value) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min))
        else:
            fraction = (value - y_min) / (y_max - y_min)
        return plot_bottom - fraction * (plot_bottom - plot_top)

    for value in y_ticks:
        if value < y_min or value > y_max:
            continue
        y = y_pixel(float(value))
        draw.line((plot_left, y, plot_right, y), fill=_rgb("#d9d9d9"), width=2)
        label = f"{value:g}"
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((plot_left - 18 - (box[2] - box[0]), y - (box[3] - box[1]) / 2), label, font=tick_font, fill="black")

    for value in (0, 50, 100, 150, 200):
        x = x_pixel(float(value))
        draw.line((x, plot_top, x, plot_bottom), fill=_rgb("#ececec"), width=2)
        label = str(value)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (box[2] - box[0]) / 2, plot_bottom + 18), label, font=tick_font, fill="black")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=3)

    if band is not None:
        lower, upper, color = band
        upper_points = [(x_pixel(float(x)), y_pixel(float(y))) for x, y in zip(x_values, upper)]
        lower_points = [(x_pixel(float(x)), y_pixel(float(y))) for x, y in zip(x_values[::-1], lower[::-1])]
        draw.polygon(upper_points + lower_points, fill=_rgb(color, 60))

    marker_shapes = ("circle", "square", "triangle")
    for line_index, (label, values, color, marker) in enumerate(lines):
        points = [(x_pixel(float(x)), y_pixel(float(y))) for x, y in zip(x_values, values) if math.isfinite(y)]
        if len(points) >= 2:
            draw.line(points, fill=_rgb(color), width=6, joint="curve")
        radius = 10
        for x, y in points:
            if marker == "square":
                draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=_rgb(color), outline="white", width=2)
            elif marker == "triangle":
                draw.polygon(((x, y - radius - 2), (x - radius - 1, y + radius), (x + radius + 1, y + radius)), fill=_rgb(color))
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_rgb(color), outline="white", width=2)

        legend_x = plot_left + 90 + line_index * int((plot_right - plot_left - 110) / max(len(lines), 1))
        legend_y = plot_top + 14
        draw.line((legend_x, legend_y + 12, legend_x + 45, legend_y + 12), fill=_rgb(color), width=6)
        draw.text((legend_x + 56, legend_y), label, font=legend_font, fill="black")

    draw.text((left + 18, top + 15), panel_label, font=panel_font, fill="black")
    x_label = "RF amplitude (Vpeak)"
    box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((plot_left + plot_right - (box[2] - box[0])) / 2, bottom - 58), x_label, font=label_font, fill="black")

    label_layer = Image.new("RGBA", (bottom - top, 110), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((0, 0), ylabel, font=label_font, fill="black")
    rotated = label_layer.rotate(90, expand=True)
    image.alpha_composite(rotated, (left + 18, int((plot_top + plot_bottom - rotated.height) / 2)))


def plot_summary_pillow(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    summary = summary.sort_values("rf_vpeak")
    rf = summary["rf_vpeak"].to_numpy(dtype=float)
    image = Image.new("RGBA", (3600, 2500), "white")
    rectangles = (
        (40, 35, 1780, 1230),
        (1820, 35, 3560, 1230),
        (40, 1270, 1780, 2460),
        (1820, 1270, 3560, 2460),
    )

    current_max = float(summary[["stable_ion_current_nA", "stable_parent_current_nA", "stable_fragment_current_nA"]].max().max())
    current_ticks = _nice_linear_ticks(current_max * 1.05)
    _draw_panel(
        image,
        rectangles[0],
        "(a)",
        rf,
        "Exit current (nA)",
        [
            ("Total", summary["stable_ion_current_nA"].to_numpy(float), "#1f77b4", "circle"),
            ("Parent", summary["stable_parent_current_nA"].to_numpy(float), "#2ca02c", "square"),
            ("Fragment", summary["stable_fragment_current_nA"].to_numpy(float), "#d62728", "triangle"),
        ],
        current_ticks,
        (0.0, float(current_ticks[-1])),
    )
    _draw_panel(
        image,
        rectangles[1],
        "(b)",
        rf,
        "Fraction (%)",
        [
            ("Transmission", summary["stable_transmission_percent"].to_numpy(float), "#1f77b4", "circle"),
            ("Intact", summary["stable_intact_transmission_percent"].to_numpy(float), "#2ca02c", "square"),
            ("Fragmented", summary["stable_fragmentation_percent_of_exit"].to_numpy(float), "#d62728", "triangle"),
        ],
        np.asarray([0, 25, 50, 75, 100], dtype=float),
        (0.0, 105.0),
    )

    ke_p05 = summary["stable_ke_ev_p05_weighted"].to_numpy(float)
    ke_p50 = summary["stable_ke_ev_p50_weighted"].to_numpy(float)
    ke_p95 = summary["stable_ke_ev_p95_weighted"].to_numpy(float)
    positive_ke = ke_p05[np.isfinite(ke_p05) & (ke_p05 > 0.0)]
    ke_min = 10.0 ** math.floor(math.log10(float(np.min(positive_ke))))
    ke_max = 10.0 ** math.ceil(math.log10(float(np.nanmax(ke_p95))))
    ke_ticks = 10.0 ** np.arange(math.log10(ke_min), math.log10(ke_max) + 0.1)
    _draw_panel(
        image,
        rectangles[2],
        "(c)",
        rf,
        "Exit KE (eV, log scale)",
        [("P50; band P05-P95", ke_p50, "#5b3f99", "circle")],
        ke_ticks,
        (ke_min, ke_max),
        log_y=True,
        band=(ke_p05, ke_p95, "#7b6fd0"),
    )

    temp_p05 = summary["stable_temperature_k_p05_weighted"].to_numpy(float)
    temp_p50 = summary["stable_temperature_k_p50_weighted"].to_numpy(float)
    temp_p95 = summary["stable_temperature_k_p95_weighted"].to_numpy(float)
    temp_ticks = _nice_linear_ticks(float(np.nanmax(temp_p95)) * 1.05)
    _draw_panel(
        image,
        rectangles[3],
        "(d)",
        rf,
        "Internal temperature (K)",
        [("P50; band P05-P95", temp_p50, "#9c5d12", "circle")],
        temp_ticks,
        (0.0, float(temp_ticks[-1])),
        band=(temp_p05, temp_p95, "#e09f3e"),
    )
    image.convert("RGB").save(output, dpi=(dpi, dpi), quality=95)


def plot_summary(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    if plt is None:
        plot_summary_pillow(summary, output, dpi)
        return
    summary = summary.sort_values("rf_vpeak")
    rf = summary["rf_vpeak"].to_numpy(dtype=float)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(rf, summary["stable_ion_current_nA"], "o-", color="#1f77b4", label="Total")
    ax.plot(rf, summary["stable_parent_current_nA"], "s-", color="#2ca02c", label="Parent")
    ax.plot(rf, summary["stable_fragment_current_nA"], "^-", color="#d62728", label="Fragment")
    ax.set_ylabel("Exit current (nA)")
    ax.legend(frameon=False, ncol=3, fontsize=8)

    ax = axes[0, 1]
    ax.plot(rf, summary["stable_transmission_percent"], "o-", color="#1f77b4", label="Transmission")
    ax.plot(
        rf,
        summary["stable_intact_transmission_percent"],
        "s-",
        color="#2ca02c",
        label="Intact transmission",
    )
    ax.plot(
        rf,
        summary["stable_fragmentation_percent_of_exit"],
        "^-",
        color="#d62728",
        label="Fragmented at exit",
    )
    ax.set_ylabel("Fraction (%)")
    ax.set_ylim(-2, 103)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ke_p05 = summary["stable_ke_ev_p05_weighted"].to_numpy(dtype=float)
    ke_p50 = summary["stable_ke_ev_p50_weighted"].to_numpy(dtype=float)
    ke_p95 = summary["stable_ke_ev_p95_weighted"].to_numpy(dtype=float)
    ax.fill_between(rf, ke_p05, ke_p95, color="#7b6fd0", alpha=0.22, linewidth=0)
    ax.plot(rf, ke_p50, "o-", color="#5b3f99", label="P50; band P05-P95")
    ax.set_yscale("log")
    ax.set_ylabel("Exit KE (eV, log scale)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    temp_p05 = summary["stable_temperature_k_p05_weighted"].to_numpy(dtype=float)
    temp_p50 = summary["stable_temperature_k_p50_weighted"].to_numpy(dtype=float)
    temp_p95 = summary["stable_temperature_k_p95_weighted"].to_numpy(dtype=float)
    ax.fill_between(rf, temp_p05, temp_p95, color="#e09f3e", alpha=0.24, linewidth=0)
    ax.plot(rf, temp_p50, "o-", color="#9c5d12", label="P50; band P05-P95")
    ax.set_ylabel("Internal temperature (K)")
    ax.legend(frameon=False, fontsize=8)

    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for ax, label in zip(axes.flat, panel_labels):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        ax.set_xlabel(r"RF amplitude ($V_{peak}$)")
        ax.set_xlim(float(np.min(rf)) - 3.0, float(np.max(rf)) + 3.0)
        ax.set_xticks(np.arange(0, max(201, int(np.max(rf)) + 1), 25))
        ax.grid(True, color="#d9d9d9", linewidth=0.55, alpha=0.75)

    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-ms RF-sweep transport and fragmentation metrics.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-summary-csv", type=Path)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--bin-width-ms", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    root = args.root.resolve()
    output_csv = args.output_csv or root / f"{root.name}_multimetric_per_ms.csv"
    output_summary_csv = args.output_summary_csv or root / f"{root.name}_stable_summary.csv"
    output_plot = args.output_plot or root / f"{root.name}_multimetric_overview.png"
    case_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and CASE_PATTERN.search(path.name)),
        key=lambda path: int(CASE_PATTERN.search(path.name).group("rf")),
    )
    if not case_dirs:
        raise SystemExit(f"No RF case directories found under {root}")

    rows: list[dict[str, object]] = []
    case_summaries: list[dict[str, object]] = []
    for case_dir in case_dirs:
        case_rows, case_summary = load_case(case_dir, args.bin_width_ms)
        rows.extend(case_rows)
        case_summaries.append(case_summary)

    per_ms = pd.DataFrame(rows).sort_values(["rf_vpeak", "ms_bin"])
    summary = pd.DataFrame(case_summaries).sort_values("rf_vpeak")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    per_ms.to_csv(output_csv, index=False, float_format="%.12g")
    summary.to_csv(output_summary_csv, index=False, float_format="%.12g")
    plot_summary(summary, output_plot, args.dpi)

    print(f"cases={len(case_dirs)} rows={len(per_ms)}")
    print(f"csv={output_csv}")
    print(f"summary_csv={output_summary_csv}")
    print(f"plot={output_plot}")


if __name__ == "__main__":
    main()
