"""Overlay external electric field, CFD flow, mask and ion snapshots on one axis.

Run with ``python -m src.data.diagnostics.plot_field_flow_ion_overlay --help``.
Ion positions and velocity arrows are sampled from one or more exported snapshots;
this diagnostic intentionally does not connect points into trajectories.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import plot_field_flow_mask_alignment as alignment
from . import plot_vr_trajectory_overlay as ions
from .plot_style import apply_paper_style

ROOT = Path(__file__).resolve().parents[3]


def resolve_input(value: str | Path) -> Path:
    """Resolve an explicit path without guessing replacements for missing fields."""
    path = Path(value)
    for candidate in (path, ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Input not found: {value}. Supply its current path explicitly.")


def electric_components(simion: dict, selection: tuple, mode: str,
                        peak_v: float, reference_v: float, phase_deg: float):
    """Return external (Er, Ez) in V/m; RF peak is amplitude, not instantaneous RF."""
    if not np.isfinite(reference_v) or reference_v <= 0:
        raise ValueError("RF reference peak voltage must be positive and finite.")
    dc = [np.asarray(simion[f"e_dc_{axis}_v_per_m"])[selection] for axis in ("r", "z")]
    rf = [np.asarray(simion[f"e_rf_{axis}_v_per_m"])[selection] * peak_v / reference_v
          for axis in ("r", "z")]
    if mode == "dc":
        return dc
    if mode == "rf-peak":
        return rf
    factor = math.cos(math.radians(phase_deg))
    return [d + factor * r for d, r in zip(dc, rf)]


def load_case(args):
    """Select requested timestamps and resolve report-based field defaults."""
    case = Path(args.case_dir).resolve()
    report_path = case / "simulation_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    config = report.get("config", {})
    recorded_field = config.get("static_field_path")
    raw_field = args.field or recorded_field
    if not raw_field:
        raise ValueError("No recorded field path; specify --field.")
    field = resolve_input(raw_field)
    override = bool(recorded_field and Path(recorded_field).as_posix() != field.as_posix())
    if recorded_field:
        try:
            override = resolve_input(recorded_field) != field
        except FileNotFoundError:
            override = True
    dt = float(config.get("macro_time_step_s", args.snapshot_time_step_s))
    records = ions._snapshot_records(case, dt)
    if not records:
        raise FileNotFoundError(f"No NPY snapshots in {case}")
    requested = args.snapshot_time_ms
    if requested is None:
        anchors = [records[-1]]
    else:
        anchors = []
        for target_ms in requested:
            anchor = min(records, key=lambda r: abs(r.time_s * 1e3 - target_ms))
            if anchor.macro_step not in {item.macro_step for item in anchors}:
                anchors.append(anchor)
        if not anchors:
            raise ValueError("At least one snapshot time is required.")
    args.case_dir = str(case)
    args.snapshot_file = ""
    # The shared loader accepts a step interval; the index timestamps are authoritative.
    anchor = anchors[-1]
    args.snapshot_time_step_s = anchor.time_s / anchor.macro_step if anchor.macro_step else dt
    peak = args.rf_peak_voltage if args.rf_peak_voltage is not None else config.get("rf_peak_voltage_v", 0.0)
    phase = args.rf_phase_deg
    if phase is None:
        cycles = float(config.get("rf_frequency_hz", 0.0)) * anchor.time_s
        phase = math.degrees(float(config.get("rf_phase_rad", 0.0))) + 360 * (cycles % 1)
    return case, config, field, recorded_field, override, records, anchors, float(peak), float(phase)


def load_local_fields(args, baked, config):
    """Subset on constant strides so streamplot receives a uniformly spaced mesh."""
    grid = baked["grid"]
    z_all, r_all = [np.asarray(grid[f"{a}_coords_m"]) * 1e3 for a in ("z", "r")]
    zi = alignment._window_indices(z_all, args.z_min_mm, args.z_max_mm, "z")
    ri = alignment._window_indices(r_all, args.r_min_mm, args.r_max_mm, "r")
    zi, ri = zi[::max(1, math.ceil(len(zi) / 900))], ri[::max(1, math.ceil(len(ri) / 400))]
    if min(len(zi), len(ri)) < 2:
        raise ValueError("Plot window must contain at least two grid points on each axis.")
    z, r = z_all[zi], r_all[ri]
    zg, rg = np.meshgrid(z, r)
    metal = np.zeros(zg.shape, dtype=bool)
    mask_input = args.electrode_mask
    if mask_input is None:
        mask_input = config.get("electrode_mask_cache_path") or config.get("electrode_mask_path")
    mask_path = None
    if mask_input and str(mask_input).lower() != "none":
        mask_path = resolve_input(mask_input)
        mask = alignment.load_electrode_mask(source_path=mask_path, field_metadata=baked,
                                            z_offset_m=config.get("electrode_mask_z_offset_m"))
        lookup = RegularGridInterpolator((mask.r_coords_m, mask.z_coords_m), mask.metal_mask,
                                         method="nearest", bounds_error=False, fill_value=0)
        metal = lookup(np.stack((rg * 1e-3, zg * 1e-3), axis=-1)).astype(bool)
    selection = np.ix_(ri, zi)
    vr, vz = [np.asarray(baked["fluent"][f"v_{a}_m_per_s"])[selection] for a in ("r", "z")]
    return z, r, selection, metal, vr, vz, mask_path


def load_snapshot_clouds(args, anchors):
    """Load one point cloud per selected snapshot without merging timestamps."""
    clouds = []
    for anchor in anchors:
        local_args = copy.copy(args)
        local_args.snapshot_file = ""
        # The shared snapshot loader accepts one scalar target per call.
        local_args.snapshot_time_ms = anchor.time_s * 1e3
        local_args.snapshot_macro = None
        local_args.snapshot_point_mode = "latest"
        cloud = ions._load_snapshot_point_cloud(local_args)
        if cloud is not None:
            clouds.append(cloud)
    return clouds


def draw_ion_velocity_arrows(ax, cloud, color: str, args) -> int:
    """Draw direction-normalized ion velocity arrows and return the arrow count."""
    if cloud is None or cloud.plotted_count == 0:
        return 0
    stride = max(1, int(args.ion_arrow_stride))
    indices = np.arange(0, cloud.plotted_count, stride, dtype=np.int64)
    if indices.size > args.max_ion_arrows:
        indices = np.linspace(0, cloud.plotted_count - 1, args.max_ion_arrows, dtype=np.int64)
    z = cloud.z_mm[indices]
    r = cloud.r_mm[indices]
    vz = cloud.vz_m_per_s[indices]
    vr = cloud.vr_m_per_s[indices]
    speed = np.hypot(vz, vr)
    valid = np.isfinite(z) & np.isfinite(r) & np.isfinite(vz) & np.isfinite(vr) & (speed > 0)
    if not np.any(valid):
        return 0
    z, r, vz, vr, speed = z[valid], r[valid], vz[valid], vr[valid], speed[valid]
    length = float(args.ion_arrow_length_mm)
    ax.quiver(z, r, length * vz / speed, length * vr / speed,
              angles="xy", scale_units="xy", scale=1.0, color=color,
              alpha=float(args.ion_arrow_alpha), width=float(args.ion_arrow_width),
              headwidth=3.2, headlength=4.2, headaxislength=3.7, zorder=6)
    return int(z.size)


def resolve_snapshot_colors(args, count: int) -> np.ndarray:
    """Assign explicit colors in snapshot order, or use the default colormap."""
    if args.snapshot_point_color is not None:
        colors = args.snapshot_point_color
        if len(colors) == 1:
            colors = colors * count
        elif len(colors) != count:
            raise ValueError(
                f"--snapshot-point-color requires one color or {count} colors "
                f"(one per distinct selected snapshot); received {len(colors)}.")
        return matplotlib.colors.to_rgba_array(colors)
    if count == 1:
        return matplotlib.colors.to_rgba_array(["#ef8a17"])
    if args.snapshot_cmap == "orange-yellow-green":
        cmap = LinearSegmentedColormap.from_list(
            "orange-yellow-green", ["#ef8a17", "#e6c62f", "#90b83e", "#23944f"])
        return cmap(np.linspace(0, 1, count))
    return plt.get_cmap(args.snapshot_cmap)(np.linspace(0.12, 0.88, count))


def make_plot(args) -> Path:
    """Compose one publication-style axis and write image plus provenance sidecar."""
    case, config, field, recorded, override, records, anchors, peak, phase = load_case(args)
    snapshot_colors = resolve_snapshot_colors(args, len(anchors))
    baked = alignment._load_baked(field)
    z, r, selection, metal, vr, vz, mask_path = load_local_fields(args, baked, config)
    reference = config.get("rf_reference_peak_voltage_v") or baked["simion"].get("rf_reference_peak_voltage_v", 1.0)
    er, ez = electric_components(baked["simion"], selection, args.electric_mode, peak, reference, phase)
    magnitude = np.ma.masked_where(metal, np.hypot(er, ez))
    clouds = load_snapshot_clouds(args, anchors)
    apply_paper_style(font_scale=args.paper_font_scale, line_scale=1.25)
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.21 if override else 0.16, top=0.82)
    gas = {"vr": vr, "vz": vz, "speed": np.hypot(vr, vz)}[args.gas_quantity]
    gas = np.ma.masked_where(metal, gas)
    norm = ions._velocity_norm(gas.compressed(), args.vmin, args.vmax)
    levels = np.linspace(norm.vmin, norm.vmax, args.levels)
    color = ax.contourf(z, r, gas, levels=levels, cmap=args.gas_cmap, norm=norm, extend="both", alpha=0.8)
    cax = fig.add_axes([0.91, 0.23, 0.018, 0.52])
    label = {"vr": r"Gas $v_r$ [m/s]", "vz": r"Gas $v_z$ [m/s]", "speed": "Gas speed [m/s]"}[args.gas_quantity]
    fig.colorbar(color, cax=cax, label=label)
    zg, rg = np.meshgrid(z, r)
    if metal.any():
        alignment._draw_mask_overlay(ax, zg, rg, metal.astype(float), alpha=1.0, contour_width=1.0)
    gas_exclusion = metal.copy()
    fmeta = baked["fluent"]
    if fmeta.get("fluent_capillary_external_mask_enabled"):
        gas_exclusion |= ((zg < float(fmeta["fluent_capillary_exit_z_m"]) * 1e3)
                          & (rg <= float(fmeta["fluent_capillary_radius_m"]) * 1e3))
    if args.gas_streamlines:
        ax.streamplot(z, r, np.ma.masked_where(gas_exclusion, vz), np.ma.masked_where(gas_exclusion, vr),
                      color="#424242", density=args.gas_streamline_density, linewidth=0.7,
                      arrowsize=0.85, zorder=3)
    positive = magnitude.compressed()
    positive = positive[np.isfinite(positive) & (positive > 0)]
    e_levels = args.electric_levels
    if e_levels is None:
        lo, hi = np.percentile(positive, [15, 97]) if positive.size else (1, 10)
        e_levels = np.geomspace(max(lo, 1e-12), max(hi, lo * 1.01), 6).tolist()
    if positive.size and np.ptp(positive) > 0:
        contours = ax.contour(z, r, magnitude, levels=e_levels, colors=args.electric_color,
                              linewidths=args.electric_linewidth, zorder=4)
        if args.electric_labels:
            ax.clabel(contours, fontsize=8 * args.paper_font_scale, fmt="%.2g", inline=True)
    snapshot_points = []
    arrow_counts = []
    for anchor, cloud, snapshot_color in zip(anchors, clouds, snapshot_colors):
        local_args = copy.copy(args)
        local_args.snapshot_point_color = matplotlib.colors.to_hex(snapshot_color)
        local_args.snapshot_point_color_by = "none"
        ions._draw_snapshot_points(ax, cloud, local_args)
        snapshot_points.append(cloud)
        arrow_counts.append(draw_ion_velocity_arrows(ax, cloud, snapshot_color, args))
    handles = [Line2D([], [], color=args.electric_color, lw=args.electric_linewidth,
                      label={"dc": r"$|E_{DC}|$ [V/m]", "rf-peak": r"$|E_{RF,peak}|$ [V/m]",
                             "instantaneous": r"$|E_{DC}+E_{RF}(t)|$ [V/m]"}[args.electric_mode])]
    if args.gas_streamlines:
        handles.append(Line2D([], [], color="#424242", lw=1, label="Gas streamlines"))
    for anchor, snapshot_color in zip(anchors, snapshot_colors):
        handles.append(Line2D([], [], marker="o", ls="-", lw=1.0, markersize=5,
                              color=snapshot_color,
                              label=f"{anchor.time_s * 1e3:.4g} ms: ions + velocity arrows"))
    handles.append(Patch(facecolor="#d9d9d9", label="Electrodes"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=3, frameon=False, borderaxespad=0, columnspacing=1.4)
    ax.set(xlim=(args.z_min_mm, args.z_max_mm), ylim=(args.r_min_mm, args.r_max_mm), xlabel="z [mm]", ylabel="r [mm]")
    ax.tick_params(direction="in", top=True, right=True)
    if args.title:
        fig.suptitle(args.title, y=0.99)
    if override:
        fig.text(0.08, 0.035, "FIELD OVERRIDE PREVIEW: supplied field differs from the recorded path; equivalence unverified.",
                 fontsize=9, color="#9b2626")
    output = Path(args.output) if args.output else case / "field_flow_ion_overlay.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    selected_times = [anchor.time_s * 1e3 for anchor in anchors]
    selected_local_points = [cloud.total_local_points for cloud in snapshot_points]
    selected_plotted_points = [cloud.plotted_count for cloud in snapshot_points]
    summary = dict(case_dir=str(case), field=str(field), recorded_field=recorded, field_override=override,
                   electrode_mask=str(mask_path) if mask_path else None,
                   snapshots=[str(anchor.path) for anchor in anchors],
                   snapshot_times_ms=selected_times,
                   snapshot_colors_hex=[matplotlib.colors.to_hex(c) for c in snapshot_colors],
                   snapshot_local_points=selected_local_points,
                   snapshot_plotted_points=selected_plotted_points,
                   ion_arrows_drawn=arrow_counts, electric_mode=args.electric_mode,
                   rf_peak_voltage_v=peak, rf_reference_peak_voltage_v=reference, rf_phase_deg=phase % 360,
                   electric_levels_v_per_m=list(e_levels), gas_quantity=args.gas_quantity,
                   config=vars(args), space_charge_field_included=False,
                   notes="External E only. Ion trajectories are disabled; points and direction-normalized velocity arrows are shown independently for each selected snapshot.")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {output.resolve()}\nSnapshots: {len(snapshot_points)}; points: {selected_plotted_points}; arrows: {arrow_counts}")
    if override:
        print("WARNING: explicit field override; this image does not establish agreement with the original run field.")
    return output


def parse_args(argv=None):
    """Expose plot-only controls; infer RF settings and input paths from the case report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", required=True)
    p.add_argument("--field", help="Defaults to the case report; explicit replacements are flagged.")
    p.add_argument("--electrode-mask", help="PATXT/NPZ or none; default is the recorded cache/source.")
    for name, default in (("z-min", 4.5), ("z-max", 14.0), ("r-min", 0.0), ("r-max", 3.0)):
        p.add_argument(f"--{name}-mm", type=float, default=default)
    p.add_argument("--snapshot-time-ms", nargs="+", type=float,
                   help="One or more nearest saved snapshot times in ms; default latest.")
    p.add_argument("--snapshot-time-step-s", type=float, default=5e-8, help="Only a fallback if report/index lacks timing.")
    p.add_argument("--snapshot-point-max", type=int, default=20000)
    p.add_argument("--snapshot-point-size", type=float, default=12)
    p.add_argument("--snapshot-point-color", nargs="+", metavar="COLOR",
                   help="One color for all snapshots, or one color per distinct selected snapshot "
                        "in input order (e.g. green orange yellow). Overrides --snapshot-cmap.")
    p.add_argument("--snapshot-point-alpha", type=float, default=0.8)
    p.add_argument("--snapshot-cmap", default="orange-yellow-green",
                   help="orange-yellow-green (default) or a Matplotlib colormap for multiple snapshot times.")
    p.add_argument("--ion-arrow-stride", type=int, default=10,
                   help="Draw every Nth point in each snapshot.")
    p.add_argument("--max-ion-arrows", type=int, default=5000,
                   help="Maximum arrows per snapshot after stride sampling.")
    p.add_argument("--ion-arrow-length-mm", type=float, default=0.25,
                   help="Direction-normalized arrow length in plot coordinates.")
    p.add_argument("--ion-arrow-width", type=float, default=0.002)
    p.add_argument("--ion-arrow-alpha", type=float, default=0.75)
    p.add_argument("--electric-mode", choices=["dc", "rf-peak", "instantaneous"], default="rf-peak")
    p.add_argument("--rf-peak-voltage", type=float, help="Vpeak, inferred from report by default.")
    p.add_argument("--rf-phase-deg", type=float, help="Instantaneous RF angle; default omega*t_snapshot + initial phase.")
    p.add_argument("--electric-levels", nargs="+", type=float, help="Electric magnitude contour levels in V/m.")
    p.add_argument("--electric-color", default="#713b95")
    p.add_argument("--electric-linewidth", type=float, default=1.2)
    p.add_argument("--electric-labels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gas-quantity", choices=["vr", "vz", "speed"], default="vr")
    p.add_argument("--gas-cmap", default="RdBu_r")
    p.add_argument("--vmin", type=float)
    p.add_argument("--vmax", type=float)
    p.add_argument("--levels", type=int, default=50)
    p.add_argument("--gas-streamlines", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gas-streamline-density", type=float, default=0.9)
    p.add_argument("--font-scale", "--paper-font-scale", dest="paper_font_scale", type=float, default=1.18,
                   help="Scale all publication figure text; aliases are equivalent.")
    p.add_argument("--figsize", nargs=2, type=float, default=[12, 5.8])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default="")
    p.add_argument("--output")
    a = p.parse_args(argv)
    if a.snapshot_point_color is not None:
        for color in a.snapshot_point_color:
            if not matplotlib.colors.is_color_like(color):
                p.error(f"Invalid --snapshot-point-color {color!r}; use color names or quoted hex codes.")
    if (a.snapshot_point_max < 1 or a.levels < 2 or a.ion_arrow_stride < 1
            or a.max_ion_arrows < 1 or a.ion_arrow_length_mm <= 0 or a.ion_arrow_width <= 0
            or a.paper_font_scale <= 0):
        p.error("Require positive snapshot/arrow/font settings and levels >= 2.")
    if a.vmin is not None and a.vmax is not None and a.vmin >= a.vmax:
        p.error("vmin must be less than vmax.")
    if a.electric_levels is not None and (not np.all(np.isfinite(a.electric_levels)) or
                                         np.any(np.diff(a.electric_levels) <= 0)):
        p.error("electric-levels must be finite and strictly increasing.")
    a.show_snapshot_points, a.snapshot_point_color_by = True, "none"
    return a


def main():
    make_plot(parse_args())


if __name__ == "__main__":
    main()
