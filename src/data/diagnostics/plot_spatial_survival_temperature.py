"""Plot local precursor survival over spatially resolved ion internal temperature."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as effects
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from ..boundary_masks import load_electrode_mask
from .plot_collision_density_comparison import select_records, temporal_weights
from .plot_vr_trajectory_overlay import _snapshot_records


def snapshot_moments(data, columns, r_edges_mm, z_edges_mm):
    required = ("r_m", "z_m", "teff_k", "represented_real_ions",
                "parent_real_ions_remaining", "status_code")
    if any(name not in columns for name in required):
        raise ValueError("Snapshots must contain position, internal temperature, weight, parent weight and status.")
    r, z, temperature, weight, parent, status = [data[:, columns.index(k)] for k in required]
    active = status == 0
    if (not np.isfinite(np.array([r[active], z[active], temperature[active], weight[active], parent[active]])).all()
            or np.any(r[active] < 0) or np.any(temperature[active] <= 0) or np.any(weight[active] < 0)
            or np.any(parent[active] < 0) or np.any(parent[active] > weight[active]+1e-7)):
        raise ValueError("Invalid active-ion position, temperature or precursor weight.")
    active &= weight > 0
    r, z, temperature, weight, parent = [v[active] for v in (r, z, temperature, weight, parent)]
    r, z = r*1e3, z*1e3
    bins = (r_edges_mm, z_edges_mm)
    return {
        "cell_weight": np.histogram2d(r, z, bins=bins, weights=weight)[0],
        "cell_weight_temperature": np.histogram2d(r, z, bins=bins, weights=weight*temperature)[0],
        "cell_observations": np.histogram2d(r, z, bins=bins)[0],
        # The survival profile includes all radii, independently of the heatmap crop.
        "axial_weight": np.histogram(z, bins=z_edges_mm, weights=weight)[0],
        "axial_parent_weight": np.histogram(z, bins=z_edges_mm, weights=parent)[0],
    }


def finalize_moments(moments):
    cell_weight, axial_weight = moments["cell_weight"], moments["axial_weight"]
    temperature = np.divide(moments["cell_weight_temperature"], cell_weight,
                            out=np.full(cell_weight.shape, np.nan), where=cell_weight > 0)
    survival = np.divide(100*moments["axial_parent_weight"], axial_weight,
                        out=np.full(axial_weight.shape, np.nan), where=axial_weight > 0)
    return temperature, survival


def matching_physics(reports):
    first = reports[0]
    for report in reports:
        if report["template"] != first["template"]:
            raise ValueError("Ion templates differ between selected cases.")
        for key in first["config"].keys() | report["config"].keys():
            if key not in ("rf_peak_voltage_v", "static_field_path") and first["config"].get(key) != report["config"].get(key):
                raise ValueError(f"Selected cases differ in {key}, not just RF/DC settings.")
        if report["config"]["fragmentation_mode"] != "transport":
            raise ValueError("Local precursor survival requires transport-mode weight bookkeeping.")


def read_source(path, provenance):
    raw = path.read_bytes()
    provenance[str(path.resolve())] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return raw


def exit_statistics(case, report, start_ms, end_ms, provenance):
    raw = read_source(case/"terminal_events.csv", provenance)
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    weight, parent, times, ids = [], [], [], set()
    for row in rows:
        time_s = float(row["event_time_s"])
        if row["status"] != "z_exit" or not start_ms*1e-3 <= time_s <= end_ms*1e-3+1e-14:
            continue
        track = int(row["track_id"])
        if track in ids:
            raise ValueError("Duplicate terminal track in the exit population.")
        ids.add(track)
        weight.append(float(row["represented_real_ions"]))
        parent.append(float(row["parent_real_ions_remaining"]))
        times.append(time_s)
    weight, parent, times = np.asarray(weight), np.asarray(parent), np.asarray(times)
    if not weight.size or not np.isfinite(weight).all() or not np.isfinite(parent).all() or np.any(weight <= 0):
        raise ValueError("No valid exit population in the requested window.")
    if np.any(parent < 0) or np.any(parent > weight+1e-7):
        raise ValueError("Invalid precursor exit weight.")
    duration_s = (end_ms-start_ms)*1e-3
    current_na = abs(report["template"]["charge_c"])*weight.sum()/duration_s*1e9
    edges = np.linspace(start_ms*1e-3, end_ms*1e-3, 7)
    bin_weight = np.histogram(times, bins=edges, weights=weight)[0]
    return {"exit_survival_percent": float(100*parent.sum()/weight.sum()),
            "exit_current_na": float(current_na), "exit_tracks": len(ids),
            "exit_transmission_percent": float(current_na*1e-9/report["config"]["ion_current_a"]*100),
            "exit_equal_time_bin_current_cv_percent": float(100*bin_weight.std()/bin_weight.mean()),
            "exit_represented_weight": float(weight.sum()), "exit_parent_weight": float(parent.sum())}


def load_case(case, rf, dc, args, r_edges, z_edges, provenance):
    report = json.loads(read_source(case/"simulation_summary.json", provenance).decode("utf-8-sig"))
    config = report["config"]
    if config["rf_peak_voltage_v"] != rf:
        raise ValueError(f"RF label does not match saved settings: {case}")
    source = report["field_metadata"]["simion"].get("dc_source_path", "")
    match = re.search(r"dc(\d+)eV", source)
    if match is None or float(match[1]) != dc:
        raise ValueError(f"DC label cannot be verified against saved field metadata: {case}")
    if report["result"].get("numerical_failure"):
        raise ValueError(f"Saved run reports a numerical failure: {case}")
    columns = json.loads(read_source(case/"metadata.json", provenance).decode("utf-8-sig"))["columns"]
    read_source(case/"snapshots_index.csv", provenance)
    records = _snapshot_records(case, config["macro_time_step_s"])
    unique = {record.time_s: record for record in records}
    records = select_records(list(unique.values()), args.start_ms, args.end_ms, 1)
    exposure = temporal_weights(np.array([record.time_s for record in records]))
    total = None
    samples = []
    for record, time_weight in zip(records, exposure):
        data = np.load(io.BytesIO(read_source(record.path, provenance)), allow_pickle=False)
        moments = snapshot_moments(data, columns, r_edges, z_edges)
        if total is None:
            total = {key: np.zeros_like(value) for key, value in moments.items()}
        for key in total:
            total[key] += (1. if key == "cell_observations" else time_weight)*moments[key]
        samples.append({"time_ms": record.time_s*1e3, "time_weight": float(time_weight), "file": str(record.path)})
    temperature, survival = finalize_moments(total)
    if not np.isfinite(temperature).any():
        raise ValueError(f"No ion-temperature samples in the display window: {case}")
    stats = {"rf_vpeak": rf, "dc_bias_v": dc, "case": str(case), "snapshots": samples,
             "heatmap_radius_coverage_percent": float(100*total["cell_weight"].sum()/total["axial_weight"].sum()),
             "cell_mean_temperature_min_k": float(np.nanmin(temperature)),
             "cell_mean_temperature_max_k": float(np.nanmax(temperature)),
             "hybrid_joint_gaussian_steps": report["collision_physics"].get("hybrid_joint_gaussian_steps"),
             **exit_statistics(case, report, args.start_ms, args.end_ms, provenance)}
    return {"rf": rf, "dc": dc, "report": report, "temperature": temperature,
            "survival": survival, "moments": total, "statistics": stats}


def geometry_grid(mask, z_min, z_max, r_max):
    zi = np.flatnonzero((mask.z_coords_m*1e3 >= z_min) & (mask.z_coords_m*1e3 <= z_max))[::5]
    ri = np.flatnonzero((mask.r_coords_m*1e3 >= 0) & (mask.r_coords_m*1e3 <= r_max))[::5]
    if min(len(ri), len(zi)) < 2:
        raise ValueError("Plot window is not resolved by the electrode mask.")
    return mask.z_coords_m[zi]*1e3, mask.r_coords_m[ri]*1e3, mask.metal_mask[np.ix_(ri, zi)].astype(bool)


def comparison_cases(cases):
    rfs = sorted({case["rf"] for case in cases})
    dcs = sorted({case["dc"] for case in cases})
    lookup = {(case["rf"], case["dc"]): case for case in cases}
    if len(lookup) != len(cases) or len(cases) != len(rfs)*len(dcs):
        raise ValueError("Comparison layout requires every RF/DC combination exactly once.")
    return [lookup[rf, dc] for dc in dcs for rf in rfs], rfs, dcs


@plt.rc_context()
def render_group(cases, r_edges, z_edges, geometry, args, output):
    plt.rcParams.update({"font.family": "Arial", "font.size": 14, "axes.linewidth": 1.2,
                         "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.default": "regular", "savefig.facecolor": "white"})
    comparison = getattr(args, "layout", "grouped") == "comparison"
    if comparison:
        cases, rfs, dcs = comparison_cases(cases)
        nrows, ncols = len(dcs), len(rfs)
        fig, axes = plt.subplots(nrows, ncols, figsize=(13.8, 7.8), sharex=True, squeeze=False)
        fig.subplots_adjust(left=.075, right=.84, bottom=.10, top=.84, hspace=.17, wspace=.06)
        for col, rf in enumerate(rfs):
            position = axes[0, col].get_position()
            fig.text((position.x0+position.x1)/2, .925, f"RF amplitude: {rf:g} V"+r"$_{peak}$",
                     ha="center", va="center", fontsize=17)
        legend_y, cbar_bounds = .995, [.885, .10, .018, .74]
    else:
        nrows, ncols = len(cases), 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, 8.8), sharex=True, squeeze=False)
        fig.subplots_adjust(left=.115, right=.80, bottom=.09, top=.885, hspace=.29)
        fig.suptitle(f"RF amplitude: {cases[0]['rf']:g} V"+r"$_{peak}$", x=.46, y=.974, fontsize=17)
        legend_y, cbar_bounds = .95, [.895, .09, .023, .795]
    legend = [Line2D([], [], color="black", marker="s", markersize=3.5, lw=1.5, label="Local survival yield"),
              Line2D([], [], color="#b2182b", marker="D", markersize=4.5, ls="none", label="Exit survival yield")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.455, legend_y), ncol=2, frameon=False, fontsize=12)
    norm = Normalize(args.temperature_min_k, args.temperature_max_k)
    z_centers = (z_edges[:-1]+z_edges[1:])/2
    zm, rm, metal = geometry
    for index, (ax, case) in enumerate(zip(axes.flat, cases)):
        col = index % ncols
        radial = ax.twinx()
        radial.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        image = radial.pcolormesh(z_edges, r_edges, np.ma.masked_invalid(case["temperature"]),
                                  cmap="turbo", norm=norm, shading="flat", rasterized=True)
        if metal.any():
            radial.contourf(zm, rm, metal.astype(float), levels=[.5, 1.5], colors=["#dedede"], zorder=2)
            radial.contour(zm, rm, metal.astype(float), levels=[.5], colors=["#858585"], linewidths=.55, zorder=3)
        radial.set(ylim=(r_edges[0], r_edges[-1]), ylabel="r / mm" if col == ncols-1 else "")
        radial.yaxis.set_minor_locator(AutoMinorLocator(2))
        radial.tick_params(axis="y", which="both", direction="in", length=5)
        if comparison:
            radial.tick_params(axis="y", labelright=col == ncols-1)
        radial.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
        radial.patch.set_visible(False)
        line, = ax.plot(z_centers, case["survival"], color="black", lw=1.65, marker="s", ms=3., markevery=8)
        line.set_path_effects([effects.Stroke(linewidth=3.1, foreground="white"), effects.Normal()])
        config = case["report"]["config"]
        for z_mm in (config["capillary_exit_z_m"]*1e3, config["detector_z_m"]*1e3):
            ax.axvline(z_mm, color="#666666", lw=.7, ls="--", zorder=1)
        exit_sy = case["statistics"]["exit_survival_percent"]
        ax.plot(config["detector_z_m"]*1e3, exit_sy, marker="D", ms=5., color="#b2182b", mec="white", mew=.7, zorder=4)
        ax.set_title(f"({chr(97+index)}) DC bias = {case['dc']:g} V", loc="left", fontsize=14, pad=10)
        sy_text = f"Exit SY = {exit_sy:.1f}%" if exit_sy >= .001 else "Exit SY < 0.001%"
        ax.text(1., 1.035, sy_text, transform=ax.transAxes, ha="right", fontsize=12)
        ax.set(xlim=(z_edges[0], z_edges[-1]), ylim=(-3, 103), ylabel="Survival yield / %" if col == 0 else "")
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xticks(np.arange(z_edges[0], z_edges[-1]+1e-9, 10.))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", direction="in", top=True, right=False, length=6)
        ax.tick_params(which="minor", direction="in", top=True, right=False, length=3)
        if comparison:
            ax.tick_params(axis="y", labelleft=col == 0)
    for ax in axes[-1, :]:
        ax.set_xlabel("z / mm")
    cax = fig.add_axes(cbar_bounds)
    all_temperatures = np.concatenate([case["temperature"].ravel() for case in cases])
    below = np.any(all_temperatures < args.temperature_min_k)
    above = np.any(all_temperatures > args.temperature_max_k)
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    bar = fig.colorbar(image, cax=cax, extend=extend, extendrect=True)
    if comparison:
        ticks = np.unique(np.r_[args.temperature_min_k,
                               np.arange(np.ceil(args.temperature_min_k/200)*200, args.temperature_max_k, 200),
                               args.temperature_max_k])
        bar.set_ticks(ticks)
    bar.set_label("Mean internal temperature / K", fontsize=15, labelpad=10)
    bar.ax.tick_params(direction="in", labelsize=12)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)


def make_plots(args):
    r_edges = np.linspace(0., args.r_max_mm, round(args.r_max_mm/args.dr_mm)+1)
    z_edges = np.linspace(args.z_min_mm, args.z_max_mm, round((args.z_max_mm-args.z_min_mm)/args.dz_mm)+1)
    provenance, cases, labels = {}, [], set()
    for rf_text, dc_text, path in args.case:
        rf, dc = float(rf_text), float(dc_text)
        if (rf, dc) in labels:
            raise ValueError("Duplicate RF/DC case label.")
        labels.add((rf, dc))
        cases.append(load_case(Path(path).resolve(), rf, dc, args, r_edges, z_edges, provenance))
    matching_physics([case["report"] for case in cases])
    first = cases[0]["report"]
    mask_path = Path(first["config"]["electrode_mask_cache_path"]).resolve()
    read_source(mask_path, provenance)
    mask = load_electrode_mask(source_path=mask_path, field_metadata=first["field_metadata"],
                               z_offset_m=first["config"]["electrode_mask_z_offset_m"])
    geometry = geometry_grid(mask, args.z_min_mm, args.z_max_mm, args.r_max_mm)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stored = {"r_edges_mm": r_edges, "z_edges_mm": z_edges}
    outputs = []
    if args.layout == "comparison":
        ordered, rfs, dcs = comparison_cases(cases)
        rf_label, dc_label = "_".join(f"{rf:g}" for rf in rfs), "_".join(f"{dc:g}" for dc in dcs)
        groups = [(f"LeuEnk_RF{rf_label}_DC{dc_label}_SY_temperature_comparison.png", ordered)]
    else:
        groups = [(f"LeuEnk_RF{rf:g}_SY_spatial_temperature.png",
                   sorted((case for case in cases if case["rf"] == rf), key=lambda case: case["dc"]))
                  for rf in sorted({case["rf"] for case in cases})]
    for filename, group in groups:
        output = out/filename
        render_group(group, r_edges, z_edges, geometry, args, output)
        outputs.append(str(output))
        for case in group:
            rf = case["rf"]
            prefix = f"RF{rf:g}_DC{case['dc']:g}"
            stored[prefix+"_temperature_k"] = case["temperature"]
            stored[prefix+"_local_survival_percent"] = case["survival"]
            for key, value in case["moments"].items():
                stored[prefix+"_"+key] = value
            print(f"RF {rf:g}, DC {case['dc']:g}: exit SY={case['statistics']['exit_survival_percent']:.8g}%, "
                  f"I={case['statistics']['exit_current_na']:.5g} nA; heatmap radius coverage="
                  f"{case['statistics']['heatmap_radius_coverage_percent']:.5f}%", flush=True)
        print(f"Saved: {output}", flush=True)
    np.savez_compressed(out/"spatial_survival_temperature.npz", **stored)
    metadata = {"outputs": outputs, "layout": args.layout,
                "panel_order": [{"rf_vpeak": case["rf"], "dc_bias_v": case["dc"]} for _, group in groups for case in group],
                "time_window_ms": [args.start_ms, args.end_ms],
                "z_window_mm": [args.z_min_mm, args.z_max_mm], "r_window_mm": [0, args.r_max_mm],
                "actual_dz_mm": float(z_edges[1]-z_edges[0]), "actual_dr_mm": float(r_edges[1]-r_edges[0]),
                "temperature_color_range_k": [args.temperature_min_k, args.temperature_max_k],
                "temperature_definition": "sum(time_weight*represented_weight*T_int)/sum(time_weight*represented_weight) in each r,z cell",
                "survival_definition": "100*sum(time_weight*remaining_parent_weight)/sum(time_weight*represented_weight) in each z bin, including all radii",
                "population": "active transport-mode ion packs; total represented weight, not surviving-parent-only temperature",
                "profile_interpretation": "local population composition, not a single trajectory or first-passage cumulative survival; no monotonicity enforcement",
                "exit_marker": "ratio of summed remaining-parent to total represented exit weight for arrivals in the same time window",
                "display": "Arial, inward ticks, black frame, black SY line, red exit marker; raw cell means, no smoothing or gap filling",
                "empty_cells": "NaN and left blank; no extrapolation beyond the injection/exit population",
                "font_family": "Arial", "dpi": args.dpi,
                "cases": [case["statistics"] for case in cases], "sources": provenance}
    (out/"spatial_survival_temperature.summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, action="append", required=True, metavar=("RF_VPEAK", "DC_V", "DIRECTORY"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layout", choices=("grouped", "comparison"), default="grouped",
                        help="Separate RF figures, or one grid with RF columns and DC rows.")
    parser.add_argument("--start-ms", type=float, default=.4)
    parser.add_argument("--end-ms", type=float, default=1.)
    parser.add_argument("--z-min-mm", type=float, default=5.)
    parser.add_argument("--z-max-mm", type=float, default=65.)
    parser.add_argument("--dz-mm", type=float, default=.5)
    parser.add_argument("--r-max-mm", type=float, default=5.)
    parser.add_argument("--dr-mm", type=float, default=.2)
    parser.add_argument("--temperature-min-k", type=float, default=0.)
    parser.add_argument("--temperature-max-k", type=float, default=2000.)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    numbers = [args.start_ms, args.end_ms, args.z_min_mm, args.z_max_mm, args.dz_mm,
               args.r_max_mm, args.dr_mm, args.temperature_min_k, args.temperature_max_k]
    if (not np.isfinite(numbers).all() or args.start_ms < 0 or args.end_ms <= args.start_ms
            or args.z_max_mm <= args.z_min_mm or args.r_max_mm <= 0
            or args.dz_mm <= 0 or args.dr_mm <= 0 or args.r_max_mm < args.dr_mm
            or args.z_max_mm-args.z_min_mm < args.dz_mm or args.dpi < 1
            or args.temperature_max_k <= args.temperature_min_k):
        parser.error("Invalid time, spatial or temperature limits.")
    try:
        labels = np.array([[float(row[0]), float(row[1])] for row in args.case])
    except ValueError:
        parser.error("RF/DC labels must be numeric.")
    if not np.isfinite(labels).all() or np.any(labels[:, 0] < 0):
        parser.error("RF/DC labels must be finite and RF must be nonnegative.")
    return args


if __name__ == "__main__":
    make_plots(parse_args())
