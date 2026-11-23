"""Plot recorded production gas/DC/RF maps with bounded display scales."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
import numpy as np

from . import plot_field_flow_ion_overlay as fields


def select_field_input(recorded_path, requested_path="", *, allow_override=False):
    recorded = fields.resolve_input(recorded_path)
    selected = fields.resolve_input(requested_path) if requested_path else recorded
    if selected != recorded and not allow_override:
        raise ValueError("Field override must match the selected production case unless --allow-field-override is explicit.")
    return recorded, selected


def display_norms(dc_max_v_per_m, electric_min_v_per_m, electric_max_v_per_m, gas_max_m_per_s, *, dc_min_v_per_m=0.):
    limits = np.array([dc_max_v_per_m, electric_min_v_per_m, electric_max_v_per_m, gas_max_m_per_s], float)
    if not np.isfinite(limits).all() or np.any(limits <= 0) or electric_min_v_per_m >= electric_max_v_per_m:
        raise ValueError("Display limits must be positive and finite, with electric minimum < maximum.")
    if not np.isfinite(dc_min_v_per_m) or not 0 <= dc_min_v_per_m < dc_max_v_per_m:
        raise ValueError("DC display limits must satisfy 0 <= minimum < maximum.")
    return {"gas": Normalize(0, gas_max_m_per_s, clip=True),
            "dc": LogNorm(dc_min_v_per_m, dc_max_v_per_m, clip=True) if dc_min_v_per_m else Normalize(0, dc_max_v_per_m, clip=True),
            "rf": LogNorm(electric_min_v_per_m, electric_max_v_per_m, clip=True),
            "total": LogNorm(electric_min_v_per_m, electric_max_v_per_m, clip=True)}


def direction_samples(z_mm, r_mm, er, ez, metal, minimum_v_per_m, spacing_mm):
    """Sample directions only in nonmetal cells above the display threshold."""
    if minimum_v_per_m <= 0 or spacing_mm <= 0 or not np.isfinite([minimum_v_per_m, spacing_mm]).all():
        raise ValueError("Vector threshold and spacing must be positive and finite.")
    sz = max(1, math.ceil(spacing_mm/np.min(np.diff(z_mm))))
    sr = max(1, math.ceil(spacing_mm/np.min(np.diff(r_mm))))
    zz, rr = np.meshgrid(z_mm[::sz], r_mm[::sr])
    er, ez, excluded = er[::sr, ::sz], ez[::sr, ::sz], metal[::sr, ::sz]
    magnitude = np.hypot(er, ez)
    valid = ~excluded & np.isfinite(magnitude) & (magnitude >= minimum_v_per_m)
    return zz[valid], rr[valid], ez[valid]/magnitude[valid], er[valid]/magnitude[valid]


def draw_electric_field_lines(ax, z, r, er, ez, metal, minimum_v_per_m, density):
    magnitude = np.hypot(er, ez)
    excluded = metal | ~np.isfinite(magnitude) | (magnitude < minimum_v_per_m)
    if excluded.all():
        return 0, 0
    before = len(ax.patches)
    streams = ax.streamplot(z, r, np.ma.masked_where(excluded, ez), np.ma.masked_where(excluded, er),
                            color="white", density=density, linewidth=.55, arrowsize=.65,
                            minlength=.08, maxlength=4., integration_direction="both", zorder=4)
    effects = [path_effects.Stroke(linewidth=.95, foreground="#36434b", alpha=.8), path_effects.Normal()]
    streams.lines.set_path_effects(effects)
    arrows = list(ax.patches)[before:]
    for arrow in arrows:
        arrow.set_path_effects(effects)
    return len(streams.lines.get_paths()), len(arrows)


def panel_statistics(data, metal, norm):
    values = np.asarray(data)[~metal]
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Field panel needs finite nonmetal samples.")
    return {"minimum": float(values.min()), "median": float(np.median(values)), "maximum": float(values.max()),
            "display_minimum": float(norm.vmin), "display_maximum": float(norm.vmax),
            "display_scale": "logarithmic" if isinstance(norm, LogNorm) else "linear",
            "fraction_below_display_minimum": float(np.mean(values < norm.vmin)),
            "fraction_above_display_maximum": float(np.mean(values > norm.vmax))}


@plt.rc_context()
def render_figure(args, config, z, r, metal, gas_speed, dc, rf, total_er, total_ez, output):
    font = args.font_family
    plt.rcParams.update({"font.family": font, "font.size": args.font_size,
                         "axes.labelsize": args.font_size+1, "axes.titlesize": args.font_size+1,
                         "xtick.labelsize": args.font_size-1, "ytick.labelsize": args.font_size-1,
                         "mathtext.fontset": "custom", "mathtext.rm": font,
                         "mathtext.it": font+":italic", "mathtext.bf": font+":bold",
                         "mathtext.sf": font, "mathtext.default": "regular", "axes.linewidth": 1.})
    norms = display_norms(args.dc_max_v_per_m, args.electric_min_v_per_m,
                          args.electric_max_v_per_m, args.gas_max_m_per_s,
                          dc_min_v_per_m=getattr(args, "dc_min_v_per_m", 0.))
    total = np.hypot(total_er, total_ez)
    peak = config["rf_peak_voltage_v"]
    gas_electric = getattr(args, "combined_panel", "electric") == "gas-electric"
    electric_overlay = getattr(args, "electric_overlay", "arrows") if gas_electric else "arrows"
    gas_ticks = None if gas_electric else np.linspace(0, args.gas_max_m_per_s, 4)
    if gas_electric:
        norms["gas_electric"] = norms["gas"]
        last_panel = ("gas_electric", f"(d) Gas speed + DC/RF {electric_overlay}, "+rf"$\theta={args.rf_phase_deg:g}^\circ$",
                      gas_speed, "plasma", r"$|\mathbf{u}_g|$ / m s$^{-1}$", gas_ticks)
    else:
        last_panel = ("total", "(d) Instantaneous DC + RF, "+rf"$\theta={args.rf_phase_deg:g}^\circ$",
                      total, "viridis", r"$|\mathbf{E}_{ext}(\theta)|$ / V m$^{-1}$", None)
    definitions = [
        ("gas", "(a) Gas speed", gas_speed, "plasma", r"$|\mathbf{u}_g|$ / m s$^{-1}$", gas_ticks),
        ("dc", "(b) DC field magnitude", dc, getattr(args, "dc_cmap", "Greys"), r"$|\mathbf{E}_{DC}|$ / V m$^{-1}$",
         None if isinstance(norms["dc"], LogNorm) else np.linspace(0, args.dc_max_v_per_m, 5)),
        ("rf", f"(c) RF peak field, {peak:g} V"+r"$_{peak}$", rf, "viridis", r"$|\mathbf{E}_{RF,peak}|$ / V m$^{-1}$", None),
        last_panel,
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), sharex=True, sharey=True,
                             dpi=args.dpi, layout="constrained")
    fig.get_layout_engine().set(rect=(0, .085, 1, .915), hspace=.075, wspace=.045)
    capillary_z_mm = config["capillary_exit_z_m"]*1e3
    capillary_visible = args.z_min_mm <= capillary_z_mm <= args.z_max_mm
    stats, arrow_count = {}, 0
    for ax, (key, title, data, cmap, unit, ticks) in zip(axes.flat, definitions):
        norm = norms[key]
        stats[key] = {"quantity": title[4:], "units": "m/s" if key in ("gas", "gas_electric") else "V/m", "colormap": cmap,
                      **panel_statistics(data, metal, norm)}
        # Only the color mapping is clipped. The field values remain unchanged.
        image = ax.pcolormesh(z, r, np.ma.masked_where(metal, data), cmap=cmap,
                              norm=norm, shading="auto", rasterized=True)
        below = bool(stats[key]["fraction_below_display_minimum"])
        above = bool(stats[key]["fraction_above_display_maximum"])
        extend = "both" if below and above else "min" if below else "max" if above else "neither"
        bar = fig.colorbar(image, ax=ax, pad=.02, fraction=.046, shrink=.94, extend=extend, extendrect=True)
        bar.set_label(unit, fontsize=args.font_size)
        if ticks is not None:
            bar.set_ticks([tick for tick in ticks if norm.vmin <= tick <= norm.vmax])
        bar.ax.tick_params(labelsize=args.font_size-2)
        if metal.any():
            ax.contourf(z, r, metal.astype(float), levels=[.5, 1.5], colors=["#cbd0d3"], zorder=5)
            ax.contour(z, r, metal.astype(float), levels=[.5], colors=["#555d63"], linewidths=.65, zorder=6)
        if capillary_visible:
            ax.axvline(capillary_z_mm, color="#269ac5", ls="--", lw=1.1, zorder=7)
        if key == "dc" and stats[key]["maximum"] < 1e-6:
            background = np.asarray(plt.get_cmap(cmap)(0.))[:3]
            text_color = "white" if background @ np.array([.2126, .7152, .0722]) < .5 else "#30373c"
            ax.text(.5, .15, r"$|\mathbf{E}_{DC}| < 10^{-6}$ V m$^{-1}$", transform=ax.transAxes,
                    ha="center", va="center", fontsize=args.font_size, color=text_color)
        if key == "total" or (key == "gas_electric" and electric_overlay == "arrows"):
            x, y, dx, dy = direction_samples(z, r, total_er, total_ez, metal,
                                              args.electric_min_v_per_m, args.arrow_spacing_mm)
            arrow_count = len(x)
            ax.quiver(x, y, .28*dx, .28*dy, angles="xy", scale_units="xy", scale=1,
                      color="white", edgecolor="#354047", linewidth=.2, width=.003,
                      headwidth=3.1, headlength=4., zorder=4)
        if key == "gas_electric":
            stats[key].update({"background_quantity": "gas speed", "overlay_quantity": "instantaneous external DC+RF electric field direction",
                               "electric_overlay": electric_overlay, "overlay_phase_deg": args.rf_phase_deg,
                               "overlay_minimum_v_per_m": args.electric_min_v_per_m,
                               "electric_field_actual_maximum_v_per_m": float(total[~metal].max())})
            if electric_overlay == "lines":
                paths, arrow_count = draw_electric_field_lines(ax, z, r, total_er, total_ez, metal,
                                                               args.electric_min_v_per_m,
                                                               getattr(args, "electric_line_density", .7))
                stats[key].update({"line_quantity": "instantaneous external DC+RF electric field",
                                   "line_units": "V/m", "line_phase_deg": args.rf_phase_deg,
                                   "line_minimum_v_per_m": args.electric_min_v_per_m,
                                   "field_line_path_count": paths})
            else:
                stats[key].update({"direction_arrow_count": arrow_count, "direction_arrow_length_mm": .28,
                                   "requested_arrow_spacing_mm": args.arrow_spacing_mm})
        ax.set_title(title, loc="left", pad=10)
        ax.set(xlim=(args.z_min_mm, args.z_max_mm), ylim=(args.r_min_mm, args.r_max_mm))
        ax.tick_params(direction="in", top=True, right=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("r / mm")
    for ax in axes[-1, :]:
        ax.set_xlabel("z / mm")
    handles = []
    if capillary_visible:
        capillary = Line2D([], [], color="#269ac5", ls="--", lw=1.1,
                           label=f"Capillary exit (z = {capillary_z_mm:g} mm)")
        handles.append(capillary)
    if gas_electric:
        electric_line = Line2D([], [], color="white", lw=.7, marker=">", markersize=4,
                               markeredgecolor="#36434b", label=("DC + RF electric field lines" if electric_overlay == "lines"
                                                                             else "DC + RF electric field direction (fixed-length arrows)"),
                               path_effects=[path_effects.Stroke(linewidth=1.25, foreground="#36434b"), path_effects.Normal()])
        handles.append(electric_line)
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), frameon=False, fontsize=12, ncol=len(handles))
    frequency = config["rf_frequency_hz"]/1e3
    fig.text(.5, .018, f"RF: {peak:g} V"+r"$_{peak}$"+f", {frequency:g} kHz; snapshot phase: {args.rf_phase_deg:g} deg; external fields only (no space charge)",
             ha="center", va="center", fontsize=11, color=".25")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    return stats, arrow_count


def make_plot(args):
    case = Path(args.case_dir).resolve()
    report = json.loads((case/"simulation_summary.json").read_text(encoding="utf-8-sig"))
    config = report["config"]
    recorded, field = select_field_input(config["static_field_path"], args.field,
                                         allow_override=getattr(args, "allow_field_override", False))
    override = field != recorded
    baked = np.load(field, allow_pickle=True).item()
    config = dict(config)
    if override and baked["grid"].get("capillary_exit_z_m") is not None:
        config["capillary_exit_z_m"] = float(baked["grid"]["capillary_exit_z_m"])
    sources = {"dc": baked["simion"].get("dc_source_path"), "rf": baked["simion"].get("rf_source_path"),
               "gas": baked["fluent"].get("fluent_source_path")}
    background_pressure_pa = baked["fluent"].get("background_pressure_pa")
    reference = float(baked["simion"].get("rf_reference_peak_voltage_v", 1.))
    z, r, selection, metal, vr, vz, mask = fields.load_local_fields(args, baked, config)
    if mask is None:
        raise ValueError("The production alignment figure requires the recorded electrode mask.")
    dc_er, dc_ez = fields.electric_components(baked["simion"], selection, "dc", config["rf_peak_voltage_v"], reference, args.rf_phase_deg)
    rf_er, rf_ez = fields.electric_components(baked["simion"], selection, "rf-peak", config["rf_peak_voltage_v"], reference, args.rf_phase_deg)
    total_er, total_ez = fields.electric_components(baked["simion"], selection, "total", config["rf_peak_voltage_v"], reference, args.rf_phase_deg)
    del baked
    gas_speed, dc, rf = np.hypot(vr, vz), np.hypot(dc_er, dc_ez), np.hypot(rf_er, rf_ez)
    output = Path(args.output).resolve()
    stats, count = render_figure(args, config, z, r, metal, gas_speed, dc, rf, total_er, total_ez, output)
    arrays_path = output.with_suffix(".fields.npz")
    np.savez_compressed(arrays_path, z_mm=z, r_mm=r, metal_mask=metal, gas_speed_m_per_s=gas_speed,
                        dc_er_v_per_m=dc_er, dc_ez_v_per_m=dc_ez, rf_peak_er_v_per_m=rf_er,
                        rf_peak_ez_v_per_m=rf_ez, total_er_v_per_m=total_er, total_ez_v_per_m=total_ez)
    metadata = {"case": str(case), "field": str(field), "mask": str(mask), "output": str(output),
                "recorded_field": str(recorded), "field_override": override, "field_sources": sources,
                "field_override_scope": "complete DC/RF/gas map" if override else None,
                "case_settings_used_for": "RF amplitude, frequency and initial phase; selected map supplies DC/RF/gas fields and geometry",
                "gas_background_pressure_pa": background_pressure_pa,
                "stored_field_arrays": str(arrays_path), "rf_peak_voltage_v": config["rf_peak_voltage_v"],
                "rf_frequency_hz": config["rf_frequency_hz"], "snapshot_rf_phase_deg": args.rf_phase_deg,
                "runtime_initial_rf_phase_deg": math.degrees(config["rf_phase_rad"]),
                "rf_reference_peak_voltage_v": reference, "space_charge_included": False,
                "capillary_exit_z_mm": config["capillary_exit_z_m"]*1e3,
                "capillary_exit_marker_visible": args.z_min_mm <= config["capillary_exit_z_m"]*1e3 <= args.z_max_mm,
                "display_z_window_mm": [args.z_min_mm, args.z_max_mm],
                "display_r_window_mm": [args.r_min_mm, args.r_max_mm], "panels": stats,
                "plot_grid_shape": [len(r), len(z)], "font_family": args.font_family,
                "base_font_size_pt": args.font_size, "direction_arrow_count": count,
                "combined_panel_mode": getattr(args, "combined_panel", "electric"),
                "electric_overlay": getattr(args, "electric_overlay", "arrows") if getattr(args, "combined_panel", "electric") == "gas-electric" else "arrows",
                "electric_line_density": getattr(args, "electric_line_density", .7),
                "requested_arrow_spacing_mm": args.arrow_spacing_mm,
                "direction_arrow_minimum_v_per_m": args.electric_min_v_per_m,
                "arrow_definition": ("arrowheads on instantaneous DC+RF electric field lines over gas-speed colors; not gas streamlines"
                                     if getattr(args, "combined_panel", "electric") == "gas-electric" and getattr(args, "electric_overlay", "arrows") == "lines"
                                     else "fixed-length directions of the instantaneous external field; metal and subthreshold points excluded"),
                "total_field_definition": "E_DC + E_RF_peak*cos(snapshot RF phase)",
                "display_policy": "bounded color normalization only; no smoothing or field-value modification",
                "colorbar_extension_shape": "rectangle"}
    output.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    for key, values in stats.items():
        print(key, "maximum:", values["maximum"], values["units"],
              "display:", values["display_minimum"], values["display_maximum"], values["display_scale"])
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, help="Production run supplying the recorded field, mask and RF settings.")
    parser.add_argument("--field", default="", help="Optional explicit input; must match the recorded field unless --allow-field-override is supplied.")
    parser.add_argument("--allow-field-override", action="store_true", help="Explicitly select another complete DC/RF/gas map for an offline comparison; provenance records the override.")
    parser.add_argument("--electrode-mask", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--z-min-mm", type=float, default=3.5)
    parser.add_argument("--z-max-mm", type=float, default=13.5)
    parser.add_argument("--r-min-mm", type=float, default=0.)
    parser.add_argument("--r-max-mm", type=float, default=10.)
    parser.add_argument("--dc-max-v-per-m", type=float, default=1.)
    parser.add_argument("--dc-min-v-per-m", type=float, default=0., help="DC color minimum: zero uses a linear scale, positive values use a logarithmic scale.")
    parser.add_argument("--dc-cmap", default="Greys", help="DC color map; use magma for the original Figure 2 palette.")
    parser.add_argument("--electric-min-v-per-m", type=float, default=1e3)
    parser.add_argument("--electric-max-v-per-m", type=float, default=1e6)
    parser.add_argument("--gas-max-m-per-s", type=float, default=900.)
    parser.add_argument("--combined-panel", choices=("electric", "gas-electric"), default="electric",
                        help="Panel d: electric magnitude with arrows, or gas-speed colors with a DC+RF electric overlay.")
    parser.add_argument("--electric-overlay", choices=("arrows", "lines"), default="arrows",
                        help="Gas-electric overlay: regular fixed-length direction arrows (default), or electric field lines.")
    parser.add_argument("--electric-line-density", type=float, default=.7)
    parser.add_argument("--rf-phase-deg", type=float, default=0., help="Instantaneous RF phase, not the runtime initial phase.")
    parser.add_argument("--arrow-spacing-mm", type=float, default=.6)
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--font-size", type=float, default=14.)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    try:
        display_norms(args.dc_max_v_per_m, args.electric_min_v_per_m, args.electric_max_v_per_m,
                      args.gas_max_m_per_s, dc_min_v_per_m=args.dc_min_v_per_m)
    except ValueError as exc:
        parser.error(str(exc))
    values = [args.z_min_mm, args.z_max_mm, args.r_min_mm, args.r_max_mm,
              args.rf_phase_deg, args.arrow_spacing_mm, args.font_size, args.electric_line_density]
    if (not np.isfinite(values).all() or args.z_min_mm >= args.z_max_mm or args.r_min_mm < 0
            or args.r_min_mm >= args.r_max_mm or args.arrow_spacing_mm <= 0 or args.font_size <= 0
            or args.electric_line_density <= 0 or args.dpi < 1):
        parser.error("Invalid window, phase, arrow spacing, font size or DPI.")
    return args


if __name__ == "__main__":
    make_plot(parse_args())
