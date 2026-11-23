"""Compare matched collision cases using time-averaged annular number density."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import plot_field_flow_ion_overlay as overlay
from .plot_style import apply_paper_style
from .plot_vr_trajectory_overlay import _snapshot_records


def annular_volumes(r_edges_m: np.ndarray, z_edges_m: np.ndarray) -> np.ndarray:
    r, z = np.asarray(r_edges_m, float), np.asarray(z_edges_m, float)
    if r.ndim != 1 or z.ndim != 1 or min(r.size, z.size) < 2:
        raise ValueError("Density axes need at least two edges each.")
    if r[0] < 0 or not np.all(np.diff(r) > 0) or not np.all(np.diff(z) > 0):
        raise ValueError("Edges must increase strictly; radius must be nonnegative.")
    return np.pi * np.diff(r**2)[:, None] * np.diff(z)[None, :]


def temporal_weights(times_s: np.ndarray) -> np.ndarray:
    """Normalized trapezoidal exposure weights, including unequal time gaps."""
    times = np.asarray(times_s, float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Time averaging needs at least two strictly increasing finite timestamps.")
    gaps = np.diff(times)
    weights = np.r_[gaps[0]/2, (gaps[:-1]+gaps[1:])/2, gaps[-1]/2]
    return weights / (times[-1]-times[0])


def density_histogram(data, columns, r_edges_m, z_edges_m):
    required = ("r_m", "z_m", "status_code", "represented_real_ions")
    if any(key not in columns for key in required):
        raise ValueError("Snapshot metadata must include radius, z, status and represented weight.")
    r, z, status, weight = [data[:, columns.index(key)] for key in required]
    valid = (status == 0) & (weight > 0) & np.isfinite(r) & np.isfinite(z) & np.isfinite(weight)
    return np.histogram2d(r[valid], z[valid], bins=(r_edges_m, z_edges_m), weights=weight[valid])[0]


def select_records(records, start_ms, end_ms, stride):
    if end_ms <= start_ms or start_ms < 0 or stride < 1 or len(records) < 2:
        raise ValueError("Invalid averaging interval, stride, or insufficient snapshots.")
    times = np.array([record.time_s for record in records])
    if not np.all(np.diff(times) > 0):
        raise ValueError("Snapshot timestamps must increase strictly.")
    targets = np.array([start_ms, end_ms]) * 1e-3
    tolerance = max(1e-12, float(np.median(np.diff(times))) * .01)
    if targets[0] < times[0]-tolerance or targets[-1] > times[-1]+tolerance:
        raise ValueError("Requested density window is not covered by the saved snapshots.")
    low, high = [int(np.argmin(abs(times-target))) for target in targets]
    if max(abs(times[[low, high]]-targets)) > tolerance or high <= low:
        raise ValueError("Window endpoints must coincide with saved snapshots (within 1% of the cadence).")
    indices = list(range(low, high+1, stride))
    if indices[-1] != high:
        indices.append(high)
    return [records[i] for i in indices]


def load_density(case, summary, args, r_edges_m, z_edges_m):
    columns = json.loads((case / "metadata.json").read_text(encoding="utf-8-sig"))["columns"]
    records = averaging_records(case, summary, args)
    weights = temporal_weights(np.array([record.time_s for record in records]))
    population = np.zeros((len(r_edges_m)-1, len(z_edges_m)-1))
    snapshots = []
    for record, exposure in zip(records, weights):
        values = np.load(record.path, allow_pickle=False)
        counts = density_histogram(values, columns, r_edges_m, z_edges_m)
        population += exposure * counts
        snapshots.append({"file": str(record.path), "time_ms": record.time_s*1e3,
                          "time_weight": float(exposure), "local_active_represented_ions": float(counts.sum())})
    volume = annular_volumes(r_edges_m, z_edges_m)
    density = population / volume
    assert np.isclose(np.sum(density*volume), population.sum(), rtol=1e-12)
    return density, {"snapshots": snapshots, "mean_local_represented_ions": float(population.sum()),
                     "actual_time_window_ms": [records[0].time_s*1e3, records[-1].time_s*1e3]}


def cutline_bounds(cut_z_mm, width_mm, z_min_mm, z_max_mm):
    centers = np.asarray(cut_z_mm, dtype=float)
    if (centers.ndim != 1 or centers.size == 0 or not np.isfinite(centers).all()
            or len(np.unique(centers)) != len(centers)
            or not np.isfinite(width_mm) or width_mm <= 0):
        raise ValueError("Cut positions must be distinct and finite; slab width must be positive.")
    bounds = np.column_stack((centers-width_mm/2, centers+width_mm/2))
    if np.any(bounds[:, 0] < z_min_mm) or np.any(bounds[:, 1] > z_max_mm):
        raise ValueError("Each complete cut slab must lie inside the gas-map axial window.")
    return bounds * 1e-3


def validate_open_slabs(mask, r_min_m, r_max_m, bounds_m):
    """Full annular volumes are valid only for slabs free of electrode material."""
    r, z = np.asarray(mask.r_coords_m), np.asarray(mask.z_coords_m)
    if r_min_m < r[0] or r_max_m > r[-1] or bounds_m.min() < z[0] or bounds_m.max() > z[-1]:
        raise ValueError("Cut slabs must be covered by the electrode mask.")
    ri = np.flatnonzero((r >= r_min_m) & (r <= r_max_m))
    for low, high in bounds_m:
        zi = np.flatnonzero((z >= low) & (z <= high))
        if not ri.size or not zi.size:
            raise ValueError("Cut slabs need resolved mask nodes.")
        if np.any(mask.metal_mask[np.ix_(ri, zi)]):
            raise ValueError(
                f"Slab {low*1e3:g}-{high*1e3:g} mm intersects electrodes. "
                "Choose an open axial gap or a smaller radial range/slab width."
            )


def load_radial_profiles(case, summary, args, r_edges_m, bounds_m):
    columns = json.loads((case / "metadata.json").read_text(encoding="utf-8-sig"))["columns"]
    records = averaging_records(case, summary, args)
    weights = temporal_weights(np.array([record.time_s for record in records]))
    population = np.zeros((len(bounds_m), len(r_edges_m)-1))
    snapshots = []
    for record, exposure in zip(records, weights):
        values = np.load(record.path, allow_pickle=False)
        counts = np.array([density_histogram(values, columns, r_edges_m, bounds)[:, 0]
                           for bounds in bounds_m])
        population += exposure * counts
        snapshots.append({"file": str(record.path), "time_ms": record.time_s*1e3,
                          "time_weight": float(exposure), "slab_represented_ions": counts.sum(axis=1).tolist()})
    volume = np.array([annular_volumes(r_edges_m, bounds)[:, 0] for bounds in bounds_m])
    density = population / volume
    np.testing.assert_allclose(density*volume, population, rtol=1e-12, atol=1e-10)
    return density, {"snapshots": snapshots, "density_m3": density.tolist(),
                     "mean_represented_ions_per_annulus": population.tolist(), "annular_volume_m3": volume.tolist(),
                     "mean_slab_represented_ions": population.sum(axis=1).tolist(),
                     "actual_time_window_ms": [records[0].time_s*1e3, records[-1].time_s*1e3]}


def averaging_records(case, summary, args):
    exported = _snapshot_records(case, summary["config"]["macro_time_step_s"])
    if np.any(np.diff([record.time_s for record in exported]) < 0):
        raise ValueError("Snapshot timestamps must not decrease.")
    # A final re-export can have a new macro index but the same physical time.
    unique = {record.time_s: record for record in exported}
    return select_records(list(unique.values()), args.start_ms, args.end_ms, args.snapshot_stride)


def contour_levels(values):
    levels = np.asarray(values, dtype=float)
    if levels.ndim != 1 or levels.size < 1 or not np.isfinite(levels).all() or np.any(levels <= 0) or np.any(np.diff(levels) <= 0):
        raise ValueError("Ion contour levels must be finite, positive and strictly increasing.")
    return levels


def conservative_density_smoothing(density, r_edges_m, z_edges_m, metal, sigma_m):
    """Finite-volume display diffusion with reflecting walls and conserved weight."""
    values = np.asarray(density, dtype=float)
    metal = np.asarray(metal, dtype=bool)
    volume = annular_volumes(r_edges_m, z_edges_m)
    if values.shape != volume.shape or metal.shape != volume.shape:
        raise ValueError("Density and mask must have the annular-grid shape.")
    if not np.isfinite(values).all() or np.any(values < 0) or not np.isfinite(sigma_m) or sigma_m < 0:
        raise ValueError("Density and smoothing length must be finite and nonnegative.")
    result = values.copy()
    if sigma_m == 0:
        return result
    r, z = np.asarray(r_edges_m), np.asarray(z_edges_m)
    rc, zc = (r[:-1]+r[1:])/2, (z[:-1]+z[1:])/2
    radial = 2*np.pi*r[1:-1, None]*np.diff(z)[None, :]/np.diff(rc)[:, None]
    axial = np.pi*np.diff(r**2)[:, None]/np.diff(zc)[None, :]
    radial *= ~(metal[:-1] | metal[1:])
    axial *= ~(metal[:, :-1] | metal[:, 1:])
    outgoing = np.zeros_like(values)
    outgoing[:-1] += radial
    outgoing[1:] += radial
    outgoing[:, :-1] += axial
    outgoing[:, 1:] += axial
    coupled = outgoing > 0
    if not coupled.any():
        return result
    diffusion_time = sigma_m**2/2
    max_step = .45*float(np.min(volume[coupled]/outgoing[coupled]))
    steps = max(1, math.ceil(diffusion_time/max_step))
    dt = diffusion_time/steps
    for _ in range(steps):
        flux_r = dt*radial*(result[1:]-result[:-1])
        flux_z = dt*axial*(result[:, 1:]-result[:, :-1])
        change = np.zeros_like(result)
        change[:-1] += flux_r
        change[1:] -= flux_r
        change[:, :-1] += flux_z
        change[:, 1:] -= flux_z
        result += change/volume
    np.testing.assert_allclose(np.sum(result*volume), np.sum(values*volume), rtol=1e-12, atol=1e-8)
    if np.any(result < 0):
        raise ValueError("Conservative display smoothing produced negative density.")
    return result


def scientific_contour_label(value):
    exponent = int(math.floor(math.log10(value)))
    mantissa = value/10.**exponent
    return (rf"$10^{{{exponent}}}$" if np.isclose(mantissa, 1)
            else rf"${mantissa:g}\times10^{{{exponent}}}$")


def suppress_repeated_streamline_loops(streams, arrow_patches):
    """Omit compact, repeatedly traced loops from the display, not the gas map."""
    segments = streams.lines.get_segments()
    if len(segments) != len(arrow_patches):
        raise ValueError("Streamline paths and arrow patches must have one-to-one correspondence.")
    kept_segments, kept_arrows, removed = [], [], []
    for segment, arrow in zip(segments, arrow_patches):
        lower, upper = segment.min(axis=0), segment.max(axis=0)
        span = upper-lower
        length = np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()
        perimeter = 2*span.sum()
        if 0 < perimeter and span.max() < .75 and length > 3*perimeter:
            arrow.remove()
            removed.append({"z_bounds_mm": [float(lower[0]), float(upper[0])],
                            "r_bounds_mm": [float(lower[1]), float(upper[1])],
                            "path_length_mm": float(length)})
        else:
            kept_segments.append(segment)
            kept_arrows.append(arrow)
    streams.lines.set_segments(kept_segments)
    streams.arrows.set_paths(kept_arrows)
    return removed


@plt.rc_context()
def make_overlay_plot(args, groups, first, field, z, r, metal, gas_vr, gas_vz,
                      mask_path, grids, details, r_edges, z_edges):
    layout = getattr(args, "layout", "density-overlay")
    filled_density = layout == "density-streamlines"
    density_rendering = getattr(args, "density_rendering", "contours")
    font_family = getattr(args, "font_family", "Arial")
    font_size = getattr(args, "font_size", 14.)
    if filled_density:
        plt.rcParams.update({"font.family": font_family, "font.size": font_size,
                             "axes.labelsize": font_size+2, "axes.titlesize": font_size+2,
                             "xtick.labelsize": font_size, "ytick.labelsize": font_size,
                             "legend.fontsize": font_size, "mathtext.fontset": "custom",
                             "mathtext.rm": font_family, "mathtext.it": font_family+":italic",
                             "mathtext.bf": font_family+":bold", "mathtext.sf": font_family,
                             "mathtext.default": "regular"})
    levels = contour_levels(args.ion_contour_levels or [3e11, 1e12, 3e12])
    if not filled_density and not any(np.max(grid) >= levels[0] for grid in grids.values()):
        raise ValueError("All density contours are above the sampled densities; lower --ion-contour-levels.")
    density_limits = None
    if filled_density:
        positive = np.concatenate([grid[grid > 0] for grid in grids.values()])
        if not positive.size:
            raise ValueError("No positive ion densities to display.")
        vmin = getattr(args, "density_min", 0.) or 10.**math.floor(math.log10(positive.min()))
        vmax = getattr(args, "density_max", 0.) or 10.**math.ceil(math.log10(positive.max()))
        if not np.isfinite([vmin, vmax]).all() or not 0 < vmin < vmax:
            raise ValueError("Density log-scale limits must satisfy 0 < minimum < maximum.")
        density_limits = [vmin, vmax]
        density_norm = LogNorm(vmin=vmin, vmax=vmax)
        fill_levels = np.geomspace(vmin, vmax, 33)
    rf_levels = sorted({key[0] for key in groups})
    zc, rc = (z_edges[:-1]+z_edges[1:])*500, (r_edges[:-1]+r_edges[1:])*500
    zz, rr = np.meshgrid(zc, rc)
    lookup = RegularGridInterpolator((r, z), metal.astype(float), method="nearest", bounds_error=False, fill_value=0)
    bin_metal = lookup(np.stack((rr, zz), axis=-1)) > .5
    density_support_mask = bin_metal | (zz < first["config"]["capillary_exit_z_m"]*1e3)
    sigma_mm = getattr(args, "density_smoothing_mm", 0.)
    display_grids = {key: conservative_density_smoothing(grid, r_edges, z_edges, density_support_mask, sigma_mm*1e-3)
                     for key, grid in grids.items()}
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), sharex=True, sharey=True, dpi=args.dpi)
    if filled_density:
        fig.subplots_adjust(left=.085, right=.85, top=.92, bottom=.20, hspace=.26, wspace=.10)
    else:
        fig.subplots_adjust(left=.07, right=.87, top=.94, bottom=.15, hspace=.20, wspace=.07)
    gas_levels = np.unique(np.r_[np.linspace(args.gas_vmin, 0, 9), np.linspace(0, args.gas_vmax, 25)])
    gas_norm = TwoSlopeNorm(vmin=args.gas_vmin, vcenter=0, vmax=args.gas_vmax)
    gas_plot = density_plot = None
    density_color = "#006c36"
    removed_loops = []
    for row, rf in enumerate(rf_levels):
        for col, model in enumerate(("explicit", "hybrid-langevin")):
            ax = axes[row, col]
            density = np.ma.masked_where(density_support_mask, display_grids[(rf, model)])
            if filled_density:
                visible_density = np.ma.masked_less_equal(density, 0)
                if density_rendering == "cells":
                    density_plot = ax.pcolormesh(z_edges*1e3, r_edges*1e3, visible_density,
                                                norm=density_norm, cmap="magma", shading="flat", rasterized=True)
                else:
                    density_plot = ax.contourf(zc, rc, visible_density, levels=fill_levels,
                                                norm=density_norm, cmap="magma", corner_mask=False, extend="both")
            else:
                gas_plot = ax.contourf(z, r, np.ma.masked_where(metal, gas_vr), levels=gas_levels,
                                       norm=gas_norm, cmap="RdBu_r", alpha=.55, extend="both")
            previous_patches = len(ax.patches)
            streams = ax.streamplot(z, r, np.ma.masked_where(metal, gas_vz), np.ma.masked_where(metal, gas_vr),
                                    color="#27313a" if filled_density else "#777d80",
                                    density=.62, linewidth=.55 if filled_density else .42, arrowsize=.62, zorder=3)
            if filled_density:
                arrow_patches = list(ax.patches)[previous_patches:]
                removed = suppress_repeated_streamline_loops(streams, arrow_patches)
                removed_loops.extend({"panel": chr(97+row*2+col), **loop} for loop in removed)
                effects = [path_effects.Stroke(linewidth=1.15, foreground="white", alpha=.8), path_effects.Normal()]
                streams.lines.set_path_effects(effects)
                for arrow in list(ax.patches)[previous_patches:]:
                    arrow.set_path_effects(effects)
            else:
                contours = ax.contour(zc, rc, density, levels=levels, colors=[density_color],
                                      linewidths=np.linspace(.85, 1.35, len(levels)), corner_mask=False, zorder=4)
                contours.set_path_effects([path_effects.Stroke(linewidth=2.15, foreground="white", alpha=.65), path_effects.Normal()])
                texts = ax.clabel(contours, inline=True, fontsize=7.2, fmt=scientific_contour_label, inline_spacing=3)
                for text in texts:
                    text.set_bbox({"facecolor":"white", "edgecolor":"none", "alpha":.78, "pad":.4})
            ax.contourf(z, r, metal.astype(float), levels=[.5, 1.5], colors=["#d1d3d4"], zorder=6)
            if metal.any():
                ax.contour(z, r, metal.astype(float), levels=[.5], colors=["#6a6f72"], linewidths=.5, zorder=7)
            ax.axvline(first["config"]["capillary_exit_z_m"]*1e3, color="#7d8489", ls=":", lw=.7)
            model_label = "" if filled_density else ("Explicit, " if model == "explicit" else "Hybrid, ")
            ax.set_title(f"({chr(97+row*2+col)}) {model_label}RF {rf:g} V"+r"$_{peak}$", loc="left",
                         fontsize=font_size+2 if filled_density else 10.5, pad=8 if filled_density else 6)
            ax.set(xlim=(args.z_min_mm, args.z_max_mm), ylim=(args.r_min_mm, args.r_max_mm))
            if row == 1: ax.set_xlabel("z / mm")
            if col == 0: ax.set_ylabel("r / mm")
            ax.tick_params(direction="in", top=True, right=True, labelsize=font_size if filled_density else 9)
            if model == "hybrid-langevin" and not filled_density:
                cfg = groups[(rf, model)][1]["config"]
                lo, hi = (cfg[k]*1e3 for k in ("langevin_z_start_m", "langevin_z_end_m"))
                height = args.r_max_mm*.94
                ax.annotate("", (lo, height), (hi, height), arrowprops={"arrowstyle":"|-|", "color":"#4f5960", "lw":.9}, zorder=8)
                ax.text((lo+hi)/2, height-.2, "Hybrid region", ha="center", va="top", fontsize=7.5,
                        bbox={"facecolor":"white", "edgecolor":"none", "alpha":.85, "pad":1}, zorder=8)
    cax = fig.add_axes([.875, .26, .022, .60] if filled_density else [.895, .235, .016, .61])
    if filled_density:
        ticks = 10.**np.arange(math.ceil(math.log10(vmin)), math.floor(math.log10(vmax))+1)
        fig.colorbar(density_plot, cax=cax, ticks=ticks, extend="both", extendrect=True,
                     label=r"Time-averaged ion number density / m$^{-3}$")
    else:
        fig.colorbar(gas_plot, cax=cax, ticks=[args.gas_vmin, 0, args.gas_vmax/3, 2*args.gas_vmax/3, args.gas_vmax],
                     label=r"Gas radial velocity $u_r$ / m s$^{-1}$")
    handles = [Line2D([], [], color="#27313a" if filled_density else "#777d80", lw=.7,
                      marker=">", markersize=4, label="Gas streamlines")]
    if not filled_density:
        handles.insert(0, Line2D([], [], color=density_color, lw=1.4, label=r"Ion-density contours / m$^{-3}$"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.47, .095 if filled_density else .060),
               ncol=2, frameon=False, handlelength=3.)
    note = f"{args.start_ms:g}-{args.end_ms:g} ms average; bins {(z_edges[1]-z_edges[0])*1e3:g} mm (z) x {(r_edges[1]-r_edges[0])*1e3:g} mm (r)."
    if args.context_note: note += ("\n" if filled_density else "  ") + args.context_note
    if sigma_mm: note += f"  Display smoothing: {sigma_mm:g} mm."
    elif filled_density: note += "  No smoothing."
    fig.text(.47, .022, note, ha="center", fontsize=max(10., font_size-3) if filled_density else 8, color=".3")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    npz = output.with_suffix(".density.npz")
    arrays = {f"density_rf{rf:g}_{'explicit' if model == 'explicit' else 'hybrid'}_m3": grid
              for (rf, model), grid in grids.items()}
    arrays.update({f"display_density_rf{rf:g}_{'explicit' if model == 'explicit' else 'hybrid'}_m3": grid
                   for (rf, model), grid in display_grids.items()})
    np.savez_compressed(npz, r_edges_m=r_edges, z_edges_m=z_edges,
                        display_metal_mask=bin_metal, density_support_mask=density_support_mask, **arrays)
    metadata = {"output": str(output.resolve()), "layout": layout, "field": str(field), "mask": str(mask_path),
                "requested_window_ms": [args.start_ms, args.end_ms], "case_details": details,
                "r_edges_mm": (r_edges*1e3).tolist(), "z_edges_mm": (z_edges*1e3).tolist(),
                "ion_contour_levels_m3": None if filled_density else levels.tolist(), "stored_density_grids": str(npz.resolve()),
                "density_color_limits_m3": density_limits,
                "density_colormap": "magma" if filled_density else None,
                "density_rendering": density_rendering if filled_density else "contour lines",
                "density_formula": "sum_k(alpha_k * sum_active_p_in_cell(W_p)) / (pi * (r_outer^2-r_inner^2) * dz); sum_k(alpha_k)=1",
                "model_labels_visible": not filled_density,
                "font_family": font_family if filled_density else None,
                "tick_font_size_pt": font_size if filled_density else 9,
                "title_and_axis_label_font_size_pt": font_size+2 if filled_density else None,
                "colorbar_extension_shape": "rectangle" if filled_density else "triangle",
                "suppressed_streamline_loops": removed_loops,
                "streamline_display_filter": "omit paths narrower than 0.75 mm on both axes and longer than three bounding-box perimeters" if filled_density else None,
                "panel_mapping": [{"panel": chr(97+row*2+col), "rf_Vpeak": rf, "collision_model": model}
                                  for row, rf in enumerate(rf_levels)
                                  for col, model in enumerate(("explicit", "hybrid-langevin"))],
                "density_definition": "active total represented ion weight per full annular grid-cell volume; azimuthally averaged",
                "smoothing": "none" if sigma_mm == 0 else "conservative axisymmetric finite-volume diffusion; no flux across mask faces",
                "display_smoothing_sigma_mm": sigma_mm, "display_diffusion_time_m2": (sigma_mm*1e-3)**2/2,
                "density_upstream_display_limit_mm": first["config"]["capillary_exit_z_m"]*1e3,
                "individual_peak_normalization": "none",
                "time_averaging": "normalized trapezoidal weights over saved snapshots; no extra RF-phase averaging",
                "duplicate_timestamp_policy": "keep last export at each physical timestamp",
                "gas_arrows": "gas streamlines, not ion velocities", "context_note": args.context_note,
                "density_line_color": None if filled_density else density_color,
                "gas_color_limits_m_per_s": None if filled_density else [args.gas_vmin, args.gas_vmax]}
    output.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    for detail in details:
        print(detail["rf_Vpeak"], detail["collision_model"], len(detail["snapshots"]), detail["mean_local_represented_ions"])
    return output


def make_radial_plot(args, groups, first, field, z, r, metal, gas_vr, gas_vz, mask_path):
    if mask_path is None:
        raise ValueError("Radial cutlines require an electrode mask to verify open annular volumes.")
    bounds = cutline_bounds(args.cut_z_mm, args.cut_width_mm, args.z_min_mm, args.z_max_mm)
    mask = overlay.alignment.load_electrode_mask(
        source_path=mask_path, field_metadata=first["field_metadata"],
        z_offset_m=first["config"].get("electrode_mask_z_offset_m"))
    validate_open_slabs(mask, args.r_min_mm*1e-3, args.r_max_mm*1e-3, bounds)
    del mask
    nr = max(1, math.ceil((args.r_max_mm-args.r_min_mm)/args.dr_mm))
    r_edges = np.linspace(args.r_min_mm, args.r_max_mm, nr+1)*1e-3
    profiles, details = {}, []
    for key, (case, summary) in sorted(groups.items()):
        profile, detail = load_radial_profiles(case, summary, args, r_edges, bounds)
        profiles[key] = profile
        details.append({"case": str(case), "rf_Vpeak": key[0], "collision_model": key[1], **detail})
    positive = np.concatenate([p[p > 0] for p in profiles.values()])
    if not positive.size:
        raise ValueError("No active ion weight found in the selected radial slabs.")
    vmin = args.density_min or 10**math.floor(math.log10(positive.min()))
    vmax = args.density_max or 10**math.ceil(math.log10(positive.max()))
    if not np.isfinite([vmin, vmax]).all() or not 0 < vmin < vmax:
        raise ValueError("Radial log-scale limits must satisfy 0 < minimum < maximum.")
    ncuts = len(bounds)
    fig = plt.figure(figsize=(10.8, 7.8), dpi=args.dpi)
    layout = fig.add_gridspec(2, ncuts, height_ratios=(1.05, 1.), hspace=.43, wspace=.11,
                             left=.09, right=.90, bottom=.13, top=.96)
    gas_ax = fig.add_subplot(layout[0, :])
    gas_cax = fig.add_axes([.92, .61, .016, .32])
    levels = np.unique(np.r_[np.linspace(args.gas_vmin, 0, 9), np.linspace(0, args.gas_vmax, 25)])
    gas_plot = gas_ax.contourf(z, r, np.ma.masked_where(metal, gas_vr), levels=levels,
                              norm=TwoSlopeNorm(vmin=args.gas_vmin, vcenter=0, vmax=args.gas_vmax),
                              cmap="RdBu_r", extend="both")
    gas_ax.streamplot(z, r, np.ma.masked_where(metal, gas_vz), np.ma.masked_where(metal, gas_vr),
                      color="#4d5357", density=.85, linewidth=.5, arrowsize=.65)
    gas_ax.contourf(z, r, metal.astype(float), levels=[.5, 1.5], colors=["#c9cdd0"])
    if metal.any():
        gas_ax.contour(z, r, metal.astype(float), levels=[.5], colors=["#60666a"], linewidths=.5)
    for i, center in enumerate(args.cut_z_mm):
        gas_ax.axvspan(bounds[i, 0]*1e3, bounds[i, 1]*1e3, facecolor="none", edgecolor="#31383d", lw=.7, ls=":")
        gas_ax.axvline(center, color="#31383d", lw=.9, ls="--")
        gas_ax.text(center, args.r_max_mm*.94, f"({chr(98+i)})", ha="center", va="top", fontsize=10,
                    bbox={"facecolor":"white", "edgecolor":"none", "alpha":.9, "pad":1.5})
    gas_ax.axvline(first["config"]["capillary_exit_z_m"]*1e3, color="#68777e", ls=":", lw=.8)
    gas_ax.set(xlim=(args.z_min_mm, args.z_max_mm), ylim=(args.r_min_mm, args.r_max_mm),
               xlabel="z / mm", ylabel="r / mm")
    gas_ax.text(.012, .96, "(a)", transform=gas_ax.transAxes, va="top", fontweight="bold")
    fig.colorbar(gas_plot, cax=gas_cax, ticks=[args.gas_vmin, 0, args.gas_vmax/3, 2*args.gas_vmax/3, args.gas_vmax],
                 label=r"Gas $u_r$ / m s$^{-1}$")
    rf_levels = sorted({key[0] for key in groups})
    colors = dict(zip(rf_levels, ("#286a9a", "#c15b35")))
    centers = (r_edges[:-1]+r_edges[1:])*500
    axes = []
    for i, center in enumerate(args.cut_z_mm):
        ax = fig.add_subplot(layout[1, i], sharey=axes[0] if axes else None)
        axes.append(ax)
        for (rf, model), profile in sorted(profiles.items()):
            explicit = model == "explicit"
            ax.plot(centers, np.where(profile[i] > 0, profile[i], np.nan), color=colors[rf],
                    lw=1.65 if explicit else 1.4, ls="-" if explicit else (0, (4, 2)),
                    alpha=1 if explicit else .85)
        ax.set(xlim=(args.r_min_mm, args.r_max_mm), ylim=(vmin, vmax), yscale="log", xlabel="r / mm")
        ax.set_xticks(np.arange(math.ceil(args.r_min_mm/2)*2, args.r_max_mm+.01, 2))
        ax.tick_params(labelleft=i == 0, direction="in", top=True, right=True)
        ax.grid(axis="y", which="major", color=".9", lw=.5)
        ax.set_title(f"({chr(98+i)})  z = {center:g} mm", fontsize=10, pad=7)
    axes[0].set_ylabel(r"Time-averaged ion density / m$^{-3}$")
    handles = [Line2D([], [], color=colors[rf], lw=1.65 if model == "explicit" else 1.4,
                      ls="-" if model == "explicit" else (0, (4, 2)),
                      label=f"RF {rf:g} V"+r"$_{peak}$, "+("explicit" if model == "explicit" else "hybrid"))
               for rf in rf_levels for model in ("explicit", "hybrid-langevin")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .049), ncol=4, frameon=False,
               columnspacing=1.4, handlelength=2.8)
    note = (f"{args.start_ms:g}-{args.end_ms:g} ms average; "
            f"slab width {args.cut_width_mm:g} mm; radial bins {(r_edges[1]-r_edges[0])*1e3:g} mm.")
    if args.context_note:
        note += "  " + args.context_note
    fig.text(.5, .012, note, ha="center", fontsize=8.3, color=".3")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    metadata = {"output": str(output.resolve()), "layout": "radial-cutlines", "field": str(field), "mask": str(mask_path),
                "case_details": details, "requested_window_ms": [args.start_ms, args.end_ms],
                "cut_z_mm": args.cut_z_mm, "slab_bounds_mm": (bounds*1e3).tolist(), "r_edges_mm": (r_edges*1e3).tolist(),
                "density_limits_m3": [vmin, vmax], "annular_volume_validation": "all cut slabs are free of metal mask nodes",
                "density_definition": "active total represented weight per full annular slab volume, azimuthally averaged",
                "time_averaging": "normalized trapezoidal weights over saved snapshots; no extra RF-phase averaging",
                "duplicate_timestamp_policy": "keep the last export at each physical timestamp",
                "smoothing": "none", "curve_peak_normalization": "none", "zero_density_on_log_axis": "gaps, not a positive floor",
                "gas_arrows": "gas streamlines, not ion velocity vectors", "context_note": args.context_note,
                "gas_color_limits_m_per_s": [args.gas_vmin, args.gas_vmax]}
    output.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    for detail in details:
        print(detail["rf_Vpeak"], detail["collision_model"], len(detail["snapshots"]), detail["mean_slab_represented_ions"])
    return output


def make_plot(args):
    apply_paper_style(font_scale=.90, line_scale=1.)
    cases = [Path(value).resolve() for value in args.case_dir]
    if len(cases) != 4 or len(set(cases)) != 4:
        raise ValueError("Provide four distinct cases: RF off/on crossed with explicit/hybrid.")
    summaries = [json.loads((case / "simulation_summary.json").read_text(encoding="utf-8-sig")) for case in cases]
    groups = {}
    first = summaries[0]
    shared_config = ("ion_current_a", "rf_frequency_hz", "rf_phase_rad", "capillary_voltage_v",
                     "capillary_exit_z_m", "detector_z_m", "source_profile", "source_radius_m",
                     "source_gaussian_sigma_m", "pic_grid_nr", "pic_grid_nz")
    for case, report in zip(cases, summaries):
        cfg = report["config"]
        model = cfg["collision_model"]
        if model not in ("explicit", "hybrid-langevin"):
            raise ValueError(f"Unsupported comparison collision model: {model}")
        key = (cfg["rf_peak_voltage_v"], model)
        if key in groups:
            raise ValueError("Duplicate RF/model combination.")
        for name in shared_config:
            if cfg.get(name) != first["config"].get(name):
                raise ValueError(f"Cases differ in non-scanned parameter {name}.")
        if report["template"] != first["template"] or report["field_metadata"] != first["field_metadata"]:
            raise ValueError("Cases must use the same ion template and imported-field metadata.")
        groups[key] = (case, report)
    rf_levels = sorted({key[0] for key in groups})
    if len(rf_levels) != 2 or rf_levels[0] != 0 or any((rf, model) not in groups for rf in rf_levels for model in ("explicit", "hybrid-langevin")):
        raise ValueError("Expected RF off plus one nonzero amplitude, each with explicit and hybrid cases.")
    field = overlay.resolve_input(args.field or first["config"]["static_field_path"])
    recorded_field = overlay.resolve_input(first["config"]["static_field_path"])
    if field != recorded_field:
        raise ValueError("Field override differs from the recorded field; use a matched input map.")
    baked = np.load(field, allow_pickle=True).item()
    z, r, _, metal, gas_vr, gas_vz, mask_path = overlay.load_local_fields(args, baked, first["config"])
    del baked
    if args.layout == "radial-cutlines":
        return make_radial_plot(args, groups, first, field, z, r, metal, gas_vr, gas_vz, mask_path)
    nz = max(1, math.ceil((args.z_max_mm-args.z_min_mm)/args.dz_mm))
    nr = max(1, math.ceil((args.r_max_mm-args.r_min_mm)/args.dr_mm))
    z_edges = np.linspace(args.z_min_mm, args.z_max_mm, nz+1)*1e-3
    r_edges = np.linspace(args.r_min_mm, args.r_max_mm, nr+1)*1e-3
    grids, details = {}, []
    for key in sorted(groups):
        case, report = groups[key]
        density, detail = load_density(case, report, args, r_edges, z_edges)
        grids[key] = density
        details.append({"case": str(case), "rf_Vpeak": key[0], "collision_model": key[1], **detail})
    positives = np.concatenate([v[v>0] for v in grids.values()])
    if not positives.size:
        raise ValueError("No active ion weight found in this spatial/time window.")
    vmin = args.density_min or 10**math.floor(math.log10(positives.min()))
    vmax = args.density_max or 10**math.ceil(math.log10(positives.max()))
    if vmin <= 0 or vmax <= vmin:
        raise ValueError("Density log-scale limits must satisfy 0 < minimum < maximum.")
    if args.layout in ("density-overlay", "density-streamlines"):
        return make_overlay_plot(args, groups, first, field, z, r, metal, gas_vr, gas_vz,
                                 mask_path, grids, details, r_edges, z_edges)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(10.6, 9.1), dpi=args.dpi, layout="constrained")
    layout = fig.add_gridspec(3, 3, width_ratios=(1, 1, .035), height_ratios=(.85, 1, 1), hspace=.08, wspace=.04)
    gas_ax = fig.add_subplot(layout[0, :2])
    gas_cax = fig.add_subplot(layout[0, 2])
    gas_norm = TwoSlopeNorm(vmin=args.gas_vmin, vcenter=0, vmax=args.gas_vmax)
    gas_levels = np.unique(np.r_[np.linspace(args.gas_vmin, 0, 9), np.linspace(0, args.gas_vmax, 25)])
    gas_plot = gas_ax.contourf(z, r, np.ma.masked_where(metal, gas_vr),
                             levels=gas_levels, norm=gas_norm,
                             cmap="RdBu_r", extend="both")
    gas_ax.streamplot(z, r, np.ma.masked_where(metal, gas_vz), np.ma.masked_where(metal, gas_vr),
                      color="#3e474d", density=.75, linewidth=.55, arrowsize=.7)
    gas_ticks = np.r_[args.gas_vmin, args.gas_vmin/2, 0, args.gas_vmax*np.array([.25,.5,.75,1])]
    fig.colorbar(gas_plot, cax=gas_cax, ticks=gas_ticks, label=r"Gas $u_r$ / m s$^{-1}$")
    density_cax = fig.add_subplot(layout[1:, 2])
    axes = [gas_ax]
    labels = ["(a)  Gas flow"]
    density_plot = None
    zz, rr = np.meshgrid((z_edges[:-1]+z_edges[1:])*.5e3, (r_edges[:-1]+r_edges[1:])*.5e3)
    metal_lookup = RegularGridInterpolator((r, z), metal.astype(float), method="nearest", bounds_error=False, fill_value=0)
    bin_metal = metal_lookup(np.stack((rr, zz), axis=-1)) > .5
    for row, rf in enumerate(rf_levels, 1):
        for col, model in enumerate(("explicit", "hybrid-langevin")):
            ax = fig.add_subplot(layout[row, col])
            density = grids[(rf, model)]
            density_plot = ax.pcolormesh(z_edges*1e3, r_edges*1e3,
                                         np.ma.masked_where((density<=0) | bin_metal, density),
                                         cmap="magma", norm=norm, shading="flat", rasterized=True)
            label = "Explicit" if model == "explicit" else "Hybrid"
            letter = chr(ord("b")+(row-1)*2+col)
            labels.append(f"({letter})  {label}, RF {rf:g} V" + r"$_{peak}$")
            axes.append(ax)
            if model == "hybrid-langevin":
                cfg = groups[(rf, model)][1]["config"]
                start, end = [cfg[key]*1e3 for key in ("langevin_z_start_m", "langevin_z_end_m")]
                y = args.r_max_mm*.86
                ax.annotate("", (start,y), (end,y), arrowprops={"arrowstyle":"|-|", "color":"#247b83", "lw":1.2})
                ax.text((start+end)/2, y-.2, "Hybrid region", fontsize=8, ha="center", va="top", color="#247b83")
    for ax, label in zip(axes, labels):
        ax.contourf(z, r, metal.astype(float), levels=[.5, 1.5], colors=["#c9cdd0"])
        if metal.any():
            ax.contour(z, r, metal.astype(float), levels=[.5], colors=["#5b6267"], linewidths=.55)
        ax.axvline(first["config"]["capillary_exit_z_m"]*1e3, color="#69767d", ls="--", lw=.8)
        ax.set(xlim=(args.z_min_mm,args.z_max_mm), ylim=(args.r_min_mm,args.r_max_mm), xlabel="z / mm", ylabel="r / mm")
        ax.text(.02, .97, label, transform=ax.transAxes, fontsize=10, va="top",
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":.85, "pad":2})
        ax.tick_params(labelsize=9)
    fig.colorbar(density_plot, cax=density_cax, label=r"Time-averaged ion number density / m$^{-3}$")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    metadata = {"output": str(output.resolve()), "field": str(field), "mask": str(mask_path),
                "requested_window_ms": [args.start_ms, args.end_ms], "case_details": details,
                "z_edges_mm": (z_edges*1e3).tolist(), "r_edges_mm": (r_edges*1e3).tolist(),
                "density_color_limits_m3": [vmin, vmax], "time_averaging": "normalized trapezoidal exposure weights",
                "density_definition": "active total represented ion weight per full annular cell volume; azimuthally averaged",
                "smoothing": "none", "gas_arrows": "gas streamlines, not ion velocities",
                "source_current_nA": first["config"]["ion_current_a"]*1e9,
                "birth_energy_voltage_V": first["config"]["capillary_voltage_v"],
                "gas_color_limits_m_per_s": [args.gas_vmin,args.gas_vmax]}
    output.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    for detail in details:
        print(detail["rf_Vpeak"], detail["collision_model"], len(detail["snapshots"]), detail["mean_local_represented_ions"])
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", action="append", required=True, help="Repeat for each of the four matched RF/model cases.")
    parser.add_argument("--field", default="", help="Optional explicit path, must match the recorded field.")
    parser.add_argument("--electrode-mask", default=None, help="Mask/cache path; defaults to recorded case input.")
    parser.add_argument("--start-ms", type=float, default=1.)
    parser.add_argument("--end-ms", type=float, default=3.)
    parser.add_argument("--snapshot-stride", type=int, default=5)
    parser.add_argument("--z-min-mm", type=float, default=4.5)
    parser.add_argument("--z-max-mm", type=float, default=13.)
    parser.add_argument("--r-min-mm", type=float, default=0.)
    parser.add_argument("--r-max-mm", type=float, default=8.)
    parser.add_argument("--dz-mm", type=float, default=.1)
    parser.add_argument("--dr-mm", type=float, default=.1)
    parser.add_argument("--layout", choices=("density-map", "radial-cutlines", "density-overlay", "density-streamlines"), default="density-map")
    parser.add_argument("--ion-contour-levels", type=float, nargs="+", default=None, help="Common absolute density contours in m^-3 for density-overlay.")
    parser.add_argument("--density-smoothing-mm", type=float, default=0., help="Optional conservative display smoothing for density-overlay or density-streamlines; zero keeps original bins.")
    parser.add_argument("--density-rendering", choices=("contours", "cells"), default="contours",
                        help="Density-streamlines rendering: filled contours, or flat annular cell values without spatial interpolation.")
    parser.add_argument("--font-family", default="Arial", help="Font used for the density-streamlines layout.")
    parser.add_argument("--font-size", type=float, default=14., help="Base tick/legend size in pt for density-streamlines; axis labels and titles add 2 pt.")
    parser.add_argument("--cut-z-mm", nargs="+", type=float, default=[8., 10., 12.])
    parser.add_argument("--cut-width-mm", type=float, default=.6)
    parser.add_argument("--context-note", default="", help="Optional factual note below the comparison figure.")
    parser.add_argument("--density-min", type=float, default=0.)
    parser.add_argument("--density-max", type=float, default=0.)
    parser.add_argument("--gas-vmin", type=float, default=-120.)
    parser.add_argument("--gas-vmax", type=float, default=720.)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    if args.dz_mm <= 0 or args.dr_mm <= 0 or args.z_max_mm <= args.z_min_mm or args.r_max_mm <= args.r_min_mm:
        parser.error("Bin widths and axis extents must be positive.")
    if not args.gas_vmin < 0 < args.gas_vmax:
        parser.error("Gas radial-velocity scale must straddle zero.")
    if args.r_min_mm < 0:
        parser.error("Radius must be nonnegative.")
    if not np.isfinite(args.density_smoothing_mm) or args.density_smoothing_mm < 0:
        parser.error("Density smoothing scale must be finite and nonnegative.")
    if not np.isfinite(args.font_size) or args.font_size <= 0:
        parser.error("Font size must be finite and positive.")
    if args.density_smoothing_mm and args.layout not in ("density-overlay", "density-streamlines"):
        parser.error("Display smoothing is available only for density-overlay or density-streamlines.")
    if args.layout == "radial-cutlines":
        try:
            cutline_bounds(args.cut_z_mm, args.cut_width_mm, args.z_min_mm, args.z_max_mm)
        except ValueError as exc:
            parser.error(str(exc))
    if args.ion_contour_levels is not None:
        try:
            contour_levels(args.ion_contour_levels)
        except ValueError as exc:
            parser.error(str(exc))
    return args


if __name__ == "__main__":
    make_plot(parse_args())
