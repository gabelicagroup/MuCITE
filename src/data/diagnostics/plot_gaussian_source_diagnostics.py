"""Generate Gaussian source and tube-gas birth velocity diagnostic figures."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from ...env.sources.continuous import sample_source_xy_offsets
    from .plot_style import add_subplot_labels, apply_paper_style, scaled, subplot_label
except ImportError:  # pragma: no cover - supports direct script execution.
    from src.data.diagnostics.plot_style import (  # type: ignore
        add_subplot_labels,
        apply_paper_style,
        scaled,
        subplot_label,
    )
    from src.env.sources.continuous import sample_source_xy_offsets  # type: ignore


DEFAULT_GAS_CSV = Path("F_field") / "slenssource_flow_interior.txt"


@dataclass(frozen=True)
class GasInterpolators:
    z_m: np.ndarray
    r_m: np.ndarray
    vz_m_per_s: np.ndarray
    vr_m_per_s: np.ndarray
    linear_vz: LinearNDInterpolator
    linear_vr: LinearNDInterpolator
    nearest_vz: NearestNDInterpolator
    nearest_vr: NearestNDInterpolator


@dataclass(frozen=True)
class SourceSamples:
    gas: GasInterpolators
    x_m: np.ndarray
    y_m: np.ndarray
    r_m: np.ndarray
    z_birth_m: np.ndarray
    vz_m_per_s: np.ndarray
    vr_raw_m_per_s: np.ndarray
    vr_m_per_s: np.ndarray
    speed_m_per_s: np.ndarray


def _load_gas_interpolators(path: Path, z_min_m: float, z_max_m: float) -> GasInterpolators:
    if not path.exists():
        raise FileNotFoundError(f"Gas CSV not found: {path}")
    if z_max_m < z_min_m:
        raise ValueError("--birth-z-max-mm must be >= --birth-z-min-mm.")

    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 4, 5), dtype=np.float64)
    raw = np.atleast_2d(raw)
    z_m = np.asarray(raw[:, 0], dtype=np.float64)
    r_m = np.asarray(raw[:, 1], dtype=np.float64)
    vz_m_per_s = np.asarray(raw[:, 2], dtype=np.float64)
    vr_m_per_s = np.asarray(raw[:, 3], dtype=np.float64)

    valid = (
        np.isfinite(z_m)
        & np.isfinite(r_m)
        & np.isfinite(vz_m_per_s)
        & np.isfinite(vr_m_per_s)
        & (z_m >= z_min_m)
        & (z_m <= z_max_m)
    )
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError(f"Fewer than 3 finite gas points in z=[{z_min_m * 1e3:g}, {z_max_m * 1e3:g}] mm.")

    z_used = z_m[valid]
    r_used = r_m[valid]
    vz_used = vz_m_per_s[valid]
    vr_used = vr_m_per_s[valid]
    points = np.column_stack((z_used, r_used))
    return GasInterpolators(
        z_m=z_used,
        r_m=r_used,
        vz_m_per_s=vz_used,
        vr_m_per_s=vr_used,
        linear_vz=LinearNDInterpolator(points, vz_used, fill_value=np.nan),
        linear_vr=LinearNDInterpolator(points, vr_used, fill_value=np.nan),
        nearest_vz=NearestNDInterpolator(points, vz_used),
        nearest_vr=NearestNDInterpolator(points, vr_used),
    )


def _sample_component(linear, nearest, z_m: np.ndarray, r_m: np.ndarray) -> np.ndarray:
    values = np.asarray(linear(z_m, r_m), dtype=np.float64)
    missing = ~np.isfinite(values)
    if np.any(missing):
        values[missing] = nearest(np.asarray(z_m)[missing], np.asarray(r_m)[missing])
    return np.asarray(values, dtype=np.float64)


def _subsample_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_points, dtype=np.int64)


def _expected_radial_density(r_m: np.ndarray, sigma_m: float) -> np.ndarray:
    density = np.exp(-0.5 * (r_m / sigma_m) ** 2)
    peak = float(np.nanmax(density)) if density.size else 1.0
    return density / max(peak, 1.0e-30)


def _parse_panels(panel_text: str) -> list[str]:
    presets = {
        "paper": ["xy-density", "birth-zr", "vz-hist", "velocity-phase"],
        "default": ["xy-density", "radial-density", "birth-zr", "vz-hist", "vr-hist", "velocity-phase"],
        "gas": ["birth-zr", "vz-hist", "vr-hist", "velocity-phase"],
        "source": ["xy-density", "radial-density", "xy-scatter"],
        "all": [
            "xy-density",
            "radial-density",
            "birth-zr",
            "vz-hist",
            "vr-hist",
            "velocity-phase",
            "gas-vz-field",
            "birth-z-hist",
            "speed-hist",
        ],
    }
    aliases = {
        "xy": "xy-density",
        "density": "xy-density",
        "xy-density": "xy-density",
        "xy-scatter": "xy-scatter",
        "radial": "radial-density",
        "radial-density": "radial-density",
        "annulus": "radial-density",
        "birth": "birth-zr",
        "birth-zr": "birth-zr",
        "zr": "birth-zr",
        "vz": "vz-hist",
        "vz-hist": "vz-hist",
        "vr": "vr-hist",
        "vr-hist": "vr-hist",
        "phase": "velocity-phase",
        "velocity-phase": "velocity-phase",
        "phase-space": "velocity-phase",
        "gas-vz": "gas-vz-field",
        "gas-vz-field": "gas-vz-field",
        "birth-z": "birth-z-hist",
        "birth-z-hist": "birth-z-hist",
        "speed": "speed-hist",
        "speed-hist": "speed-hist",
    }
    raw = str(panel_text).strip().lower()
    if raw in presets:
        return presets[raw]
    panels: list[str] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key in presets:
            for panel in presets[key]:
                if panel not in panels:
                    panels.append(panel)
            continue
        panel = aliases.get(key)
        if panel is None:
            valid = sorted(set(aliases) | set(presets))
            raise ValueError(f"Unknown panel {key!r}. Valid values include: {', '.join(valid)}")
        if panel not in panels:
            panels.append(panel)
    if not panels:
        raise ValueError("--panels must select at least one panel.")
    return panels


def _layout_shape(layout: str, panel_count: int) -> tuple[int, int]:
    text = str(layout).strip().lower()
    if text == "auto":
        if panel_count <= 1:
            return 1, 1
        if panel_count == 2:
            return 1, 2
        if panel_count <= 4:
            return 2, 2
        if panel_count <= 6:
            return 2, 3
        return int(math.ceil(panel_count / 3.0)), 3
    parts = text.split("x", 1)
    if len(parts) != 2:
        raise ValueError("--layout must be auto or rowsxcols, for example 1x1, 2x3.")
    rows = int(parts[0])
    cols = int(parts[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("--layout rows and columns must be positive.")
    if rows * cols < panel_count:
        raise ValueError(f"--layout {layout!r} has {rows * cols} slots for {panel_count} panels.")
    return rows, cols


def _figure_size(rows: int, cols: int) -> tuple[float, float]:
    if rows == 1 and cols == 1:
        return 5.4, 4.5
    return max(5.0, cols * 4.6), max(4.0, rows * 3.75)


def _build_samples(args: argparse.Namespace) -> SourceSamples:
    rng = np.random.default_rng(int(args.seed))
    count = int(args.n_samples)
    radius_m = float(args.source_radius_mm) * 1.0e-3
    sigma_m = float(args.source_gaussian_sigma_mm) * 1.0e-3
    z_min_m = float(args.birth_z_min_mm) * 1.0e-3
    z_max_m = float(args.birth_z_max_mm) * 1.0e-3
    if count <= 0:
        raise ValueError("--n-samples must be positive.")
    if radius_m <= 0.0 or sigma_m <= 0.0:
        raise ValueError("--source-radius-mm and --source-gaussian-sigma-mm must be positive.")

    x_m, y_m = sample_source_xy_offsets(
        rng,
        count,
        source_radius_m=radius_m,
        source_profile="gaussian",
        source_gaussian_sigma_m=sigma_m,
    )
    r_m = np.hypot(x_m, y_m)
    z_birth_m = rng.uniform(z_min_m, z_max_m, size=count) if z_max_m > z_min_m else np.full(count, z_min_m)

    gas = _load_gas_interpolators(Path(args.gas_csv), z_min_m, z_max_m)
    vz_m_per_s = _sample_component(gas.linear_vz, gas.nearest_vz, z_birth_m, r_m)
    vr_raw_m_per_s = _sample_component(gas.linear_vr, gas.nearest_vr, z_birth_m, r_m)
    vr_m_per_s = float(args.radial_velocity_scale) * vr_raw_m_per_s
    speed_m_per_s = np.hypot(vz_m_per_s, vr_m_per_s)

    return SourceSamples(
        gas=gas,
        x_m=x_m,
        y_m=y_m,
        r_m=r_m,
        z_birth_m=z_birth_m,
        vz_m_per_s=vz_m_per_s,
        vr_raw_m_per_s=vr_raw_m_per_s,
        vr_m_per_s=vr_m_per_s,
        speed_m_per_s=speed_m_per_s,
    )


def _panel_title(panel: str, args: argparse.Namespace) -> str:
    titles = {
        "xy-density": "Gaussian source density",
        "xy-scatter": "Transverse source samples",
        "radial-density": "Radial source density",
        "birth-zr": "Birth-position samples",
        "vz-hist": "Axial-velocity distribution",
        "vr-hist": "Radial-velocity distribution",
        "velocity-phase": "Velocity phase space",
        "gas-vz-field": "Axial gas-velocity field",
        "birth-z-hist": "Birth-position distribution",
        "speed-hist": "Speed distribution",
    }
    return titles[panel]


def _set_panel_title(ax, panel: str, args: argparse.Namespace) -> None:
    if bool(args.panel_titles):
        ax.set_title(_panel_title(panel, args), loc="left")


def _plot_indices(samples: SourceSamples, args: argparse.Namespace) -> np.ndarray:
    return _subsample_indices(len(samples.r_m), int(args.max_plot_points))


def _add_colorbar(fig, mappable, ax, cax, label: str):
    if cax is None:
        colorbar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.055)
    else:
        colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_label(label)
    return colorbar


def _panel_uses_colorbar(panel: str, args: argparse.Namespace) -> bool:
    return panel in {"xy-density", "birth-zr", "gas-vz-field"} or (
        panel == "velocity-phase" and bool(args.phase_colorbar)
    )


def _draw_xy_density(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    limit = (
        float(args.xy_limit_mm)
        if float(args.xy_limit_mm) > 0.0
        else max(args.source_radius_mm * 1.2, args.source_gaussian_sigma_mm * 3.2)
    )
    hist = ax.hist2d(
        samples.x_m * 1.0e3,
        samples.y_m * 1.0e3,
        bins=int(args.xy_bins),
        range=[[-limit, limit], [-limit, limit]],
        cmap=args.xy_cmap,
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            args.source_radius_mm,
            fill=False,
            color="white",
            linewidth=scaled(0.9, args.paper_line_scale),
            alpha=0.9,
        )
    )
    _add_colorbar(fig, hist[3], ax, cax, "macro count/bin")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$y$ [mm]")
    _set_panel_title(ax, "xy-density", args)


def _draw_xy_scatter(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    plot_idx = _plot_indices(samples, args)
    x_plot_mm = samples.x_m[plot_idx] * 1.0e3
    y_plot_mm = samples.y_m[plot_idx] * 1.0e3
    ax.scatter(x_plot_mm, y_plot_mm, s=float(args.point_size), alpha=float(args.point_alpha), linewidths=0, color="#275d8c")
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            args.source_radius_mm,
            fill=False,
            color="black",
            linewidth=scaled(1.0, args.paper_line_scale),
        )
    )
    limit = (
        float(args.xy_limit_mm)
        if float(args.xy_limit_mm) > 0.0
        else max(args.source_radius_mm * 1.08, args.source_gaussian_sigma_mm * 3.0)
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$y$ [mm]")
    _set_panel_title(ax, "xy-scatter", args)
    ax.grid(True, alpha=0.22)


def _draw_radial_density(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    bins = np.linspace(0.0, args.source_radius_mm, int(args.radial_bins) + 1)
    counts, edges = np.histogram(samples.r_m * 1.0e3, bins=bins)
    shell_area_mm2 = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    area_density = counts / np.maximum(shell_area_mm2, 1.0e-30)
    if np.nanmax(area_density) > 0:
        area_density = area_density / np.nanmax(area_density)
    centers_mm = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers_mm, area_density, color="black", linewidth=scaled(1.65, args.paper_line_scale), label="sample / annulus area")
    if args.show_expected_radial:
        sigma_m = float(args.source_gaussian_sigma_mm) * 1.0e-3
        ax.plot(
            centers_mm,
            _expected_radial_density(centers_mm * 1.0e-3, sigma_m),
            color="#d62728",
            linestyle="--",
            linewidth=scaled(1.2, args.paper_line_scale),
            label="ideal Gaussian density",
        )
        ax.legend()
    ax.set_xlabel(r"$r$ [mm]")
    ax.set_ylabel("normalized areal density")
    _set_panel_title(ax, "radial-density", args)
    ax.grid(True, alpha=0.22)


def _draw_birth_z_hist(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    ax.hist(samples.z_birth_m * 1.0e3, bins=int(args.z_bins), color="#4c78a8", alpha=0.82)
    ax.set_xlabel("birth z used for gas velocity [mm]")
    ax.set_ylabel("count")
    _set_panel_title(ax, "birth-z-hist", args)
    ax.grid(True, alpha=0.22)


def _draw_birth_zr(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    plot_idx = _plot_indices(samples, args)
    scatter = ax.scatter(
        samples.z_birth_m[plot_idx] * 1.0e3,
        samples.r_m[plot_idx] * 1.0e3,
        c=samples.vz_m_per_s[plot_idx],
        s=float(args.point_size),
        alpha=float(args.point_alpha),
        cmap=args.velocity_cmap,
        linewidths=0,
    )
    _add_colorbar(fig, scatter, ax, cax, r"$v_z$ [m/s]")
    ax.set_xlabel(r"$z$ [mm]")
    ax.set_ylabel(r"$r$ [mm]")
    _set_panel_title(ax, "birth-zr", args)
    ax.grid(True, alpha=0.18)


def _draw_gas_vz_field(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    gas = samples.gas
    z_grid_mm = np.linspace(args.birth_z_min_mm, args.birth_z_max_mm, 160)
    if float(args.gas_vz_r_max_mm) > 0.0:
        r_max_mm = max(float(args.gas_vz_r_max_mm), float(args.source_radius_mm))
    else:
        r_max_mm = max(args.source_radius_mm, min(float(np.nanmax(gas.r_m)) * 1.0e3, 2.0))
    r_grid_mm = np.linspace(0.0, r_max_mm, 120)
    zg_mm, rg_mm = np.meshgrid(z_grid_mm, r_grid_mm)
    vz_grid = _sample_component(gas.linear_vz, gas.nearest_vz, zg_mm.ravel() * 1.0e-3, rg_mm.ravel() * 1.0e-3).reshape(zg_mm.shape)
    plot_idx = _plot_indices(samples, args)
    im = ax.contourf(zg_mm, rg_mm, vz_grid, levels=60, cmap="viridis")
    ax.scatter(samples.z_birth_m[plot_idx] * 1.0e3, samples.r_m[plot_idx] * 1.0e3, s=3, c="white", alpha=0.25, linewidths=0)
    _add_colorbar(fig, im, ax, cax, r"gas axial velocity $v_z$ [m/s]")
    ax.set_xlabel(r"$z$ [mm]")
    ax.set_ylabel(r"$r$ [mm]")
    ax.set_ylim(0.0, r_max_mm)
    _set_panel_title(ax, "gas-vz-field", args)


def _draw_vz_hist(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    ax.hist(samples.vz_m_per_s, bins=int(args.velocity_bins), color="#3b7ddd", alpha=0.9)
    ax.set_xlabel(r"$v_z$ [m/s]")
    ax.set_ylabel("count")
    _set_panel_title(ax, "vz-hist", args)
    ax.grid(True, alpha=0.22)


def _draw_vr_hist(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    ax.hist(samples.vr_m_per_s, bins=int(args.velocity_bins), color="#d45745", alpha=0.9)
    ax.set_xlabel(r"$v_r$ [m/s]")
    ax.set_ylabel("count")
    _set_panel_title(ax, "vr-hist", args)
    ax.grid(True, alpha=0.22)


def _draw_velocity_phase(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    plot_idx = _plot_indices(samples, args)
    color_by = str(args.phase_color_by).strip().lower()
    if color_by == "z":
        colors = samples.z_birth_m[plot_idx] * 1.0e3
        color_label = "birth z [mm]"
    elif color_by == "r":
        colors = samples.r_m[plot_idx] * 1.0e3
        color_label = "source r [mm]"
    elif color_by == "vz":
        colors = samples.vz_m_per_s[plot_idx]
        color_label = r"$v_z$ [m/s]"
    else:
        raise ValueError("--phase-color-by must be z, r, or vz.")
    scatter = ax.scatter(
        samples.vz_m_per_s[plot_idx],
        samples.vr_m_per_s[plot_idx],
        c=colors,
        s=float(args.point_size),
        alpha=float(args.point_alpha),
        cmap=args.phase_cmap,
        linewidths=0,
    )
    if args.phase_colorbar:
        _add_colorbar(fig, scatter, ax, cax, color_label)
    ax.axhline(0.0, color="black", linewidth=scaled(0.8, args.paper_line_scale))
    ax.set_xlabel(r"$v_z$ [m/s]")
    ax.set_ylabel(r"$v_r$ [m/s]")
    _set_panel_title(ax, "velocity-phase", args)
    ax.grid(True, alpha=0.22)


def _draw_speed_hist(ax, fig, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    ax.hist(samples.speed_m_per_s, bins=int(args.velocity_bins), color="#6f55aa", alpha=0.9)
    ax.set_xlabel(r"$|\mathbf{v}|$ [m/s]")
    ax.set_ylabel("count")
    _set_panel_title(ax, "speed-hist", args)
    ax.grid(True, alpha=0.22)


def _draw_panel(ax, fig, panel: str, samples: SourceSamples, args: argparse.Namespace, cax=None) -> None:
    draw = {
        "xy-density": _draw_xy_density,
        "xy-scatter": _draw_xy_scatter,
        "radial-density": _draw_radial_density,
        "birth-zr": _draw_birth_zr,
        "vz-hist": _draw_vz_hist,
        "vr-hist": _draw_vr_hist,
        "velocity-phase": _draw_velocity_phase,
        "gas-vz-field": _draw_gas_vz_field,
        "birth-z-hist": _draw_birth_z_hist,
        "speed-hist": _draw_speed_hist,
    }[panel]
    draw(ax, fig, samples, args, cax=cax)


def _create_group_axes(fig, rows: int, cols: int, panels: list[str], args: argparse.Namespace):
    """Create aligned main axes with an equally sized colorbar slot per panel."""

    width_ratios = [ratio for _ in range(cols) for ratio in (1.0, 0.055)]
    grid = fig.add_gridspec(
        rows,
        cols * 2,
        width_ratios=width_ratios,
        wspace=0.08,
    )
    axes = []
    colorbar_axes = []
    for index in range(rows * cols):
        row, col = divmod(index, cols)
        ax = fig.add_subplot(grid[row, 2 * col])
        cax = fig.add_subplot(grid[row, 2 * col + 1])
        if index >= len(panels):
            ax.axis("off")
            cax.axis("off")
        elif not _panel_uses_colorbar(panels[index], args):
            cax.axis("off")
        axes.append(ax)
        colorbar_axes.append(cax)
    return axes, colorbar_axes


def _summary(samples: SourceSamples, args: argparse.Namespace, outputs: list[Path], panels: list[str]) -> dict[str, object]:
    gas = samples.gas
    return {
        "gas_csv": str(args.gas_csv),
        "outputs": [str(path) for path in outputs],
        "panels": panels,
        "separate_panels": bool(args.separate_panels),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "source_gaussian_sigma_mm": float(args.source_gaussian_sigma_mm),
        "source_radius_mm": float(args.source_radius_mm),
        "birth_z_min_mm": float(args.birth_z_min_mm),
        "birth_z_max_mm": float(args.birth_z_max_mm),
        "radial_velocity_scale": float(args.radial_velocity_scale),
        "xy_limit_mm": float(args.xy_limit_mm),
        "gas_vz_r_max_mm": float(args.gas_vz_r_max_mm),
        "phase_cmap": str(args.phase_cmap),
        "panel_titles": bool(args.panel_titles),
        "sample_r_p50_mm": float(np.nanpercentile(samples.r_m * 1.0e3, 50)),
        "sample_r_p95_mm": float(np.nanpercentile(samples.r_m * 1.0e3, 95)),
        "vz_p50_m_per_s": float(np.nanpercentile(samples.vz_m_per_s, 50)),
        "vz_p95_m_per_s": float(np.nanpercentile(samples.vz_m_per_s, 95)),
        "vr_p50_m_per_s": float(np.nanpercentile(samples.vr_m_per_s, 50)),
        "vr_p95_m_per_s": float(np.nanpercentile(samples.vr_m_per_s, 95)),
        "speed_p50_m_per_s": float(np.nanpercentile(samples.speed_m_per_s, 50)),
        "speed_p95_m_per_s": float(np.nanpercentile(samples.speed_m_per_s, 95)),
        "gas_source_z_min_mm": float(np.nanmin(gas.z_m) * 1.0e3),
        "gas_source_z_max_mm": float(np.nanmax(gas.z_m) * 1.0e3),
        "gas_source_r_max_mm": float(np.nanmax(gas.r_m) * 1.0e3),
    }


def _output_base(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).with_suffix("")
    return Path(args.output_dir) / str(args.output_prefix)


def _save_summary(args: argparse.Namespace, samples: SourceSamples, outputs: list[Path], panels: list[str]) -> Path:
    base = _output_base(args)
    summary_path = base.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_summary(samples, args, outputs, panels), indent=2), encoding="utf-8")
    return summary_path


def make_plot(args: argparse.Namespace) -> list[Path]:
    apply_paper_style(font_scale=args.paper_font_scale, line_scale=args.paper_line_scale)
    panels = _parse_panels(args.panels)
    samples = _build_samples(args)
    outputs: list[Path] = []
    base = _output_base(args)

    if args.separate_panels:
        for panel in panels:
            fig, ax = plt.subplots(figsize=_figure_size(1, 1), dpi=args.dpi, constrained_layout=True)
            _draw_panel(ax, fig, panel, samples, args)
            output = base.parent / f"{base.name}_{panel}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output)
            plt.close(fig)
            outputs.append(output)
    else:
        rows, cols = _layout_shape(args.layout, len(panels))
        fig = plt.figure(figsize=_figure_size(rows, cols), dpi=args.dpi, constrained_layout=True)
        axes, colorbar_axes = _create_group_axes(fig, rows, cols, panels, args)
        for ax, cax, panel in zip(axes, colorbar_axes, panels):
            _draw_panel(ax, fig, panel, samples, args, cax=cax)
        if str(args.title).strip():
            fig.suptitle(str(args.title).strip())
        labels_enabled = bool(args.subplot_labels) and len(panels) > 1
        if bool(args.panel_titles) and labels_enabled:
            for index, (ax, panel) in enumerate(zip(axes, panels)):
                ax.set_title(f"{subplot_label(index)} {_panel_title(panel, args)}", loc="left")
        else:
            add_subplot_labels(
                axes[: len(panels)],
                enabled=labels_enabled,
                x=0.0,
                y=1.04,
                font_scale=args.paper_font_scale,
                line_scale=args.paper_line_scale,
            )
        output = Path(args.output) if args.output else base.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)
        outputs.append(output)

    summary_path = _save_summary(args, samples, outputs, panels)
    for output in outputs:
        print(f"Saved: {output}")
    print(f"Saved: {summary_path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw source-only diagnostics for a truncated 2D Gaussian ion source and "
            "birth velocities sampled from raw tube Fluent gas data."
        )
    )
    parser.add_argument("--gas-csv", default=str(DEFAULT_GAS_CSV), help="Raw Fluent tube gas CSV path.")
    parser.add_argument("--output-dir", default="outputs/paper_figures", help="Output directory when --output is omitted.")
    parser.add_argument("--output", default="", help="Output PNG path.")
    parser.add_argument("--output-prefix", default="gaussian_source_diagnostics", help="Output filename prefix when --output is omitted.")
    parser.add_argument(
        "--panels",
        default="paper",
        help=(
            "Comma-separated panels or preset. Presets: paper, source, gas, all. "
            "Panels: xy-density, radial-density, birth-zr, vz-hist, vr-hist, "
            "velocity-phase, gas-vz-field, xy-scatter, birth-z-hist, speed-hist."
        ),
    )
    parser.add_argument("--layout", default="auto", help="Group layout: auto or rowsxcols, for example 1x1, 1x3, 2x3.")
    parser.add_argument("--separate-panels", action="store_true", help="Write each selected panel as a separate PNG.")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of synthetic source samples.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--source-gaussian-sigma-mm", type=float, default=0.05, help="Gaussian sigma for x/y [mm].")
    parser.add_argument("--source-radius-mm", type=float, default=0.15, help="Hard radial truncation radius [mm].")
    parser.add_argument("--birth-z-min-mm", type=float, default=0.0, help="Minimum raw tube z used for birth velocity [mm].")
    parser.add_argument("--birth-z-max-mm", type=float, default=4.4, help="Maximum raw tube z used for birth velocity [mm].")
    parser.add_argument("--radial-velocity-scale", type=float, default=1.0, help="Multiplier applied to sampled radial gas velocity.")
    parser.add_argument("--xy-limit-mm", type=float, default=0.0, help="Half-width for xy-density/xy-scatter axes [mm]. Use 0 for automatic.")
    parser.add_argument("--gas-vz-r-max-mm", type=float, default=0.0, help="Maximum r shown in gas-vz-field panel [mm]. Use 0 for automatic.")
    parser.add_argument("--max-plot-points", type=int, default=25000, help="Maximum points shown in scatter panels.")
    parser.add_argument("--radial-bins", type=int, default=50, help="Radial density histogram bin count.")
    parser.add_argument("--xy-bins", type=int, default=90, help="2D source-density histogram bin count per axis.")
    parser.add_argument("--z-bins", type=int, default=45, help="Birth-z histogram bin count.")
    parser.add_argument("--velocity-bins", type=int, default=70, help="Velocity histogram bin count.")
    parser.add_argument("--point-size", type=float, default=4.0, help="Scatter point size.")
    parser.add_argument("--point-alpha", type=float, default=0.36, help="Scatter point alpha.")
    parser.add_argument("--xy-cmap", default="magma", help="Colormap for 2D source density.")
    parser.add_argument("--velocity-cmap", default="viridis", help="Colormap for birth z-r velocity plot.")
    parser.add_argument("--phase-cmap", default="magma", help="Colormap for velocity phase-space plot.")
    parser.add_argument("--phase-color-by", choices=["z", "r", "vz"], default="z", help="Quantity used to color velocity phase space.")
    parser.add_argument(
        "--phase-colorbar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show colorbar on velocity phase-space panel.",
    )
    parser.add_argument("--show-expected-radial", action="store_true", help="Overlay ideal Gaussian areal density on radial panel.")
    parser.add_argument("--title", default="", help="Optional group-figure title.")
    parser.add_argument(
        "--panel-titles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show concise per-panel titles. Disabled by default for manuscript figures.",
    )
    parser.add_argument(
        "--subplot-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label grouped subplots as (a), (b), ... for manuscript figures.",
    )
    parser.add_argument("--paper-font-scale", type=float, default=1.18, help="Global font-size multiplier for paper figures.")
    parser.add_argument("--paper-line-scale", type=float, default=1.25, help="Line-width multiplier for paper figures.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    make_plot(parse_args())


if __name__ == "__main__":
    main()
