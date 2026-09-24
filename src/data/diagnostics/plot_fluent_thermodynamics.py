"""Plot Mach number, temperature, and pressure from a Fluent axisymmetric export."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from scipy.spatial import cKDTree

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from .plot_style import add_subplot_labels, apply_paper_style
except ImportError:  # pragma: no cover - supports direct script execution.
    from src.data.diagnostics.plot_style import (  # type: ignore
        add_subplot_labels,
        apply_paper_style,
    )


DEFAULT_INPUT = (
    Path("manuscript")
    / "slens_Efield_Fluent_baker"
    / "P1baked_capaligned_L101_exit6_0p01"
    / "slens_inlet1p5_P1mbar_400K.txt"
)

COLUMN_ALIASES = {
    "z_m": ("xcoordinate", "zcoordinate", "zm", "z"),
    "r_m": ("ycoordinate", "rcoordinate", "rm", "r"),
    "pressure_pa": ("absolutepressure", "staticpressure", "pressurepa", "pressure"),
    "v_z_m_per_s": ("axialvelocity", "xvelocity", "velocityz", "vz"),
    "v_r_m_per_s": ("radialvelocity", "yvelocity", "velocityr", "vr"),
    "temperature_k": ("statictemperature", "temperaturek", "temperature"),
}


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _resolve_columns(path: Path) -> dict[str, str]:
    header = pd.read_csv(path, nrows=0, skipinitialspace=True)
    normalized = {_normalize_header(column): str(column) for column in header.columns}
    resolved: dict[str, str] = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                resolved[field_name] = normalized[alias]
                break
        else:
            available = ", ".join(str(column).strip() for column in header.columns)
            raise KeyError(
                f"Could not identify {field_name!r} in {path}. "
                f"Expected one of {aliases}; available columns: {available}"
            )
    return resolved


def _load_fluent_points(
    path: Path,
    *,
    z_offset_mm: float,
    gamma: float,
    specific_gas_constant_j_per_kg_k: float,
) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Fluent export not found: {path}")

    columns = _resolve_columns(path)
    frame = pd.read_csv(path, usecols=list(columns.values()), skipinitialspace=True)
    values = {
        field_name: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        for field_name, column in columns.items()
    }

    finite = np.ones(len(frame), dtype=bool)
    for array in values.values():
        finite &= np.isfinite(array)
    finite &= values["temperature_k"] > 0.0
    finite &= values["pressure_pa"] > 0.0
    if not np.any(finite):
        raise ValueError(f"No finite positive-pressure/temperature points found in {path}")

    result = {key: array[finite] for key, array in values.items()}
    result["z_mm"] = result.pop("z_m") * 1.0e3 + float(z_offset_mm)
    result["r_mm"] = result.pop("r_m") * 1.0e3
    speed_m_per_s = np.hypot(result["v_z_m_per_s"], result["v_r_m_per_s"])
    sound_speed_m_per_s = np.sqrt(
        float(gamma) * float(specific_gas_constant_j_per_kg_k) * result["temperature_k"]
    )
    result["mach_number"] = speed_m_per_s / sound_speed_m_per_s
    result["pressure_mbar"] = result["pressure_pa"] / 100.0
    return result


def _crop_points(
    points: dict[str, np.ndarray],
    *,
    z_min_mm: float | None,
    z_max_mm: float | None,
    r_min_mm: float | None,
    r_max_mm: float | None,
) -> tuple[dict[str, np.ndarray], tuple[float, float, float, float]]:
    z_mm = points["z_mm"]
    r_mm = points["r_mm"]
    resolved_z_min = float(np.nanmin(z_mm)) if z_min_mm is None else float(z_min_mm)
    resolved_z_max = float(np.nanmax(z_mm)) if z_max_mm is None else float(z_max_mm)
    resolved_r_min = float(np.nanmin(r_mm)) if r_min_mm is None else float(r_min_mm)
    resolved_r_max = float(np.nanmax(r_mm)) if r_max_mm is None else float(r_max_mm)
    if resolved_z_max <= resolved_z_min:
        raise ValueError("z max must be greater than z min.")
    if resolved_r_max <= resolved_r_min:
        raise ValueError("r max must be greater than r min.")

    selected = (
        (z_mm >= resolved_z_min)
        & (z_mm <= resolved_z_max)
        & (r_mm >= resolved_r_min)
        & (r_mm <= resolved_r_max)
    )
    if int(np.count_nonzero(selected)) < 4:
        raise ValueError(
            "Fewer than four Fluent points remain in the requested window: "
            f"z=[{resolved_z_min:g}, {resolved_z_max:g}] mm, "
            f"r=[{resolved_r_min:g}, {resolved_r_max:g}] mm."
        )
    return (
        {key: value[selected] for key, value in points.items()},
        (resolved_z_min, resolved_z_max, resolved_r_min, resolved_r_max),
    )


def _query_grid(
    z_mm: np.ndarray,
    r_mm: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    nz: int,
    nr: int,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_min_mm, z_max_mm, r_min_mm, r_max_mm = bounds
    z_grid_mm = np.linspace(z_min_mm, z_max_mm, int(nz), dtype=np.float64)
    r_grid_mm = np.linspace(r_min_mm, r_max_mm, int(nr), dtype=np.float64)
    z_mesh_mm, r_mesh_mm = np.meshgrid(z_grid_mm, r_grid_mm)

    tree = cKDTree(np.column_stack((z_mm, r_mm)))
    query_points = np.column_stack((z_mesh_mm.ravel(), r_mesh_mm.ravel()))
    distances, indices = tree.query(query_points, k=int(neighbors), workers=-1)
    if int(neighbors) == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return z_mesh_mm, r_mesh_mm, np.asarray(distances), np.asarray(indices)


def _interpolate_idw(
    values: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, int],
    *,
    power: float,
    max_distance_mm: float | None,
) -> np.ma.MaskedArray:
    sampled = np.asarray(values, dtype=np.float64)[indices]
    exact = distances[:, 0] <= 1.0e-12
    safe_distances = np.maximum(distances, 1.0e-12)
    weights = np.power(safe_distances, -float(power))
    interpolated = np.sum(weights * sampled, axis=1) / np.sum(weights, axis=1)
    interpolated[exact] = sampled[exact, 0]

    mask = ~np.isfinite(interpolated)
    if max_distance_mm is not None:
        mask |= distances[:, 0] > float(max_distance_mm)
    return np.ma.array(interpolated.reshape(shape), mask=mask.reshape(shape))


def _finite_values(values: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        finite = np.asarray(values.compressed(), dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
    return finite[np.isfinite(finite)]


def _linear_norm(
    values: np.ndarray | np.ma.MaskedArray,
    *,
    lower_percentile: float,
    upper_percentile: float,
    explicit_min: float | None,
    explicit_max: float | None,
) -> Normalize:
    finite = _finite_values(values)
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.percentile(finite, lower_percentile)) if explicit_min is None else float(explicit_min)
    vmax = float(np.percentile(finite, upper_percentile)) if explicit_max is None else float(explicit_max)
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin) * 1.0e-6, 1.0e-9)
    return Normalize(vmin=vmin, vmax=vmax)


def _pressure_norm(
    values: np.ndarray | np.ma.MaskedArray,
    *,
    lower_percentile: float,
    upper_percentile: float,
    explicit_min: float | None,
    explicit_max: float | None,
    logarithmic: bool,
) -> Normalize:
    if not logarithmic:
        return _linear_norm(
            values,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            explicit_min=explicit_min,
            explicit_max=explicit_max,
        )
    positive = _finite_values(values)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return LogNorm(vmin=1.0e-3, vmax=1.0)
    vmin = float(np.percentile(positive, lower_percentile)) if explicit_min is None else float(explicit_min)
    vmax = float(np.percentile(positive, upper_percentile)) if explicit_max is None else float(explicit_max)
    vmin = max(vmin, float(np.min(positive)), 1.0e-12)
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def _mach_norm(
    values: np.ndarray | np.ma.MaskedArray,
    *,
    lower_percentile: float,
    upper_percentile: float,
    explicit_min: float | None,
    explicit_max: float | None,
    center: float,
) -> Normalize:
    linear = _linear_norm(
        values,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        explicit_min=explicit_min,
        explicit_max=explicit_max,
    )
    if float(linear.vmin) < float(center) < float(linear.vmax):
        return TwoSlopeNorm(vmin=float(linear.vmin), vcenter=float(center), vmax=float(linear.vmax))
    return linear


def _stats(values: np.ndarray | np.ma.MaskedArray) -> dict[str, float]:
    finite = _finite_values(values)
    if finite.size == 0:
        return {"min": float("nan"), "p50": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(finite)),
        "p50": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _resolve_z_offset(args: argparse.Namespace) -> float:
    if args.capillary_length_mm is not None:
        if args.capillary_exit_z_mm is None:
            raise ValueError("--capillary-length-mm requires --capillary-exit-z-mm.")
        if args.z_offset_mm is not None:
            raise ValueError("Use either --z-offset-mm or capillary alignment, not both.")
        return float(args.capillary_exit_z_mm) - float(args.capillary_length_mm)
    return 0.0 if args.z_offset_mm is None else float(args.z_offset_mm)


def plot_fields(args: argparse.Namespace) -> tuple[Path, Path | None]:
    apply_paper_style(font_scale=1.0, line_scale=1.0)
    input_path = Path(args.gas_csv)
    z_offset_mm = _resolve_z_offset(args)
    points_all = _load_fluent_points(
        input_path,
        z_offset_mm=z_offset_mm,
        gamma=args.gamma,
        specific_gas_constant_j_per_kg_k=args.specific_gas_constant_j_per_kg_k,
    )
    points, bounds = _crop_points(
        points_all,
        z_min_mm=args.z_min_mm,
        z_max_mm=args.z_max_mm,
        r_min_mm=args.r_min_mm,
        r_max_mm=args.r_max_mm,
    )
    z_mesh_mm, r_mesh_mm, distances, indices = _query_grid(
        points["z_mm"],
        points["r_mm"],
        bounds,
        nz=args.grid_nz,
        nr=args.grid_nr,
        neighbors=args.neighbors,
    )
    grid_shape = z_mesh_mm.shape
    mach = _interpolate_idw(
        points["mach_number"],
        distances,
        indices,
        grid_shape,
        power=args.idw_power,
        max_distance_mm=args.max_distance_mm,
    )
    temperature_k = _interpolate_idw(
        points["temperature_k"],
        distances,
        indices,
        grid_shape,
        power=args.idw_power,
        max_distance_mm=args.max_distance_mm,
    )
    pressure_mbar = _interpolate_idw(
        points["pressure_mbar"],
        distances,
        indices,
        grid_shape,
        power=args.idw_power,
        max_distance_mm=args.max_distance_mm,
    )

    linear_kwargs = {
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
    }
    mach_norm = _mach_norm(
        mach,
        **linear_kwargs,
        explicit_min=args.mach_min,
        explicit_max=args.mach_max,
        center=args.mach_center,
    )
    temperature_norm = _linear_norm(
        temperature_k,
        **linear_kwargs,
        explicit_min=args.temperature_min_k,
        explicit_max=args.temperature_max_k,
    )
    pressure_norm = _pressure_norm(
        pressure_mbar,
        **linear_kwargs,
        explicit_min=args.pressure_min_mbar,
        explicit_max=args.pressure_max_mbar,
        logarithmic=not args.linear_pressure,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(args.figure_width_in, args.figure_height_in),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = (
        (mach, "Mach number", "Mach number", args.mach_cmap, mach_norm),
        (temperature_k, "Static temperature", "Temperature [K]", args.temperature_cmap, temperature_norm),
        (pressure_mbar, "Absolute pressure", "Pressure [mbar]", args.pressure_cmap, pressure_norm),
    )
    for ax, (values, title, colorbar_label, cmap, norm) in zip(axes, panels):
        image = ax.pcolormesh(z_mesh_mm, r_mesh_mm, values, shading="auto", cmap=cmap, norm=norm)
        colorbar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.035)
        colorbar.set_label(colorbar_label)
        ax.set_ylabel("r [mm]")
        if not args.no_panel_titles:
            ax.set_title(title, loc="left", pad=5.0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        if args.capillary_exit_z_mm is not None and not args.no_exit_line:
            ax.axvline(float(args.capillary_exit_z_mm), color="white", linestyle="--", linewidth=1.1, alpha=0.9)

    if not args.no_sonic_contour:
        mach_finite = _finite_values(mach)
        if mach_finite.size and float(np.min(mach_finite)) <= 1.0 <= float(np.max(mach_finite)):
            axes[0].contour(
                z_mesh_mm,
                r_mesh_mm,
                mach,
                levels=[1.0],
                colors="black",
                linewidths=1.0,
            )
    axes[-1].set_xlabel("z [mm]")
    add_subplot_labels(axes, x=0.012, y=0.965, font_scale=1.0, line_scale=1.0)
    if args.title:
        fig.suptitle(args.title)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "plots" / f"{input_path.stem}_mach_temperature_pressure_3x1.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, facecolor="white")
    plt.close(fig)

    summary_path: Path | None = None
    if not args.no_summary:
        summary_path = output_path.with_suffix(".summary.json")
        summary: dict[str, Any] = {
            "input": str(input_path),
            "output": str(output_path),
            "raw_finite_point_count": int(points_all["z_mm"].size),
            "window_point_count": int(points["z_mm"].size),
            "z_offset_mm": float(z_offset_mm),
            "bounds_mm": {
                "z_min": float(bounds[0]),
                "z_max": float(bounds[1]),
                "r_min": float(bounds[2]),
                "r_max": float(bounds[3]),
            },
            "grid": {"nz": int(args.grid_nz), "nr": int(args.grid_nr)},
            "interpolation": {
                "method": "inverse_distance_weighted",
                "neighbors": int(args.neighbors),
                "power": float(args.idw_power),
                "max_distance_mm": args.max_distance_mm,
            },
            "mach_definition": {
                "formula": "hypot(v_z, v_r) / sqrt(gamma * R_specific * T)",
                "gamma": float(args.gamma),
                "specific_gas_constant_j_per_kg_k": float(args.specific_gas_constant_j_per_kg_k),
            },
            "display": {
                "lower_percentile": float(args.lower_percentile),
                "upper_percentile": float(args.upper_percentile),
                "pressure_logarithmic": bool(not args.linear_pressure),
                "mach_center": float(args.mach_center),
                "mach_limits": [float(mach_norm.vmin), float(mach_norm.vmax)],
                "temperature_limits_k": [float(temperature_norm.vmin), float(temperature_norm.vmax)],
                "pressure_limits_mbar": [float(pressure_norm.vmin), float(pressure_norm.vmax)],
            },
            "grid_statistics": {
                "mach_number": _stats(mach),
                "temperature_k": _stats(temperature_k),
                "pressure_mbar": _stats(pressure_mbar),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Saved figure: {output_path}")
    if summary_path is not None:
        print(f"Saved summary: {summary_path}")
    return output_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Fluent Mach number, static temperature, and absolute pressure as a 3x1 field figure."
    )
    parser.add_argument("gas_csv", nargs="?", default=str(DEFAULT_INPUT), help="Fluent comma-separated text export.")
    parser.add_argument("--output", default="", help="Output image path; defaults to a plots folder beside the input.")
    parser.add_argument("--z-offset-mm", type=float, default=None, help="Manual offset added to raw Fluent x/z [mm].")
    parser.add_argument("--capillary-length-mm", type=float, default=None, help="Raw Fluent capillary length [mm].")
    parser.add_argument("--capillary-exit-z-mm", type=float, default=None, help="Aligned capillary-exit z [mm].")
    parser.add_argument("--z-min-mm", type=float, default=None)
    parser.add_argument("--z-max-mm", type=float, default=None)
    parser.add_argument("--r-min-mm", type=float, default=None)
    parser.add_argument("--r-max-mm", type=float, default=None)
    parser.add_argument("--grid-nz", type=int, default=800)
    parser.add_argument("--grid-nr", type=int, default=240)
    parser.add_argument("--neighbors", type=int, default=4, help="KD-tree neighbors used by IDW interpolation.")
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--max-distance-mm", type=float, default=None, help="Mask grid cells farther from source data.")
    parser.add_argument("--gamma", type=float, default=1.4, help="Heat-capacity ratio used for Mach calculation.")
    parser.add_argument(
        "--specific-gas-constant-j-per-kg-k",
        type=float,
        default=296.80,
        help="Specific gas constant; default is nitrogen.",
    )
    parser.add_argument("--lower-percentile", type=float, default=0.0, help="Lower color-limit percentile.")
    parser.add_argument("--upper-percentile", type=float, default=100.0, help="Upper color-limit percentile.")
    parser.add_argument("--mach-min", type=float, default=None)
    parser.add_argument("--mach-max", type=float, default=None)
    parser.add_argument("--mach-center", type=float, default=1.0, help="Center of the red-blue Mach color scale.")
    parser.add_argument("--temperature-min-k", type=float, default=None)
    parser.add_argument("--temperature-max-k", type=float, default=None)
    parser.add_argument("--pressure-min-mbar", type=float, default=None)
    parser.add_argument("--pressure-max-mbar", type=float, default=None)
    parser.add_argument("--linear-pressure", action="store_true", help="Use a linear rather than logarithmic pressure scale.")
    parser.add_argument("--mach-cmap", default="RdBu_r")
    parser.add_argument("--temperature-cmap", default="inferno")
    parser.add_argument("--pressure-cmap", default="viridis")
    parser.add_argument("--figure-width-in", type=float, default=8.2)
    parser.add_argument("--figure-height-in", type=float, default=9.6)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--title", default="", help="Optional overall title; empty by default.")
    parser.add_argument("--no-panel-titles", action="store_true")
    parser.add_argument("--no-sonic-contour", action="store_true")
    parser.add_argument("--no-exit-line", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.grid_nz < 2 or args.grid_nr < 2:
        parser.error("--grid-nz and --grid-nr must both be at least 2.")
    if args.neighbors < 1:
        parser.error("--neighbors must be at least 1.")
    if args.idw_power <= 0.0:
        parser.error("--idw-power must be positive.")
    if not (0.0 <= args.lower_percentile < args.upper_percentile <= 100.0):
        parser.error("Percentiles must satisfy 0 <= lower < upper <= 100.")
    if args.gamma <= 0.0 or args.specific_gas_constant_j_per_kg_k <= 0.0:
        parser.error("Mach-number gas properties must be positive.")
    plot_fields(args)


if __name__ == "__main__":
    main()
