"""Diagnostic plots generated from immutable report rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...config import SimulationConfig


def _as_float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, np.nan)))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _sample_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_points, dtype=np.int64)


def _load_pyplot(error_path: Path) -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        error_path.write_text(repr(exc), encoding="utf-8")
        return None
    return plt


def _active_plot_data(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    active = [
        row for row in rows
        if str(row.get("final_status", "")) == "active"
    ]
    if not active:
        return None
    z_mm = _as_float_array(active, "final_z_m") * 1.0e3
    r_mm = _as_float_array(active, "final_r_m") * 1.0e3
    tof_ms = _as_float_array(active, "tof_s") * 1.0e3
    weights = _as_float_array(active, "represented_real_ions")
    finite = np.isfinite(z_mm) & np.isfinite(r_mm) & np.isfinite(tof_ms)
    if not np.any(finite):
        return None
    finite_weights = weights[finite]
    finite_weights = np.where(
        np.isfinite(finite_weights) & (finite_weights > 0.0),
        finite_weights,
        1.0,
    )
    return z_mm[finite], r_mm[finite], tof_ms[finite], finite_weights


def _active_plot_limits(
    z_mm: np.ndarray,
    r_mm: np.ndarray,
    config: SimulationConfig,
) -> tuple[float, float, float, float, float]:
    detector_z_mm = float(
        config.detector_z_m or config.domain_length_m
    ) * 1.0e3
    radial_limit_mm = float(
        config.radial_limit_m or config.domain_radius_m
    ) * 1.0e3
    capillary_exit_mm = (
        float(config.capillary_exit_z_m) * 1.0e3
        if config.capillary_exit_z_m is not None
        else np.nan
    )
    z_max = max(detector_z_mm, float(np.nanpercentile(z_mm, 99.5)))
    r_max = max(min(radial_limit_mm, 20.0), float(np.nanpercentile(r_mm, 99.5)))
    r_max = min(max(r_max * 1.08, 1.0), radial_limit_mm)
    return detector_z_mm, radial_limit_mm, capillary_exit_mm, z_max, r_max


def _draw_active_scatter(
    fig: Any,
    axis: Any,
    z_mm: np.ndarray,
    r_mm: np.ndarray,
    tof_ms: np.ndarray,
    sample: np.ndarray,
    capillary_exit_mm: float,
    z_max: float,
    r_max: float,
) -> None:
    scatter = axis.scatter(
        z_mm[sample],
        r_mm[sample],
        c=tof_ms[sample],
        s=3,
        alpha=0.55,
        cmap="viridis",
        linewidths=0,
    )
    if np.isfinite(capillary_exit_mm):
        axis.axvline(
            capillary_exit_mm,
            color="cyan",
            linestyle=":",
            linewidth=1.1,
        )
    axis.set_xlabel("z [mm]")
    axis.set_ylabel("r [mm]")
    axis.set_title("Final active ions only, colored by TOF")
    axis.set_xlim(0.0, max(z_max, 1.0))
    axis.set_ylim(0.0, max(r_max, 1.0))
    axis.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axis).set_label("TOF / ion age [ms]")


def _draw_active_histogram(
    axis: Any,
    z_mm: np.ndarray,
    weights: np.ndarray,
    z_max: float,
) -> None:
    bin_count = min(80, max(20, int(np.sqrt(len(z_mm)))))
    bins = np.linspace(
        max(0.0, float(np.nanmin(z_mm))),
        max(z_max, float(np.nanmax(z_mm))),
        bin_count,
    )
    axis.hist(z_mm, bins=bins, weights=weights, color="#4C78A8", alpha=0.82)
    axis.set_xlabel("z [mm]")
    axis.set_ylabel("represented active real ions / bin")
    axis.set_title("Final active ion axial distribution")
    axis.grid(True, alpha=0.25)


def write_final_active_spatial_plot(
    path: Path,
    final_particle_rows: list[dict[str, Any]],
    config: SimulationConfig,
    *,
    max_points: int = 200_000,
) -> Path | None:
    """Write the default final active-ion r-z plot."""

    active_data = _active_plot_data(final_particle_rows)
    if active_data is None:
        return None
    plt = _load_pyplot(path.parent / "final_active_spatial_plot_error.txt")
    if plt is None:
        return None
    z_mm, r_mm, tof_ms, weights = active_data
    _, _, capillary_exit_mm, z_max, r_max = _active_plot_limits(
        z_mm,
        r_mm,
        config,
    )
    sample = _sample_indices(len(z_mm), max_points)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180)
    _draw_active_scatter(
        fig, axes[0], z_mm, r_mm, tof_ms, sample,
        capillary_exit_mm, z_max, r_max,
    )
    _draw_active_histogram(axes[1], z_mm, weights, z_max)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _exit_plot_data(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[np.ndarray, ...]]:
    exits = [row for row in rows if str(row.get("status", "")) == "z_exit"]
    if not exits:
        return exits, ()
    x_mm = _as_float_array(exits, "x_m") * 1.0e3
    y_mm = _as_float_array(exits, "y_m") * 1.0e3
    r_mm = _as_float_array(exits, "r_m") * 1.0e3
    tof_us = _as_float_array(exits, "tof_s") * 1.0e6
    weights = _as_float_array(exits, "represented_real_ions")
    finite = (
        np.isfinite(x_mm)
        & np.isfinite(y_mm)
        & np.isfinite(r_mm)
        & np.isfinite(tof_us)
    )
    finite_weights = weights[finite]
    finite_weights = np.where(
        np.isfinite(finite_weights) & (finite_weights > 0.0),
        finite_weights,
        1.0,
    )
    return exits, (
        x_mm[finite],
        y_mm[finite],
        r_mm[finite],
        tof_us[finite],
        finite_weights,
    )


def _draw_empty_exit(axes: Any) -> None:
    axes[0].text(
        0.5, 0.5, "No z_exit events",
        ha="center", va="center", transform=axes[0].transAxes,
    )
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("z_exit x-y distribution")
    axes[1].text(
        0.5, 0.5, "No z_exit events",
        ha="center", va="center", transform=axes[1].transAxes,
    )
    axes[1].set_xlabel("r [mm]")
    axes[1].set_ylabel("represented real ions / bin")


def _draw_exit_scatter(
    plt: Any,
    fig: Any,
    axis: Any,
    arrays: tuple[np.ndarray, ...],
    detector_radius_mm: float,
    max_points: int,
) -> None:
    x_mm, y_mm, _, tof_us, _ = arrays
    sample = _sample_indices(len(x_mm), max_points)
    scatter = axis.scatter(
        x_mm[sample], y_mm[sample], c=tof_us[sample],
        s=4, alpha=0.55, cmap="viridis", linewidths=0,
    )
    axis.add_patch(
        plt.Circle(
            (0.0, 0.0),
            detector_radius_mm,
            fill=False,
            color="black",
            linewidth=1.0,
        )
    )
    extent = max(
        detector_radius_mm,
        float(np.nanpercentile(np.abs(np.r_[x_mm, y_mm]), 99.5)),
    ) * 1.08
    extent = max(extent, 1.0)
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_title("z_exit x-y distribution")
    axis.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axis).set_label("TOF [us]")


def _draw_exit_histogram(
    axis: Any,
    arrays: tuple[np.ndarray, ...],
    detector_radius_mm: float,
    row_count: int,
) -> None:
    _, _, r_mm, _, weights = arrays
    bins = np.linspace(
        0.0,
        max(detector_radius_mm, float(np.nanmax(r_mm))),
        60,
    )
    axis.hist(r_mm, bins=bins, weights=weights, color="#4C78A8", alpha=0.82)
    axis.axvline(
        detector_radius_mm,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="detector radius",
    )
    axis.set_xlabel("exit radius [mm]")
    axis.set_ylabel("represented real ions / bin")
    axis.set_title(f"z_exit radial distribution, rows={row_count}")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)


def write_z_exit_xy_plot(
    path: Path,
    terminal_event_rows: list[dict[str, Any]],
    config: SimulationConfig,
    *,
    max_points: int = 200_000,
) -> Path | None:
    """Write an x-y distribution plot for cumulative z_exit events."""

    plt = _load_pyplot(path.parent / "z_exit_xy_plot_error.txt")
    if plt is None:
        return None
    exit_rows, arrays = _exit_plot_data(terminal_event_rows)
    detector_radius_mm = float(
        config.detector_radius_m
        or config.radial_limit_m
        or config.domain_radius_m
    ) * 1.0e3
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=180)
    if not exit_rows:
        _draw_empty_exit(axes)
    else:
        _draw_exit_scatter(
            plt, fig, axes[0], arrays, detector_radius_mm, max_points,
        )
        _draw_exit_histogram(
            axes[1], arrays, detector_radius_mm, len(exit_rows),
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
