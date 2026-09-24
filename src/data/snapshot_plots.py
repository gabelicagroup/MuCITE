"""Plot adapters for persisted particle snapshots and trajectories."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SnapshotPlotData:
    points: np.ndarray
    sample: np.ndarray
    z_mm: np.ndarray
    r_mm: np.ndarray
    weights: np.ndarray
    colors: np.ndarray
    color_label: str
    z_col: int
    x_col: int
    y_col: int


def _load_pyplot(error_path: Path) -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        error_path.write_text(repr(exc), encoding="utf-8")
        return None
    return plt


def _final_snapshot(logger: Any) -> tuple[Any | None, np.ndarray | None]:
    final_record = None
    final_snapshot = None
    for record, snapshot in logger._iter_snapshot_arrays():
        if snapshot.size:
            final_record = record
            final_snapshot = snapshot
    return final_record, final_snapshot


def _snapshot_plot_data(logger: Any, snapshot: np.ndarray) -> SnapshotPlotData:
    points = np.asarray(snapshot, dtype=np.float64)
    r_col = logger._column_index("r_m")
    z_col = logger._column_index("z_m")
    weight_col = logger._column_index("represented_real_ions")
    color_col = logger._column_index("kinetic_energy_ev")
    sample = np.arange(points.shape[0], dtype=np.int64)
    if sample.size > logger.plot_max_points:
        sample = np.linspace(
            0,
            sample.size - 1,
            logger.plot_max_points,
            dtype=np.int64,
        )
    z_mm = points[:, z_col] * 1.0e3
    r_mm = points[:, r_col] * 1.0e3
    weights = points[:, weight_col]
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    colors, label = _color_values(points[:, color_col], r_mm)
    return SnapshotPlotData(
        points=points,
        sample=sample,
        z_mm=z_mm,
        r_mm=r_mm,
        weights=weights,
        colors=colors,
        color_label=label,
        z_col=z_col,
        x_col=logger._column_index("x_m"),
        y_col=logger._column_index("y_m"),
    )


def _color_values(
    kinetic_energy_ev: np.ndarray,
    radius_mm: np.ndarray,
) -> tuple[np.ndarray, str]:
    if np.any(np.isfinite(kinetic_energy_ev)):
        return kinetic_energy_ev, "KE [eV]"
    return radius_mm, "r [mm]"


def _draw_axial_histogram(axis: Any, data: SnapshotPlotData) -> None:
    finite_z = data.z_mm[np.isfinite(data.z_mm)]
    if finite_z.size:
        z_min = max(0.0, float(np.nanmin(finite_z)))
        z_max = max(z_min + 1.0, float(np.nanmax(finite_z)))
        bin_count = min(80, max(20, int(np.sqrt(finite_z.size))))
        axis.hist(
            data.z_mm,
            bins=np.linspace(z_min, z_max, bin_count),
            weights=data.weights,
            color="#4C78A8",
            alpha=0.82,
        )
    axis.set_xlabel("z [mm]")
    axis.set_ylabel("represented active real ions / bin")
    axis.set_title("Final active ion axial distribution")
    axis.grid(True, alpha=0.22)


def _write_rz_plot(
    logger: Any,
    plt: Any,
    record: Any,
    data: SnapshotPlotData,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=160)
    scatter = axes[0].scatter(
        data.z_mm[data.sample],
        data.r_mm[data.sample],
        c=data.colors[data.sample],
        s=3,
        alpha=0.55,
        cmap="viridis",
        linewidths=0,
    )
    axes[0].set_xlabel("z [mm]")
    axes[0].set_ylabel("r [mm]")
    axes[0].set_title(
        f"Final active snapshot, t={record.time_s * 1.0e3:.3g} ms"
    )
    axes[0].grid(True, alpha=0.22)
    fig.colorbar(scatter, ax=axes[0]).set_label(data.color_label)
    _draw_axial_histogram(axes[1], data)
    fig.tight_layout()
    path = logger.output_dir / "ion_spatial_distribution_rz.png"
    fig.savefig(path)
    plt.close(fig)
    logger.generated_artifacts.append(path)


def _write_xyz_plot(
    logger: Any,
    plt: Any,
    data: SnapshotPlotData,
) -> None:
    coordinates = data.points[:, [data.x_col, data.y_col, data.z_col]]
    if not np.all(np.isfinite(coordinates)):
        return
    fig = plt.figure(figsize=(8.0, 6.2), dpi=160)
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        data.points[data.sample, data.z_col] * 1.0e3,
        data.points[data.sample, data.x_col] * 1.0e3,
        data.points[data.sample, data.y_col] * 1.0e3,
        c=data.colors[data.sample],
        s=2,
        alpha=0.32,
        cmap="viridis",
        linewidths=0,
    )
    axis.set_xlabel("z [mm]")
    axis.set_ylabel("x [mm]")
    axis.set_zlabel("y [mm]")
    axis.set_title("Final Active Ion Snapshot (3D)")
    fig.colorbar(scatter, ax=axis, pad=0.12, shrink=0.75).set_label(
        data.color_label
    )
    fig.tight_layout()
    path = logger.output_dir / "ion_spatial_distribution_xyz.png"
    fig.savefig(path)
    plt.close(fig)
    logger.generated_artifacts.append(path)


def write_spatial_plots(logger: Any) -> None:
    """Render final RZ/XYZ distributions from a snapshot logger."""

    if not logger.make_plots or not logger.records:
        return
    plt = _load_pyplot(logger.output_dir / "snapshot_plot_error.txt")
    if plt is None:
        return
    record, snapshot = _final_snapshot(logger)
    if record is None or snapshot is None or snapshot.size == 0:
        return
    data = _snapshot_plot_data(logger, snapshot)
    _write_rz_plot(logger, plt, record, data)
    _write_xyz_plot(logger, plt, data)


def _read_tracks(path: Path) -> dict[int, list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tracks: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            track_id = int(float(row["track_id"]))
        except (KeyError, ValueError):
            continue
        tracks.setdefault(track_id, []).append(row)
    return tracks


def _write_trajectory_figure(
    logger: Any,
    plt: Any,
    tracks: dict[int, list[dict[str, str]]],
) -> None:
    selected = sorted(tracks)[
        : min(logger.trajectory_plot_max_tracks, len(tracks))
    ]
    fig, axis = plt.subplots(figsize=(8.5, 5.8), dpi=160)
    for track_id in selected:
        rows = sorted(
            tracks[track_id],
            key=lambda item: float(item["time_s"]),
        )
        z_mm = [float(item["z_m"]) * 1.0e3 for item in rows]
        r_mm = [float(item["r_m"]) * 1.0e3 for item in rows]
        axis.plot(z_mm, r_mm, linewidth=0.8, alpha=0.55)
    axis.set_xlabel("z [mm]")
    axis.set_ylabel("r [mm]")
    axis.set_title(f"Representative Ion Trajectories ({len(selected)} tracks)")
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    path = logger.output_dir / "representative_trajectories_rz.png"
    fig.savefig(path)
    plt.close(fig)
    logger.generated_artifacts.append(path)


def write_trajectory_plots(logger: Any) -> None:
    """Render sampled trajectories from their stable CSV artifact."""

    path = logger.output_dir / "representative_trajectories.csv"
    if (
        not logger.make_plots
        or logger.trajectory_plot_max_tracks <= 0
        or not path.exists()
    ):
        return
    plt = _load_pyplot(logger.output_dir / "trajectory_plot_error.txt")
    if plt is None:
        return
    tracks = _read_tracks(path)
    if tracks:
        _write_trajectory_figure(logger, plt, tracks)
