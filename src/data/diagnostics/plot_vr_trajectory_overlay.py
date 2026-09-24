"""Plot local gas radial velocity contours with representative ion tracks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize, TwoSlopeNorm

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from .plot_results import load_snapshots
    from .plot_style import add_subplot_labels, apply_paper_style, scaled
except ImportError:  # pragma: no cover - supports direct script execution.
    from plot_results import load_snapshots  # type: ignore
    from plot_style import add_subplot_labels, apply_paper_style, scaled  # type: ignore


DEFAULT_FIELD_PATH = (
    Path("outputs") / "slens100x_bake_0p01_zmax65_fluent_interior_clipped_capexit4p5" / "baked_fields.npy"
)
DEFAULT_COLUMNS = (
    "r_m",
    "z_m",
    "teff_k",
    "kinetic_energy_j",
    "x_m",
    "y_m",
    "vx_m_per_s",
    "vy_m_per_s",
    "vz_m_per_s",
    "speed_m_per_s",
    "kinetic_energy_ev",
    "represented_real_ions",
    "parent_real_ions_remaining",
    "fragmented_real_ions_represented",
    "track_id",
    "slot_id",
    "status_code",
    "collision_count_per_ion",
)


@dataclass(frozen=True)
class Trajectory:
    z_mm: np.ndarray
    r_mm: np.ndarray
    time_ms: np.ndarray
    vz_m_per_s: np.ndarray
    vr_m_per_s: np.ndarray


@dataclass(frozen=True)
class SnapshotPointCloud:
    z_mm: np.ndarray
    r_mm: np.ndarray
    vz_m_per_s: np.ndarray
    vr_m_per_s: np.ndarray
    color_values: np.ndarray | None
    color_label: str
    macro_steps: list[int]
    time_ms: list[float]
    total_local_points: int

    @property
    def plotted_count(self) -> int:
        return int(self.z_mm.size)


@dataclass(frozen=True)
class SnapshotRecord:
    macro_step: int
    time_s: float
    path: Path


def _load_baked_field(path: Path) -> dict[str, object]:
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a baked field dictionary.")
    for group in ("grid", "fluent"):
        if group not in payload:
            raise KeyError(f"{path} is missing required group {group!r}.")
    return payload


def _window_indices(coords_mm: np.ndarray, lower_mm: float, upper_mm: float, axis_name: str) -> np.ndarray:
    if upper_mm <= lower_mm:
        raise ValueError(f"{axis_name} max must be greater than min.")
    indices = np.flatnonzero((coords_mm >= lower_mm) & (coords_mm <= upper_mm))
    if indices.size == 0:
        raise ValueError(f"No {axis_name} grid points in [{lower_mm:g}, {upper_mm:g}] mm.")
    return indices


def _selected_ids(track_ids: Iterable[int], max_tracks: int) -> set[int]:
    unique = np.asarray(sorted(set(int(value) for value in track_ids)), dtype=np.int64)
    if unique.size == 0:
        return set()
    if max_tracks <= 0 or unique.size <= max_tracks:
        return {int(value) for value in unique.tolist()}
    offsets = np.linspace(0, unique.size - 1, max_tracks, dtype=np.int64)
    return {int(value) for value in unique[offsets].tolist()}


def _filled_velocity_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_ms = np.asarray(values[:, 0], dtype=np.float64)
    z_mm = np.asarray(values[:, 1], dtype=np.float64)
    r_mm = np.asarray(values[:, 2], dtype=np.float64)
    vz_m_per_s = np.asarray(values[:, 3], dtype=np.float64)
    vr_m_per_s = np.asarray(values[:, 4], dtype=np.float64)
    missing = ~np.isfinite(vz_m_per_s) | ~np.isfinite(vr_m_per_s)
    if np.any(missing) and values.shape[0] >= 2:
        time_s = time_ms * 1.0e-3
        if np.nanmax(time_s) > np.nanmin(time_s):
            vz_est = np.gradient(z_mm * 1.0e-3, time_s)
            vr_est = np.gradient(r_mm * 1.0e-3, time_s)
            vz_m_per_s = np.where(np.isfinite(vz_m_per_s), vz_m_per_s, vz_est)
            vr_m_per_s = np.where(np.isfinite(vr_m_per_s), vr_m_per_s, vr_est)
    return z_mm, r_mm, time_ms, vz_m_per_s, vr_m_per_s


def _row_radial_velocity(row: dict[str, str]) -> tuple[float, float]:
    required = ("x_m", "y_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s")
    if any(name not in row for name in required):
        return float("nan"), float("nan")
    try:
        x_m = float(row["x_m"])
        y_m = float(row["y_m"])
        vx_m_per_s = float(row["vx_m_per_s"])
        vy_m_per_s = float(row["vy_m_per_s"])
        vz_m_per_s = float(row["vz_m_per_s"])
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    r_m = float(np.hypot(x_m, y_m))
    if r_m <= 1.0e-16:
        return vz_m_per_s, 0.0
    return vz_m_per_s, float((x_m * vx_m_per_s + y_m * vy_m_per_s) / r_m)


def _load_trajectories_csv(path: Path, max_tracks: int) -> dict[int, Trajectory]:
    rows_by_track: dict[int, list[tuple[float, float, float, float, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time_s", "r_m", "z_m", "track_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing trajectory columns: {sorted(missing)}")
        for row in reader:
            try:
                track_id = int(float(row["track_id"]))
                time_ms = float(row["time_s"]) * 1.0e3
                r_mm = float(row["r_m"]) * 1.0e3
                z_mm = float(row["z_m"]) * 1.0e3
                vz_m_per_s, vr_m_per_s = _row_radial_velocity(row)
            except (TypeError, ValueError):
                continue
            if track_id < 0 or not (np.isfinite(time_ms) and np.isfinite(r_mm) and np.isfinite(z_mm)):
                continue
            rows_by_track.setdefault(track_id, []).append((time_ms, z_mm, r_mm, vz_m_per_s, vr_m_per_s))

    selected = _selected_ids(rows_by_track.keys(), max_tracks)
    trajectories: dict[int, Trajectory] = {}
    for track_id in sorted(selected):
        rows = sorted(rows_by_track[track_id], key=lambda item: item[0])
        if len(rows) < 2:
            continue
        values = np.asarray(rows, dtype=np.float64)
        z_mm, r_mm, time_ms, vz_m_per_s, vr_m_per_s = _filled_velocity_arrays(values)
        trajectories[int(track_id)] = Trajectory(z_mm, r_mm, time_ms, vz_m_per_s, vr_m_per_s)
    return trajectories


def _snapshot_radial_velocity(row: np.ndarray, columns: list[str]) -> tuple[float, float]:
    required = ("x_m", "y_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s")
    if any(name not in columns for name in required):
        return float("nan"), float("nan")
    x_m = float(row[columns.index("x_m")])
    y_m = float(row[columns.index("y_m")])
    vx_m_per_s = float(row[columns.index("vx_m_per_s")])
    vy_m_per_s = float(row[columns.index("vy_m_per_s")])
    vz_m_per_s = float(row[columns.index("vz_m_per_s")])
    if not all(np.isfinite(value) for value in (x_m, y_m, vx_m_per_s, vy_m_per_s, vz_m_per_s)):
        return float("nan"), float("nan")
    r_m = float(np.hypot(x_m, y_m))
    if r_m <= 1.0e-16:
        return vz_m_per_s, 0.0
    return vz_m_per_s, float((x_m * vx_m_per_s + y_m * vy_m_per_s) / r_m)


def _load_trajectories_from_snapshots(case_dir: Path, max_tracks: int) -> dict[int, Trajectory]:
    snapshots, metadata = load_snapshots(case_dir)
    columns = list(metadata.get("columns", DEFAULT_COLUMNS))
    for required in ("r_m", "z_m", "track_id"):
        if required not in columns:
            raise ValueError(f"{case_dir} snapshot metadata is missing column {required!r}.")
    r_col = columns.index("r_m")
    z_col = columns.index("z_m")
    track_col = columns.index("track_id")

    all_track_ids: list[int] = []
    for snapshot in snapshots:
        if snapshot.data.size == 0:
            continue
        track_values = snapshot.data[:, track_col]
        valid = track_values[np.isfinite(track_values) & (track_values >= 0)].astype(np.int64)
        all_track_ids.extend(int(value) for value in valid.tolist())
    selected = _selected_ids(all_track_ids, max_tracks)
    if not selected:
        return {}

    rows_by_track: dict[int, list[tuple[float, float, float, float, float]]] = {track_id: [] for track_id in selected}
    for snapshot in snapshots:
        if snapshot.data.size == 0:
            continue
        track_values = snapshot.data[:, track_col]
        valid_track_mask = np.isfinite(track_values) & (track_values >= 0)
        if not np.any(valid_track_mask):
            continue
        valid_track_ids = track_values[valid_track_mask].astype(np.int64)
        valid_rows = snapshot.data[valid_track_mask]
        for track_id in selected:
            mask = valid_track_ids == track_id
            if not np.any(mask):
                continue
            # One row per track per snapshot is expected; average if duplicates exist.
            z_mm = float(np.nanmean(valid_rows[mask, z_col]) * 1.0e3)
            r_mm = float(np.nanmean(valid_rows[mask, r_col]) * 1.0e3)
            vz_values: list[float] = []
            vr_values: list[float] = []
            for row in valid_rows[mask]:
                vz_value, vr_value = _snapshot_radial_velocity(row, columns)
                vz_values.append(vz_value)
                vr_values.append(vr_value)
            vz_m_per_s = float(np.nanmean(vz_values)) if np.any(np.isfinite(vz_values)) else float("nan")
            vr_m_per_s = float(np.nanmean(vr_values)) if np.any(np.isfinite(vr_values)) else float("nan")
            rows_by_track[track_id].append((snapshot.time_s * 1.0e3, z_mm, r_mm, vz_m_per_s, vr_m_per_s))

    trajectories: dict[int, Trajectory] = {}
    for track_id, rows in rows_by_track.items():
        rows = sorted(rows, key=lambda item: item[0])
        if len(rows) < 2:
            continue
        values = np.asarray(rows, dtype=np.float64)
        z_mm, r_mm, time_ms, vz_m_per_s, vr_m_per_s = _filled_velocity_arrays(values)
        trajectories[int(track_id)] = Trajectory(z_mm, r_mm, time_ms, vz_m_per_s, vr_m_per_s)
    return trajectories


def _resolve_trajectories(args: argparse.Namespace) -> dict[int, Trajectory]:
    if int(args.max_trajectories) <= 0:
        return {}
    trajectory_csv = Path(args.trajectory_csv) if args.trajectory_csv else None
    case_dir = Path(args.case_dir) if args.case_dir else None
    if trajectory_csv is None and case_dir is not None:
        candidate = case_dir / "representative_trajectories.csv"
        if candidate.exists():
            trajectory_csv = candidate
    if trajectory_csv is not None:
        return _load_trajectories_csv(trajectory_csv, args.max_trajectories)
    if case_dir is not None:
        return _load_trajectories_from_snapshots(case_dir, args.max_trajectories)
    return {}


def _velocity_norm(values: np.ndarray, vmin: float | None, vmax: float | None) -> Normalize:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=-1.0, vmax=1.0)
    lower = float(np.nanpercentile(finite, 1.0)) if vmin is None else float(vmin)
    upper = float(np.nanpercentile(finite, 99.0)) if vmax is None else float(vmax)
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    if lower < 0.0 < upper:
        return TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
    return Normalize(vmin=lower, vmax=upper)


def _metadata_columns(case_dir: Path) -> list[str]:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        return list(DEFAULT_COLUMNS)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return list(metadata.get("columns", DEFAULT_COLUMNS))


def _snapshot_records(case_dir: Path, fallback_dt_s: float) -> list[SnapshotRecord]:
    records: list[SnapshotRecord] = []
    index_path = case_dir / "snapshots_index.csv"
    if index_path.exists():
        with index_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                storage_ref = row.get("storage_ref", "")
                if not storage_ref.lower().endswith(".npy"):
                    continue
                records.append(
                    SnapshotRecord(
                        macro_step=int(row["macro_step"]),
                        time_s=float(row["time_s"]),
                        path=case_dir / storage_ref,
                    )
                )
    if records:
        return sorted(records, key=lambda item: item.macro_step)

    for path in case_dir.glob("snapshot_macro_*.npy"):
        match = re.match(r"snapshot_macro_(\d+)\.npy$", path.name)
        if match is None:
            continue
        macro_step = int(match.group(1))
        records.append(SnapshotRecord(macro_step=macro_step, time_s=macro_step * fallback_dt_s, path=path))
    return sorted(records, key=lambda item: item.macro_step)


def _selected_snapshot_records(args: argparse.Namespace) -> list[SnapshotRecord]:
    snapshot_file = Path(args.snapshot_file) if args.snapshot_file else None
    if snapshot_file is not None:
        match = re.match(r"snapshot_macro_(\d+)\.npy$", snapshot_file.name)
        macro_step = int(match.group(1)) if match is not None else -1
        return [SnapshotRecord(macro_step=macro_step, time_s=macro_step * float(args.snapshot_time_step_s), path=snapshot_file)]

    case_dir = Path(args.case_dir) if args.case_dir else None
    if case_dir is None:
        raise ValueError("--show-snapshot-points requires --case-dir or --snapshot-file.")
    records = _snapshot_records(case_dir, float(args.snapshot_time_step_s))
    if not records:
        raise FileNotFoundError(f"No snapshot_macro_*.npy files found in {case_dir}.")
    if args.snapshot_point_mode == "all":
        return records

    if args.snapshot_macro is not None:
        target_macro = int(args.snapshot_macro)
        return [min(records, key=lambda item: abs(item.macro_step - target_macro))]
    if args.snapshot_time_ms is not None:
        target_time_s = float(args.snapshot_time_ms) * 1.0e-3
        return [min(records, key=lambda item: abs(item.time_s - target_time_s))]
    return [records[-1]]


def _snapshot_color_values(data: np.ndarray, columns: list[str], color_by: str) -> tuple[np.ndarray | None, str]:
    if color_by.lower() == "none":
        return None, ""
    if color_by in columns:
        return data[:, columns.index(color_by)], color_by
    if color_by in {"vr_m_per_s", "radial_velocity_m_per_s"}:
        values = np.asarray([_snapshot_radial_velocity(row, columns)[1] for row in data], dtype=np.float64)
        return values, "ion radial velocity [m/s]"
    if color_by in {"vz_m_per_s", "axial_velocity_m_per_s"}:
        if "vz_m_per_s" not in columns:
            raise ValueError("Snapshot metadata does not include vz_m_per_s.")
        return data[:, columns.index("vz_m_per_s")], "ion axial velocity [m/s]"
    raise ValueError(f"Unknown --snapshot-point-color-by value {color_by!r}.")


def _load_snapshot_point_cloud(args: argparse.Namespace) -> SnapshotPointCloud | None:
    if not bool(args.show_snapshot_points):
        return None
    case_dir = Path(args.case_dir) if args.case_dir else Path(args.snapshot_file).parent
    columns = _metadata_columns(case_dir)
    for required in ("r_m", "z_m"):
        if required not in columns:
            raise ValueError(f"Snapshot metadata is missing column {required!r}.")
    r_col = columns.index("r_m")
    z_col = columns.index("z_m")

    records = _selected_snapshot_records(args)
    z_blocks: list[np.ndarray] = []
    r_blocks: list[np.ndarray] = []
    vz_blocks: list[np.ndarray] = []
    vr_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    color_label = ""
    macro_steps: list[int] = []
    time_ms: list[float] = []
    total_local_points = 0

    for record in records:
        data = np.asarray(np.load(record.path), dtype=np.float64)
        if data.size == 0:
            continue
        z_mm = data[:, z_col] * 1.0e3
        r_mm = data[:, r_col] * 1.0e3
        finite = np.isfinite(z_mm) & np.isfinite(r_mm)
        in_window = (
            finite
            & (z_mm >= float(args.z_min_mm))
            & (z_mm <= float(args.z_max_mm))
            & (r_mm >= float(args.r_min_mm))
            & (r_mm <= float(args.r_max_mm))
        )
        if not np.any(in_window):
            continue
        local = data[in_window]
        local_z_mm = z_mm[in_window]
        local_r_mm = r_mm[in_window]
        vz_values = np.asarray([_snapshot_radial_velocity(row, columns)[0] for row in local], dtype=np.float64)
        vr_values = np.asarray([_snapshot_radial_velocity(row, columns)[1] for row in local], dtype=np.float64)
        color_values, color_label = _snapshot_color_values(local, columns, args.snapshot_point_color_by)
        z_blocks.append(local_z_mm)
        r_blocks.append(local_r_mm)
        vz_blocks.append(vz_values)
        vr_blocks.append(vr_values)
        if color_values is not None:
            color_blocks.append(np.asarray(color_values, dtype=np.float64))
        total_local_points += int(local_z_mm.size)
        macro_steps.append(int(record.macro_step))
        time_ms.append(float(record.time_s * 1.0e3))

    if not z_blocks:
        return SnapshotPointCloud(
            z_mm=np.empty(0),
            r_mm=np.empty(0),
            vz_m_per_s=np.empty(0),
            vr_m_per_s=np.empty(0),
            color_values=None,
            color_label=color_label,
            macro_steps=[int(record.macro_step) for record in records],
            time_ms=[float(record.time_s * 1.0e3) for record in records],
            total_local_points=0,
        )

    z_all = np.concatenate(z_blocks)
    r_all = np.concatenate(r_blocks)
    vz_all = np.concatenate(vz_blocks)
    vr_all = np.concatenate(vr_blocks)
    color_all = np.concatenate(color_blocks) if color_blocks else None

    max_points = max(1, int(args.snapshot_point_max))
    if z_all.size > max_points:
        offsets = np.linspace(0, z_all.size - 1, max_points, dtype=np.int64)
        z_all = z_all[offsets]
        r_all = r_all[offsets]
        vz_all = vz_all[offsets]
        vr_all = vr_all[offsets]
        if color_all is not None:
            color_all = color_all[offsets]

    return SnapshotPointCloud(
        z_mm=z_all,
        r_mm=r_all,
        vz_m_per_s=vz_all,
        vr_m_per_s=vr_all,
        color_values=color_all,
        color_label=color_label,
        macro_steps=macro_steps,
        time_ms=time_ms,
        total_local_points=total_local_points,
    )


def _draw_gas_streamlines(ax, z_mm: np.ndarray, r_mm: np.ndarray, vz: np.ndarray, vr: np.ndarray, args: argparse.Namespace) -> None:
    try:
        ax.streamplot(
            z_mm,
            r_mm,
            vz,
            vr,
            density=float(args.gas_streamline_density),
            color=args.gas_streamline_color,
            linewidth=scaled(args.gas_streamline_linewidth, args.paper_line_scale),
            arrowsize=float(args.gas_streamline_arrowsize),
            minlength=0.03,
            zorder=4,
        )
    except ValueError as exc:
        print(f"Skipped gas streamlines: {exc}")


def _draw_snapshot_points(ax, point_cloud: SnapshotPointCloud, args: argparse.Namespace) -> None:
    if point_cloud.plotted_count <= 0:
        return
    if point_cloud.color_values is None:
        ax.scatter(
            point_cloud.z_mm,
            point_cloud.r_mm,
            s=float(args.snapshot_point_size),
            c=args.snapshot_point_color,
            alpha=float(args.snapshot_point_alpha),
            linewidths=0.0,
            zorder=5,
        )
        return
    norm = None
    if args.snapshot_point_vmin is not None or args.snapshot_point_vmax is not None:
        finite = point_cloud.color_values[np.isfinite(point_cloud.color_values)]
        vmin = float(args.snapshot_point_vmin) if args.snapshot_point_vmin is not None else float(np.nanmin(finite))
        vmax = float(args.snapshot_point_vmax) if args.snapshot_point_vmax is not None else float(np.nanmax(finite))
        norm = Normalize(vmin=vmin, vmax=vmax)
    ax.scatter(
        point_cloud.z_mm,
        point_cloud.r_mm,
        s=float(args.snapshot_point_size),
        c=point_cloud.color_values,
        cmap=args.snapshot_point_cmap,
        norm=norm,
        alpha=float(args.snapshot_point_alpha),
        linewidths=0.0,
        zorder=5,
    )


def _draw_ion_velocity_arrows(ax, trajectories: dict[int, Trajectory], args: argparse.Namespace) -> int:
    points: list[tuple[float, float, float, float]] = []
    stride = max(1, int(args.ion_arrow_stride))
    for trajectory in trajectories.values():
        for index in range(0, trajectory.z_mm.size, stride):
            z_mm = float(trajectory.z_mm[index])
            r_mm = float(trajectory.r_mm[index])
            vz_m_per_s = float(trajectory.vz_m_per_s[index])
            vr_m_per_s = float(trajectory.vr_m_per_s[index])
            if not all(np.isfinite(value) for value in (z_mm, r_mm, vz_m_per_s, vr_m_per_s)):
                continue
            if not (args.z_min_mm <= z_mm <= args.z_max_mm and args.r_min_mm <= r_mm <= args.r_max_mm):
                continue
            speed = float(np.hypot(vz_m_per_s, vr_m_per_s))
            if speed <= 1.0e-30:
                continue
            points.append((z_mm, r_mm, vz_m_per_s / speed, vr_m_per_s / speed))
    if not points:
        return 0
    max_arrows = max(1, int(args.max_ion_arrows))
    if len(points) > max_arrows:
        offsets = np.linspace(0, len(points) - 1, max_arrows, dtype=np.int64)
        points = [points[int(index)] for index in offsets]
    values = np.asarray(points, dtype=np.float64)
    length_mm = float(args.ion_arrow_length_mm)
    ax.quiver(
        values[:, 0],
        values[:, 1],
        values[:, 2] * length_mm,
        values[:, 3] * length_mm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=args.ion_arrow_color,
        alpha=float(args.ion_arrow_alpha),
        width=scaled(args.ion_arrow_width, args.paper_line_scale),
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.5,
        zorder=6,
    )
    return int(values.shape[0])


def _draw_snapshot_velocity_arrows(ax, point_cloud: SnapshotPointCloud | None, args: argparse.Namespace) -> int:
    if point_cloud is None or point_cloud.plotted_count <= 0:
        return 0
    values = np.column_stack((point_cloud.z_mm, point_cloud.r_mm, point_cloud.vz_m_per_s, point_cloud.vr_m_per_s))
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    if values.size == 0:
        return 0
    speeds = np.hypot(values[:, 2], values[:, 3])
    moving = speeds > 1.0e-30
    values = values[moving]
    speeds = speeds[moving]
    if values.size == 0:
        return 0
    stride = max(1, int(args.ion_arrow_stride))
    values = values[::stride]
    speeds = speeds[::stride]
    max_arrows = max(1, int(args.max_ion_arrows))
    if values.shape[0] > max_arrows:
        offsets = np.linspace(0, values.shape[0] - 1, max_arrows, dtype=np.int64)
        values = values[offsets]
        speeds = speeds[offsets]
    length_mm = float(args.ion_arrow_length_mm)
    ax.quiver(
        values[:, 0],
        values[:, 1],
        values[:, 2] / speeds * length_mm,
        values[:, 3] / speeds * length_mm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=args.ion_arrow_color,
        alpha=float(args.ion_arrow_alpha),
        width=scaled(args.ion_arrow_width, args.paper_line_scale),
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.5,
        zorder=6,
    )
    return int(values.shape[0])


def _parse_panels(panel_text: str) -> list[str]:
    presets = {
        "paper": ["streamlines", "ion-velocity"],
        "comparison": ["streamlines", "ion-velocity"],
        "all": ["gas", "snapshot", "streamlines", "trajectories", "ion-velocity"],
    }
    aliases = {
        "combined": "combined",
        "legacy": "combined",
        "gas": "gas",
        "vr": "gas",
        "v_r": "gas",
        "snapshot": "snapshot",
        "points": "snapshot",
        "ions": "snapshot",
        "streamline": "streamlines",
        "streamlines": "streamlines",
        "gas-streamlines": "streamlines",
        "trajectory": "trajectories",
        "trajectories": "trajectories",
        "tracks": "trajectories",
        "ion-velocity": "ion-velocity",
        "velocity": "ion-velocity",
        "arrows": "ion-velocity",
        "ion-arrows": "ion-velocity",
    }
    raw = str(panel_text).strip().lower()
    if not raw:
        return ["combined"]
    if raw in presets:
        return list(presets[raw])

    panels: list[str] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key in presets:
            for preset_panel in presets[key]:
                if preset_panel not in panels:
                    panels.append(preset_panel)
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
        return int(math.ceil(panel_count / 3.0)), 3
    parts = text.split("x", 1)
    if len(parts) != 2:
        raise ValueError("--layout must be auto or rowsxcols, for example 1x1, 1x2, 2x2.")
    rows = int(parts[0])
    cols = int(parts[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("--layout rows and columns must be positive.")
    if rows * cols < panel_count:
        raise ValueError(f"--layout {layout!r} has {rows * cols} slots for {panel_count} panels.")
    return rows, cols


def _figure_size(rows: int, cols: int) -> tuple[float, float]:
    if rows == 1 and cols == 1:
        return 9.6, 6.2
    return max(7.2, cols * 7.0), max(5.4, rows * 5.2)


def _panel_title(panel: str, args: argparse.Namespace, point_cloud: SnapshotPointCloud | None) -> str:
    titles = {
        "combined": "v_r contour with selected overlays",
        "gas": "gas radial velocity v_r(r,z)",
        "snapshot": "local ion snapshot points",
        "streamlines": "gas streamlines over v_r + local ion snapshots",
        "trajectories": "representative ion trajectories over v_r",
        "ion-velocity": "local snapshot points and ion velocity arrows",
    }
    title = titles[panel]
    if panel in {"snapshot", "streamlines", "ion-velocity"} and point_cloud is not None and point_cloud.time_ms:
        if len(point_cloud.time_ms) == 1:
            title += f"\nt={point_cloud.time_ms[0]:.2f} ms, points={point_cloud.total_local_points}"
        else:
            title += f"\n{len(point_cloud.time_ms)} snapshots, local points={point_cloud.total_local_points}"
    return title


def _draw_trajectory_lines(ax, trajectories: dict[int, Trajectory], args: argparse.Namespace) -> int:
    drawn = 0
    for trajectory in trajectories.values():
        z_track_mm = trajectory.z_mm
        r_track_mm = trajectory.r_mm
        inside = (
            (z_track_mm >= args.z_min_mm)
            & (z_track_mm <= args.z_max_mm)
            & (r_track_mm >= args.r_min_mm)
            & (r_track_mm <= args.r_max_mm)
        )
        if int(np.count_nonzero(inside)) < 2:
            continue
        (line,) = ax.plot(
            z_track_mm[inside],
            r_track_mm[inside],
            color=args.trajectory_color,
            linewidth=scaled(args.trajectory_linewidth, args.paper_line_scale),
            alpha=float(args.trajectory_alpha),
            zorder=5,
        )
        if args.trajectory_outline_width > 0.0:
            line.set_path_effects(
                [
                    path_effects.Stroke(
                        linewidth=scaled(args.trajectory_linewidth + args.trajectory_outline_width, args.paper_line_scale),
                        foreground=args.trajectory_outline_color,
                    ),
                    path_effects.Normal(),
                ]
            )
        drawn += 1
    return drawn


def _draw_vr_background(
    ax,
    zg: np.ndarray,
    rg: np.ndarray,
    vr: np.ndarray,
    norm: Normalize,
    args: argparse.Namespace,
):
    contour = ax.contourf(zg, rg, vr, levels=int(args.levels), cmap=args.cmap, norm=norm)
    ax.contour(zg, rg, vr, levels=[0.0], colors="black", linewidths=scaled(0.9, args.paper_line_scale), alpha=0.78)
    return contour


def _decorate_vr_axis(ax, grid: dict[str, object], args: argparse.Namespace, title: str) -> None:
    capillary_exit_m = grid.get("capillary_exit_z_m")
    if capillary_exit_m is not None:
        ax.axvline(
            float(capillary_exit_m) * 1.0e3,
            color="white",
            linestyle="--",
            linewidth=scaled(1.15, args.paper_line_scale),
            alpha=0.9,
        )
    for reference_z_mm in args.reference_z_mm:
        ax.axvline(float(reference_z_mm), color="cyan", linestyle=":", linewidth=scaled(1.15, args.paper_line_scale), alpha=0.9)
    ax.set_xlim(args.z_min_mm, args.z_max_mm)
    ax.set_ylim(args.r_min_mm, args.r_max_mm)
    ax.set_title(title)
    ax.grid(True, color="white", linewidth=scaled(0.5, args.paper_line_scale), alpha=0.2)


def _draw_vr_panel(
    ax,
    panel: str,
    *,
    zg: np.ndarray,
    rg: np.ndarray,
    z_window_mm: np.ndarray,
    r_window_mm: np.ndarray,
    vr: np.ndarray,
    vz: np.ndarray,
    norm: Normalize,
    trajectories: dict[int, Trajectory],
    snapshot_points: SnapshotPointCloud | None,
    grid: dict[str, object],
    args: argparse.Namespace,
) -> tuple[object, int, int]:
    contour = _draw_vr_background(ax, zg, rg, vr, norm, args)
    trajectory_count = 0
    ion_arrow_count = 0

    if panel == "combined":
        if bool(args.show_gas_streamlines):
            _draw_gas_streamlines(ax, z_window_mm, r_window_mm, vz, vr, args)
        if snapshot_points is not None:
            _draw_snapshot_points(ax, snapshot_points, args)
        trajectory_count = _draw_trajectory_lines(ax, trajectories, args)
        if bool(args.show_ion_velocity_arrows):
            if args.ion_arrow_source == "snapshot":
                ion_arrow_count = _draw_snapshot_velocity_arrows(ax, snapshot_points, args)
            elif args.ion_arrow_source == "auto" and snapshot_points is not None and snapshot_points.plotted_count > 0:
                ion_arrow_count = _draw_snapshot_velocity_arrows(ax, snapshot_points, args)
            else:
                ion_arrow_count = _draw_ion_velocity_arrows(ax, trajectories, args)
    elif panel == "gas":
        pass
    elif panel == "snapshot":
        if snapshot_points is not None:
            _draw_snapshot_points(ax, snapshot_points, args)
    elif panel == "streamlines":
        _draw_gas_streamlines(ax, z_window_mm, r_window_mm, vz, vr, args)
        if snapshot_points is not None:
            _draw_snapshot_points(ax, snapshot_points, args)
    elif panel == "trajectories":
        if snapshot_points is not None:
            _draw_snapshot_points(ax, snapshot_points, args)
        trajectory_count = _draw_trajectory_lines(ax, trajectories, args)
    elif panel == "ion-velocity":
        if snapshot_points is not None:
            _draw_snapshot_points(ax, snapshot_points, args)
        if args.ion_arrow_source in {"snapshot", "auto"} and snapshot_points is not None and snapshot_points.plotted_count > 0:
            ion_arrow_count = _draw_snapshot_velocity_arrows(ax, snapshot_points, args)
        else:
            ion_arrow_count = _draw_ion_velocity_arrows(ax, trajectories, args)
    else:  # pragma: no cover - protected by _parse_panels.
        raise ValueError(f"Unknown panel {panel!r}")

    _decorate_vr_axis(ax, grid, args, _panel_title(panel, args, snapshot_points))
    return contour, trajectory_count, ion_arrow_count


def make_plot(args: argparse.Namespace) -> Path:
    apply_paper_style(font_scale=args.paper_font_scale, line_scale=args.paper_line_scale)
    field_path = Path(args.field)
    baked = _load_baked_field(field_path)
    grid = baked["grid"]
    fluent = baked["fluent"]
    if not isinstance(grid, dict) or not isinstance(fluent, dict):
        raise TypeError("baked field grid/fluent groups must be dictionaries.")

    z_mm = np.asarray(grid["z_coords_m"], dtype=np.float64) * 1.0e3
    r_mm = np.asarray(grid["r_coords_m"], dtype=np.float64) * 1.0e3
    z_idx = _window_indices(z_mm, args.z_min_mm, args.z_max_mm, "z")
    r_idx = _window_indices(r_mm, args.r_min_mm, args.r_max_mm, "r")

    vr = np.asarray(fluent["v_r_m_per_s"], dtype=np.float64)[np.ix_(r_idx, z_idx)]
    vz = np.asarray(fluent["v_z_m_per_s"], dtype=np.float64)[np.ix_(r_idx, z_idx)]
    zg, rg = np.meshgrid(z_mm[z_idx], r_mm[r_idx])
    trajectories = _resolve_trajectories(args)
    snapshot_points = _load_snapshot_point_cloud(args)
    panels = _parse_panels(args.panels)
    nrows, ncols = _layout_shape(args.layout, len(panels))

    output = Path(args.output) if args.output else Path(args.output_dir) / "vr_contour_trajectory_overlay.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=_figure_size(nrows, ncols),
        dpi=args.dpi,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    norm = _velocity_norm(vr, args.vmin, args.vmax)
    contours: list[object] = []
    trajectory_counts: list[int] = []
    ion_arrow_counts: list[int] = []
    for ax, panel in zip(axes_flat, panels):
        contour, trajectory_count, ion_arrow_count = _draw_vr_panel(
            ax,
            panel,
            zg=zg,
            rg=rg,
            z_window_mm=z_mm[z_idx],
            r_window_mm=r_mm[r_idx],
            vr=vr,
            vz=vz,
            norm=norm,
            trajectories=trajectories,
            snapshot_points=snapshot_points,
            grid=grid,
            args=args,
        )
        contours.append(contour)
        trajectory_counts.append(trajectory_count)
        ion_arrow_counts.append(ion_arrow_count)

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    axes_grid = np.asarray(axes).reshape(nrows, ncols)
    for ax in axes_grid[:, 0]:
        ax.set_ylabel("r [mm]")
    for ax in axes_grid[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("z [mm]")

    active_axes = axes_flat[: len(panels)]
    cbar = fig.colorbar(contours[0], ax=active_axes.tolist(), pad=0.015, shrink=0.92)
    cbar.set_label("gas radial velocity v_r [m/s]")
    if str(args.title).strip() and len(panels) == 1:
        active_axes[0].set_title(str(args.title).strip())
    add_subplot_labels(
        active_axes,
        enabled=bool(args.subplot_labels) and len(panels) > 1,
        font_scale=args.paper_font_scale,
        line_scale=args.paper_line_scale,
    )
    if str(args.title).strip() and len(panels) > 1:
        fig.suptitle(str(args.title).strip())
    fig.savefig(output)
    plt.close(fig)

    summary = {
        "field": str(field_path),
        "case_dir": args.case_dir,
        "trajectory_csv": args.trajectory_csv,
        "output": str(output),
        "panels": panels,
        "layout": str(args.layout),
        "layout_shape": [int(nrows), int(ncols)],
        "z_window_mm": [float(args.z_min_mm), float(args.z_max_mm)],
        "r_window_mm": [float(args.r_min_mm), float(args.r_max_mm)],
        "trajectory_count_loaded": int(len(trajectories)),
        "trajectory_count_drawn_by_panel": [int(value) for value in trajectory_counts],
        "gas_streamlines_shown": bool(args.show_gas_streamlines),
        "snapshot_points_shown": snapshot_points is not None,
        "snapshot_point_mode": args.snapshot_point_mode,
        "snapshot_point_count_plotted": int(snapshot_points.plotted_count) if snapshot_points is not None else 0,
        "snapshot_point_count_local_available": int(snapshot_points.total_local_points) if snapshot_points is not None else 0,
        "snapshot_point_macro_steps": snapshot_points.macro_steps if snapshot_points is not None else [],
        "snapshot_point_time_ms": snapshot_points.time_ms if snapshot_points is not None else [],
        "snapshot_point_color_by": args.snapshot_point_color_by,
        "ion_velocity_arrows_shown": bool(args.show_ion_velocity_arrows),
        "ion_arrow_source": args.ion_arrow_source,
        "ion_velocity_arrow_count_by_panel": [int(value) for value in ion_arrow_counts],
        "ion_velocity_arrow_count": int(max(ion_arrow_counts) if ion_arrow_counts else 0),
        "vr_min_m_per_s": float(np.nanmin(vr)),
        "vr_max_m_per_s": float(np.nanmax(vr)),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    print(f"Saved: {summary_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a local contour map of baked Fluent radial gas velocity v_r(r,z) "
            "and overlay representative ion trajectories in r-z."
        )
    )
    parser.add_argument("--field", default=str(DEFAULT_FIELD_PATH), help="baked_fields.npy path.")
    parser.add_argument("--case-dir", default="", help="Simulation output directory containing snapshots/trajectories.")
    parser.add_argument("--trajectory-csv", default="", help="Optional representative_trajectories.csv path.")
    parser.add_argument("--output-dir", default="outputs/paper_figures", help="Output directory when --output is omitted.")
    parser.add_argument("--output", default="", help="Output PNG path.")
    parser.add_argument(
        "--panels",
        default="combined",
        help=(
            "Comma-separated panels or preset. Presets: paper/comparison -> streamlines,ion-velocity; "
            "all -> gas,snapshot,streamlines,trajectories,ion-velocity. Panels: combined, gas, snapshot, "
            "streamlines, trajectories, ion-velocity."
        ),
    )
    parser.add_argument("--layout", default="auto", help="Grouped figure layout: auto or rowsxcols, for example 1x1, 1x2, 2x2.")
    parser.add_argument("--z-min-mm", type=float, default=4.0, help="Left edge of plotted z window [mm].")
    parser.add_argument("--z-max-mm", type=float, default=12.0, help="Right edge of plotted z window [mm].")
    parser.add_argument("--r-min-mm", type=float, default=0.0, help="Lower edge of plotted r window [mm].")
    parser.add_argument("--r-max-mm", type=float, default=8.0, help="Upper edge of plotted r window [mm].")
    parser.add_argument("--reference-z-mm", type=float, nargs="*", default=[5.0], help="Optional vertical reference lines [mm].")
    parser.add_argument("--max-trajectories", type=int, default=80, help="Maximum representative tracks to draw.")
    parser.add_argument("--trajectory-alpha", type=float, default=0.55, help="Trajectory line alpha.")
    parser.add_argument("--trajectory-linewidth", type=float, default=1.0, help="Trajectory line width before paper scaling.")
    parser.add_argument("--trajectory-color", default="#111111", help="Trajectory line color.")
    parser.add_argument("--trajectory-outline-color", default="white", help="Trajectory outline color.")
    parser.add_argument("--trajectory-outline-width", type=float, default=1.2, help="Additional trajectory outline width.")
    parser.add_argument("--show-snapshot-points", action="store_true", help="Overlay ion positions from exported snapshots.")
    parser.add_argument(
        "--snapshot-point-mode",
        choices=("latest", "all"),
        default="latest",
        help="Use the latest/selected snapshot or aggregate all snapshots before local-window sampling.",
    )
    parser.add_argument("--snapshot-file", default="", help="Optional explicit snapshot_macro_*.npy file for ion points.")
    parser.add_argument("--snapshot-macro", type=int, default=None, help="Use the snapshot nearest this macro step.")
    parser.add_argument("--snapshot-time-ms", type=float, default=None, help="Use the snapshot nearest this time [ms].")
    parser.add_argument(
        "--snapshot-time-step-s",
        type=float,
        default=5.0e-8,
        help="Fallback macro time step for snapshot files when snapshots_index.csv is absent.",
    )
    parser.add_argument("--snapshot-point-max", type=int, default=4000, help="Maximum ion snapshot points to draw.")
    parser.add_argument("--snapshot-point-size", type=float, default=5.0, help="Ion snapshot point marker size.")
    parser.add_argument("--snapshot-point-alpha", type=float, default=0.55, help="Ion snapshot point alpha.")
    parser.add_argument("--snapshot-point-color", default="#66c2a5", help="Ion point color when --snapshot-point-color-by none.")
    parser.add_argument(
        "--snapshot-point-color-by",
        default="track_id",
        help="Ion point color source: none, any snapshot column, vr_m_per_s, or vz_m_per_s.",
    )
    parser.add_argument("--snapshot-point-cmap", default="viridis", help="Ion point colormap.")
    parser.add_argument("--snapshot-point-vmin", type=float, default=None, help="Ion point color scale minimum.")
    parser.add_argument("--snapshot-point-vmax", type=float, default=None, help="Ion point color scale maximum.")
    parser.add_argument("--show-gas-streamlines", action="store_true", help="Overlay gas streamlines using (v_z, v_r).")
    parser.add_argument("--gas-streamline-density", type=float, default=1.15, help="Matplotlib streamplot density.")
    parser.add_argument("--gas-streamline-color", default="#222222", help="Gas streamline color.")
    parser.add_argument("--gas-streamline-linewidth", type=float, default=0.85, help="Gas streamline line width before paper scaling.")
    parser.add_argument("--gas-streamline-arrowsize", type=float, default=0.85, help="Gas streamline arrow size.")
    parser.add_argument("--show-ion-velocity-arrows", action="store_true", help="Overlay ion velocity direction arrows.")
    parser.add_argument(
        "--ion-arrow-source",
        choices=("trajectories", "snapshot", "auto"),
        default="trajectories",
        help="Draw ion arrows from representative trajectories or from the selected snapshot point cloud.",
    )
    parser.add_argument("--ion-arrow-stride", type=int, default=12, help="Use every Nth trajectory row for candidate arrows.")
    parser.add_argument("--max-ion-arrows", type=int, default=120, help="Maximum ion velocity arrows to draw.")
    parser.add_argument("--ion-arrow-length-mm", type=float, default=0.35, help="Displayed ion velocity arrow length [mm].")
    parser.add_argument("--ion-arrow-color", default="#f5f5f5", help="Ion velocity arrow color.")
    parser.add_argument("--ion-arrow-alpha", type=float, default=0.85, help="Ion velocity arrow alpha.")
    parser.add_argument("--ion-arrow-width", type=float, default=0.0030, help="Ion velocity arrow shaft width before paper scaling.")
    parser.add_argument("--vmin", type=float, default=None, help="Manual v_r color scale minimum [m/s].")
    parser.add_argument("--vmax", type=float, default=None, help="Manual v_r color scale maximum [m/s].")
    parser.add_argument("--levels", type=int, default=80, help="Contour level count.")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap.")
    parser.add_argument(
        "--subplot-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label grouped subplots as (a), (b), ... for manuscript figures.",
    )
    parser.add_argument("--paper-font-scale", type=float, default=1.18, help="Global font-size multiplier for paper figures.")
    parser.add_argument("--paper-line-scale", type=float, default=1.25, help="Line-width multiplier for paper figures.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    parser.add_argument("--title", default="", help="Optional plot title.")
    return parser.parse_args()


def main() -> None:
    make_plot(parse_args())


if __name__ == "__main__":
    main()
