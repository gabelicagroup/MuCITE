"""Plot the closest SIMION/Python trajectory with axial error bars."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from ..benchmarks.simion_trajectory_compare import parse_simion_trajectory
except ImportError:  # pragma: no cover - supports direct script execution.
    from src.data.benchmarks.simion_trajectory_compare import (  # type: ignore
        parse_simion_trajectory,
    )


DEFAULT_ROOT = Path("outputs") / "sim_py_traj_benchmark"
DEFAULT_OUTPUT_SUBDIR = DEFAULT_ROOT / "python_traj_compare_phase_m90"


@dataclass(frozen=True)
class TrajectoryMetrics:
    ion_id: int
    sample_count: int
    tof_final_us: float
    simion_final_z_mm: float
    python_final_z_mm: float
    simion_final_r_mm: float
    python_final_r_mm: float
    axial_rmse_mm: float
    axial_abs_p95_mm: float
    axial_abs_max_mm: float
    transverse_rmse_mm: float
    position_rmse_mm: float
    final_abs_axial_mm: float
    final_abs_transverse_mm: float
    final_abs_ke_ev: float
    simion_final_ke_ev: float
    python_final_ke_ev: float
    final_event_class: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ion_id": self.ion_id,
            "sample_count": self.sample_count,
            "tof_final_us": self.tof_final_us,
            "simion_final_z_mm": self.simion_final_z_mm,
            "python_final_z_mm": self.python_final_z_mm,
            "simion_final_r_mm": self.simion_final_r_mm,
            "python_final_r_mm": self.python_final_r_mm,
            "axial_rmse_mm": self.axial_rmse_mm,
            "axial_abs_p95_mm": self.axial_abs_p95_mm,
            "axial_abs_max_mm": self.axial_abs_max_mm,
            "transverse_rmse_mm": self.transverse_rmse_mm,
            "position_rmse_mm": self.position_rmse_mm,
            "final_abs_axial_mm": self.final_abs_axial_mm,
            "final_abs_transverse_mm": self.final_abs_transverse_mm,
            "final_abs_ke_ev": self.final_abs_ke_ev,
            "simion_final_ke_ev": self.simion_final_ke_ev,
            "python_final_ke_ev": self.python_final_ke_ev,
            "final_event_class": self.final_event_class,
        }


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _load_sample_groups(path: Path) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "ion_id",
            "sample_index",
            "tof_us",
            "simion_x_axial_mm",
            "python_z_axial_mm",
            "axial_z_diff_mm",
            "simion_r_mm",
            "python_r_mm",
            "r_diff_mm",
            "transverse_distance_diff_mm",
            "simion_ke_ev",
            "python_ke_ev",
            "ke_diff_ev",
            "event_class",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            ion_id = int(row["ion_id"])
            groups.setdefault(ion_id, []).append(row)

    if not groups:
        raise ValueError(f"No trajectory samples were loaded from {path}")

    for rows in groups.values():
        rows.sort(key=lambda item: int(item["sample_index"]))
    return dict(sorted(groups.items()))


def _trajectory_metrics(ion_id: int, rows: list[dict[str, str]]) -> TrajectoryMetrics:
    axial = np.asarray([_float(row, "axial_z_diff_mm") for row in rows], dtype=np.float64)
    transverse = np.asarray([_float(row, "transverse_distance_diff_mm") for row in rows], dtype=np.float64)
    position = np.hypot(axial, transverse)
    final = rows[-1]

    return TrajectoryMetrics(
        ion_id=int(ion_id),
        sample_count=len(rows),
        tof_final_us=_float(final, "tof_us"),
        simion_final_z_mm=_float(final, "simion_x_axial_mm"),
        python_final_z_mm=_float(final, "python_z_axial_mm"),
        simion_final_r_mm=_float(final, "simion_r_mm"),
        python_final_r_mm=_float(final, "python_r_mm"),
        axial_rmse_mm=float(np.sqrt(np.mean(axial * axial))),
        axial_abs_p95_mm=float(np.percentile(np.abs(axial), 95)),
        axial_abs_max_mm=float(np.max(np.abs(axial))),
        transverse_rmse_mm=float(np.sqrt(np.mean(transverse * transverse))),
        position_rmse_mm=float(np.sqrt(np.mean(position * position))),
        final_abs_axial_mm=abs(_float(final, "axial_z_diff_mm")),
        final_abs_transverse_mm=abs(_float(final, "transverse_distance_diff_mm")),
        final_abs_ke_ev=abs(_float(final, "ke_diff_ev")),
        simion_final_ke_ev=_float(final, "simion_ke_ev"),
        python_final_ke_ev=_float(final, "python_ke_ev"),
        final_event_class=str(final["event_class"]),
    )


def _metric_value(metrics: TrajectoryMetrics, selection_metric: str) -> float:
    if selection_metric == "position-rmse":
        return metrics.position_rmse_mm
    if selection_metric == "axial-rmse":
        return metrics.axial_rmse_mm
    if selection_metric == "final-axial":
        return metrics.final_abs_axial_mm
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _thin_indices(count: int, max_count: int) -> np.ndarray:
    if count <= max_count:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, max_count, dtype=int))


def _format_um(value_mm: float) -> str:
    return f"{value_mm * 1.0e3:.3g}"


def _errorbar_values(rows: list[dict[str, str]], mode: str) -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
    if mode == "axial":
        values = np.abs(np.asarray([_float(row, "axial_z_diff_mm") for row in rows], dtype=np.float64))
        return values, None, "|axial error|", "horizontal xerr"
    if mode == "radial":
        values = np.abs(np.asarray([_float(row, "r_diff_mm") for row in rows], dtype=np.float64))
        return None, values, "|radial error|", "vertical yerr"
    if mode == "transverse":
        values = np.abs(np.asarray([_float(row, "transverse_distance_diff_mm") for row in rows], dtype=np.float64))
        return None, values, "|transverse position error|", "vertical yerr"
    raise ValueError(f"Unsupported errorbar mode: {mode}")


def _plot_selected(
    *,
    path: Path,
    rows: list[dict[str, str]],
    selected: TrajectoryMetrics,
    selection_metric: str,
    errorbar_mode: str,
    simion_ion_count: int,
    errorbar_count: int,
    dpi: int,
) -> None:
    simion_z = np.asarray([_float(row, "simion_x_axial_mm") for row in rows], dtype=np.float64)
    simion_r = np.asarray([_float(row, "simion_r_mm") for row in rows], dtype=np.float64)
    python_z = np.asarray([_float(row, "python_z_axial_mm") for row in rows], dtype=np.float64)
    python_r = np.asarray([_float(row, "python_r_mm") for row in rows], dtype=np.float64)
    xerr, yerr, error_label, error_direction = _errorbar_values(rows, errorbar_mode)

    fig, ax = plt.subplots(figsize=(10.6, 5.2), dpi=dpi)
    fig.subplots_adjust(right=0.72)
    python_line, = ax.plot(
        python_z,
        python_r,
        linestyle="-",
        color="#c2410c",
        linewidth=2.3,
        label=f"Field baker trajectory: solid, {error_direction} = {error_label}",
        zorder=2,
    )
    simion_line, = ax.plot(
        simion_z,
        simion_r,
        linestyle="--",
        color="#1f2937",
        linewidth=1.7,
        label="SIMION trajectory: dashed, z = SIMION X",
        zorder=4,
    )

    error_indices = _thin_indices(len(rows), max(2, int(errorbar_count)))
    sampled_xerr = None if xerr is None else xerr[error_indices]
    sampled_yerr = None if yerr is None else yerr[error_indices]
    ax.errorbar(
        python_z[error_indices],
        python_r[error_indices],
        xerr=sampled_xerr,
        yerr=sampled_yerr,
        fmt="none",
        ecolor="#c2410c",
        elinewidth=0.8,
        capsize=2.0,
        alpha=0.75,
        zorder=3,
    )

    ax.set_xlabel("axial z [mm]")
    ax.set_ylabel("radius r [mm]")
    ax.set_title(f"Closest trajectory comparison, ion {selected.ion_id}")
    ax.grid(True, alpha=0.25)
    ax.legend(handles=[simion_line, python_line], loc="upper left", frameon=True)

    property_text = "\n".join(
        [
            "Line properties",
            f"selected ion: {selected.ion_id} of {simion_ion_count}",
            f"selection: min {selection_metric}",
            f"error bars: {error_label}",
            f"direction: {error_direction}",
            f"samples: {selected.sample_count}",
            f"final TOF: {selected.tof_final_us:.4g} us",
            f"axial RMSE: {_format_um(selected.axial_rmse_mm)} um",
            f"axial p95/max: {_format_um(selected.axial_abs_p95_mm)} / {_format_um(selected.axial_abs_max_mm)} um",
            f"position RMSE: {_format_um(selected.position_rmse_mm)} um",
            f"final KE: SIMION {selected.simion_final_ke_ev:.5g} eV, Python {selected.python_final_ke_ev:.5g} eV",
        ]
    )
    ax.text(
        1.02,
        0.5,
        property_text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#b8b8b8", "alpha": 0.92},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_best_trajectory_plot(args: argparse.Namespace) -> dict[str, Path]:
    simion_path = Path(args.simion_traj)
    samples_path = Path(args.samples)
    output_dir = Path(args.output_dir)

    simion_by_ion = parse_simion_trajectory(simion_path)
    sample_groups = _load_sample_groups(samples_path)
    missing_in_samples = sorted(set(simion_by_ion).difference(sample_groups))
    if missing_in_samples:
        raise ValueError(f"SIMION ions missing from comparison samples: {missing_in_samples[:10]}")

    metrics = [_trajectory_metrics(ion_id, rows) for ion_id, rows in sample_groups.items()]
    metrics.sort(key=lambda item: _metric_value(item, args.selection_metric))
    selected = metrics[0]
    selected_rows = sample_groups[selected.ion_id]

    metrics_csv = output_dir / "best_trajectory_metrics.csv"
    selected_csv = output_dir / f"best_trajectory_ion{selected.ion_id}_samples.csv"
    summary_json = output_dir / f"best_trajectory_{args.errorbar_mode}_summary.json"
    if args.output:
        plot_path = Path(args.output)
    else:
        plot_path = output_dir / f"best_trajectory_{args.errorbar_mode}_errorbar.png"

    _write_dict_rows(metrics_csv, [item.as_dict() for item in metrics])
    _write_dict_rows(selected_csv, selected_rows)
    _plot_selected(
        path=plot_path,
        rows=selected_rows,
        selected=selected,
        selection_metric=args.selection_metric,
        errorbar_mode=args.errorbar_mode,
        simion_ion_count=len(simion_by_ion),
        errorbar_count=args.errorbar_count,
        dpi=args.dpi,
    )

    summary = {
        "simion_traj": str(simion_path),
        "comparison_samples": str(samples_path),
        "selection_metric": args.selection_metric,
        "errorbar_mode": args.errorbar_mode,
        "selected": selected.as_dict(),
        "simion_ion_count": len(simion_by_ion),
        "sample_ion_count": len(sample_groups),
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "selected_samples_csv": str(selected_csv),
            "plot": str(plot_path),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "metrics_csv": metrics_csv,
        "selected_samples_csv": selected_csv,
        "summary_json": summary_json,
        "plot": plot_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the closest time-matched SIMION/Python trajectory.")
    parser.add_argument("--simion-traj", default=str(DEFAULT_ROOT / "simion_benchmark_traj.csv"))
    parser.add_argument("--samples", default=str(DEFAULT_OUTPUT_SUBDIR / "trajectory_comparison_samples.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_SUBDIR))
    parser.add_argument(
        "--selection-metric",
        choices=("position-rmse", "axial-rmse", "final-axial"),
        default="position-rmse",
        help="Metric used to choose the closest trajectory.",
    )
    parser.add_argument(
        "--errorbar-mode",
        choices=("axial", "radial", "transverse"),
        default="axial",
        help="Error component drawn on the field-baker/Python trajectory.",
    )
    parser.add_argument("--errorbar-count", type=int, default=80, help="Maximum number of error bars to draw.")
    parser.add_argument("--output", default="", help="Optional output PNG path. Defaults to mode-specific filename.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    outputs = build_best_trajectory_plot(parse_args())
    print("Saved best trajectory comparison:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
