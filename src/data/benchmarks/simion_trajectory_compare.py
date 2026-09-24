"""Time-matched trajectory comparison against a SIMION flight log.

This script complements simion_field_benchmark.py. It parses SIMION trajectory
records, initializes the same ions in the Python pure-field model, samples the
Python trajectory at each SIMION TOF, and writes compact comparison artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .simion_field_benchmark import (
    AMU_KG,
    DEFAULT_FIELD_PATH,
    ELEMENTARY_CHARGE_C,
    BenchmarkConfig,
    _json_ready,
    _simion_value,
    load_baked_field,
    rk4_step,
)


DEFAULT_SIMION_TRAJ_PATH = (
    Path("outputs") / "sim_py_traj_benchmark" / "simion_benchmark_traj.csv"
)


@dataclass(frozen=True)
class SimionTrajectoryPoint:
    ion_id: int
    event: str
    tof_us: float
    simion_x_mm: float
    simion_y_mm: float
    simion_z_mm: float
    simion_vx_mm_per_us: float
    simion_vy_mm_per_us: float
    simion_vz_mm_per_us: float
    simion_vt_mm_per_us: float
    simion_ke_ev: float


def parse_simion_trajectory(path: Path) -> Dict[int, List[SimionTrajectoryPoint]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    starts = [match.start() for match in re.finditer(r"Begin Fly'm", text)]
    block = text[starts[-1] :] if starts else text

    by_ion: Dict[int, List[SimionTrajectoryPoint]] = {}
    for chunk in re.split(r"\n\s*\n", block):
        if "Ion(" not in chunk or "Event(" not in chunk:
            continue
        record = " ".join(chunk.split())
        ion_match = re.search(r"Ion\((\d+)\)", record)
        event_match = re.search(r"Event\(([^)]*)\)", record)
        if ion_match is None or event_match is None:
            continue
        try:
            point = SimionTrajectoryPoint(
                ion_id=int(ion_match.group(1)),
                event=event_match.group(1).strip(),
                tof_us=_simion_value(record, "TOF", "usec"),
                simion_x_mm=_simion_value(record, "X", "mm"),
                simion_y_mm=_simion_value(record, "Y", "mm"),
                simion_z_mm=_simion_value(record, "Z", "mm"),
                simion_vx_mm_per_us=_simion_value(record, "Vx", "mm/usec"),
                simion_vy_mm_per_us=_simion_value(record, "Vy", "mm/usec"),
                simion_vz_mm_per_us=_simion_value(record, "Vz", "mm/usec"),
                simion_vt_mm_per_us=_simion_value(record, "Vt", "mm/usec"),
                simion_ke_ev=_simion_value(record, "KE", "eV"),
            )
        except ValueError:
            continue
        by_ion.setdefault(point.ion_id, []).append(point)

    for ion_id, points in by_ion.items():
        points.sort(key=lambda point: point.tof_us)
        if not points:
            raise ValueError(f"SIMION trajectory for ion {ion_id} is empty.")
    if not by_ion:
        raise ValueError(f"No SIMION trajectory records were parsed from {path}.")
    return dict(sorted(by_ion.items()))


def downsample_simion_trajectory(
    simion_by_ion: Dict[int, List[SimionTrajectoryPoint]],
    max_samples_per_ion: int,
) -> tuple[Dict[int, List[SimionTrajectoryPoint]], dict[str, object]]:
    """Limit SIMION time samples per ion while preserving first and final points."""

    if max_samples_per_ion <= 0:
        sample_counts = {ion_id: len(points) for ion_id, points in simion_by_ion.items()}
        return simion_by_ion, {
            "enabled": False,
            "max_samples_per_ion": 0,
            "original_sample_count": int(sum(sample_counts.values())),
            "retained_sample_count": int(sum(sample_counts.values())),
            "original_sample_count_by_ion": {str(k): int(v) for k, v in sample_counts.items()},
            "retained_sample_count_by_ion": {str(k): int(v) for k, v in sample_counts.items()},
        }

    downsampled: Dict[int, List[SimionTrajectoryPoint]] = {}
    original_counts: dict[str, int] = {}
    retained_counts: dict[str, int] = {}
    for ion_id, points in simion_by_ion.items():
        n_points = len(points)
        original_counts[str(ion_id)] = int(n_points)
        if n_points <= max_samples_per_ion:
            selected = list(points)
        elif max_samples_per_ion <= 1:
            selected = [points[-1]]
        else:
            indices = np.linspace(0, n_points - 1, int(max_samples_per_ion), dtype=int)
            indices = np.unique(indices)
            if indices[0] != 0:
                indices = np.insert(indices, 0, 0)
            if indices[-1] != n_points - 1:
                indices = np.append(indices, n_points - 1)
            selected = [points[int(index)] for index in indices]
        downsampled[ion_id] = selected
        retained_counts[str(ion_id)] = int(len(selected))

    return dict(sorted(downsampled.items())), {
        "enabled": True,
        "max_samples_per_ion": int(max_samples_per_ion),
        "original_sample_count": int(sum(original_counts.values())),
        "retained_sample_count": int(sum(retained_counts.values())),
        "original_sample_count_by_ion": original_counts,
        "retained_sample_count_by_ion": retained_counts,
    }


def _simion_position_m(point: SimionTrajectoryPoint) -> np.ndarray:
    # Project convention: SIMION X -> Python z, SIMION Y -> Python x, SIMION Z -> Python y.
    return np.array(
        [point.simion_y_mm, point.simion_z_mm, point.simion_x_mm],
        dtype=float,
    ) * 1.0e-3


def _simion_velocity_m_per_s(point: SimionTrajectoryPoint) -> np.ndarray:
    return np.array(
        [point.simion_vy_mm_per_us, point.simion_vz_mm_per_us, point.simion_vx_mm_per_us],
        dtype=float,
    ) * 1.0e3


def _first_created(points: Iterable[SimionTrajectoryPoint]) -> SimionTrajectoryPoint:
    for point in points:
        if "Ion Created" in point.event:
            return point
    first = next(iter(points), None)
    if first is None:
        raise ValueError("SIMION trajectory has no points.")
    return first


def _event_class(event: str) -> str:
    if "Hit Electrode" in event:
        return "hit_electrode"
    if "Outside Work Bench" in event:
        return "outside_work_bench"
    if "Ion Created" in event:
        return "ion_created"
    return "next_time_step"


def _stats(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan, "rmse": math.nan}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
    }


def _signed_stats(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "p05": math.nan,
            "p95": math.nan,
            "rmse": math.nan,
        }
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.percentile(finite, 50)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
    }


def run_time_matched_compare(
    simion_by_ion: Dict[int, List[SimionTrajectoryPoint]],
    config: BenchmarkConfig,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    field = load_baked_field(config.field_path)
    mass_kg = config.mass_amu * AMU_KG
    charge_c = config.charge_state * ELEMENTARY_CHARGE_C

    rows: List[Dict[str, object]] = []
    final_rows: List[Dict[str, object]] = []
    ion_ids = sorted(simion_by_ion)
    detector_z_mm = config.detector_z_m * 1.0e3
    max_substeps_per_ion = 0

    for ion_id in ion_ids:
        simion_points = simion_by_ion[ion_id]
        created = _first_created(simion_points)
        position_m = _simion_position_m(created).reshape(1, 3)
        velocity_m_per_s = _simion_velocity_m_per_s(created).reshape(1, 3)
        time_s = float(created.tof_us) * 1.0e-6
        substeps = 0

        for sample_index, point in enumerate(simion_points):
            target_s = float(point.tof_us) * 1.0e-6
            if target_s + 1.0e-18 < time_s:
                raise ValueError(
                    f"SIMION trajectory for ion {ion_id} is not time sorted at sample {sample_index}."
                )
            while time_s + config.dt_s < target_s:
                position_m, velocity_m_per_s = rk4_step(
                    field,
                    position_m,
                    velocity_m_per_s,
                    time_s,
                    config.dt_s,
                    mass_kg,
                    charge_c,
                    config,
                )
                time_s += config.dt_s
                substeps += 1
            remaining_s = target_s - time_s
            if remaining_s > 1.0e-18:
                position_m, velocity_m_per_s = rk4_step(
                    field,
                    position_m,
                    velocity_m_per_s,
                    time_s,
                    remaining_s,
                    mass_kg,
                    charge_c,
                    config,
                )
                time_s = target_s
                substeps += 1

            sim_pos_m = _simion_position_m(point)
            sim_vel_m_per_s = _simion_velocity_m_per_s(point)
            py_pos_m = position_m[0].copy()
            py_vel_m_per_s = velocity_m_per_s[0].copy()
            sim_r_mm = float(np.hypot(sim_pos_m[0], sim_pos_m[1]) * 1.0e3)
            py_r_mm = float(np.hypot(py_pos_m[0], py_pos_m[1]) * 1.0e3)
            pos_diff_mm = (py_pos_m - sim_pos_m) * 1.0e3
            vel_diff_mm_per_us = (py_vel_m_per_s - sim_vel_m_per_s) * 1.0e-3
            py_speed_mm_per_us = float(np.linalg.norm(py_vel_m_per_s) * 1.0e-3)
            py_ke_ev = float(0.5 * mass_kg * np.dot(py_vel_m_per_s, py_vel_m_per_s) / ELEMENTARY_CHARGE_C)
            event_class = _event_class(point.event)
            row = {
                "ion_id": ion_id,
                "sample_index": sample_index,
                "simion_event": point.event,
                "event_class": event_class,
                "tof_us": point.tof_us,
                "simion_x_axial_mm": point.simion_x_mm,
                "python_z_axial_mm": float(py_pos_m[2] * 1.0e3),
                "axial_z_diff_mm": float(pos_diff_mm[2]),
                "simion_y_to_python_x_mm": point.simion_y_mm,
                "python_x_mm": float(py_pos_m[0] * 1.0e3),
                "x_diff_mm": float(pos_diff_mm[0]),
                "simion_z_to_python_y_mm": point.simion_z_mm,
                "python_y_mm": float(py_pos_m[1] * 1.0e3),
                "y_diff_mm": float(pos_diff_mm[1]),
                "simion_r_mm": sim_r_mm,
                "python_r_mm": py_r_mm,
                "r_diff_mm": float(py_r_mm - sim_r_mm),
                "transverse_distance_diff_mm": float(np.linalg.norm(pos_diff_mm[:2])),
                "simion_vx_axial_mm_per_us": point.simion_vx_mm_per_us,
                "python_vz_axial_mm_per_us": float(py_vel_m_per_s[2] * 1.0e-3),
                "axial_vz_diff_mm_per_us": float(vel_diff_mm_per_us[2]),
                "simion_vy_to_python_vx_mm_per_us": point.simion_vy_mm_per_us,
                "python_vx_mm_per_us": float(py_vel_m_per_s[0] * 1.0e-3),
                "vx_diff_mm_per_us": float(vel_diff_mm_per_us[0]),
                "simion_vz_to_python_vy_mm_per_us": point.simion_vz_mm_per_us,
                "python_vy_mm_per_us": float(py_vel_m_per_s[1] * 1.0e-3),
                "vy_diff_mm_per_us": float(vel_diff_mm_per_us[1]),
                "simion_vt_mm_per_us": point.simion_vt_mm_per_us,
                "python_speed_mm_per_us": py_speed_mm_per_us,
                "speed_diff_mm_per_us": float(py_speed_mm_per_us - point.simion_vt_mm_per_us),
                "simion_ke_ev": point.simion_ke_ev,
                "python_ke_ev": py_ke_ev,
                "ke_diff_ev": float(py_ke_ev - point.simion_ke_ev),
            }
            rows.append(row)
        max_substeps_per_ion = max(max_substeps_per_ion, substeps)
        final_row = dict(rows[-1])
        final_row["python_detector_residual_mm"] = (
            float(final_row["python_z_axial_mm"]) - detector_z_mm
        )
        final_rows.append(final_row)

    axial_abs = np.abs(np.asarray([row["axial_z_diff_mm"] for row in rows], dtype=float))
    transverse_abs = np.asarray([row["transverse_distance_diff_mm"] for row in rows], dtype=float)
    r_abs = np.abs(np.asarray([row["r_diff_mm"] for row in rows], dtype=float))
    speed_abs = np.abs(np.asarray([row["speed_diff_mm_per_us"] for row in rows], dtype=float))
    ke_abs = np.abs(np.asarray([row["ke_diff_ev"] for row in rows], dtype=float))

    final_axial_abs = np.abs(np.asarray([row["axial_z_diff_mm"] for row in final_rows], dtype=float))
    final_transverse_abs = np.asarray(
        [row["transverse_distance_diff_mm"] for row in final_rows], dtype=float
    )
    final_ke_abs = np.abs(np.asarray([row["ke_diff_ev"] for row in final_rows], dtype=float))
    final_detector_residual = np.asarray(
        [row["python_detector_residual_mm"] for row in final_rows], dtype=float
    )

    simion_event_counts: Dict[str, int] = {}
    for row in final_rows:
        label = str(row["event_class"])
        simion_event_counts[label] = simion_event_counts.get(label, 0) + 1

    initial_radius = np.asarray(
        [math.hypot(_first_created(points).simion_y_mm, _first_created(points).simion_z_mm) for points in simion_by_ion.values()],
        dtype=float,
    )
    initial_angle_deg = []
    for points in simion_by_ion.values():
        created = _first_created(points)
        transverse_v = math.hypot(created.simion_vy_mm_per_us, created.simion_vz_mm_per_us)
        initial_angle_deg.append(math.degrees(math.atan2(transverse_v, created.simion_vx_mm_per_us)))

    summary = {
        "config": _json_ready(config.__dict__),
        "field_metadata": _json_ready(field.metadata),
        "simion": {
            "ion_count": len(ion_ids),
            "sample_count": len(rows),
            "sample_count_by_ion": {str(ion_id): len(simion_by_ion[ion_id]) for ion_id in ion_ids},
            "final_event_counts": simion_event_counts,
            "initial_radius_mm": _stats(initial_radius),
            "initial_angle_deg": _stats(np.asarray(initial_angle_deg, dtype=float)),
        },
        "integration": {
            "dt_s": config.dt_s,
            "rf_phase_deg": math.degrees(config.rf_phase_rad),
            "rf_peak_voltage_v": config.rf_peak_voltage_v,
            "rf_frequency_hz": config.rf_frequency_hz,
            "max_substeps_per_ion": max_substeps_per_ion,
            "coordinate_mapping": "SIMION X->Python z, SIMION Y->Python x, SIMION Z->Python y",
            "detector_z_mm": detector_z_mm,
            "gas_collisions_space_charge": "disabled",
        },
        "all_samples_abs_error": {
            "axial_z_mm": _stats(axial_abs),
            "transverse_distance_mm": _stats(transverse_abs),
            "radius_mm": _stats(r_abs),
            "speed_mm_per_us": _stats(speed_abs),
            "kinetic_energy_ev": _stats(ke_abs),
        },
        "all_samples_signed_error": {
            "axial_z_mm": _signed_stats(np.asarray([row["axial_z_diff_mm"] for row in rows], dtype=float)),
            "radius_mm": _signed_stats(np.asarray([row["r_diff_mm"] for row in rows], dtype=float)),
            "speed_mm_per_us": _signed_stats(np.asarray([row["speed_diff_mm_per_us"] for row in rows], dtype=float)),
            "kinetic_energy_ev": _signed_stats(np.asarray([row["ke_diff_ev"] for row in rows], dtype=float)),
        },
        "final_abs_error": {
            "axial_z_mm": _stats(final_axial_abs),
            "transverse_distance_mm": _stats(final_transverse_abs),
            "kinetic_energy_ev": _stats(final_ke_abs),
        },
        "final_detector_residual_mm": _signed_stats(final_detector_residual),
    }
    return summary, rows, final_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _downsample_rows(rows: List[Dict[str, object]], max_points: int = 30000) -> List[Dict[str, object]]:
    if len(rows) <= max_points:
        return rows
    indices = np.linspace(0, len(rows) - 1, max_points, dtype=int)
    return [rows[int(index)] for index in indices]


def plot_rz_overlay(path: Path, rows: List[Dict[str, object]]) -> None:
    by_ion: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_ion.setdefault(int(row["ion_id"]), []).append(row)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for offset, (ion_id, ion_rows) in enumerate(by_ion.items()):
        color = color_cycle[offset % len(color_cycle)]
        simion_z = np.asarray([row["simion_x_axial_mm"] for row in ion_rows], dtype=float)
        simion_r = np.asarray([row["simion_r_mm"] for row in ion_rows], dtype=float)
        python_z = np.asarray([row["python_z_axial_mm"] for row in ion_rows], dtype=float)
        python_r = np.asarray([row["python_r_mm"] for row in ion_rows], dtype=float)
        ax.plot(simion_z, simion_r, color=color, linewidth=0.8, alpha=0.75)
        ax.plot(python_z, python_r, color=color, linewidth=0.8, alpha=0.75, linestyle="--")
        if offset < 10:
            ax.text(python_z[-1], python_r[-1], str(ion_id), fontsize=6, color=color)
    ax.set_xlabel("axial z / SIMION X [mm]")
    ax.set_ylabel("radius [mm]")
    ax.set_title("SIMION solid vs Python dashed trajectory r(z)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rz_by_ion_pdf(path: Path, rows: List[Dict[str, object]], ions_per_page: int = 25) -> None:
    by_ion: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_ion.setdefault(int(row["ion_id"]), []).append(row)

    ion_items = sorted(by_ion.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for page_start in range(0, len(ion_items), ions_per_page):
            page_items = ion_items[page_start : page_start + ions_per_page]
            ncols = 5
            nrows = int(math.ceil(len(page_items) / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(13.5, max(2.2 * nrows, 2.8)),
                squeeze=False,
                sharex=False,
                sharey=False,
            )
            for ax in axes.ravel():
                ax.set_visible(False)
            for ax, (ion_id, ion_rows) in zip(axes.ravel(), page_items):
                ax.set_visible(True)
                simion_z = np.asarray([row["simion_x_axial_mm"] for row in ion_rows], dtype=float)
                simion_r = np.asarray([row["simion_r_mm"] for row in ion_rows], dtype=float)
                python_z = np.asarray([row["python_z_axial_mm"] for row in ion_rows], dtype=float)
                python_r = np.asarray([row["python_r_mm"] for row in ion_rows], dtype=float)
                axial_um = np.abs(np.asarray([row["axial_z_diff_mm"] for row in ion_rows], dtype=float)) * 1.0e3
                trans_um = (
                    np.asarray([row["transverse_distance_diff_mm"] for row in ion_rows], dtype=float) * 1.0e3
                )
                ax.plot(simion_z, simion_r, color="black", linewidth=1.0, label="SIMION")
                ax.plot(python_z, python_r, color="#d55e00", linewidth=1.0, linestyle="--", label="Python")
                ax.set_title(
                    f"ion {ion_id}: z p95 {np.percentile(axial_um, 95):.1f} um, "
                    f"r p95 {np.percentile(trans_um, 95):.1f} um",
                    fontsize=7.0,
                )
                ax.grid(True, alpha=0.2)
                ax.tick_params(labelsize=6.5)
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
            fig.supxlabel("axial z / SIMION X [mm]", fontsize=9)
            fig.supylabel("radius [mm]", fontsize=9)
            fig.suptitle(
                f"Per-ion SIMION/Python r-z trajectory comparison, ions {page_start + 1}-"
                f"{page_start + len(page_items)}",
                fontsize=11,
            )
            fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def plot_error_vs_time(path: Path, rows: List[Dict[str, object]]) -> None:
    sampled = _downsample_rows(rows)
    tof_us = np.asarray([row["tof_us"] for row in sampled], dtype=float)
    axial_um = np.abs(np.asarray([row["axial_z_diff_mm"] for row in sampled], dtype=float)) * 1.0e3
    transverse_um = np.asarray([row["transverse_distance_diff_mm"] for row in sampled], dtype=float) * 1.0e3
    ke_ev = np.abs(np.asarray([row["ke_diff_ev"] for row in sampled], dtype=float))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)
    axes[0].scatter(tof_us, axial_um, s=3, alpha=0.35)
    axes[0].set_ylabel("|axial error| [um]")
    axes[0].grid(True, alpha=0.25)
    axes[1].scatter(tof_us, transverse_um, s=3, alpha=0.35)
    axes[1].set_ylabel("transverse error [um]")
    axes[1].grid(True, alpha=0.25)
    axes[2].scatter(tof_us, ke_ev, s=3, alpha=0.35)
    axes[2].set_ylabel("|KE error| [eV]")
    axes[2].set_xlabel("time [us]")
    axes[2].grid(True, alpha=0.25)
    fig.suptitle("Time-matched trajectory errors")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fmt(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_report(
    path: Path,
    simion_path: Path,
    summary_path: Path,
    sample_csv_path: Path,
    final_csv_path: Path,
    rz_plot_path: Path,
    error_plot_path: Path,
    summary: Dict[str, object],
) -> None:
    all_abs = summary["all_samples_abs_error"]
    final_abs = summary["final_abs_error"]
    residual = summary["final_detector_residual_mm"]
    simion = summary["simion"]
    integration = summary["integration"]
    field = summary["field_metadata"]

    lines = [
        "# SIMION/Python trajectory comparison",
        "",
        f"SIMION trajectory log: `{simion_path}`",
        "",
        "## Setup",
        "",
        f"- ions: `{simion['ion_count']}`",
        f"- SIMION trajectory samples: `{simion['sample_count']}`",
        f"- field: `{summary['config']['field_path']}`",
        f"- RF phase used in Python: `{_fmt(integration['rf_phase_deg'], 6)} deg`",
        f"- RF peak voltage: `{_fmt(integration['rf_peak_voltage_v'], 6)} V`",
        f"- RF frequency: `{_fmt(integration['rf_frequency_hz'], 6)} Hz`",
        f"- RK4 max step: `{_fmt(integration['dt_s'], 6)} s`",
        f"- coordinate mapping: `{integration['coordinate_mapping']}`",
        f"- physics disabled for this benchmark: `{integration['gas_collisions_space_charge']}`",
        f"- field grid: `dr={_fmt(field['dr_m'] * 1.0e3, 6)} mm`, `dz={_fmt(field['dz_m'] * 1.0e3, 6)} mm`",
        "",
        "## SIMION input summary",
        "",
        f"- final events: `{simion['final_event_counts']}`",
        f"- initial radius p95: `{_fmt(simion['initial_radius_mm']['p95'], 6)} mm`",
        f"- initial angle p95: `{_fmt(simion['initial_angle_deg']['p95'], 6)} deg`",
        "",
        "## Time-matched trajectory error",
        "",
        "| quantity | mean abs | median abs | p95 abs | max abs | RMSE |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| axial position [mm] | "
            f"`{_fmt(all_abs['axial_z_mm']['mean'])}` | `{_fmt(all_abs['axial_z_mm']['median'])}` | "
            f"`{_fmt(all_abs['axial_z_mm']['p95'])}` | `{_fmt(all_abs['axial_z_mm']['max'])}` | `{_fmt(all_abs['axial_z_mm']['rmse'])}` |"
        ),
        (
            "| transverse distance [mm] | "
            f"`{_fmt(all_abs['transverse_distance_mm']['mean'])}` | `{_fmt(all_abs['transverse_distance_mm']['median'])}` | "
            f"`{_fmt(all_abs['transverse_distance_mm']['p95'])}` | `{_fmt(all_abs['transverse_distance_mm']['max'])}` | `{_fmt(all_abs['transverse_distance_mm']['rmse'])}` |"
        ),
        (
            "| radius [mm] | "
            f"`{_fmt(all_abs['radius_mm']['mean'])}` | `{_fmt(all_abs['radius_mm']['median'])}` | "
            f"`{_fmt(all_abs['radius_mm']['p95'])}` | `{_fmt(all_abs['radius_mm']['max'])}` | `{_fmt(all_abs['radius_mm']['rmse'])}` |"
        ),
        (
            "| speed [mm/usec] | "
            f"`{_fmt(all_abs['speed_mm_per_us']['mean'])}` | `{_fmt(all_abs['speed_mm_per_us']['median'])}` | "
            f"`{_fmt(all_abs['speed_mm_per_us']['p95'])}` | `{_fmt(all_abs['speed_mm_per_us']['max'])}` | `{_fmt(all_abs['speed_mm_per_us']['rmse'])}` |"
        ),
        (
            "| kinetic energy [eV] | "
            f"`{_fmt(all_abs['kinetic_energy_ev']['mean'])}` | `{_fmt(all_abs['kinetic_energy_ev']['median'])}` | "
            f"`{_fmt(all_abs['kinetic_energy_ev']['p95'])}` | `{_fmt(all_abs['kinetic_energy_ev']['max'])}` | `{_fmt(all_abs['kinetic_energy_ev']['rmse'])}` |"
        ),
        "",
        "## Endpoint check",
        "",
        (
            f"- final axial p95 abs error: `{_fmt(final_abs['axial_z_mm']['p95'])} mm`; "
            f"max abs error: `{_fmt(final_abs['axial_z_mm']['max'])} mm`"
        ),
        (
            f"- final transverse p95 distance error: `{_fmt(final_abs['transverse_distance_mm']['p95'])} mm`; "
            f"max: `{_fmt(final_abs['transverse_distance_mm']['max'])} mm`"
        ),
        (
            f"- final KE p95 abs error: `{_fmt(final_abs['kinetic_energy_ev']['p95'])} eV`; "
            f"max: `{_fmt(final_abs['kinetic_energy_ev']['max'])} eV`"
        ),
        (
            f"- Python z minus detector plane at SIMION final TOF: mean "
            f"`{_fmt(residual['mean'])} mm`, p05 `{_fmt(residual['p05'])} mm`, "
            f"p95 `{_fmt(residual['p95'])} mm`"
        ),
        "",
        "## Outputs",
        "",
        f"- summary JSON: `{summary_path}`",
        f"- time-matched samples CSV: `{sample_csv_path}`",
        f"- final samples CSV: `{final_csv_path}`",
        f"- r-z overlay plot: `{rz_plot_path}`",
        f"- per-ion r-z comparison PDF: `{rz_plot_path.with_name('trajectory_rz_by_ion.pdf')}`",
        f"- error plot: `{error_plot_path}`",
        "",
        "## Notes",
        "",
        (
            "The comparison uses the current empirical convention from AGENTS.md: "
            "a SIMION export labelled +90 deg aligns with Python `--rf-phase-deg -90`."
        ),
        "Gas, collisions, IonSPA, and space charge are disabled; this is a pure electric-field benchmark.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    simion_path: Path,
    summary: Dict[str, object],
    rows: List[Dict[str, object]],
    final_rows: List[Dict[str, object]],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "trajectory_comparison_summary.json"
    samples_path = output_dir / "trajectory_comparison_samples.csv"
    final_path = output_dir / "trajectory_comparison_final.csv"
    rz_plot_path = output_dir / "trajectory_rz_overlay.png"
    rz_by_ion_pdf_path = output_dir / "trajectory_rz_by_ion.pdf"
    error_plot_path = output_dir / "trajectory_error_vs_time.png"
    report_path = output_dir / "trajectory_comparison_report.md"

    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    write_csv(samples_path, rows)
    write_csv(final_path, final_rows)
    plot_rz_overlay(rz_plot_path, rows)
    plot_rz_by_ion_pdf(rz_by_ion_pdf_path, rows)
    plot_error_vs_time(error_plot_path, rows)
    write_report(
        report_path,
        simion_path,
        summary_path,
        samples_path,
        final_path,
        rz_plot_path,
        error_plot_path,
        summary,
    )
    return {
        "summary": summary_path,
        "samples": samples_path,
        "final": final_path,
        "rz_plot": rz_plot_path,
        "rz_by_ion_pdf": rz_by_ion_pdf_path,
        "error_plot": error_plot_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python trajectories to a SIMION trajectory log.")
    parser.add_argument("--simion-traj", default=str(DEFAULT_SIMION_TRAJ_PATH))
    parser.add_argument("--field", default=str(DEFAULT_FIELD_PATH))
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs") / "sim_py_traj_benchmark" / "python_traj_compare_phase_m90"),
    )
    parser.add_argument("--mass-amu", type=float, default=2000.0)
    parser.add_argument("--charge-state", type=int, default=5)
    parser.add_argument("--rf-peak-voltage", type=float, default=50.0)
    parser.add_argument("--rf-frequency", type=float, default=6.5e5)
    parser.add_argument("--rf-phase-deg", type=float, default=-90.0)
    parser.add_argument("--detector-z-mm", type=float, default=65.0)
    parser.add_argument("--dt", type=float, default=2.0e-9)
    parser.add_argument(
        "--max-simion-samples-per-ion",
        type=int,
        default=0,
        help="Downsample SIMION trajectory samples per ion before comparison; 0 keeps all samples.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, n_ions: int) -> BenchmarkConfig:
    if args.mass_amu <= 0.0:
        raise ValueError("--mass-amu must be positive.")
    if args.charge_state <= 0:
        raise ValueError("--charge-state must be positive.")
    if args.rf_frequency <= 0.0:
        raise ValueError("--rf-frequency must be positive.")
    if args.detector_z_mm <= 0.0:
        raise ValueError("--detector-z-mm must be positive.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    return BenchmarkConfig(
        field_path=Path(args.field),
        output_dir=Path(args.output_dir),
        n_ions=n_ions,
        mass_amu=float(args.mass_amu),
        charge_state=int(args.charge_state),
        rf_peak_voltage_v=float(args.rf_peak_voltage),
        rf_frequency_hz=float(args.rf_frequency),
        rf_phase_rad=math.radians(float(args.rf_phase_deg)),
        detector_z_m=float(args.detector_z_mm) * 1.0e-3,
        dt_s=float(args.dt),
        t_max_s=1.0,
        sample_trajectories=n_ions,
        record_every_steps=1,
        use_electrode_mask=False,
    )


def main() -> None:
    args = parse_args()
    simion_path = Path(args.simion_traj)
    simion_by_ion = parse_simion_trajectory(simion_path)
    simion_by_ion, downsample_summary = downsample_simion_trajectory(
        simion_by_ion,
        int(args.max_simion_samples_per_ion),
    )
    config = build_config(args, n_ions=len(simion_by_ion))
    summary, rows, final_rows = run_time_matched_compare(simion_by_ion, config)
    summary["simion_downsample"] = downsample_summary
    output_paths = write_outputs(Path(args.output_dir), simion_path, summary, rows, final_rows)

    print("Saved SIMION/Python trajectory comparison outputs:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")
    all_abs = summary["all_samples_abs_error"]
    final_abs = summary["final_abs_error"]
    print(
        "Trajectory errors: "
        f"all p95 axial={all_abs['axial_z_mm']['p95']:.6g} mm, "
        f"all p95 transverse={all_abs['transverse_distance_mm']['p95']:.6g} mm, "
        f"final p95 axial={final_abs['axial_z_mm']['p95']:.6g} mm, "
        f"final p95 KE={final_abs['kinetic_energy_ev']['p95']:.6g} eV"
    )


if __name__ == "__main__":
    main()
