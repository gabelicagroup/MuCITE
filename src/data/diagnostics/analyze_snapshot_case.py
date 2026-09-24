from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COL_R_M = 0
COL_Z_M = 1
COL_TEFF_K = 2
COL_VZ = 8
COL_SPEED = 9
COL_KE_EV = 10
COL_WEIGHT = 11
COL_COLLISIONS = 17


def _percentiles(values: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{int(q):02d}": float("nan") for q in qs}
    out = np.percentile(values, qs)
    return {f"p{int(q):02d}": float(v) for q, v in zip(qs, out)}


def _summary_row(macro_step: int, time_s: float, data: np.ndarray) -> dict[str, float | int]:
    if data.size == 0:
        return {
            "macro_step": int(macro_step),
            "time_s": float(time_s),
            "active_macro_particles": 0,
        }
    z_mm = data[:, COL_Z_M] * 1.0e3
    r_mm = data[:, COL_R_M] * 1.0e3
    teff_k = data[:, COL_TEFF_K]
    ke_ev = data[:, COL_KE_EV]
    vz = data[:, COL_VZ]
    collisions = data[:, COL_COLLISIONS]
    weights = data[:, COL_WEIGHT]
    row: dict[str, float | int] = {
        "macro_step": int(macro_step),
        "time_s": float(time_s),
        "time_ms": float(time_s * 1.0e3),
        "active_macro_particles": int(data.shape[0]),
        "represented_real_ions": float(np.sum(weights)),
        "z_4p5_20mm_count": int(np.count_nonzero((z_mm >= 4.5) & (z_mm < 20.0))),
        "z_20_40mm_count": int(np.count_nonzero((z_mm >= 20.0) & (z_mm < 40.0))),
        "z_40_65mm_count": int(np.count_nonzero((z_mm >= 40.0) & (z_mm <= 65.0))),
        "z_ge_64mm_count": int(np.count_nonzero(z_mm >= 64.0)),
        "z_ge_65mm_count": int(np.count_nonzero(z_mm >= 65.0)),
        "teff_gt_1000k_count": int(np.count_nonzero(teff_k > 1000.0)),
        "teff_gt_2000k_count": int(np.count_nonzero(teff_k > 2000.0)),
        "teff_gt_5000k_count": int(np.count_nonzero(teff_k > 5000.0)),
        "ke_gt_1ev_count": int(np.count_nonzero(ke_ev > 1.0)),
        "ke_gt_2ev_count": int(np.count_nonzero(ke_ev > 2.0)),
        "ke_gt_5ev_count": int(np.count_nonzero(ke_ev > 5.0)),
        "teff_mean_k": float(np.mean(teff_k)),
        "teff_max_k": float(np.max(teff_k)),
        "ke_mean_ev": float(np.mean(ke_ev)),
        "ke_max_ev": float(np.max(ke_ev)),
        "z_mean_mm": float(np.mean(z_mm)),
        "z_max_mm": float(np.max(z_mm)),
        "r_mean_mm": float(np.mean(r_mm)),
        "r_max_mm": float(np.max(r_mm)),
        "vz_mean_m_per_s": float(np.mean(vz)),
        "collision_mean": float(np.mean(collisions)),
        "collision_max": float(np.max(collisions)),
    }
    for prefix, values in (
        ("z_mm", z_mm),
        ("r_mm", r_mm),
        ("teff_k", teff_k),
        ("ke_ev", ke_ev),
        ("vz_m_per_s", vz),
        ("collision_count", collisions),
    ):
        for key, value in _percentiles(values, (5, 10, 50, 90, 95, 99)).items():
            row[f"{prefix}_{key}"] = value
    return row


def _load_index(case_dir: Path) -> list[tuple[int, float, Path]]:
    rows: list[tuple[int, float, Path]] = []
    index_path = case_dir / "snapshots_index.csv"
    with index_path.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            macro_text, time_text, _count_text, storage_ref = line.strip().split(",")
            rows.append((int(macro_text), float(time_text), case_dir / storage_ref))
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(key, "")) for key in keys) + "\n")


def _make_plots(case_dir: Path, out_dir: Path, rows: list[dict[str, float | int]], selected: list[tuple[int, float, Path]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times_ms = np.asarray([float(row.get("time_ms", np.nan)) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(times_ms, [row.get("active_macro_particles", np.nan) for row in rows], marker="o")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("active weighted ion packs")
    axes[0, 0].set_title("Active particle pool")

    axes[0, 1].plot(times_ms, [row.get("z_mm_p50", np.nan) for row in rows], label="p50")
    axes[0, 1].plot(times_ms, [row.get("z_mm_p95", np.nan) for row in rows], label="p95")
    axes[0, 1].plot(times_ms, [row.get("z_max_mm", np.nan) for row in rows], label="max")
    axes[0, 1].set_xlabel("time (ms)")
    axes[0, 1].set_ylabel("z (mm)")
    axes[0, 1].set_title("Axial motion")
    axes[0, 1].legend()

    axes[1, 0].plot(times_ms, [row.get("ke_ev_p50", np.nan) for row in rows], label="p50")
    axes[1, 0].plot(times_ms, [row.get("ke_ev_p95", np.nan) for row in rows], label="p95")
    axes[1, 0].plot(times_ms, [row.get("ke_ev_p99", np.nan) for row in rows], label="p99")
    axes[1, 0].set_xlabel("time (ms)")
    axes[1, 0].set_ylabel("KE (eV)")
    axes[1, 0].set_title("Kinetic energy")
    axes[1, 0].legend()

    axes[1, 1].plot(times_ms, [row.get("teff_k_p50", np.nan) for row in rows], label="p50")
    axes[1, 1].plot(times_ms, [row.get("teff_k_p95", np.nan) for row in rows], label="p95")
    axes[1, 1].plot(times_ms, [row.get("teff_k_p99", np.nan) for row in rows], label="p99")
    axes[1, 1].set_xlabel("time (ms)")
    axes[1, 1].set_ylabel("internal effective T (K)")
    axes[1, 1].set_title("Internal temperature")
    axes[1, 1].legend()

    fig.savefig(out_dir / "snapshot_time_trends.png", dpi=180)
    plt.close(fig)

    selected_macros = {1000, 10000, 20000, 30000, 39000}
    selected_arrays = []
    for macro_step, time_s, path in selected:
        if macro_step in selected_macros:
            selected_arrays.append((macro_step, time_s, np.load(path)))
    if selected_arrays:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for macro_step, time_s, data in selected_arrays:
            if data.size == 0:
                continue
            label = f"{time_s * 1e3:.2f} ms"
            axes[0].hist(data[:, COL_KE_EV], bins=80, histtype="step", density=True, label=label)
            axes[1].hist(data[:, COL_TEFF_K], bins=80, histtype="step", density=True, label=label)
            axes[2].hist(data[:, COL_Z_M] * 1e3, bins=80, histtype="step", density=True, label=label)
        axes[0].set_xlabel("KE (eV)")
        axes[0].set_ylabel("density")
        axes[0].set_title("KE distribution")
        axes[1].set_xlabel("internal effective T (K)")
        axes[1].set_title("Internal temperature distribution")
        axes[2].set_xlabel("z (mm)")
        axes[2].set_title("Axial distribution")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.savefig(out_dir / "snapshot_distribution_overlay.png", dpi=180)
        plt.close(fig)

    last_macro, last_time, last_path = selected[-1]
    last = np.load(last_path)
    if last.size:
        z_mm = last[:, COL_Z_M] * 1e3
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        sc = axes[0].scatter(z_mm, last[:, COL_R_M] * 1e3, c=last[:, COL_KE_EV], s=2, cmap="viridis", vmin=0)
        axes[0].set_xlabel("z (mm)")
        axes[0].set_ylabel("r (mm)")
        axes[0].set_title(f"Last snapshot KE, {last_time * 1e3:.2f} ms")
        fig.colorbar(sc, ax=axes[0], label="KE (eV)")

        sc = axes[1].scatter(z_mm, last[:, COL_R_M] * 1e3, c=last[:, COL_TEFF_K], s=2, cmap="magma", vmin=0)
        axes[1].set_xlabel("z (mm)")
        axes[1].set_ylabel("r (mm)")
        axes[1].set_title("Internal T by position")
        fig.colorbar(sc, ax=axes[1], label="K")

        axes[2].hist2d(z_mm, last[:, COL_VZ], bins=(90, 90), cmap="Blues")
        axes[2].set_xlabel("z (mm)")
        axes[2].set_ylabel("vz (m/s)")
        axes[2].set_title("Axial velocity phase space")
        fig.savefig(out_dir / "last_snapshot_position_phase.png", dpi=180)
        plt.close(fig)


def _zone_rows(macro_step: int, time_s: float, data: np.ndarray, detector_z_mm: float) -> list[dict[str, float | int | str]]:
    z_mm = data[:, COL_Z_M] * 1.0e3 if data.size else np.asarray([])
    zones = (
        ("4.5-20mm", 4.5, 20.0),
        ("20-40mm", 20.0, 40.0),
        ("40-64mm", 40.0, detector_z_mm - 1.0),
        ("near_exit_64-65mm", detector_z_mm - 1.0, detector_z_mm),
        ("at_or_past_exit_ge65mm", detector_z_mm, np.inf),
    )
    rows: list[dict[str, float | int | str]] = []
    for name, lo, hi in zones:
        mask = (z_mm >= lo) & (z_mm < hi)
        zone_data = data[mask]
        if zone_data.size == 0:
            rows.append(
                {
                    "macro_step": int(macro_step),
                    "time_s": float(time_s),
                    "zone": name,
                    "active_macro_particles": 0,
                }
            )
            continue
        teff_k = zone_data[:, COL_TEFF_K]
        ke_ev = zone_data[:, COL_KE_EV]
        vz = zone_data[:, COL_VZ]
        row: dict[str, float | int | str] = {
            "macro_step": int(macro_step),
            "time_s": float(time_s),
            "zone": name,
            "active_macro_particles": int(zone_data.shape[0]),
            "represented_real_ions": float(np.sum(zone_data[:, COL_WEIGHT])),
            "z_min_mm": float(np.min(zone_data[:, COL_Z_M] * 1.0e3)),
            "z_max_mm": float(np.max(zone_data[:, COL_Z_M] * 1.0e3)),
            "teff_mean_k": float(np.mean(teff_k)),
            "teff_max_k": float(np.max(teff_k)),
            "ke_mean_ev": float(np.mean(ke_ev)),
            "ke_max_ev": float(np.max(ke_ev)),
            "vz_mean_m_per_s": float(np.mean(vz)),
        }
        for prefix, values in (("teff_k", teff_k), ("ke_ev", ke_ev), ("vz_m_per_s", vz)):
            for key, value in _percentiles(values, (5, 50, 95, 99)).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--detector-z-mm", type=float, default=65.0)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    out_dir = case_dir / "snapshot_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots = _load_index(case_dir)
    rows = []
    for macro_step, time_s, path in snapshots:
        rows.append(_summary_row(macro_step, time_s, np.load(path)))

    _write_csv(out_dir / "snapshot_ke_temp_summary.csv", rows)
    _make_plots(case_dir, out_dir, rows, snapshots)

    last_macro, last_time_s, last_path = snapshots[-1]
    last = np.load(last_path)
    zone_summary = _zone_rows(last_macro, last_time_s, last, args.detector_z_mm)
    _write_csv(out_dir / "last_snapshot_zone_summary.csv", zone_summary)
    z_mm = last[:, COL_Z_M] * 1.0e3 if last.size else np.asarray([])
    weights = last[:, COL_WEIGHT] if last.size else np.asarray([])
    near_exit = z_mm >= (args.detector_z_mm - 1.0)
    at_exit = z_mm >= args.detector_z_mm
    transmission_estimate = {
        "note": "Incomplete run. This is only the active-particle snapshot estimate; terminal z_exit rows are unavailable because the run did not finish.",
        "last_macro_step": int(last_macro),
        "last_time_s": float(last_time_s),
        "active_macro_particles": int(last.shape[0]),
        "active_near_exit_z_ge_64mm_count": int(np.count_nonzero(near_exit)),
        "active_at_or_past_detector_z_ge_65mm_count": int(np.count_nonzero(at_exit)),
        "represented_real_ions_near_exit_z_ge_64mm": float(np.sum(weights[near_exit])) if weights.size else 0.0,
        "represented_real_ions_at_or_past_detector_z_ge_65mm": float(np.sum(weights[at_exit])) if weights.size else 0.0,
    }
    with (out_dir / "snapshot_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "case_dir": str(case_dir),
                "snapshot_count": len(snapshots),
                "transmission_estimate": transmission_estimate,
                "last_snapshot_summary": rows[-1],
                "last_snapshot_zone_summary": zone_summary,
            },
            handle,
            indent=2,
        )

    print(json.dumps({"out_dir": str(out_dir), "last_snapshot": rows[-1], "transmission_estimate": transmission_estimate}, indent=2))


if __name__ == "__main__":
    main()
