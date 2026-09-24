"""Generate diagnostic plots for completed continuous-current runs.

This utility is intentionally lightweight:

- `r-z` cloud: final active-particle density with terminal-status overlays
- bundle evolution: true snapshot envelope when snapshots exist, otherwise a
  proxy plot from `macro_history` (`mean z` and active count vs time)
- `z-KE` and `r-KE` phase plots: final active-particle phase-space density

The script accepts either one case directory or a root directory that contains
multiple case subdirectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATUS_COLORS = {
    "z_exit": "#1f77b4",
    "electrode_hit": "#d62728",
    "domain_out": "#ff7f0e",
    "radial_out": "#9467bd",
    "fragmented": "#8c564b",
}


def _load_case_paths(path: Path) -> list[Path]:
    path = path.resolve()
    if (path / "simulation_summary.json").exists() and (path / "final_particles.csv").exists():
        return [path]
    cases = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "simulation_summary.json").exists() and (child / "final_particles.csv").exists():
            cases.append(child)
    if not cases:
        raise FileNotFoundError(f"No completed case directories found under {path}.")
    return cases


def _active_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["final_status"] == "active"].copy()


def _terminal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["final_status"] != "active"].copy()


def _status_groups(frame: pd.DataFrame, statuses: Iterable[str]) -> Iterable[tuple[str, pd.DataFrame]]:
    for status in statuses:
        group = frame.loc[frame["final_status"] == status]
        if not group.empty:
            yield status, group


def _plot_rz_cloud(case_dir: Path, frame: pd.DataFrame, summary: dict) -> Path:
    active = _active_frame(frame)
    terminal = _terminal_frame(frame)

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    if not active.empty:
        hb = ax.hexbin(
            active["final_z_m"].to_numpy() * 1.0e3,
            active["final_r_m"].to_numpy() * 1.0e3,
            gridsize=120,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        cbar = fig.colorbar(hb, ax=ax, pad=0.02)
        cbar.set_label("Active weighted ion pack density (log10 count)")
    for status, group in _status_groups(terminal, ("z_exit", "electrode_hit", "domain_out", "radial_out", "fragmented")):
        ax.scatter(
            group["final_z_m"].to_numpy() * 1.0e3,
            group["final_r_m"].to_numpy() * 1.0e3,
            s=12,
            alpha=0.85,
            color=STATUS_COLORS.get(status, "#444444"),
            label=status,
        )

    detector_z_mm = float(summary["config"].get("detector_z_m") or summary["config"]["domain_length_m"]) * 1.0e3
    ax.axvline(detector_z_mm, color="#333333", linestyle="--", linewidth=1.0, label="detector z")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title(f"{case_dir.name}: final r-z cloud")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    if not terminal.empty:
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    output = case_dir / "rz_cloud_final.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _load_snapshot_envelope(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    index_path = case_dir / "snapshots_index.csv"
    if not index_path.exists():
        return None

    index = pd.read_csv(index_path)
    if index.empty:
        return None

    time_us = []
    z_p05_mm = []
    z_p50_mm = []
    z_p95_mm = []
    for _, row in index.iterrows():
        storage_ref = str(row["storage_ref"])
        if storage_ref.endswith(".npy"):
            snapshot = np.load(case_dir / storage_ref)
        else:
            continue
        if snapshot.size == 0:
            continue
        z_mm = snapshot[:, 1] * 1.0e3
        time_us.append(float(row["time_s"]) * 1.0e6)
        z_p05_mm.append(float(np.percentile(z_mm, 5)))
        z_p50_mm.append(float(np.percentile(z_mm, 50)))
        z_p95_mm.append(float(np.percentile(z_mm, 95)))
    if not time_us:
        return None
    return (
        np.asarray(time_us, dtype=float),
        np.asarray(z_p05_mm, dtype=float),
        np.asarray(z_p50_mm, dtype=float),
        np.asarray(z_p95_mm, dtype=float),
    )


def _plot_bundle_evolution(case_dir: Path, summary: dict) -> Path:
    macro_history = pd.DataFrame(summary["result"].get("macro_history", []))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.8, 7.2), constrained_layout=True)

    snapshot_envelope = _load_snapshot_envelope(case_dir)
    if snapshot_envelope is not None:
        time_us, z_p05_mm, z_p50_mm, z_p95_mm = snapshot_envelope
        ax0.fill_between(time_us, z_p05_mm, z_p95_mm, color="#9ecae1", alpha=0.5, label="z p05-p95")
        ax0.plot(time_us, z_p50_mm, color="#08519c", linewidth=1.6, label="z p50")
        ax0.set_title(f"{case_dir.name}: bundle envelope over time")
        ax0.legend(loc="best", fontsize=8)
    else:
        time_us = macro_history["time_s"].to_numpy(dtype=float) * 1.0e6
        mean_z_mm = macro_history["mean_axial_position_m"].to_numpy(dtype=float) * 1.0e3
        ax0.plot(time_us, mean_z_mm, color="#08519c", linewidth=1.6)
        ax0.set_title(f"{case_dir.name}: bundle evolution proxy (no snapshot exports)")
        ax0.text(
            0.02,
            0.95,
            "Proxy: mean z only; true percentile envelope needs --export-every snapshots.",
            transform=ax0.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#bbbbbb"),
        )

    ax0.set_xlabel("time [us]")
    ax0.set_ylabel("axial position [mm]")
    ax0.set_xlim(left=0.0)
    ax0.set_ylim(bottom=0.0)

    if not macro_history.empty:
        time_us = macro_history["time_s"].to_numpy(dtype=float) * 1.0e6
        active_count = macro_history["ion_count"].to_numpy(dtype=float)
        terminal_count = macro_history["terminal_count"].to_numpy(dtype=float)
        ax1.plot(time_us, active_count, color="#2ca02c", linewidth=1.4, label="active weighted ion packs")
        ax1.plot(time_us, terminal_count, color="#d62728", linewidth=1.2, label="terminal events (cumulative)")
        ax1.legend(loc="best", fontsize=8)
    ax1.set_xlabel("time [us]")
    ax1.set_ylabel("count")
    ax1.set_xlim(left=0.0)
    ax1.set_ylim(bottom=0.0)

    output = case_dir / "bundle_evolution.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _plot_phase_space(case_dir: Path, frame: pd.DataFrame) -> Path:
    active = _active_frame(frame)
    if active.empty:
        raise ValueError(f"{case_dir} has no active particles to plot.")

    z_mm = active["final_z_m"].to_numpy(dtype=float) * 1.0e3
    r_mm = active["final_r_m"].to_numpy(dtype=float) * 1.0e3
    ke_ev = active["final_ke_ev"].to_numpy(dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    hb0 = ax0.hexbin(z_mm, ke_ev, gridsize=110, mincnt=1, bins="log", cmap="plasma")
    cbar0 = fig.colorbar(hb0, ax=ax0, pad=0.02)
    cbar0.set_label("log10 count")
    ax0.set_xlabel("z [mm]")
    ax0.set_ylabel("KE [eV]")
    ax0.set_title("z-KE phase space")
    ax0.set_xlim(left=0.0)
    ax0.set_ylim(bottom=0.0)

    hb1 = ax1.hexbin(r_mm, ke_ev, gridsize=110, mincnt=1, bins="log", cmap="magma")
    cbar1 = fig.colorbar(hb1, ax=ax1, pad=0.02)
    cbar1.set_label("log10 count")
    ax1.set_xlabel("r [mm]")
    ax1.set_ylabel("KE [eV]")
    ax1.set_title("r-KE phase space")
    ax1.set_xlim(left=0.0)
    ax1.set_ylim(bottom=0.0)

    fig.suptitle(case_dir.name, fontsize=11)
    output = case_dir / "phase_space_final.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _write_index(root: Path, records: list[dict[str, str]]) -> Path:
    index_path = root / "diagnostic_plots_index.json"
    index_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for completed continuous-current runs.")
    parser.add_argument("path", type=Path, help="Case directory or root directory containing completed cases.")
    args = parser.parse_args()

    cases = _load_case_paths(args.path)
    records = []
    for case_dir in cases:
        summary = json.loads((case_dir / "simulation_summary.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(case_dir / "final_particles.csv")
        rz_path = _plot_rz_cloud(case_dir, frame, summary)
        envelope_path = _plot_bundle_evolution(case_dir, summary)
        phase_path = _plot_phase_space(case_dir, frame)
        records.append(
            {
                "case": case_dir.name,
                "rz_cloud_final": str(rz_path),
                "bundle_evolution": str(envelope_path),
                "phase_space_final": str(phase_path),
            }
        )

    index_path = _write_index(cases[0].parent if len(cases) > 1 else cases[0], records)
    print(f"Wrote plots for {len(records)} case(s).")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
