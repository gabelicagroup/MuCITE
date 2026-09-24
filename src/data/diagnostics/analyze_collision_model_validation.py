"""Summarize collision-model validation runs and extract matched exit tracks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze_rf_sweep_multimetric_per_ms import (
    ELEMENTARY_CHARGE_C,
    weighted_quantile,
)


CASE_PATTERN = re.compile(r"CM_RF(?P<rf>\d+)V_(?P<model>explicit|hybrid)")
EVENT_COLUMNS = [
    "track_id",
    "status_code",
    "status",
    "event_time_s",
    "birth_time_s",
    "tof_s",
    "x_m",
    "y_m",
    "z_m",
    "r_m",
    "vx_m_per_s",
    "vy_m_per_s",
    "vz_m_per_s",
    "speed_m_per_s",
    "ke_ev",
    "internal_temperature_k",
    "represented_real_ions",
    "parent_real_ions_remaining",
    "fragmented_real_ions_represented",
    "collision_count_per_ion",
]
OUTCOME_STATUSES = ("electrode_hit", "domain_out", "radial_out")
SNAPSHOT_COLUMNS = [
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
]


def _current_na(weight: float, charge_state: int, duration_s: float) -> float:
    if duration_s <= 0.0:
        return math.nan
    return weight * abs(charge_state) * ELEMENTARY_CHARGE_C / duration_s * 1.0e9


def _weighted_median(events: pd.DataFrame, column: str) -> float:
    return weighted_quantile(
        events[column].to_numpy(dtype=np.float64),
        events["represented_real_ions"].to_numpy(dtype=np.float64),
        0.50,
    )


def _exit_current_cv_percent(
    exits: pd.DataFrame,
    start_s: float,
    end_s: float,
    charge_state: int,
    bin_width_s: float,
) -> float:
    currents: list[float] = []
    left = start_s
    while left < end_s - 1.0e-15:
        right = min(left + bin_width_s, end_s)
        if right >= end_s - 1.0e-15:
            mask = exits["event_time_s"].ge(left) & exits["event_time_s"].le(right + 1.0e-15)
        else:
            mask = exits["event_time_s"].ge(left) & exits["event_time_s"].lt(right)
        weight = float(exits.loc[mask, "represented_real_ions"].sum())
        currents.append(_current_na(weight, charge_state, right - left))
        left = right
    values = np.asarray(currents, dtype=np.float64)
    mean = float(np.mean(values)) if values.size else math.nan
    if values.size < 2 or not math.isfinite(mean) or mean <= 0.0:
        return math.nan
    return 100.0 * float(np.std(values, ddof=1)) / mean


def _read_case(case_dir: Path, stable_start_ms: float, stability_bin_ms: float) -> tuple[dict[str, object], dict[str, object]]:
    match = CASE_PATTERN.search(case_dir.name)
    if match is None:
        raise ValueError(f"Cannot parse collision-model case name: {case_dir.name}")

    with (case_dir / "simulation_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)

    config = summary["config"]
    source = summary["source_runtime"]
    template = summary["template"]
    performance = summary["performance"]
    result = summary["result"]
    simulated_time_s = float(performance["simulated_time_s"])
    requested_time_s = float(config["total_time_s"])
    stable_start_s = min(stable_start_ms * 1.0e-3, simulated_time_s)
    stable_duration_s = simulated_time_s - stable_start_s
    source_current_na = float(source["ion_current_a"]) * 1.0e9
    charge_state = int(template["charge_state"])

    events = pd.read_csv(case_dir / "terminal_events.csv", usecols=EVENT_COLUMNS)
    events = events.loc[np.isfinite(events["event_time_s"])].copy()
    stable_events = events.loc[
        events["event_time_s"].ge(stable_start_s)
        & events["event_time_s"].le(simulated_time_s + 1.0e-15)
    ].copy()
    exits = stable_events.loc[stable_events["status"].eq("z_exit")].copy()
    exit_weight = float(exits["represented_real_ions"].sum())
    exit_current_na = _current_na(exit_weight, charge_state, stable_duration_s)
    fragment_weight = float(exits["fragmented_real_ions_represented"].sum())

    row: dict[str, object] = {
        "case": case_dir.name,
        "collision_model": str(config["collision_model"]),
        "rf_vpeak": float(config["rf_peak_voltage_v"]),
        "source_current_nA": source_current_na,
        "stable_window_start_ms": stable_start_s * 1.0e3,
        "stable_window_end_ms": simulated_time_s * 1.0e3,
        "stability_bin_width_ms": stability_bin_ms,
        "completed_requested_time": simulated_time_s >= requested_time_s - 1.0e-12,
        "termination_reason": str(result["termination_reason"]),
        "stable_exit_event_count": int(len(exits)),
        "stable_exit_ion_current_nA": exit_current_na,
        "stable_transmission_percent": 100.0 * exit_current_na / source_current_na,
        "stable_exit_current_cv_percent": _exit_current_cv_percent(
            exits,
            stable_start_s,
            simulated_time_s,
            charge_state,
            stability_bin_ms * 1.0e-3,
        ),
        "stable_ke_eV_weighted_median": _weighted_median(exits, "ke_ev"),
        "stable_temperature_K_weighted_median": _weighted_median(exits, "internal_temperature_k"),
        "stable_fragmentation_percent_of_exit": (
            100.0 * fragment_weight / exit_weight if exit_weight > 0.0 else math.nan
        ),
    }
    accounted_weight = exit_weight
    for status in OUTCOME_STATUSES:
        weight = float(
            stable_events.loc[stable_events["status"].eq(status), "represented_real_ions"].sum()
        )
        current_na = _current_na(weight, charge_state, stable_duration_s)
        row[f"stable_{status}_percent_of_input"] = 100.0 * current_na / source_current_na
        accounted_weight += weight
    row["stable_terminal_accounted_percent_of_input"] = (
        100.0 * _current_na(accounted_weight, charge_state, stable_duration_s) / source_current_na
    )
    wall_time_s = float(performance.get("run_wall_time_s", math.nan))
    row["total_runtime_s"] = wall_time_s
    row["total_runtime_h"] = wall_time_s / 3600.0

    all_exits = events.loc[events["status"].eq("z_exit")].copy()
    payload = {
        "case_dir": case_dir,
        "row": row,
        "events": events,
        "all_exits": all_exits,
    }
    return row, payload


def _select_matched_tracks(case_payloads: list[dict[str, object]], count: int) -> list[tuple[str, int]]:
    by_rf: dict[float, list[dict[str, object]]] = {}
    for payload in case_payloads:
        rf = float(payload["row"]["rf_vpeak"])
        by_rf.setdefault(rf, []).append(payload)

    for rf in sorted(by_rf, reverse=True):
        payloads = by_rf[rf]
        if len(payloads) < 2:
            continue
        id_sets = [set(payload["all_exits"]["track_id"].astype(int)) for payload in payloads]
        common_ids = set.intersection(*id_sets) if id_sets else set()
        if len(common_ids) < count:
            continue

        energy_maps = [
            payload["all_exits"].set_index("track_id")["ke_ev"].to_dict()
            for payload in payloads
        ]
        scores: list[tuple[int, float]] = []
        for track_id in common_ids:
            energies = [float(energy_map[track_id]) for energy_map in energy_maps]
            scores.append((int(track_id), float(np.mean(energies))))
        scores.sort(key=lambda item: item[1])

        labels = ("lower-exit-KE", "median-exit-KE", "upper-exit-KE")
        quantiles = np.linspace(0.25, 0.75, count)
        selected: list[tuple[str, int]] = []
        used: set[int] = set()
        for label, quantile in zip(labels, quantiles):
            target = int(round(float(quantile) * (len(scores) - 1)))
            candidates = sorted(range(len(scores)), key=lambda index: abs(index - target))
            index = next(index for index in candidates if scores[index][0] not in used)
            track_id = scores[index][0]
            selected.append((label, track_id))
            used.add(track_id)
        return selected
    return []


def _extract_tracks(
    case_payloads: list[dict[str, object]],
    selected: list[tuple[str, int]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for payload in case_payloads:
        all_exits = payload["all_exits"]
        model = str(payload["row"]["collision_model"])
        rf_vpeak = float(payload["row"]["rf_vpeak"])
        case = str(payload["row"]["case"])
        case_dir = Path(payload["case_dir"])
        relevant_ids = [track_id for _label, track_id in selected if track_id in set(all_exits["track_id"].astype(int))]
        snapshot_rows: list[pd.DataFrame] = []
        if relevant_ids:
            index = pd.read_csv(case_dir / "snapshots_index.csv")
            for record in index.itertuples(index=False):
                snapshot = np.load(case_dir / str(record.storage_ref))
                if snapshot.size == 0:
                    continue
                ids = snapshot[:, SNAPSHOT_COLUMNS.index("track_id")].astype(np.int64)
                keep = np.isin(ids, relevant_ids)
                if not np.any(keep):
                    continue
                frame = pd.DataFrame(snapshot[keep], columns=SNAPSHOT_COLUMNS)
                frame["track_id"] = frame["track_id"].astype(np.int64)
                frame["slot_id"] = frame["slot_id"].astype(np.int64)
                frame.insert(0, "time_s", float(record.time_s))
                frame.insert(0, "macro_step", int(record.macro_step))
                snapshot_rows.append(frame)
        snapshots = pd.concat(snapshot_rows, ignore_index=True) if snapshot_rows else pd.DataFrame()

        for label, track_id in selected:
            matched_event = all_exits.loc[all_exits["track_id"].eq(track_id)]
            if matched_event.empty:
                continue
            event = matched_event.iloc[0]
            birth_time_s = float(event["birth_time_s"])
            track = snapshots.loc[snapshots["track_id"].eq(track_id)].copy()
            track = track.sort_values("time_s")
            track["sample_origin"] = "snapshot"

            terminal = pd.DataFrame(
                [{
                    "macro_step": math.nan,
                    "time_s": float(event["event_time_s"]),
                    "r_m": float(event["r_m"]),
                    "z_m": float(event["z_m"]),
                    "teff_k": float(event["internal_temperature_k"]),
                    "kinetic_energy_j": float(event["ke_ev"]) * ELEMENTARY_CHARGE_C,
                    "x_m": float(event["x_m"]),
                    "y_m": float(event["y_m"]),
                    "vx_m_per_s": float(event["vx_m_per_s"]),
                    "vy_m_per_s": float(event["vy_m_per_s"]),
                    "vz_m_per_s": float(event["vz_m_per_s"]),
                    "speed_m_per_s": float(event["speed_m_per_s"]),
                    "kinetic_energy_ev": float(event["ke_ev"]),
                    "represented_real_ions": float(event["represented_real_ions"]),
                    "parent_real_ions_remaining": float(event["parent_real_ions_remaining"]),
                    "fragmented_real_ions_represented": float(event["fragmented_real_ions_represented"]),
                    "track_id": int(track_id),
                    "slot_id": math.nan,
                    "status_code": int(event["status_code"]),
                    "collision_count_per_ion": float(event["collision_count_per_ion"]),
                    "sample_origin": "terminal_event",
                }]
            )
            if track.empty or float(track["time_s"].max()) < float(event["event_time_s"]) - 1.0e-15:
                track = pd.concat([track, terminal], ignore_index=True)

            track.insert(0, "representative_group", label)
            track.insert(0, "collision_model", model)
            track.insert(0, "rf_vpeak", rf_vpeak)
            track.insert(0, "case", case)
            track["time_since_birth_us"] = (track["time_s"] - birth_time_s) * 1.0e6
            track["r_mm"] = track["r_m"] * 1.0e3
            track["z_mm"] = track["z_m"] * 1.0e3
            track["fragmentation_percent"] = np.where(
                track["represented_real_ions"] > 0.0,
                100.0 * track["fragmented_real_ions_represented"] / track["represented_real_ions"],
                np.nan,
            )
            rows.append(track)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["representative_group", "track_id", "collision_model", "time_s"]
    )


def _plot_tracks(tracks: pd.DataFrame, output: Path, dpi: int) -> None:
    if tracks.empty:
        return
    colors = {
        "lower-exit-KE": "#0072B2",
        "median-exit-KE": "#009E73",
        "upper-exit-KE": "#D55E00",
    }
    linestyles = {"explicit": "-", "hybrid-langevin": "--"}
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 10.2), sharex=True, constrained_layout=True)
    for (label, model, track_id), subset in tracks.groupby(
        ["representative_group", "collision_model", "track_id"], sort=False
    ):
        subset = subset.sort_values("time_s")
        line_label = f"{label}, {model}, track {int(track_id)}"
        style = linestyles.get(str(model), "-")
        color = colors.get(str(label), "#444444")
        axes[0].plot(subset["z_mm"], subset["r_mm"], style, color=color, lw=1.2, label=line_label)
        axes[1].plot(subset["z_mm"], subset["teff_k"], style, color=color, lw=1.2)
        axes[2].plot(subset["z_mm"], subset["kinetic_energy_ev"], style, color=color, lw=1.2)
        axes[3].plot(subset["z_mm"], subset["fragmentation_percent"], style, color=color, lw=1.2)

    axes[0].set_ylabel("Radial position (mm)")
    axes[1].set_ylabel(r"$T_{eff}$ (K)")
    axes[2].set_ylabel("Kinetic energy (eV)")
    axes[3].set_ylabel("Dissociated weight (%)")
    axes[3].set_xlabel("Axial position z (mm)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    for index, axis in enumerate(axes):
        axis.text(0.01, 0.97, f"({chr(97 + index)})", transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(True, color="#d9d9d9", linewidth=0.55, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stable-start-ms", type=float, default=1.0)
    parser.add_argument("--stability-bin-ms", type=float, default=0.5)
    parser.add_argument("--trajectory-count", type=int, default=3)
    parser.add_argument("--output-summary-csv", type=Path)
    parser.add_argument("--output-trajectories-csv", type=Path)
    parser.add_argument("--output-trajectories-plot", type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    root = args.root.resolve()
    case_dirs = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and CASE_PATTERN.search(path.name)
            and (path / "terminal_events.csv").exists()
        ),
        key=lambda path: (
            int(CASE_PATTERN.search(path.name).group("rf")),
            CASE_PATTERN.search(path.name).group("model"),
        ),
    )
    if not case_dirs:
        raise FileNotFoundError(f"No collision-model cases found under {root}")

    summary_rows: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []
    for case_dir in case_dirs:
        row, payload = _read_case(case_dir, args.stable_start_ms, args.stability_bin_ms)
        summary_rows.append(row)
        payloads.append(payload)

    summary = pd.DataFrame(summary_rows).sort_values(["rf_vpeak", "collision_model"])
    selected = _select_matched_tracks(payloads, args.trajectory_count)
    tracks = _extract_tracks(payloads, selected)

    summary_output = (args.output_summary_csv or root / "01_collision_model_stable_summary.csv").resolve()
    trajectory_output = (
        args.output_trajectories_csv
        or root / "01_collision_model_representative_exit_trajectories.csv"
    ).resolve()
    plot_output = (
        args.output_trajectories_plot
        or root / "01_collision_model_representative_exit_trajectories.png"
    ).resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False, float_format="%.12g")
    tracks.to_csv(trajectory_output, index=False, float_format="%.12g")
    _plot_tracks(tracks, plot_output, args.dpi)

    print(f"cases={len(summary)}")
    print(f"selected_tracks={selected}")
    print(f"summary_csv={summary_output}")
    print(f"trajectories_csv={trajectory_output}")
    print(f"trajectory_plot={plot_output}")


if __name__ == "__main__":
    main()
