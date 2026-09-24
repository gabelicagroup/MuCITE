"""Summarize RF-sweep validation runs and extract matched exit trajectories."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analyze_rf_sweep_multimetric_per_ms import (
    ELEMENTARY_CHARGE_C,
    load_case,
)


RF_PATTERN = re.compile(r"RF_RF(?P<rf>\d+)V", re.IGNORECASE)
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


def _rf_vpeak(case_dir: Path) -> int:
    match = RF_PATTERN.search(case_dir.name)
    if match is None:
        raise ValueError(f"Cannot parse RF amplitude from {case_dir.name}")
    return int(match.group("rf"))


def _exit_current_cv_percent(
    exits: pd.DataFrame,
    start_s: float,
    end_s: float,
    source_current_na: float,
    bin_width_s: float,
) -> float:
    transmissions: list[float] = []
    left = start_s
    while left < end_s - 1.0e-15:
        right = min(left + bin_width_s, end_s)
        if right >= end_s - 1.0e-15:
            mask = exits["event_time_s"].ge(left) & exits["event_time_s"].le(right + 1.0e-15)
        else:
            mask = exits["event_time_s"].ge(left) & exits["event_time_s"].lt(right)
        weight = float(exits.loc[mask, "represented_real_ions"].sum())
        current_na = weight * ELEMENTARY_CHARGE_C / (right - left) * 1.0e9
        transmissions.append(100.0 * current_na / source_current_na)
        left = right
    values = np.asarray(transmissions, dtype=np.float64)
    mean = float(np.mean(values)) if values.size else math.nan
    if values.size < 2 or not math.isfinite(mean) or mean <= 0.0:
        return math.nan
    return 100.0 * float(np.std(values, ddof=1)) / mean


def _case_payload(case_dir: Path, stability_bin_ms: float) -> tuple[dict[str, object], dict[str, object]]:
    _rows, legacy = load_case(case_dir, stability_bin_ms)
    with (case_dir / "simulation_summary.json").open(encoding="utf-8") as stream:
        source_summary = json.load(stream)

    simulated_time_s = float(source_summary["performance"]["simulated_time_s"])
    requested_time_s = float(source_summary["config"]["total_time_s"])
    stable_start_s = min(1.0e-3, simulated_time_s)
    source_current_na = float(legacy["source_current_nA"])
    diagnostics = source_summary["result"]["terminal_event_diagnostics"]
    events = pd.read_csv(case_dir / "terminal_events.csv", usecols=EVENT_COLUMNS)
    events = events.loc[np.isfinite(events["event_time_s"])].copy()
    stable_events = events.loc[
        events["event_time_s"].ge(stable_start_s)
        & events["event_time_s"].le(simulated_time_s + 1.0e-15)
    ]
    stable_exits = stable_events.loc[stable_events["status"].eq("z_exit")]
    wall_time_s = float(source_summary["performance"].get("run_wall_time_s", math.nan))

    row: dict[str, object] = {
        "case": case_dir.name,
        "rf_vpeak": float(legacy["rf_vpeak"]),
        "source_current_nA": source_current_na,
        "collision_model": str(source_summary["config"]["collision_model"]),
        "stable_window_start_ms": stable_start_s * 1.0e3,
        "stable_window_end_ms": simulated_time_s * 1.0e3,
        "stability_bin_width_ms": stability_bin_ms,
        "completed_requested_time": simulated_time_s >= requested_time_s - 1.0e-12,
        "termination_reason": str(source_summary["result"]["termination_reason"]),
        "terminal_event_rows_capped": bool(diagnostics["max_rows_reached"]),
        "stable_exit_event_count": int(len(stable_exits)),
        "stable_exit_ion_current_nA": float(legacy["stable_ion_current_nA"]),
        "stable_transmission_percent": float(legacy["stable_transmission_percent"]),
        "stable_exit_current_cv_percent": _exit_current_cv_percent(
            stable_exits,
            stable_start_s,
            simulated_time_s,
            source_current_na,
            stability_bin_ms * 1.0e-3,
        ),
        "stable_ke_eV_weighted_median": float(legacy["stable_ke_ev_p50_weighted"]),
        "stable_temperature_K_weighted_median": float(
            legacy["stable_temperature_k_p50_weighted"]
        ),
        "stable_exit_beam_radius_mm_weighted_median": float(
            legacy["stable_exit_beam_radius_mm_p50_weighted"]
        ),
        "stable_exit_beam_radius_mm_weighted_p95": float(
            legacy["stable_exit_beam_radius_mm_p95_weighted"]
        ),
        "stable_fragmentation_percent_of_exit": float(
            legacy["stable_fragmentation_percent_of_exit"]
        ),
        "stable_electrode_hit_percent_of_input": float(
            legacy["stable_electrode_hit_percent_of_input"]
        ),
        "stable_domain_out_percent_of_input": float(
            legacy["stable_domain_out_percent_of_input"]
        ),
        "stable_radial_out_percent_of_input": float(
            legacy["stable_radial_out_percent_of_input"]
        ),
        "stable_terminal_accounted_percent_of_input": float(
            legacy["stable_terminal_accounted_percent_of_input"]
        ),
        "total_runtime_s": wall_time_s,
        "total_runtime_h": wall_time_s / 3600.0,
    }
    payload = {
        "case_dir": case_dir,
        "row": row,
        "all_exits": events.loc[events["status"].eq("z_exit")].copy(),
    }
    return row, payload


def _select_cases(payloads: list[dict[str, object]], targets: tuple[float, ...]) -> list[dict[str, object]]:
    remaining = list(payloads)
    selected: list[dict[str, object]] = []
    for target in targets:
        choice = min(remaining, key=lambda item: abs(float(item["row"]["rf_vpeak"]) - target))
        selected.append(choice)
        remaining.remove(choice)
    return sorted(selected, key=lambda item: float(item["row"]["rf_vpeak"]))


def _select_matched_tracks(
    payloads: list[dict[str, object]],
    count: int,
) -> list[tuple[str, int]]:
    id_sets = [set(payload["all_exits"]["track_id"].astype(int)) for payload in payloads]
    common_ids = set.intersection(*id_sets) if id_sets else set()
    if len(common_ids) < count:
        raise ValueError(f"Only {len(common_ids)} common exit tracks; {count} requested.")
    energy_maps = [
        payload["all_exits"].set_index("track_id")["ke_ev"].to_dict()
        for payload in payloads
    ]
    scores = sorted(
        (
            int(track_id),
            float(np.mean([float(energy_map[track_id]) for energy_map in energy_maps])),
        )
        for track_id in common_ids
    )
    scores.sort(key=lambda item: item[1])
    labels = ("lower-exit-KE", "median-exit-KE", "upper-exit-KE")
    quantiles = np.linspace(0.25, 0.75, count)
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, quantile in zip(labels, quantiles):
        target = int(round(float(quantile) * (len(scores) - 1)))
        index = next(
            index
            for index in sorted(range(len(scores)), key=lambda value: abs(value - target))
            if scores[index][0] not in used
        )
        selected.append((label, scores[index][0]))
        used.add(scores[index][0])
    return selected


def _load_selected_snapshots(case_dir: Path, selected_ids: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    index = pd.read_csv(case_dir / "snapshots_index.csv")
    track_index = SNAPSHOT_COLUMNS.index("track_id")
    for record in index.itertuples(index=False):
        snapshot = np.load(case_dir / str(record.storage_ref))
        if snapshot.size == 0:
            continue
        ids = snapshot[:, track_index].astype(np.int64)
        keep = np.isin(ids, selected_ids)
        if not np.any(keep):
            continue
        frame = pd.DataFrame(snapshot[keep], columns=SNAPSHOT_COLUMNS)
        frame["track_id"] = frame["track_id"].astype(np.int64)
        frame["slot_id"] = frame["slot_id"].astype(np.int64)
        frame.insert(0, "time_s", float(record.time_s))
        frame.insert(0, "macro_step", int(record.macro_step))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _extract_tracks(
    payloads: list[dict[str, object]],
    selected: list[tuple[str, int]],
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    selected_ids = [track_id for _label, track_id in selected]
    for payload in payloads:
        case_dir = Path(payload["case_dir"])
        case = str(payload["row"]["case"])
        rf_vpeak = float(payload["row"]["rf_vpeak"])
        exits = payload["all_exits"]
        snapshots = _load_selected_snapshots(case_dir, selected_ids)
        for label, track_id in selected:
            event_rows = exits.loc[exits["track_id"].eq(track_id)]
            if event_rows.empty:
                continue
            event = event_rows.iloc[0]
            birth_time_s = float(event["birth_time_s"])
            track = snapshots.loc[snapshots["track_id"].eq(track_id)].copy()
            track = track.sort_values("time_s")
            track["sample_origin"] = "snapshot"
            terminal = pd.DataFrame(
                [
                    {
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
                        "fragmented_real_ions_represented": float(
                            event["fragmented_real_ions_represented"]
                        ),
                        "track_id": int(track_id),
                        "slot_id": math.nan,
                        "status_code": int(event["status_code"]),
                        "collision_count_per_ion": float(event["collision_count_per_ion"]),
                        "sample_origin": "terminal_event",
                    }
                ]
            )
            if track.empty or float(track["time_s"].max()) < float(event["event_time_s"]) - 1.0e-15:
                track = pd.concat([track, terminal], ignore_index=True)
            track.insert(0, "representative_group", label)
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
            output.append(track)
    if not output:
        return pd.DataFrame()
    return pd.concat(output, ignore_index=True).sort_values(
        ["representative_group", "track_id", "rf_vpeak", "time_s"]
    )


def _plot_tracks(tracks: pd.DataFrame, output: Path, dpi: int) -> None:
    if tracks.empty:
        return
    rf_values = sorted(tracks["rf_vpeak"].unique())
    colors = dict(zip(rf_values, ("#0072B2", "#009E73", "#D55E00")))
    styles = {"lower-exit-KE": "-", "median-exit-KE": "--", "upper-exit-KE": ":"}
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 10.2), sharex=True, constrained_layout=True)
    for (rf_vpeak, label, track_id), subset in tracks.groupby(
        ["rf_vpeak", "representative_group", "track_id"], sort=False
    ):
        subset = subset.sort_values("time_s")
        style = styles.get(str(label), "-")
        color = colors[float(rf_vpeak)]
        axes[0].plot(subset["z_mm"], subset["r_mm"], style, color=color, lw=1.25)
        axes[1].plot(subset["z_mm"], subset["teff_k"], style, color=color, lw=1.25)
        axes[2].plot(subset["z_mm"], subset["kinetic_energy_ev"], style, color=color, lw=1.25)
        axes[3].plot(subset["z_mm"], subset["fragmentation_percent"], style, color=color, lw=1.25)

    axes[0].set_ylabel("Radial position (mm)")
    axes[1].set_ylabel(r"$T_{eff}$ (K)")
    axes[2].set_ylabel("Kinetic energy (eV)")
    axes[3].set_ylabel("Dissociated weight (%)")
    axes[3].set_xlabel("Axial position z (mm)")
    rf_handles = [
        Line2D([0], [0], color=colors[value], lw=1.6, label=f"RF {value:g} Vpeak")
        for value in rf_values
    ]
    group_handles = [
        Line2D([0], [0], color="#333333", lw=1.4, linestyle=style, label=label.replace("-", " "))
        for label, style in styles.items()
    ]
    axes[0].legend(
        handles=rf_handles + group_handles,
        frameon=False,
        fontsize=7.2,
        ncol=3,
        loc="upper right",
    )
    for index, axis in enumerate(axes):
        axis.text(0.01, 0.97, f"({chr(97 + index)})", transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(True, color="#d9d9d9", linewidth=0.55, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records", double_precision=15))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stability-bin-ms", type=float, default=0.5)
    parser.add_argument("--trajectory-rf-vpeak", type=float, nargs="+", default=(70.0, 100.0, 160.0))
    parser.add_argument("--trajectory-count", type=int, default=3)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--trajectories-json", type=Path, required=True)
    parser.add_argument("--trajectory-plot", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    root = args.root.resolve()
    case_dirs = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and RF_PATTERN.search(path.name)
            and (path / "terminal_events.csv").exists()
        ),
        key=_rf_vpeak,
    )
    if not case_dirs:
        raise FileNotFoundError(f"No RF-sweep cases found under {root}")

    rows: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []
    for case_dir in case_dirs:
        row, payload = _case_payload(case_dir, args.stability_bin_ms)
        rows.append(row)
        payloads.append(payload)
    summary = pd.DataFrame(rows).sort_values("rf_vpeak")

    trajectory_payloads = _select_cases(payloads, tuple(args.trajectory_rf_vpeak))
    selected = _select_matched_tracks(trajectory_payloads, args.trajectory_count)
    tracks = _extract_tracks(trajectory_payloads, selected)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.trajectories_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(_json_records(summary), indent=2), encoding="utf-8")
    args.trajectories_json.write_text(json.dumps(_json_records(tracks), indent=2), encoding="utf-8")
    _plot_tracks(tracks, args.trajectory_plot.resolve(), args.dpi)

    print(f"cases={len(summary)}")
    print(f"trajectory_rf_vpeak={[float(item['row']['rf_vpeak']) for item in trajectory_payloads]}")
    print(f"common_exit_tracks={len(set.intersection(*[set(item['all_exits']['track_id'].astype(int)) for item in trajectory_payloads]))}")
    print(f"selected_tracks={selected}")
    print(f"summary_json={args.summary_json.resolve()}")
    print(f"trajectories_json={args.trajectories_json.resolve()}")
    print(f"trajectory_plot={args.trajectory_plot.resolve()}")


if __name__ == "__main__":
    main()
