"""Summarize stable-window transport, energy, fragmentation, and terminal losses."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_rf_sweep_multimetric_per_ms import ELEMENTARY_CHARGE_C, weighted_quantile


EVENT_COLUMNS = [
    "status",
    "event_time_s",
    "ke_ev",
    "internal_temperature_k",
    "represented_real_ions",
    "fragmented_real_ions_represented",
]
OUTCOME_STATUSES = ("electrode_hit", "domain_out", "radial_out")


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


def _load_case(case_dir: Path, dataset: str, stable_start_ms: float) -> dict[str, object]:
    with (case_dir / "simulation_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)

    config = summary["config"]
    source = summary["source_runtime"]
    template = summary["template"]
    simulated_time_s = float(summary["performance"]["simulated_time_s"])
    requested_time_s = float(config["total_time_s"])
    stable_start_s = min(stable_start_ms * 1.0e-3, simulated_time_s)
    stable_duration_s = simulated_time_s - stable_start_s
    source_current_na = float(source["ion_current_a"]) * 1.0e9
    charge_state = int(template["charge_state"])

    events = pd.read_csv(case_dir / "terminal_events.csv", usecols=EVENT_COLUMNS)
    events = events.loc[
        np.isfinite(events["event_time_s"])
        & events["event_time_s"].ge(stable_start_s)
        & events["event_time_s"].le(simulated_time_s + 1.0e-15)
    ].copy()
    exits = events.loc[events["status"].eq("z_exit")]
    exit_weight = float(exits["represented_real_ions"].sum())
    exit_current_na = _current_na(exit_weight, charge_state, stable_duration_s)
    fragment_weight = float(exits["fragmented_real_ions_represented"].sum())

    row: dict[str, object] = {
        "dataset": dataset,
        "case": case_dir.name,
        "collision_model": str(config["collision_model"]),
        "rf_vpeak": float(config["rf_peak_voltage_v"]),
        "source_current_nA": source_current_na,
        "stable_window_start_ms": stable_start_s * 1.0e3,
        "stable_window_end_ms": simulated_time_s * 1.0e3,
        "completed_requested_time": simulated_time_s >= requested_time_s - 1.0e-12,
        "stable_exit_ion_current_nA": exit_current_na,
        "stable_transmission_percent": 100.0 * exit_current_na / source_current_na,
        "stable_ke_ev_median_weighted": _weighted_median(exits, "ke_ev"),
        "stable_temperature_k_median_weighted": _weighted_median(exits, "internal_temperature_k"),
        "stable_fragmentation_percent_of_exit": (
            100.0 * fragment_weight / exit_weight if exit_weight > 0.0 else math.nan
        ),
    }
    for status in OUTCOME_STATUSES:
        weight = float(events.loc[events["status"].eq(status), "represented_real_ions"].sum())
        current_na = _current_na(weight, charge_state, stable_duration_s)
        row[f"stable_{status}_percent_of_input"] = 100.0 * current_na / source_current_na

    del summary
    gc.collect()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Directories containing case subdirectories.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--stable-start-ms", default=1.0, type=float)
    args = parser.parse_args()

    if args.stable_start_ms < 0.0:
        raise ValueError("--stable-start-ms must be non-negative.")

    rows: list[dict[str, object]] = []
    for root in args.roots:
        root = root.resolve()
        cases = sorted(path for path in root.iterdir() if path.is_dir() and (path / "terminal_events.csv").exists())
        if not cases:
            raise FileNotFoundError(f"No terminal-event case directories found under {root}")
        rows.extend(_load_case(case, root.name, args.stable_start_ms) for case in cases)

    output = args.output_csv.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.12g")
    print(f"cases={len(rows)}")
    print(f"csv={output}")


if __name__ == "__main__":
    main()
