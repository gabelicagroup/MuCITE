"""Extract per-ms transport, energy, fragmentation, and loss metrics for a current sweep."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .analyze_rf_sweep_multimetric_per_ms import (
        EVENT_COLUMNS,
        current_na,
        distribution_stats,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from src.data.diagnostics.analyze_rf_sweep_multimetric_per_ms import (  # type: ignore
        EVENT_COLUMNS,
        current_na,
        distribution_stats,
    )


CURRENT_PATTERN = re.compile(r"I(?P<current>\d+(?:p\d+)?)nA", re.IGNORECASE)
RF_PATTERN = re.compile(r"RF(?P<rf>\d+)V?", re.IGNORECASE)
LOSS_STATUSES = ("electrode_hit", "domain_out", "radial_out")


def _label_number(pattern: re.Pattern[str], text: str, group: str) -> float:
    match = pattern.search(text)
    if not match:
        return math.nan
    return float(match.group(group).replace("p", "."))


def _percent(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    return 100.0 * float(numerator) / float(denominator)


def _event_mask(events: pd.DataFrame, start_s: float, end_s: float, *, include_end: bool) -> pd.Series:
    mask = events["event_time_s"].ge(start_s)
    if include_end:
        return mask & events["event_time_s"].le(end_s + 1.0e-15)
    return mask & events["event_time_s"].lt(end_s)


def _status_metrics(
    events: pd.DataFrame,
    status: str,
    duration_s: float,
    source_current_na: float,
    suffix: str,
) -> dict[str, float | int]:
    subset = events.loc[events["status"].eq(status)]
    weight = float(subset["represented_real_ions"].sum())
    event_current_na = current_na(weight, duration_s)
    return {
        f"{status}_macro_events_{suffix}": int(len(subset)),
        f"{status}_weight_{suffix}": weight,
        f"{status}_current_nA_{suffix}": event_current_na,
        f"{status}_percent_of_source_{suffix}": _percent(event_current_na, source_current_na),
    }


def _exit_metrics(
    events: pd.DataFrame,
    duration_s: float,
    source_current_na: float,
    suffix: str,
) -> dict[str, object]:
    exits = events.loc[events["status"].eq("z_exit")]
    total_weight = float(exits["represented_real_ions"].sum())
    parent_weight = float(exits["parent_real_ions_remaining"].sum())
    fragment_weight = float(exits["fragmented_real_ions_represented"].sum())
    total_current_na = current_na(total_weight, duration_s)
    parent_current_na = current_na(parent_weight, duration_s)
    fragment_current_na = current_na(fragment_weight, duration_s)
    stats = distribution_stats(exits)
    result: dict[str, object] = {
        f"z_exit_macro_events_{suffix}": int(len(exits)),
        f"z_exit_weight_{suffix}": total_weight,
        f"z_exit_parent_weight_{suffix}": parent_weight,
        f"z_exit_fragment_weight_{suffix}": fragment_weight,
        f"ion_current_nA_{suffix}": total_current_na,
        f"parent_current_nA_{suffix}": parent_current_na,
        f"fragment_current_nA_{suffix}": fragment_current_na,
        f"transmission_percent_{suffix}": _percent(total_current_na, source_current_na),
        f"intact_transmission_percent_{suffix}": _percent(parent_current_na, source_current_na),
        f"fragment_transmission_percent_{suffix}": _percent(fragment_current_na, source_current_na),
        f"fragmentation_percent_of_exit_{suffix}": _percent(fragment_weight, total_weight),
    }
    result.update({f"{key}_{suffix}": value for key, value in stats.items()})
    return result


def _load_run_metadata(case_dir: Path) -> dict[str, object]:
    with (case_dir / "simulation_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    terminal_diagnostics = summary["result"]["terminal_event_diagnostics"]
    metadata = {
        "requested_time_s": float(summary["config"]["total_time_s"]),
        "simulated_time_s": float(summary["performance"]["simulated_time_s"]),
        "source_current_a": float(summary["source_runtime"]["ion_current_a"]),
        "represented_source_current_a": float(summary["source_runtime"]["represented_injected_current_a"]),
        "actual_rf_vpeak": float(summary["config"]["rf_peak_voltage_v"]),
        "charge_state": int(summary["template"]["charge_state"]),
        "termination_reason": str(summary["result"]["termination_reason"]),
        "injected_real_ions": float(summary["source_runtime"]["injected_real_ions"]),
        "terminal_rows": int(terminal_diagnostics["rows"]),
        "terminal_rows_capped": bool(terminal_diagnostics["max_rows_reached"]),
        "omitted_rows_by_status": dict(terminal_diagnostics["omitted_rows_by_status"]),
        "static_field": str(summary["config"]["static_field_path"]),
    }
    del summary
    gc.collect()
    return metadata


def load_case(case_dir: Path, root: Path, bin_width_ms: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    metadata = _load_run_metadata(case_dir)
    requested_time_s = float(metadata["requested_time_s"])
    simulated_time_s = float(metadata["simulated_time_s"])
    source_current_na = float(metadata["source_current_a"]) * 1.0e9
    represented_source_current_na = float(metadata["represented_source_current_a"]) * 1.0e9
    completed = simulated_time_s >= requested_time_s - 1.0e-12

    events = pd.read_csv(case_dir / "terminal_events.csv", usecols=EVENT_COLUMNS)
    events = events.loc[np.isfinite(events["event_time_s"])].copy()
    final_particles = pd.read_csv(
        case_dir / "final_particles.csv",
        usecols=["final_status", "represented_real_ions"],
    )
    active_at_end = final_particles.loc[final_particles["final_status"].isin(["active", "capillary_buffer"])]
    active_at_end_weight = float(active_at_end["represented_real_ions"].sum())

    terminal_weight = float(events["represented_real_ions"].sum())
    observed_weight = terminal_weight + active_at_end_weight
    injected_weight = float(metadata["injected_real_ions"])
    conservation_error_weight = observed_weight - injected_weight
    conservation_error_percent = _percent(conservation_error_weight, injected_weight)

    root_rf_label = _label_number(RF_PATTERN, root.name, "rf")
    case_rf_label = _label_number(RF_PATTERN, case_dir.name, "rf")
    case_current_label = _label_number(CURRENT_PATTERN, case_dir.name, "current")
    actual_rf = float(metadata["actual_rf_vpeak"])

    common: dict[str, object] = {
        "case": case_dir.name,
        "dataset_directory": root.name,
        "directory_rf_label_vpeak": root_rf_label,
        "case_rf_label_vpeak": case_rf_label,
        "actual_rf_vpeak": actual_rf,
        "directory_rf_label_matches_actual": bool(math.isfinite(root_rf_label) and abs(root_rf_label - actual_rf) < 1.0e-12),
        "case_rf_label_matches_actual": bool(math.isfinite(case_rf_label) and abs(case_rf_label - actual_rf) < 1.0e-12),
        "case_current_label_nA": case_current_label,
        "source_current_nA": source_current_na,
        "represented_source_current_nA": represented_source_current_na,
        "requested_total_time_ms": requested_time_s * 1.0e3,
        "simulated_time_ms": simulated_time_s * 1.0e3,
        "completed_requested_time": completed,
        "termination_reason": metadata["termination_reason"],
        "terminal_event_rows": metadata["terminal_rows"],
        "terminal_event_rows_capped": metadata["terminal_rows_capped"],
        "omitted_terminal_rows": int(sum(metadata["omitted_rows_by_status"].values())),
        "static_field": metadata["static_field"],
        "injected_real_ions": injected_weight,
        "terminal_weight_0_to_end": terminal_weight,
        "active_at_end_weight": active_at_end_weight,
        "observed_weight_terminal_plus_active": observed_weight,
        "weight_conservation_error": conservation_error_weight,
        "weight_conservation_error_percent": conservation_error_percent,
    }

    overall: dict[str, object] = {}
    overall.update(_exit_metrics(events, simulated_time_s, source_current_na, "0_to_end"))
    for status in LOSS_STATUSES:
        overall.update(_status_metrics(events, status, simulated_time_s, source_current_na, "0_to_end"))

    stable_start_s = min(1.0e-3, simulated_time_s)
    stable_duration_s = simulated_time_s - stable_start_s
    stable_events = events.loc[events["event_time_s"].ge(stable_start_s)]
    stable: dict[str, object] = {
        "stable_window_ms": f"{stable_start_s * 1.0e3:g}-{simulated_time_s * 1.0e3:g}",
    }
    stable.update(_exit_metrics(stable_events, stable_duration_s, source_current_na, "stable"))
    for status in LOSS_STATUSES:
        stable.update(_status_metrics(stable_events, status, stable_duration_s, source_current_na, "stable"))

    requested_time_ms = requested_time_s * 1.0e3
    bin_count = int(math.ceil(requested_time_ms / bin_width_ms - 1.0e-12))
    rows: list[dict[str, object]] = []
    for bin_index in range(bin_count):
        start_ms = bin_index * bin_width_ms
        end_ms = min((bin_index + 1) * bin_width_ms, requested_time_ms)
        duration_ms = end_ms - start_ms
        start_s = start_ms * 1.0e-3
        end_s = end_ms * 1.0e-3
        subset = events.loc[
            _event_mask(events, start_s, end_s, include_end=bin_index == bin_count - 1)
        ]
        duration_s = duration_ms * 1.0e-3
        terminal_bin_weight = float(subset["represented_real_ions"].sum())
        terminal_bin_current_na = current_na(terminal_bin_weight, duration_s)
        row: dict[str, object] = {
            **common,
            "ms_bin": bin_index + 1,
            "time_start_ms": start_ms,
            "time_end_ms": end_ms,
            "bin_duration_ms": duration_ms,
            "bin_data_state": "complete" if simulated_time_s >= end_s - 1.0e-12 else "partial_or_missing",
            "terminal_macro_events_this_ms": int(len(subset)),
            "terminal_weight_this_ms": terminal_bin_weight,
            "terminal_outcome_current_nA_this_ms": terminal_bin_current_na,
            "terminal_outcome_percent_of_source_this_ms": _percent(terminal_bin_current_na, source_current_na),
        }
        row.update(_exit_metrics(subset, duration_s, source_current_na, "this_ms"))
        for status in LOSS_STATUSES:
            row.update(_status_metrics(subset, status, duration_s, source_current_na, "this_ms"))
        row.update(overall)
        row.update(stable)
        rows.append(row)

    case_summary = {
        **common,
        **overall,
        **stable,
    }
    return rows, case_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-summary-csv", type=Path)
    parser.add_argument("--bin-width-ms", type=float, default=1.0)
    args = parser.parse_args()

    if args.bin_width_ms <= 0.0:
        raise ValueError("--bin-width-ms must be positive.")
    root = args.root.resolve()
    output_csv = args.output_csv or root / f"{root.name}_multimetric_per_ms.csv"
    output_summary_csv = args.output_summary_csv
    case_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and CURRENT_PATTERN.search(path.name)),
        key=lambda path: _label_number(CURRENT_PATTERN, path.name, "current"),
    )
    if not case_dirs:
        raise SystemExit(f"No current-sweep case directories found under {root}")

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for case_dir in case_dirs:
        case_rows, case_summary = load_case(case_dir, root, args.bin_width_ms)
        rows.extend(case_rows)
        summaries.append(case_summary)

    per_ms = pd.DataFrame(rows).sort_values(["source_current_nA", "ms_bin"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    per_ms.to_csv(output_csv, index=False, float_format="%.12g")
    if output_summary_csv is not None:
        summary = pd.DataFrame(summaries).sort_values("source_current_nA")
        output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_summary_csv, index=False, float_format="%.12g")

    print(f"cases={len(case_dirs)} rows={len(per_ms)}")
    print(f"csv={output_csv}")
    if output_summary_csv is not None:
        print(f"summary_csv={output_summary_csv}")


if __name__ == "__main__":
    main()
