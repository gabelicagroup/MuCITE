"""Summarize trap-only PIC capacity scan outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ELEMENTARY_CHARGE_C = 1.602176634e-19


def _to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stage_durations_us(stage_history: list[dict[str, Any]]) -> tuple[float, float]:
    loading_us = math.nan
    hold_us = 0.0
    for stage in stage_history:
        duration_us = _to_float(stage.get("duration_s")) * 1.0e6
        if bool(stage.get("source_enabled")) and math.isnan(loading_us):
            loading_us = duration_us
        if not bool(stage.get("source_enabled")):
            hold_us += duration_us
    return loading_us, hold_us


def _max_active_charge_pC(summary: dict[str, Any], source: dict[str, Any]) -> tuple[float, float, float, float]:
    target_weight = _to_float(source.get("source_target_macro_weight"), 0.0)
    max_count = 0.0
    max_time_us = math.nan
    max_ratio = math.nan
    for row in summary.get("result", {}).get("macro_history", []) or []:
        count = _to_float(row.get("ion_count"), 0.0)
        if count > max_count:
            max_count = count
            max_time_us = _to_float(row.get("time_s")) * 1.0e6
        ratio = _to_float(row.get("sce_external_ratio_p95"))
        if math.isfinite(ratio):
            max_ratio = ratio if not math.isfinite(max_ratio) else max(max_ratio, ratio)
    return max_count * target_weight * ELEMENTARY_CHARGE_C * 1.0e12, max_count, max_time_us, max_ratio


def summarize_run(summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    source = summary.get("source_runtime", {})
    config = summary.get("config", {})
    result = summary.get("result", {})
    final_macro = result.get("final_macro", {})
    terminal = source.get("terminal_real_ions_by_status", {}) or {}
    stage_history = result.get("stage_history", []) or []

    loading_us, hold_us = _stage_durations_us(stage_history)
    emitted_ions = _to_float(source.get("emitted_real_ions"), 0.0)
    active_ions = _to_float(source.get("active_real_ions"), 0.0)
    capillary_ions = _to_float(source.get("capillary_buffer_real_ions"), 0.0)
    blocked_ions = _to_float(source.get("blocked_real_ions"), 0.0)
    max_active_charge_pC, max_active_macros, max_active_time_us, max_ratio = _max_active_charge_pC(summary, source)
    sor_max = int(_to_float(config.get("sor_max_iters"), 0.0))
    final_sor = _to_float(final_macro.get("sor_iters"))

    def status_ions(name: str) -> float:
        return _to_float(terminal.get(name), 0.0)

    active_charge_pC = active_ions * ELEMENTARY_CHARGE_C * 1.0e12
    emitted_charge_pC = emitted_ions * ELEMENTARY_CHARGE_C * 1.0e12

    return {
        "run": summary_path.parent.name,
        "summary_path": str(summary_path),
        "generate_time_us": loading_us,
        "trap_time_us": hold_us,
        "pic_space_charge_scale": _to_float(config.get("pic_space_charge_scale")),
        "ion_current_a": _to_float(config.get("ion_current_a")),
        "rf_peak_voltage_v": _to_float(config.get("rf_peak_voltage_v")),
        "rf_frequency_hz": _to_float(config.get("rf_frequency_hz")),
        "initial_ke_ev": _to_float(summary.get("template", {}).get("initial_kinetic_energy_ev")),
        "source_z_mm": _to_float(source.get("capillary_exit_z_m")) * 1.0e3,
        "source_radius_mm": _to_float(config.get("source_radius_m")) * 1.0e3,
        "pic_grid_nr": summary.get("grid_runtime", {}).get("pic_grid_nr"),
        "pic_grid_nz": summary.get("grid_runtime", {}).get("pic_grid_nz"),
        "macro_time_step_s": _to_float(config.get("macro_time_step_s")),
        "sor_max_iters": sor_max,
        "final_sor_iters": final_sor,
        "final_sor_hit_limit": bool(sor_max > 0 and math.isfinite(final_sor) and final_sor >= sor_max),
        "emitted_real_ions": emitted_ions,
        "emitted_charge_pC": emitted_charge_pC,
        "active_real_ions": active_ions,
        "active_charge_pC": active_charge_pC,
        "active_fraction": active_ions / emitted_ions if emitted_ions > 0.0 else math.nan,
        "capillary_buffer_real_ions": capillary_ions,
        "blocked_real_ions": blocked_ions,
        "blocked_charge_pC": blocked_ions * ELEMENTARY_CHARGE_C * 1.0e12,
        "max_active_real_ions_est": max_active_macros * _to_float(source.get("source_target_macro_weight"), 0.0),
        "max_active_charge_pC_est": max_active_charge_pC,
        "max_active_time_us": max_active_time_us,
        "sce_external_ratio_p95_max": max_ratio,
        "z_exit_real_ions": status_ions("z_exit"),
        "z_exit_charge_pC": status_ions("z_exit") * ELEMENTARY_CHARGE_C * 1.0e12,
        "electrode_hit_real_ions": status_ions("electrode_hit"),
        "electrode_hit_charge_pC": status_ions("electrode_hit") * ELEMENTARY_CHARGE_C * 1.0e12,
        "radial_out_real_ions": status_ions("radial_out"),
        "radial_out_charge_pC": status_ions("radial_out") * ELEMENTARY_CHARGE_C * 1.0e12,
        "domain_out_real_ions": status_ions("domain_out"),
        "domain_out_charge_pC": status_ions("domain_out") * ELEMENTARY_CHARGE_C * 1.0e12,
        "termination_reason": result.get("termination_reason", ""),
        "final_time_us": _to_float(result.get("final_macro", {}).get("time_s")) * 1.0e6,
    }


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["run"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional diagnostic path
        print(f"plot skipped: matplotlib unavailable ({exc})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_to_float(row.get("pic_space_charge_scale")), []).append(row)

    def label(scale: float) -> str:
        return "PIC on" if scale > 0.0 else "PIC off"

    for y_key, y_label, file_name in (
        ("active_charge_pC", "Q_active_end [pC]", "active_charge_vs_generate_time.png"),
        ("active_fraction", "Q_active_end / Q_emitted", "active_fraction_vs_generate_time.png"),
        ("max_active_charge_pC_est", "max active charge estimate [pC]", "max_active_charge_vs_generate_time.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
        for scale, group_rows in sorted(groups.items()):
            ordered = sorted(group_rows, key=lambda item: _to_float(item.get("generate_time_us")))
            ax.plot(
                [_to_float(item.get("generate_time_us")) for item in ordered],
                [_to_float(item.get(y_key)) for item in ordered],
                marker="o",
                label=label(scale),
            )
        ax.set_xlabel("generation time [us]")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / file_name)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    for status, color in (
        ("z_exit_charge_pC", "tab:green"),
        ("electrode_hit_charge_pC", "tab:red"),
        ("radial_out_charge_pC", "tab:orange"),
        ("domain_out_charge_pC", "tab:blue"),
    ):
        ordered = sorted(rows, key=lambda item: (_to_float(item.get("pic_space_charge_scale")), _to_float(item.get("generate_time_us"))))
        ax.scatter(
            [_to_float(item.get("generate_time_us")) for item in ordered],
            [_to_float(item.get(status)) for item in ordered],
            s=18,
            label=status.replace("_charge_pC", ""),
            color=color,
            alpha=0.75,
        )
    ax.set_xlabel("generation time [us]")
    ax.set_ylabel("terminal charge [pC]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "terminal_charge_split.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize trap-only PIC capacity scan outputs.")
    parser.add_argument("run_root", help="Directory containing trap-only run subdirectories.")
    parser.add_argument("--output-csv", default="", help="Output CSV path. Defaults to run_root/trap_only_summary.csv.")
    parser.add_argument("--plots", action="store_true", help="Write diagnostic PNG plots under run_root/plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    summaries = sorted(run_root.glob("*/simulation_summary.json"))
    rows = [summarize_run(path) for path in summaries]
    rows.sort(key=lambda item: (_to_float(item.get("generate_time_us")), _to_float(item.get("pic_space_charge_scale")), item.get("run", "")))

    output_csv = Path(args.output_csv).resolve() if args.output_csv else run_root / "trap_only_summary.csv"
    write_csv(rows, output_csv)
    print(f"wrote {len(rows)} rows: {output_csv}")

    if args.plots:
        write_plots(rows, run_root / "plots")
        print(f"plots: {run_root / 'plots'}")


if __name__ == "__main__":
    main()
