"""Prepare ring-electrode trap PIC validation inputs.

This tool is intentionally narrow: it validates the SIMION PA0-derived trap
``.patxt`` files, optionally bakes the three DC stages with the shared RF basis,
and writes a stage schedule plus sweep matrix for paired PIC on/off runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_TRAP_DIR = Path("E_field") / "Iontrap"
DEFAULT_RF_PATXT = "ret_rf base.patxt"
DEFAULT_STAGE_FILES = {
    "injection": "ret_inject.patxt",
    "trapping": "ret_trapping.patxt",
    "exit": "ret_exit.patxt",
}
DEFAULT_STAGE_DURATIONS_S = {
    "injection": 30.0e-6,
    "trapping": 0.3e-3,
    "exit": 2.0e-3,
}
DEFAULT_SOURCE_ENABLED = {
    "injection": True,
    "trapping": False,
    "exit": False,
}


@dataclass(frozen=True)
class PatxtSummary:
    path: str
    header: dict[str, Any]
    point_count: int
    electrode_count: int
    potential_min_v: Optional[float]
    potential_max_v: Optional[float]
    electrode_min_v: Optional[float]
    electrode_max_v: Optional[float]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def summarize_patxt(path: Path) -> PatxtSummary:
    path = _resolve(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw_header: dict[str, str] = {}
    in_header = False
    in_points = False
    point_count = 0
    electrode_count = 0
    potential_min_v = np.inf
    potential_max_v = -np.inf
    electrode_min_v = np.inf
    electrode_max_v = -np.inf

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "begin_header":
                in_header = True
                continue
            if line == "end_header":
                in_header = False
                continue
            if line == "begin_points":
                in_points = True
                continue
            if line == "end_points":
                break
            if in_header:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    raw_header[parts[0]] = parts[1]
                continue
            if not in_points:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            point_count += 1
            value_v = float(parts[4])
            potential_min_v = min(potential_min_v, value_v)
            potential_max_v = max(potential_max_v, value_v)
            if parts[3] != "1":
                continue
            electrode_count += 1
            electrode_min_v = min(electrode_min_v, value_v)
            electrode_max_v = max(electrode_max_v, value_v)

    header: dict[str, Any] = {}
    for key, value in raw_header.items():
        try:
            header[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            header[key] = float(value)
            continue
        except ValueError:
            header[key] = value

    return PatxtSummary(
        path=str(path),
        header=header,
        point_count=point_count,
        electrode_count=electrode_count,
        potential_min_v=None if point_count == 0 else float(potential_min_v),
        potential_max_v=None if point_count == 0 else float(potential_max_v),
        electrode_min_v=None if electrode_count == 0 else float(electrode_min_v),
        electrode_max_v=None if electrode_count == 0 else float(electrode_max_v),
    )


def validate_patxt_set(trap_dir: Path, rf_name: str, stage_files: dict[str, str]) -> dict[str, Any]:
    trap_dir = _resolve(trap_dir)
    summaries = {"rf": summarize_patxt(trap_dir / rf_name)}
    for stage_name, file_name in stage_files.items():
        summaries[stage_name] = summarize_patxt(trap_dir / file_name)

    reference = summaries["rf"].header
    compare_keys = ("symmetry", "nx", "ny", "nz", "ng", "data_format")
    mismatches: list[str] = []
    for label, summary in summaries.items():
        for key in compare_keys:
            if summary.header.get(key) != reference.get(key):
                mismatches.append(
                    f"{label}.{key}={summary.header.get(key)!r} != rf.{key}={reference.get(key)!r}"
                )
        if summary.electrode_count != summaries["rf"].electrode_count:
            mismatches.append(
                f"{label}.electrode_count={summary.electrode_count} != rf.electrode_count={summaries['rf'].electrode_count}"
            )

    rf_min = summaries["rf"].electrode_min_v
    rf_max = summaries["rf"].electrode_max_v
    rf_global_min = summaries["rf"].potential_min_v
    rf_global_max = summaries["rf"].potential_max_v
    rf_electrode_normalized = (
        rf_min is not None
        and rf_max is not None
        and np.isclose(rf_min, -1.0, atol=1.0e-9, rtol=0.0)
        and np.isclose(rf_max, 1.0, atol=1.0e-9, rtol=0.0)
    )
    rf_global_normalized = (
        rf_global_min is not None
        and rf_global_max is not None
        and np.isclose(rf_global_min, -1.0, atol=1.0e-9, rtol=0.0)
        and np.isclose(rf_global_max, 1.0, atol=1.0e-9, rtol=0.0)
    )
    rf_normalized = rf_electrode_normalized or rf_global_normalized
    if not rf_normalized:
        mismatches.append(
            f"RF range check failed: electrode={rf_min}..{rf_max} V, "
            f"global={rf_global_min}..{rf_global_max} V, expected -1..+1 V."
        )

    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "rf_normalization_check": "electrode_potential_range" if rf_electrode_normalized else "global_potential_range",
        "summaries": {label: asdict(summary) for label, summary in summaries.items()},
    }


def stage_files_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "injection": str(args.injection_patxt),
        "trapping": str(args.trapping_patxt),
        "exit": str(args.exit_patxt),
    }


def bake_stage_fields(args: argparse.Namespace, output_dir: Path) -> dict[str, str]:
    from .field_baker import FieldBakeConfig, run_field_bake

    output_paths: dict[str, str] = {}
    trap_dir = _resolve(Path(args.iontrap_dir))
    rf_path = trap_dir / args.rf_patxt
    for stage_name, dc_file in stage_files_from_args(args).items():
        stage_dir = output_dir / "baked" / stage_name
        output_npy = stage_dir / "baked_fields.npy"
        plot_dir = stage_dir / "plots"
        config = FieldBakeConfig(
            simion_dc_csv=trap_dir / dc_file,
            simion_rf_csv=rf_path,
            fluent_csv=None,
            offset_z_simion_mm=0.0,
            offset_z_fluent_mm=0.0,
            output_npy=output_npy,
            output_plot_dir=plot_dir,
            background_pressure_pa=float(args.background_pressure_pa),
            background_temperature_k=float(args.background_temperature_k),
            phi_key_for_plot="phi_dc_v",
            z_min_mm=float(args.z_min_mm),
            z_max_mm=float(args.z_max_mm),
            r_min_mm=float(args.r_min_mm),
            r_max_mm=float(args.r_max_mm),
            dz_mm=float(args.dz_mm),
            dr_mm=float(args.dr_mm),
            capillary_exit_z_mm=float(args.source_z_mm),
            clip_fluent_before_capillary_exit=False,
            simion_pa_effective_grids_per_mm=float(args.simion_pa_grids_per_mm),
            simion_dc_voltage_scale=1.0,
        )
        run_field_bake(config)
        output_paths[stage_name] = str(output_npy)
    return output_paths


def write_stage_schedule(output_dir: Path, stage_field_paths: dict[str, str]) -> Path:
    schedule_path = output_dir / "smoke_stage_schedule.json"
    stages = []
    for stage_name in ("injection", "trapping", "exit"):
        stages.append(
            {
                "name": stage_name,
                "duration_s": DEFAULT_STAGE_DURATIONS_S[stage_name],
                "static_field_path": stage_field_paths[stage_name],
                "source_enabled": DEFAULT_SOURCE_ENABLED[stage_name],
            }
        )
    schedule = {
        "schema": "simu_ionsource.stage_schedule.v1",
        "description": "Ring trap smoke schedule: injection 30 us, trapping 0.3 ms, exit 2 ms.",
        "stages": stages,
    }
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    return schedule_path


def write_sweep_matrix(output_dir: Path) -> Path:
    matrix_path = output_dir / "pic_sweep_matrix.csv"
    currents_a = [1.0e-12, 10.0e-12, 100.0e-12]
    injection_us = [10.0, 30.0, 100.0]
    trapping_ms = [0.1, 0.3, 1.0, 3.0]
    rf_peak_v = [25.0, 50.0, 100.0]
    rf_frequency_hz = [0.5e6, 1.0e6, 2.0e6]
    seeds = [7, 17, 37]
    pic_scales = [1.0, 0.0]
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ion_current_a",
                "injection_us",
                "trapping_ms",
                "exit_ms",
                "rf_peak_voltage_v",
                "rf_frequency_hz",
                "random_seed",
                "pic_space_charge_scale",
            ],
        )
        writer.writeheader()
        for current_a in currents_a:
            for inj_us in injection_us:
                for trap_ms in trapping_ms:
                    for peak_v in rf_peak_v:
                        for frequency_hz in rf_frequency_hz:
                            for seed in seeds:
                                for pic_scale in pic_scales:
                                    writer.writerow(
                                        {
                                            "ion_current_a": current_a,
                                            "injection_us": inj_us,
                                            "trapping_ms": trap_ms,
                                            "exit_ms": 2.0,
                                            "rf_peak_voltage_v": peak_v,
                                            "rf_frequency_hz": frequency_hz,
                                            "random_seed": seed,
                                            "pic_space_charge_scale": pic_scale,
                                        }
                                    )
    return matrix_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and prepare ring-trap PIC inputs.")
    parser.add_argument("--iontrap-dir", default=str(DEFAULT_TRAP_DIR), help="Directory containing the PA0 .patxt files.")
    parser.add_argument("--rf-patxt", default=DEFAULT_RF_PATXT, help="Shared RF basis .patxt filename.")
    parser.add_argument("--injection-patxt", default=DEFAULT_STAGE_FILES["injection"], help="Injection DC .patxt filename.")
    parser.add_argument("--trapping-patxt", default=DEFAULT_STAGE_FILES["trapping"], help="Trapping DC .patxt filename.")
    parser.add_argument("--exit-patxt", default=DEFAULT_STAGE_FILES["exit"], help="Exit DC .patxt filename.")
    parser.add_argument("--output-dir", default=str(Path("outputs") / "ring_trap_pic"), help="Output directory.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate inputs and write metadata files.")
    parser.add_argument("--bake", action="store_true", help="Bake the three stage fields. This can be slow and memory-heavy.")
    parser.add_argument("--z-min-mm", type=float, default=0.0)
    parser.add_argument("--z-max-mm", type=float, default=80.0)
    parser.add_argument("--r-min-mm", type=float, default=0.0)
    parser.add_argument("--r-max-mm", type=float, default=20.0)
    parser.add_argument("--dz-mm", type=float, default=0.01)
    parser.add_argument("--dr-mm", type=float, default=0.01)
    parser.add_argument("--simion-pa-grids-per-mm", type=float, default=100.0)
    parser.add_argument("--background-pressure-pa", type=float, default=1.0e-5)
    parser.add_argument("--background-temperature-k", type=float, default=300.0)
    parser.add_argument("--source-z-mm", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_files = stage_files_from_args(args)
    validation = validate_patxt_set(Path(args.iontrap_dir), args.rf_patxt, stage_files)
    validation_path = output_dir / "patxt_validation_summary.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["ok"]:
        raise ValueError(
            "PATXT validation failed; see "
            f"{validation_path}. First mismatch: {validation['mismatches'][0]}"
        )

    stage_field_paths = {
        stage_name: str((output_dir / "baked" / stage_name / "baked_fields.npy").resolve())
        for stage_name in stage_files
    }
    if args.bake and not args.validate_only:
        stage_field_paths = bake_stage_fields(args, output_dir)
    schedule_path = write_stage_schedule(output_dir, stage_field_paths)
    matrix_path = write_sweep_matrix(output_dir)

    print(f"validation: {validation_path}")
    print(f"stage schedule: {schedule_path}")
    print(f"sweep matrix: {matrix_path}")
    if not args.bake:
        print("bake skipped; rerun with --bake to create the stage baked_fields.npy files.")


if __name__ == "__main__":
    main()
