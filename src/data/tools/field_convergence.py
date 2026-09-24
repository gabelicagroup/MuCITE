"""Field-baker convergence diagnostics for SIMION refined PA text fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import distance_transform_edt

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .field_baker import (
    FieldBakeConfig,
    GlobalGrid,
    RF_REFERENCE_PEAK_VOLTAGE_V,
    _read_simion_patxt,
    _validate_rf_patxt_normalization,
    run_field_bake,
)


DEFAULT_GRID_MM = (0.20, 0.10, 0.05)
DEFAULT_Z_MIN_MM = 0.0
DEFAULT_Z_MAX_MM = 70.0
DEFAULT_R_MIN_MM = 0.0
DEFAULT_R_MAX_MM = 20.0
DEFAULT_SIMION_GRIDS_PER_MM = 100.0
DEFAULT_DC_SCALE = 1.0
BOUNDARY_MARGIN_FACTOR = 2.0
SMOOTH_ELECTRODE_DISTANCE_MM = 0.30
NEAR_ELECTRODE_MIN_DISTANCE_MM = 0.10
NEAR_ELECTRODE_MAX_DISTANCE_MM = 0.30
TINY = 1.0e-300


@dataclass(frozen=True)
class SourceReference:
    label: str
    path: Path
    spline: RectBivariateSpline
    electrode_distance_spline: RectBivariateSpline
    local_r_coords_m: np.ndarray
    shifted_z_coords_m: np.ndarray
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CaseData:
    h_mm: float
    case_dir: Path
    baked_path: Path
    plot_paths: Tuple[Path, ...]
    grid: GlobalGrid
    baked: Dict[str, Any]


def _case_name(h_mm: float) -> str:
    return f"h_{h_mm:.2f}mm".replace(".", "p")


def _load_baked(path: Path) -> Dict[str, Any]:
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a baked field dictionary.")
    return payload


def _build_grid(h_mm: float) -> GlobalGrid:
    return GlobalGrid(
        z_min_m=DEFAULT_Z_MIN_MM * 1.0e-3,
        z_max_m=DEFAULT_Z_MAX_MM * 1.0e-3,
        r_min_m=DEFAULT_R_MIN_MM * 1.0e-3,
        r_max_m=DEFAULT_R_MAX_MM * 1.0e-3,
        dz_m=h_mm * 1.0e-3,
        dr_m=h_mm * 1.0e-3,
    )


def _build_source_reference(
    *,
    label: str,
    path: Path,
    voltage_scale: float,
    pa_effective_grids_per_mm: float,
    z_offset_mm: float,
) -> SourceReference:
    header, pa_potential_v, electrode_mask = _read_simion_patxt(path)
    if str(header["symmetry"]).lower() != "cylindrical":
        raise ValueError(f"{path} is not a cylindrical SIMION PA text file.")

    if label == "rf":
        validation_metadata = _validate_rf_patxt_normalization(path, pa_potential_v, electrode_mask)
        voltage_scale = RF_REFERENCE_PEAK_VOLTAGE_V
    else:
        validation_metadata = {}

    effective_grids_per_mm = float(pa_effective_grids_per_mm)
    spacing_m = 1.0e-3 / effective_grids_per_mm
    local_r_coords_m = np.arange(int(header["ny"]), dtype=float) * spacing_m
    local_z_coords_m = np.arange(int(header["nx"]), dtype=float) * spacing_m
    shifted_z_coords_m = local_z_coords_m + float(z_offset_mm) * 1.0e-3
    local_phi_v = pa_potential_v * float(voltage_scale)

    spline = RectBivariateSpline(
        local_r_coords_m,
        shifted_z_coords_m,
        local_phi_v,
        kx=min(3, len(local_r_coords_m) - 1),
        ky=min(3, len(shifted_z_coords_m) - 1),
    )
    distance_to_electrode_m = distance_transform_edt(~electrode_mask, sampling=(spacing_m, spacing_m))
    electrode_distance_spline = RectBivariateSpline(
        local_r_coords_m,
        shifted_z_coords_m,
        distance_to_electrode_m,
        kx=1,
        ky=1,
    )

    metadata = {
        "source_path": str(path),
        "source_format": "patxt",
        "pa_nx": int(header["nx"]),
        "pa_ny": int(header["ny"]),
        "pa_effective_grids_per_mm": effective_grids_per_mm,
        "pa_source_spacing_mm": 1.0 / effective_grids_per_mm,
        "pa_coordinate_mapping": "x_index_to_global_z__y_index_to_global_r",
        "voltage_scale": float(voltage_scale),
        **validation_metadata,
    }
    return SourceReference(
        label=label,
        path=path,
        spline=spline,
        electrode_distance_spline=electrode_distance_spline,
        local_r_coords_m=local_r_coords_m,
        shifted_z_coords_m=shifted_z_coords_m,
        metadata=metadata,
    )


def _evaluate_source_reference(reference: SourceReference, grid: GlobalGrid) -> Dict[str, np.ndarray]:
    grid_r_m, grid_z_m = grid.mesh_rz
    phi_v = reference.spline(grid.r_coords_m, grid.z_coords_m)
    e_r_v_per_m = -reference.spline(grid.r_coords_m, grid.z_coords_m, dx=1, dy=0)
    e_z_v_per_m = -reference.spline(grid.r_coords_m, grid.z_coords_m, dx=0, dy=1)
    electrode_distance_m = reference.electrode_distance_spline(grid.r_coords_m, grid.z_coords_m)

    source_r_min_m = float(np.min(reference.local_r_coords_m))
    source_r_max_m = float(np.max(reference.local_r_coords_m))
    source_z_min_m = float(np.min(reference.shifted_z_coords_m))
    source_z_max_m = float(np.max(reference.shifted_z_coords_m))
    inside_source = (
        (grid_r_m >= source_r_min_m)
        & (grid_r_m <= source_r_max_m)
        & (grid_z_m >= source_z_min_m)
        & (grid_z_m <= source_z_max_m)
    )
    return {
        "phi_v": phi_v,
        "e_r_v_per_m": e_r_v_per_m,
        "e_z_v_per_m": e_z_v_per_m,
        "electrode_distance_m": electrode_distance_m,
        "inside_source": inside_source,
        "grid_r_m": grid_r_m,
        "grid_z_m": grid_z_m,
    }


def _region_masks(reference_eval: Mapping[str, np.ndarray], *, boundary_margin_m: float) -> Dict[str, np.ndarray]:
    inside_source = np.asarray(reference_eval["inside_source"], dtype=bool)
    grid_r_m = np.asarray(reference_eval["grid_r_m"], dtype=float)
    grid_z_m = np.asarray(reference_eval["grid_z_m"], dtype=float)
    electrode_distance_m = np.asarray(reference_eval["electrode_distance_m"], dtype=float)
    if not np.any(inside_source):
        raise ValueError("Reference source mask is empty.")

    source_r_min_m = float(np.min(grid_r_m[inside_source]))
    source_r_max_m = float(np.max(grid_r_m[inside_source]))
    source_z_min_m = float(np.min(grid_z_m[inside_source]))
    source_z_max_m = float(np.max(grid_z_m[inside_source]))
    boundary_excluded = (
        inside_source
        & (grid_r_m >= source_r_min_m + boundary_margin_m)
        & (grid_r_m <= source_r_max_m - boundary_margin_m)
        & (grid_z_m >= source_z_min_m + boundary_margin_m)
        & (grid_z_m <= source_z_max_m - boundary_margin_m)
    )
    near_min_m = NEAR_ELECTRODE_MIN_DISTANCE_MM * 1.0e-3
    near_max_m = NEAR_ELECTRODE_MAX_DISTANCE_MM * 1.0e-3
    smooth_min_m = SMOOTH_ELECTRODE_DISTANCE_MM * 1.0e-3
    return {
        "valid": boundary_excluded & (electrode_distance_m >= near_min_m),
        "smooth": boundary_excluded & (electrode_distance_m >= smooth_min_m),
        "near_electrode": boundary_excluded & (electrode_distance_m >= near_min_m) & (electrode_distance_m < near_max_m),
    }


def _vector_error_metrics(
    *,
    label: str,
    h_mm: float,
    region: str,
    method_pair: str,
    er_test: np.ndarray,
    ez_test: np.ndarray,
    er_ref: np.ndarray,
    ez_ref: np.ndarray,
    mask: np.ndarray,
    pass_fail: str = "INFO",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    delta = np.sqrt((np.asarray(er_test) - np.asarray(er_ref)) ** 2 + (np.asarray(ez_test) - np.asarray(ez_ref)) ** 2)
    ref_mag = np.sqrt(np.asarray(er_ref) ** 2 + np.asarray(ez_ref) ** 2)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(delta) & np.isfinite(ref_mag)
    result = {
        "field_label": label,
        "grid_h_mm": float(h_mm),
        "region": region,
        "method_pair": method_pair,
        "point_count": int(np.count_nonzero(valid)),
        "pass_fail": pass_fail,
    }
    if not np.any(valid):
        result.update(
            {
                "rms_abs_error_v_per_m": math.nan,
                "reference_p95_v_per_m": math.nan,
                "normalized_rms_error": math.nan,
                "p50_relative_error": math.nan,
                "p95_relative_error": math.nan,
                "p99_relative_error": math.nan,
                "max_relative_error": math.nan,
                "pass_fail": "FAIL",
            }
        )
        if extra:
            result.update(extra)
        return result

    delta_values = delta[valid]
    ref_values = ref_mag[valid]
    reference_p95 = float(np.percentile(ref_values, 95))
    rms_abs = float(np.sqrt(np.mean(delta_values * delta_values)))
    e_floor = max(0.01 * reference_p95, TINY)
    relative_error = delta_values / np.maximum(ref_values, e_floor)
    result.update(
        {
            "rms_abs_error_v_per_m": rms_abs,
            "reference_p95_v_per_m": reference_p95,
            "normalized_rms_error": rms_abs / max(reference_p95, TINY),
            "p50_relative_error": float(np.percentile(relative_error, 50)),
            "p95_relative_error": float(np.percentile(relative_error, 95)),
            "p99_relative_error": float(np.percentile(relative_error, 99)),
            "max_relative_error": float(np.max(relative_error)),
        }
    )
    if extra:
        result.update(extra)
    return result


def _baked_vs_spline_status(h_mm: float, region: str, metric: Mapping[str, Any]) -> str:
    norm = float(metric["normalized_rms_error"])
    p95 = float(metric["p95_relative_error"])
    if region == "smooth":
        if math.isclose(h_mm, 0.10, abs_tol=1.0e-9):
            return "PASS" if norm <= 0.010 and p95 <= 0.020 else "FAIL"
        if h_mm <= 0.05 + 1.0e-12:
            return "PASS" if norm <= 0.0035 and p95 <= 0.010 else "FAIL"
    if region == "near_electrode":
        if math.isclose(h_mm, 0.10, abs_tol=1.0e-9):
            return "PASS" if norm <= 0.050 and p95 <= 0.100 else "FAIL"
        if h_mm <= 0.05 + 1.0e-12:
            return "PASS" if norm <= 0.030 and p95 <= 0.080 else "FAIL"
    return "INFO"


def _convergence_ratio_status(region: str, ratio: float) -> str:
    if region == "smooth":
        return "PASS" if ratio >= 2.5 else "FAIL"
    if region == "near_electrode":
        return "PASS" if ratio >= 1.5 else "FAIL"
    return "INFO"


def _downsample_to_coarser(fine_array: np.ndarray, fine_h_mm: float, coarse_h_mm: float) -> np.ndarray:
    ratio = int(round(coarse_h_mm / fine_h_mm))
    if ratio <= 0 or not math.isclose(ratio * fine_h_mm, coarse_h_mm, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Grid spacing {fine_h_mm} mm does not nest into {coarse_h_mm} mm.")
    return np.asarray(fine_array)[::ratio, ::ratio]


def _run_bakes(args: argparse.Namespace, output_dir: Path) -> List[CaseData]:
    cases: List[CaseData] = []
    for h_mm in args.grid_mm:
        case_dir = output_dir / _case_name(h_mm)
        baked_path = case_dir / "baked_fields.npy"
        plots_dir = case_dir / "plots"
        config = FieldBakeConfig(
            simion_dc_csv=Path(args.simion_dc_patxt),
            simion_rf_csv=Path(args.simion_rf_patxt),
            offset_z_simion_mm=args.offset_z_simion_mm,
            output_npy=baked_path,
            output_plot_dir=plots_dir,
            z_min_mm=DEFAULT_Z_MIN_MM,
            z_max_mm=DEFAULT_Z_MAX_MM,
            r_min_mm=DEFAULT_R_MIN_MM,
            r_max_mm=DEFAULT_R_MAX_MM,
            dz_mm=h_mm,
            dr_mm=h_mm,
            simion_pa_effective_grids_per_mm=args.simion_pa_grids_per_mm,
            simion_dc_voltage_scale=args.simion_dc_voltage_scale,
        )
        if args.skip_bake and baked_path.exists():
            baked = _load_baked(baked_path)
            plot_paths: Tuple[Path, ...] = tuple(sorted(plots_dir.glob("*.png")))
        else:
            result = run_field_bake(config)
            baked = _load_baked(Path(result["output_npy"]))
            plot_paths = tuple(Path(path) for path in result["plot_paths"])
        cases.append(CaseData(float(h_mm), case_dir, baked_path, plot_paths, _build_grid(h_mm), baked))
    return cases


def _make_error_map(
    *,
    output_path: Path,
    title: str,
    grid: GlobalGrid,
    er_test: np.ndarray,
    ez_test: np.ndarray,
    er_ref: np.ndarray,
    ez_ref: np.ndarray,
    mask: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    delta = np.sqrt((er_test - er_ref) ** 2 + (ez_test - ez_ref) ** 2)
    ref_mag = np.sqrt(er_ref**2 + ez_ref**2)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(ref_mag)
    reference_p95 = float(np.percentile(ref_mag[valid], 95)) if np.any(valid) else 1.0
    rel = delta / np.maximum(ref_mag, max(0.01 * reference_p95, TINY))
    rel_plot = np.where(mask, rel, np.nan)

    _, ax = plt.subplots(figsize=(9, 4.8))
    extent = [
        grid.z_min_m * 1.0e3,
        grid.z_max_m * 1.0e3,
        grid.r_min_m * 1.0e3,
        grid.r_max_m * 1.0e3,
    ]
    vmax = float(np.nanpercentile(rel_plot, 99)) if np.any(np.isfinite(rel_plot)) else 1.0
    image = ax.imshow(
        rel_plot,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0.0,
        vmax=max(vmax, TINY),
        cmap="magma",
    )
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title(title)
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("relative vector-E error")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _collect_metrics(
    cases: List[CaseData],
    references: Mapping[str, SourceReference],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[float, Dict[str, Dict[str, np.ndarray]]]]]:
    max_h_mm = max(case.h_mm for case in cases)
    boundary_margin_m = BOUNDARY_MARGIN_FACTOR * max_h_mm * 1.0e-3
    metrics: List[Dict[str, Any]] = []
    evaluations: Dict[str, Dict[float, Dict[str, Dict[str, np.ndarray]]]] = {"dc": {}, "rf": {}}

    for case in cases:
        for label, reference in references.items():
            reference_eval = _evaluate_source_reference(reference, case.grid)
            masks = _region_masks(reference_eval, boundary_margin_m=boundary_margin_m)
            evaluations[label][case.h_mm] = {"reference_eval": reference_eval, "masks": masks}

            simion_fields = case.baked["simion"]
            er_baked = np.asarray(simion_fields[f"e_{label}_r_v_per_m"], dtype=float)
            ez_baked = np.asarray(simion_fields[f"e_{label}_z_v_per_m"], dtype=float)
            er_spline = np.asarray(reference_eval["e_r_v_per_m"], dtype=float)
            ez_spline = np.asarray(reference_eval["e_z_v_per_m"], dtype=float)

            for region, mask in masks.items():
                metric = _vector_error_metrics(
                    label=label,
                    h_mm=case.h_mm,
                    region=region,
                    method_pair="baked_field_vs_spline_derivative",
                    er_test=er_baked,
                    ez_test=ez_baked,
                    er_ref=er_spline,
                    ez_ref=ez_spline,
                    mask=mask,
                )
                metric["pass_fail"] = _baked_vs_spline_status(case.h_mm, region, metric)
                metrics.append(metric)

            if math.isclose(case.h_mm, 0.10, abs_tol=1.0e-9) or math.isclose(case.h_mm, 0.05, abs_tol=1.0e-9):
                _make_error_map(
                    output_path=output_dir / "error_maps" / f"{label}_{_case_name(case.h_mm)}_baked_vs_spline.png",
                    title=f"{label.upper()} h={case.h_mm:.2f} mm: baked field vs spline derivative",
                    grid=case.grid,
                    er_test=er_baked,
                    ez_test=ez_baked,
                    er_ref=er_spline,
                    ez_ref=ez_spline,
                    mask=masks["valid"],
                )

    _append_nested_grid_metrics(cases, references, evaluations, metrics)
    return metrics, evaluations


def _append_nested_grid_metrics(
    cases: List[CaseData],
    references: Mapping[str, SourceReference],
    evaluations: Dict[str, Dict[float, Dict[str, Dict[str, np.ndarray]]]],
    metrics: List[Dict[str, Any]],
) -> None:
    by_h = {case.h_mm: case for case in cases}
    required = {0.20, 0.10, 0.05}
    if not required.issubset(set(by_h)):
        return

    pair_metrics: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for label in references:
        case_020 = by_h[0.20]
        case_010 = by_h[0.10]
        case_005 = by_h[0.05]
        comparisons = [
            (0.20, 0.10, case_020, case_020.baked["simion"], case_010.baked["simion"], "grid_0p20_vs_0p10"),
            (0.10, 0.05, case_010, case_010.baked["simion"], case_005.baked["simion"], "grid_0p10_vs_0p05"),
        ]
        for coarse_h, fine_h, _coarse_case, coarse_fields, fine_fields, method_pair in comparisons:
            er_coarse = np.asarray(coarse_fields[f"e_{label}_r_v_per_m"], dtype=float)
            ez_coarse = np.asarray(coarse_fields[f"e_{label}_z_v_per_m"], dtype=float)
            er_fine = _downsample_to_coarser(np.asarray(fine_fields[f"e_{label}_r_v_per_m"], dtype=float), fine_h, coarse_h)
            ez_fine = _downsample_to_coarser(np.asarray(fine_fields[f"e_{label}_z_v_per_m"], dtype=float), fine_h, coarse_h)
            masks = evaluations[label][coarse_h]["masks"]
            for region, mask in masks.items():
                metric = _vector_error_metrics(
                    label=label,
                    h_mm=coarse_h,
                    region=region,
                    method_pair=method_pair,
                    er_test=er_coarse,
                    ez_test=ez_coarse,
                    er_ref=er_fine,
                    ez_ref=ez_fine,
                    mask=mask,
                )
                pair_metrics.setdefault((label, region), {})[method_pair] = metric
                metrics.append(metric)

    for (label, region), pair_group in pair_metrics.items():
        coarse = pair_group.get("grid_0p20_vs_0p10")
        fine = pair_group.get("grid_0p10_vs_0p05")
        if coarse is None or fine is None:
            continue
        numerator = float(coarse["rms_abs_error_v_per_m"])
        denominator = float(fine["rms_abs_error_v_per_m"])
        convergence_note = ""
        if numerator <= TINY and denominator <= TINY:
            ratio = math.inf
            convergence_note = "exact_on_nested_nodes"
            pass_fail = "PASS"
        else:
            ratio = numerator / max(denominator, TINY)
            pass_fail = _convergence_ratio_status(region, ratio)
        metrics.append(
            {
                "field_label": label,
                "grid_h_mm": "triplet",
                "region": region,
                "method_pair": "convergence_ratio_(0p20_vs_0p10)/(0p10_vs_0p05)",
                "point_count": min(int(coarse["point_count"]), int(fine["point_count"])),
                "rms_abs_error_v_per_m": math.nan,
                "reference_p95_v_per_m": math.nan,
                "normalized_rms_error": math.nan,
                "p50_relative_error": math.nan,
                "p95_relative_error": math.nan,
                "p99_relative_error": math.nan,
                "max_relative_error": math.nan,
                "convergence_ratio": ratio,
                "convergence_note": convergence_note,
                "pass_fail": pass_fail,
            }
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_metrics(metrics: List[Dict[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    json_path = output_dir / "convergence_metrics.json"
    csv_path = output_dir / "convergence_metrics.csv"
    json_path.write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    fieldnames = sorted({key for metric in metrics for key in metric.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)
    return json_path, csv_path


def _format_percent(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{100.0 * numeric:.3g}%"


def _format_float(value: Any, *, precision: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{precision}g}"


def _format_ratio(metric: Mapping[str, Any]) -> str:
    if metric.get("convergence_note") == "exact_on_nested_nodes":
        return "exact"
    return _format_float(metric.get("convergence_ratio"), precision=3)


def _table(rows: Iterable[Mapping[str, Any]], columns: List[Tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _strict_metrics(metrics: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [metric for metric in metrics if metric.get("pass_fail") in {"PASS", "FAIL"}]


def _write_report(
    *,
    output_dir: Path,
    cases: List[CaseData],
    references: Mapping[str, SourceReference],
    metrics: List[Dict[str, Any]],
    metrics_json_path: Path,
    metrics_csv_path: Path,
) -> Path:
    strict = _strict_metrics(metrics)
    failed = [metric for metric in strict if metric.get("pass_fail") == "FAIL"]
    verdict = "PASS" if not failed else "FAIL"

    gradient_rows = []
    for metric in metrics:
        if metric.get("method_pair") != "baked_field_vs_spline_derivative":
            continue
        if metric.get("region") not in {"smooth", "near_electrode"}:
            continue
        if float(metric.get("grid_h_mm", 999.0)) > 0.10 + 1.0e-12:
            continue
        gradient_rows.append(
            {
                "field": metric["field_label"],
                "h": f"{float(metric['grid_h_mm']):.2f}",
                "region": metric["region"],
                "norm_rms": _format_percent(metric["normalized_rms_error"]),
                "p95": _format_percent(metric["p95_relative_error"]),
                "p99": _format_percent(metric["p99_relative_error"]),
                "status": metric["pass_fail"],
            }
        )

    ratio_rows = []
    for metric in metrics:
        if metric.get("method_pair") != "convergence_ratio_(0p20_vs_0p10)/(0p10_vs_0p05)":
            continue
        if metric.get("region") not in {"smooth", "near_electrode"}:
            continue
        ratio_rows.append(
            {
                "field": metric["field_label"],
                "region": metric["region"],
                "ratio": _format_ratio(metric),
                "status": metric["pass_fail"],
            }
        )

    case_rows = [
        {"h": f"{case.h_mm:.2f}", "shape": f"{len(case.grid.r_coords_m)} x {len(case.grid.z_coords_m)}", "baked": str(case.baked_path)}
        for case in cases
    ]
    dc_source_spacing_mm = references["dc"].metadata.get("pa_source_spacing_mm", "unknown")
    dc_grids_per_mm = references["dc"].metadata.get("pa_effective_grids_per_mm", "unknown")
    grid_h_text = " / ".join(f"{case.h_mm:.2f}" for case in cases)
    rf_meta = cases[-1].baked["simion"]
    report_path = output_dir / "field_convergence_report.md"
    lines = [
        "# Field Convergence Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Overall verdict: **{verdict}**",
        "",
        "## Assumptions",
        "",
        "- SIMION has already refined the DC and RF potential arrays.",
        "- Python only maps coordinates, interpolates the refined potential, and differentiates it.",
        "- RF input is fixed to adjacent RF electrodes at `+1 V / -1 V`, so `Vref = 1 V`.",
        f"- The native PA spacing is `{dc_source_spacing_mm} mm` from `{dc_grids_per_mm} grids/mm`; tested global spacings are `{grid_h_text} mm`.",
        "- Vector field error is `sqrt(dEr^2 + dEz^2)`, normalized by the region's P95 reference field magnitude.",
        "",
        "## Baked Cases",
        "",
        _table(case_rows, [("h [mm]", "h"), ("shape [nr x nz]", "shape"), ("baked file", "baked")]),
        "",
        "## RF Metadata Check",
        "",
        f"- `rf_normalization`: `{rf_meta.get('rf_normalization')}`",
        f"- `rf_reference_peak_voltage_v`: `{rf_meta.get('rf_reference_peak_voltage_v')}`",
        f"- `rf_pa_electrode_min_v`: `{rf_meta.get('rf_pa_electrode_min_v')}`",
        f"- `rf_pa_electrode_max_v`: `{rf_meta.get('rf_pa_electrode_max_v')}`",
        f"- `rf_pa_coordinate_mapping`: `{rf_meta.get('rf_pa_coordinate_mapping')}`",
        f"- new `rf_voltage_scale` key present: `{('rf_voltage_scale' in rf_meta)}`",
        "",
        "## Baked Field vs Spline Derivative",
        "",
        _table(
            gradient_rows,
            [
                ("field", "field"),
                ("h [mm]", "h"),
                ("region", "region"),
                ("norm RMS", "norm_rms"),
                ("P95 rel", "p95"),
                ("P99 rel", "p99"),
                ("status", "status"),
            ],
        ),
        "",
        "Pass criteria:",
        "",
        "- smooth `h=0.10 mm`: normalized RMS <= 1.0%, P95 <= 2.0%.",
        "- smooth `h<=0.05 mm`: normalized RMS <= 0.35%, P95 <= 1.0%.",
        "- near-electrode `h=0.10 mm`: normalized RMS <= 5.0%, P95 <= 10.0%.",
        "- near-electrode `h<=0.05 mm`: normalized RMS <= 3.0%, P95 <= 8.0%.",
        "",
        "## Nested-Grid Convergence",
        "",
        _table(ratio_rows, [("field", "field"), ("region", "region"), ("RMS ratio", "ratio"), ("status", "status")]),
        "",
        "The ratio is `RMS(0.20 mm vs 0.10 mm) / RMS(0.10 mm vs 0.05 mm)`. A second-order finite-difference method should approach 4 in smooth regions. If the baker exports spline analytic derivatives, nested-grid values at common nodes can agree exactly; those rows are marked `exact`.",
        "",
        "## Output Files",
        "",
        f"- Metrics JSON: `{metrics_json_path}`",
        f"- Metrics CSV: `{metrics_csv_path}`",
        f"- Error maps: `{output_dir / 'error_maps'}`",
        "",
    ]

    if failed:
        failed_rows = [
            {
                "field": metric["field_label"],
                "h": str(metric["grid_h_mm"]),
                "region": metric["region"],
                "pair": metric["method_pair"],
                "norm_rms": _format_percent(metric.get("normalized_rms_error")),
                "p95": _format_percent(metric.get("p95_relative_error")),
                "ratio": _format_ratio(metric),
            }
            for metric in failed
        ]
        lines.extend(
            [
                "## Failed Strict Checks",
                "",
                _table(
                    failed_rows,
                    [
                        ("field", "field"),
                        ("h", "h"),
                        ("region", "region"),
                        ("pair", "pair"),
                        ("norm RMS", "norm_rms"),
                        ("P95 rel", "p95"),
                        ("ratio", "ratio"),
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- If baked-vs-spline fails while nested-grid convergence passes, inspect the baker derivative method and boundary masking before changing physics settings.",
            "- If smooth-region nested-grid convergence fails, Python-side grid refinement is not enough; repeat the test with a finer SIMION PA/refine export.",
            "- Near-electrode failures should be interpreted together with trajectory/electrode-hit logic, because ions entering this region may be physically absorbed rather than freely propagated.",
            "",
            "## Source References",
            "",
        ]
    )
    for label, reference in references.items():
        lines.append(f"- `{label}`: `{reference.path}`, metadata: `{reference.metadata}`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run field-baker convergence diagnostics and write a Markdown report.")
    parser.add_argument("--simion-dc-patxt", default="data/fields/simion_refined/slens_dc.patxt")
    parser.add_argument("--simion-rf-patxt", default="data/fields/simion_refined/slens_rf.patxt")
    parser.add_argument("--output-dir", default="outputs/field_convergence")
    parser.add_argument("--simion-pa-grids-per-mm", type=float, default=DEFAULT_SIMION_GRIDS_PER_MM)
    parser.add_argument("--simion-dc-voltage-scale", type=float, default=DEFAULT_DC_SCALE)
    parser.add_argument("--offset-z-simion-mm", type=float, default=0.0)
    parser.add_argument("--grid-mm", type=float, nargs="+", default=list(DEFAULT_GRID_MM))
    parser.add_argument("--skip-bake", action="store_true", help="Reuse existing baked files when present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = {
        "dc": _build_source_reference(
            label="dc",
            path=Path(args.simion_dc_patxt),
            voltage_scale=args.simion_dc_voltage_scale,
            pa_effective_grids_per_mm=args.simion_pa_grids_per_mm,
            z_offset_mm=args.offset_z_simion_mm,
        ),
        "rf": _build_source_reference(
            label="rf",
            path=Path(args.simion_rf_patxt),
            voltage_scale=RF_REFERENCE_PEAK_VOLTAGE_V,
            pa_effective_grids_per_mm=args.simion_pa_grids_per_mm,
            z_offset_mm=args.offset_z_simion_mm,
        ),
    }
    cases = _run_bakes(args, output_dir)
    metrics, _ = _collect_metrics(cases, references, output_dir)
    metrics_json_path, metrics_csv_path = _write_metrics(metrics, output_dir)
    report_path = _write_report(
        output_dir=output_dir,
        cases=cases,
        references=references,
        metrics=metrics,
        metrics_json_path=metrics_json_path,
        metrics_csv_path=metrics_csv_path,
    )
    print(f"Saved field convergence report: {report_path}")
    print(f"Saved metrics JSON: {metrics_json_path}")
    print(f"Saved metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()
