"""Bake regular Cartesian 3D electric fields for multipole ion guides.

The existing ``field_baker.py`` remains the validated 2D axisymmetric baker. This
tool creates a separate ``grid3d/electric3d`` artifact that can be added to the
runtime as a true Cartesian static electric field.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


RF_REFERENCE_PEAK_VOLTAGE_V = 1.0
RF_RUNTIME_VOLTAGE_PARAMETER = "rf_peak_voltage_v"
RF_SCALING_EQUATION = (
    "E_rf(t)=E_rf_ref*(rf_peak_voltage_v/rf_reference_peak_voltage_v)"
    "*cos(2*pi*rf_frequency_hz*t+rf_phase_rad)"
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


@dataclass(frozen=True)
class CartesianField3DBakeConfig:
    """One 3D Cartesian electric-field bake job."""

    dc_csv: Optional[Path] = None
    rf_csv: Optional[Path] = None
    output_npy: Path = Path("outputs/baked_fields_3d.npy")
    coordinate_frame: str = "global"
    length_unit: str = "mm"
    dc_voltage_scale: float = 1.0
    rf_reference_peak_voltage_v: float = RF_REFERENCE_PEAK_VOLTAGE_V


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", header.strip().lower())


def _read_csv_columns(csv_path: Path) -> dict[str, np.ndarray]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        raw_fieldnames = next(reader, None)
        if raw_fieldnames is None:
            raise ValueError(f"CSV file is missing a header: {csv_path}")

        normalized_names: list[str] = []
        seen: dict[str, int] = {}
        for raw_name in raw_fieldnames:
            base_name = _normalize_header(raw_name)
            count = seen.get(base_name, 0) + 1
            seen[base_name] = count
            normalized_names.append(base_name if count == 1 else f"{base_name}{count}")

        buckets: dict[str, list[float]] = {key: [] for key in normalized_names}
        for row in reader:
            for index, normalized_name in enumerate(normalized_names):
                raw_value = row[index] if index < len(row) else ""
                if raw_value is None or raw_value.strip() == "":
                    buckets[normalized_name].append(np.nan)
                else:
                    buckets[normalized_name].append(float(raw_value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in buckets.items()}


def _pick_column(columns: dict[str, np.ndarray], aliases: Sequence[str], *, csv_path: Path, required: bool = True) -> Optional[np.ndarray]:
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    if not required:
        return None
    available = ", ".join(sorted(columns.keys()))
    alias_text = ", ".join(aliases)
    raise KeyError(f"Could not find any of [{alias_text}] in {csv_path}; available columns: {available}")


def _length_scale(unit: str) -> float:
    normalized = str(unit).strip().lower()
    if normalized == "m":
        return 1.0
    if normalized == "mm":
        return 1.0e-3
    raise ValueError("--length-unit must be 'm' or 'mm'.")


def _field_key_aliases(component: str) -> tuple[str, ...]:
    return (
        f"e{component}vperm",
        f"e{component}vm",
        f"electricfield{component}vperm",
        f"electricfield{component}vm",
        f"{component}electricfieldvperm",
        f"{component}electricfieldvm",
    )


def _extract_global_coordinates(
    columns: dict[str, np.ndarray],
    *,
    csv_path: Path,
    coordinate_frame: str,
    length_unit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_x = _pick_column(columns, ("xm", "xmm", "x", "xcoordinate", "xcoord"), csv_path=csv_path)
    raw_y = _pick_column(columns, ("ym", "ymm", "y", "ycoordinate", "ycoord"), csv_path=csv_path)
    raw_z = _pick_column(columns, ("zm", "zmm", "z", "zcoordinate", "zcoord"), csv_path=csv_path)
    assert raw_x is not None and raw_y is not None and raw_z is not None
    scale = _length_scale(length_unit)
    x = np.asarray(raw_x, dtype=np.float64) * scale
    y = np.asarray(raw_y, dtype=np.float64) * scale
    z = np.asarray(raw_z, dtype=np.float64) * scale

    frame = str(coordinate_frame).strip().lower()
    if frame == "global":
        return x, y, z
    if frame == "simion":
        # Project convention: SIMION x -> global z, y -> global x, z -> global y.
        return y, z, x
    raise ValueError("--coordinate-frame must be 'global' or 'simion'.")


def _extract_global_vector_components(
    columns: dict[str, np.ndarray],
    *,
    csv_path: Path,
    coordinate_frame: str,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    raw_ex = _pick_column(columns, _field_key_aliases("x"), csv_path=csv_path, required=False)
    raw_ey = _pick_column(columns, _field_key_aliases("y"), csv_path=csv_path, required=False)
    raw_ez = _pick_column(columns, _field_key_aliases("z"), csv_path=csv_path, required=False)
    if raw_ex is None or raw_ey is None or raw_ez is None:
        return None

    ex = np.asarray(raw_ex, dtype=np.float64)
    ey = np.asarray(raw_ey, dtype=np.float64)
    ez = np.asarray(raw_ez, dtype=np.float64)
    frame = str(coordinate_frame).strip().lower()
    if frame == "global":
        return ex, ey, ez
    if frame == "simion":
        # Vector mapping mirrors the project velocity convention:
        # SIMION Vx -> global vz, Vy -> global vx, Vz -> global vy.
        return ey, ez, ex
    raise ValueError("--coordinate-frame must be 'global' or 'simion'.")


def _extract_potential(columns: dict[str, np.ndarray], *, csv_path: Path, voltage_scale: float) -> Optional[np.ndarray]:
    raw_phi = _pick_column(
        columns,
        ("potentialv", "potential", "voltagev", "voltage", "phiv", "phi", "v"),
        csv_path=csv_path,
        required=False,
    )
    if raw_phi is None:
        return None
    return np.asarray(raw_phi, dtype=np.float64) * float(voltage_scale)


def _regular_axes_and_indices(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_valid = np.asarray(x_m[valid_mask], dtype=np.float64)
    y_valid = np.asarray(y_m[valid_mask], dtype=np.float64)
    z_valid = np.asarray(z_m[valid_mask], dtype=np.float64)

    x_coords = np.unique(x_valid)
    y_coords = np.unique(y_valid)
    z_coords = np.unique(z_valid)
    if x_coords.size < 2 or y_coords.size < 2 or z_coords.size < 2:
        raise ValueError(
            "3D field CSV must contain at least two unique nodes on each global axis "
            f"(got nx={x_coords.size}, ny={y_coords.size}, nz={z_coords.size})."
        )
    _validate_uniform_axis("x", x_coords)
    _validate_uniform_axis("y", y_coords)
    _validate_uniform_axis("z", z_coords)

    i = np.searchsorted(x_coords, x_valid)
    j = np.searchsorted(y_coords, y_valid)
    k = np.searchsorted(z_coords, z_valid)
    return x_coords, y_coords, z_coords, i, j, k


def _validate_uniform_axis(name: str, coords_m: np.ndarray) -> None:
    diffs = np.diff(coords_m)
    if not np.all(diffs > 0.0):
        raise ValueError(f"{name} coordinates must be strictly increasing.")
    step = float(np.median(diffs))
    tolerance = max(abs(step) * 1.0e-9, 1.0e-15)
    max_error = float(np.max(np.abs(diffs - step)))
    if max_error > tolerance:
        raise ValueError(f"{name} coordinates must be uniformly spaced; max spacing error={max_error:.6g} m.")


def _scatter_regular_values(
    values: np.ndarray,
    shape: tuple[int, int, int],
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    for ii, jj, kk, value in zip(i, j, k, values):
        out[ii, jj, kk] += float(value)
        counts[ii, jj, kk] += 1
    missing = counts == 0
    if np.any(missing):
        raise ValueError(f"Regular 3D grid is incomplete; missing {int(np.count_nonzero(missing))} nodes.")
    return out / counts


def _electric_from_potential(
    phi_v: np.ndarray,
    x_coords_m: np.ndarray,
    y_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_order = 2 if min(phi_v.shape) >= 3 else 1
    dphi_dx, dphi_dy, dphi_dz = np.gradient(
        phi_v,
        x_coords_m,
        y_coords_m,
        z_coords_m,
        edge_order=edge_order,
    )
    return -dphi_dx, -dphi_dy, -dphi_dz


def _load_regular_field_csv(
    csv_path: Path,
    *,
    coordinate_frame: str,
    length_unit: str,
    voltage_scale: float,
) -> dict[str, Any]:
    columns = _read_csv_columns(csv_path)
    x_m, y_m, z_m = _extract_global_coordinates(
        columns,
        csv_path=csv_path,
        coordinate_frame=coordinate_frame,
        length_unit=length_unit,
    )
    phi_v = _extract_potential(columns, csv_path=csv_path, voltage_scale=voltage_scale)
    direct_e = _extract_global_vector_components(columns, csv_path=csv_path, coordinate_frame=coordinate_frame)

    if phi_v is None and direct_e is None:
        raise ValueError(
            f"{csv_path} must contain either potential columns or all Ex/Ey/Ez field-component columns."
        )

    valid_mask = np.isfinite(x_m) & np.isfinite(y_m) & np.isfinite(z_m)
    if phi_v is not None:
        valid_mask &= np.isfinite(phi_v)
    if direct_e is not None:
        valid_mask &= np.isfinite(direct_e[0]) & np.isfinite(direct_e[1]) & np.isfinite(direct_e[2])
    if not np.any(valid_mask):
        raise ValueError(f"No valid 3D field rows remain after filtering: {csv_path}")

    x_coords, y_coords, z_coords, i, j, k = _regular_axes_and_indices(x_m, y_m, z_m, valid_mask)
    shape = (len(x_coords), len(y_coords), len(z_coords))
    phi_grid_v = None
    if phi_v is not None:
        phi_grid_v = _scatter_regular_values(phi_v[valid_mask], shape, i, j, k)
    if direct_e is not None:
        ex_v_per_m = _scatter_regular_values(direct_e[0][valid_mask], shape, i, j, k)
        ey_v_per_m = _scatter_regular_values(direct_e[1][valid_mask], shape, i, j, k)
        ez_v_per_m = _scatter_regular_values(direct_e[2][valid_mask], shape, i, j, k)
        derivative_method = "direct_field_components"
    elif phi_grid_v is not None:
        ex_v_per_m, ey_v_per_m, ez_v_per_m = _electric_from_potential(phi_grid_v, x_coords, y_coords, z_coords)
        derivative_method = "numpy_gradient_regular_cartesian"
    else:
        raise AssertionError("3D field CSV validation allowed neither potential nor field components.")

    return {
        "x_coords_m": x_coords,
        "y_coords_m": y_coords,
        "z_coords_m": z_coords,
        "phi_v": phi_grid_v,
        "e_x_v_per_m": ex_v_per_m,
        "e_y_v_per_m": ey_v_per_m,
        "e_z_v_per_m": ez_v_per_m,
        "source_path": str(csv_path),
        "source_rows": int(x_m.size),
        "valid_rows": int(np.count_nonzero(valid_mask)),
        "derivative_method": derivative_method,
    }


def _same_grid(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return (
        np.array_equal(first["x_coords_m"], second["x_coords_m"])
        and np.array_equal(first["y_coords_m"], second["y_coords_m"])
        and np.array_equal(first["z_coords_m"], second["z_coords_m"])
    )


def bake_cartesian_field3d(config: CartesianField3DBakeConfig) -> dict[str, Any]:
    """Bake DC/RF CSV inputs into the canonical ``grid3d/electric3d`` dictionary."""

    if config.dc_csv is None and config.rf_csv is None:
        raise ValueError("At least one of dc_csv or rf_csv must be provided.")
    if config.rf_reference_peak_voltage_v <= 0.0:
        raise ValueError("rf_reference_peak_voltage_v must be positive.")

    dc = None
    if config.dc_csv is not None:
        dc = _load_regular_field_csv(
            Path(config.dc_csv),
            coordinate_frame=config.coordinate_frame,
            length_unit=config.length_unit,
            voltage_scale=config.dc_voltage_scale,
        )
    rf = None
    if config.rf_csv is not None:
        rf = _load_regular_field_csv(
            Path(config.rf_csv),
            coordinate_frame=config.coordinate_frame,
            length_unit=config.length_unit,
            voltage_scale=1.0,
        )

    reference = dc if dc is not None else rf
    assert reference is not None
    if dc is not None and not _same_grid(reference, dc):
        raise ValueError("DC field grid does not match the reference 3D grid.")
    if rf is not None and not _same_grid(reference, rf):
        raise ValueError("RF field grid does not match the reference 3D grid.")

    shape = (
        len(reference["x_coords_m"]),
        len(reference["y_coords_m"]),
        len(reference["z_coords_m"]),
    )
    zeros = np.zeros(shape, dtype=np.float64)
    output = {
        "grid3d": {
            "x_coords_m": reference["x_coords_m"],
            "y_coords_m": reference["y_coords_m"],
            "z_coords_m": reference["z_coords_m"],
            "x_min_m": float(reference["x_coords_m"][0]),
            "x_max_m": float(reference["x_coords_m"][-1]),
            "y_min_m": float(reference["y_coords_m"][0]),
            "y_max_m": float(reference["y_coords_m"][-1]),
            "z_min_m": float(reference["z_coords_m"][0]),
            "z_max_m": float(reference["z_coords_m"][-1]),
            "dx_m": float(np.median(np.diff(reference["x_coords_m"]))),
            "dy_m": float(np.median(np.diff(reference["y_coords_m"]))),
            "dz_m": float(np.median(np.diff(reference["z_coords_m"]))),
            "nx": int(shape[0]),
            "ny": int(shape[1]),
            "nz": int(shape[2]),
            "coordinate_frame": "global_python_cartesian",
            "source_coordinate_frame": str(config.coordinate_frame),
            "source_length_unit": str(config.length_unit),
            "shape_order": "x_y_z",
        },
        "electric3d": {
            "e_dc_x_v_per_m": zeros.copy() if dc is None else dc["e_x_v_per_m"],
            "e_dc_y_v_per_m": zeros.copy() if dc is None else dc["e_y_v_per_m"],
            "e_dc_z_v_per_m": zeros.copy() if dc is None else dc["e_z_v_per_m"],
            "e_rf_x_v_per_m": zeros.copy() if rf is None else rf["e_x_v_per_m"],
            "e_rf_y_v_per_m": zeros.copy() if rf is None else rf["e_y_v_per_m"],
            "e_rf_z_v_per_m": zeros.copy() if rf is None else rf["e_z_v_per_m"],
            "rf_reference_peak_voltage_v": float(config.rf_reference_peak_voltage_v),
            "rf_runtime_voltage_parameter": RF_RUNTIME_VOLTAGE_PARAMETER,
            "rf_scaling_equation": RF_SCALING_EQUATION,
            "dc_source_path": None if dc is None else dc["source_path"],
            "rf_source_path": None if rf is None else rf["source_path"],
            "dc_derivative_method": None if dc is None else dc["derivative_method"],
            "rf_derivative_method": None if rf is None else rf["derivative_method"],
            "dc_voltage_scale": float(config.dc_voltage_scale),
        },
    }
    if dc is not None and dc["phi_v"] is not None:
        output["electric3d"]["phi_dc_v"] = dc["phi_v"]
    if rf is not None and rf["phi_v"] is not None:
        output["electric3d"]["phi_rf_v"] = rf["phi_v"]
    return output


def run_bake(config: CartesianField3DBakeConfig) -> dict[str, Any]:
    baked = bake_cartesian_field3d(config)
    output_path = Path(config.output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, baked, allow_pickle=True)
    grid = baked["grid3d"]
    return {
        "output_npy": output_path,
        "shape": (int(grid["nx"]), int(grid["ny"]), int(grid["nz"])),
        "x_range_m": (float(grid["x_min_m"]), float(grid["x_max_m"])),
        "y_range_m": (float(grid["y_min_m"]), float(grid["y_max_m"])),
        "z_range_m": (float(grid["z_min_m"]), float(grid["z_max_m"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake a regular Cartesian 3D DC/RF electric field for multipole ion-guide simulations."
    )
    parser.add_argument("--dc-csv", default="", help="Regular-grid DC potential or Ex/Ey/Ez CSV.")
    parser.add_argument("--rf-csv", default="", help="Regular-grid RF reference potential or Ex/Ey/Ez CSV.")
    parser.add_argument("--output-npy", default="outputs/baked_fields_3d.npy", help="Output 3D baked field .npy path.")
    parser.add_argument(
        "--coordinate-frame",
        choices=("global", "simion"),
        default="global",
        help="Interpret CSV x/y/z as global Python coordinates or SIMION coordinates.",
    )
    parser.add_argument("--length-unit", choices=("m", "mm"), default="mm", help="Coordinate unit used by CSV x/y/z columns.")
    parser.add_argument(
        "--dc-voltage-scale",
        type=float,
        default=1.0,
        help="Scale applied to DC potential values before differentiating. Direct Ex/Ey/Ez columns are not scaled.",
    )
    parser.add_argument(
        "--rf-reference-peak-voltage",
        type=float,
        default=RF_REFERENCE_PEAK_VOLTAGE_V,
        help="Reference peak voltage represented by the RF arrays. Runtime still uses --rf-peak-voltage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bake(
        CartesianField3DBakeConfig(
            dc_csv=Path(args.dc_csv) if args.dc_csv else None,
            rf_csv=Path(args.rf_csv) if args.rf_csv else None,
            output_npy=Path(args.output_npy),
            coordinate_frame=str(args.coordinate_frame),
            length_unit=str(args.length_unit),
            dc_voltage_scale=float(args.dc_voltage_scale),
            rf_reference_peak_voltage_v=float(args.rf_reference_peak_voltage),
        )
    )
    shape = result["shape"]
    print(
        "Saved 3D baked field: "
        f"{result['output_npy']} shape={shape}, "
        f"x={result['x_range_m'][0] * 1.0e3:.6g}..{result['x_range_m'][1] * 1.0e3:.6g} mm, "
        f"y={result['y_range_m'][0] * 1.0e3:.6g}..{result['y_range_m'][1] * 1.0e3:.6g} mm, "
        f"z={result['z_range_m'][0] * 1.0e3:.6g}..{result['z_range_m'][1] * 1.0e3:.6g} mm"
    )


if __name__ == "__main__":
    main()
