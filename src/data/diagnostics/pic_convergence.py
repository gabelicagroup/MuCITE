"""PIC mesh and macro-coupling convergence diagnostics.

This module deliberately separates two error sources:

* a manufactured axisymmetric Poisson problem exercises the production
  finite-volume sparse solver on successively refined meshes;
* a linear plasma-oscillation proxy measures the error introduced when the
  self field is refreshed only once per PIC macro step.

The oscillator is a coupling diagnostic, not a replacement for an
application-specific particle convergence study.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.constants import elementary_charge, physical_constants

from ...env.pic.poisson import EPSILON_0, AxisymmetricSparsePoissonSolver


ATOMIC_MASS_CONSTANT_KG = float(physical_constants["atomic mass constant"][0])
DEFAULT_NUMBER_DENSITY_M3 = 1.0e14
DEFAULT_ION_MASS_AMU = 100.0
DEFAULT_MACRO_DT_S = (2.0e-7, 1.0e-7, 5.0e-8, 2.5e-8)
DEFAULT_REFERENCE_MACRO_DT_S = 5.0e-8
DEFAULT_MAX_MACRO_PHASE_RAD = 0.2
DEFAULT_GRID_SHAPES = ((17, 49), (33, 97), (65, 193))


@dataclass(frozen=True)
class _GeometryGrid:
    """Minimal geometry consumed by the production sparse Poisson backend."""

    nr: int
    nz: int
    r_max_m: float
    z_min_m: float
    z_max_m: float

    @property
    def dr(self) -> float:
        return self.r_max_m / float(self.nr - 1)

    @property
    def dz(self) -> float:
        return (self.z_max_m - self.z_min_m) / float(self.nz - 1)


def plasma_frequency_rad_per_s(
    number_density_m3: float,
    *,
    charge_c: float,
    mass_kg: float,
) -> float:
    """Return ``omega_p = sqrt(n q^2 / (epsilon_0 m))``."""

    number_density_m3 = float(number_density_m3)
    charge_c = float(charge_c)
    mass_kg = float(mass_kg)
    if not math.isfinite(number_density_m3) or number_density_m3 <= 0.0:
        raise ValueError("number_density_m3 must be finite and positive.")
    if not math.isfinite(charge_c) or charge_c == 0.0:
        raise ValueError("charge_c must be finite and non-zero.")
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be finite and positive.")
    return math.sqrt(number_density_m3 * charge_c * charge_c / (EPSILON_0 * mass_kg))


def _manufactured_solution(
    grid: _GeometryGrid,
    *,
    amplitude_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact potential and the charge density that generates it.

    The selected solution

    ``phi = A [1 - (r/R)^2] sin(pi z/L)``

    obeys the runtime boundary conditions: axial and outer-radius Dirichlet
    zero, plus symmetry ``dphi/dr = 0`` at the axis.
    """

    r_m = np.arange(grid.nr, dtype=np.float64) * grid.dr
    z_m = grid.z_min_m + np.arange(grid.nz, dtype=np.float64) * grid.dz
    rr_m, zz_m = np.meshgrid(r_m, z_m, indexing="ij")
    length_m = grid.z_max_m - grid.z_min_m
    axial_phase = math.pi * (zz_m - grid.z_min_m) / length_m
    radial_shape = 1.0 - (rr_m / grid.r_max_m) ** 2
    wave_number = math.pi / length_m

    phi_exact_v = float(amplitude_v) * radial_shape * np.sin(axial_phase)
    # Cylindrical Laplacian:
    # div(grad(phi)) = -A sin(kz) [4/R^2 + k^2(1-r^2/R^2)]
    rho_c_per_m3 = (
        EPSILON_0
        * float(amplitude_v)
        * np.sin(axial_phase)
        * (4.0 / grid.r_max_m**2 + wave_number**2 * radial_shape)
    )
    return phi_exact_v, rho_c_per_m3


def run_manufactured_poisson_convergence(
    *,
    grid_shapes: Sequence[tuple[int, int]] = DEFAULT_GRID_SHAPES,
    r_max_m: float = 4.0e-3,
    z_max_m: float = 12.0e-3,
    amplitude_v: float = 5.0,
) -> dict[str, Any]:
    """Run a nested-grid convergence study through the production solver."""

    if len(grid_shapes) < 2:
        raise ValueError("At least two grid shapes are required for convergence.")
    rows: list[dict[str, Any]] = []
    previous_error: float | None = None
    previous_h_m: float | None = None

    for nr_value, nz_value in grid_shapes:
        nr = int(nr_value)
        nz = int(nz_value)
        if nr < 5 or nz < 5:
            raise ValueError("Manufactured-solution grids require at least 5x5 nodes.")
        grid = _GeometryGrid(
            nr=nr,
            nz=nz,
            r_max_m=float(r_max_m),
            z_min_m=0.0,
            z_max_m=float(z_max_m),
        )
        phi_exact_v, rho_c_per_m3 = _manufactured_solution(
            grid,
            amplitude_v=float(amplitude_v),
        )
        solver = AxisymmetricSparsePoissonSolver(
            mode="sparse_direct",
            tolerance=1.0e-11,
            max_iters=1,
            warm_start=False,
        )
        solver.setup(grid)
        result = solver.solve(rho_c_per_m3)
        active = ~np.asarray(solver.dirichlet_mask, dtype=bool)
        delta_v = np.asarray(result.phi, dtype=np.float64)[active] - phi_exact_v[active]
        exact_active_v = phi_exact_v[active]
        rms_error_v = float(np.sqrt(np.mean(delta_v * delta_v)))
        reference_rms_v = float(np.sqrt(np.mean(exact_active_v * exact_active_v)))
        relative_l2_error = rms_error_v / max(reference_rms_v, np.finfo(float).tiny)
        h_m = max(grid.dr, grid.dz)

        error_ratio = None
        observed_order = None
        if previous_error is not None and previous_h_m is not None:
            error_ratio = previous_error / relative_l2_error
            observed_order = math.log(error_ratio) / math.log(previous_h_m / h_m)
        rows.append(
            {
                "nr": nr,
                "nz": nz,
                "node_count": nr * nz,
                "dr_m": grid.dr,
                "dz_m": grid.dz,
                "h_m": h_m,
                "relative_l2_phi_error": relative_l2_error,
                "max_abs_phi_error_v": float(np.max(np.abs(delta_v))),
                "error_ratio_from_previous": error_ratio,
                "observed_order": observed_order,
                "poisson_relative_residual": float(result.relative_residual),
                "poisson_converged": bool(result.converged),
                "setup_time_ms": float(result.setup_time_ms),
                "solve_time_ms": float(result.runtime_ms),
            }
        )
        previous_error = relative_l2_error
        previous_h_m = h_m

    refinement_rows = rows[1:]
    passed = bool(
        all(row["poisson_converged"] for row in rows)
        and all(float(row["error_ratio_from_previous"]) > 1.0 for row in refinement_rows)
        and all(float(row["observed_order"]) >= 1.8 for row in refinement_rows)
    )
    return {
        "method": "production_axisymmetric_finite_volume_sparse_direct",
        "manufactured_potential": "A*(1-(r/R)^2)*sin(pi*z/L)",
        "acceptance": {
            "all_poisson_solves_converged": True,
            "error_decreases_each_refinement": True,
            "minimum_observed_order": 1.8,
        },
        "rows": rows,
        "passed": passed,
    }


def _frozen_field_oscillator_error(
    *,
    omega_rad_per_s: float,
    macro_dt_s: float,
    duration_s: float,
) -> dict[str, float]:
    """Integrate ``x''=-omega^2*x_n`` with the field frozen per macro step."""

    x = 1.0
    velocity = 0.0
    time_s = 0.0
    step_count = 0
    time_tolerance_s = max(1.0e-18, 8.0 * np.finfo(float).eps * duration_s)
    while time_s < duration_s - time_tolerance_s:
        dt_s = min(macro_dt_s, duration_s - time_s)
        acceleration = -(omega_rad_per_s**2) * x
        x += velocity * dt_s + 0.5 * acceleration * dt_s * dt_s
        velocity += acceleration * dt_s
        time_s += dt_s
        step_count += 1

    exact_x = math.cos(omega_rad_per_s * duration_s)
    exact_velocity = -omega_rad_per_s * math.sin(omega_rad_per_s * duration_s)
    normalized_velocity_error = (velocity - exact_velocity) / omega_rad_per_s
    phase_space_error = math.hypot(x - exact_x, normalized_velocity_error)
    return {
        "step_count": step_count,
        "final_x": x,
        "final_velocity_over_omega": velocity / omega_rad_per_s,
        "phase_space_error": phase_space_error,
    }


def run_macro_step_convergence(
    *,
    number_density_m3: float = DEFAULT_NUMBER_DENSITY_M3,
    ion_mass_amu: float = DEFAULT_ION_MASS_AMU,
    charge_state: int = 1,
    macro_dt_values_s: Sequence[float] = DEFAULT_MACRO_DT_S,
    reference_macro_dt_s: float = DEFAULT_REFERENCE_MACRO_DT_S,
    max_macro_phase_rad: float = DEFAULT_MAX_MACRO_PHASE_RAD,
    plasma_periods: float = 1.0,
) -> dict[str, Any]:
    """Measure frozen-self-field coupling error across macro-step refinement."""

    ion_mass_amu = float(ion_mass_amu)
    if not math.isfinite(ion_mass_amu) or ion_mass_amu <= 0.0:
        raise ValueError("ion_mass_amu must be finite and positive.")
    if int(charge_state) == 0:
        raise ValueError("charge_state must be non-zero.")
    if not math.isfinite(plasma_periods) or plasma_periods <= 0.0:
        raise ValueError("plasma_periods must be finite and positive.")
    if not math.isfinite(max_macro_phase_rad) or max_macro_phase_rad <= 0.0:
        raise ValueError("max_macro_phase_rad must be finite and positive.")

    omega_rad_per_s = plasma_frequency_rad_per_s(
        number_density_m3,
        charge_c=float(charge_state) * elementary_charge,
        mass_kg=ion_mass_amu * ATOMIC_MASS_CONSTANT_KG,
    )
    plasma_period_s = 2.0 * math.pi / omega_rad_per_s
    duration_s = float(plasma_periods) * plasma_period_s
    dt_values = sorted((float(value) for value in macro_dt_values_s), reverse=True)
    if len(dt_values) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in dt_values):
        raise ValueError("At least two finite, positive macro_dt_values_s are required.")

    rows: list[dict[str, Any]] = []
    previous_error: float | None = None
    previous_dt_s: float | None = None
    for macro_dt_s in dt_values:
        integration = _frozen_field_oscillator_error(
            omega_rad_per_s=omega_rad_per_s,
            macro_dt_s=macro_dt_s,
            duration_s=duration_s,
        )
        error = float(integration["phase_space_error"])
        error_ratio = None
        observed_order = None
        if previous_error is not None and previous_dt_s is not None:
            error_ratio = previous_error / error
            observed_order = math.log(error_ratio) / math.log(previous_dt_s / macro_dt_s)
        macro_phase_rad = omega_rad_per_s * macro_dt_s
        rows.append(
            {
                "macro_dt_s": macro_dt_s,
                "macro_phase_rad": macro_phase_rad,
                "updates_per_plasma_period": 2.0 * math.pi / macro_phase_rad,
                **integration,
                "error_ratio_from_previous": error_ratio,
                "observed_order": observed_order,
                "phase_resolution_passed": macro_phase_rad <= max_macro_phase_rad,
            }
        )
        previous_error = error
        previous_dt_s = macro_dt_s

    refinement_rows = rows[1:]
    convergence_passed = bool(
        all(float(row["error_ratio_from_previous"]) > 1.0 for row in refinement_rows)
        and all(float(row["observed_order"]) >= 0.8 for row in refinement_rows)
    )
    reference_phase_rad = omega_rad_per_s * float(reference_macro_dt_s)
    return {
        "method": "linear_plasma_oscillator_with_self_field_frozen_per_macro_step",
        "scope": "coupling-error proxy; application trajectories still require a mesh/time-step study",
        "number_density_m3": float(number_density_m3),
        "ion_mass_amu": ion_mass_amu,
        "charge_state": int(charge_state),
        "omega_p_rad_per_s": omega_rad_per_s,
        "plasma_period_s": plasma_period_s,
        "duration_s": duration_s,
        "reference_macro_dt_s": float(reference_macro_dt_s),
        "reference_macro_phase_rad": reference_phase_rad,
        "max_macro_phase_rad": float(max_macro_phase_rad),
        "reference_phase_resolution_passed": reference_phase_rad <= max_macro_phase_rad,
        "acceptance": {
            "error_decreases_each_refinement": True,
            "minimum_observed_order": 0.8,
            "reference_omega_p_times_macro_dt_max": float(max_macro_phase_rad),
        },
        "rows": rows,
        "convergence_passed": convergence_passed,
        "passed": convergence_passed and reference_phase_rad <= max_macro_phase_rad,
    }


def run_pic_convergence(**kwargs: Any) -> dict[str, Any]:
    """Return the combined grid and macro-coupling convergence result."""

    grid_result = run_manufactured_poisson_convergence()
    macro_result = run_macro_step_convergence(**kwargs)
    return {
        "schema_version": 1,
        "grid_convergence": grid_result,
        "macro_step_convergence": macro_result,
        "passed": bool(grid_result["passed"] and macro_result["passed"]),
    }


def _write_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pic_convergence.json"
    csv_path = output_dir / "pic_convergence.csv"
    report_path = output_dir / "pic_convergence_report.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    for row in result["grid_convergence"]["rows"]:
        csv_rows.append({"study": "grid", **row})
    for row in result["macro_step_convergence"]["rows"]:
        csv_rows.append({"study": "macro_step", **row})
    fieldnames = sorted({key for row in csv_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    grid_rows = result["grid_convergence"]["rows"]
    macro = result["macro_step_convergence"]
    macro_rows = macro["rows"]
    lines = [
        "# PIC Convergence Report",
        "",
        f"Overall: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "## Grid convergence",
        "",
        "| nr x nz | relative L2(phi) | ratio | observed order | residual |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in grid_rows:
        ratio = "—" if row["error_ratio_from_previous"] is None else f"{row['error_ratio_from_previous']:.4g}"
        order = "—" if row["observed_order"] is None else f"{row['observed_order']:.4f}"
        lines.append(
            f"| {row['nr']} x {row['nz']} | {row['relative_l2_phi_error']:.6e} "
            f"| {ratio} | {order} | {row['poisson_relative_residual']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Gate: every Poisson solve converges, error decreases on each refinement, "
            "and observed order is at least 1.8.",
            "",
            "## Macro-step / plasma-frequency convergence",
            "",
            f"`omega_p = {macro['omega_p_rad_per_s']:.6e} rad/s`, "
            f"`T_p = {macro['plasma_period_s']:.6e} s`.",
            "",
            "| macro dt [s] | omega_p dt [rad] | phase-space error | ratio | observed order | phase gate |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in macro_rows:
        ratio = "—" if row["error_ratio_from_previous"] is None else f"{row['error_ratio_from_previous']:.4g}"
        order = "—" if row["observed_order"] is None else f"{row['observed_order']:.4f}"
        lines.append(
            f"| {row['macro_dt_s']:.3e} | {row['macro_phase_rad']:.5f} "
            f"| {row['phase_space_error']:.6e} | {ratio} | {order} "
            f"| {'PASS' if row['phase_resolution_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Gate: halving the macro step must reduce the frozen-field oscillator error "
            "with observed order at least 0.8. The operational reference also requires "
            f"`omega_p * DeltaT <= {macro['max_macro_phase_rad']}`. This phase limit is a "
            "conservative resolution policy, not a universal stability theorem.",
            "",
            "## Formulas",
            "",
            "- `omega_p = sqrt(n q^2 / (epsilon_0 m))`.",
            "- During one frozen-field macro step: "
            "`x_(n+1)=x_n+v_n*DeltaT-0.5*omega_p^2*x_n*DeltaT^2`, "
            "`v_(n+1)=v_n-omega_p^2*x_n*DeltaT`.",
            "- Manufactured Poisson field: "
            "`phi=A[1-(r/R)^2]sin(pi z/L)`, with `rho=-epsilon_0 laplacian(phi)`.",
            "",
            "The oscillator isolates macro-coupling lag. Quantitative production runs "
            "must additionally repeat their own observables on finer PIC grids and "
            "smaller macro steps.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path, report_path


def _parse_grid_shapes(values: Iterable[str]) -> tuple[tuple[int, int], ...]:
    shapes = []
    for value in values:
        parts = value.lower().split("x", maxsplit=1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid grid shape {value!r}; use NRxNZ.")
        shapes.append((int(parts[0]), int(parts[1])))
    return tuple(shapes)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase2_pic_convergence_20260720")
    parser.add_argument("--number-density-m3", type=float, default=DEFAULT_NUMBER_DENSITY_M3)
    parser.add_argument("--ion-mass-amu", type=float, default=DEFAULT_ION_MASS_AMU)
    parser.add_argument("--charge-state", type=int, default=1)
    parser.add_argument("--macro-dt-s", type=float, nargs="+", default=list(DEFAULT_MACRO_DT_S))
    parser.add_argument("--reference-macro-dt-s", type=float, default=DEFAULT_REFERENCE_MACRO_DT_S)
    parser.add_argument("--max-macro-phase-rad", type=float, default=DEFAULT_MAX_MACRO_PHASE_RAD)
    parser.add_argument(
        "--grid-shape",
        action="append",
        default=None,
        metavar="NRxNZ",
        help="Nested grid shape; repeat at least twice.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    grid_shapes = DEFAULT_GRID_SHAPES if args.grid_shape is None else _parse_grid_shapes(args.grid_shape)
    grid_result = run_manufactured_poisson_convergence(grid_shapes=grid_shapes)
    macro_result = run_macro_step_convergence(
        number_density_m3=args.number_density_m3,
        ion_mass_amu=args.ion_mass_amu,
        charge_state=args.charge_state,
        macro_dt_values_s=args.macro_dt_s,
        reference_macro_dt_s=args.reference_macro_dt_s,
        max_macro_phase_rad=args.max_macro_phase_rad,
    )
    result = {
        "schema_version": 1,
        "grid_convergence": grid_result,
        "macro_step_convergence": macro_result,
        "passed": bool(grid_result["passed"] and macro_result["passed"]),
    }
    json_path, csv_path, report_path = _write_outputs(result, Path(args.output_dir))
    print(f"PIC convergence: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    return result


if __name__ == "__main__":
    main()
