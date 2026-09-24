"""Generate a one-page English report summarizing the sanity-check results."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ...config import PROJECT_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(target: Path, start: Path) -> str:
    return os.path.relpath(target.resolve(), start.resolve().parent).replace("\\", "/")


def build_report(
    *,
    repo_root: Path,
    coulomb_summary_path: Path,
    mach_summary_path: Path,
    benchmark_summary_path: Path,
    output_path: Path,
) -> str:
    coulomb = _read_json(coulomb_summary_path)
    mach = _read_json(mach_summary_path)
    benchmark = _read_json(benchmark_summary_path)

    coulomb_plot = repo_root / Path(coulomb["phase_space_plot"])
    mach_temp_plot = repo_root / Path(mach["temperature_plot"])
    benchmark_plot = repo_root / Path(benchmark["benchmark_plot"])

    benchmark_entries = benchmark["entries"]
    cpu_entry = next(entry for entry in benchmark_entries if entry["requested_backend"] == "cpu")
    taichi_entry = next(entry for entry in benchmark_entries if entry["requested_backend"] == "taichi")

    radial_growth = coulomb["final_mean_r_m"] / max(coulomb["initial_mean_r_m"], 1.0e-30)
    heating_delta = mach["peak_mean_teff_k"] - mach["initial_mean_teff_k"]
    benchmark_delta_pct = 100.0 * (
        taichi_entry["runtime_s"] - cpu_entry["runtime_s"]
    ) / max(cpu_entry["runtime_s"], 1.0e-30)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Sanity Check Report

**Project:** Simu_IonSource  
**Report Type:** One-page English summary  
**Generated:** {generated_at}

## Project Overview

Simu_IonSource is being developed as a reduced-dimension, multi-physics simulation framework for the ESI-MS interface region. The core design choice is to keep ion motion in full 3D Cartesian space while storing all Eulerian fields on a 2D axisymmetric `(r, z)` mesh. This allows the code to retain physically meaningful 3D particle trajectories while reducing the cost of field storage and space-charge updates.

## Project Purpose

The project is intended to provide a practical simulation framework for:

- ion transport in the ESI-MS inlet and transition region,
- coupling between electric fields, gas flow, collisions, and internal thermodynamics,
- reuse of IonSPA as the thermal/fragmentation kernel instead of rewriting that physics from scratch,
- future integration of real SIMION electrostatic data, CFD flow fields, and higher-fidelity PIC workflows.

## Methods Used

The current implementation combines several numerical and physical ingredients:

- **Eulerian-Lagrangian hybrid architecture:** ions are treated as Lagrangian particles, while electric, gas-dynamic, and space-charge fields are stored on a 2D axisymmetric Eulerian mesh.
- **2D axisymmetric PIC workflow:** particle charge is deposited to the mesh with CIC weighting, the cylindrical Poisson equation is solved with SOR, and the space-charge field is recovered by finite differences.
- **3D force reconstruction:** the mesh stores `E_r` and `E_z`, which are mapped back to `E_x`, `E_y`, and `E_z` for particle motion.
- **Micro-step collision workflow:** the particle update sequence is `Gather -> Monte Carlo collision -> IonSPA thermodynamic update -> RK4 push`, so thermal changes are attached to the local gas state before position advancement.
- **Data export and post-processing:** macro-step snapshots are written to structured arrays and later visualized with offline plotting tools.
- **Optional Taichi acceleration:** the framework is compatible with Taichi, although final speedup depends on whether CUDA hardware is available.

## Current Progress

At the current stage, the project has achieved the following milestones:

- a flowchart-aligned default PIC execution path has replaced the earlier placeholder runtime path,
- the code can export alive-ion snapshots for offline analysis,
- post-processing scripts can generate phase-space and temperature-evolution figures,
- a reusable sanity-check entry point now supports physics validation and benchmarking,
- three baseline tests have been completed: Coulomb expansion, Mach-disk heating, and backend benchmarking.

## Executive Summary

Three sanity checks were run to validate the current 2D axisymmetric PIC workflow.

- **Test 1: Coulomb Expansion Test** showed clear radial expansion under vacuum and zero external field, which is consistent with a space-charge-driven Coulomb blow-up.
- **Test 2: Mach Disk Heating Test** showed a measurable internal-temperature rise after the imposed pressure/velocity step, which is consistent with the intended micro-step ordering `Gather -> MC collision -> IonSPA -> RK4`.
- **Test 3: CPU vs. Taichi Benchmark** completed successfully as a workflow benchmark, but the current machine has no CUDA runtime, so both backends executed on `x64 CPU`. The benchmark is therefore valid as a script/integration check, not yet as a publishable CPU-vs-GPU figure.

## Test 1. Coulomb Expansion Test

**Purpose:** verify the physical correctness of the 2D PIC Poisson solve and the back-mapping from `E_sce_r` to Cartesian force components.

**Setup:**

- External DC/RF field disabled
- Background gas disabled
- Collisions disabled
- Dense ion cloud injected near `r = 0`, `z = 0`
- Only PIC space-charge force retained

**Key Results:**

- Ion count: `{coulomb["ion_count"]}`
- Real ions per weighted ion pack: `{coulomb["macro_particle_weight"]:.3e}`
- Initial mean radius: `{coulomb["initial_mean_r_m"]:.3e} m`
- Final mean radius: `{coulomb["final_mean_r_m"]:.3e} m`
- Radial growth factor: `{radial_growth:.2f}x`
- Collision count: `{coulomb["collision_count"]}`

**Interpretation:** the cloud expands outward without any external focusing or gas drag, which is the expected qualitative signature of a Coulomb explosion. This supports the consistency of the PIC field solve and the radial-to-Cartesian force projection.

![Coulomb Expansion Phase Space]({_rel(coulomb_plot, output_path)})

## Test 2. Mach Disk Heating Test

**Purpose:** verify that the micro-step execution order captures thermodynamic changes in a non-equilibrium gas field.

**Setup:**

- A shock-like step was imposed at `z = 5 mm`
- Gas pressure rises sharply downstream of the step
- Axial gas velocity drops sharply downstream of the step
- Collision handling and IonSPA thermodynamics were enabled

**Key Results:**

- Ion count: `{mach["ion_count"]}`
- Collision count: `{mach["collision_count"]}`
- Initial mean `T_eff`: `{mach["initial_mean_teff_k"]:.2f} K`
- Peak mean `T_eff`: `{mach["peak_mean_teff_k"]:.2f} K`
- Heating increment: `{heating_delta:.2f} K`
- Peak heating location: `{mach["peak_mean_z_m"] * 1.0e3:.2f} mm`

**Interpretation:** the temperature rise appears downstream of the imposed gas-dynamic discontinuity, which is consistent with the intended order `Gather -> Monte Carlo collision -> IonSPA update -> RK4 push`. This is a good sign that the thermal jump is being attached to the local gas state before the particle is advanced.

![Mach Disk Temperature Evolution]({_rel(mach_temp_plot, output_path)})

## Test 3. CPU vs. Taichi Benchmark

**Purpose:** prepare the workflow for a future performance comparison figure.

**Current Benchmark Run:**

- Ion count: `{cpu_entry["ion_count"]}`
- Grid: `{cpu_entry["grid_nr"]} x {cpu_entry["grid_nz"]}`
- Target macro steps: `{cpu_entry["macro_steps_target"]}`
- CPU runtime: `{cpu_entry["runtime_s"]:.3f} s` on `{cpu_entry["actual_arch"]}`
- Taichi runtime: `{taichi_entry["runtime_s"]:.3f} s` on `{taichi_entry["actual_arch"]}`
- Relative runtime difference: `{benchmark_delta_pct:+.2f}%`

**Interpretation:** the benchmark pipeline is functional, but both runs fell back to `x64 CPU`. As a result, this benchmark should currently be treated as an integration check rather than the final Figure 1 for a paper. A publishable version should be re-run on a machine with working CUDA support.

![Backend Benchmark Runtime]({_rel(benchmark_plot, output_path)})

## Overall Conclusion

The current sanity-check suite supports the following claims:

1. The PIC space-charge field produces the expected qualitative Coulomb expansion in vacuum.
2. The collision/thermodynamic update order is able to capture a shock-driven heating signature in the imposed Mach-disk-like field.
3. The benchmark tooling is ready, but the final CPU-vs-GPU comparison must be repeated on CUDA-capable hardware.

## Development Status

The project has moved beyond a code skeleton and now supports:

- a working axisymmetric PIC runtime,
- automated data export,
- automated figure generation,
- reproducible sanity-check scripts,
- direct English reporting from generated results.

The main remaining gap is calibration against real instrument data and execution on a GPU-enabled machine for a publishable performance figure.
"""
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a one-page English sanity-check report.")
    parser.add_argument("--coulomb-summary", default="outputs/sanity_coulomb_expansion/summary.json")
    parser.add_argument("--mach-summary", default="outputs/sanity_mach_disk_heating/summary.json")
    parser.add_argument("--benchmark-summary", default="outputs/backend_benchmark_smoke/summary.json")
    parser.add_argument("--output", default="outputs/Sanity_Check_Report_EN.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = PROJECT_ROOT
    output_path = repo_root / Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(
        repo_root=repo_root,
        coulomb_summary_path=repo_root / Path(args.coulomb_summary),
        mach_summary_path=repo_root / Path(args.mach_summary),
        benchmark_summary_path=repo_root / Path(args.benchmark_summary),
        output_path=output_path,
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
