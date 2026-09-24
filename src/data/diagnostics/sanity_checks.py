"""Physics sanity checks for the axisymmetric PIC workflow."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time as wall_time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

from ...config import IonTemplate, SimulationConfig
from ...core.simulation import FlowchartPicSimulation
from ..logger import DataLogger
from .plot_results import load_snapshots, plot_phase_space, plot_temperature_evolution

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _write_summary(summary_path: Path, payload: dict[str, object]) -> None:
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_coulomb_expansion_test(
    *,
    output_dir: Path,
    backend: str = "cpu",
    ion_count: int = 256,
    macro_particle_weight: float = 2.0e4,
    total_time_s: float = 2.0e-7,
    macro_time_step_s: float = 2.0e-9,
) -> dict[str, object]:
    """Run the space-charge-only Coulomb expansion sanity check.

    Test setup
    ----------
    - external DC / RF field disabled,
    - background gas pressure forced to zero,
    - collisions therefore vanish automatically,
    - a compact, high-density ion cloud is injected near ``r = 0`` and ``z = 0``.
    """

    output_dir = Path(output_dir)
    export_dir = output_dir / "exports"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        ion_count=int(ion_count),
        total_time_s=float(total_time_s),
        macro_time_step_s=float(macro_time_step_s),
        random_seed=11,
        domain_radius_m=8.0e-3,
        domain_length_m=25.0e-3,
        rf_frequency_hz=0.0,
        initial_position_jitter_m=2.0e-5,
        initial_velocity_jitter_m_per_s=1.0,
        grid_nr=128,
        grid_nz=256,
        macro_particle_weight=float(macro_particle_weight),
        sor_omega=1.75,
        sor_max_iters=300,
        sor_tolerance=1.0e-6,
    )
    template = IonTemplate(
        name="coulomb_cloud",
        charge_state=10,
        initial_position_m=np.array([0.0, 0.0, 0.0], dtype=float),
        initial_velocity_m_per_s=np.array([0.0, 0.0, 50.0], dtype=float),
        initial_internal_temperature_k=300.0,
    )
    logger = DataLogger(export_dir, export_every_macro_steps=1, file_format="npy")
    simulation = FlowchartPicSimulation(template, config, particle_backend=backend, data_logger=logger)

    # Remove all externally imposed fields and neutral-gas effects so any observed
    # expansion comes solely from the solved PIC space-charge field.
    simulation.replace_static_fields(
        lambda grid: grid.bake_zero_static_fields(
            background_pressure_pa=0.0,
            background_temperature_k=300.0,
        )
    )
    simulation.backend_message = (
        "Running Coulomb expansion sanity check: external fields off, vacuum background, "
        "collisions disabled, PIC space-charge only."
    )

    result = simulation.run()
    snapshots, _ = load_snapshots(export_dir)
    phase_space_path = plot_dir / "coulomb_expansion_phase_space.png"
    plot_phase_space(snapshots, phase_space_path)

    first_nonempty = next((snapshot for snapshot in snapshots if snapshot.data.size > 0), None)
    last_nonempty = next((snapshot for snapshot in reversed(snapshots) if snapshot.data.size > 0), None)
    if first_nonempty is None or last_nonempty is None:
        raise RuntimeError("Coulomb expansion test produced no non-empty snapshots.")

    initial_r = first_nonempty.data[:, 0]
    final_r = last_nonempty.data[:, 0]
    summary = {
        "test_name": "coulomb_expansion",
        "backend": backend,
        "export_directory": str(export_dir),
        "phase_space_plot": str(phase_space_path),
        "ion_count": int(ion_count),
        "macro_particle_weight": float(macro_particle_weight),
        "survivors": len(result.survivors),
        "collision_count": int(result.collision_count),
        "macro_steps_logged": len(snapshots),
        "initial_mean_r_m": float(np.mean(initial_r)),
        "final_mean_r_m": float(np.mean(final_r)),
        "initial_max_r_m": float(np.max(initial_r)),
        "final_max_r_m": float(np.max(final_r)),
        "final_mean_z_m": float(np.mean(last_nonempty.data[:, 1])),
    }
    _write_summary(output_dir / "summary.json", summary)
    return summary


def run_mach_disk_heating_test(
    *,
    output_dir: Path,
    backend: str = "cpu",
    ion_count: int = 128,
    total_time_s: float = 1.0e-5,
    macro_time_step_s: float = 1.0e-7,
) -> dict[str, object]:
    """Run the Mach-disk heating sanity check."""

    output_dir = Path(output_dir)
    export_dir = output_dir / "exports"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        ion_count=int(ion_count),
        total_time_s=float(total_time_s),
        macro_time_step_s=float(macro_time_step_s),
        random_seed=19,
        domain_radius_m=8.0e-3,
        domain_length_m=25.0e-3,
        rf_frequency_hz=0.0,
        initial_position_jitter_m=3.0e-5,
        initial_velocity_jitter_m_per_s=3.0,
        grid_nr=128,
        grid_nz=320,
        macro_particle_weight=1.0,
        sor_omega=1.75,
        sor_max_iters=250,
        sor_tolerance=1.0e-6,
    )
    template = IonTemplate(
        name="mach_disk_probe",
        charge_state=10,
        initial_position_m=np.array([0.0, 0.0, 0.0], dtype=float),
        initial_velocity_m_per_s=np.array([0.0, 0.0, 620.0], dtype=float),
        initial_internal_temperature_k=300.0,
    )
    logger = DataLogger(export_dir, export_every_macro_steps=1, file_format="npy")
    simulation = FlowchartPicSimulation(template, config, particle_backend=backend, data_logger=logger)

    # The gas field contains a sharp pressure rise and velocity drop near z = 5 mm,
    # which acts as a Mach-disk-like thermalization region.
    simulation.replace_static_fields(
        lambda grid: grid.bake_mach_disk_step_fields(
            shock_z_m=5.0e-3,
            transition_width_m=2.0e-4,
            upstream_pressure_pa=2.0,
            downstream_pressure_pa=80.0,
            upstream_temperature_k=300.0,
            downstream_temperature_k=520.0,
            upstream_velocity_z_m_per_s=650.0,
            downstream_velocity_z_m_per_s=80.0,
        )
    )
    simulation.backend_message = (
        "Running Mach-disk heating sanity check: pressure step at z=5 mm, velocity drop, "
        "collisions and IonSPA enabled."
    )

    result = simulation.run()
    snapshots, _ = load_snapshots(export_dir)
    temperature_plot_path = plot_dir / "mach_disk_temperature_evolution.png"
    phase_space_path = plot_dir / "mach_disk_phase_space.png"
    plot_temperature_evolution(snapshots, temperature_plot_path)
    plot_phase_space(snapshots, phase_space_path)

    nonempty = [snapshot for snapshot in snapshots if snapshot.data.size > 0]
    if not nonempty:
        raise RuntimeError("Mach-disk heating test produced no non-empty snapshots.")

    mean_temperatures = [float(np.mean(snapshot.data[:, 2])) for snapshot in nonempty]
    mean_positions = [float(np.mean(snapshot.data[:, 1])) for snapshot in nonempty]
    peak_index = int(np.argmax(mean_temperatures))
    summary = {
        "test_name": "mach_disk_heating",
        "backend": backend,
        "export_directory": str(export_dir),
        "temperature_plot": str(temperature_plot_path),
        "phase_space_plot": str(phase_space_path),
        "ion_count": int(ion_count),
        "survivors": len(result.survivors),
        "collision_count": int(result.collision_count),
        "macro_steps_logged": len(snapshots),
        "initial_mean_teff_k": mean_temperatures[0],
        "peak_mean_teff_k": mean_temperatures[peak_index],
        "peak_mean_z_m": mean_positions[peak_index],
        "final_mean_teff_k": mean_temperatures[-1],
        "final_mean_z_m": mean_positions[-1],
    }
    _write_summary(output_dir / "summary.json", summary)
    return summary


def _plot_benchmark_runtime(entries: list[dict[str, object]], output_path: Path) -> None:
    labels = [f"{entry['requested_backend']}\n({entry['actual_arch']})" for entry in entries]
    runtimes = [float(entry["runtime_s"]) for entry in entries]

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=160)
    bars = ax.bar(labels, runtimes, color=["#4c78a8", "#f58518"][: len(entries)])
    ax.set_ylabel("Wall Time [s]")
    ax.set_title("CPU vs Taichi Benchmark")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, runtime in zip(bars, runtimes):
        ax.text(bar.get_x() + bar.get_width() / 2.0, runtime, f"{runtime:.2f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _run_backend_benchmark_entry(
    requested_backend: str,
    ion_count: int = 100000,
    grid_nr: int = 200,
    grid_nz: int = 500,
    macro_steps: int = 1000,
    macro_time_step_s: float = 1.0e-9,
) -> dict[str, object]:
    """Run one backend benchmark inside its own process."""

    config = SimulationConfig(
        ion_count=int(ion_count),
        total_time_s=float(macro_steps) * float(macro_time_step_s),
        macro_time_step_s=float(macro_time_step_s),
        random_seed=23,
        domain_radius_m=8.0e-3,
        domain_length_m=25.0e-3,
        rf_frequency_hz=0.0,
        initial_position_jitter_m=1.0e-4,
        initial_velocity_jitter_m_per_s=0.0,
        grid_nr=int(grid_nr),
        grid_nz=int(grid_nz),
        macro_particle_weight=1.0,
        sor_omega=1.75,
        sor_max_iters=120,
        sor_tolerance=1.0e-6,
    )
    template = IonTemplate(
        name=f"benchmark_{requested_backend}",
        initial_position_m=np.array([0.0, 0.0, 0.0], dtype=float),
        initial_velocity_m_per_s=np.array([0.0, 0.0, 200.0], dtype=float),
    )
    simulation = FlowchartPicSimulation(template, config, particle_backend=requested_backend, data_logger=None)
    simulation.replace_static_fields(
        lambda grid: grid.bake_zero_static_fields(
            background_pressure_pa=0.0,
            background_temperature_k=300.0,
        )
    )
    start = wall_time.perf_counter()
    result = simulation.run(progress_callback=None)
    runtime_s = wall_time.perf_counter() - start
    return {
        "requested_backend": requested_backend,
        "actual_arch": simulation.arch_label,
        "runtime_s": runtime_s,
        "survivors": len(result.survivors),
        "macro_steps_target": int(macro_steps),
        "ion_count": int(ion_count),
        "grid_nr": int(grid_nr),
        "grid_nz": int(grid_nz),
    }


def run_backend_benchmark(
    *,
    output_dir: Path,
    ion_count: int = 100000,
    grid_nr: int = 200,
    grid_nz: int = 500,
    macro_steps: int = 1000,
    macro_time_step_s: float = 1.0e-9,
) -> dict[str, object]:
    """Benchmark CPU and accelerator requests in separate spawned processes."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    process_context = multiprocessing.get_context("spawn")
    for requested_backend in ("cpu", "taichi"):
        # Recreate the one-worker executor for every request so architecture
        # selection can never leak from one benchmark entry to the next.
        with ProcessPoolExecutor(max_workers=1, mp_context=process_context) as executor:
            entries.append(
                executor.submit(
                    _run_backend_benchmark_entry,
                    requested_backend,
                    int(ion_count),
                    int(grid_nr),
                    int(grid_nz),
                    int(macro_steps),
                    float(macro_time_step_s),
                ).result()
            )

    benchmark_plot_path = output_dir / "benchmark_runtime.png"
    _plot_benchmark_runtime(entries, benchmark_plot_path)
    summary = {
        "test_name": "backend_benchmark",
        "entries": entries,
        "benchmark_plot": str(benchmark_plot_path),
    }
    _write_summary(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physics sanity checks for Simu_IonSource.")
    subparsers = parser.add_subparsers(dest="test_name", required=True)

    coulomb = subparsers.add_parser("coulomb-expansion", help="Run the space-charge Coulomb explosion sanity check.")
    coulomb.add_argument("--backend", choices=["cpu", "taichi"], default="cpu")
    coulomb.add_argument("--output-dir", default="outputs/sanity_coulomb_expansion")
    coulomb.add_argument("--ion-count", type=int, default=256)
    coulomb.add_argument("--macro-weight", type=float, default=2.0e4)
    coulomb.add_argument("--total-time", type=float, default=2.0e-7)
    coulomb.add_argument("--macro-time-step", type=float, default=2.0e-9)

    mach = subparsers.add_parser("mach-disk-heating", help="Run the shock-heating / Mach-disk thermodynamic sanity check.")
    mach.add_argument("--backend", choices=["cpu", "taichi"], default="cpu")
    mach.add_argument("--output-dir", default="outputs/sanity_mach_disk_heating")
    mach.add_argument("--ion-count", type=int, default=128)
    mach.add_argument("--total-time", type=float, default=1.0e-5)
    mach.add_argument("--macro-time-step", type=float, default=1.0e-7)

    bench = subparsers.add_parser("backend-benchmark", help="Benchmark requested CPU and Taichi backends.")
    bench.add_argument("--output-dir", default="outputs/backend_benchmark")
    bench.add_argument("--ion-count", type=int, default=100000)
    bench.add_argument("--grid-nr", type=int, default=200)
    bench.add_argument("--grid-nz", type=int, default=500)
    bench.add_argument("--macro-steps", type=int, default=1000)
    bench.add_argument("--macro-time-step", type=float, default=1.0e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.test_name == "coulomb-expansion":
        summary = run_coulomb_expansion_test(
            output_dir=Path(args.output_dir),
            backend=args.backend,
            ion_count=args.ion_count,
            macro_particle_weight=args.macro_weight,
            total_time_s=args.total_time,
            macro_time_step_s=args.macro_time_step,
        )
        print(f"Summary: {Path(args.output_dir) / 'summary.json'}")
        print(f"Phase Space Plot: {summary['phase_space_plot']}")
        print(
            "Initial mean r = {:.3e} m, final mean r = {:.3e} m".format(
                summary["initial_mean_r_m"],
                summary["final_mean_r_m"],
            )
        )
    elif args.test_name == "mach-disk-heating":
        summary = run_mach_disk_heating_test(
            output_dir=Path(args.output_dir),
            backend=args.backend,
            ion_count=args.ion_count,
            total_time_s=args.total_time,
            macro_time_step_s=args.macro_time_step,
        )
        print(f"Summary: {Path(args.output_dir) / 'summary.json'}")
        print(f"Temperature Plot: {summary['temperature_plot']}")
        print(
            "Initial mean Teff = {:.2f} K, peak mean Teff = {:.2f} K at z = {:.3e} m".format(
                summary["initial_mean_teff_k"],
                summary["peak_mean_teff_k"],
                summary["peak_mean_z_m"],
            )
        )
    elif args.test_name == "backend-benchmark":
        summary = run_backend_benchmark(
            output_dir=Path(args.output_dir),
            ion_count=args.ion_count,
            grid_nr=args.grid_nr,
            grid_nz=args.grid_nz,
            macro_steps=args.macro_steps,
            macro_time_step_s=args.macro_time_step,
        )
        print(f"Summary: {Path(args.output_dir) / 'summary.json'}")
        print(f"Benchmark Plot: {summary['benchmark_plot']}")
        for entry in summary["entries"]:
            print(
                "backend={} arch={} runtime={:.3f} s".format(
                    entry["requested_backend"],
                    entry["actual_arch"],
                    entry["runtime_s"],
                )
            )


if __name__ == "__main__":
    main()
