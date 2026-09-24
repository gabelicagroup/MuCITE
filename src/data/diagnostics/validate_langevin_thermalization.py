"""Validate the high-collision Langevin thermalization approximation.

This diagnostic is intentionally separate from the production trajectory
runtime. It checks whether a single aggregated Ornstein-Uhlenbeck velocity
update reproduces the analytic mean and variance expected after many neutral
momentum-transfer collisions in a uniform gas bath.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import Boltzmann

from ...config import ATOMIC_MASS_CONSTANT


def _langevin_update(
    velocities_m_per_s: np.ndarray,
    *,
    gas_velocity_m_per_s: np.ndarray,
    gas_temperature_k: float,
    ion_mass_kg: float,
    dt_s: float,
    momentum_relaxation_time_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one exact OU/Langevin velocity update for a uniform gas bath."""

    velocities = np.asarray(velocities_m_per_s, dtype=np.float64)
    gas_velocity = np.asarray(gas_velocity_m_per_s, dtype=np.float64)
    tau_s = max(float(momentum_relaxation_time_s), 1.0e-30)
    alpha = float(np.exp(-max(float(dt_s), 0.0) / tau_s))
    thermal_std_m_per_s = float(np.sqrt(Boltzmann * max(float(gas_temperature_k), 1.0) / max(float(ion_mass_kg), 1.0e-30)))
    diffusion_scale = thermal_std_m_per_s * float(np.sqrt(max(1.0 - alpha * alpha, 0.0)))
    return gas_velocity + alpha * (velocities - gas_velocity) + diffusion_scale * rng.normal(
        0.0,
        1.0,
        size=velocities.shape,
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    n_particles = max(1, int(args.particles))
    ion_mass_kg = max(float(args.ion_mass_amu), 1.0e-30) * ATOMIC_MASS_CONSTANT
    gas_velocity = np.array([0.0, 0.0, float(args.gas_velocity_z_m_per_s)], dtype=np.float64)
    initial_relative = np.array([0.0, 0.0, float(args.initial_relative_vz_m_per_s)], dtype=np.float64)
    velocities0 = np.tile(gas_velocity + initial_relative, (n_particles, 1))

    total_time_s = max(float(args.total_time_s), 0.0)
    tau_s = max(float(args.tau_s), 1.0e-30)
    gas_temperature_k = max(float(args.gas_temperature_k), 1.0)
    thermal_std_m_per_s = float(np.sqrt(Boltzmann * gas_temperature_k / ion_mass_kg))
    alpha_total = float(np.exp(-total_time_s / tau_s))

    aggregated = _langevin_update(
        velocities0,
        gas_velocity_m_per_s=gas_velocity,
        gas_temperature_k=gas_temperature_k,
        ion_mass_kg=ion_mass_kg,
        dt_s=total_time_s,
        momentum_relaxation_time_s=tau_s,
        rng=rng,
    )

    rng = np.random.default_rng(int(args.seed))
    substep_count = max(1, int(args.substeps))
    substep = np.array(velocities0, copy=True)
    dt_sub_s = total_time_s / float(substep_count)
    for _ in range(substep_count):
        substep = _langevin_update(
            substep,
            gas_velocity_m_per_s=gas_velocity,
            gas_temperature_k=gas_temperature_k,
            ion_mass_kg=ion_mass_kg,
            dt_s=dt_sub_s,
            momentum_relaxation_time_s=tau_s,
            rng=rng,
        )

    expected_relative_mean = alpha_total * initial_relative
    expected_relative_std = thermal_std_m_per_s * float(np.sqrt(max(1.0 - alpha_total * alpha_total, 0.0)))
    aggregated_relative = aggregated - gas_velocity
    substep_relative = substep - gas_velocity

    mean_error = np.mean(aggregated_relative, axis=0) - expected_relative_mean
    std_error = np.std(aggregated_relative, axis=0) - expected_relative_std
    substep_mean_error = np.mean(substep_relative, axis=0) - expected_relative_mean
    substep_std_error = np.std(substep_relative, axis=0) - expected_relative_std

    result = {
        "particles": n_particles,
        "seed": int(args.seed),
        "ion_mass_amu": float(args.ion_mass_amu),
        "gas_temperature_k": gas_temperature_k,
        "gas_velocity_z_m_per_s": float(args.gas_velocity_z_m_per_s),
        "tau_s": tau_s,
        "total_time_s": total_time_s,
        "substeps": substep_count,
        "thermal_std_m_per_s": thermal_std_m_per_s,
        "alpha_total": alpha_total,
        "expected_relative_mean_m_per_s": expected_relative_mean.tolist(),
        "expected_relative_std_m_per_s": expected_relative_std,
        "aggregated_relative_mean_m_per_s": np.mean(aggregated_relative, axis=0).tolist(),
        "aggregated_relative_std_m_per_s": np.std(aggregated_relative, axis=0).tolist(),
        "aggregated_mean_error_m_per_s": mean_error.tolist(),
        "aggregated_std_error_m_per_s": std_error.tolist(),
        "substep_relative_mean_m_per_s": np.mean(substep_relative, axis=0).tolist(),
        "substep_relative_std_m_per_s": np.std(substep_relative, axis=0).tolist(),
        "substep_mean_error_m_per_s": substep_mean_error.tolist(),
        "substep_std_error_m_per_s": substep_std_error.tolist(),
        "aggregated_speed_m_per_s": _summary(np.linalg.norm(aggregated, axis=1)),
        "substep_speed_m_per_s": _summary(np.linalg.norm(substep, axis=1)),
    }
    result["pass"] = bool(
        np.max(np.abs(mean_error)) < float(args.mean_tolerance_m_per_s)
        and np.max(np.abs(std_error)) < float(args.std_tolerance_m_per_s)
        and np.max(np.abs(substep_mean_error)) < float(args.mean_tolerance_m_per_s)
        and np.max(np.abs(substep_std_error)) < float(args.std_tolerance_m_per_s)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an aggregated Langevin thermalization update.")
    parser.add_argument("--particles", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ion-mass-amu", type=float, default=2000.0)
    parser.add_argument("--gas-temperature-k", type=float, default=208.65)
    parser.add_argument("--gas-velocity-z-m-per-s", type=float, default=539.178)
    parser.add_argument("--initial-relative-vz-m-per-s", type=float, default=1500.0)
    parser.add_argument("--tau-s", type=float, default=3.0e-11)
    parser.add_argument("--total-time-s", type=float, default=3.0e-9)
    parser.add_argument("--substeps", type=int, default=200)
    parser.add_argument("--mean-tolerance-m-per-s", type=float, default=1.0)
    parser.add_argument("--std-tolerance-m-per-s", type=float, default=1.0)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_validation(args)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / (
        "langevin_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "langevin_validation_summary.json"
    report_path = output_dir / "langevin_validation_report.md"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Langevin Thermalization Validation",
                "",
                f"- pass: `{result['pass']}`",
                f"- particles: `{result['particles']}`",
                f"- tau: `{result['tau_s']} s`",
                f"- total time: `{result['total_time_s']} s`",
                f"- expected relative mean: `{result['expected_relative_mean_m_per_s']}` m/s",
                f"- aggregated relative mean: `{result['aggregated_relative_mean_m_per_s']}` m/s",
                f"- aggregated mean error: `{result['aggregated_mean_error_m_per_s']}` m/s",
                f"- expected relative std: `{result['expected_relative_std_m_per_s']}` m/s",
                f"- aggregated relative std: `{result['aggregated_relative_std_m_per_s']}` m/s",
                f"- aggregated std error: `{result['aggregated_std_error_m_per_s']}` m/s",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Pass: {result['pass']}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
