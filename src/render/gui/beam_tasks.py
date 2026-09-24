"""Beam preview sampling and offline smoke-test tasks."""

from __future__ import annotations

import queue
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import elementary_charge

from ...config import PROJECT_ROOT
from ...env.gas import AxisymmetricGasVelocitySampler
from ...env.sources import sample_source_xy_offsets
from .config_adapter import build_ion_template, direction_vector
from .models import AppConfig, BeamConfig
from .worker_lifecycle import TaskStopped, _post


def _validate_beam_preview(beam: BeamConfig) -> None:
    if int(beam.particle_count) <= 0:
        raise ValueError("particle_count must be positive.")
    if float(beam.beam_radius_mm) < 0.0:
        raise ValueError("beam_radius_mm must be non-negative.")
    if (
        beam_velocity_source(beam) == "static"
        and not (0.0 <= float(beam.cone_half_angle_deg) < 90.0)
    ):
        raise ValueError("cone_half_angle_deg must be in [0, 90).")


def beam_velocity_source(beam: BeamConfig) -> str:
    """Return the velocity source represented by a beam presentation model."""

    mode = str(beam.gas_velocity_init_mode).strip().lower()
    if mode in {"off", "static"}:
        return "static"
    if mode == "from-gas-field":
        return "gas-field-csv"
    raise ValueError(
        "gas_velocity_init_mode must be 'static' or 'from-gas-field'."
    )


def _resolved_gas_velocity_path(beam: BeamConfig) -> Path:
    value = str(beam.source_birth_velocity_gas_csv).strip()
    if not value:
        raise ValueError(
            "Gas-field beam preview needs a Birth gas CSV. The baked-field "
            "fallback remains available to the simulation runtime, but the "
            "standalone beam preview cannot infer it from BeamConfig."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Source birth velocity gas CSV not found: {path}"
        )
    return path


def _gas_velocity_bounds_m(beam: BeamConfig) -> tuple[float, float]:
    z_min_m = max(float(beam.source_birth_velocity_z_min_mm), 0.0) * 1.0e-3
    raw_z_max_mm = beam.source_birth_velocity_z_max_mm
    if raw_z_max_mm is None or str(raw_z_max_mm).strip() == "":
        raw_z_max_mm = getattr(beam, "capillary_exit_z_mm", beam.initial_z_mm)
    z_max_m = max(float(raw_z_max_mm), 0.0) * 1.0e-3
    if z_max_m < z_min_m:
        raise ValueError("Birth gas z max must be greater than or equal to z min.")
    return z_min_m, z_max_m


def _sample_gas_field_velocities(
    beam: BeamConfig,
    positions_m: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    z_min_m, z_max_m = _gas_velocity_bounds_m(beam)
    raw_radius_mm = beam.source_birth_velocity_radius_mm
    radius_m = (
        None
        if raw_radius_mm is None or str(raw_radius_mm).strip() == ""
        else float(raw_radius_mm) * 1.0e-3
    )
    sampler = AxisymmetricGasVelocitySampler.from_fluent_csv(
        _resolved_gas_velocity_path(beam),
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        r_max_m=radius_m,
    )
    velocities = np.asarray(
        sampler.sample_velocity_vectors(positions_m, rng),
        dtype=np.float64,
    )
    radial_scale = max(float(beam.source_radial_velocity_scale), 0.0)
    velocities[:, :2] *= radial_scale
    velocities[:, 2] += float(beam.source_velocity_delta_m_per_s)
    return velocities


def _sample_positions(
    beam: BeamConfig,
    base_position: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    positions = np.tile(base_position, (count, 1))
    radius_m = max(float(beam.beam_radius_mm), 0.0) * 1.0e-3
    jitter_m = max(float(beam.initial_position_jitter_mm), 0.0) * 1.0e-3
    if radius_m > 0.0:
        x_offsets, y_offsets = sample_source_xy_offsets(
            rng,
            count,
            source_radius_m=radius_m,
            source_profile=str(beam.source_profile),
            source_gaussian_sigma_m=(
                max(float(beam.source_gaussian_sigma_mm), 0.0) * 1.0e-3
            ),
        )
        positions[:, 0] = base_position[0] + x_offsets
        positions[:, 1] = base_position[1] + y_offsets
        if jitter_m > 0.0:
            positions[:, 2] = (
                base_position[2] + rng.normal(0.0, jitter_m, size=count)
            )
    elif jitter_m > 0.0:
        positions += rng.normal(0.0, jitter_m, size=(count, 3))
    return positions


def _rotate_cone_directions(
    local_dirs: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    if np.allclose(axis, np.array([0.0, 0.0, 1.0])):
        return local_dirs
    dominant = int(np.argmax(np.abs(axis)))
    sign = float(np.sign(axis[dominant]))
    rotated = np.zeros_like(local_dirs)
    transverse = [index for index in range(3) if index != dominant]
    rotated[:, transverse[0]] = local_dirs[:, 0]
    rotated[:, transverse[1]] = local_dirs[:, 1]
    rotated[:, dominant] = sign * local_dirs[:, 2]
    return rotated


def _sample_cone_directions(
    rng: np.random.Generator,
    count: int,
    cone_half_angle_rad: float,
    axis: np.ndarray,
) -> np.ndarray:
    cos_theta = 1.0 - rng.random(count) * (
        1.0 - np.cos(cone_half_angle_rad)
    )
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    azimuth_rad = 2.0 * np.pi * rng.random(count)
    local_dirs = np.column_stack(
        (
            sin_theta * np.cos(azimuth_rad),
            sin_theta * np.sin(azimuth_rad),
            cos_theta,
        )
    )
    return _rotate_cone_directions(local_dirs, axis)


def _sample_velocities(
    beam: BeamConfig,
    base_velocity: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    speed = float(np.linalg.norm(base_velocity))
    velocities = np.tile(base_velocity, (count, 1))
    cone_half_angle_rad = np.radians(float(beam.cone_half_angle_deg))
    if cone_half_angle_rad > 0.0 and speed > 0.0:
        axis = direction_vector(beam.direction_axis)
        directions = _sample_cone_directions(
            rng,
            count,
            cone_half_angle_rad,
            axis,
        )
        velocities = speed * directions
    velocity_jitter = max(float(beam.velocity_jitter_m_per_s), 0.0)
    if velocity_jitter > 0.0:
        velocities += rng.normal(
            0.0,
            velocity_jitter,
            size=(count, 3),
        )
    return velocities


def _beam_diagnostics(
    beam: BeamConfig,
    mass_kg: float,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    speeds = np.linalg.norm(velocities, axis=1)
    kinetic_energy_ev = 0.5 * mass_kg * speeds**2 / elementary_charge
    axis = direction_vector(beam.direction_axis)
    dot = velocities @ axis
    denom = np.maximum(speeds, 1.0e-30)
    angle_deg = np.degrees(
        np.arccos(np.clip(dot / denom, -1.0, 1.0))
    )
    return kinetic_energy_ev, angle_deg


def _gas_field_diagnostics(
    mass_kg: float,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    speeds = np.linalg.norm(velocities, axis=1)
    kinetic_energy_ev = 0.5 * mass_kg * speeds**2 / elementary_charge
    axial_speed = velocities[:, 2]
    angle_deg = np.degrees(
        np.arccos(
            np.clip(axial_speed / np.maximum(speeds, 1.0e-30), -1.0, 1.0)
        )
    )
    return kinetic_energy_ev, angle_deg


def sample_beam_phase_space(
    beam: BeamConfig,
    *,
    max_particles: int = 20000,
) -> dict[str, np.ndarray]:
    _validate_beam_preview(beam)
    velocity_source = beam_velocity_source(beam)
    template_beam = beam
    if velocity_source != "static":
        template_beam = replace(
            beam,
            kinetic_energy_ev=None,
            direction_axis="+z",
            cone_half_angle_deg=0.0,
            velocity_jitter_m_per_s=0.0,
        )
    template = build_ion_template(template_beam)
    rng = np.random.default_rng(7)
    count = min(max(1, int(beam.particle_count)), max_particles)
    positions = _sample_positions(
        beam,
        np.asarray(template.initial_position_m, dtype=np.float64),
        rng,
        count,
    )
    if velocity_source == "static":
        velocities = _sample_velocities(
            beam,
            np.asarray(template.initial_velocity_m_per_s, dtype=np.float64),
            rng,
            count,
        )
        kinetic_energy_ev, angle_deg = _beam_diagnostics(
            beam,
            template.mass_kg,
            velocities,
        )
    else:
        velocities = _sample_gas_field_velocities(beam, positions, rng)
        kinetic_energy_ev, angle_deg = _gas_field_diagnostics(
            template.mass_kg,
            velocities,
        )
    return {
        "positions_m": positions,
        "velocities_m_per_s": velocities,
        "kinetic_energy_ev": kinetic_energy_ev,
        "angle_deg": angle_deg,
    }


def run_beam_smoke_task(
    message_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    app_config: AppConfig,
) -> None:
    if stop_event.is_set():
        raise TaskStopped()
    sample = sample_beam_phase_space(app_config.beam)
    output_dir = Path(app_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "beam_smoke_samples.npz"
    np.savez(sample_path, **sample)
    from .plotting import make_beam_preview_figure

    preview_path = output_dir / "beam_smoke_preview.png"
    fig = make_beam_preview_figure(sample)
    fig.savefig(preview_path, dpi=140)
    stats = {
        "sample_count": int(sample["positions_m"].shape[0]),
        "velocity_source": beam_velocity_source(app_config.beam),
        "energy_ev_mean": float(np.mean(sample["kinetic_energy_ev"])),
        "energy_ev_p95": float(
            np.percentile(sample["kinetic_energy_ev"], 95)
        ),
        "angle_deg_p95": float(np.percentile(sample["angle_deg"], 95)),
        "sample_path": str(sample_path),
        "preview_path": str(preview_path),
    }
    _post(
        message_queue,
        "beam_smoke_finished",
        sample=sample,
        stats=stats,
    )
