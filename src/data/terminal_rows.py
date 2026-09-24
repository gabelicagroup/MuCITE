"""Normalization and row construction for terminal-event batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import elementary_charge

from ..agents.events import TerminalEventBatch
from ..config import PARTICLE_STATUS_NAMES


@dataclass(frozen=True)
class NormalizedTerminalBatch:
    particle_indices: np.ndarray
    status_codes: np.ndarray
    event_times_s: np.ndarray
    positions_m: np.ndarray
    velocities_m_per_s: np.ndarray
    temperatures_k: np.ndarray
    represented_real_ions: np.ndarray
    parent_real_ions_remaining: np.ndarray
    fragmented_real_ions_represented: np.ndarray
    masses_kg: np.ndarray
    track_ids: np.ndarray
    birth_time_s: np.ndarray
    collision_counts: np.ndarray
    tof_s: np.ndarray
    electrode_ids: np.ndarray
    surface_distance_m: np.ndarray


def normalize_terminal_batch(
    batch: TerminalEventBatch,
) -> NormalizedTerminalBatch:
    """Convert a public event DTO into validated numerical arrays."""

    particle_indices = np.asarray(batch.particle_indices, dtype=np.int32)
    represented = np.asarray(batch.represented_real_ions, dtype=np.float64)
    parent = _parent_weights(batch, represented)
    fragments = _fragment_weights(batch, represented, parent)
    count = particle_indices.size
    _require_length(parent, count, "parent_real_ions_remaining")
    _require_length(
        fragments,
        count,
        "fragmented_real_ions_represented",
    )
    return NormalizedTerminalBatch(
        particle_indices=particle_indices,
        status_codes=np.asarray(batch.status_codes, dtype=np.int16),
        event_times_s=_event_times(batch.event_time_s, count),
        positions_m=np.asarray(batch.positions_m, dtype=np.float64),
        velocities_m_per_s=np.asarray(batch.velocities_m_per_s, dtype=np.float64),
        temperatures_k=np.asarray(batch.temperatures_k, dtype=np.float64),
        represented_real_ions=represented,
        parent_real_ions_remaining=parent,
        fragmented_real_ions_represented=fragments,
        masses_kg=np.asarray(batch.masses_kg, dtype=np.float64),
        track_ids=np.asarray(batch.track_ids, dtype=np.int64),
        birth_time_s=np.asarray(batch.birth_time_s, dtype=np.float64),
        collision_counts=np.asarray(batch.collision_counts, dtype=np.int64),
        tof_s=np.asarray(batch.tof_s, dtype=np.float64),
        electrode_ids=np.asarray(batch.electrode_ids, dtype=np.int32),
        surface_distance_m=np.asarray(batch.surface_distance_m, dtype=np.float64),
    )


def _parent_weights(
    batch: TerminalEventBatch,
    represented: np.ndarray,
) -> np.ndarray:
    if batch.parent_real_ions_remaining is None:
        return represented.copy()
    return np.asarray(batch.parent_real_ions_remaining, dtype=np.float64)


def _fragment_weights(
    batch: TerminalEventBatch,
    represented: np.ndarray,
    parent: np.ndarray,
) -> np.ndarray:
    if batch.fragmented_real_ions_represented is None:
        return np.maximum(represented - parent, 0.0)
    return np.asarray(
        batch.fragmented_real_ions_represented,
        dtype=np.float64,
    )


def _require_length(values: np.ndarray, count: int, name: str) -> None:
    if values.shape[0] != count:
        raise ValueError(f"{name} must have length n_particles.")


def _event_times(value: np.ndarray | float, count: int) -> np.ndarray:
    event_time_s = np.asarray(value, dtype=np.float64)
    if event_time_s.ndim == 0:
        result = np.full(count, float(event_time_s), dtype=np.float64)
    elif event_time_s.shape == (count,):
        result = event_time_s
    else:
        raise ValueError(
            "event_time_s must be scalar or have length n_particles."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("event_time_s contains non-finite values.")
    return result


def build_terminal_rows(
    batch: NormalizedTerminalBatch,
    offsets: np.ndarray,
    event_id_start: int,
) -> list[dict[str, float | int | str]]:
    """Build stable CSV dictionaries for selected batch offsets."""

    positions = batch.positions_m[offsets]
    velocities = batch.velocities_m_per_s[offsets]
    radii_m = np.linalg.norm(positions[:, :2], axis=1)
    speeds_m_per_s = np.linalg.norm(velocities, axis=1)
    kinetic_energy_ev = (
        0.5 * batch.masses_kg[offsets] * speeds_m_per_s**2
        / elementary_charge
    )
    return [
        _build_row(
            batch,
            int(offset),
            int(event_id_start + row_index),
            float(radii_m[row_index]),
            float(speeds_m_per_s[row_index]),
            float(kinetic_energy_ev[row_index]),
        )
        for row_index, offset in enumerate(offsets)
    ]


def _build_row(
    batch: NormalizedTerminalBatch,
    offset: int,
    event_id: int,
    radius_m: float,
    speed_m_per_s: float,
    kinetic_energy_ev: float,
) -> dict[str, float | int | str]:
    status_code = int(batch.status_codes[offset])
    position = batch.positions_m[offset]
    velocity = batch.velocities_m_per_s[offset]
    return {
        "event_id": event_id,
        "track_id": int(batch.track_ids[offset]),
        "slot_id": int(batch.particle_indices[offset]),
        "status_code": status_code,
        "status": PARTICLE_STATUS_NAMES.get(status_code, str(status_code)),
        "event_time_s": float(batch.event_times_s[offset]),
        "birth_time_s": float(batch.birth_time_s[offset]),
        "tof_s": float(batch.tof_s[offset]),
        "x_m": float(position[0]),
        "y_m": float(position[1]),
        "z_m": float(position[2]),
        "r_m": radius_m,
        "vx_m_per_s": float(velocity[0]),
        "vy_m_per_s": float(velocity[1]),
        "vz_m_per_s": float(velocity[2]),
        "speed_m_per_s": speed_m_per_s,
        "ke_ev": kinetic_energy_ev,
        "internal_temperature_k": float(batch.temperatures_k[offset]),
        "represented_real_ions": float(batch.represented_real_ions[offset]),
        "parent_real_ions_remaining": float(
            batch.parent_real_ions_remaining[offset]
        ),
        "fragmented_real_ions_represented": float(
            batch.fragmented_real_ions_represented[offset]
        ),
        "collision_count_per_ion": int(batch.collision_counts[offset]),
        "electrode_id": int(batch.electrode_ids[offset]),
        "surface_distance_m": float(batch.surface_distance_m[offset]),
    }
