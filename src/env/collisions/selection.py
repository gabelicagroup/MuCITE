"""Whole-macro collision candidate selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.constants import Avogadro

from ...config import CollisionBatchUpdate, LocalStateBatch
from .iict_lite.physics import mean_relative_speed_m_per_s
from .probability import collision_probability_from_lambda_dt


def _collision_inputs(
    operator: Any,
    state: dict[str, np.ndarray],
    active_indices: np.ndarray,
    positions_m: np.ndarray,
    samples: LocalStateBatch,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runtime = operator.runtime
    velocities = np.asarray(
        state["velocities"][active_indices],
        dtype=np.float64,
    )
    relative_speed = np.linalg.norm(
        velocities - samples.gas_velocity_m_per_s,
        axis=1,
    )
    if str(runtime.config.collision_physics_backend).lower() == "iict-lite":
        relative_speed = np.asarray(
            mean_relative_speed_m_per_s(
                relative_speed,
                samples.temperature_k,
                runtime.config.gas_molar_mass_kg_per_mol / Avogadro,
            ),
            dtype=np.float64,
        )
    collision_rate = (
        samples.number_density_m3
        * runtime.template.collision_cross_section_m2
        * np.maximum(relative_speed, 1.0e-9)
    )
    lambda_dt = collision_rate * float(dt_s)
    hybrid_mask = np.zeros(active_indices.size, dtype=bool)
    if str(runtime.config.collision_model).lower() == "hybrid-langevin":
        z_m = positions_m[:, 2]
        hybrid_mask = (
            (z_m >= float(runtime.config.langevin_z_start_m))
            & (z_m <= float(runtime.config.langevin_z_end_m))
            & (lambda_dt >= float(runtime.config.langevin_switch_probability))
        )
    return velocities, lambda_dt, hybrid_mask


def _hybrid_update(
    operator: Any,
    state: dict[str, np.ndarray],
    active_indices: np.ndarray,
    velocities: np.ndarray,
    samples: LocalStateBatch,
    lambda_dt: np.ndarray,
    hybrid_mask: np.ndarray,
) -> CollisionBatchUpdate:
    runtime = operator.runtime
    if not np.any(hybrid_mask):
        return runtime._empty_collision_update()
    return operator._apply_hybrid_selected(
        state,
        np.asarray(active_indices[hybrid_mask], dtype=np.int32),
        np.asarray(velocities[hybrid_mask], dtype=np.float64),
        np.asarray(samples.gas_velocity_m_per_s[hybrid_mask], dtype=np.float64),
        np.asarray(samples.temperature_k[hybrid_mask], dtype=np.float64),
        np.asarray(lambda_dt[hybrid_mask], dtype=np.float64),
    )


def _explicit_hit_offsets(
    runtime: Any,
    lambda_dt: np.ndarray,
    hybrid_mask: np.ndarray,
) -> np.ndarray:
    explicit_mask = ~hybrid_mask
    probability = collision_probability_from_lambda_dt(lambda_dt[explicit_mask])
    random_values = runtime.rng.random(int(np.count_nonzero(explicit_mask)))
    hit_mask = random_values < probability
    return np.flatnonzero(explicit_mask)[hit_mask]


def apply_collision_updates(
    operator: Any,
    state: dict[str, np.ndarray],
    active_indices: np.ndarray,
    positions_m: np.ndarray,
    samples: LocalStateBatch,
    dt_s: float,
    time_s: float,
) -> CollisionBatchUpdate:
    runtime = operator.runtime
    active_indices = np.asarray(active_indices, dtype=np.int32)
    positions_m = np.asarray(positions_m, dtype=np.float64)
    if active_indices.size == 0:
        return runtime._empty_collision_update()
    velocities, lambda_dt, hybrid_mask = _collision_inputs(
        operator, state, active_indices, positions_m, samples, dt_s,
    )
    hybrid_update = _hybrid_update(
        operator, state, active_indices, velocities, samples,
        lambda_dt, hybrid_mask,
    )
    hit_offsets = _explicit_hit_offsets(runtime, lambda_dt, hybrid_mask)
    if hit_offsets.size == 0:
        return hybrid_update
    hit_indices = np.asarray(active_indices[hit_offsets], dtype=np.int32)
    explicit_update = operator._apply_explicit_hits(
        state,
        hit_indices,
        np.asarray(positions_m[hit_offsets], dtype=np.float64),
        np.asarray(samples.gas_velocity_m_per_s[hit_offsets], dtype=np.float64),
        np.asarray(samples.temperature_k[hit_offsets], dtype=np.float64),
        np.asarray(samples.number_density_m3[hit_offsets], dtype=np.float64),
        np.asarray(samples.mass_density_kg_per_m3[hit_offsets], dtype=np.float64),
        dt_s=dt_s,
        time_s=time_s,
    )
    return operator._combine_updates(hybrid_update, explicit_update)


__all__ = ["apply_collision_updates"]
