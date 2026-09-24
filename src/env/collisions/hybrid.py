"""Hybrid Langevin collision updates."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.constants import Boltzmann

from ...config import CollisionBatchUpdate


def _hybrid_parameters(
    state: dict[str, np.ndarray],
    hybrid_indices: np.ndarray,
    velocities_m_per_s: np.ndarray,
    gas_velocities_m_per_s: np.ndarray,
    gas_temperatures_k: np.ndarray,
    lambda_dt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    velocities = np.asarray(velocities_m_per_s, dtype=np.float64)
    gas_velocities = np.asarray(gas_velocities_m_per_s, dtype=np.float64)
    temperatures_k = np.maximum(
        np.asarray(gas_temperatures_k, dtype=np.float64),
        1.0,
    )
    masses_kg = np.maximum(
        np.asarray(state["mass"][hybrid_indices], dtype=np.float64),
        1.0e-30,
    )
    lambda_dt = np.maximum(np.asarray(lambda_dt, dtype=np.float64), 0.0)
    return velocities, gas_velocities, temperatures_k, masses_kg, lambda_dt


def _record_hybrid_update(
    runtime: Any,
    state: dict[str, np.ndarray],
    hybrid_indices: np.ndarray,
    hybrid_lambda_dt: np.ndarray,
) -> CollisionBatchUpdate:
    estimated_counts = np.maximum(
        np.rint(hybrid_lambda_dt).astype(np.int64),
        1,
    )
    runtime._particle_collision_counts[hybrid_indices] += estimated_counts
    weights = np.asarray(state["weight"][hybrid_indices], dtype=np.float64)
    represented_real_ions = float(np.sum(weights))
    event_count = int(hybrid_indices.size)
    runtime._hybrid_langevin_macro_events += event_count
    runtime._hybrid_langevin_real_ions_represented += represented_real_ions
    runtime._hybrid_langevin_estimated_binary_collisions += float(
        np.sum(hybrid_lambda_dt * weights)
    )
    runtime._collision_real_ions_represented += represented_real_ions
    runtime._collision_macro_events += event_count
    return CollisionBatchUpdate(
        collision_hits=event_count,
        fragmentation_hits=0,
        collision_real_ions_represented=represented_real_ions,
        fragmented_real_ions_represented=0.0,
        updated_indices=hybrid_indices.copy(),
        deactivated_indices=np.zeros(0, dtype=np.int32),
    )


class HybridCollisionMixin:
    """Apply aggregate OU updates to high-collision candidates."""

    def _apply_hybrid_selected(
        self,
        state: dict[str, np.ndarray],
        particle_indices: np.ndarray,
        velocities_m_per_s: np.ndarray,
        gas_velocities_m_per_s: np.ndarray,
        gas_temperatures_k: np.ndarray,
        lambda_dt: np.ndarray,
    ) -> CollisionBatchUpdate:
        runtime = self.runtime
        hybrid_indices = np.asarray(particle_indices, dtype=np.int32)
        if hybrid_indices.size == 0:
            return runtime._empty_collision_update()
        velocities, gas_velocities, temperatures_k, masses_kg, lambda_dt = (
            _hybrid_parameters(
                state,
                hybrid_indices,
                velocities_m_per_s,
                gas_velocities_m_per_s,
                gas_temperatures_k,
                lambda_dt,
            )
        )
        alpha = np.exp(-lambda_dt)
        thermal_std = np.sqrt(Boltzmann * temperatures_k / masses_kg)
        diffusion_scale = thermal_std * np.sqrt(
            np.maximum(1.0 - alpha * alpha, 0.0)
        )
        state["velocities"][hybrid_indices] = (
            gas_velocities
            + alpha[:, None] * (velocities - gas_velocities)
            + diffusion_scale[:, None]
            * runtime.rng.normal(0.0, 1.0, size=velocities.shape)
        )
        return _record_hybrid_update(runtime, state, hybrid_indices, lambda_dt)


__all__ = ["HybridCollisionMixin"]
