"""Explicit binary collisions; dissociation belongs to the time scheduler."""

from __future__ import annotations

import numpy as np
from typing import Any

from ...config import CollisionBatchUpdate


def _apply_hit_physics(
    runtime: Any, state: dict[str, np.ndarray], hit_indices: np.ndarray,
    gas_velocities: np.ndarray, gas_temperatures: np.ndarray,
    gas_number_density: np.ndarray, gas_mass_density: np.ndarray, dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    velocity, temperature = runtime.adapter.apply_collision_state_batch(
        velocities_m_per_s=state["velocities"][hit_indices],
        masses_kg=state["mass"][hit_indices],
        internal_temperatures_k=state["temperature"][hit_indices],
        gas_velocities_m_per_s=np.asarray(gas_velocities, dtype=np.float64),
        gas_temperatures_k=np.asarray(gas_temperatures, dtype=np.float64),
        gas_number_density_m3=np.asarray(gas_number_density, dtype=np.float64),
        gas_mass_density_kg_per_m3=np.asarray(gas_mass_density, dtype=np.float64),
        template=runtime.template, gas_name=str(runtime.config.gas_species),
        dt_s=dt_s, rng=runtime.rng,
    )
    state["velocities"][hit_indices] = velocity
    state["temperature"][hit_indices] = temperature
    runtime._particle_collision_counts[hit_indices] += 1
    return velocity, temperature


class ExplicitCollisionMixin:
    """Update only velocity, internal temperature, and collision counters."""

    def _apply_explicit_hits(
        self,
        state: dict[str, np.ndarray],
        hit_indices: np.ndarray,
        hit_positions: np.ndarray,
        gas_velocities: np.ndarray,
        gas_temperatures: np.ndarray,
        gas_number_density: np.ndarray,
        gas_mass_density: np.ndarray,
        dt_s: float,
        time_s: float,
    ) -> CollisionBatchUpdate:
        runtime = self.runtime
        indices = np.asarray(hit_indices, dtype=np.int32)
        if indices.size == 0:
            return runtime._empty_collision_update()
        _apply_hit_physics(
            runtime, state, indices, gas_velocities, gas_temperatures,
            gas_number_density, gas_mass_density, dt_s,
        )
        represented = float(np.sum(state["weight"][indices]))
        runtime._collision_real_ions_represented += represented
        runtime._collision_macro_events += int(indices.size)
        return CollisionBatchUpdate(
            collision_hits=int(indices.size),
            fragmentation_hits=0,
            collision_real_ions_represented=represented,
            fragmented_real_ions_represented=0.0,
            updated_indices=indices,
            deactivated_indices=np.zeros(0, dtype=np.int32),
        )


__all__ = ["ExplicitCollisionMixin"]
