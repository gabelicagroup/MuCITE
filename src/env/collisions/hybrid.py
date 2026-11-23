"""Joint IICT velocity/internal-energy collision updates in the hybrid window."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config import CollisionBatchUpdate
from .joint_step import JointIictCollisionStep, JointCollisionResult


def _record_joint_update(
    runtime: Any, state: dict[str, np.ndarray], indices: np.ndarray,
    result: JointCollisionResult,
) -> CollisionBatchUpdate:
    """Binary counts are actual events; dense-step counts remain estimates."""
    counts = result.estimated_counts
    runtime._particle_collision_counts[indices] += counts
    weights = np.asarray(state["weight"][indices], dtype=np.float64)
    represented = float(np.sum(weights * counts))
    gaussian_steps = int(np.sum(result.gaussian_steps))
    runtime._collision_real_ions_represented += represented
    runtime._collision_macro_events += int(np.sum(counts))
    runtime._hybrid_langevin_macro_events += gaussian_steps
    runtime._hybrid_langevin_real_ions_represented += float(np.sum(weights * result.gaussian_steps))
    runtime._hybrid_langevin_estimated_binary_collisions += float(
        np.sum(weights * result.gaussian_expected_counts))
    info = runtime.dissociation_runtime_info
    info["hybrid_joint_gaussian_steps"] += gaussian_steps
    info["hybrid_joint_binary_events"] += int(np.sum(result.binary_counts))
    info["hybrid_joint_guard_fallbacks"] += result.guard_fallbacks
    changed = (result.binary_counts > 0) | (result.gaussian_steps > 0)
    state["velocities"][indices[changed]] = result.velocities_m_per_s[changed]
    state["temperature"][indices[changed]] = result.temperatures_k[changed]
    return CollisionBatchUpdate(
        collision_hits=int(np.sum(counts)), fragmentation_hits=0,
        collision_real_ions_represented=represented,
        fragmented_real_ions_represented=0.0,
        updated_indices=indices[changed], deactivated_indices=np.zeros(0, dtype=np.int32),
    )


class HybridCollisionMixin:
    """Use guarded joint diffusion or actual IICT events, without weight decay."""

    def _apply_hybrid_selected(
        self, state: dict[str, np.ndarray], particle_indices: np.ndarray,
        velocities_m_per_s: np.ndarray, gas_velocities_m_per_s: np.ndarray,
        gas_temperatures_k: np.ndarray, *,
        gas_number_density_m3: np.ndarray, dt_s: float,
    ) -> CollisionBatchUpdate:
        runtime = self.runtime
        indices = np.asarray(particle_indices, dtype=np.int32)
        if indices.size == 0:
            return runtime._empty_collision_update()
        result = JointIictCollisionStep(
            backend=runtime.adapter, template=runtime.template, config=runtime.config,
            velocities_m_per_s=velocities_m_per_s, masses_kg=state["mass"][indices],
            temperatures_k=state["temperature"][indices],
            gas_velocities_m_per_s=gas_velocities_m_per_s,
            gas_temperatures_k=gas_temperatures_k,
            gas_number_density_m3=gas_number_density_m3, dt_s=dt_s, rng=runtime.rng,
        ).run()
        return _record_joint_update(runtime, state, indices, result)


__all__ = ["HybridCollisionMixin"]
