"""Particle-slot mutations for continuous source injection."""

from __future__ import annotations

import numpy as np
from scipy.constants import elementary_charge

from ...config import (
    PARTICLE_ACTIVE,
    PARTICLE_CAPILLARY_BUFFER,
    SLOT_ACTIVE,
    SLOT_CAPILLARY,
)


class SourceSlotRuntimeMixin:
    """Translate source samples into host and device particle slots."""

    def _populate_capillary_slots(
        self,
        state: dict[str, np.ndarray],
        indices: np.ndarray,
        positions_m: np.ndarray,
        weights: np.ndarray,
        *,
        source_entry_time_s: float,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int32)
        if indices.size == 0:
            return
        positions = np.asarray(positions_m, dtype=np.float64)
        velocities = np.zeros((indices.size, 3), dtype=np.float64)
        velocities[:, 2] = self._source_axial_velocity_m_per_s()
        weights = np.asarray(weights, dtype=np.float64)
        self._assign_source_state(
            state,
            indices,
            positions,
            velocities,
            weights,
            active=0,
            phase=SLOT_CAPILLARY,
        )
        self._assign_source_metadata(
            indices,
            weights,
            status=PARTICLE_CAPILLARY_BUFFER,
            birth_time_s=float("nan"),
            phase=SLOT_CAPILLARY,
        )

    def _activate_external_slots(
        self,
        state: dict[str, np.ndarray],
        indices: np.ndarray,
        *,
        event_time_s: float,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int32)
        if indices.size == 0:
            return
        state["positions"][indices, 2] = self._capillary_exit_z_m()
        positions = np.asarray(state["positions"][indices], dtype=np.float64)
        velocities = self._sample_external_emission_velocities(positions)
        state["velocities"][indices] = velocities
        state["active"][indices] = 1
        state["phase"][indices] = SLOT_ACTIVE
        state["temperature"][indices] = (
            self.template.initial_internal_temperature_k
        )
        self._mark_slots_active(indices, event_time_s)
        self._activate_cloud_slots(
            state,
            indices,
            positions,
            velocities,
        )

    def _populate_external_source_slots(
        self,
        state: dict[str, np.ndarray],
        indices: np.ndarray,
        weights: np.ndarray,
        *,
        event_time_s: float,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int32)
        if indices.size == 0:
            return
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != indices.size:
            raise ValueError("weights must have length n_particles.")
        z_values_m = np.full(
            indices.size,
            self._capillary_exit_z_m(),
            dtype=np.float64,
        )
        positions = self._sample_source_disk_positions(
            int(indices.size),
            z_values_m,
        )
        velocities = self._sample_external_emission_velocities(positions)
        self._assign_source_state(
            state,
            indices,
            positions,
            velocities,
            weights,
            active=1,
            phase=SLOT_ACTIVE,
        )
        self._assign_source_metadata(
            indices,
            weights,
            status=PARTICLE_ACTIVE,
            birth_time_s=float(event_time_s),
            phase=SLOT_ACTIVE,
        )
        self._activate_cloud_slots(state, indices, positions, velocities)

    def _assign_source_state(
        self,
        state: dict[str, np.ndarray],
        indices: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        weights: np.ndarray,
        *,
        active: int,
        phase: int,
    ) -> None:
        state["positions"][indices] = positions
        state["velocities"][indices] = velocities
        state["charge"][indices] = (
            self.template.charge_state * elementary_charge
        )
        state["mass"][indices] = self.template.mass_kg
        state["weight"][indices] = weights
        state["active"][indices] = active
        state["temperature"][indices] = (
            self.template.initial_internal_temperature_k
        )
        state["phase"][indices] = phase

    def _assign_source_metadata(
        self,
        indices: np.ndarray,
        weights: np.ndarray,
        *,
        status: int,
        birth_time_s: float,
        phase: int,
    ) -> None:
        self._particle_charge_c[indices] = (
            self.template.charge_state * elementary_charge
        )
        self._particle_mass_kg[indices] = self.template.mass_kg
        self._particle_weight[indices] = weights
        self._particle_parent_weight[indices] = weights
        self._particle_collision_counts[indices] = 0
        self._particle_status_codes[indices] = status
        self._particle_tof_s[indices] = np.nan
        self._particle_electrode_id[indices] = -1
        self._particle_surface_distance_m[indices] = np.nan
        self._particle_track_id[indices] = self._next_track_ids(indices.size)
        self._particle_birth_time_s[indices] = birth_time_s
        self._particle_phase_codes[indices] = phase
        self._particle_ever_used[indices] = True

    def _next_track_ids(self, count: int) -> np.ndarray:
        stop = self._next_track_id + int(count)
        track_ids = np.arange(
            self._next_track_id,
            stop,
            dtype=np.int64,
        )
        self._next_track_id = stop
        return track_ids

    def _mark_slots_active(
        self,
        indices: np.ndarray,
        event_time_s: float,
    ) -> None:
        self._particle_status_codes[indices] = PARTICLE_ACTIVE
        self._particle_tof_s[indices] = np.nan
        self._particle_electrode_id[indices] = -1
        self._particle_surface_distance_m[indices] = np.nan
        self._particle_birth_time_s[indices] = float(event_time_s)
        self._particle_phase_codes[indices] = SLOT_ACTIVE

    def _activate_cloud_slots(
        self,
        state: dict[str, np.ndarray],
        indices: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        self.cloud.activate_particles(
            indices,
            positions,
            velocities,
            np.asarray(state["weight"][indices], dtype=np.float64),
            float(self.template.charge_state * elementary_charge),
            float(self.template.mass_kg),
            float(self.template.initial_internal_temperature_k),
            int(indices.size),
        )


__all__ = ["SourceSlotRuntimeMixin"]
