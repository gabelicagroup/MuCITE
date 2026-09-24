"""Taichi cloud upload/download adapters for the coupled runtime."""

from __future__ import annotations

import numpy as np
from scipy.constants import elementary_charge

from ...config import SLOT_FREE


class CloudStateIOMixin:
    """Translate between NumPy runtime state and Taichi entity fields."""

    def _upload_cloud_state(self, state: dict[str, np.ndarray]) -> None:
        self.cloud.x.from_numpy(
            np.asarray(state["positions"][:, 0], dtype=np.float64)
        )
        self.cloud.y.from_numpy(
            np.asarray(state["positions"][:, 1], dtype=np.float64)
        )
        self.cloud.z.from_numpy(
            np.asarray(state["positions"][:, 2], dtype=np.float64)
        )
        self.cloud.vx.from_numpy(
            np.asarray(state["velocities"][:, 0], dtype=np.float64)
        )
        self.cloud.vy.from_numpy(
            np.asarray(state["velocities"][:, 1], dtype=np.float64)
        )
        self.cloud.vz.from_numpy(
            np.asarray(state["velocities"][:, 2], dtype=np.float64)
        )
        self.cloud.q.from_numpy(np.asarray(state["charge"], dtype=np.float64))
        self.cloud.m.from_numpy(np.asarray(state["mass"], dtype=np.float64))
        self.cloud.weight.from_numpy(
            np.asarray(state["weight"], dtype=np.float64)
        )
        self.cloud.active.from_numpy(
            np.asarray(state["active"], dtype=np.int32)
        )
        self.cloud.internal_temperature.from_numpy(
            np.asarray(state["temperature"], dtype=np.float64)
        )

    def _release_free_slot(
        self,
        state: dict[str, np.ndarray],
        slot_index: int,
    ) -> None:
        state["active"][slot_index] = 0
        state["phase"][slot_index] = SLOT_FREE
        state["weight"][slot_index] = 0.0
        self._particle_weight[slot_index] = 0.0
        self._particle_parent_weight[slot_index] = 0.0
        self._particle_phase_codes[slot_index] = SLOT_FREE

    def _upload_active_state_updates(
        self,
        state: dict[str, np.ndarray],
        particle_indices: np.ndarray,
    ) -> None:
        indices = np.asarray(particle_indices, dtype=np.int32)
        if indices.size == 0:
            return
        self.cloud.apply_state_updates(
            indices,
            np.asarray(state["positions"][indices], dtype=np.float64),
            np.asarray(state["velocities"][indices], dtype=np.float64),
            np.asarray(state["weight"][indices], dtype=np.float64),
            np.asarray(state["temperature"][indices], dtype=np.float64),
            float(self.template.charge_state * elementary_charge),
            float(self.template.mass_kg),
            int(indices.size),
        )

    def _upload_deactivated_indices(
        self,
        particle_indices: np.ndarray,
    ) -> None:
        indices = np.asarray(particle_indices, dtype=np.int32)
        if indices.size:
            self.cloud.deactivate_indices(indices, int(indices.size))

    def _download_selected_motion_state(
        self,
        particle_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(particle_indices, dtype=np.int32)
        if indices.size == 0:
            empty = np.zeros((0, 3), dtype=np.float64)
            return empty, empty.copy()
        positions = np.zeros((indices.size, 3), dtype=np.float64)
        velocities = np.zeros((indices.size, 3), dtype=np.float64)
        self.cloud.gather_motion_state(
            indices,
            positions,
            velocities,
            int(indices.size),
        )
        return positions, velocities

    def _download_cloud_state(self) -> dict[str, np.ndarray]:
        positions = np.column_stack(
            (self.cloud.x.to_numpy(), self.cloud.y.to_numpy(), self.cloud.z.to_numpy())
        )
        velocities = np.column_stack(
            (
                self.cloud.vx.to_numpy(),
                self.cloud.vy.to_numpy(),
                self.cloud.vz.to_numpy(),
            )
        )
        return {
            "positions": np.asarray(positions, dtype=np.float64),
            "velocities": np.asarray(velocities, dtype=np.float64),
            "charge": self._particle_charge_c,
            "mass": self._particle_mass_kg,
            "weight": self._particle_weight,
            "active": np.asarray(self.cloud.active.to_numpy(), dtype=np.int32),
            "temperature": np.asarray(
                self.cloud.internal_temperature.to_numpy(),
                dtype=np.float64,
            ),
            "phase": np.asarray(self._particle_phase_codes, dtype=np.int16),
        }

    def _download_motion_state_into(
        self,
        state: dict[str, np.ndarray],
    ) -> None:
        indices = self._active_indices(state)
        if indices.size == 0:
            return
        positions, velocities = self._download_selected_motion_state(indices)
        state["positions"][indices] = positions
        state["velocities"][indices] = velocities

    def _active_indices(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return np.flatnonzero(state["active"] != 0)
