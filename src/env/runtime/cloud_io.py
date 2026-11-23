"""Taichi cloud upload/download adapters for the coupled runtime."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
from scipy.constants import elementary_charge

from ...config import SLOT_FREE
from .step_cache import ParticleStepCache, invalidate_particle_cache


class CloudStateIOMixin:
    """Translate between NumPy runtime state and Taichi entity fields."""

    @contextmanager
    def _micro_step_scope(self, state: dict[str, np.ndarray]) -> Iterator[None]:
        """Bound cached active slots and motion to one engine microstep."""
        self._particle_step_cache = ParticleStepCache(state, motion_enabled=False)
        try:
            yield
        finally:
            invalidate_particle_cache(self)

    def _enable_motion_cache(self) -> None:
        """Enable reuse when hybrid coordination needs repeated motion reads."""
        cache = getattr(self, "_particle_step_cache", None)
        if cache is not None:
            cache.motion_enabled = True

    def _invalidate_motion_cache(self, *, after_push: bool = False) -> None:
        """Discard pre-update values and choose reuse for the new stage."""
        cache = getattr(self, "_particle_step_cache", None)
        if cache is not None:
            cache.invalidate_motion()
            if after_push:
                cache.motion_enabled = (
                    getattr(self, "electrode_mask", None) is not None
                    or getattr(self, "electrode_mask3d", None) is not None
                    or getattr(getattr(self, "config", None), "fragmentation_mode", "") == "loss"
                )

    def _upload_cloud_state(self, state: dict[str, np.ndarray]) -> None:
        invalidate_particle_cache(self)
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
        invalidate_particle_cache(self)
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
        CloudStateIOMixin._invalidate_motion_cache(self)
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
        cache = getattr(self, "_particle_step_cache", None)
        gather = lambda selected: CloudStateIOMixin._gather_motion_rows(self, selected)
        return gather(indices) if cache is None else cache.read(indices, gather)

    def _gather_motion_rows(
        self, indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Copy only requested slots, bypassing the stage cache."""
        positions = np.empty((indices.size, 3), dtype=np.float64)
        velocities = np.empty_like(positions)
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
            "dissociation_hazard": self._particle_dissociation_hazard,
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
        cache = getattr(self, "_particle_step_cache", None)
        if cache is not None and cache.state is state:
            return cache.indices
        return np.flatnonzero(state["active"] != 0)
