"""Progress, survivor, and particle-table projections."""

from __future__ import annotations

import numpy as np
from scipy.constants import elementary_charge

from ...agents.ion import IonState
from ...config import SimulationProgress


class ResultProjectionMixin:
    """Build read-only runtime projections consumed by engine/output ports."""

    def _build_progress_snapshot(
        self,
        state: dict[str, np.ndarray],
        time_s: float,
        macro_step_index: int,
        micro_step_index: int,
        collision_count: int,
    ) -> SimulationProgress:
        active_indices = self._active_indices(state)
        if len(active_indices) > 0:
            mean_temperature = float(
                np.mean(state["temperature"][active_indices])
            )
            mean_axial_position = float(
                np.mean(state["positions"][active_indices, 2])
            )
        else:
            mean_temperature = 0.0
            mean_axial_position = 0.0
        return SimulationProgress(
            time_s=float(time_s),
            macro_step_index=macro_step_index,
            micro_step_index=micro_step_index,
            ion_count=int(len(active_indices)),
            collision_count=collision_count,
            mean_internal_temperature_k=mean_temperature,
            mean_axial_position_m=mean_axial_position,
            particle_backend=self.particle_backend,
            backend_message=self.backend_message,
        )

    def _collect_survivors(
        self,
        state: dict[str, np.ndarray],
    ) -> list[IonState]:
        survivors: list[IonState] = []
        for particle_index in self._active_indices(state):
            survivors.append(
                IonState(
                    position_m=np.asarray(
                        state["positions"][particle_index],
                        dtype=float,
                    ),
                    velocity_m_per_s=np.asarray(
                        state["velocities"][particle_index],
                        dtype=float,
                    ),
                    mass_kg=float(state["mass"][particle_index]),
                    charge_c=float(state["charge"][particle_index]),
                    collision_cross_section_m2=(
                        self.template.collision_cross_section_m2
                    ),
                    internal_temperature_k=float(
                        state["temperature"][particle_index]
                    ),
                    species_name=(
                        f"{self.template.name}_{particle_index:04d}"
                    ),
                    alive=True,
                )
            )
        return survivors

    def _build_particle_snapshot(
        self,
        state: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return the established 18-column active-particle table."""

        indices = self._active_indices(state)
        if len(indices) == 0:
            return np.zeros((0, 18), dtype=np.float64)
        positions = np.asarray(state["positions"][indices], dtype=np.float64)
        velocities = np.asarray(state["velocities"][indices], dtype=np.float64)
        masses = np.asarray(state["mass"][indices], dtype=np.float64)
        temperatures = np.asarray(state["temperature"][indices], dtype=np.float64)
        weights = np.asarray(state["weight"][indices], dtype=np.float64)
        parents = np.asarray(
            self._particle_parent_weight[indices],
            dtype=np.float64,
        )
        radius_m = np.linalg.norm(positions[:, :2], axis=1)
        speed_m_per_s = np.linalg.norm(velocities, axis=1)
        kinetic_energy_j = 0.5 * masses * speed_m_per_s**2
        parents = np.minimum(
            np.maximum(parents, 0.0),
            np.maximum(weights, 0.0),
        )
        return np.column_stack(
            (
                radius_m, positions[:, 2], temperatures, kinetic_energy_j,
                positions[:, 0], positions[:, 1], velocities[:, 0],
                velocities[:, 1], velocities[:, 2], speed_m_per_s,
                kinetic_energy_j / elementary_charge, weights, parents,
                np.maximum(weights - parents, 0.0),
                self._particle_track_id[indices].astype(np.float64),
                indices.astype(np.float64),
                self._particle_status_codes[indices].astype(np.float64),
                self._particle_collision_counts[indices].astype(np.float64),
            )
        )
