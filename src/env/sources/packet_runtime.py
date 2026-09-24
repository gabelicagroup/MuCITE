"""Packet-source initialization behavior for the simulation model."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...agents.slot_state import ParticleRuntimeState
from ...config import SLOT_ACTIVE
from .continuous import sample_source_xy_offsets


class PacketSourceRuntimeMixin:
    """Initialize one finite packet without owning engine scheduling."""

    def initialize_ions(self) -> None:
        """Create configured source particles directly in the cloud."""

        self._reset_collision_statistics()
        if self.config.source_mode == "continuous-current":
            self._initialize_continuous_current_source()
            return
        self._prepare_packet_slots()
        positions, velocities = self._sample_packet_phase_space()
        state = self._packet_cloud_state(positions, velocities)
        self._particle_ever_used = np.ones(
            self.config.ion_count,
            dtype=bool,
        )
        self._upload_cloud_state(state)

    def _prepare_packet_slots(self) -> None:
        ParticleRuntimeState.packet(
            self.config.ion_count,
            self.template,
            macro_weight=self.config.macro_particle_weight,
        ).bind_to(self)
        self.terminal_event_recorder.clear()
        self._terminal_event_rows = self.terminal_event_recorder.rows

    def _sample_packet_phase_space(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = self.config.ion_count
        base_position = np.asarray(
            self.template.initial_position_m,
            dtype=float,
        )
        base_velocity = np.asarray(
            self.template.initial_velocity_m_per_s,
            dtype=float,
        )
        positions = self._sample_packet_positions(count, base_position)
        velocities = self._sample_packet_velocities(
            count,
            base_velocity,
        )
        positions[:, 2] = np.clip(
            positions[:, 2],
            0.0,
            self.config.domain_length_m,
        )
        return positions, velocities

    def _sample_packet_positions(
        self,
        count: int,
        base_position: np.ndarray,
    ) -> np.ndarray:
        positions = np.tile(base_position, (count, 1))
        radius_m = self.config.source_radius_m
        if radius_m is None or radius_m <= 0.0:
            positions += self.rng.normal(
                0.0,
                self.config.initial_position_jitter_m,
                size=(count, 3),
            )
            return positions
        x_offsets_m, y_offsets_m = sample_source_xy_offsets(
            self.rng,
            count,
            source_radius_m=radius_m,
            source_profile=self.config.source_profile,
            source_gaussian_sigma_m=self.config.source_gaussian_sigma_m,
        )
        positions[:, 0] = base_position[0] + x_offsets_m
        positions[:, 1] = base_position[1] + y_offsets_m
        positions[:, 2] = base_position[2] + self.rng.normal(
            0.0,
            self.config.initial_position_jitter_m,
            size=count,
        )
        return positions

    def _sample_packet_velocities(
        self,
        count: int,
        base_velocity: np.ndarray,
    ) -> np.ndarray:
        velocities = np.tile(base_velocity, (count, 1))
        base_speed = float(np.linalg.norm(base_velocity))
        if self.config.cone_half_angle_rad > 0.0 and base_speed > 0.0:
            velocities = self._sample_conical_velocities(count, base_speed)
        velocities += self.rng.normal(
            0.0,
            self.config.initial_velocity_jitter_m_per_s,
            size=(count, 3),
        )
        return velocities

    def _sample_conical_velocities(
        self,
        count: int,
        base_speed: float,
    ) -> np.ndarray:
        cosine_limit = np.cos(self.config.cone_half_angle_rad)
        cos_theta = 1.0 - self.rng.random(count) * (1.0 - cosine_limit)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
        azimuth_rad = 2.0 * np.pi * self.rng.random(count)
        return base_speed * np.column_stack(
            (
                sin_theta * np.cos(azimuth_rad),
                sin_theta * np.sin(azimuth_rad),
                cos_theta,
            )
        )

    def _packet_cloud_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> dict[str, np.ndarray]:
        count = self.config.ion_count
        return {
            "positions": np.asarray(positions, dtype=np.float64),
            "velocities": np.asarray(velocities, dtype=np.float64),
            "charge": self._particle_charge_c,
            "mass": self._particle_mass_kg,
            "weight": self._particle_weight,
            "active": np.ones(count, dtype=np.int32),
            "temperature": np.full(
                count,
                self.template.initial_internal_temperature_k,
                dtype=np.float64,
            ),
            "phase": np.full(count, SLOT_ACTIVE, dtype=np.int16),
        }


__all__ = ["PacketSourceRuntimeMixin"]
