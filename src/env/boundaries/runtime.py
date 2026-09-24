"""Runtime collection, ordering, and commit of terminal boundary events."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...config import (
    PARTICLE_ACTIVE,
    PARTICLE_DOMAIN_OUT,
    PARTICLE_ELECTRODE_HIT,
    PARTICLE_RADIAL_OUT,
    PARTICLE_Z_EXIT,
    SLOT_FREE,
)
from .events import TerminalEvents, concatenate_terminal_events
from .localization import _boundary_limits


def _localized_events(
    owner: object,
    events: TerminalEvents,
    start_positions_m: np.ndarray,
    start_velocities_m_per_s: np.ndarray,
    step_start_time_s: float,
    step_dt_s: float,
) -> TerminalEvents:
    localized = owner._localize_terminal_segments(
        start_positions_m=start_positions_m,
        end_positions_m=events.positions_m,
        start_velocities_m_per_s=start_velocities_m_per_s,
        end_velocities_m_per_s=events.velocities_m_per_s,
        endpoint_status=events.statuses,
        endpoint_electrode_id=events.electrode_ids,
        endpoint_surface_distance_m=events.surface_distances_m,
        step_start_time_s=float(step_start_time_s),
        step_dt_s=float(step_dt_s),
    )
    return TerminalEvents(
        indices=events.indices,
        statuses=localized[0],
        positions_m=localized[1],
        velocities_m_per_s=localized[2],
        event_times_s=localized[3],
        electrode_ids=localized[4],
        surface_distances_m=localized[5],
    )


class TerminalBoundaryRuntimeMixin:
    """Endpoint and swept-boundary orchestration for the runtime."""

    def _taichi_runtime_apply_terminal_boundary_events(
        self,
        state: dict[str, np.ndarray],
        event_time_s: float,
        *,
        step_start_time_s: Optional[float] = None,
        step_dt_s: Optional[float] = None,
    ) -> np.ndarray:
        swept = TerminalBoundaryRuntimeMixin._collect_swept_terminal_events(
            self,
            state,
            step_start_time_s,
            step_dt_s,
        )
        endpoint = TerminalBoundaryRuntimeMixin._collect_endpoint_terminal_events(
            self,
            state,
            event_time_s,
            step_start_time_s,
            step_dt_s,
        )
        events = concatenate_terminal_events(swept, endpoint).ordered()
        if events.count == 0:
            return np.zeros(0, dtype=np.int32)
        TerminalBoundaryRuntimeMixin._commit_terminal_events(
            self,
            state,
            events,
        )
        return events.indices

    def _collect_swept_terminal_events(
        self,
        state: dict[str, np.ndarray],
        step_start_time_s: Optional[float],
        step_dt_s: Optional[float],
    ) -> TerminalEvents:
        if step_start_time_s is None or step_dt_s is None:
            return TerminalEvents.empty()
        if self.electrode_mask is None and self.electrode_mask3d is None:
            return TerminalEvents.empty()
        indices = self._active_indices(state).astype(np.int32, copy=False)
        if indices.size == 0:
            return TerminalEvents.empty()
        end_positions, end_velocities = (
            self._download_selected_motion_state(indices)
        )
        start_positions = np.asarray(
            state["positions"][indices],
            dtype=np.float64,
        )
        start_velocities = np.asarray(
            state["velocities"][indices],
            dtype=np.float64,
        )
        fractions = self._find_swept_electrode_hits(
            start_positions,
            end_positions,
        )
        offsets = np.flatnonzero(np.isfinite(fractions))
        if offsets.size == 0:
            return TerminalEvents.empty()
        events = TerminalBoundaryRuntimeMixin._localize_swept_terminal_events(
            self,
            indices[offsets], start_positions[offsets], end_positions[offsets],
            start_velocities[offsets], end_velocities[offsets],
            float(step_start_time_s), float(step_dt_s),
        )
        self._upload_deactivated_indices(events.indices)
        return events

    def _localize_swept_terminal_events(
        self,
        indices: np.ndarray,
        start_positions_m: np.ndarray,
        end_positions_m: np.ndarray,
        start_velocities_m_per_s: np.ndarray,
        end_velocities_m_per_s: np.ndarray,
        step_start_time_s: float,
        step_dt_s: float,
    ) -> TerminalEvents:
        count = int(indices.size)
        provisional = TerminalEvents(
            indices=indices,
            statuses=np.full(
                count,
                PARTICLE_ELECTRODE_HIT,
                dtype=np.int16,
            ),
            electrode_ids=np.full(count, -1, dtype=np.int32),
            surface_distances_m=np.full(count, np.nan, dtype=np.float64),
            positions_m=end_positions_m,
            velocities_m_per_s=end_velocities_m_per_s,
            event_times_s=np.full(
                count,
                step_start_time_s + step_dt_s,
                dtype=np.float64,
            ),
        )
        return _localized_events(
            self,
            provisional,
            start_positions_m,
            start_velocities_m_per_s,
            step_start_time_s,
            step_dt_s,
        )

    def _collect_endpoint_terminal_events(
        self,
        state: dict[str, np.ndarray],
        event_time_s: float,
        step_start_time_s: Optional[float],
        step_dt_s: Optional[float],
    ) -> TerminalEvents:
        TerminalBoundaryRuntimeMixin._select_terminal_endpoint_kernel(self)
        events = TerminalBoundaryRuntimeMixin._download_terminal_endpoint_events(
            self,
            event_time_s,
        )
        if (
            events.count == 0
            or step_start_time_s is None
            or step_dt_s is None
        ):
            return events
        return _localized_events(
            self,
            events,
            np.asarray(
                state["positions"][events.indices],
                dtype=np.float64,
            ),
            np.asarray(
                state["velocities"][events.indices],
                dtype=np.float64,
            ),
            float(step_start_time_s),
            float(step_dt_s),
        )

    def _select_terminal_endpoint_kernel(self) -> None:
        limits = _boundary_limits(self)
        self.particle_pusher.select_terminal_boundary_hits(
            self.cloud,
            limits.detector_z_m,
            limits.detector_radius_m,
            limits.radial_limit_m,
            limits.domain_radius_m,
            limits.domain_length_m,
            float(max(self.config.electrode_hit_distance_m, 0.0)),
            int(1 if self.electrode_mask is not None else 0),
            int(PARTICLE_ACTIVE),
            int(PARTICLE_Z_EXIT),
            int(PARTICLE_RADIAL_OUT),
            int(PARTICLE_ELECTRODE_HIT),
            int(PARTICLE_DOMAIN_OUT),
        )

    def _download_terminal_endpoint_events(
        self,
        event_time_s: float,
    ) -> TerminalEvents:
        count = int(self.cloud.terminal_count.to_numpy().item())
        indices = np.asarray(
            self.cloud.terminal_indices.to_numpy()[:count],
            dtype=np.int32,
        )
        positions, velocities = self._download_selected_motion_state(indices)
        return TerminalEvents(
            indices=indices,
            statuses=np.asarray(
                self.cloud.terminal_status.to_numpy()[:count],
                dtype=np.int16,
            ),
            electrode_ids=np.asarray(
                self.cloud.terminal_electrode_id.to_numpy()[:count],
                dtype=np.int32,
            ),
            surface_distances_m=np.asarray(
                self.cloud.terminal_surface_distance.to_numpy()[:count],
                dtype=np.float64,
            ),
            positions_m=positions,
            velocities_m_per_s=velocities,
            event_times_s=np.full(count, float(event_time_s), dtype=np.float64),
        )

    def _commit_terminal_events(
        self,
        state: dict[str, np.ndarray],
        events: TerminalEvents,
    ) -> None:
        state["positions"][events.indices] = events.positions_m
        state["velocities"][events.indices] = events.velocities_m_per_s
        self._record_terminal_events(
            events.indices,
            events.statuses,
            event_time_s=events.event_times_s,
            positions_m=events.positions_m,
            velocities_m_per_s=events.velocities_m_per_s,
            temperatures_k=state["temperature"][events.indices],
            electrode_ids=events.electrode_ids,
            surface_distance_m=events.surface_distances_m,
        )
        TerminalBoundaryRuntimeMixin._commit_terminal_slot_metadata(
            self,
            state,
            events,
        )

    def _commit_terminal_slot_metadata(
        self,
        state: dict[str, np.ndarray],
        events: TerminalEvents,
    ) -> None:
        indices = events.indices
        self._particle_status_codes[indices] = events.statuses
        self._particle_tof_s[indices] = self._particle_tof_from_event_time(
            indices,
            events.event_times_s,
        )
        self._particle_phase_codes[indices] = SLOT_FREE
        electrode_hits = events.statuses == PARTICLE_ELECTRODE_HIT
        if np.any(electrode_hits):
            hit_indices = indices[electrode_hits]
            self._particle_electrode_id[hit_indices] = (
                events.electrode_ids[electrode_hits]
            )
            self._particle_surface_distance_m[hit_indices] = (
                events.surface_distances_m[electrode_hits]
            )
        state["active"][indices] = 0
        state["phase"][indices] = SLOT_FREE

    def _classify_domain_exit(self, position_m: np.ndarray) -> int:
        z_m = float(position_m[2])
        r_m = float(np.linalg.norm(position_m[:2]))
        detector_z_m = (
            self.config.detector_z_m or self.config.domain_length_m
        )
        detector_radius_m = (
            self.config.detector_radius_m
            or self.config.radial_limit_m
            or self.config.domain_radius_m
        )
        radial_limit_m = (
            self.config.radial_limit_m or self.config.domain_radius_m
        )
        if (
            r_m > radial_limit_m
            or (z_m >= detector_z_m and r_m > detector_radius_m)
        ):
            return PARTICLE_RADIAL_OUT
        if z_m >= detector_z_m:
            return PARTICLE_Z_EXIT
        if z_m < 0.0 or z_m > self.config.domain_length_m:
            return PARTICLE_DOMAIN_OUT
        if r_m > self.config.domain_radius_m:
            return PARTICLE_RADIAL_OUT
        return PARTICLE_DOMAIN_OUT
