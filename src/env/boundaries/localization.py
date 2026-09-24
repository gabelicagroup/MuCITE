"""Continuous localization of the earliest terminal segment event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ...config import (
    PARTICLE_DOMAIN_OUT,
    PARTICLE_ELECTRODE_HIT,
    PARTICLE_RADIAL_OUT,
    PARTICLE_Z_EXIT,
)
from ...utils.geometry import (
    interpolate_segment_state,
    locate_first_segment_hit,
    segment_cylinder_exit_fraction,
    segment_plane_crossing_fraction,
)
from .electrodes import ElectrodeSamplingMixin


TerminalCandidate = tuple[float, int, int]
LocalizedTerminalTuple = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


@dataclass(frozen=True)
class BoundaryLimits:
    detector_z_m: float
    detector_radius_m: float
    radial_limit_m: float
    domain_radius_m: float
    domain_length_m: float


@dataclass
class TerminalSegments:
    starts_m: np.ndarray
    ends_m: np.ndarray
    start_velocities_m_per_s: np.ndarray
    end_velocities_m_per_s: np.ndarray
    statuses: np.ndarray
    electrode_ids: np.ndarray
    surface_distances_m: np.ndarray


def _boundary_limits(owner: Any) -> BoundaryLimits:
    config = owner.config
    detector_z_m = (
        config.detector_z_m
        if config.detector_z_m is not None
        else config.domain_length_m
    )
    detector_radius_m = (
        config.detector_radius_m
        if config.detector_radius_m is not None
        else (
            config.radial_limit_m
            if config.radial_limit_m is not None
            else config.domain_radius_m
        )
    )
    radial_limit_m = (
        config.radial_limit_m
        if config.radial_limit_m is not None
        else config.domain_radius_m
    )
    return BoundaryLimits(
        detector_z_m=float(detector_z_m),
        detector_radius_m=float(detector_radius_m),
        radial_limit_m=float(radial_limit_m),
        domain_radius_m=float(config.domain_radius_m),
        domain_length_m=float(config.domain_length_m),
    )


def _terminal_segments(
    start_positions_m: np.ndarray,
    end_positions_m: np.ndarray,
    start_velocities_m_per_s: np.ndarray,
    end_velocities_m_per_s: np.ndarray,
    endpoint_status: np.ndarray,
    endpoint_electrode_id: np.ndarray,
    endpoint_surface_distance_m: np.ndarray,
) -> TerminalSegments:
    starts = np.asarray(start_positions_m, dtype=np.float64)
    ends = np.asarray(end_positions_m, dtype=np.float64)
    start_velocities = np.asarray(
        start_velocities_m_per_s,
        dtype=np.float64,
    )
    end_velocities = np.asarray(end_velocities_m_per_s, dtype=np.float64)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("Terminal segment positions must have shape (n, 3).")
    if (
        start_velocities.shape != starts.shape
        or end_velocities.shape != starts.shape
    ):
        raise ValueError(
            "Terminal segment velocities must match position shape."
        )
    return TerminalSegments(
        starts_m=starts,
        ends_m=ends,
        start_velocities_m_per_s=start_velocities,
        end_velocities_m_per_s=end_velocities,
        statuses=np.asarray(endpoint_status, dtype=np.int16).copy(),
        electrode_ids=np.asarray(endpoint_electrode_id, dtype=np.int32).copy(),
        surface_distances_m=np.asarray(
            endpoint_surface_distance_m,
            dtype=np.float64,
        ).copy(),
    )


def _electrode_candidate(
    owner: Any,
    start: np.ndarray,
    end: np.ndarray,
    sample_spacing_m: Optional[float],
) -> Optional[TerminalCandidate]:
    if sample_spacing_m is None:
        return None
    fraction = locate_first_segment_hit(
        start,
        end,
        lambda position: owner._sample_electrode_event(position)[0],
        max_sample_spacing_m=sample_spacing_m,
        bisection_iterations=int(
            getattr(
                owner.config,
                "electrode_hit_bisection_iterations",
                24,
            )
        ),
    )
    if fraction is None:
        return None
    return fraction, 0, int(PARTICLE_ELECTRODE_HIT)


def _detector_candidate(
    start: np.ndarray,
    end: np.ndarray,
    limits: BoundaryLimits,
) -> Optional[TerminalCandidate]:
    fraction = segment_plane_crossing_fraction(
        start,
        end,
        limits.detector_z_m,
        direction=1,
    )
    if fraction is None:
        return None
    position = start + fraction * (end - start)
    radius_m = float(np.linalg.norm(position[:2]))
    status = (
        PARTICLE_Z_EXIT
        if radius_m <= limits.detector_radius_m
        else PARTICLE_RADIAL_OUT
    )
    priority = 2 if status == PARTICLE_Z_EXIT else 1
    return fraction, priority, int(status)


def _radial_candidates(
    start: np.ndarray,
    end: np.ndarray,
    limits: BoundaryLimits,
) -> list[TerminalCandidate]:
    candidates: list[TerminalCandidate] = []
    for radius_m in {limits.radial_limit_m, limits.domain_radius_m}:
        fraction = segment_cylinder_exit_fraction(start, end, radius_m)
        if fraction is not None:
            candidates.append(
                (fraction, 1, int(PARTICLE_RADIAL_OUT))
            )
    return candidates


def _domain_candidates(
    start: np.ndarray,
    end: np.ndarray,
    limits: BoundaryLimits,
) -> list[TerminalCandidate]:
    candidates: list[TerminalCandidate] = []
    upstream = segment_plane_crossing_fraction(
        start,
        end,
        0.0,
        direction=-1,
    )
    if upstream is not None:
        candidates.append((upstream, 3, int(PARTICLE_DOMAIN_OUT)))
    downstream = segment_plane_crossing_fraction(
        start,
        end,
        limits.domain_length_m,
        direction=1,
    )
    if downstream is not None:
        candidates.append((downstream, 3, int(PARTICLE_DOMAIN_OUT)))
    return candidates


def _terminal_candidates(
    owner: Any,
    start: np.ndarray,
    end: np.ndarray,
    limits: BoundaryLimits,
    electrode_spacing_m: Optional[float],
) -> list[TerminalCandidate]:
    candidates: list[TerminalCandidate] = []
    electrode = _electrode_candidate(
        owner,
        start,
        end,
        electrode_spacing_m,
    )
    if electrode is not None:
        candidates.append(electrode)
    detector = _detector_candidate(start, end, limits)
    if detector is not None:
        candidates.append(detector)
    candidates.extend(_radial_candidates(start, end, limits))
    candidates.extend(_domain_candidates(start, end, limits))
    return candidates


def _localized_electrode_values(
    owner: Any,
    status: int,
    position_m: np.ndarray,
    electrode_id: int,
    surface_distance_m: float,
) -> tuple[int, float]:
    if status != PARTICLE_ELECTRODE_HIT:
        return -1, float("nan")
    hit, sampled_id, sampled_distance_m = owner._sample_electrode_event(
        position_m
    )
    if hit:
        return sampled_id, sampled_distance_m
    return electrode_id, surface_distance_m


def _localize_one_segment(
    owner: Any,
    segments: TerminalSegments,
    offset: int,
    limits: BoundaryLimits,
    electrode_spacing_m: Optional[float],
    step_start_time_s: float,
    step_dt_s: float,
) -> tuple[int, np.ndarray, np.ndarray, float, int, float]:
    start = segments.starts_m[offset]
    end = segments.ends_m[offset]
    candidates = _terminal_candidates(
        owner,
        start,
        end,
        limits,
        electrode_spacing_m,
    )
    status = int(segments.statuses[offset])
    fraction = 1.0
    if candidates:
        fraction, _priority, status = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
    position, velocity = interpolate_segment_state(
        start,
        end,
        segments.start_velocities_m_per_s[offset],
        segments.end_velocities_m_per_s[offset],
        fraction,
    )
    event_time_s = float(step_start_time_s) + float(fraction) * float(step_dt_s)
    electrode_id = int(segments.electrode_ids[offset])
    surface_distance_m = float(segments.surface_distances_m[offset])
    electrode_id, surface_distance_m = _localized_electrode_values(
        owner, status, position, electrode_id, surface_distance_m,
    )
    return (
        status,
        position,
        velocity,
        event_time_s,
        electrode_id,
        surface_distance_m,
    )


class TerminalLocalizationMixin(ElectrodeSamplingMixin):
    """Continuous earliest-hit localization for terminal segments."""

    def _localize_terminal_segments(
        self,
        *,
        start_positions_m: np.ndarray,
        end_positions_m: np.ndarray,
        start_velocities_m_per_s: np.ndarray,
        end_velocities_m_per_s: np.ndarray,
        endpoint_status: np.ndarray,
        endpoint_electrode_id: np.ndarray,
        endpoint_surface_distance_m: np.ndarray,
        step_start_time_s: float,
        step_dt_s: float,
    ) -> LocalizedTerminalTuple:
        """Locate the earliest swept terminal event on each segment."""

        segments = _terminal_segments(
            start_positions_m, end_positions_m,
            start_velocities_m_per_s, end_velocities_m_per_s,
            endpoint_status, endpoint_electrode_id,
            endpoint_surface_distance_m,
        )
        positions = segments.ends_m.copy()
        velocities = segments.end_velocities_m_per_s.copy()
        event_times_s = np.full(
            segments.starts_m.shape[0],
            float(step_start_time_s) + float(step_dt_s),
            dtype=np.float64,
        )
        limits = _boundary_limits(self)
        spacing_m = self._electrode_segment_sample_spacing_m()
        for offset in range(segments.starts_m.shape[0]):
            localized = _localize_one_segment(
                self, segments, offset, limits, spacing_m,
                step_start_time_s, step_dt_s,
            )
            segments.statuses[offset] = localized[0]
            positions[offset], velocities[offset] = localized[1:3]
            event_times_s[offset] = localized[3]
            segments.electrode_ids[offset] = localized[4]
            segments.surface_distances_m[offset] = localized[5]
        return (
            segments.statuses,
            positions,
            velocities,
            event_times_s,
            segments.electrode_ids,
            segments.surface_distances_m,
        )
