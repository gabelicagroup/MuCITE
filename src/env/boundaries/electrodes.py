"""Electrode occupancy sampling and swept thin-solid detection."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .mask2d import sample_electrode_mask
from .mask3d import sample_electrode_mask3d
from ...utils.geometry import locate_first_segment_hit


def _empty_electrode_samples(
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(count, dtype=bool),
        np.full(count, -1, dtype=np.int32),
        np.full(count, np.nan, dtype=np.float64),
    )


def _mask_hits(
    metal: np.ndarray,
    surface_distance_m: np.ndarray,
    inside: np.ndarray,
    hit_distance_m: float,
    available: np.ndarray,
) -> np.ndarray:
    return available & inside & (
        metal
        | (
            np.isfinite(surface_distance_m)
            & (surface_distance_m <= hit_distance_m)
        )
    )


def _apply_mask_samples(
    samples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    hit_distance_m: float,
    available: np.ndarray,
    hits: np.ndarray,
    electrode_ids: np.ndarray,
    surface_distances_m: np.ndarray,
) -> None:
    metal, sampled_ids, sampled_distances_m, inside = samples
    selected = _mask_hits(
        metal,
        sampled_distances_m,
        inside,
        hit_distance_m,
        available,
    )
    hits[selected] = True
    electrode_ids[selected] = sampled_ids[selected]
    surface_distances_m[selected] = sampled_distances_m[selected]


def _validated_segments(
    start_positions_m: np.ndarray,
    end_positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(start_positions_m, dtype=np.float64)
    ends = np.asarray(end_positions_m, dtype=np.float64)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("Swept electrode segments must have shape (n, 3).")
    return starts, ends


def _segment_sample_counts(
    starts: np.ndarray,
    ends: np.ndarray,
    sample_spacing_m: float,
) -> np.ndarray:
    lengths_m = np.linalg.norm(ends - starts, axis=1)
    return np.maximum(
        1,
        np.ceil(lengths_m / sample_spacing_m).astype(np.int32),
    )


class ElectrodeSamplingMixin:
    """CPU-side 3D-first electrode sampling and swept-entry location."""

    def _sample_electrode_events(
        self,
        positions_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_m must have shape (n, 3).")
        hits, electrode_ids, distances_m = _empty_electrode_samples(
            positions.shape[0]
        )
        hit_distance_m = float(
            max(self.config.electrode_hit_distance_m, 0.0)
        )
        if self.electrode_mask3d is not None:
            _apply_mask_samples(
                sample_electrode_mask3d(self.electrode_mask3d, positions),
                hit_distance_m,
                np.ones(positions.shape[0], dtype=bool),
                hits,
                electrode_ids,
                distances_m,
            )
        if self.electrode_mask is not None:
            _apply_mask_samples(
                sample_electrode_mask(self.electrode_mask, positions),
                hit_distance_m,
                ~hits,
                hits,
                electrode_ids,
                distances_m,
            )
        return hits, electrode_ids, distances_m

    def _sample_electrode_event(
        self,
        position_m: np.ndarray,
    ) -> tuple[bool, int, float]:
        hits, electrode_ids, surface_distances_m = (
            self._sample_electrode_events(
                np.asarray(position_m, dtype=np.float64).reshape(1, 3)
            )
        )
        return (
            bool(hits[0]),
            int(electrode_ids[0]),
            float(surface_distances_m[0]),
        )

    def _electrode_segment_sample_spacing_m(self) -> Optional[float]:
        spacings_m: list[float] = []
        if self.electrode_mask is not None:
            spacings_m.extend(
                [
                    float(np.median(np.diff(self.electrode_mask.r_coords_m))),
                    float(np.median(np.diff(self.electrode_mask.z_coords_m))),
                ]
            )
        if self.electrode_mask3d is not None:
            spacings_m.extend(
                [
                    float(np.median(np.diff(self.electrode_mask3d.x_coords_m))),
                    float(np.median(np.diff(self.electrode_mask3d.y_coords_m))),
                    float(np.median(np.diff(self.electrode_mask3d.z_coords_m))),
                ]
            )
        valid_spacings_m = [
            value
            for value in spacings_m
            if np.isfinite(value) and value > 0.0
        ]
        if not valid_spacings_m:
            return None
        return (
            float(
                getattr(
                    self.config,
                    "electrode_sweep_spacing_fraction",
                    0.25,
                )
            )
            * min(valid_spacings_m)
        )

    def _find_swept_electrode_hits(
        self,
        start_positions_m: np.ndarray,
        end_positions_m: np.ndarray,
    ) -> np.ndarray:
        """Return the first sampled/bisected electrode-entry fraction."""

        starts, ends = _validated_segments(
            start_positions_m,
            end_positions_m,
        )
        fractions = np.full(starts.shape[0], np.nan, dtype=np.float64)
        if starts.shape[0] == 0:
            return fractions
        sample_spacing_m = self._electrode_segment_sample_spacing_m()
        if sample_spacing_m is None:
            return fractions
        start_hits, _ids, _distances = self._sample_electrode_events(starts)
        fractions[start_hits] = 0.0
        self._scan_swept_samples(
            starts,
            ends,
            _segment_sample_counts(starts, ends, sample_spacing_m),
            sample_spacing_m,
            ~start_hits,
            fractions,
        )
        return fractions

    def _scan_swept_samples(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        sample_counts: np.ndarray,
        sample_spacing_m: float,
        unresolved: np.ndarray,
        fractions: np.ndarray,
    ) -> None:
        delta = ends - starts
        for sample_index in range(1, int(np.max(sample_counts)) + 1):
            offsets = np.flatnonzero(
                unresolved & (sample_counts >= sample_index)
            )
            if offsets.size == 0:
                continue
            upper_fractions = np.minimum(
                sample_index / sample_counts[offsets].astype(np.float64),
                1.0,
            )
            sample_positions = (
                starts[offsets]
                + upper_fractions[:, None] * delta[offsets]
            )
            sample_hits, _ids, _distances = self._sample_electrode_events(
                sample_positions
            )
            self._refine_swept_hits(
                starts, delta, sample_counts, sample_index, offsets,
                upper_fractions, sample_hits, sample_spacing_m,
                unresolved, fractions,
            )

    def _refine_swept_hits(
        self,
        starts: np.ndarray,
        delta: np.ndarray,
        sample_counts: np.ndarray,
        sample_index: int,
        offsets: np.ndarray,
        upper_fractions: np.ndarray,
        sample_hits: np.ndarray,
        sample_spacing_m: float,
        unresolved: np.ndarray,
        fractions: np.ndarray,
    ) -> None:
        for local_offset in np.flatnonzero(sample_hits):
            particle_offset = int(offsets[local_offset])
            upper_fraction = float(upper_fractions[local_offset])
            lower_fraction = float(
                (sample_index - 1) / sample_counts[particle_offset]
            )
            lower_position = (
                starts[particle_offset]
                + lower_fraction * delta[particle_offset]
            )
            upper_position = (
                starts[particle_offset]
                + upper_fraction * delta[particle_offset]
            )
            local_fraction = locate_first_segment_hit(
                lower_position,
                upper_position,
                lambda position: self._sample_electrode_event(position)[0],
                max_sample_spacing_m=sample_spacing_m,
                bisection_iterations=int(
                    getattr(
                        self.config,
                        "electrode_hit_bisection_iterations",
                        24,
                    )
                ),
            )
            fractions[particle_offset] = (
                upper_fraction
                if local_fraction is None
                else lower_fraction
                + local_fraction * (upper_fraction - lower_fraction)
            )
            unresolved[particle_offset] = False
