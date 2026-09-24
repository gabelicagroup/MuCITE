"""Terminal time-of-flight, weighted accounting, and recorder adaptation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...agents.events import TerminalEventBatch
from ...config import PARTICLE_FRAGMENTED


def _terminal_motion(
    owner: Any,
    particle_indices: np.ndarray,
    positions_m: Optional[np.ndarray],
    velocities_m_per_s: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if positions_m is None or velocities_m_per_s is None:
        downloaded_positions, downloaded_velocities = (
            owner._download_selected_motion_state(particle_indices)
        )
        positions_m = downloaded_positions if positions_m is None else positions_m
        velocities_m_per_s = (
            downloaded_velocities
            if velocities_m_per_s is None
            else velocities_m_per_s
        )
    positions = np.asarray(positions_m, dtype=np.float64)
    velocities = np.asarray(velocities_m_per_s, dtype=np.float64)
    expected_shape = (particle_indices.size, 3)
    if positions.shape != expected_shape:
        raise ValueError("positions_m must have shape (n_particles, 3).")
    if velocities.shape != expected_shape:
        raise ValueError("velocities_m_per_s must have shape (n_particles, 3).")
    return positions, velocities


def _length_checked(
    name: str,
    values: np.ndarray,
    particle_count: int,
    dtype: np.dtype,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.shape[0] != particle_count:
        raise ValueError(f"{name} must have length n_particles.")
    return result


def _terminal_weights(
    owner: Any,
    particle_indices: np.ndarray,
    status_codes: np.ndarray,
    represented_real_ions: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slot_weights = np.asarray(
        owner._particle_weight[particle_indices],
        dtype=np.float64,
    )
    weights = (
        slot_weights
        if represented_real_ions is None
        else _length_checked(
            "represented_real_ions",
            represented_real_ions,
            particle_indices.size,
            np.float64,
        )
    )
    parent_source = getattr(
        owner,
        "_particle_parent_weight",
        owner._particle_weight,
    )
    slot_parent_weights = np.asarray(
        parent_source[particle_indices],
        dtype=np.float64,
    )
    parent_weights = _scaled_parent_weights(
        slot_weights,
        slot_parent_weights,
        weights,
        represented_real_ions is not None,
    )
    if represented_real_ions is not None:
        parent_weights[status_codes == PARTICLE_FRAGMENTED] = 0.0
    return weights, parent_weights, np.maximum(weights - parent_weights, 0.0)


def _scaled_parent_weights(
    slot_weights: np.ndarray,
    slot_parent_weights: np.ndarray,
    weights: np.ndarray,
    rescale: bool,
) -> np.ndarray:
    parent_weights = slot_parent_weights
    if rescale:
        scale = np.zeros_like(weights, dtype=np.float64)
        valid_slot_weight = slot_weights > 0.0
        scale[valid_slot_weight] = (
            weights[valid_slot_weight] / slot_weights[valid_slot_weight]
        )
        parent_weights = slot_parent_weights * scale
    return np.minimum(
        np.maximum(parent_weights, 0.0),
        np.maximum(weights, 0.0),
    )


def _terminal_track_ids(
    owner: Any,
    particle_indices: np.ndarray,
    track_ids: Optional[np.ndarray],
) -> np.ndarray:
    values = (
        owner._particle_track_id[particle_indices]
        if track_ids is None
        else track_ids
    )
    return _length_checked(
        "track_ids",
        values,
        particle_indices.size,
        np.int64,
    )


def _terminal_electrode_values(
    owner: Any,
    particle_indices: np.ndarray,
    electrode_ids: Optional[np.ndarray],
    surface_distance_m: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    ids = (
        owner._particle_electrode_id[particle_indices]
        if electrode_ids is None
        else electrode_ids
    )
    distances = (
        owner._particle_surface_distance_m[particle_indices]
        if surface_distance_m is None
        else surface_distance_m
    )
    return (
        _length_checked(
            "electrode_ids",
            ids,
            particle_indices.size,
            np.int32,
        ),
        _length_checked(
            "surface_distance_m",
            distances,
            particle_indices.size,
            np.float64,
        ),
    )


def _build_terminal_batch(
    owner: Any,
    particle_indices: np.ndarray,
    status_codes: np.ndarray,
    event_time_s: float | np.ndarray,
    positions_m: np.ndarray,
    velocities_m_per_s: np.ndarray,
    temperatures_k: np.ndarray,
    weights: np.ndarray,
    parent_weights: np.ndarray,
    fragment_weights: np.ndarray,
    track_ids: np.ndarray,
    electrode_ids: np.ndarray,
    surface_distance_m: np.ndarray,
) -> TerminalEventBatch:
    return TerminalEventBatch(
        particle_indices=particle_indices,
        status_codes=status_codes,
        event_time_s=np.asarray(event_time_s, dtype=np.float64),
        positions_m=positions_m,
        velocities_m_per_s=velocities_m_per_s,
        temperatures_k=temperatures_k,
        represented_real_ions=weights,
        parent_real_ions_remaining=parent_weights,
        fragmented_real_ions_represented=fragment_weights,
        masses_kg=np.asarray(
            owner._particle_mass_kg[particle_indices],
            dtype=np.float64,
        ),
        track_ids=track_ids,
        birth_time_s=np.asarray(
            owner._particle_birth_time_s[particle_indices],
            dtype=np.float64,
        ),
        collision_counts=np.asarray(
            owner._particle_collision_counts[particle_indices],
            dtype=np.int64,
        ),
        tof_s=owner._particle_tof_from_event_time(
            particle_indices,
            event_time_s,
        ),
        electrode_ids=electrode_ids,
        surface_distance_m=surface_distance_m,
    )


def _submit_terminal_batch(owner: Any, batch: TerminalEventBatch) -> None:
    owner.terminal_event_recorder.record(
        batch,
        source_mode=owner.config.source_mode,
        source_terminal_accounting=owner._source_terminal_accounting,
        terminal_event_mode=owner.config.terminal_event_mode,
        max_terminal_event_rows=owner.config.max_terminal_event_rows,
    )


class TerminalAccountingMixin:
    """Terminal accounting behavior exposed through the simulation runtime."""

    def _particle_tof_from_event_time(
        self,
        particle_indices: np.ndarray,
        event_time_s: float | np.ndarray,
    ) -> np.ndarray:
        particle_indices = np.asarray(particle_indices, dtype=np.int32)
        if particle_indices.size == 0:
            return np.zeros(0, dtype=np.float64)
        event_time_array = np.asarray(event_time_s, dtype=np.float64)
        if event_time_array.ndim == 0:
            event_times_s = np.full(
                particle_indices.size,
                float(event_time_array),
                dtype=np.float64,
            )
        elif event_time_array.shape == (particle_indices.size,):
            event_times_s = event_time_array
        else:
            raise ValueError(
                "event_time_s must be scalar or have length n_particles."
            )
        if not np.all(np.isfinite(event_times_s)):
            raise ValueError("event_time_s contains non-finite values.")
        birth_time_s = np.asarray(
            self._particle_birth_time_s[particle_indices],
            dtype=np.float64,
        )
        tof_s = event_times_s.copy()
        valid_birth = np.isfinite(birth_time_s)
        tof_s[valid_birth] = np.maximum(
            event_times_s[valid_birth] - birth_time_s[valid_birth],
            0.0,
        )
        return tof_s

    def _accumulate_terminal_accounting(
        self,
        status_code: int,
        macro_event_count: int,
        represented_real_ions: float,
        parent_real_ions_remaining: Optional[float] = None,
        fragmented_real_ions_represented: Optional[float] = None,
    ) -> None:
        """Accumulate weighted terminal summaries when rows are not emitted."""

        if self.config.source_mode != "continuous-current":
            return
        self._source_terminal_accounting.add_summary(
            status_code,
            macro_event_count,
            represented_real_ions,
            parent_real_ions_remaining=parent_real_ions_remaining,
            fragmented_real_ions_represented=fragmented_real_ions_represented,
        )

    def _record_terminal_events(
        self,
        particle_indices: np.ndarray,
        status_codes: np.ndarray,
        *,
        event_time_s: float | np.ndarray,
        positions_m: Optional[np.ndarray] = None,
        velocities_m_per_s: Optional[np.ndarray] = None,
        temperatures_k: Optional[np.ndarray] = None,
        electrode_ids: Optional[np.ndarray] = None,
        surface_distance_m: Optional[np.ndarray] = None,
        represented_real_ions: Optional[np.ndarray] = None,
        track_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Normalize runtime slot state into a terminal-event batch."""

        indices = np.asarray(particle_indices, dtype=np.int32)
        statuses = np.asarray(status_codes, dtype=np.int16)
        if indices.size == 0:
            return
        positions, velocities = _terminal_motion(
            self,
            indices,
            positions_m,
            velocities_m_per_s,
        )
        temperatures = _length_checked(
            "temperatures_k",
            np.full(indices.size, np.nan) if temperatures_k is None else temperatures_k,
            indices.size,
            np.float64,
        )
        weights, parents, fragments = _terminal_weights(
            self,
            indices,
            statuses,
            represented_real_ions,
        )
        ids, distances = _terminal_electrode_values(
            self,
            indices,
            electrode_ids,
            surface_distance_m,
        )
        batch = _build_terminal_batch(
            self, indices, statuses, event_time_s, positions, velocities,
            temperatures, weights, parents, fragments,
            _terminal_track_ids(self, indices, track_ids), ids, distances,
        )
        _submit_terminal_batch(self, batch)
