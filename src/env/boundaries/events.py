"""Internal event batches exchanged by terminal-boundary runtime steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerminalEvents:
    """Columnar terminal events with one row per particle slot."""

    indices: np.ndarray
    statuses: np.ndarray
    electrode_ids: np.ndarray
    surface_distances_m: np.ndarray
    positions_m: np.ndarray
    velocities_m_per_s: np.ndarray
    event_times_s: np.ndarray

    @property
    def count(self) -> int:
        return int(self.indices.size)

    @classmethod
    def empty(cls) -> "TerminalEvents":
        return cls(
            indices=np.zeros(0, dtype=np.int32),
            statuses=np.zeros(0, dtype=np.int16),
            electrode_ids=np.zeros(0, dtype=np.int32),
            surface_distances_m=np.zeros(0, dtype=np.float64),
            positions_m=np.zeros((0, 3), dtype=np.float64),
            velocities_m_per_s=np.zeros((0, 3), dtype=np.float64),
            event_times_s=np.zeros(0, dtype=np.float64),
        )

    def take(self, offsets: np.ndarray) -> "TerminalEvents":
        return TerminalEvents(
            indices=self.indices[offsets],
            statuses=self.statuses[offsets],
            electrode_ids=self.electrode_ids[offsets],
            surface_distances_m=self.surface_distances_m[offsets],
            positions_m=self.positions_m[offsets],
            velocities_m_per_s=self.velocities_m_per_s[offsets],
            event_times_s=self.event_times_s[offsets],
        )

    def ordered(self) -> "TerminalEvents":
        """Order by event time and then particle index."""

        if self.count == 0:
            return self
        return self.take(np.lexsort((self.indices, self.event_times_s)))


def concatenate_terminal_events(
    first: TerminalEvents,
    second: TerminalEvents,
) -> TerminalEvents:
    """Concatenate swept events before endpoint events, as in the legacy flow."""

    return TerminalEvents(
        indices=np.concatenate((first.indices, second.indices)),
        statuses=np.concatenate((first.statuses, second.statuses)),
        electrode_ids=np.concatenate((first.electrode_ids, second.electrode_ids)),
        surface_distances_m=np.concatenate(
            (first.surface_distances_m, second.surface_distances_m)
        ),
        positions_m=np.concatenate(
            (first.positions_m, second.positions_m),
            axis=0,
        ),
        velocities_m_per_s=np.concatenate(
            (first.velocities_m_per_s, second.velocities_m_per_s),
            axis=0,
        ),
        event_times_s=np.concatenate((first.event_times_s, second.event_times_s)),
    )
