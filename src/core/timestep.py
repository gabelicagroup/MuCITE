"""Backend-independent time-step validation and selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InvalidTimeStepError(RuntimeError):
    """Raised when an active particle has an unusable time-step candidate."""

    def __init__(
        self,
        particle_indices: np.ndarray,
        candidates_s: np.ndarray,
    ) -> None:
        self.particle_indices = np.asarray(particle_indices, dtype=np.int64)
        self.candidates_s = np.asarray(candidates_s, dtype=np.float64)
        pairs = ", ".join(
            f"{int(index)}={float(value)!r}"
            for index, value in zip(
                self.particle_indices[:8],
                self.candidates_s[:8],
            )
        )
        if self.particle_indices.size > 8:
            pairs += ", ..."
        super().__init__(
            f"Invalid dt candidate(s) for active particle slot(s): {pairs}"
        )


@dataclass(frozen=True)
class SelectedTimeStep:
    dt_s: float
    particle_index: int


def select_active_timestep(
    dt_candidates_s: np.ndarray,
    active: np.ndarray,
    *,
    inactive_sentinel_s: float = 1.0e90,
) -> SelectedTimeStep:
    """Validate every active candidate, then return the global minimum."""

    candidates = np.asarray(dt_candidates_s, dtype=np.float64)
    active_mask = np.asarray(active, dtype=bool)
    if (
        candidates.ndim != 1
        or active_mask.ndim != 1
        or candidates.shape != active_mask.shape
    ):
        raise ValueError(
            "dt_candidates_s and active must be same-length 1D arrays."
        )
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        raise ValueError("Cannot select a time step without active particles.")
    active_candidates = candidates[active_indices]
    valid = (
        np.isfinite(active_candidates)
        & (active_candidates > 0.0)
        & (active_candidates < float(inactive_sentinel_s))
    )
    if not np.all(valid):
        raise InvalidTimeStepError(
            active_indices[~valid],
            active_candidates[~valid],
        )
    local_index = int(np.argmin(active_candidates))
    return SelectedTimeStep(
        dt_s=float(active_candidates[local_index]),
        particle_index=int(active_indices[local_index]),
    )


def velocity_cfl_timestep(
    velocity_m_per_s: np.ndarray,
    cell_size_m: float,
    *,
    safety_factor: float = 0.5,
) -> np.ndarray:
    """Return ``C * cell_size / |v|``; stationary particles get infinity."""

    velocity = np.asarray(velocity_m_per_s, dtype=np.float64)
    if velocity.shape == () or velocity.shape[-1] != 3:
        raise ValueError(
            "velocity_m_per_s must end in a Cartesian axis of length 3."
        )
    if not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_m_per_s contains non-finite values.")
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive.")
    if not np.isfinite(safety_factor) or not 0.0 < safety_factor <= 1.0:
        raise ValueError("safety_factor must satisfy 0 < C <= 1.")
    speed_m_per_s = np.linalg.norm(velocity, axis=-1)
    dt_s = np.full(speed_m_per_s.shape, np.inf, dtype=np.float64)
    moving = speed_m_per_s > 0.0
    dt_s[moving] = (
        safety_factor * float(cell_size_m) / speed_m_per_s[moving]
    )
    return dt_s
