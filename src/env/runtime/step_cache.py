"""Host caches valid only within one microstep and one motion stage."""

from __future__ import annotations

from typing import Callable

import numpy as np


class ParticleStepCache:
    """Keep slot order stable; cached motion never aliases mutable host state."""

    def __init__(self, state: dict[str, np.ndarray], *, motion_enabled: bool = True) -> None:
        self.state = state
        self.indices = np.flatnonzero(state["active"] != 0).astype(np.int32)
        self.motion_enabled = motion_enabled
        self.positions_m = self.velocities_m_per_s = self.valid = None

    def invalidate_motion(self) -> None:
        """Call after a velocity upload or RK4, before any subsequent read."""
        if self.valid is not None:
            self.valid.fill(False)

    def read(
        self, indices: np.ndarray,
        gather: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.motion_enabled:
            return gather(indices)
        if self.valid is None:
            self.positions_m = np.empty((self.indices.size, 3), dtype=np.float64)
            self.velocities_m_per_s = np.empty_like(self.positions_m)
            self.valid = np.zeros(self.indices.size, dtype=bool)
        if indices is self.indices or np.array_equal(indices, self.indices):
            return self._read_all(gather)
        offsets = np.searchsorted(self.indices, indices)
        if (np.any(offsets >= self.indices.size)
                or np.any(self.indices[offsets] != indices)):
            return gather(indices)
        missing = ~self.valid[offsets]
        if np.any(missing):
            positions, velocities = gather(indices[missing])
            self.positions_m[offsets[missing]] = positions
            self.velocities_m_per_s[offsets[missing]] = velocities
            self.valid[offsets[missing]] = True
        return self.positions_m[offsets], self.velocities_m_per_s[offsets]

    def _read_all(
        self, gather: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Avoid repeated index searches for full active-population reads."""
        missing = ~self.valid
        if np.any(missing):
            positions, velocities = gather(self.indices[missing])
            self.positions_m[missing] = positions
            self.velocities_m_per_s[missing] = velocities
            self.valid[missing] = True
        return self.positions_m.copy(), self.velocities_m_per_s.copy()


def invalidate_particle_cache(owner: object) -> None:
    """Discard active-slot membership after release, activation or injection."""
    owner._particle_step_cache = None
