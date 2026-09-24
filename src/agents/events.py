"""Entity-level terminal event payloads and weighted accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (
    PARTICLE_ACTIVE,
    PARTICLE_CAPILLARY_BUFFER,
    PARTICLE_DOMAIN_OUT,
    PARTICLE_ELECTRODE_HIT,
    PARTICLE_RADIAL_OUT,
    PARTICLE_STATUS_NAMES,
    PARTICLE_Z_EXIT,
)


TRANSPORT_TERMINAL_STATUS_CODES = {
    PARTICLE_Z_EXIT,
    PARTICLE_RADIAL_OUT,
    PARTICLE_ELECTRODE_HIT,
    PARTICLE_DOMAIN_OUT,
}


@dataclass(frozen=True)
class TerminalEventBatch:
    """Materialized per-particle data for one terminal-event emission."""

    particle_indices: np.ndarray
    status_codes: np.ndarray
    event_time_s: float | np.ndarray
    positions_m: np.ndarray
    velocities_m_per_s: np.ndarray
    temperatures_k: np.ndarray
    represented_real_ions: np.ndarray
    masses_kg: np.ndarray
    track_ids: np.ndarray
    birth_time_s: np.ndarray
    collision_counts: np.ndarray
    tof_s: np.ndarray
    electrode_ids: np.ndarray
    surface_distance_m: np.ndarray
    parent_real_ions_remaining: np.ndarray | None = None
    fragmented_real_ions_represented: np.ndarray | None = None


@dataclass
class TerminalAccounting:
    """Weighted cumulative terminal accounting keyed by particle status."""

    real_ions_by_code: dict[int, float]
    macro_particles_by_code: dict[int, int]
    parent_real_ions_by_code: dict[int, float]
    fragment_real_ions_by_code: dict[int, float]

    @classmethod
    def empty(cls) -> "TerminalAccounting":
        excluded = {PARTICLE_ACTIVE, PARTICLE_CAPILLARY_BUFFER}
        codes = [code for code in PARTICLE_STATUS_NAMES if code not in excluded]
        return cls(
            real_ions_by_code={code: 0.0 for code in codes},
            macro_particles_by_code={code: 0 for code in codes},
            parent_real_ions_by_code={code: 0.0 for code in codes},
            fragment_real_ions_by_code={code: 0.0 for code in codes},
        )

    def add_summary(
        self,
        status_code: int,
        macro_event_count: int,
        represented_real_ions: float,
        parent_real_ions_remaining: float | None = None,
        fragmented_real_ions_represented: float | None = None,
    ) -> None:
        macro_count = int(macro_event_count)
        if macro_count <= 0:
            return
        code = int(status_code)
        self.real_ions_by_code[code] = (
            self.real_ions_by_code.get(code, 0.0) + float(represented_real_ions)
        )
        self.macro_particles_by_code[code] = (
            self.macro_particles_by_code.get(code, 0) + macro_count
        )
        if parent_real_ions_remaining is not None:
            self.parent_real_ions_by_code[code] = (
                self.parent_real_ions_by_code.get(code, 0.0)
                + float(parent_real_ions_remaining)
            )
        if fragmented_real_ions_represented is not None:
            self.fragment_real_ions_by_code[code] = (
                self.fragment_real_ions_by_code.get(code, 0.0)
                + float(fragmented_real_ions_represented)
            )

    def add_batch(
        self,
        status_codes: np.ndarray,
        represented_real_ions: np.ndarray,
        parent_real_ions_remaining: np.ndarray,
        fragmented_real_ions_represented: np.ndarray,
    ) -> None:
        for code in np.unique(status_codes):
            code_int = int(code)
            if code_int == PARTICLE_ACTIVE:
                continue
            mask = status_codes == code_int
            self.add_summary(
                code_int,
                int(np.count_nonzero(mask)),
                float(np.sum(represented_real_ions[mask])),
                parent_real_ions_remaining=float(
                    np.sum(parent_real_ions_remaining[mask])
                ),
                fragmented_real_ions_represented=float(
                    np.sum(fragmented_real_ions_represented[mask])
                ),
            )
