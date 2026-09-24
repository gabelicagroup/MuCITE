"""Continuous-current source lifecycle for the simulation model."""

from __future__ import annotations

import numpy as np

from ...agents.slot_state import ParticleRuntimeState
from ...config import SLOT_CAPILLARY, SLOT_FREE


class ContinuousSourceRuntimeMixin:
    """Manage continuous source reservoirs, injection, and capillary transit."""

    def _advance_capillary_source(
        self,
        state: dict[str, np.ndarray],
        dt_s: float,
        event_time_s: float,
    ) -> np.ndarray:
        capillary = np.flatnonzero(
            np.asarray(state["phase"], dtype=np.int16) == SLOT_CAPILLARY
        )
        if capillary.size == 0 or dt_s <= 0.0:
            return np.zeros(0, dtype=np.int32)
        capillary = np.asarray(capillary, dtype=np.int32)
        state["positions"][capillary, 2] += (
            state["velocities"][capillary, 2] * float(dt_s)
        )
        emitted_mask = (
            state["positions"][capillary, 2] >= self._capillary_exit_z_m()
        )
        if not np.any(emitted_mask):
            return np.zeros(0, dtype=np.int32)
        emitted = capillary[emitted_mask]
        self._activate_external_slots(
            state,
            emitted,
            event_time_s=event_time_s,
        )
        weights = np.asarray(state["weight"][emitted], dtype=np.float64)
        self._source_emitted_real_ions += float(np.sum(weights))
        self._source_emitted_macro_particles += int(emitted.size)
        return np.asarray(emitted, dtype=np.int32)

    def _initialize_continuous_current_source(self) -> None:
        count = self.config.ion_count
        self._prepare_continuous_slots(count)
        state = self._empty_continuous_state(count)
        if self._continuous_source_enabled():
            self._prefill_continuous_state(state)
        self._upload_cloud_state(state)

    def _prepare_continuous_slots(self, count: int) -> None:
        ParticleRuntimeState.continuous(
            count,
            self.template,
        ).bind_to(self)
        self.terminal_event_recorder.clear()
        self._terminal_event_rows = self.terminal_event_recorder.rows
        self._reset_source_counters()
        self._reset_terminal_accounting()
        self._reset_collision_statistics()

    def _reset_source_counters(self) -> None:
        self._source_real_ion_accumulator = 0.0
        self._source_prefill_real_ions = 0.0
        self._source_prefill_macro_particles = 0
        self._source_boundary_injected_real_ions = 0.0
        self._source_boundary_injected_macro_particles = 0
        self._source_emitted_real_ions = 0.0
        self._source_emitted_macro_particles = 0
        self._source_injected_real_ions = 0.0
        self._source_injected_macro_particles = 0
        self._source_blocked_real_ions = 0.0

    def _empty_continuous_state(
        self,
        count: int,
    ) -> dict[str, np.ndarray]:
        return {
            "positions": np.zeros((count, 3), dtype=np.float64),
            "velocities": np.zeros((count, 3), dtype=np.float64),
            "charge": self._particle_charge_c,
            "mass": self._particle_mass_kg,
            "weight": self._particle_weight,
            "active": np.zeros(count, dtype=np.int32),
            "temperature": np.full(
                count,
                self.template.initial_internal_temperature_k,
                dtype=np.float64,
            ),
            "phase": np.full(count, SLOT_FREE, dtype=np.int16),
        }

    def _prefill_continuous_state(
        self,
        state: dict[str, np.ndarray],
    ) -> None:
        length_m = max(
            float(self.config.capillary_prefill_length_m),
            0.0,
        )
        if length_m <= 0.0:
            return
        axial_speed = max(
            self._source_axial_velocity_m_per_s(),
            1.0e-30,
        )
        real_ions = (
            float(self.config.ion_current_a)
            * (length_m / axial_speed)
            / self._source_charge_per_ion_c()
        )
        macro_count = self._prefill_macro_count(real_ions)
        indices = np.arange(macro_count, dtype=np.int32)
        positions, _ = self._sample_capillary_phase_space(
            macro_count,
            prefill=True,
        )
        weights = np.full(
            macro_count,
            real_ions / float(macro_count),
            dtype=np.float64,
        )
        self._populate_capillary_slots(
            state,
            indices,
            positions,
            weights,
            source_entry_time_s=0.0,
        )
        self._record_prefill(real_ions, macro_count)

    def _prefill_macro_count(self, real_ions: float) -> int:
        configured = int(self.config.capillary_prefill_macro_particles)
        if configured > 0:
            macro_count = configured
        else:
            macro_count = max(
                1,
                int(self.config.macro_particles_per_injection),
            )
            if self.config.max_macro_particle_weight > 0.0:
                macro_count = max(
                    macro_count,
                    int(
                        np.ceil(
                            real_ions
                            / float(self.config.max_macro_particle_weight)
                        )
                    ),
                )
        macro_count = max(1, macro_count)
        if macro_count > self.config.ion_count:
            raise ValueError(
                f"--ion-count={self.config.ion_count} is too small for the "
                "requested capillary buffer; need at least "
                f"{macro_count} weighted ion pack slots to represent the initial "
                "steady-state reservoir."
            )
        return macro_count

    def _record_prefill(self, real_ions: float, macro_count: int) -> None:
        self._source_prefill_real_ions += float(real_ions)
        self._source_prefill_macro_particles += int(macro_count)
        self._source_injected_real_ions += float(real_ions)
        self._source_injected_macro_particles += int(macro_count)

    def _inject_continuous_source(
        self,
        state: dict[str, np.ndarray],
        time_s: float,
        dt_s: float,
    ) -> int:
        if not self._continuous_source_enabled() or dt_s <= 0.0:
            return 0
        pending, target, full_count = self._accumulate_source_current(dt_s)
        if full_count <= 0:
            return 0
        selected = self._select_injection_slots(
            state,
            pending,
            target,
            full_count,
        )
        if selected is None:
            return 0
        indices, emitted_real_ions = selected
        weights = np.full(indices.size, target, dtype=np.float64)
        injected_real_ions = float(indices.size) * target
        self._populate_injected_slots(
            state,
            indices,
            weights,
            time_s,
            injected_real_ions,
        )
        self._record_boundary_injection(indices.size, injected_real_ions)
        self._source_real_ion_accumulator = max(
            pending - emitted_real_ions,
            0.0,
        )
        return int(indices.size)

    def _accumulate_source_current(
        self,
        dt_s: float,
    ) -> tuple[float, float, int]:
        real_ions = (
            float(self.config.ion_current_a)
            * float(dt_s)
            / self._source_charge_per_ion_c()
        )
        self._source_real_ion_accumulator += max(real_ions, 0.0)
        pending = float(self._source_real_ion_accumulator)
        target = self._source_target_macro_weight()
        full_count = int(
            np.floor((pending + 1.0e-12 * target) / target)
        )
        return pending, target, full_count

    def _select_injection_slots(
        self,
        state: dict[str, np.ndarray],
        pending: float,
        target: float,
        full_count: int,
    ) -> tuple[np.ndarray, float] | None:
        free = np.flatnonzero(
            np.asarray(state["phase"], dtype=np.int16) == SLOT_FREE
        )
        emitted_real_ions = float(full_count) * target
        if free.size == 0:
            self._discard_blocked_source(pending, emitted_real_ions)
            return None
        macro_count = min(full_count, int(free.size))
        if macro_count <= 0:
            self._discard_blocked_source(pending, emitted_real_ions)
            return None
        blocked_count = full_count - macro_count
        if blocked_count > 0:
            self._source_blocked_real_ions += float(blocked_count) * target
        return (
            np.asarray(free[:macro_count], dtype=np.int32),
            emitted_real_ions,
        )

    def _discard_blocked_source(
        self,
        pending: float,
        emitted_real_ions: float,
    ) -> None:
        self._source_blocked_real_ions += emitted_real_ions
        self._source_real_ion_accumulator = max(
            pending - emitted_real_ions,
            0.0,
        )

    def _populate_injected_slots(
        self,
        state: dict[str, np.ndarray],
        indices: np.ndarray,
        weights: np.ndarray,
        time_s: float,
        injected_real_ions: float,
    ) -> None:
        if max(float(self.config.capillary_prefill_length_m), 0.0) <= 0.0:
            self._populate_external_source_slots(
                state,
                indices,
                weights,
                event_time_s=float(time_s),
            )
            self._source_emitted_real_ions += injected_real_ions
            self._source_emitted_macro_particles += int(indices.size)
            return
        positions, _ = self._sample_capillary_phase_space(
            int(indices.size),
            prefill=False,
        )
        self._populate_capillary_slots(
            state,
            indices,
            positions,
            weights,
            source_entry_time_s=float(time_s),
        )

    def _record_boundary_injection(
        self,
        macro_count: int,
        real_ions: float,
    ) -> None:
        self._source_boundary_injected_real_ions += real_ions
        self._source_boundary_injected_macro_particles += int(macro_count)
        self._source_injected_real_ions += real_ions
        self._source_injected_macro_particles += int(macro_count)


__all__ = ["ContinuousSourceRuntimeMixin"]
