"""Shared collision-runtime state and dispatch methods."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...config import CollisionBatchUpdate, LocalStateBatch
from .dissociation import (
    advance_dissociation, DISSOCIATION_SCHEDULE_VERSION, LOSS_WEIGHT_UNDERFLOW_LIMIT,
    check_final_transport_survival,
)
from ..boundaries.events import TerminalEvents
from ..gas.kinetics import COLLISION_RATE_MODEL
from .joint_step import JOINT_LANGEVIN_VERSION


class CollisionStateMixin:
    """Collision counters, empty updates, and runtime dispatch."""

    def _check_dissociation_survival(self, state: dict[str, np.ndarray]) -> None:
        check_final_transport_survival(self, state)

    def _advance_dissociation(
        self, state: dict[str, np.ndarray], *, step_start_time_s: float,
        step_dt_s: float, events: TerminalEvents,
    ) -> TerminalEvents:
        return advance_dissociation(
            self, state, step_start_time_s=step_start_time_s,
            step_dt_s=step_dt_s, events=events,
        )

    def _reset_collision_statistics(self) -> None:
        self._particle_step_cache = None
        self._dissociation_rate_cache = None
        self._dissociation_step_count = 0
        self.dissociation_runtime_info = {
            "dissociation_schedule_version": DISSOCIATION_SCHEDULE_VERSION,
            "collision_rate_model": COLLISION_RATE_MODEL,
            "hazard_definition": "integral k(T_internal) dt; dimensionless",
            "transport_survival_max_abs_error": 0.0,
            "transport_survival_max_relative_error": 0.0,
            "transport_survival_checks": 0,
            "dissociation_check_every": int(getattr(self.config, "dissociation_check_every", 100)),
            "dissociation_rate_evaluated_particles": 0,
            "dissociation_rate_cache_hits": 0,
            "loss_weight_underflow_limit": LOSS_WEIGHT_UNDERFLOW_LIMIT,
            "loss_underflow_depletions": 0,
            "hybrid_internal_energy_corrected": True,
            "hybrid_joint_model": JOINT_LANGEVIN_VERSION,
            "hybrid_joint_status": "experimental; convergence required for each physical regime",
            "hybrid_joint_gaussian_steps": 0,
            "hybrid_joint_binary_events": 0,
            "hybrid_joint_guard_fallbacks": 0,
            "hybrid_joint_energy_boundary_guard_sigmas": 8,
            "hybrid_joint_gas_environment": "frozen over the outer collision operator",
            "hybrid_collision_count_note": "actual binary events plus rounded expected dense-step counts",
        }
        self._collision_real_ions_represented = 0.0
        self._fragmented_real_ions_represented = 0.0
        self._collision_macro_events = 0
        self._hybrid_langevin_macro_events = 0
        self._hybrid_langevin_real_ions_represented = 0.0
        self._hybrid_langevin_estimated_binary_collisions = 0.0

    def _apply_collision_updates(
        self,
        state: dict[str, np.ndarray],
        active_indices: np.ndarray,
        positions_m: np.ndarray,
        samples: LocalStateBatch,
        dt_s: float,
        time_s: float,
    ) -> CollisionBatchUpdate:
        return self.collision_operator.apply_updates(
            state,
            active_indices,
            positions_m,
            samples,
            dt_s,
            time_s,
        )

    def _empty_collision_update(self) -> CollisionBatchUpdate:
        empty_indices = np.zeros(0, dtype=np.int32)
        return CollisionBatchUpdate(
            collision_hits=0,
            fragmentation_hits=0,
            collision_real_ions_represented=0.0,
            fragmented_real_ions_represented=0.0,
            updated_indices=empty_indices,
            deactivated_indices=empty_indices,
        )

    def _taichi_runtime_apply_collisions_before_push(
        self,
        state: dict[str, np.ndarray],
        dt_s: float,
        time_s: float,
        cached_collision_rates_hz: Optional[np.ndarray] = None,
    ) -> CollisionBatchUpdate:
        if not self.config.collision_preselection_enabled:
            return self._taichi_runtime_apply_collisions_before_push_legacy(
                state,
                dt_s,
                time_s,
            )
        return self._taichi_runtime_apply_collisions_before_push_preselected(
            state,
            dt_s,
            time_s,
            cached_collision_rates_hz=cached_collision_rates_hz,
        )


__all__ = ["CollisionStateMixin"]
