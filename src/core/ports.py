"""Dependency-inversion ports used by the control layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


class ProgressObserver(Protocol):
    """Receive immutable progress projections from the engine."""

    def __call__(self, progress: Any) -> None: ...


class EventPublisher(Protocol):
    """Publish lifecycle events without depending on a renderer."""

    def publish(self, event: Any) -> None: ...


@runtime_checkable
class SnapshotSink(Protocol):
    """Optional output sink observed by the engine at safe checkpoints."""

    output_dir: Path
    last_logged_macro_step: int | None

    def should_log(self, macro_step_index: int) -> bool: ...

    def log_snapshot(
        self,
        macro_step_index: int,
        time_s: float,
        particle_rows: np.ndarray,
    ) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class TerminalEventSink(Protocol):
    """Event collector/writer port; concrete CSV handling belongs to data/."""

    rows: list[dict[str, float | int | str]]

    def clear(self) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...

    def record(self, batch: Any, **options: Any) -> None: ...


class SimulationRuntime(Protocol):
    """Operations required by the macro/micro control engine.

    Concrete field, particle, collision, source, and terminal implementations
    remain outside :mod:`src.core`; the engine coordinates them through this
    structural port.
    """

    config: Any
    template: Any
    integrator: Any
    particle_pusher: Any
    cloud: Any
    data_logger: SnapshotSink | None
    terminal_event_recorder: TerminalEventSink
    event_bus: EventPublisher
    particle_backend: str
    particle_backend_requested: str
    particle_backend_effective: str
    backend_message: str
    current_stage_index: int
    stage_history: list[dict[str, Any]]
    collision_model_code: int
    collision_rate_model_code: int
    rf_enabled: bool
    rf_angular_frequency_rad_s: float
    rf_peak_to_reference_scale: float
    _current_stage_source_enabled: bool
    _particle_status_codes: np.ndarray
    _source_terminal_macro_particles_by_code: dict[int, int]

    def initialize_ions(self) -> None: ...

    def is_cancel_requested(self) -> bool: ...

    def _active_indices(self, state: dict[str, np.ndarray]) -> np.ndarray: ...

    def _continuous_source_enabled(self) -> bool: ...

    def _apply_stage_for_time(self, time_s: float) -> None: ...

    def _next_stage_boundary_s(self, time_s: float) -> float: ...

    def _current_stage_name(self) -> str: ...

    def _update_space_charge_fields(self) -> Any: ...

    def _download_cloud_state(self) -> dict[str, np.ndarray]: ...

    def _download_motion_state_into(self, state: dict[str, np.ndarray]) -> None: ...

    def _download_selected_motion_state(
        self,
        particle_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def _sample_local_state_batch(
        self,
        positions_m: np.ndarray,
        time_s: float,
    ) -> Any: ...

    def _taichi_runtime_apply_collisions_before_push(
        self,
        state: dict[str, np.ndarray],
        dt_s: float,
        time_s: float,
        cached_collision_rates_hz: np.ndarray | None = None,
    ) -> Any: ...

    def _upload_active_state_updates(
        self,
        state: dict[str, np.ndarray],
        particle_indices: np.ndarray,
    ) -> None: ...

    def _upload_deactivated_indices(
        self,
        particle_indices: np.ndarray,
    ) -> None: ...

    def _taichi_runtime_apply_terminal_boundary_events(
        self,
        state: dict[str, np.ndarray],
        event_time_s: float,
        *,
        step_start_time_s: float | None = None,
        step_dt_s: float | None = None,
    ) -> np.ndarray: ...

    def _advance_capillary_source(
        self,
        state: dict[str, np.ndarray],
        dt_s: float,
        event_time_s: float,
    ) -> np.ndarray: ...

    def _inject_continuous_source(
        self,
        state: dict[str, np.ndarray],
        time_s: float,
        dt_s: float,
    ) -> int: ...

    def _build_progress_snapshot(
        self,
        state: dict[str, np.ndarray],
        time_s: float,
        macro_step_index: int,
        micro_step_index: int,
        collision_count: int,
    ) -> Any: ...

    def _space_charge_external_ratio_p95(
        self,
        state: dict[str, np.ndarray],
        time_s: float,
    ) -> float: ...

    def _build_particle_snapshot(
        self,
        state: dict[str, np.ndarray],
    ) -> np.ndarray: ...

    def _collect_survivors(self, state: dict[str, np.ndarray]) -> list[Any]: ...
