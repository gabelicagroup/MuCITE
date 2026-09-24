"""Injected outer-layer services used by the model-runtime composition root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BootstrapServices:
    """Factories/loaders retained as injection points for tests and adapters."""

    make_config: Callable[..., Any]
    resolve_path: Callable[..., Any]
    load_stage_schedule: Callable[..., Any]
    load_baked_field: Callable[..., Any]
    config_with_baked_grid: Callable[..., Any]
    load_cartesian_field: Callable[..., Any]
    baked_metadata_summary: Callable[..., Any]
    stage_schedule_summary: Callable[..., Any]
    infer_rf_reference: Callable[..., Any]
    infer_cartesian_rf_reference: Callable[..., Any]
    has_baked_rf: Callable[..., Any]
    has_cartesian_rf: Callable[..., Any]
    load_static_fields: Callable[..., Any]
    load_dc_fields: Callable[..., Any]
    load_electrode_mask: Callable[..., Any]
    load_electrode_mask3d: Callable[..., Any]
    validate_baked_grid: Callable[..., Any]
    baked_field_message: Callable[..., str]
    rf_convention_message: Callable[..., str]
    cartesian_field_message: Callable[..., str]
    json_safe: Callable[..., Any]
    ensure_taichi: Callable[..., Any]
    cancellation_token_factory: Callable[..., Any]
    run_guard_factory: Callable[..., Any]
    event_bus_factory: Callable[..., Any]
    integrator_factory: Callable[..., Any]
    collision_adapter_factory: Callable[..., Any]
    seed_manager_factory: Callable[..., Any]
    gas_velocity_sampler_type: Any
    field_sampler_factory: Callable[..., Any]
    particle_state_type: Any
    terminal_recorder_factory: Callable[..., Any]
    continuous_source_factory: Callable[..., Any]
    collision_operator_factory: Callable[..., Any]
    project_root: Path
