"""Install control objects, deterministic RNG, and stage metadata."""

from __future__ import annotations

from typing import Any, Optional

from ....core.ports import SnapshotSink
from .artifacts import RuntimeArtifacts
from .services import BootstrapServices


def install_control_state(
    owner: Any,
    template: Any,
    artifacts: RuntimeArtifacts,
    *,
    data_logger: Optional[SnapshotSink],
    event_bus: Optional[Any],
    services: BootstrapServices,
) -> None:
    config = artifacts.config
    owner.template = template
    owner.config = config
    owner.cancellation_token = services.cancellation_token_factory()
    owner.run_guard = services.run_guard_factory()
    owner.event_bus = event_bus or services.event_bus_factory()
    if config.source_mode not in {"packet", "continuous-current"}:
        raise ValueError(
            "source_mode must be 'packet' or 'continuous-current'."
        )
    owner.integrator = services.integrator_factory(
        rf_frequency_hz=config.rf_frequency_hz,
        collision_safety_factor=config.collision_safety_factor,
        rf_safety_factor=config.rf_safety_factor,
        acceleration_safety_factor=config.acceleration_safety_factor,
        spatial_tolerance_m=config.spatial_tolerance_m,
        min_dt_s=config.min_time_step_s,
        max_dt_s=config.max_time_step_s,
    )
    owner.adapter = services.collision_adapter_factory(
        config=config,
        template=template,
        project_root=getattr(services, "project_root", None),
    )
    owner.seed_manager = services.seed_manager_factory(config.random_seed)
    owner.rng = owner.seed_manager.physical
    owner.data_logger = data_logger
    owner.static_field_path = artifacts.static_field_path
    owner.stage_schedule_path = artifacts.stage_schedule_path
    owner.stage_schedule = artifacts.stage_schedule
    owner.current_stage_index = 0
    owner._current_stage_source_enabled = bool(
        not artifacts.stage_schedule
        or artifacts.stage_schedule[0].source_enabled
    )
    owner.stage_history = []
    owner.static_field_3d_path = artifacts.static_field_3d_path
    owner.cartesian_field3d = artifacts.cartesian_field3d
    owner._stage_services = services


def install_field_metadata(
    owner: Any,
    artifacts: RuntimeArtifacts,
    services: BootstrapServices,
) -> None:
    metadata = services.baked_metadata_summary(artifacts.baked_fields)
    metadata["stage_schedule"] = services.stage_schedule_summary(
        artifacts.stage_schedule_path,
        artifacts.stage_schedule,
    )
    if artifacts.cartesian_field3d is not None:
        metadata["cartesian3d"] = (
            artifacts.cartesian_field3d.metadata_summary()
        )
    owner.baked_field_metadata = metadata
