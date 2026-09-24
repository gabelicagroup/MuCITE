"""Resolve immutable configuration and field artifacts before allocation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from ....config import SimulationConfig
from .services import BootstrapServices


@dataclass(frozen=True)
class RuntimeArtifacts:
    """Prepared inputs needed by subsequent bootstrap phases."""

    config: SimulationConfig
    stage_schedule_path: Optional[Path]
    stage_schedule: list[Any]
    static_field_path: Optional[Path]
    baked_fields: Optional[dict[str, Any]]
    static_field_3d_path: Optional[Path]
    cartesian_field3d: Optional[Any]


def prepare_runtime_artifacts(
    owner: Any,
    config: SimulationConfig,
    services: BootstrapServices,
) -> RuntimeArtifacts:
    normalized = services.make_config(config)
    owner.requested_config = normalized
    stage_path = services.resolve_path(normalized.stage_schedule_path)
    stage_schedule = services.load_stage_schedule(stage_path)
    static_path = services.resolve_path(normalized.static_field_path)
    if stage_schedule:
        static_path = stage_schedule[0].static_field_path
        normalized = replace(
            normalized,
            static_field_path=static_path,
            total_time_s=float(stage_schedule[-1].end_s),
        )
    baked_fields = None
    if static_path is not None:
        baked_fields = services.load_baked_field(static_path)
        normalized = services.config_with_baked_grid(
            normalized,
            baked_fields,
        )
    field3d_path = services.resolve_path(normalized.static_field_3d_path)
    field3d = (
        None
        if field3d_path is None
        else services.load_cartesian_field(field3d_path)
    )
    return RuntimeArtifacts(
        config=services.make_config(normalized),
        stage_schedule_path=stage_path,
        stage_schedule=stage_schedule,
        static_field_path=static_path,
        baked_fields=baked_fields,
        static_field_3d_path=field3d_path,
        cartesian_field3d=field3d,
    )
