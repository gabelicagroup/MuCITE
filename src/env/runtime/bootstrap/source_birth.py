"""Resolve optional Fluent-based source birth velocity sampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .artifacts import RuntimeArtifacts
from .services import BootstrapServices


@dataclass(frozen=True)
class SourceBirthSelection:
    path: Optional[Path]
    z_min_m: float
    z_max_m: float
    radius_m: Optional[float]
    origin: str


def _initial_source_birth_selection(
    template: Any,
    config: Any,
    services: BootstrapServices,
) -> SourceBirthSelection:
    path = services.resolve_path(config.source_birth_velocity_gas_path)
    z_min_m = float(max(config.source_birth_velocity_z_min_m, 0.0))
    if config.source_birth_velocity_z_max_m is None:
        z_max_m = (
            float(config.capillary_exit_z_m)
            if config.capillary_exit_z_m is not None
            else float(
                np.clip(
                    template.initial_position_m[2],
                    0.0,
                    config.domain_length_m,
                )
            )
        )
    else:
        z_max_m = float(config.source_birth_velocity_z_max_m)
    return SourceBirthSelection(
        path=path,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        radius_m=config.source_birth_velocity_radius_m,
        origin=(
            "explicit_cli"
            if config.source_birth_velocity_gas_path is not None
            else "none"
        ),
    )


def _selection_from_baked_metadata(
    selection: SourceBirthSelection,
    baked_fields: Optional[dict[str, Any]],
    services: BootstrapServices,
) -> SourceBirthSelection:
    if selection.path is not None or baked_fields is None:
        return selection
    metadata = baked_fields.get("fluent", {})
    path = metadata.get("fluent_source_path")
    z_min_m = metadata.get(
        "fluent_capillary_source_velocity_raw_z_min_m"
    )
    z_max_m = metadata.get(
        "fluent_capillary_source_velocity_raw_z_max_m"
    )
    radius_m = metadata.get("fluent_capillary_radius_m")
    if not path or z_min_m is None or z_max_m is None:
        return selection
    return replace(
        selection,
        path=services.resolve_path(Path(str(path))),
        z_min_m=float(z_min_m),
        z_max_m=float(z_max_m),
        radius_m=(
            float(radius_m)
            if selection.radius_m is None and radius_m is not None
            else selection.radius_m
        ),
        origin="baked_fluent_capillary_metadata",
    )


def configure_source_birth_velocity(
    owner: Any,
    template: Any,
    artifacts: RuntimeArtifacts,
    services: BootstrapServices,
) -> tuple[
    Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]],
    Optional[float],
]:
    owner.source_birth_velocity_metadata = {"loaded": False}
    config = owner.config
    if not config.source_velocity_from_gas_field:
        return None, None
    selection = _selection_from_baked_metadata(
        _initial_source_birth_selection(
            template,
            config,
            services,
        ),
        artifacts.baked_fields,
        services,
    )
    if selection.path is None:
        return None, None
    sampler = services.gas_velocity_sampler_type.from_fluent_csv(
        selection.path,
        z_min_m=selection.z_min_m,
        z_max_m=selection.z_max_m,
        r_max_m=selection.radius_m,
        reference_sample_count=config.source_reference_velocity_sample_count,
    )
    owner.source_birth_velocity_metadata = services.json_safe(
        {**sampler.metadata(), "origin": selection.origin}
    )
    return (
        sampler.sample_velocity_vectors,
        sampler.reference_axial_velocity_m_per_s(),
    )
