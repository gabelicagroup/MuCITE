"""Ordered composition root for the coupled simulation runtime."""

from __future__ import annotations

from typing import Any, Optional

from ....core.ports import SnapshotSink
from .artifacts import RuntimeArtifacts, prepare_runtime_artifacts
from .control import install_control_state, install_field_metadata
from .grids import (
    install_effective_grid_config,
    install_electrode_masks,
    install_grid_state,
    install_particle_operators,
    install_taichi_backend,
)
from .messages import install_backend_message
from .rf import install_rf_state
from .services import BootstrapServices
from .source_birth import configure_source_birth_velocity


BOOTSTRAP_PHASE_ORDER = (
    "artifacts",
    "control",
    "rf",
    "taichi",
    "grids",
    "masks",
    "operators",
    "source_birth",
    "message",
    "field_sampler",
    "particle_state",
    "source_collision_runtime",
)


def _install_field_sampler(
    owner: Any,
    artifacts: RuntimeArtifacts,
    services: BootstrapServices,
) -> None:
    owner.field_sampler = services.field_sampler_factory(
        config=owner.config,
        static_grid=owner.static_grid,
        pic_grid=owner.pic_grid,
        cartesian_field3d=owner.cartesian_field3d,
        rf_enabled=owner.rf_enabled,
        rf_peak_to_reference_scale=owner.rf_peak_to_reference_scale,
        rf_angular_frequency_rad_s=owner.rf_angular_frequency_rad_s,
    )
    owner._field_cache = owner.field_sampler.cache
    owner._static_field_cache = owner.field_sampler.static_cache
    owner._pic_field_cache = owner.field_sampler.pic_cache


def _install_particle_state(
    owner: Any,
    template: Any,
    services: BootstrapServices,
) -> None:
    services.particle_state_type.allocate(
        owner.config.ion_count,
        template,
        default_weight=owner.config.macro_particle_weight,
    ).bind_to(owner)
    owner.terminal_event_recorder = services.terminal_recorder_factory()
    owner._terminal_event_rows = owner.terminal_event_recorder.rows


def _install_source_and_collision_runtime(
    owner: Any,
    template: Any,
    birth_velocity_sampler: Any,
    birth_axial_velocity_m_per_s: Optional[float],
    services: BootstrapServices,
) -> None:
    config = owner.config
    owner.continuous_source = services.continuous_source_factory(
        template,
        config,
        owner.rng,
        domain_length_m=config.domain_length_m,
        temperature_sampler=owner._scalar_bilinear,
        field_sampler=owner._scalar_bilinear,
        birth_velocity_sampler=birth_velocity_sampler,
        birth_axial_velocity_m_per_s=birth_axial_velocity_m_per_s,
    )
    owner.collision_operator = services.collision_operator_factory(owner)
    owner._source_real_ion_accumulator = 0.0
    owner._source_prefill_real_ions = 0.0
    owner._source_prefill_macro_particles = 0
    owner._source_boundary_injected_real_ions = 0.0
    owner._source_boundary_injected_macro_particles = 0
    owner._source_emitted_real_ions = 0.0
    owner._source_emitted_macro_particles = 0
    owner._source_injected_real_ions = 0.0
    owner._source_injected_macro_particles = 0
    owner._source_blocked_real_ions = 0.0
    owner._reset_terminal_accounting()
    owner._reset_collision_statistics()


def _install_model_allocations(
    owner: Any,
    artifacts: RuntimeArtifacts,
    requested_backend: str,
    services: BootstrapServices,
) -> str:
    install_taichi_backend(owner, requested_backend, services)
    config, pic_nr, pic_nz = install_effective_grid_config(
        owner,
        artifacts.config,
        services,
    )
    static_message = install_grid_state(
        owner,
        config,
        pic_nr,
        pic_nz,
        artifacts,
        services,
    )
    install_electrode_masks(owner, config, artifacts, services)
    install_particle_operators(
        owner,
        config,
        artifacts,
        requested_backend,
    )
    return static_message


def bootstrap_simulation(
    owner: Any,
    template: Any,
    config: Any,
    *,
    particle_backend: str,
    data_logger: Optional[SnapshotSink],
    event_bus: Optional[Any],
    services: BootstrapServices,
) -> None:
    """Build the runtime in the legacy side-effect order."""

    requested_backend = particle_backend.lower()
    if requested_backend not in {"cpu", "taichi"}:
        raise ValueError(
            f"Unsupported particle backend: {particle_backend}"
        )
    artifacts = prepare_runtime_artifacts(owner, config, services)
    install_control_state(
        owner, template, artifacts, data_logger=data_logger,
        event_bus=event_bus, services=services,
    )
    install_field_metadata(owner, artifacts, services)
    install_rf_state(owner, artifacts, services)
    static_message = _install_model_allocations(
        owner, artifacts, requested_backend, services,
    )
    birth_sampler, birth_axial_velocity = configure_source_birth_velocity(
        owner, template, artifacts, services,
    )
    install_backend_message(owner, artifacts, static_message, services)
    _install_field_sampler(owner, artifacts, services)
    _install_particle_state(owner, template, services)
    _install_source_and_collision_runtime(
        owner, template, birth_sampler, birth_axial_velocity, services,
    )
    owner._bootstrap_phase_history = BOOTSTRAP_PHASE_ORDER
