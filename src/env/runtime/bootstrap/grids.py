"""Taichi backend, shared/decoupled grids, masks, and model operators."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from scipy.constants import Avogadro

from ....agents import IonCloud3D
from ....config import SimulationConfig
from ...fields import (
    UnifiedGrid2D,
    estimate_runtime_grid_resources,
    resolve_pic_grid_shape,
)
from ...pic import PICSolver
from ...transport import ParticlePusher
from .artifacts import RuntimeArtifacts
from .services import BootstrapServices


def install_taichi_backend(
    owner: Any,
    requested_backend: str,
    services: BootstrapServices,
) -> None:
    _, owner.arch_label = services.ensure_taichi(requested_backend)
    owner.particle_backend_requested = requested_backend
    owner.particle_backend_effective = owner.arch_label


def install_effective_grid_config(
    owner: Any,
    config: SimulationConfig,
    services: BootstrapServices,
) -> tuple[SimulationConfig, int, int]:
    pic_nr, pic_nz, source = resolve_pic_grid_shape(
        static_nr=config.grid_nr,
        static_nz=config.grid_nz,
        r_max_m=config.domain_radius_m,
        z_min_m=0.0,
        z_max_m=config.domain_length_m,
        requested_nr=config.pic_grid_nr,
        requested_nz=config.pic_grid_nz,
    )
    effective = services.make_config(
        replace(config, pic_grid_nr=pic_nr, pic_grid_nz=pic_nz)
    )
    owner.config = effective
    owner.effective_config = effective
    owner.pic_grid_resolution_source = source
    owner.pic_grid_is_decoupled = (
        pic_nr != effective.grid_nr or pic_nz != effective.grid_nz
    )
    return effective, pic_nr, pic_nz


def _install_static_grid(
    owner: Any,
    config: SimulationConfig,
    artifacts: RuntimeArtifacts,
    services: BootstrapServices,
) -> str:
    storage_mode = "static" if owner.pic_grid_is_decoupled else "unified"
    owner.static_grid = UnifiedGrid2D(
        nr=config.grid_nr,
        nz=config.grid_nz,
        r_max_m=config.domain_radius_m,
        z_min_m=0.0,
        z_max_m=config.domain_length_m,
        storage_mode=storage_mode,
    )
    if (
        artifacts.baked_fields is not None
        and artifacts.static_field_path is not None
    ):
        services.load_static_fields(
            owner.static_grid,
            artifacts.baked_fields,
        )
        return services.baked_field_message(
            artifacts.static_field_path,
            artifacts.baked_fields,
        )
    owner.static_grid.bake_dummy_fields()
    return "Using analytic dummy static fields."


def _install_pic_grid(
    owner: Any,
    config: SimulationConfig,
    pic_nr: int,
    pic_nz: int,
) -> None:
    if owner.pic_grid_is_decoupled:
        owner.pic_grid = UnifiedGrid2D(
            nr=pic_nr,
            nz=pic_nz,
            r_max_m=config.domain_radius_m,
            z_min_m=0.0,
            z_max_m=config.domain_length_m,
            storage_mode="pic",
        )
    else:
        owner.pic_grid = owner.static_grid
    owner.grid_storage_estimate = estimate_runtime_grid_resources(
        static_nr=config.grid_nr,
        static_nz=config.grid_nz,
        pic_nr=pic_nr,
        pic_nz=pic_nz,
    )
    owner.grid = owner.static_grid


def install_grid_state(
    owner: Any,
    config: SimulationConfig,
    pic_nr: int,
    pic_nz: int,
    artifacts: RuntimeArtifacts,
    services: BootstrapServices,
) -> str:
    static_message = _install_static_grid(
        owner,
        config,
        artifacts,
        services,
    )
    _install_pic_grid(owner, config, pic_nr, pic_nz)
    if owner.stage_schedule:
        owner._record_stage_activation(
            owner.stage_schedule[0],
            event_time_s=0.0,
            loaded_dc=False,
        )
    return static_message


def install_electrode_masks(
    owner: Any,
    config: SimulationConfig,
    artifacts: RuntimeArtifacts,
    services: BootstrapServices,
) -> None:
    owner.electrode_mask = (
        None
        if config.electrode_mask_path is None
        else services.load_electrode_mask(
            source_path=config.electrode_mask_path,
            cache_path=config.electrode_mask_cache_path,
            z_offset_m=config.electrode_mask_z_offset_m,
            field_metadata=artifacts.baked_fields,
            project_root=services.project_root,
        )
    )
    owner.electrode_mask_metadata = (
        {"loaded": False}
        if owner.electrode_mask is None
        else services.json_safe(owner.electrode_mask.metadata)
    )
    owner.electrode_mask3d = services.load_electrode_mask3d(
        services.resolve_path(config.electrode_mask_3d_path)
    )
    owner.electrode_mask_3d_metadata = (
        {"loaded": False}
        if owner.electrode_mask3d is None
        else services.json_safe(owner.electrode_mask3d.metadata)
    )


def install_particle_operators(
    owner: Any,
    config: SimulationConfig,
    artifacts: RuntimeArtifacts,
    requested_backend: str,
) -> None:
    owner.pic_solver = PICSolver(
        owner.pic_grid,
        sor_omega=config.sor_omega,
        max_sor_iters=config.sor_max_iters,
        sor_tolerance=config.sor_tolerance,
        poisson_backend=config.pic_poisson_backend,
        poisson_preconditioner=config.pic_poisson_preconditioner,
        poisson_warm_start=config.pic_poisson_warm_start,
        poisson_amg_mode=config.pic_amg_mode,
        poisson_amg_solver=config.pic_amg_solver,
        poisson_amg_tolerance=config.pic_amg_tolerance,
        poisson_amg_max_iters=config.pic_amg_max_iters,
        poisson_amg_fallback_backend=config.pic_amg_fallback_backend,
        electrode_mask=owner.electrode_mask,
    )
    owner.cloud = IonCloud3D(config.ion_count)
    owner.particle_pusher = ParticlePusher(
        owner.static_grid,
        pic_grid=owner.pic_grid,
        advection_cfl_fraction=config.advection_cfl_fraction,
        collision_rate_model_code=(
            1
            if str(config.collision_physics_backend).lower() == "iict-lite"
            else 0
        ),
        gas_molecule_mass_kg=config.gas_molar_mass_kg_per_mol / Avogadro,
        electrode_mask=owner.electrode_mask,
        electrode_mask3d=owner.electrode_mask3d,
        cartesian_field3d=artifacts.cartesian_field3d,
    )
    owner.particle_backend = requested_backend
    owner.collision_model_code = (
        1
        if str(config.collision_model).lower() == "hybrid-langevin"
        else 0
    )
    owner.collision_rate_model_code = (
        1
        if str(config.collision_physics_backend).lower() == "iict-lite"
        else 0
    )
