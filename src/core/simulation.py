"""Core ion-transport simulation orchestration and PIC runtime."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.constants import elementary_charge

from .cancellation import CancellationToken
from .concurrency import RunGuard
from .engine import SimulationEngine
from .events import EventBus
from .policy import EnginePolicy
from .ports import SnapshotSink
from .randomness import SeedManager
from ..data.fields.runtime_artifacts import (
    FieldStage,
    _baked_field_message,
    _baked_metadata_summary,
    _cartesian_field3d_has_rf,
    _cartesian_field3d_message,
    _config_with_baked_grid,
    _has_nonzero_baked_rf_field,
    _infer_cartesian_field3d_reference_peak_voltage_v,
    _infer_rf_reference_peak_voltage_v,
    _load_baked_dc_fields_into_grid,
    _load_baked_field_file,
    _load_baked_static_fields_into_grid,
    _load_field_stage_schedule,
    _optional_baked_array,
    _require_baked_array,
    _resolve_schedule_relative_path,
    _resolve_static_field_path,
    _rf_convention_message,
    _stage_duration_s,
    _stage_schedule_summary,
    _validate_baked_grid_matches_runtime,
)
from ..data.boundary_masks import (
    load_electrode_mask,
    load_electrode_mask3d,
)
from ..env.collisions import (
    CollisionOperator,
    CollisionRuntimeMixin,
    build_collision_physics_backend,
)
from ..env.transport.adaptive_integrator import AdaptiveRK4Integrator
from ..env.fields import CartesianField3D, load_baked_cartesian_field3d
from ..env.gas import AxisymmetricGasVelocitySampler
from ..agents.slot_state import ParticleRuntimeState
from ..env.boundaries import BoundaryRuntimeMixin
from ..env.fields import RuntimeFieldSampler
from ..env.runtime import ModelRuntimeMixin
from ..env.runtime.bootstrap import (
    BootstrapServices,
    bootstrap_simulation,
)
from ..env.sources import ContinuousCurrentSource, SourceRuntimeMixin
from ..agents.events import TerminalAccounting
from ..data.terminal_recorder import TerminalEventRecorder
from ..config import (
    PARTICLE_ACTIVE,
    SLOT_FREE,
    PROJECT_ROOT,
    IonTemplate,
    ProgressCallback,
    SimulationConfig,
    SimulationProgress,
    SimulationResult,
    make_simulation_config,
)

_PIC_TAICHI = None
_PIC_TAICHI_ARCH_LABEL = "uninitialized"
_PIC_TAICHI_INITIALIZED = False
_PIC_TAICHI_REQUESTED_BACKEND: Optional[str] = None
_PIC_TAICHI_OWNER_THREAD_ID: Optional[int] = None
_PIC_TAICHI_INIT_LOCK = threading.Lock()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _ensure_pic_taichi_initialized(requested_backend: str) -> tuple[Any, str]:
    """Initialize Taichi once for the flowchart-aligned PIC runtime."""

    with _PIC_TAICHI_INIT_LOCK:
        return _ensure_pic_taichi_initialized_locked(requested_backend)


def _require_pic_taichi_owner(operation: str) -> None:
    current_thread_id = threading.get_ident()
    if current_thread_id == _PIC_TAICHI_OWNER_THREAD_ID:
        return
    raise RuntimeError(
        "Taichi runtime is owned by thread "
        f"{_PIC_TAICHI_OWNER_THREAD_ID}; thread {current_thread_id} cannot "
        f"{operation} its device context. Use the owner thread."
    )


def _ensure_pic_taichi_initialized_locked(
    requested_backend: str,
) -> tuple[Any, str]:
    global _PIC_TAICHI, _PIC_TAICHI_ARCH_LABEL, _PIC_TAICHI_INITIALIZED
    global _PIC_TAICHI_REQUESTED_BACKEND, _PIC_TAICHI_OWNER_THREAD_ID

    requested_backend = str(requested_backend).strip().lower()
    if requested_backend not in {"cpu", "taichi"}:
        raise ValueError(f"Unsupported particle backend: {requested_backend}")

    if _PIC_TAICHI_INITIALIZED:
        _require_pic_taichi_owner("reuse")
        if requested_backend != _PIC_TAICHI_REQUESTED_BACKEND:
            raise RuntimeError(
                "Taichi architecture is process-global and this process was already "
                f"initialized for requested backend {_PIC_TAICHI_REQUESTED_BACKEND!r} "
                f"(effective arch {_PIC_TAICHI_ARCH_LABEL!r}); cannot switch to "
                f"{requested_backend!r}. Start a fresh process for the new backend."
            )
        return _PIC_TAICHI, _PIC_TAICHI_ARCH_LABEL

    try:
        import taichi as ti
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Taichi is required for the axisymmetric PIC workflow.") from exc

    if requested_backend == "taichi":
        try:
            ti.init(arch=ti.cuda, default_fp=ti.f64, offline_cache=False)
        except Exception:
            ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    else:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    try:
        current_arch = ti.lang.impl.current_cfg().arch
        _PIC_TAICHI_ARCH_LABEL = str(current_arch).split(".")[-1].lower()
    except Exception:  # pragma: no cover - defensive
        _PIC_TAICHI_ARCH_LABEL = "unknown"

    _PIC_TAICHI = ti
    _PIC_TAICHI_INITIALIZED = True
    _PIC_TAICHI_REQUESTED_BACKEND = requested_backend
    _PIC_TAICHI_OWNER_THREAD_ID = threading.get_ident()
    return _PIC_TAICHI, _PIC_TAICHI_ARCH_LABEL


def release_pic_taichi_runtime() -> None:
    """Synchronize and reset Taichi from the thread that initialized it."""

    global _PIC_TAICHI, _PIC_TAICHI_ARCH_LABEL, _PIC_TAICHI_INITIALIZED
    global _PIC_TAICHI_REQUESTED_BACKEND, _PIC_TAICHI_OWNER_THREAD_ID

    with _PIC_TAICHI_INIT_LOCK:
        if not _PIC_TAICHI_INITIALIZED:
            return
        _require_pic_taichi_owner("release")
        try:
            try:
                _PIC_TAICHI.sync()
            finally:
                _PIC_TAICHI.reset()
        finally:
            _PIC_TAICHI = None
            _PIC_TAICHI_ARCH_LABEL = "uninitialized"
            _PIC_TAICHI_INITIALIZED = False
            _PIC_TAICHI_REQUESTED_BACKEND = None
            _PIC_TAICHI_OWNER_THREAD_ID = None


def _bootstrap_services() -> BootstrapServices:
    """Capture composition dependencies at construction time for patchability."""

    return BootstrapServices(
        make_config=make_simulation_config,
        resolve_path=_resolve_static_field_path,
        load_stage_schedule=_load_field_stage_schedule,
        load_baked_field=_load_baked_field_file,
        config_with_baked_grid=_config_with_baked_grid,
        load_cartesian_field=load_baked_cartesian_field3d,
        baked_metadata_summary=_baked_metadata_summary,
        stage_schedule_summary=_stage_schedule_summary,
        infer_rf_reference=_infer_rf_reference_peak_voltage_v,
        infer_cartesian_rf_reference=(
            _infer_cartesian_field3d_reference_peak_voltage_v
        ),
        has_baked_rf=_has_nonzero_baked_rf_field,
        has_cartesian_rf=_cartesian_field3d_has_rf,
        load_static_fields=_load_baked_static_fields_into_grid,
        load_dc_fields=_load_baked_dc_fields_into_grid,
        load_electrode_mask=load_electrode_mask,
        load_electrode_mask3d=load_electrode_mask3d,
        validate_baked_grid=_validate_baked_grid_matches_runtime,
        baked_field_message=_baked_field_message,
        rf_convention_message=_rf_convention_message,
        cartesian_field_message=_cartesian_field3d_message,
        json_safe=_json_safe,
        ensure_taichi=_ensure_pic_taichi_initialized,
        cancellation_token_factory=CancellationToken,
        run_guard_factory=RunGuard,
        event_bus_factory=EventBus,
        integrator_factory=AdaptiveRK4Integrator,
        collision_adapter_factory=build_collision_physics_backend,
        seed_manager_factory=SeedManager,
        gas_velocity_sampler_type=AxisymmetricGasVelocitySampler,
        field_sampler_factory=RuntimeFieldSampler,
        particle_state_type=ParticleRuntimeState,
        terminal_recorder_factory=TerminalEventRecorder,
        continuous_source_factory=ContinuousCurrentSource,
        collision_operator_factory=CollisionOperator,
        project_root=PROJECT_ROOT,
    )


class FlowchartPicSimulation(
    BoundaryRuntimeMixin,
    SourceRuntimeMixin,
    CollisionRuntimeMixin,
    ModelRuntimeMixin,
):
    """Flowchart-aligned 2D axisymmetric PIC controller.

    This runtime makes the execution order explicit:

    1. macro step: deposit charge on the ``(r, z)`` mesh, solve for ``E_sce``,
    2. micro step: gather local gas/electric state at the current particle location,
    3. evaluate Monte Carlo collision probability in the current cell,
    4. if a collision occurs, update velocity/internal temperature with IonSPA,
    5. push the particle with the updated local ``E_total`` using RK4.
    """

    def __init__(
        self,
        template: IonTemplate,
        config: SimulationConfig,
        *,
        particle_backend: str = "cpu",
        data_logger: Optional[SnapshotSink] = None,
        event_bus: Optional[EventBus] = None,
        engine_policy: Optional[EnginePolicy] = None,
    ) -> None:
        bootstrap_simulation(
            self,
            template,
            config,
            particle_backend=particle_backend,
            data_logger=data_logger,
            event_bus=event_bus,
            services=_bootstrap_services(),
        )
        self.engine_policy = engine_policy or EnginePolicy()

    def request_cancel(self) -> None:
        """Request cooperative termination at the next safe run-loop checkpoint."""

        self.cancellation_token.request()

    def clear_cancel_request(self) -> None:
        """Clear a previous cancellation request before explicitly reusing the object."""

        self.cancellation_token.clear()

    def is_cancel_requested(self) -> bool:
        """Return whether another thread has requested cooperative termination."""

        return self.cancellation_token.requested

    def _reset_terminal_accounting(self) -> None:
        """Reset cumulative terminal counters while preserving legacy attribute names."""

        self._source_terminal_accounting = TerminalAccounting.empty()
        self._source_terminal_real_ions_by_code = self._source_terminal_accounting.real_ions_by_code
        self._source_terminal_macro_particles_by_code = self._source_terminal_accounting.macro_particles_by_code
        self._source_terminal_parent_real_ions_by_code = self._source_terminal_accounting.parent_real_ions_by_code
        self._source_terminal_fragment_real_ions_by_code = self._source_terminal_accounting.fragment_real_ions_by_code

    def run(
        self,
        progress_callback: Optional[ProgressCallback] = None,
        *,
        engine_policy: Optional[EnginePolicy] = None,
    ) -> SimulationResult:
        """Delegate time advancement to the control-layer engine."""

        policy = engine_policy or self.engine_policy
        return SimulationEngine(self, policy).run(progress_callback)

