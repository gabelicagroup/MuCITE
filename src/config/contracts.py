"""Runtime result and callback contracts shared across package boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass(frozen=True)
class CollisionOutcome:
    """Result of one Monte Carlo collision event."""

    velocity_m_per_s: np.ndarray
    internal_temperature_k: float
    transferred_internal_energy_j: float
    fragmentation_probability: float


@dataclass(frozen=True)
class LocalStateBatch:
    """Vectorized local gas/electric samples for active particles."""

    gas_velocity_m_per_s: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    number_density_m3: np.ndarray
    mass_density_kg_per_m3: np.ndarray
    mach_number: np.ndarray
    external_field_v_per_m: np.ndarray
    sce_field_v_per_m: np.ndarray
    total_field_v_per_m: np.ndarray


@dataclass(frozen=True)
class CollisionBatchUpdate:
    """Sparse particle updates produced by one collision pre-push step."""

    collision_hits: int
    fragmentation_hits: int
    collision_real_ions_represented: float
    fragmented_real_ions_represented: float
    updated_indices: np.ndarray
    deactivated_indices: np.ndarray


@dataclass
class SimulationResult:
    """Collected outputs from the coupled simulation."""

    survivors: list[Any]
    macro_history: list[dict[str, Any]]
    collision_count: int
    particle_backend: str
    macro_history_summary: dict[str, Any] = field(default_factory=dict)
    particle_backend_requested: str = ""
    particle_backend_effective: str = ""
    export_directory: Optional[str] = None
    final_state: Optional[dict[str, np.ndarray]] = None
    final_time_s: float = 0.0
    micro_step_count: int = 0
    run_wall_time_s: float = 0.0
    run_started_at: str = ""
    run_ended_at: str = ""
    termination_reason: str = ""
    performance_breakdown: dict[str, float] = field(default_factory=dict)
    dt_statistics: dict[str, float | int | bool] = field(default_factory=dict)
    dt_limiter_counts: dict[str, int] = field(default_factory=dict)
    dt_raw_limiter_counts: dict[str, int] = field(default_factory=dict)
    collision_statistics: dict[str, float] = field(default_factory=dict)
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    numerical_failure: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationProgress:
    """Progress snapshot for stdout logging or live UI updates."""

    time_s: float
    macro_step_index: int
    micro_step_index: int
    ion_count: int
    collision_count: int
    mean_internal_temperature_k: float
    mean_axial_position_m: float
    particle_backend: str
    backend_message: str = ""


ProgressCallback = Callable[[SimulationProgress], None]
