"""Mutable state for exactly one engine run."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .metrics import BoundedMacroHistory, OnlineTimeStepStatistics
from .metrics import (
    DEFAULT_DT_QUANTILE_RESERVOIR_SIZE,
    DEFAULT_MACRO_HISTORY_SAMPLE_SIZE,
)


PERFORMANCE_SECTIONS = (
    "time_pic_poisson",
    "time_download_state",
    "time_sample_fields",
    "time_dt_estimate",
    "time_collision_loop",
    "time_upload_state",
    "time_rk4_push",
    "time_terminal_events",
    "time_source_injection",
)

DT_LIMITER_NAMES = (
    "max_dt",
    "macro_end",
    "collision",
    "rf",
    "acceleration",
    "advection",
    "min_dt_floor",
    "langevin_max",
)


@dataclass
class RunClock:
    time_s: float = 0.0
    macro_step_index: int = 0
    micro_step_index: int = 0


@dataclass
class RunTermination:
    reason: str = ""
    numerical_failure: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return bool(self.reason)

    def finish(self, reason: str) -> None:
        if not self.reason:
            self.reason = str(reason)


@dataclass
class PerformanceBreakdown:
    values: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in PERFORMANCE_SECTIONS}
    )

    def start(self) -> float:
        return time.perf_counter()

    def finish(self, section: str, started_at: float) -> None:
        self.values[section] += time.perf_counter() - float(started_at)

    def finalize(self, wall_time_s: float) -> dict[str, float]:
        timed_total_s = float(sum(self.values.values()))
        self.values["time_timed_sections_total"] = timed_total_s
        self.values["time_unaccounted"] = max(float(wall_time_s) - timed_total_s, 0.0)
        return dict(self.values)


@dataclass
class RunState:
    """Control state kept separate from particle/environment arrays."""

    particle_state: dict[str, Any]
    stable_window_steps: int
    random_seed: int
    run_wall_start: float
    run_started_at: str
    progress_interval_s: float = 0.25
    dt_quantile_reservoir_size: int = DEFAULT_DT_QUANTILE_RESERVOIR_SIZE
    macro_history_max_rows: int = DEFAULT_MACRO_HISTORY_SAMPLE_SIZE
    clock: RunClock = field(default_factory=RunClock)
    termination: RunTermination = field(default_factory=RunTermination)
    performance: PerformanceBreakdown = field(default_factory=PerformanceBreakdown)
    collision_count: int = 0
    export_directory: str | None = None
    last_progress_wall_time: float = field(default_factory=time.monotonic)
    terminal_count_history: deque[int] = field(init=False)
    dt_tracker: OnlineTimeStepStatistics = field(init=False)
    macro_tracker: BoundedMacroHistory = field(init=False)

    def __post_init__(self) -> None:
        self.terminal_count_history = deque(
            maxlen=max(int(self.stable_window_steps) + 1, 1)
        )
        self.dt_tracker = OnlineTimeStepStatistics(
            quantile_reservoir_size=int(self.dt_quantile_reservoir_size),
            limiter_names=DT_LIMITER_NAMES,
            seed=int(self.random_seed),
        )
        self.macro_tracker = BoundedMacroHistory(
            max_sample_rows=int(self.macro_history_max_rows),
            seed=int(self.random_seed),
        )
