"""Strict validation for externalized runtime and retention tuning."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np

from .models import ExecutionConfig, OutputConfig, SimulationConfig


def _finite_real(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real number.")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    return numeric


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _positive_real(value: Any, field_name: str) -> float:
    numeric = _finite_real(value, field_name)
    if numeric <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return numeric


def _fraction(value: Any, field_name: str) -> float:
    numeric = _positive_real(value, field_name)
    if numeric > 1.0:
        raise ValueError(f"{field_name} must be in (0, 1].")
    return numeric


def _nonnegative_integer(value: Any, field_name: str) -> int:
    numeric = _integer(value, field_name)
    if numeric < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return numeric


def _positive_integer(value: Any, field_name: str) -> int:
    numeric = _integer(value, field_name)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return numeric


def validate_execution_tuning(config: ExecutionConfig) -> None:
    """Validate cadence controls kept outside the physical model."""

    _positive_real(config.progress_interval_s, "progress_interval_s")


def validate_output_tuning(config: OutputConfig) -> None:
    """Validate bounded-memory and plot-retention controls."""

    for field_name in (
        "dt_quantile_reservoir_size",
        "macro_history_max_rows",
        "terminal_event_sample_rows",
        "trajectory_plot_max_tracks",
    ):
        _nonnegative_integer(getattr(config, field_name), field_name)
    _integer(config.terminal_event_sample_seed, "terminal_event_sample_seed")
    _positive_integer(config.report_plot_max_points, "report_plot_max_points")


def validate_simulation_tuning(config: SimulationConfig) -> None:
    """Validate externalized numerical knobs without changing their formulas."""

    for field_name in (
        "collision_safety_factor",
        "rf_safety_factor",
        "acceleration_safety_factor",
        "advection_cfl_fraction",
        "electrode_sweep_spacing_fraction",
    ):
        _fraction(getattr(config, field_name), field_name)
    _positive_real(config.spatial_tolerance_m, "spatial_tolerance_m")
    min_dt_s = _positive_real(config.min_time_step_s, "min_time_step_s")
    max_dt_s = _positive_real(config.max_time_step_s, "max_time_step_s")
    if min_dt_s > max_dt_s:
        raise ValueError("min_time_step_s must not exceed max_time_step_s.")
    gamma = _positive_real(config.gas_gamma, "gas_gamma")
    if gamma <= 1.0:
        raise ValueError("gas_gamma must be greater than 1.")
    _positive_real(
        config.gas_molar_mass_kg_per_mol,
        "gas_molar_mass_kg_per_mol",
    )
    for field_name in (
        "source_reference_velocity_sample_count",
        "source_gaussian_max_attempts",
        "source_gaussian_min_batch",
        "source_gaussian_oversample_factor",
    ):
        _positive_integer(getattr(config, field_name), field_name)
    _nonnegative_integer(
        config.electrode_hit_bisection_iterations,
        "electrode_hit_bisection_iterations",
    )


__all__ = [
    "validate_execution_tuning",
    "validate_output_tuning",
    "validate_simulation_tuning",
]
