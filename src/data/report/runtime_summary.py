"""Runtime, source-accounting, survivor, and performance report summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import elementary_charge

from ...config import PARTICLE_STATUS_NAMES, SLOT_CAPILLARY, SimulationResult
from .values import _array_stats


@dataclass(frozen=True)
class _SourceValues:
    source_temperature_k: float
    source_velocity_m_per_s: float
    source_exit_speed_m_per_s: float
    active_real_ions: float
    active_parent_real_ions: float
    capillary_real_ions: float
    charge_per_ion_c: float
    elapsed_s: float
    capillary_number_density_m3: float
    terminal_real_ions: dict[str, float]
    terminal_macro_particles: dict[str, int]
    terminal_parent_real_ions: dict[str, float]
    terminal_fragment_real_ions: dict[str, float]
    terminal_current_a: dict[str, float]
    terminal_parent_current_a: dict[str, float]
    terminal_fragment_current_a: dict[str, float]
    boundary_injected_current_a: float
    emitted_current_a: float
    z_exit_real_ions: float
    eta_exit: float


def _survivor_summary(survivors: list[Any]) -> dict[str, Any]:
    if not survivors:
        return {"count": 0}
    positions = np.asarray([ion.position_m for ion in survivors], dtype=float)
    velocities = np.asarray([ion.velocity_m_per_s for ion in survivors], dtype=float)
    temperatures = np.asarray(
        [ion.internal_temperature_k for ion in survivors],
        dtype=float,
    )
    masses = np.asarray([ion.mass_kg for ion in survivors], dtype=float)
    r_m = np.linalg.norm(positions[:, :2], axis=1)
    speed_m_per_s = np.linalg.norm(velocities, axis=1)
    kinetic_energy_ev = 0.5 * masses * speed_m_per_s**2 / elementary_charge
    return {
        "count": int(len(survivors)),
        "r_m": _array_stats(r_m),
        "z_m": _array_stats(positions[:, 2]),
        "speed_m_per_s": _array_stats(speed_m_per_s),
        "kinetic_energy_ev": _array_stats(kinetic_energy_ev),
        "internal_temperature_k": _array_stats(temperatures),
    }


def _runtime_field_summary(simulation: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    field_names = (
        "E_sce_r",
        "E_sce_z",
        "E_dc_r",
        "E_dc_z",
        "E_rf_r",
        "E_rf_z",
        "P_gas",
        "T_gas",
    )
    for field_name in field_names:
        values = simulation._field_cache.get(field_name)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        abs_values = np.abs(values)
        summary[field_name] = {
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "max_abs": float(np.nanmax(abs_values)),
            "mean_abs": float(np.nanmean(abs_values)),
        }
    return summary


def _active_inventory(
    simulation: Any,
    result: SimulationResult,
) -> tuple[float, float, float]:
    if result.final_state is not None:
        active = np.asarray(result.final_state["active"], dtype=np.int32) != 0
        phase = np.asarray(
            result.final_state.get("phase", simulation._particle_phase_codes),
            dtype=np.int16,
        )
    else:
        active = np.zeros(0, dtype=bool)
        phase = np.zeros(0, dtype=np.int16)
    active_real = (
        float(np.sum(simulation._particle_weight[active])) if active.size else 0.0
    )
    parent = np.asarray(
        getattr(simulation, "_particle_parent_weight", simulation._particle_weight),
        dtype=float,
    )
    weight = np.asarray(simulation._particle_weight, dtype=float)
    if active.size and parent.shape[0] == active.shape[0] and weight.shape[0] == active.shape[0]:
        parent_active = np.minimum(
            np.maximum(parent[active], 0.0),
            np.maximum(weight[active], 0.0),
        )
        active_parent = float(np.sum(parent_active))
    else:
        active_parent = active_real
    capillary = phase == SLOT_CAPILLARY
    capillary_real = (
        float(np.sum(simulation._particle_weight[capillary])) if phase.size else 0.0
    )
    return active_real, active_parent, capillary_real


def _named_status_values(
    values: dict[int, float | int],
    cast: type[float] | type[int],
) -> dict[str, float] | dict[str, int]:
    return {
        PARTICLE_STATUS_NAMES.get(code, str(code)): cast(value)
        for code, value in values.items()
    }


def _terminal_accounting(
    simulation: Any,
    charge_per_ion_c: float,
    elapsed_s: float,
) -> tuple[dict[str, Any], ...]:
    real_ions = _named_status_values(
        simulation._source_terminal_real_ions_by_code,
        float,
    )
    macro_particles = _named_status_values(
        simulation._source_terminal_macro_particles_by_code,
        int,
    )
    parent_real_ions = _named_status_values(
        getattr(simulation, "_source_terminal_parent_real_ions_by_code", {}),
        float,
    )
    fragment_real_ions = _named_status_values(
        getattr(simulation, "_source_terminal_fragment_real_ions_by_code", {}),
        float,
    )
    current = {
        status: float(value * charge_per_ion_c / elapsed_s)
        for status, value in real_ions.items()
    }
    parent_current = {
        status: float(value * charge_per_ion_c / elapsed_s)
        for status, value in parent_real_ions.items()
    }
    fragment_current = {
        status: float(value * charge_per_ion_c / elapsed_s)
        for status, value in fragment_real_ions.items()
    }
    return (
        real_ions,
        macro_particles,
        parent_real_ions,
        fragment_real_ions,
        current,
        parent_current,
        fragment_current,
    )


def _source_values(simulation: Any, result: SimulationResult) -> _SourceValues:
    temperature_k = simulation._source_temperature_for_mach_k()
    velocity_m_per_s = simulation._source_axial_velocity_m_per_s()
    exit_speed_m_per_s = simulation._source_exit_speed_m_per_s()
    active_real, active_parent, capillary_real = _active_inventory(
        simulation,
        result,
    )
    charge_per_ion_c = simulation._source_charge_per_ion_c()
    elapsed_s = max(float(result.final_time_s), 1.0e-30)
    terminal = _terminal_accounting(simulation, charge_per_ion_c, elapsed_s)
    boundary_current = (
        simulation._source_boundary_injected_real_ions
        * charge_per_ion_c
        / elapsed_s
    )
    emitted_current = (
        simulation._source_emitted_real_ions * charge_per_ion_c / elapsed_s
    )
    number_density_m3 = simulation._capillary_number_density_m3()
    z_exit_real = float(terminal[0].get("z_exit", 0.0))
    eta_exit = (
        float(z_exit_real / simulation._source_injected_real_ions)
        if simulation._source_injected_real_ions > 0.0
        else float("nan")
    )
    return _SourceValues(
        temperature_k,
        velocity_m_per_s,
        exit_speed_m_per_s,
        active_real,
        active_parent,
        capillary_real,
        charge_per_ion_c,
        elapsed_s,
        number_density_m3,
        *terminal,
        boundary_current,
        emitted_current,
        z_exit_real,
        eta_exit,
    )


def _source_physics_payload(simulation: Any, values: _SourceValues) -> dict[str, Any]:
    config = simulation.config
    number_density = values.capillary_number_density_m3
    return {
        "source_mode": config.source_mode,
        "collision_operator": str(config.collision_model),
        "ion_current_a": float(config.ion_current_a),
        "ion_emission_rate_per_s": float(simulation._source_ion_emission_rate_per_s()),
        "source_mach_number": float(config.source_mach_number),
        "source_gas_gamma": float(config.source_gas_gamma),
        "source_temperature_k": float(values.source_temperature_k),
        "source_axial_velocity_m_per_s": float(values.source_velocity_m_per_s),
        "source_profile": str(config.source_profile),
        "source_gaussian_sigma_m": (
            None
            if config.source_gaussian_sigma_m is None
            else float(config.source_gaussian_sigma_m)
        ),
        "source_velocity_from_gas_field": bool(config.source_velocity_from_gas_field),
        "source_birth_velocity": getattr(
            simulation,
            "source_birth_velocity_metadata",
            {"loaded": False},
        ),
        "source_radial_velocity_scale": float(config.source_radial_velocity_scale),
        "source_velocity_delta_m_per_s": float(config.source_velocity_delta_m_per_s),
        "source_exit_speed_m_per_s": float(values.source_exit_speed_m_per_s),
        "source_exit_kinetic_energy_ev": float(
            simulation._source_exit_kinetic_energy_j() / elementary_charge
        ),
        "capillary_exit_z_m": float(simulation._capillary_exit_z_m()),
        "capillary_inlet_z_m": float(simulation._capillary_inlet_z_m()),
        "capillary_voltage_v": float(config.capillary_voltage_v),
        "capillary_prefill_length_m": float(config.capillary_prefill_length_m),
        "capillary_number_density_m3": float(number_density),
        "capillary_charge_density_c_per_m3": (
            float(number_density * values.charge_per_ion_c)
            if np.isfinite(number_density)
            else float("nan")
        ),
    }


def _source_particle_accounting_payload(
    simulation: Any,
    values: _SourceValues,
) -> dict[str, Any]:
    config = simulation.config
    return {
        "macro_particles_per_injection": int(config.macro_particles_per_injection),
        "max_macro_particle_weight": float(config.max_macro_particle_weight),
        "source_target_macro_weight": float(simulation._source_target_macro_weight()),
        "source_pending_real_ion_accumulator": float(
            simulation._source_real_ion_accumulator
        ),
        "prefill_real_ions": float(simulation._source_prefill_real_ions),
        "prefill_macro_particles": int(simulation._source_prefill_macro_particles),
        "boundary_injected_real_ions": float(
            simulation._source_boundary_injected_real_ions
        ),
        "boundary_injected_macro_particles": int(
            simulation._source_boundary_injected_macro_particles
        ),
        "emitted_real_ions": float(simulation._source_emitted_real_ions),
        "emitted_macro_particles": int(simulation._source_emitted_macro_particles),
        "injected_real_ions": float(simulation._source_injected_real_ions),
        "injected_macro_particles": int(simulation._source_injected_macro_particles),
        "blocked_real_ions": float(simulation._source_blocked_real_ions),
        "active_real_ions": values.active_real_ions,
        "active_parent_real_ions": values.active_parent_real_ions,
        "active_fragment_real_ions": float(
            max(values.active_real_ions - values.active_parent_real_ions, 0.0)
        ),
        "active_charge_c": float(values.active_real_ions * values.charge_per_ion_c),
        "capillary_buffer_real_ions": values.capillary_real_ions,
        "capillary_buffer_charge_c": float(
            values.capillary_real_ions * values.charge_per_ion_c
        ),
        "represented_boundary_current_a": float(values.boundary_injected_current_a),
        "represented_emitted_current_a": float(values.emitted_current_a),
        "represented_injected_current_a": float(values.boundary_injected_current_a),
    }


def _source_terminal_accounting_payload(
    values: _SourceValues,
) -> dict[str, Any]:
    return {
        "terminal_real_ions_by_status": values.terminal_real_ions,
        "terminal_macro_particles_by_status": values.terminal_macro_particles,
        "terminal_parent_real_ions_by_status": values.terminal_parent_real_ions,
        "terminal_fragment_real_ions_by_status": values.terminal_fragment_real_ions,
        "terminal_current_a_by_status": values.terminal_current_a,
        "terminal_parent_current_a_by_status": values.terminal_parent_current_a,
        "terminal_fragment_current_a_by_status": values.terminal_fragment_current_a,
        "eta_exit": values.eta_exit,
        "n_exit_real_ions": values.z_exit_real_ions,
        "i_exit_cycle_a": float(
            values.z_exit_real_ions * values.charge_per_ion_c / values.elapsed_s
        ),
        "current_transmission": (
            float(
                values.terminal_current_a.get("z_exit", 0.0)
                / values.emitted_current_a
            )
            if values.emitted_current_a > 0.0
            else float("nan")
        ),
    }


def _source_runtime_summary(
    simulation: Any,
    result: SimulationResult,
) -> dict[str, Any]:
    """Build source and terminal-current accounting for the final report."""

    values = _source_values(simulation, result)
    summary = _source_physics_payload(simulation, values)
    summary.update(_source_particle_accounting_payload(simulation, values))
    summary.update(_source_terminal_accounting_payload(values))
    return summary


def _performance_summary(result: SimulationResult) -> dict[str, Any]:
    run_wall_time_s = float(result.run_wall_time_s)
    simulated_time_s = float(result.final_time_s)
    simulated_us = simulated_time_s * 1.0e6
    breakdown = {
        str(key): float(value)
        for key, value in result.performance_breakdown.items()
    }
    dt_statistics: dict[str, float | int | bool] = {}
    for key, value in result.dt_statistics.items():
        normalized_key = str(key)
        if isinstance(value, (bool, np.bool_)):
            dt_statistics[normalized_key] = bool(value)
        elif normalized_key.endswith(("_count", "_size")):
            dt_statistics[normalized_key] = int(value)
        else:
            dt_statistics[normalized_key] = float(value)
    dt_limiter_counts = {
        str(key): int(value) for key, value in result.dt_limiter_counts.items()
    }
    dt_raw_limiter_counts = {
        str(key): int(value) for key, value in result.dt_raw_limiter_counts.items()
    }
    macro_summary = getattr(result, "macro_history_summary", {}) or {}
    macro_step_count = int(
        macro_summary.get("macro_step_count", len(result.macro_history))
    )
    return {
        "run_started_at": result.run_started_at,
        "run_ended_at": result.run_ended_at,
        "run_wall_time_s": run_wall_time_s,
        "simulated_time_s": simulated_time_s,
        "simulated_us": simulated_us,
        "macro_step_count": macro_step_count,
        "micro_step_count": int(result.micro_step_count),
        "simulated_us_per_wall_s": (
            simulated_us / run_wall_time_s
            if run_wall_time_s > 0.0
            else float("nan")
        ),
        "wall_s_per_simulated_us": (
            run_wall_time_s / simulated_us
            if simulated_us > 0.0
            else float("nan")
        ),
        "breakdown": breakdown,
        "dt_statistics": dt_statistics,
        "dt_limiter_counts": dt_limiter_counts,
        "dt_raw_limiter_counts": dt_raw_limiter_counts,
    }
