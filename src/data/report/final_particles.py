"""Final-particle table and transport summaries for reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import elementary_charge

from ...config import (
    PARTICLE_ACTIVE,
    PARTICLE_CAPILLARY_BUFFER,
    PARTICLE_STATUS_NAMES,
    SLOT_CAPILLARY,
    SimulationResult,
)
from .values import _array_stats, _weighted_mean, _weighted_quantile


@dataclass(frozen=True)
class _FinalArrays:
    positions: np.ndarray
    velocities: np.ndarray
    temperatures: np.ndarray
    status_codes: np.ndarray
    tof_s: np.ndarray
    collision_counts: np.ndarray
    weights: np.ndarray
    parent_weights: np.ndarray
    electrode_ids: np.ndarray
    surface_distance_m: np.ndarray
    r_m: np.ndarray
    speed_m_per_s: np.ndarray
    kinetic_energy_ev: np.ndarray
    particle_indices: np.ndarray


def _state_arrays(
    simulation: Any,
    state: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    positions = np.asarray(state["positions"], dtype=float)
    velocities = np.asarray(state["velocities"], dtype=float)
    masses = np.asarray(state["mass"], dtype=float)
    temperatures = np.asarray(state["temperature"], dtype=float)
    active = np.asarray(state["active"], dtype=np.int32) != 0
    phase = np.asarray(
        state.get("phase", simulation._particle_phase_codes),
        dtype=np.int16,
    )
    return positions, velocities, masses, temperatures, active, phase


def _status_codes(
    simulation: Any,
    positions: np.ndarray,
    active: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    count = int(positions.shape[0])
    status_codes = np.asarray(
        simulation._particle_status_codes,
        dtype=np.int16,
    ).copy()
    if status_codes.shape[0] != count:
        status_codes = np.full(count, PARTICLE_ACTIVE, dtype=np.int16)
    status_codes[active] = PARTICLE_ACTIVE
    status_codes[phase == SLOT_CAPILLARY] = PARTICLE_CAPILLARY_BUFFER
    unclassified = np.flatnonzero(~active & (status_codes == PARTICLE_ACTIVE))
    for particle_index in unclassified:
        status_codes[particle_index] = simulation._classify_domain_exit(
            positions[particle_index]
        )
    return status_codes


def _tof_values(
    simulation: Any,
    state: dict[str, Any],
    result: SimulationResult,
    status_codes: np.ndarray,
) -> np.ndarray:
    count = int(status_codes.shape[0])
    tof_s = np.asarray(simulation._particle_tof_s, dtype=float).copy()
    if tof_s.shape[0] != count:
        tof_s = np.full(count, np.nan, dtype=float)
    birth_time_s = np.asarray(simulation._particle_birth_time_s, dtype=float)
    if birth_time_s.shape[0] != count:
        birth_time_s = np.full(count, np.nan, dtype=float)
    missing_tof = ~np.isfinite(tof_s)
    valid_birth = (
        missing_tof
        & np.isfinite(birth_time_s)
        & (status_codes != PARTICLE_CAPILLARY_BUFFER)
    )
    tof_s[valid_birth] = np.maximum(
        float(result.final_time_s) - birth_time_s[valid_birth],
        0.0,
    )
    tof_s[missing_tof & ~np.isfinite(birth_time_s)] = np.nan
    return tof_s


def _slot_diagnostics(
    simulation: Any,
    count: int,
) -> tuple[np.ndarray, ...]:
    collision_counts = np.asarray(
        simulation._particle_collision_counts,
        dtype=np.int64,
    )
    if collision_counts.shape[0] != count:
        collision_counts = np.zeros(count, dtype=np.int64)
    weights = np.asarray(simulation._particle_weight, dtype=float)
    if weights.shape[0] != count:
        weights = np.ones(count, dtype=float)
    parent_weights = np.asarray(
        getattr(simulation, "_particle_parent_weight", weights),
        dtype=float,
    )
    if parent_weights.shape[0] != count:
        parent_weights = weights.copy()
    parent_weights = np.minimum(
        np.maximum(parent_weights, 0.0),
        np.maximum(weights, 0.0),
    )
    electrode_ids = np.asarray(simulation._particle_electrode_id, dtype=np.int32)
    if electrode_ids.shape[0] != count:
        electrode_ids = np.full(count, -1, dtype=np.int32)
    surface_distance_m = np.asarray(
        simulation._particle_surface_distance_m,
        dtype=float,
    )
    if surface_distance_m.shape[0] != count:
        surface_distance_m = np.full(count, np.nan, dtype=float)
    return (
        collision_counts,
        weights,
        parent_weights,
        electrode_ids,
        surface_distance_m,
    )


def _reported_particle_indices(simulation: Any, count: int) -> np.ndarray:
    if simulation.config.source_mode == "continuous-current":
        return np.flatnonzero(
            np.asarray(simulation._particle_ever_used, dtype=bool)
        )
    return np.arange(count, dtype=np.int64)


def _final_arrays(
    simulation: Any,
    result: SimulationResult,
) -> _FinalArrays:
    state = result.final_state
    positions, velocities, masses, temperatures, active, phase = _state_arrays(
        simulation,
        state,
    )
    count = int(positions.shape[0])
    status_codes = _status_codes(simulation, positions, active, phase)
    tof_s = _tof_values(simulation, state, result, status_codes)
    diagnostics = _slot_diagnostics(simulation, count)
    r_m = np.linalg.norm(positions[:, :2], axis=1)
    speed_m_per_s = np.linalg.norm(velocities, axis=1)
    kinetic_energy_ev = 0.5 * masses * speed_m_per_s**2 / elementary_charge
    return _FinalArrays(
        positions,
        velocities,
        temperatures,
        status_codes,
        tof_s,
        *diagnostics,
        r_m,
        speed_m_per_s,
        kinetic_energy_ev,
        _reported_particle_indices(simulation, count),
    )


def _sample_final_gas(
    simulation: Any,
    result: SimulationResult,
    arrays: _FinalArrays,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = arrays.particle_indices
    gas_velocities = np.full((indices.size, 3), np.nan, dtype=np.float64)
    gas_speed_m_per_s = np.full(indices.size, np.nan, dtype=np.float64)
    relative_speed_m_per_s = np.full(indices.size, np.nan, dtype=np.float64)
    if indices.size:
        sampled_positions = arrays.positions[indices]
        sampled_velocities = arrays.velocities[indices]
        try:
            samples = simulation._sample_local_state_batch(
                sampled_positions,
                float(result.final_time_s),
            )
            gas_velocities = np.asarray(
                samples.gas_velocity_m_per_s,
                dtype=np.float64,
            )
            gas_speed_m_per_s = np.linalg.norm(gas_velocities, axis=1)
            relative_speed_m_per_s = np.linalg.norm(
                sampled_velocities - gas_velocities,
                axis=1,
            )
        except Exception:
            pass
    return gas_velocities, gas_speed_m_per_s, relative_speed_m_per_s


def _particle_row(
    arrays: _FinalArrays,
    particle_index: int,
    row_index: int,
    gas: tuple[np.ndarray, np.ndarray, np.ndarray],
    status: str,
) -> dict[str, Any]:
    gas_velocities, gas_speed_m_per_s, relative_speed_m_per_s = gas
    represented = arrays.weights[particle_index]
    parent = arrays.parent_weights[particle_index]
    return {
        "ion_id": int(particle_index),
        "final_status": status,
        "final_x_m": float(arrays.positions[particle_index, 0]),
        "final_y_m": float(arrays.positions[particle_index, 1]),
        "final_z_m": float(arrays.positions[particle_index, 2]),
        "final_r_m": float(arrays.r_m[particle_index]),
        "final_vx_m_per_s": float(arrays.velocities[particle_index, 0]),
        "final_vy_m_per_s": float(arrays.velocities[particle_index, 1]),
        "final_vz_m_per_s": float(arrays.velocities[particle_index, 2]),
        "final_speed_m_per_s": float(arrays.speed_m_per_s[particle_index]),
        "local_gas_vx_m_per_s": float(gas_velocities[row_index, 0]),
        "local_gas_vy_m_per_s": float(gas_velocities[row_index, 1]),
        "local_gas_vz_m_per_s": float(gas_velocities[row_index, 2]),
        "local_gas_speed_m_per_s": float(gas_speed_m_per_s[row_index]),
        "relative_speed_m_per_s": float(relative_speed_m_per_s[row_index]),
        "tof_s": float(arrays.tof_s[particle_index]),
        "represented_real_ions": float(represented),
        "parent_real_ions_remaining": float(parent),
        "fragmented_real_ions_represented": float(max(represented - parent, 0.0)),
        "collision_count_per_ion": int(arrays.collision_counts[particle_index]),
        "final_ke_ev": float(arrays.kinetic_energy_ev[particle_index]),
        "final_internal_temperature_k": float(
            arrays.temperatures[particle_index]
        ),
        "electrode_id": int(arrays.electrode_ids[particle_index]),
        "surface_distance_m": float(arrays.surface_distance_m[particle_index]),
    }


def _particle_rows(
    arrays: _FinalArrays,
    gas: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    status_counts = {name: 0 for name in PARTICLE_STATUS_NAMES.values()}
    for row_index, particle_index in enumerate(arrays.particle_indices):
        status = PARTICLE_STATUS_NAMES.get(
            int(arrays.status_codes[particle_index]),
            "domain_out",
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        rows.append(
            _particle_row(arrays, particle_index, row_index, gas, status)
        )
    return rows, status_counts


def _final_particle_table(
    simulation: Any,
    result: SimulationResult,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert final slot state into weighted per-particle CSV/report rows."""

    if result.final_state is None:
        return [], {}
    arrays = _final_arrays(simulation, result)
    gas = _sample_final_gas(simulation, result, arrays)
    return _particle_rows(arrays, gas)


def _final_particle_transport_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    active_rows = [
        row for row in rows if str(row.get("final_status", "")) == "active"
    ]
    if not active_rows:
        return {"active_count": 0}
    weights = np.asarray(
        [
            float(row.get("represented_real_ions", 0.0))
            for row in active_rows
        ],
        dtype=np.float64,
    )
    fields = {
        "particle_vz_m_per_s": "final_vz_m_per_s",
        "particle_speed_m_per_s": "final_speed_m_per_s",
        "local_gas_vz_m_per_s": "local_gas_vz_m_per_s",
        "local_gas_speed_m_per_s": "local_gas_speed_m_per_s",
        "relative_speed_m_per_s": "relative_speed_m_per_s",
    }
    summary: dict[str, Any] = {"active_count": int(len(active_rows))}
    for output_name, row_key in fields.items():
        values = np.asarray(
            [
                float(row.get(row_key, float("nan")))
                for row in active_rows
            ],
            dtype=np.float64,
        )
        summary[output_name] = {
            **_array_stats(values),
            "weighted_mean": _weighted_mean(values, weights),
            "weighted_p50": _weighted_quantile(values, weights, 0.50),
            "weighted_p95": _weighted_quantile(values, weights, 0.95),
        }
    return summary
