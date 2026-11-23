"""Guarded joint IICT diffusion with sparse exact collision-event fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import Avogadro

from ..gas.kinetics import collision_frequency_s_inverse
from .joint_moments import gaussian_collision_window, gaussian_joint_step, iict_jump_moments

JOINT_LANGEVIN_VERSION = "iict-joint-frozen-diffusion-with-events-v1"


@dataclass(frozen=True)
class JointCollisionResult:
    velocities_m_per_s: np.ndarray
    temperatures_k: np.ndarray
    binary_counts: np.ndarray
    gaussian_steps: np.ndarray
    gaussian_expected_counts: np.ndarray
    guard_fallbacks: int

    @property
    def estimated_counts(self) -> np.ndarray:
        return self.binary_counts + np.rint(self.gaussian_expected_counts).astype(np.int64)


class JointIictCollisionStep:
    """Advance the collision operator at frozen position/gas, in physical time.

    Sparse events are selected by exponential waiting times, with nu refreshed
    after each event. Dense steps use IICT conditional jump moments. Gas/field
    variation along the trajectory remains limited by the outer microstep.
    """

    def __init__(
        self, *, backend: Any, template: Any, config: Any,
        velocities_m_per_s: np.ndarray, masses_kg: np.ndarray,
        temperatures_k: np.ndarray, gas_velocities_m_per_s: np.ndarray,
        gas_temperatures_k: np.ndarray, gas_number_density_m3: np.ndarray,
        dt_s: float, rng: np.random.Generator,
    ) -> None:
        if not np.isfinite(dt_s) or dt_s < 0:
            raise ValueError("Joint collision dt_s must be finite non-negative.")
        self.backend, self.template, self.config, self.rng = backend, template, config, rng
        self.v = np.asarray(velocities_m_per_s, dtype=np.float64).copy()
        self.t = np.asarray(temperatures_k, dtype=np.float64).copy()
        self.mass = np.asarray(masses_kg, dtype=np.float64)
        self.gas_v = np.asarray(gas_velocities_m_per_s, dtype=np.float64)
        self.gas_t = np.asarray(gas_temperatures_k, dtype=np.float64)
        self.density = np.asarray(gas_number_density_m3, dtype=np.float64)
        self.gas_mass = config.gas_molar_mass_kg_per_mol / Avogadro
        self.u = np.asarray(backend.energy_from_temperature(self.t), dtype=np.float64).copy()
        self.remaining = np.full(self.t.size, dt_s)
        self.binary_counts = np.zeros(self.t.size, dtype=np.int64)
        self.gaussian_steps = np.zeros(self.t.size, dtype=np.int64)
        self.expected_counts = np.zeros(self.t.size)
        self.guard_fallbacks = 0

    def _rates(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(collision_frequency_s_inverse(
            ion_speed_m_per_s=np.linalg.norm(self.v[indices] - self.gas_v[indices], axis=1),
            gas_temperature_k=self.gas_t[indices], gas_mass_kg=self.gas_mass,
            gas_number_density_m3=self.density[indices],
            collision_cross_section_m2=self.template.collision_cross_section_m2,
        ))

    def _moments(self, indices: np.ndarray):
        drift = self.v[indices] - self.gas_v[indices]
        atom_mass = self.backend.pseudoatom_mass(
            self.t[indices], np.linalg.norm(drift, axis=1), self.config.gas_species)
        return iict_jump_moments(drift, self.mass[indices], self.t[indices],
                                 self.gas_t[indices], self.gas_mass, atom_mass)

    def _advance_diffusion(self, indices: np.ndarray, rates: np.ndarray) -> np.ndarray:
        candidate = rates * self.remaining[indices] >= self.config.langevin_min_collisions
        indices, rates = indices[candidate], rates[candidate]
        if indices.size == 0:
            return indices
        moments = self._moments(indices)
        window = gaussian_collision_window(
            self.u[indices], moments,
            max_relaxation=self.config.langevin_max_relaxation,
            max_energy_fraction=self.config.langevin_max_energy_fraction,
            energy_bounds_j=self.backend.internal_energy_bounds_j,
        )
        lam = np.minimum(rates * self.remaining[indices], window)
        selected = np.flatnonzero(lam >= self.config.langevin_min_collisions)
        self.guard_fallbacks += int(indices.size - selected.size)
        indices, rates, lam = indices[selected], rates[selected], lam[selected]
        if indices.size == 0:
            return indices
        w, energy = gaussian_joint_step(self.v[indices] - self.gas_v[indices],
                                       self.u[indices], moments.take(selected), lam, self.rng)
        self.t[indices] = self.backend.temperature_from_energy(energy)
        self.u[indices], self.v[indices] = energy, self.gas_v[indices] + w
        self.remaining[indices] = np.maximum(self.remaining[indices] - lam / rates, 0)
        self.gaussian_steps[indices] += 1
        self.expected_counts[indices] += lam
        return indices

    def _advance_events(self, indices: np.ndarray, rates: np.ndarray) -> None:
        """Advance to the next actual event, or consume the no-event interval."""
        waiting = self.rng.exponential(1.0 / rates)
        hit = waiting <= self.remaining[indices]
        self.remaining[indices[~hit]] = 0
        selected = indices[hit]
        if selected.size == 0:
            return
        velocity, temperature = self.backend.apply_collision_state_batch(
            velocities_m_per_s=self.v[selected], masses_kg=self.mass[selected],
            internal_temperatures_k=self.t[selected],
            gas_velocities_m_per_s=self.gas_v[selected], gas_temperatures_k=self.gas_t[selected],
            gas_number_density_m3=self.density[selected],
            gas_mass_density_kg_per_m3=self.density[selected] * self.gas_mass,
            template=self.template, gas_name=self.config.gas_species, dt_s=0.0, rng=self.rng,
        )
        self.v[selected], self.t[selected] = velocity, temperature
        self.u[selected] = self.backend.energy_from_temperature(temperature)
        self.binary_counts[selected] += 1
        self.remaining[selected] = np.maximum(self.remaining[selected] - waiting[hit], 0)

    def run(self) -> JointCollisionResult:
        for _ in range(100_000):
            indices = np.flatnonzero(self.remaining > 0)
            if indices.size == 0:
                return JointCollisionResult(self.v, self.t, self.binary_counts,
                                            self.gaussian_steps, self.expected_counts, self.guard_fallbacks)
            rates = self._rates(indices)
            self.remaining[indices[rates == 0]] = 0
            positive = rates > 0
            indices, rates = indices[positive], rates[positive]
            if indices.size == 0:
                continue
            advanced = self._advance_diffusion(indices, rates)
            exact = ~np.isin(indices, advanced, assume_unique=True)
            self._advance_events(indices[exact], rates[exact])
        raise FloatingPointError("Joint collision subcycling exceeded 100000 iterations; reduce the microstep.")
