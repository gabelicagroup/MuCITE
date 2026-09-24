"""Vectorized IonSPA collision updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import Boltzmann

from ...config import IonTemplate


@dataclass(frozen=True)
class _BatchInputs:
    velocities: np.ndarray
    masses: np.ndarray
    temperatures: np.ndarray
    gas_velocities: np.ndarray
    gas_temperatures: np.ndarray
    gas_number_density: np.ndarray
    gas_mass_density: np.ndarray


def _coerce_batch_inputs(
    velocities_m_per_s: np.ndarray,
    masses_kg: np.ndarray,
    internal_temperatures_k: np.ndarray,
    gas_velocities_m_per_s: np.ndarray,
    gas_temperatures_k: np.ndarray,
    gas_number_density_m3: np.ndarray,
    gas_mass_density_kg_per_m3: np.ndarray,
) -> _BatchInputs:
    return _BatchInputs(
        np.asarray(velocities_m_per_s, dtype=np.float64),
        np.asarray(masses_kg, dtype=np.float64),
        np.asarray(internal_temperatures_k, dtype=np.float64),
        np.asarray(gas_velocities_m_per_s, dtype=np.float64),
        np.asarray(gas_temperatures_k, dtype=np.float64),
        np.asarray(gas_number_density_m3, dtype=np.float64),
        np.asarray(gas_mass_density_kg_per_m3, dtype=np.float64),
    )


def _sample_batch_velocities(
    inputs: _BatchInputs,
    gas_molecule_mass_kg: np.ndarray,
    pseudoatom_mass_kg: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    gas_std = np.sqrt(
        Boltzmann * inputs.gas_temperatures
        / np.maximum(gas_molecule_mass_kg, 1.0e-30)
    )
    atom_std = np.sqrt(
        Boltzmann * inputs.temperatures
        / np.maximum(pseudoatom_mass_kg, 1.0e-30)
    )
    sampled_gas = inputs.gas_velocities + rng.normal(
        0.0,
        gas_std[:, None],
        size=inputs.velocities.shape,
    )
    sampled_atom = rng.normal(
        0.0,
        atom_std[:, None],
        size=inputs.velocities.shape,
    )
    return sampled_gas, sampled_atom


def _batch_velocity_after(
    inputs: _BatchInputs,
    sampled_gas: np.ndarray,
    sampled_atom: np.ndarray,
    gas_molecule_mass_kg: np.ndarray,
    pseudoatom_mass_kg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dvelscale = (
        2.0 * gas_molecule_mass_kg
        / (gas_molecule_mass_kg + pseudoatom_mass_kg)
    )
    delta_term = sampled_atom - (sampled_gas - inputs.velocities)
    velocity_after = inputs.velocities + (
        pseudoatom_mass_kg / np.maximum(inputs.masses, 1.0e-30)
    )[:, None] * (sampled_atom - dvelscale[:, None] * delta_term)
    return velocity_after, delta_term


def _batch_energy_transfer(
    sampled_atom: np.ndarray,
    delta_term: np.ndarray,
    gas_molecule_mass_kg: np.ndarray,
    pseudoatom_mass_kg: np.ndarray,
) -> np.ndarray:
    pseudo_dot_delta = np.einsum("ij,ij->i", sampled_atom, delta_term)
    delta_dot_delta = np.einsum("ij,ij->i", delta_term, delta_term)
    transferred = 0.5 * pseudoatom_mass_kg * (
        -4.0
        * gas_molecule_mass_kg
        / (pseudoatom_mass_kg + gas_molecule_mass_kg)
        * pseudo_dot_delta
        + 4.0
        * gas_molecule_mass_kg**2
        / (pseudoatom_mass_kg + gas_molecule_mass_kg) ** 2
        * delta_dot_delta
    )
    return np.asarray(transferred, dtype=np.float64)


def _simulate_collision_batch(
    adapter: Any,
    inputs: _BatchInputs,
    template: IonTemplate,
    gas_name: str,
    dt_s: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ion_model = adapter._build_ion_model(template)
    u_before = np.asarray(
        ion_model.UfromT(inputs.temperatures),
        dtype=np.float64,
    )
    relative_velocity = inputs.velocities - inputs.gas_velocities
    relative_speed = np.maximum(
        np.linalg.norm(relative_velocity, axis=1),
        1.0e-9,
    )
    pseudoatom_mass_kg = adapter._pseudoatom_mass_kg_many(
        gas_name, inputs.temperatures, relative_speed,
    )
    gas_mass_kg = inputs.gas_mass_density / np.maximum(
        inputs.gas_number_density,
        1.0,
    )
    sampled_gas, sampled_atom = _sample_batch_velocities(
        inputs, gas_mass_kg, pseudoatom_mass_kg, rng,
    )
    velocity_after, delta_term = _batch_velocity_after(
        inputs, sampled_gas, sampled_atom, gas_mass_kg, pseudoatom_mass_kg,
    )
    transferred = _batch_energy_transfer(
        sampled_atom, delta_term, gas_mass_kg, pseudoatom_mass_kg,
    )
    u_after = np.maximum(u_before + transferred, 0.0)
    temperature_after = np.asarray(ion_model.TfromU(u_after), dtype=np.float64)
    increment = adapter._fracloss_many(
        dt_s, temperature_after, template.delta_h_kj_per_mol,
        template.delta_s_j_per_mol_k,
    )
    probability = 1.0 - np.exp(-np.maximum(increment, 0.0))
    return (
        np.asarray(velocity_after, dtype=np.float64),
        temperature_after,
        np.asarray(probability, dtype=np.float64),
    )


class BatchCollisionMixin:
    """Public vector collision API."""

    def apply_collision_batch(
        self,
        *,
        velocities_m_per_s: np.ndarray,
        masses_kg: np.ndarray,
        internal_temperatures_k: np.ndarray,
        gas_velocities_m_per_s: np.ndarray,
        gas_temperatures_k: np.ndarray,
        gas_number_density_m3: np.ndarray,
        gas_mass_density_kg_per_m3: np.ndarray,
        template: IonTemplate,
        gas_name: str,
        dt_s: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        velocities = np.asarray(velocities_m_per_s, dtype=np.float64)
        if velocities.size == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        inputs = _coerce_batch_inputs(
            velocities, masses_kg, internal_temperatures_k,
            gas_velocities_m_per_s, gas_temperatures_k,
            gas_number_density_m3, gas_mass_density_kg_per_m3,
        )
        return _simulate_collision_batch(
            self,
            inputs,
            template,
            gas_name,
            dt_s,
            rng,
        )


__all__ = ["BatchCollisionMixin"]
