"""Single-ion IonSPA collision update."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.constants import Boltzmann

from ...agents.ion import IonState
from ...config import CollisionOutcome, IonTemplate
from ..gas.layered import GasState


def _single_sampled_velocities(
    ion: IonState,
    gas_state: GasState,
    gas_molecule_mass_kg: float,
    pseudoatom_mass_kg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    gas_std = np.sqrt(
        Boltzmann * gas_state.temperature_k
        / max(gas_molecule_mass_kg, 1.0e-30)
    )
    atom_std = np.sqrt(
        Boltzmann * ion.internal_temperature_k
        / max(pseudoatom_mass_kg, 1.0e-30)
    )
    sampled_gas = gas_state.velocity_m_per_s + rng.normal(
        0.0,
        gas_std,
        size=3,
    )
    sampled_atom = rng.normal(0.0, atom_std, size=3)
    return sampled_gas, sampled_atom


def _single_velocity_after(
    ion: IonState,
    sampled_gas: np.ndarray,
    sampled_atom: np.ndarray,
    gas_molecule_mass_kg: float,
    pseudoatom_mass_kg: float,
) -> tuple[np.ndarray, np.ndarray]:
    dvelscale = (
        2.0 * gas_molecule_mass_kg
        / (gas_molecule_mass_kg + pseudoatom_mass_kg)
    )
    delta_term = sampled_atom - (sampled_gas - ion.velocity_m_per_s)
    velocity_after = ion.velocity_m_per_s + pseudoatom_mass_kg / ion.mass_kg * (
        sampled_atom - dvelscale * delta_term
    )
    return velocity_after, delta_term


def _single_energy_transfer(
    sampled_atom: np.ndarray,
    delta_term: np.ndarray,
    gas_molecule_mass_kg: float,
    pseudoatom_mass_kg: float,
) -> float:
    transferred = 0.5 * pseudoatom_mass_kg * (
        -4.0
        * gas_molecule_mass_kg
        / (pseudoatom_mass_kg + gas_molecule_mass_kg)
        * np.dot(sampled_atom, delta_term)
        + 4.0
        * gas_molecule_mass_kg**2
        / (pseudoatom_mass_kg + gas_molecule_mass_kg) ** 2
        * np.dot(delta_term, delta_term)
    )
    return float(transferred)


def _simulate_single_collision(
    adapter: Any,
    ion: IonState,
    template: IonTemplate,
    gas_state: GasState,
    gas_name: str,
    dt_s: float,
    rng: np.random.Generator,
) -> CollisionOutcome:
    ion_model = adapter._build_ion_model(template)
    u_before = float(ion_model.UfromT(ion.internal_temperature_k))
    relative_velocity = ion.velocity_m_per_s - gas_state.velocity_m_per_s
    relative_speed = max(np.linalg.norm(relative_velocity), 1.0e-9)
    pseudoatom_mass_kg = adapter._pseudoatom_mass_kg(
        gas_state, gas_name, ion.internal_temperature_k, relative_speed,
    )
    gas_mass_kg = gas_state.mass_density_kg_per_m3 / max(
        gas_state.number_density_m3,
        1.0,
    )
    sampled_gas, sampled_atom = _single_sampled_velocities(
        ion, gas_state, gas_mass_kg, pseudoatom_mass_kg, rng,
    )
    velocity_after, delta_term = _single_velocity_after(
        ion, sampled_gas, sampled_atom, gas_mass_kg, pseudoatom_mass_kg,
    )
    transferred = _single_energy_transfer(
        sampled_atom, delta_term, gas_mass_kg, pseudoatom_mass_kg,
    )
    u_after = max(u_before + transferred, 0.0)
    temperature_after = float(ion_model.TfromU(u_after))
    increment = adapter._fracloss(
        dt_s, temperature_after, template.delta_h_kj_per_mol,
        template.delta_s_j_per_mol_k,
    )
    probability = float(1.0 - np.exp(-max(increment, 0.0)))
    return CollisionOutcome(
        velocity_m_per_s=np.asarray(velocity_after, dtype=float),
        internal_temperature_k=temperature_after,
        transferred_internal_energy_j=transferred,
        fragmentation_probability=probability,
    )


class SingleCollisionMixin:
    """Public scalar collision API."""

    def apply_collision(
        self,
        ion: IonState,
        template: IonTemplate,
        gas_state: GasState,
        *,
        gas_name: str,
        dt_s: float,
        rng: np.random.Generator,
    ) -> CollisionOutcome:
        return _simulate_single_collision(
            self,
            ion,
            template,
            gas_state,
            gas_name,
            dt_s,
            rng,
        )


__all__ = ["SingleCollisionMixin"]
