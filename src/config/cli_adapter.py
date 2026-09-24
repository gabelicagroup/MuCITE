"""Conversion of parsed CLI arguments into canonical ion requests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np
from scipy.constants import elementary_charge

from .constants import ATOMIC_MASS_CONSTANT
from .factories import make_ion_template
from .models import IonTemplate


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _initial_speed_from_ke(
    mass_kg: float,
    kinetic_energy_ev: Optional[float],
    fallback_velocity: np.ndarray,
) -> np.ndarray:
    if kinetic_energy_ev is None:
        return np.asarray(fallback_velocity, dtype=float)
    kinetic_energy_ev = _positive_float(
        kinetic_energy_ev,
        "--initial-kinetic-energy-ev",
    )
    speed_m_per_s = np.sqrt(
        2.0 * kinetic_energy_ev * elementary_charge / mass_kg
    )
    return np.array([0.0, 0.0, speed_m_per_s], dtype=float)


def _validated_ion_scalars(args: Any) -> dict[str, Any]:
    mass_amu = _positive_float(args.ion_mass_amu, "--ion-mass-amu")
    charge_state = int(args.charge_state)
    if charge_state <= 0:
        raise ValueError("--charge-state must be positive.")
    cross_section = _positive_float(
        args.collision_cross_section_m2,
        "--collision-cross-section-m2",
    )
    num_atoms = int(args.num_atoms)
    if num_atoms <= 2:
        raise ValueError("--num-atoms must be greater than 2 for IonSPA heat capacity.")
    profile = str(args.heat_capacity_profile).strip()
    if not profile:
        raise ValueError("--heat-capacity-profile must not be empty.")
    delta_h = _positive_float(args.delta_h_kj_per_mol, "--delta-h-kj-per-mol")
    temperature = _positive_float(
        args.initial_internal_temperature_k,
        "--initial-internal-temperature-k",
    )
    return {
        "mass_kg": mass_amu * ATOMIC_MASS_CONSTANT,
        "charge_state": charge_state,
        "collision_cross_section_m2": cross_section,
        "num_atoms": num_atoms,
        "heat_capacity_profile": profile,
        "delta_h_kj_per_mol": delta_h,
        "initial_internal_temperature_k": temperature,
    }


def _directed_velocity(velocity: np.ndarray, args: Any) -> np.ndarray:
    direction_axis = str(
        getattr(args, "initial_direction_axis", "+z")
    ).strip().lower()
    directions = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    if direction_axis not in directions:
        raise ValueError(f"Unsupported initial direction axis: {direction_axis!r}.")
    return float(np.linalg.norm(velocity)) * directions[direction_axis]


def build_ion_template_from_args(args: Any) -> IonTemplate:
    """Convert an argparse namespace into the canonical ion template."""

    defaults = IonTemplate()
    scalars = _validated_ion_scalars(args)
    velocity = _initial_speed_from_ke(
        scalars["mass_kg"],
        args.initial_kinetic_energy_ev,
        defaults.initial_velocity_m_per_s,
    )
    position = np.array(
        [args.initial_x_mm, args.initial_y_mm, args.initial_z_mm],
        dtype=float,
    ) * 1.0e-3
    return make_ion_template(
        replace(
            defaults,
            name=str(args.ion_name),
            delta_s_j_per_mol_k=float(args.delta_s_j_per_mol_k),
            initial_position_m=position,
            initial_velocity_m_per_s=_directed_velocity(velocity, args),
            **scalars,
        )
    )
