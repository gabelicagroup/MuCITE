"""Reduced-order layered gas dynamics for ESI-MS interface simulations.

The goal of this module is not to replace CFD. It provides a fast analytic field
that captures the main trends we need for trajectory work:

- isentropic free-jet expansion in the zone of silence,
- a smoothed Mach-disk transition,
- post-shock pressure and temperature recovery,
- a three-dimensional bulk velocity direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
from scipy.constants import Avogadro, Boltzmann, gas_constant


VectorLike = Union[Iterable[float], np.ndarray]


def _normalized_direction(direction: VectorLike) -> np.ndarray:
    axis = np.asarray(direction, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("axial_direction must be non-zero.")
    return axis / axis_norm


@dataclass(frozen=True)
class GasState:
    """Local gas properties sampled at a point in space."""

    pressure_pa: float
    temperature_k: float
    velocity_m_per_s: np.ndarray
    number_density_m3: float
    mass_density_kg_per_m3: float
    mach_number: float


class GasFlowField:
    """Analytic free-jet gas model with a smoothed Mach-disk transition.

    Notes
    -----
    The centerline Mach correlation is implemented as an Ashkenas-Sherman style
    power-law fit:

    ``M(x) = 1 + a * ((x / D) - x_offset) ** b``

    for ``x > D``. The coefficients are intentionally configurable because they
    should be calibrated against the actual capillary/nozzle geometry.
    """

    def __init__(
        self,
        *,
        stagnation_pressure_pa: float = 1.2e5,
        background_pressure_pa: float = 300.0,
        stagnation_temperature_k: float = 300.0,
        ambient_temperature_k: float = 300.0,
        capillary_diameter_m: float = 5.0e-4,
        specific_heat_ratio: float = 1.4,
        molar_mass_kg_per_mol: float = 0.0280134,
        nozzle_origin_m: VectorLike = (0.0, 0.0, 0.0),
        axial_direction: VectorLike = (0.0, 0.0, 1.0),
        ashkenas_a: float = 3.26,
        ashkenas_b: float = 0.67,
        ashkenas_x_offset: float = 0.075,
        shock_position_m: Optional[float] = None,
        shock_smoothing_m: Optional[float] = None,
        radial_spread_rate: float = 0.25,
        radial_decay_strength: float = 2.0,
    ) -> None:
        self.stagnation_pressure_pa = float(stagnation_pressure_pa)
        self.background_pressure_pa = float(background_pressure_pa)
        self.stagnation_temperature_k = float(stagnation_temperature_k)
        self.ambient_temperature_k = float(ambient_temperature_k)
        self.capillary_diameter_m = float(capillary_diameter_m)
        self.gamma = float(specific_heat_ratio)
        self.molar_mass_kg_per_mol = float(molar_mass_kg_per_mol)
        self.nozzle_origin_m = np.asarray(nozzle_origin_m, dtype=float)
        self.axial_direction = _normalized_direction(axial_direction)
        self.ashkenas_a = float(ashkenas_a)
        self.ashkenas_b = float(ashkenas_b)
        self.ashkenas_x_offset = float(ashkenas_x_offset)
        self.radial_spread_rate = float(radial_spread_rate)
        self.radial_decay_strength = float(radial_decay_strength)

        if shock_position_m is None:
            pressure_ratio = max(self.stagnation_pressure_pa / max(self.background_pressure_pa, 1.0), 1.0)
            self.shock_position_m = 0.67 * self.capillary_diameter_m * np.sqrt(pressure_ratio)
        else:
            self.shock_position_m = float(shock_position_m)

        if shock_smoothing_m is None:
            self.shock_smoothing_m = max(0.25 * self.capillary_diameter_m, 1.0e-6)
        else:
            self.shock_smoothing_m = float(shock_smoothing_m)

    @property
    def molecular_mass_kg(self) -> float:
        """Mass of one gas molecule."""

        return self.molar_mass_kg_per_mol / Avogadro

    @property
    def specific_gas_constant(self) -> float:
        """Specific gas constant ``R / M`` in SI units."""

        return gas_constant / self.molar_mass_kg_per_mol

    def _sigmoid(self, argument: float) -> float:
        argument = float(np.clip(argument, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-argument))

    def _decompose_position(self, position_m: VectorLike) -> tuple[float, np.ndarray, float]:
        position = np.asarray(position_m, dtype=float)
        relative = position - self.nozzle_origin_m
        axial_distance = float(np.dot(relative, self.axial_direction))
        radial_vector = relative - axial_distance * self.axial_direction
        radial_distance = float(np.linalg.norm(radial_vector))
        return axial_distance, radial_vector, radial_distance

    def _mach_zone_of_silence(self, axial_distance_m: float) -> float:
        x = max(axial_distance_m, self.capillary_diameter_m)
        scaled_x = max(x / self.capillary_diameter_m - self.ashkenas_x_offset, 0.0)
        return max(1.0, 1.0 + self.ashkenas_a * scaled_x ** self.ashkenas_b)

    def _isentropic_state(self, mach_number: float) -> tuple[float, float, float]:
        temperature_ratio = 1.0 / (1.0 + 0.5 * (self.gamma - 1.0) * mach_number**2)
        pressure_ratio = temperature_ratio ** (self.gamma / (self.gamma - 1.0))
        density_ratio = temperature_ratio ** (1.0 / (self.gamma - 1.0))
        return pressure_ratio, temperature_ratio, density_ratio

    def _normal_shock_jump(self, mach_upstream: float) -> tuple[float, float, float, float]:
        m1 = max(float(mach_upstream), 1.0 + 1.0e-8)
        g = self.gamma

        m2_sq = (1.0 + 0.5 * (g - 1.0) * m1**2) / (g * m1**2 - 0.5 * (g - 1.0))
        m2 = np.sqrt(max(m2_sq, 1.0e-12))
        pressure_ratio = 1.0 + 2.0 * g / (g + 1.0) * (m1**2 - 1.0)
        density_ratio = ((g + 1.0) * m1**2) / ((g - 1.0) * m1**2 + 2.0)
        temperature_ratio = pressure_ratio / density_ratio
        return m2, pressure_ratio, temperature_ratio, density_ratio

    def mach_number(self, position_m: VectorLike) -> float:
        """Return the local Mach number with sigmoid-smoothed shock blending."""

        axial_distance, _, _ = self._decompose_position(position_m)
        pre_shock_mach = self._mach_zone_of_silence(axial_distance)

        upstream_at_shock = self._mach_zone_of_silence(self.shock_position_m)
        post_shock_mach, _, _, _ = self._normal_shock_jump(upstream_at_shock)
        shock_weight = self._sigmoid((axial_distance - self.shock_position_m) / self.shock_smoothing_m)
        return (1.0 - shock_weight) * pre_shock_mach + shock_weight * post_shock_mach

    def _free_jet_state(
        self,
        axial_distance_m: float,
    ) -> tuple[float, float, float, float]:
        mach = self._mach_zone_of_silence(axial_distance_m)
        pressure_ratio, temperature_ratio, density_ratio = (
            self._isentropic_state(mach)
        )
        pressure = self.stagnation_pressure_pa * pressure_ratio
        temperature = self.stagnation_temperature_k * temperature_ratio
        density = (
            self.stagnation_pressure_pa * density_ratio
            / (self.specific_gas_constant * self.stagnation_temperature_k)
        )
        return mach, pressure, temperature, density

    def _post_shock_state(self) -> tuple[float, float, float, float]:
        upstream_mach = self._mach_zone_of_silence(self.shock_position_m)
        mach, pressure_jump, temperature_jump, density_jump = (
            self._normal_shock_jump(upstream_mach)
        )
        pressure_ratio, temperature_ratio, density_ratio = (
            self._isentropic_state(upstream_mach)
        )
        pressure = (
            self.stagnation_pressure_pa * pressure_ratio * pressure_jump
        )
        temperature = (
            self.stagnation_temperature_k
            * temperature_ratio * temperature_jump
        )
        density = (
            self.stagnation_pressure_pa * density_ratio * density_jump
            / (self.specific_gas_constant * self.stagnation_temperature_k)
        )
        return mach, pressure, temperature, density

    def _radial_decay(
        self,
        axial_distance_m: float,
        radial_distance_m: float,
    ) -> float:
        stream_radius = (
            0.5 * self.capillary_diameter_m
            + self.radial_spread_rate * max(axial_distance_m, 0.0)
        )
        stream_radius = max(stream_radius, 0.5 * self.capillary_diameter_m)
        return float(np.exp(
            -self.radial_decay_strength
            * (radial_distance_m / stream_radius) ** 2
        ))

    def _flow_direction(
        self,
        radial_vector: np.ndarray,
        radial_distance_m: float,
    ) -> np.ndarray:
        radial_unit = (
            radial_vector / radial_distance_m
            if radial_distance_m > 0.0
            else np.zeros(3, dtype=float)
        )
        direction = self.axial_direction + self.radial_spread_rate * radial_unit
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0.0:
            return self.axial_direction
        return direction / direction_norm

    def sample(self, position_m: VectorLike) -> GasState:
        """Return local pressure, temperature, flow velocity and density."""

        axial_distance, radial_vector, radial_distance = self._decompose_position(position_m)
        axial_distance = max(axial_distance, 0.0)
        pre_mach, pre_pressure, pre_temperature, pre_density = (
            self._free_jet_state(axial_distance)
        )
        shock_mach, shock_pressure, shock_temperature, shock_density = (
            self._post_shock_state()
        )
        shock_weight = self._sigmoid((axial_distance - self.shock_position_m) / self.shock_smoothing_m)
        radial_decay = self._radial_decay(axial_distance, radial_distance)
        pressure = (1.0 - shock_weight) * pre_pressure + shock_weight * shock_pressure
        pressure = self.background_pressure_pa + radial_decay * max(pressure - self.background_pressure_pa, 0.0)
        temperature = (1.0 - shock_weight) * pre_temperature + shock_weight * shock_temperature
        temperature = self.ambient_temperature_k + radial_decay * max(temperature - self.ambient_temperature_k, 0.0)
        density = (1.0 - shock_weight) * pre_density + shock_weight * shock_density
        ambient_density = self.background_pressure_pa / (self.specific_gas_constant * self.ambient_temperature_k)
        density = ambient_density + radial_decay * max(density - ambient_density, 0.0)
        mach_number = (1.0 - shock_weight) * pre_mach + shock_weight * shock_mach
        sound_speed = np.sqrt(self.gamma * self.specific_gas_constant * max(temperature, 1.0))
        speed = mach_number * sound_speed
        velocity = speed * self._flow_direction(radial_vector, radial_distance)
        number_density = max(pressure, 0.0) / (Boltzmann * max(temperature, 1.0))
        return GasState(
            pressure_pa=float(pressure),
            temperature_k=float(temperature),
            velocity_m_per_s=np.asarray(velocity, dtype=float),
            number_density_m3=float(number_density),
            mass_density_kg_per_m3=float(density),
            mach_number=float(mach_number),
        )
