"""Time integration and macro-scale space-charge utilities."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Union

import numpy as np

from ...agents.ion import IonState
from ..gas.layered import GasState


VectorLike = Union[Iterable[float], np.ndarray]
ElectricFieldCallable = Callable[[np.ndarray, float], np.ndarray]
SpaceChargeSampler = Callable[[np.ndarray], np.ndarray]


class AdaptiveRK4Integrator:
    """Adaptive fourth-order Runge-Kutta integrator for ion trajectories.

    The step-size controller enforces the two requested stability limits:

    - ``dt < alpha * tau_collision``
    - ``dt < beta * T_rf``

    and adds an acceleration-based limiter so the particle does not cross too much
    space in a single micro step.
    """

    def __init__(
        self,
        *,
        rf_frequency_hz: float = 1.0e6,
        collision_safety_factor: float = 0.1,
        rf_safety_factor: float = 0.05,
        acceleration_safety_factor: float = 0.2,
        spatial_tolerance_m: float = 5.0e-5,
        min_dt_s: float = 1.0e-12,
        max_dt_s: float = 5.0e-8,
    ) -> None:
        self.rf_frequency_hz = float(rf_frequency_hz)
        self.collision_safety_factor = float(collision_safety_factor)
        self.rf_safety_factor = float(rf_safety_factor)
        self.acceleration_safety_factor = float(acceleration_safety_factor)
        self.spatial_tolerance_m = float(spatial_tolerance_m)
        self.min_dt_s = float(min_dt_s)
        self.max_dt_s = float(max_dt_s)

    def estimate_collision_time(self, ion: IonState, gas_state: GasState) -> float:
        """Estimate the mean collision time from the local neutral density."""

        relative_speed = np.linalg.norm(ion.velocity_m_per_s - gas_state.velocity_m_per_s)
        collision_rate = gas_state.number_density_m3 * ion.collision_cross_section_m2 * max(relative_speed, 1.0e-9)
        if collision_rate <= 0.0:
            return np.inf
        return 1.0 / collision_rate

    def suggest_timestep(
        self,
        ion: IonState,
        gas_state: GasState,
        total_electric_field_v_per_m: VectorLike,
        *,
        dt_upper_bound_s: float = np.inf,
    ) -> float:
        """Return an adaptive micro time step."""

        electric_field = np.asarray(total_electric_field_v_per_m, dtype=float)
        tau_collision = self.estimate_collision_time(ion, gas_state)
        rf_period = np.inf if self.rf_frequency_hz <= 0.0 else 1.0 / self.rf_frequency_hz

        acceleration = abs(ion.charge_c) * np.linalg.norm(electric_field) / max(ion.mass_kg, 1.0e-30)
        tau_accel = np.inf if acceleration <= 0.0 else np.sqrt(2.0 * self.spatial_tolerance_m / acceleration)

        dt = min(
            self.max_dt_s,
            dt_upper_bound_s,
            self.collision_safety_factor * tau_collision,
            self.rf_safety_factor * rf_period,
            self.acceleration_safety_factor * tau_accel,
        )
        dt = float(np.clip(dt, self.min_dt_s, self.max_dt_s))
        if np.isfinite(dt_upper_bound_s):
            dt = min(dt, max(float(dt_upper_bound_s), 0.0))
        return dt

    def _acceleration(
        self,
        position_m: np.ndarray,
        velocity_m_per_s: np.ndarray,
        charge_c: float,
        mass_kg: float,
        time_s: float,
        external_field: ElectricFieldCallable,
        space_charge_field: SpaceChargeSampler,
    ) -> np.ndarray:
        del velocity_m_per_s
        electric_field = external_field(position_m, time_s) + space_charge_field(position_m)
        return charge_c * electric_field / mass_kg

    def step(
        self,
        ion: IonState,
        *,
        time_s: float,
        dt_s: float,
        external_field: ElectricFieldCallable,
        space_charge_field: SpaceChargeSampler,
    ) -> IonState:
        """Advance one ion by a prescribed adaptive step."""

        dt = float(dt_s)
        x0 = np.asarray(ion.position_m, dtype=float)
        v0 = np.asarray(ion.velocity_m_per_s, dtype=float)

        def rhs(position_m: np.ndarray, velocity_m_per_s: np.ndarray, time_local_s: float) -> tuple[np.ndarray, np.ndarray]:
            return velocity_m_per_s, self._acceleration(
                position_m,
                velocity_m_per_s,
                ion.charge_c,
                ion.mass_kg,
                time_local_s,
                external_field,
                space_charge_field,
            )

        k1_x, k1_v = rhs(x0, v0, time_s)
        k2_x, k2_v = rhs(x0 + 0.5 * dt * k1_x, v0 + 0.5 * dt * k1_v, time_s + 0.5 * dt)
        k3_x, k3_v = rhs(x0 + 0.5 * dt * k2_x, v0 + 0.5 * dt * k2_v, time_s + 0.5 * dt)
        k4_x, k4_v = rhs(x0 + dt * k3_x, v0 + dt * k3_v, time_s + dt)

        x1 = x0 + dt / 6.0 * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        v1 = v0 + dt / 6.0 * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        return replace(
            ion,
            position_m=np.asarray(x1, dtype=float),
            velocity_m_per_s=np.asarray(v1, dtype=float),
        )
