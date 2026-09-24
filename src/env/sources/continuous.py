"""Continuous ion-current source model and capillary phase-space sampling."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.constants import elementary_charge

from ...config import IonTemplate, SimulationConfig, UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K


TemperatureSampler = Callable[[str, float, float], float]
FieldSampler = Callable[[str, float, float], float]
BirthVelocitySampler = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def sample_source_xy_offsets(
    rng: np.random.Generator,
    count: int,
    *,
    source_radius_m: Optional[float],
    source_profile: str,
    source_gaussian_sigma_m: Optional[float],
    gaussian_max_attempts: int = 1_000,
    gaussian_min_batch: int = 1_024,
    gaussian_oversample_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample source-plane x/y offsets for supported transverse profiles."""

    if count <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    profile = str(source_profile).strip().lower()
    radius_limit_m = 0.0 if source_radius_m is None else float(source_radius_m)
    if profile == "uniform-disk":
        if radius_limit_m <= 0.0:
            return np.zeros(count, dtype=np.float64), np.zeros(count, dtype=np.float64)
        radius_m = radius_limit_m * np.sqrt(rng.random(count))
        azimuth_rad = 2.0 * np.pi * rng.random(count)
        return radius_m * np.cos(azimuth_rad), radius_m * np.sin(azimuth_rad)

    if profile != "gaussian":
        raise ValueError(f"Unsupported source_profile: {source_profile!r}")
    sigma_m = 0.0 if source_gaussian_sigma_m is None else float(source_gaussian_sigma_m)
    if sigma_m <= 0.0:
        raise ValueError("source_gaussian_sigma_m must be positive when source_profile='gaussian'.")
    if radius_limit_m <= 0.0:
        raise ValueError("source_radius_m must be positive to truncate a gaussian source profile.")
    return _sample_truncated_gaussian(
        rng,
        count,
        sigma_m,
        radius_limit_m,
        max_attempts=gaussian_max_attempts,
        min_batch=gaussian_min_batch,
        oversample_factor=gaussian_oversample_factor,
    )


def _sample_truncated_gaussian(
    rng: np.random.Generator,
    count: int,
    sigma_m: float,
    radius_limit_m: float,
    *,
    max_attempts: int = 1_000,
    min_batch: int = 1_024,
    oversample_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    x_batches: list[np.ndarray] = []
    y_batches: list[np.ndarray] = []
    accepted = 0
    attempts = 0
    while accepted < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Could not sample truncated gaussian source profile. "
                "Check that --source-radius-mm is not too small relative to --source-gaussian-sigma-mm."
            )
        batch_count = max(
            int(min_batch),
            int(oversample_factor) * (count - accepted),
        )
        x_trial = rng.normal(0.0, sigma_m, size=batch_count)
        y_trial = rng.normal(0.0, sigma_m, size=batch_count)
        keep = np.hypot(x_trial, y_trial) <= radius_limit_m
        if not np.any(keep):
            continue
        x_keep = x_trial[keep]
        y_keep = y_trial[keep]
        x_batches.append(x_keep)
        y_batches.append(y_keep)
        accepted += int(x_keep.size)

    return np.concatenate(x_batches)[:count], np.concatenate(y_batches)[:count]


class ContinuousCurrentSource:
    """Continuous source helper for current-to-macro-particle injection.

    This class owns only source-side calculations and random phase-space
    sampling. Slot allocation and Taichi uploads stay in the simulation runtime.
    """

    def __init__(
        self,
        template: IonTemplate,
        config: SimulationConfig,
        rng: np.random.Generator,
        *,
        domain_length_m: float,
        temperature_sampler: TemperatureSampler,
        field_sampler: FieldSampler,
        birth_velocity_sampler: Optional[BirthVelocitySampler] = None,
        birth_axial_velocity_m_per_s: Optional[float] = None,
    ) -> None:
        self.template = template
        self.config = config
        self.rng = rng
        self.domain_length_m = float(domain_length_m)
        self.temperature_sampler = temperature_sampler
        self.field_sampler = field_sampler
        self.birth_velocity_sampler = birth_velocity_sampler
        self.birth_axial_velocity_m_per_s = birth_axial_velocity_m_per_s

    def enabled(self) -> bool:
        return self.config.source_mode == "continuous-current" and self.config.ion_current_a > 0.0

    def charge_per_ion_c(self) -> float:
        return max(abs(float(self.template.charge_state) * elementary_charge), 1.0e-30)

    def temperature_for_mach_k(self) -> float:
        if self.config.source_temperature_k is not None:
            return max(float(self.config.source_temperature_k), 1.0)
        source_position = np.asarray(self.template.initial_position_m, dtype=float)
        source_r_m = float(np.linalg.norm(source_position[:2]))
        source_z_m = float(np.clip(source_position[2], 0.0, self.domain_length_m))
        try:
            return max(float(self.temperature_sampler("T_gas", source_r_m, source_z_m)), 1.0)
        except Exception:
            return float(self.template.initial_internal_temperature_k)

    def axial_velocity_m_per_s(self) -> float:
        if self.config.source_axial_velocity_m_per_s is not None:
            return max(float(self.config.source_axial_velocity_m_per_s), 0.0)
        if self.config.source_velocity_from_gas_field:
            if self.birth_axial_velocity_m_per_s is not None:
                return max(
                    float(self.birth_axial_velocity_m_per_s) + float(self.config.source_velocity_delta_m_per_s),
                    0.0,
                )
            z_exit_m = self.capillary_exit_z_m()
            try:
                gas_vz_m_per_s = float(self.field_sampler("v_gas_z", 0.0, z_exit_m))
            except Exception:
                gas_vz_m_per_s = 0.0
            return max(gas_vz_m_per_s + float(self.config.source_velocity_delta_m_per_s), 0.0)
        gamma = max(float(self.config.source_gas_gamma), 1.0e-12)
        molar_mass = max(float(self.config.source_gas_molar_mass_kg_per_mol), 1.0e-30)
        temperature_k = self.temperature_for_mach_k()
        sound_speed_m_per_s = np.sqrt(gamma * UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K * temperature_k / molar_mass)
        return float(max(self.config.source_mach_number, 0.0) * sound_speed_m_per_s)

    def capillary_exit_z_m(self) -> float:
        if self.config.capillary_exit_z_m is not None:
            return float(np.clip(self.config.capillary_exit_z_m, 0.0, self.domain_length_m))
        return float(np.clip(self.template.initial_position_m[2], 0.0, self.domain_length_m))

    def capillary_inlet_z_m(self) -> float:
        return float(
            max(
                0.0,
                self.capillary_exit_z_m() - max(float(self.config.capillary_prefill_length_m), 0.0),
            )
        )

    def ion_emission_rate_per_s(self) -> float:
        return float(max(self.config.ion_current_a, 0.0) / self.charge_per_ion_c())

    def exit_kinetic_energy_j(self) -> float:
        transport_speed = max(self.axial_velocity_m_per_s(), 0.0)
        transport_ke_j = 0.5 * self.template.mass_kg * transport_speed * transport_speed
        electrostatic_ke_j = self.charge_per_ion_c() * float(self.config.capillary_voltage_v)
        return float(max(transport_ke_j + electrostatic_ke_j, 0.0))

    def exit_speed_m_per_s(self) -> float:
        kinetic_energy_j = self.exit_kinetic_energy_j()
        if kinetic_energy_j <= 0.0:
            return 0.0
        return float(np.sqrt(2.0 * kinetic_energy_j / max(self.template.mass_kg, 1.0e-30)))

    def _sample_local_gas_velocity_vectors(self, positions_m: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_m, dtype=np.float64)
        velocities = np.zeros((positions.shape[0], 3), dtype=np.float64)
        if positions.size == 0:
            return velocities
        if self.birth_velocity_sampler is not None:
            velocities = np.asarray(self.birth_velocity_sampler(positions, self.rng), dtype=np.float64)
            if velocities.shape != (positions.shape[0], 3):
                raise ValueError("birth_velocity_sampler must return shape (n_particles, 3).")
            radial_scale = max(float(self.config.source_radial_velocity_scale), 0.0)
            velocities[:, 0] *= radial_scale
            velocities[:, 1] *= radial_scale
            velocities[:, 2] += float(self.config.source_velocity_delta_m_per_s)
            return velocities
        for index, position_m in enumerate(positions):
            x_m = float(position_m[0])
            y_m = float(position_m[1])
            z_m = float(np.clip(position_m[2], 0.0, self.domain_length_m))
            r_m = float(np.sqrt(x_m * x_m + y_m * y_m))
            try:
                gas_vr_m_per_s = float(self.field_sampler("v_gas_r", r_m, z_m))
                gas_vz_m_per_s = float(self.field_sampler("v_gas_z", r_m, z_m))
            except Exception:
                gas_vr_m_per_s = 0.0
                gas_vz_m_per_s = 0.0
            if r_m > 1.0e-16:
                radial_scale = max(float(self.config.source_radial_velocity_scale), 0.0)
                velocities[index, 0] = radial_scale * gas_vr_m_per_s * x_m / r_m
                velocities[index, 1] = radial_scale * gas_vr_m_per_s * y_m / r_m
            velocities[index, 2] = gas_vz_m_per_s + float(self.config.source_velocity_delta_m_per_s)
        return velocities

    def _apply_capillary_voltage_to_axial_velocity(self, velocities_m_per_s: np.ndarray) -> np.ndarray:
        velocities = np.asarray(velocities_m_per_s, dtype=np.float64).copy()
        if velocities.size == 0:
            return velocities
        electrostatic_ke_j = self.charge_per_ion_c() * float(self.config.capillary_voltage_v)
        if electrostatic_ke_j == 0.0:
            return velocities
        mass_kg = max(float(self.template.mass_kg), 1.0e-30)
        axial_energy_after_j = 0.5 * mass_kg * velocities[:, 2] * velocities[:, 2] + electrostatic_ke_j
        axial_speed_after = np.sqrt(np.maximum(2.0 * axial_energy_after_j / mass_kg, 0.0))
        if electrostatic_ke_j > 0.0:
            velocities[:, 2] = axial_speed_after
        else:
            axial_sign = np.where(velocities[:, 2] < 0.0, -1.0, 1.0)
            velocities[:, 2] = axial_sign * axial_speed_after
        return velocities

    def capillary_number_density_m3(self) -> float:
        source_radius_m = float(self.config.source_radius_m or 0.0)
        if source_radius_m <= 0.0:
            return float("nan")
        cross_section_area_m2 = np.pi * source_radius_m * source_radius_m
        transport_speed = max(self.axial_velocity_m_per_s(), 1.0e-30)
        return float(self.config.ion_current_a / (self.charge_per_ion_c() * cross_section_area_m2 * transport_speed))

    def sample_source_disk_positions(self, count: int, z_values_m: np.ndarray) -> np.ndarray:
        base_position = np.asarray(self.template.initial_position_m, dtype=float)
        positions = np.tile(base_position, (count, 1))
        x_offsets_m, y_offsets_m = sample_source_xy_offsets(
            self.rng,
            count,
            source_radius_m=self.config.source_radius_m,
            source_profile=self.config.source_profile,
            source_gaussian_sigma_m=self.config.source_gaussian_sigma_m,
            gaussian_max_attempts=getattr(
                self.config,
                "source_gaussian_max_attempts",
                1_000,
            ),
            gaussian_min_batch=getattr(
                self.config,
                "source_gaussian_min_batch",
                1_024,
            ),
            gaussian_oversample_factor=(
                getattr(
                    self.config,
                    "source_gaussian_oversample_factor",
                    2,
                )
            ),
        )
        positions[:, 0] = base_position[0] + x_offsets_m
        positions[:, 1] = base_position[1] + y_offsets_m
        positions[:, 2] = np.clip(np.asarray(z_values_m, dtype=np.float64), 0.0, self.domain_length_m)
        return np.asarray(positions, dtype=np.float64)

    def sample_capillary_phase_space(self, count: int, *, prefill: bool) -> tuple[np.ndarray, np.ndarray]:
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        z_exit_m = self.capillary_exit_z_m()
        z_inlet_m = self.capillary_inlet_z_m()
        if prefill:
            z_values_m = self.rng.uniform(z_inlet_m, z_exit_m, size=count)
        else:
            z_values_m = np.full(count, z_inlet_m, dtype=np.float64)
        positions = self.sample_source_disk_positions(count, z_values_m)
        velocities = np.zeros((count, 3), dtype=np.float64)
        velocities[:, 2] = self.axial_velocity_m_per_s()
        return np.asarray(positions, dtype=np.float64), velocities

    def sample_external_emission_velocities(self, positions_m: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_m, dtype=np.float64)
        count = int(positions.shape[0])
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float64)
        if self.config.source_velocity_from_gas_field:
            velocities = self._sample_local_gas_velocity_vectors(positions)
            velocities = self._apply_capillary_voltage_to_axial_velocity(velocities)
        else:
            exit_speed_m_per_s = self.exit_speed_m_per_s()
            velocities = np.zeros((count, 3), dtype=np.float64)
            velocities[:, 2] = exit_speed_m_per_s
        if (
            not self.config.source_velocity_from_gas_field
            and self.config.cone_half_angle_rad > 0.0
            and self.exit_speed_m_per_s() > 0.0
        ):
            exit_speed_m_per_s = self.exit_speed_m_per_s()
            cos_theta = 1.0 - self.rng.random(count) * (1.0 - np.cos(self.config.cone_half_angle_rad))
            sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
            azimuth_rad = 2.0 * np.pi * self.rng.random(count)
            velocities = exit_speed_m_per_s * np.column_stack(
                (
                    sin_theta * np.cos(azimuth_rad),
                    sin_theta * np.sin(azimuth_rad),
                    cos_theta,
                )
            )
        if self.config.initial_velocity_jitter_m_per_s > 0.0:
            velocities += self.rng.normal(0.0, self.config.initial_velocity_jitter_m_per_s, size=(count, 3))
        return np.asarray(velocities, dtype=np.float64)
