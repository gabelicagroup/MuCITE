"""CPU-side runtime field cache and axisymmetric gather boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.constants import Avogadro, Boltzmann

from ...config import (
    LocalStateBatch,
    SimulationConfig,
    UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K,
)


@dataclass(frozen=True)
class _CylindricalSamples:
    x_m: np.ndarray
    y_m: np.ndarray
    r_m: np.ndarray
    e_external_r_v_per_m: np.ndarray
    e_external_z_v_per_m: np.ndarray
    e_sce_r_v_per_m: np.ndarray
    e_sce_z_v_per_m: np.ndarray
    gas_r_m_per_s: np.ndarray
    gas_z_m_per_s: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    rf_modulation: float


def _empty_local_state_batch() -> LocalStateBatch:
    vectors = np.zeros((0, 3), dtype=np.float64)
    scalars = np.zeros(0, dtype=np.float64)
    return LocalStateBatch(
        gas_velocity_m_per_s=vectors,
        pressure_pa=scalars,
        temperature_k=scalars,
        number_density_m3=scalars,
        mass_density_kg_per_m3=scalars,
        mach_number=scalars,
        external_field_v_per_m=vectors,
        sce_field_v_per_m=vectors,
        total_field_v_per_m=vectors,
    )


def _project_cylindrical_vectors(
    samples: _CylindricalSamples,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = samples.r_m.size
    external = np.zeros((count, 3), dtype=np.float64)
    sce = np.zeros((count, 3), dtype=np.float64)
    gas = np.zeros((count, 3), dtype=np.float64)
    external[:, 2] = samples.e_external_z_v_per_m
    sce[:, 2] = samples.e_sce_z_v_per_m
    gas[:, 2] = samples.gas_z_m_per_s
    radial_mask = samples.r_m > 1.0e-16
    if not np.any(radial_mask):
        return external, sce, gas
    inv_r = np.zeros_like(samples.r_m)
    inv_r[radial_mask] = 1.0 / samples.r_m[radial_mask]
    x_over_r = samples.x_m * inv_r
    y_over_r = samples.y_m * inv_r
    external[:, 0] = samples.e_external_r_v_per_m * x_over_r
    external[:, 1] = samples.e_external_r_v_per_m * y_over_r
    sce[:, 0] = samples.e_sce_r_v_per_m * x_over_r
    sce[:, 1] = samples.e_sce_r_v_per_m * y_over_r
    gas[:, 0] = samples.gas_r_m_per_s * x_over_r
    gas[:, 1] = samples.gas_r_m_per_s * y_over_r
    return external, sce, gas


def _gas_properties(
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    gas_velocity_m_per_s: np.ndarray,
    *,
    gas_gamma: float,
    gas_molar_mass_kg_per_mol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    number_density_m3 = pressure_pa / (Boltzmann * temperature_k)
    molecule_mass_kg = gas_molar_mass_kg_per_mol / Avogadro
    mass_density_kg_per_m3 = number_density_m3 * molecule_mass_kg
    sound_speed = np.sqrt(
        np.maximum(
            gas_gamma
            * UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K
            / gas_molar_mass_kg_per_mol
            * temperature_k,
            1.0e-30,
        )
    )
    mach_number = (
        np.linalg.norm(gas_velocity_m_per_s, axis=1) / sound_speed
    )
    return number_density_m3, mass_density_kg_per_m3, mach_number


class RuntimeFieldSampler:
    """Own NumPy field caches and local gas/electric sampling.

    Taichi grids remain the authoritative device buffers. This component owns
    their CPU mirrors used by collision updates, source sampling, diagnostics,
    and report capture.
    """

    def __init__(
        self,
        *,
        config: SimulationConfig,
        static_grid: Any,
        pic_grid: Any,
        cartesian_field3d: Any,
        rf_enabled: bool,
        rf_peak_to_reference_scale: float,
        rf_angular_frequency_rad_s: float,
    ) -> None:
        self.config = config
        self.static_grid = static_grid
        self.pic_grid = pic_grid
        self.cartesian_field3d = cartesian_field3d
        self.rf_enabled = bool(rf_enabled)
        self.rf_peak_to_reference_scale = float(rf_peak_to_reference_scale)
        self.rf_angular_frequency_rad_s = float(rf_angular_frequency_rad_s)
        self.cache: dict[str, np.ndarray] = {}
        self.static_cache: dict[str, np.ndarray] = {}
        self.pic_cache: dict[str, np.ndarray] = {}
        self.refresh_all()

    def refresh_static(self) -> None:
        self.static_cache.clear()
        self.static_cache.update(
            E_dc_r=self.static_grid.E_dc_r.to_numpy(),
            E_dc_z=self.static_grid.E_dc_z.to_numpy(),
            E_rf_r=self.static_grid.E_rf_r.to_numpy(),
            E_rf_z=self.static_grid.E_rf_z.to_numpy(),
            P_gas=self.static_grid.P_gas.to_numpy(),
            T_gas=self.static_grid.T_gas.to_numpy(),
            v_gas_r=self.static_grid.v_gas_r.to_numpy(),
            v_gas_z=self.static_grid.v_gas_z.to_numpy(),
        )
        self.cache.update(self.static_cache)

    def refresh_pic(self) -> None:
        self.pic_cache.clear()
        self.pic_cache.update(
            E_sce_r=self.pic_grid.E_sce_r.to_numpy(),
            E_sce_z=self.pic_grid.E_sce_z.to_numpy(),
        )
        self.cache.update(self.pic_cache)

    def refresh_all(self) -> None:
        self.cache.clear()
        self.refresh_static()
        self.refresh_pic()

    def replace_static_fields(self, updater: Callable[[Any], None]) -> None:
        """Apply one grid update and atomically refresh both CPU cache groups."""

        if not callable(updater):
            raise TypeError("updater must be callable.")
        updater(self.static_grid)
        # Static and PIC storage may alias in the default shared-grid mode.
        self.refresh_all()

    def scalar_bilinear(self, field_name: str, r_m: float, z_m: float) -> float:
        if field_name in {"E_sce_r", "E_sce_z"}:
            field = self.pic_cache[field_name]
            grid = self.pic_grid
        else:
            field = self.static_cache[field_name]
            grid = self.static_grid
        r_clamped = float(np.clip(r_m, 0.0, self.config.domain_radius_m))
        z_clamped = float(np.clip(z_m, 0.0, self.config.domain_length_m))

        fr = r_clamped / grid.dr
        fz = (z_clamped - grid.z_min_m) / grid.dz
        i0 = int(np.clip(np.floor(fr), 0, grid.nr - 2))
        k0 = int(np.clip(np.floor(fz), 0, grid.nz - 2))
        i1 = i0 + 1
        k1 = k0 + 1
        wr = fr - float(i0)
        wz = fz - float(k0)
        return float(
            (1.0 - wr) * (1.0 - wz) * field[i0, k0]
            + wr * (1.0 - wz) * field[i1, k0]
            + (1.0 - wr) * wz * field[i0, k1]
            + wr * wz * field[i1, k1]
        )

    def scalar_bilinear_many(
        self,
        field_name: str,
        r_m: np.ndarray,
        z_m: np.ndarray,
    ) -> np.ndarray:
        if field_name in {"E_sce_r", "E_sce_z"}:
            field = self.pic_cache[field_name]
            grid = self.pic_grid
        else:
            field = self.static_cache[field_name]
            grid = self.static_grid

        r_clamped = np.clip(np.asarray(r_m, dtype=np.float64), 0.0, float(grid.r_max_m))
        z_clamped = np.clip(
            np.asarray(z_m, dtype=np.float64),
            float(grid.z_min_m),
            float(grid.z_max_m),
        )
        fr = r_clamped / float(grid.dr)
        fz = (z_clamped - float(grid.z_min_m)) / float(grid.dz)
        i0 = np.clip(np.floor(fr).astype(np.int64), 0, int(grid.nr) - 2)
        k0 = np.clip(np.floor(fz).astype(np.int64), 0, int(grid.nz) - 2)
        i1 = i0 + 1
        k1 = k0 + 1
        wr = fr - i0.astype(np.float64)
        wz = fz - k0.astype(np.float64)
        return (
            (1.0 - wr) * (1.0 - wz) * field[i0, k0]
            + wr * (1.0 - wz) * field[i1, k0]
            + (1.0 - wr) * wz * field[i0, k1]
            + wr * wz * field[i1, k1]
        )

    def rf_modulation(self, time_s: float) -> float:
        if not self.rf_enabled:
            return 0.0
        return float(
            self.rf_peak_to_reference_scale
            * np.cos(
                self.rf_angular_frequency_rad_s * float(time_s)
                + self.config.rf_phase_rad
            )
        )

    def sample_local_state_batch(
        self,
        positions_m: np.ndarray,
        time_s: float,
    ) -> LocalStateBatch:
        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.size == 0:
            return _empty_local_state_batch()
        samples = self._sample_cylindrical_components(positions, time_s)
        external, sce, gas = _project_cylindrical_vectors(samples)
        if self.cartesian_field3d is not None:
            external += self.cartesian_field3d.sample_many(
                positions,
                rf_modulation=samples.rf_modulation,
            )
        number_density, mass_density, mach = _gas_properties(
            samples.pressure_pa,
            samples.temperature_k,
            gas,
            gas_gamma=float(getattr(self.config, "gas_gamma", 1.4)),
            gas_molar_mass_kg_per_mol=float(
                getattr(
                    self.config,
                    "gas_molar_mass_kg_per_mol",
                    28.0134e-3,
                )
            ),
        )
        return LocalStateBatch(
            gas_velocity_m_per_s=gas,
            pressure_pa=samples.pressure_pa,
            temperature_k=samples.temperature_k,
            number_density_m3=number_density,
            mass_density_kg_per_m3=mass_density,
            mach_number=mach,
            external_field_v_per_m=external,
            sce_field_v_per_m=sce,
            total_field_v_per_m=external + sce,
        )

    def _sample_cylindrical_components(
        self,
        positions: np.ndarray,
        time_s: float,
    ) -> _CylindricalSamples:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        r = np.sqrt(x * x + y * y)
        e_dc_r = self.scalar_bilinear_many("E_dc_r", r, z)
        e_dc_z = self.scalar_bilinear_many("E_dc_z", r, z)
        rf_modulation = self.rf_modulation(time_s)
        e_rf_r = rf_modulation * self.scalar_bilinear_many("E_rf_r", r, z)
        e_rf_z = rf_modulation * self.scalar_bilinear_many("E_rf_z", r, z)
        e_sce_r = self.scalar_bilinear_many("E_sce_r", r, z)
        e_sce_z = self.scalar_bilinear_many("E_sce_z", r, z)
        v_gas_r = self.scalar_bilinear_many("v_gas_r", r, z)
        v_gas_z = self.scalar_bilinear_many("v_gas_z", r, z)
        pressure_pa = np.maximum(self.scalar_bilinear_many("P_gas", r, z), 0.0)
        temperature_k = np.maximum(self.scalar_bilinear_many("T_gas", r, z), 1.0)
        return _CylindricalSamples(
            x_m=x,
            y_m=y,
            r_m=r,
            e_external_r_v_per_m=e_dc_r + e_rf_r,
            e_external_z_v_per_m=e_dc_z + e_rf_z,
            e_sce_r_v_per_m=e_sce_r,
            e_sce_z_v_per_m=e_sce_z,
            gas_r_m_per_s=v_gas_r,
            gas_z_m_per_s=v_gas_z,
            pressure_pa=pressure_pa,
            temperature_k=temperature_k,
            rf_modulation=rf_modulation,
        )

    def space_charge_external_ratio_p95(
        self,
        state: dict[str, np.ndarray],
        time_s: float,
    ) -> float:
        active_indices = np.flatnonzero(state["active"] != 0)
        if active_indices.size == 0:
            return float("nan")
        positions_m = np.asarray(state["positions"][active_indices], dtype=np.float64)
        samples = self.sample_local_state_batch(positions_m, time_s)
        external_norm = np.linalg.norm(samples.external_field_v_per_m, axis=1)
        sce_norm = np.linalg.norm(samples.sce_field_v_per_m, axis=1)
        ratio = sce_norm / np.maximum(external_norm, 1.0e-30)
        finite_ratio = ratio[np.isfinite(ratio)]
        if finite_ratio.size == 0:
            return float("nan")
        return float(np.percentile(finite_ratio, 95))
