"""Small source-model accessors used by the control-layer runtime port."""

from __future__ import annotations

import numpy as np


class SourceAccessMixin:
    """Expose source physics through stable runtime method names."""

    def _continuous_source_enabled(self) -> bool:
        return bool(
            self._current_stage_source_enabled
            and self.continuous_source.enabled()
        )

    def _source_charge_per_ion_c(self) -> float:
        return self.continuous_source.charge_per_ion_c()

    def _source_temperature_for_mach_k(self) -> float:
        return self.continuous_source.temperature_for_mach_k()

    def _source_axial_velocity_m_per_s(self) -> float:
        return self.continuous_source.axial_velocity_m_per_s()

    def _capillary_exit_z_m(self) -> float:
        return self.continuous_source.capillary_exit_z_m()

    def _capillary_inlet_z_m(self) -> float:
        return self.continuous_source.capillary_inlet_z_m()

    def _source_ion_emission_rate_per_s(self) -> float:
        return self.continuous_source.ion_emission_rate_per_s()

    def _source_target_macro_weight(self) -> float:
        """Return real-ion weight represented by one new macro particle."""

        if self.config.max_macro_particle_weight > 0.0:
            return max(
                float(self.config.max_macro_particle_weight),
                1.0e-30,
            )
        macro_count = max(
            1,
            int(self.config.macro_particles_per_injection),
        )
        reference_dt_s = max(
            float(self.config.macro_time_step_s),
            1.0e-30,
        )
        represented = (
            self._source_ion_emission_rate_per_s()
            * reference_dt_s
            / float(macro_count)
        )
        return max(represented, 1.0e-30)

    def _source_exit_kinetic_energy_j(self) -> float:
        return self.continuous_source.exit_kinetic_energy_j()

    def _source_exit_speed_m_per_s(self) -> float:
        return self.continuous_source.exit_speed_m_per_s()

    def _capillary_number_density_m3(self) -> float:
        return self.continuous_source.capillary_number_density_m3()

    def _sample_source_disk_positions(
        self,
        count: int,
        z_values_m: np.ndarray,
    ) -> np.ndarray:
        return self.continuous_source.sample_source_disk_positions(
            count,
            z_values_m,
        )

    def _sample_capillary_phase_space(
        self,
        count: int,
        *,
        prefill: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.continuous_source.sample_capillary_phase_space(
            count,
            prefill=prefill,
        )

    def _sample_external_emission_velocities(
        self,
        positions_m: np.ndarray,
    ) -> np.ndarray:
        return self.continuous_source.sample_external_emission_velocities(
            positions_m
        )


__all__ = ["SourceAccessMixin"]
