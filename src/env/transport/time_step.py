"""Taichi timestep-candidate preparation and diagnostic download."""

import numpy as np
import taichi as ti

from ...agents.taichi_cloud import IonCloud3D
from .collision_rate import iict_mean_relative_speed


class TimeStepConstraintMixin:
    """Collision/RF/acceleration/CFL timestep constraints."""

    @ti.func
    def _particle_state_invalid(
        self,
        x: ti.f64,
        y: ti.f64,
        z: ti.f64,
        vx: ti.f64,
        vy: ti.f64,
        vz: ti.f64,
        q: ti.f64,
        m: ti.f64,
    ) -> ti.i32:
        invalid = 0
        if (
            ti.math.isnan(x) or ti.math.isnan(y) or ti.math.isnan(z)
            or ti.math.isnan(vx) or ti.math.isnan(vy) or ti.math.isnan(vz)
            or ti.math.isnan(q) or ti.math.isnan(m)
            or ti.math.isinf(x) or ti.math.isinf(y) or ti.math.isinf(z)
            or ti.math.isinf(vx) or ti.math.isinf(vy) or ti.math.isinf(vz)
            or ti.math.isinf(q) or ti.math.isinf(m)
        ):
            invalid = 1
        return invalid

    @ti.func
    def _gather_invalid(self, gathered: ti.template()) -> ti.i32:
        invalid = 0
        for component in ti.static(range(8)):
            if ti.math.isnan(gathered[component]) or ti.math.isinf(gathered[component]):
                invalid = 1
        return invalid

    @ti.func
    def _collision_constraints(
        self,
        gathered: ti.template(),
        vx: ti.f64,
        vy: ti.f64,
        vz: ti.f64,
        cross_section_m2: ti.f64,
        boltzmann_j_per_k: ti.f64,
        safety_factor: ti.f64,
    ) -> ti.types.vector(3, ti.f64):
        temperature_k = ti.max(gathered[6], 1.0)
        pressure_pa = ti.max(gathered[7], 0.0)
        number_density_m3 = pressure_pa / (boltzmann_j_per_k * temperature_k)
        dvx, dvy, dvz = vx - gathered[3], vy - gathered[4], vz - gathered[5]
        relative_speed = ti.max(
            ti.sqrt(dvx * dvx + dvy * dvy + dvz * dvz),
            1.0e-9,
        )
        if ti.static(self.collision_rate_model_code == 1):
            relative_speed = iict_mean_relative_speed(
                relative_speed,
                temperature_k,
                self.gas_molecule_mass_kg,
                boltzmann_j_per_k,
            )
        rate_hz = number_density_m3 * cross_section_m2 * relative_speed
        invalid = 0.0
        if rate_hz < 0.0 or ti.math.isnan(rate_hz) or ti.math.isinf(rate_hz):
            invalid, rate_hz = 1.0, 0.0
        collision_dt_s = 1.0e99
        if rate_hz > 0.0:
            collision_dt_s = safety_factor / rate_hz
        return ti.Vector([rate_hz, collision_dt_s, invalid])

    @ti.func
    def _acceleration_dt(
        self,
        gathered: ti.template(),
        q_c: ti.f64,
        mass_kg: ti.f64,
        safety_factor: ti.f64,
        spatial_tolerance_m: ti.f64,
    ) -> ti.f64:
        electric_norm = ti.sqrt(
            gathered[0] * gathered[0]
            + gathered[1] * gathered[1]
            + gathered[2] * gathered[2]
        )
        acceleration = ti.abs(q_c) * electric_norm / ti.max(mass_kg, 1.0e-30)
        result = 1.0e99
        if acceleration > 0.0:
            result = safety_factor * ti.sqrt(2.0 * spatial_tolerance_m / acceleration)
        return result

    @ti.func
    def _advection_dt(
        self,
        vx: ti.f64,
        vy: ti.f64,
        vz: ti.f64,
    ) -> ti.f64:
        speed = ti.sqrt(vx * vx + vy * vy + vz * vz)
        result = 1.0e99
        if speed > 0.0:
            result = self.advection_cfl_fraction * self.advection_cell_size_m / speed
        return result

    @ti.func
    def _noncollision_choice(
        self,
        macro_remaining_s: ti.f64,
        rf_dt_s: ti.f64,
        acceleration_dt_s: ti.f64,
        advection_dt_s: ti.f64,
        max_dt_s: ti.f64,
    ) -> ti.types.vector(2, ti.f64):
        dt_s, limiter = max_dt_s, 0
        if macro_remaining_s < dt_s:
            dt_s, limiter = macro_remaining_s, 1
        if rf_dt_s < dt_s:
            dt_s, limiter = rf_dt_s, 3
        if acceleration_dt_s < dt_s:
            dt_s, limiter = acceleration_dt_s, 4
        if advection_dt_s < dt_s:
            dt_s, limiter = advection_dt_s, 7
        return ti.Vector([dt_s, float(limiter)])

    @ti.func
    def _raw_dt_choice(
        self,
        noncollision: ti.template(),
        collision_dt_s: ti.f64,
        collision_rate_hz: ti.f64,
        collision_model_code: ti.i32,
        original_z_m: ti.f64,
        langevin_z_start_m: ti.f64,
        langevin_z_end_m: ti.f64,
        langevin_switch_probability: ti.f64,
        langevin_max_dt_s: ti.f64,
    ) -> ti.types.vector(2, ti.f64):
        raw_dt, raw_limiter = noncollision[0], int(noncollision[1])
        use_langevin_dt = False
        if collision_model_code == 1:
            if original_z_m >= langevin_z_start_m and original_z_m <= langevin_z_end_m:
                gated_dt, gated_limiter = raw_dt, raw_limiter
                if langevin_max_dt_s > 0.0 and langevin_max_dt_s < gated_dt:
                    gated_dt, gated_limiter = langevin_max_dt_s, 6
                if collision_rate_hz > 0.0 and collision_rate_hz * gated_dt >= langevin_switch_probability:
                    use_langevin_dt = True
                    raw_dt, raw_limiter = gated_dt, gated_limiter
        if (not use_langevin_dt) and collision_dt_s < raw_dt:
            raw_dt, raw_limiter = collision_dt_s, 2
        return ti.Vector([raw_dt, float(raw_limiter)])

    @ti.func
    def _final_dt_choice(
        self,
        raw: ti.template(),
        macro_remaining_s: ti.f64,
        min_dt_s: ti.f64,
        max_dt_s: ti.f64,
        invalid_local_state: ti.i32,
    ) -> ti.types.vector(2, ti.f64):
        raw_dt, raw_limiter = raw[0], int(raw[1])
        final_dt = ti.min(macro_remaining_s, ti.min(max_dt_s, ti.max(min_dt_s, raw_dt)))
        final_limiter = raw_limiter
        if raw_limiter == 7 and raw_dt < final_dt:
            final_dt, final_limiter = raw_dt, 7
        elif macro_remaining_s < min_dt_s:
            final_limiter = 1
        elif raw_dt < min_dt_s:
            final_limiter = 5
        if invalid_local_state != 0:
            final_dt, final_limiter = 0.0, -2
        return ti.Vector([final_dt, float(final_limiter)])

class TimeStepMixin(TimeStepConstraintMixin):
    """Public candidate kernel and per-particle orchestration."""

    @ti.func
    def _sample_active_constraints(
        self,
        ions: ti.template(),
        p: ti.i32,
        time_s: ti.f64,
        rf_angular_frequency_rad_s: ti.f64,
        rf_phase_rad: ti.f64,
        rf_peak_to_reference_scale: ti.f64,
        collision_cross_section_m2: ti.f64,
        boltzmann_j_per_k: ti.f64,
        collision_safety_factor: ti.f64,
        acceleration_safety_factor: ti.f64,
        spatial_tolerance_m: ti.f64,
    ) -> ti.types.vector(5, ti.f64):
        x, y, z = ions.x[p], ions.y[p], ions.z[p]
        vx, vy, vz = ions.vx[p], ions.vy[p], ions.vz[p]
        q_c, mass_kg = ions.q[p], ions.m[p]
        invalid = self._particle_state_invalid(x, y, z, vx, vy, vz, q_c, mass_kg)
        if invalid != 0:
            x, y, z, vx, vy, vz, q_c, mass_kg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        gathered = self.gather_fields(
            x, y, z, time_s, rf_angular_frequency_rad_s,
            rf_phase_rad, rf_peak_to_reference_scale,
        )
        if self._gather_invalid(gathered) != 0:
            invalid = 1
        collision = self._collision_constraints(
            gathered, vx, vy, vz, collision_cross_section_m2,
            boltzmann_j_per_k, collision_safety_factor,
        )
        if collision[2] != 0.0:
            invalid = 1
        ions.collision_rate_hz[p] = collision[0]
        acceleration_dt = self._acceleration_dt(
            gathered, q_c, mass_kg, acceleration_safety_factor,
            spatial_tolerance_m,
        )
        advection_dt = self._advection_dt(vx, vy, vz)
        return ti.Vector(
            [collision[0], collision[1], acceleration_dt, advection_dt, float(invalid)]
        )

    @ti.func
    def _select_and_store_candidate(
        self, ions: ti.template(), p: ti.i32, constraints: ti.template(),
        macro_remaining_s: ti.f64, rf_dt_s: ti.f64,
        min_dt_s: ti.f64, max_dt_s: ti.f64,
        collision_model_code: ti.i32, langevin_z_start_m: ti.f64,
        langevin_z_end_m: ti.f64, langevin_switch_probability: ti.f64,
        langevin_max_dt_s: ti.f64,
    ):
        noncollision = self._noncollision_choice(
            macro_remaining_s, rf_dt_s, constraints[2], constraints[3], max_dt_s,
        )
        raw = self._raw_dt_choice(
            noncollision, constraints[1], constraints[0], collision_model_code,
            ions.z[p], langevin_z_start_m, langevin_z_end_m,
            langevin_switch_probability, langevin_max_dt_s,
        )
        final = self._final_dt_choice(
            raw, macro_remaining_s, min_dt_s, max_dt_s, int(constraints[4]),
        )
        if constraints[4] != 0.0:
            ions.collision_rate_hz[p] = -1.0
        ions.dt_candidate[p] = final[0]
        ions.dt_limiter[p] = int(final[1])
        ions.dt_raw_limiter[p] = int(raw[1])

    @ti.func
    def _prepare_active_candidate(
        self, ions: ti.template(), p: ti.i32, time_s: ti.f64,
        macro_remaining_s: ti.f64, rf_dt_s: ti.f64,
        rf_angular_frequency_rad_s: ti.f64, rf_phase_rad: ti.f64,
        rf_peak_to_reference_scale: ti.f64,
        collision_cross_section_m2: ti.f64, boltzmann_j_per_k: ti.f64,
        collision_safety_factor: ti.f64, acceleration_safety_factor: ti.f64,
        spatial_tolerance_m: ti.f64, min_dt_s: ti.f64, max_dt_s: ti.f64,
        collision_model_code: ti.i32, langevin_z_start_m: ti.f64,
        langevin_z_end_m: ti.f64, langevin_switch_probability: ti.f64,
        langevin_max_dt_s: ti.f64,
    ):
        constraints = self._sample_active_constraints(
            ions, p, time_s, rf_angular_frequency_rad_s, rf_phase_rad,
            rf_peak_to_reference_scale, collision_cross_section_m2,
            boltzmann_j_per_k, collision_safety_factor,
            acceleration_safety_factor, spatial_tolerance_m,
        )
        self._select_and_store_candidate(
            ions, p, constraints, macro_remaining_s, rf_dt_s,
            min_dt_s, max_dt_s, collision_model_code,
            langevin_z_start_m, langevin_z_end_m,
            langevin_switch_probability, langevin_max_dt_s,
        )

    @ti.kernel
    def prepare_dt_candidates(
        self,
        ions: ti.template(),
        time_s: ti.f64,
        macro_end_s: ti.f64,
        rf_angular_frequency_rad_s: ti.f64,
        rf_phase_rad: ti.f64,
        rf_peak_to_reference_scale: ti.f64,
        collision_cross_section_m2: ti.f64,
        boltzmann_j_per_k: ti.f64,
        collision_safety_factor: ti.f64,
        rf_frequency_hz: ti.f64,
        rf_safety_factor: ti.f64,
        acceleration_safety_factor: ti.f64,
        spatial_tolerance_m: ti.f64,
        min_dt_s: ti.f64,
        max_dt_s: ti.f64,
        collision_model_code: ti.i32,
        langevin_z_start_m: ti.f64,
        langevin_z_end_m: ti.f64,
        langevin_switch_probability: ti.f64,
        langevin_max_dt_s: ti.f64,
    ):
        macro_remaining_s = ti.max(0.0, macro_end_s - time_s)
        rf_dt_s = 1.0e99
        if rf_frequency_hz > 0.0:
            rf_dt_s = rf_safety_factor / rf_frequency_hz
        for p in range(ions.n_particles):
            if ions.active[p] != 0:
                self._prepare_active_candidate(
                    ions, p, time_s, macro_remaining_s, rf_dt_s,
                    rf_angular_frequency_rad_s, rf_phase_rad,
                    rf_peak_to_reference_scale, collision_cross_section_m2,
                    boltzmann_j_per_k, collision_safety_factor,
                    acceleration_safety_factor, spatial_tolerance_m,
                    min_dt_s, max_dt_s, collision_model_code,
                    langevin_z_start_m, langevin_z_end_m,
                    langevin_switch_probability, langevin_max_dt_s,
                )
            else:
                ions.collision_rate_hz[p] = 0.0
                ions.dt_candidate[p] = 1.0e99
                ions.dt_limiter[p] = -1
                ions.dt_raw_limiter[p] = -1

    @ti.kernel
    def _download_dt_diagnostics_kernel(
        self,
        ions: ti.template(),
        dt_candidates_s: ti.types.ndarray(dtype=ti.f64, ndim=1),
        limiter_codes: ti.types.ndarray(dtype=ti.i32, ndim=1),
        raw_limiter_codes: ti.types.ndarray(dtype=ti.i32, ndim=1),
        collision_rates_hz: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        for p in range(ions.n_particles):
            dt_candidates_s[p] = ions.dt_candidate[p]
            limiter_codes[p] = ions.dt_limiter[p]
            raw_limiter_codes[p] = ions.dt_raw_limiter[p]
            collision_rates_hz[p] = ions.collision_rate_hz[p]

    def download_dt_diagnostics(
        self,
        ions: IonCloud3D,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Download timestep selection and cached collision rates."""

        dt_candidates_s = np.empty(ions.n_particles, dtype=np.float64)
        limiter_codes = np.empty(ions.n_particles, dtype=np.int32)
        raw_limiter_codes = np.empty(ions.n_particles, dtype=np.int32)
        collision_rates_hz = np.empty(ions.n_particles, dtype=np.float64)
        self._download_dt_diagnostics_kernel(
            ions,
            dt_candidates_s,
            limiter_codes,
            raw_limiter_codes,
            collision_rates_hz,
        )
        return (
            dt_candidates_s,
            limiter_codes,
            raw_limiter_codes,
            collision_rates_hz,
        )
