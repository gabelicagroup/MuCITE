"""Taichi RK4 transport under gathered electric and optional drag fields."""

import taichi as ti


class RK4TransportMixin:
    """Fixed-step fourth-order Runge-Kutta particle transport."""

    @ti.func
    def _rhs(
        self,
        q_c: ti.f64,
        m_kg: ti.f64,
        x: ti.f64,
        y: ti.f64,
        z: ti.f64,
        vx: ti.f64,
        vy: ti.f64,
        vz: ti.f64,
        time_s: ti.f64,
        rf_angular_frequency_rad_s: ti.f64,
        rf_phase_rad: ti.f64,
        rf_peak_to_reference_scale: ti.f64,
    ) -> ti.types.vector(6, ti.f64):
        gathered = self.gather_fields(
            x,
            y,
            z,
            time_s,
            rf_angular_frequency_rad_s,
            rf_phase_rad,
            rf_peak_to_reference_scale,
        )
        electric_force = ti.Vector([gathered[0], gathered[1], gathered[2]]) * q_c
        acceleration = electric_force / m_kg
        gas_velocity = ti.Vector([gathered[3], gathered[4], gathered[5]])
        acceleration += self.drag_frequency_hz * (
            gas_velocity - ti.Vector([vx, vy, vz])
        )
        return ti.Vector(
            [vx, vy, vz, acceleration[0], acceleration[1], acceleration[2]]
        )

    @ti.func
    def _rhs_for_state(
        self,
        q_c: ti.f64,
        m_kg: ti.f64,
        state: ti.template(),
        time_s: ti.f64,
        rf_angular_frequency_rad_s: ti.f64,
        rf_phase_rad: ti.f64,
        rf_peak_to_reference_scale: ti.f64,
    ) -> ti.types.vector(6, ti.f64):
        return self._rhs(
            q_c,
            m_kg,
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            state[5],
            time_s,
            rf_angular_frequency_rad_s,
            rf_phase_rad,
            rf_peak_to_reference_scale,
        )

    @ti.func
    def _rk4_advance(
        self,
        state0: ti.template(),
        q_c: ti.f64,
        m_kg: ti.f64,
        time_s: ti.f64,
        dt_s: ti.f64,
        rf_angular_frequency_rad_s: ti.f64,
        rf_phase_rad: ti.f64,
        rf_peak_to_reference_scale: ti.f64,
    ) -> ti.types.vector(6, ti.f64):
        k1 = self._rhs_for_state(
            q_c, m_kg, state0, time_s, rf_angular_frequency_rad_s,
            rf_phase_rad, rf_peak_to_reference_scale,
        )
        s1 = state0 + 0.5 * dt_s * k1
        k2 = self._rhs_for_state(
            q_c, m_kg, s1, time_s + 0.5 * dt_s,
            rf_angular_frequency_rad_s, rf_phase_rad,
            rf_peak_to_reference_scale,
        )
        s2 = state0 + 0.5 * dt_s * k2
        k3 = self._rhs_for_state(
            q_c, m_kg, s2, time_s + 0.5 * dt_s,
            rf_angular_frequency_rad_s, rf_phase_rad,
            rf_peak_to_reference_scale,
        )
        s3 = state0 + dt_s * k3
        k4 = self._rhs_for_state(
            q_c, m_kg, s3, time_s + dt_s,
            rf_angular_frequency_rad_s, rf_phase_rad,
            rf_peak_to_reference_scale,
        )
        return state0 + dt_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @ti.kernel
    def push_rk4(
        self,
        ions: ti.template(),
        time_s: ti.f64,
        dt_s: ti.f64,
        rf_angular_frequency_rad_s: ti.f64,
        rf_phase_rad: ti.f64,
        rf_peak_to_reference_scale: ti.f64,
    ):
        """Advance all active particles by one fixed RK4 micro-step."""

        for p in range(ions.n_particles):
            if ions.active[p] != 0:
                state0 = ti.Vector(
                    [
                        ions.x[p],
                        ions.y[p],
                        ions.z[p],
                        ions.vx[p],
                        ions.vy[p],
                        ions.vz[p],
                    ]
                )
                state = self._rk4_advance(
                    state0, ions.q[p], ions.m[p], time_s, dt_s,
                    rf_angular_frequency_rad_s, rf_phase_rad,
                    rf_peak_to_reference_scale,
                )
                ions.x[p], ions.y[p], ions.z[p] = state[0], state[1], state[2]
                ions.vx[p], ions.vy[p], ions.vz[p] = state[3], state[4], state[5]
