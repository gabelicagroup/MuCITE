"""Taichi-backed storage for 3D ion macro-particle entities."""

import taichi as ti


@ti.data_oriented
class IonCloud3D:
    """3D ion macro-particle storage without transport behavior."""

    def __init__(self, n_particles: int) -> None:
        self.n_particles = int(n_particles)
        if self.n_particles <= 0:
            raise ValueError("IonCloud3D requires at least one particle.")
        self.x = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.y = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.z = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.vx = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.vy = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.vz = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.q = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.m = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.weight = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.active = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.internal_temperature = ti.field(
            dtype=ti.f64,
            shape=self.n_particles,
        )
        self.dt_candidate = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.dt_limiter = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.dt_raw_limiter = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.collision_rate_hz = ti.field(dtype=ti.f64, shape=self.n_particles)
        self.terminal_count = ti.field(dtype=ti.i32, shape=())
        self.terminal_indices = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.terminal_status = ti.field(dtype=ti.i32, shape=self.n_particles)
        self.terminal_electrode_id = ti.field(
            dtype=ti.i32,
            shape=self.n_particles,
        )
        self.terminal_surface_distance = ti.field(
            dtype=ti.f64,
            shape=self.n_particles,
        )

    @ti.kernel
    def apply_state_updates(
        self,
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        velocities: ti.types.ndarray(dtype=ti.f64, ndim=2),
        weights: ti.types.ndarray(dtype=ti.f64, ndim=1),
        temperatures_k: ti.types.ndarray(dtype=ti.f64, ndim=1),
        q_c: ti.f64,
        m_kg: ti.f64,
        n_updates: ti.i32,
    ):
        for j in range(n_updates):
            p = indices[j]
            self.x[p] = positions[j, 0]
            self.y[p] = positions[j, 1]
            self.z[p] = positions[j, 2]
            self.vx[p] = velocities[j, 0]
            self.vy[p] = velocities[j, 1]
            self.vz[p] = velocities[j, 2]
            self.q[p] = q_c
            self.m[p] = m_kg
            self.weight[p] = weights[j]
            self.internal_temperature[p] = temperatures_k[j]
            self.dt_candidate[p] = 1.0e99
            self.dt_limiter[p] = -1
            self.dt_raw_limiter[p] = -1
            self.active[p] = 1

    @ti.kernel
    def gather_motion_state(
        self,
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        velocities: ti.types.ndarray(dtype=ti.f64, ndim=2),
        n_updates: ti.i32,
    ):
        for j in range(n_updates):
            p = indices[j]
            positions[j, 0] = self.x[p]
            positions[j, 1] = self.y[p]
            positions[j, 2] = self.z[p]
            velocities[j, 0] = self.vx[p]
            velocities[j, 1] = self.vy[p]
            velocities[j, 2] = self.vz[p]

    @ti.kernel
    def deactivate_indices(
        self,
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        n_updates: ti.i32,
    ):
        for j in range(n_updates):
            self.active[indices[j]] = 0

    @ti.kernel
    def activate_particles(
        self,
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        positions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        velocities: ti.types.ndarray(dtype=ti.f64, ndim=2),
        weights: ti.types.ndarray(dtype=ti.f64, ndim=1),
        q_c: ti.f64,
        m_kg: ti.f64,
        temperature_k: ti.f64,
        n_updates: ti.i32,
    ):
        for j in range(n_updates):
            p = indices[j]
            self.x[p] = positions[j, 0]
            self.y[p] = positions[j, 1]
            self.z[p] = positions[j, 2]
            self.vx[p] = velocities[j, 0]
            self.vy[p] = velocities[j, 1]
            self.vz[p] = velocities[j, 2]
            self.q[p] = q_c
            self.m[p] = m_kg
            self.weight[p] = weights[j]
            self.internal_temperature[p] = temperature_k
            self.dt_candidate[p] = 1.0e99
            self.dt_limiter[p] = -1
            self.dt_raw_limiter[p] = -1
            self.active[p] = 1
