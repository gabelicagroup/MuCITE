"""Manufactured and analytic contracts for numerical safety fixes."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import PARTICLE_ACTIVE, PARTICLE_ELECTRODE_HIT
from src.core.timestep import (
    InvalidTimeStepError,
    select_active_timestep,
    velocity_cfl_timestep,
)
from src.utils.geometry import (
    interpolate_segment_state,
    locate_first_segment_hit,
    segment_cylinder_exit_fraction,
    segment_plane_crossing_fraction,
)
from src.env.pic.operator import PICSolver, PoissonConvergenceError
from src.env.pic.poisson import (
    EPSILON_0,
    AxisymmetricSparsePoissonSolver,
    PoissonSolveResult,
    SORSolver,
)
from src.core.simulation import FlowchartPicSimulation


pytestmark = pytest.mark.physics


def _manufactured_problem(nr: int = 25, nz: int = 41):
    radius_m = 1.0
    length_m = 2.0
    grid = SimpleNamespace(
        nr=nr,
        nz=nz,
        r_max_m=radius_m,
        z_min_m=0.0,
        z_max_m=length_m,
        dr=radius_m / (nr - 1),
        dz=length_m / (nz - 1),
    )
    r_m = np.arange(nr, dtype=np.float64)[:, None] * grid.dr
    z_m = np.arange(nz, dtype=np.float64)[None, :] * grid.dz
    wave_number = math.pi / length_m
    phi_exact_v = (radius_m**2 - r_m**2) * np.sin(wave_number * z_m)
    rho_c_per_m3 = EPSILON_0 * (
        4.0 + wave_number**2 * (radius_m**2 - r_m**2)
    ) * np.sin(wave_number * z_m)
    return grid, phi_exact_v, rho_c_per_m3


def test_sparse_poisson_manufactured_solution_and_axis_symmetry() -> None:
    grid, phi_exact_v, rho_c_per_m3 = _manufactured_problem()
    solver = AxisymmetricSparsePoissonSolver(
        mode="sparse_direct",
        tolerance=1.0e-10,
        warm_start=False,
    )
    solver.setup(grid)

    result = solver.solve(rho_c_per_m3)

    assert result.converged
    assert result.relative_residual < 1.0e-10
    interior = np.s_[:, 1:-1]
    relative_l2_error = np.linalg.norm(
        result.phi[interior] - phi_exact_v[interior]
    ) / np.linalg.norm(phi_exact_v[interior])
    assert relative_l2_error < 1.0e-2
    np.testing.assert_array_equal(result.E_r[0, :], 0.0)


def test_sor_rejects_unstable_relaxation_factors() -> None:
    with pytest.raises(ValueError, match=r"0 < omega < 2"):
        SORSolver(sor_omega=2.0)
    with pytest.raises(ValueError, match=r"0 < omega < 2"):
        SORSolver(sor_omega=0.0)


class _ArrayField:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.writes = 0

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()

    def from_numpy(self, values: np.ndarray) -> None:
        self.values = np.asarray(values).copy()
        self.writes += 1


def test_pic_solver_fails_closed_before_installing_nonconverged_field() -> None:
    zeros = np.zeros((3, 3), dtype=np.float64)
    result = PoissonSolveResult(
        phi=zeros,
        E_r=zeros,
        E_z=zeros,
        residual_initial=1.0,
        residual_final=0.5,
        relative_residual=0.5,
        iterations=2,
        converged=False,
        runtime_ms=0.1,
        backend_name="synthetic",
    )
    grid = SimpleNamespace(
        rho_charge=_ArrayField(zeros),
        phi_sce=_ArrayField(np.ones_like(zeros)),
        E_sce_r=_ArrayField(np.ones_like(zeros)),
        E_sce_z=_ArrayField(np.ones_like(zeros)),
    )
    fake_solver = SimpleNamespace(solve=lambda _rho: result)
    owner = SimpleNamespace(
        grid=grid,
        poisson_solver=fake_solver,
        last_poisson_result=None,
        setup_poisson_solver=lambda _grid: None,
    )

    with pytest.raises(PoissonConvergenceError, match="relative_residual"):
        PICSolver.solve_poisson(owner)

    assert owner.last_poisson_result is result
    assert grid.phi_sce.writes == 0
    assert grid.E_sce_r.writes == 0
    assert grid.E_sce_z.writes == 0


def test_active_timestep_rejects_partial_nan_instead_of_ignoring_slot() -> None:
    with pytest.raises(InvalidTimeStepError) as exc_info:
        select_active_timestep(
            np.array([2.0e-9, np.nan, 1.0e99]),
            np.array([1, 1, 0]),
        )

    np.testing.assert_array_equal(exc_info.value.particle_indices, [1])


def test_active_timestep_ignores_inactive_sentinel_and_keeps_global_index() -> None:
    selection = select_active_timestep(
        np.array([1.0e99, 4.0e-9, 2.0e-9]),
        np.array([0, 1, 1]),
    )
    assert selection.dt_s == pytest.approx(2.0e-9)
    assert selection.particle_index == 2


def test_velocity_cfl_formula() -> None:
    dt_s = velocity_cfl_timestep(
        np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
        2.0e-3,
        safety_factor=0.5,
    )
    assert dt_s[0] == pytest.approx(2.0e-4)
    assert np.isinf(dt_s[1])


def test_continuous_plane_and_cylinder_crossings() -> None:
    start = np.array([0.2, 0.0, 0.4])
    end = np.array([1.2, 0.0, 0.8])

    plane_fraction = segment_plane_crossing_fraction(start, end, 0.5)
    cylinder_fraction = segment_cylinder_exit_fraction(start, end, 1.0)

    assert plane_fraction == pytest.approx(0.25)
    assert cylinder_fraction == pytest.approx(0.8)


def test_thin_solid_hit_is_localized_between_clear_endpoints() -> None:
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.0, 0.0, 1.0])

    fraction = locate_first_segment_hit(
        start,
        end,
        lambda position: 0.49 <= position[2] <= 0.51,
        max_sample_spacing_m=5.0e-3,
    )

    assert fraction == pytest.approx(0.49, abs=1.0e-8)
    event_position, event_velocity = interpolate_segment_state(
        start,
        end,
        np.array([0.0, 0.0, 2.0]),
        np.array([0.0, 0.0, 4.0]),
        fraction,
    )
    assert event_position[2] == pytest.approx(0.49, abs=1.0e-8)
    assert event_velocity[2] == pytest.approx(2.98, abs=2.0e-8)


def test_runtime_segment_localizer_uses_earliest_thin_electrode_entry() -> None:
    def sample_electrode(position_m: np.ndarray) -> tuple[bool, int, float]:
        hit = 0.49 <= float(position_m[2]) <= 0.51
        return hit, 12 if hit else -1, -1.0e-6 if hit else float("nan")

    owner = SimpleNamespace(
        config=SimpleNamespace(
            detector_z_m=2.0,
            detector_radius_m=2.0,
            radial_limit_m=2.0,
            domain_radius_m=2.0,
            domain_length_m=2.0,
        ),
        _electrode_segment_sample_spacing_m=lambda: 5.0e-3,
        _sample_electrode_event=sample_electrode,
    )

    (
        statuses,
        positions_m,
        velocities_m_per_s,
        event_times_s,
        electrode_ids,
        _surface_distance_m,
    ) = FlowchartPicSimulation._localize_terminal_segments(
        owner,
        start_positions_m=np.array([[0.0, 0.0, 0.0]]),
        end_positions_m=np.array([[0.0, 0.0, 1.0]]),
        start_velocities_m_per_s=np.array([[0.0, 0.0, 2.0]]),
        end_velocities_m_per_s=np.array([[0.0, 0.0, 4.0]]),
        endpoint_status=np.array([PARTICLE_ACTIVE], dtype=np.int16),
        endpoint_electrode_id=np.array([-1], dtype=np.int32),
        endpoint_surface_distance_m=np.array([np.nan]),
        step_start_time_s=10.0,
        step_dt_s=2.0,
    )

    assert statuses[0] == PARTICLE_ELECTRODE_HIT
    assert positions_m[0, 2] == pytest.approx(0.49, abs=1.0e-8)
    assert velocities_m_per_s[0, 2] == pytest.approx(2.98, abs=2.0e-8)
    assert event_times_s[0] == pytest.approx(10.98, abs=2.0e-8)
    assert electrode_ids[0] == 12


def test_taichi_dt_kernel_applies_velocity_cfl() -> None:
    ti = pytest.importorskip("taichi")
    from src.agents.taichi_cloud import IonCloud3D
    from src.env.fields.axisymmetric_grid import UnifiedGrid2D
    from src.env.transport import ParticlePusher
    from src.core.simulation import _ensure_pic_taichi_initialized

    _ensure_pic_taichi_initialized("cpu")
    grid = UnifiedGrid2D(3, 3, 2.0e-3, 0.0, 2.0e-3)
    cloud = IonCloud3D(1)
    pusher = ParticlePusher(grid, advection_cfl_fraction=0.5)

    cloud.x.from_numpy(np.array([0.0]))
    cloud.y.from_numpy(np.array([0.0]))
    cloud.z.from_numpy(np.array([0.5e-3]))
    cloud.vx.from_numpy(np.array([100.0]))
    cloud.vy.from_numpy(np.array([0.0]))
    cloud.vz.from_numpy(np.array([0.0]))
    cloud.q.from_numpy(np.array([1.0]))
    cloud.m.from_numpy(np.array([1.0]))
    cloud.active.from_numpy(np.array([1], dtype=np.int32))

    pusher.prepare_dt_candidates(
        cloud,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.380649e-23,
        0.1,
        0.0,
        0.05,
        0.2,
        1.0e-3,
        1.0e-9,
        1.0e-3,
        0,
        0.0,
        0.0,
        0.05,
        1.0e-9,
    )

    assert cloud.dt_candidate.to_numpy()[0] == pytest.approx(5.0e-6)
    assert cloud.dt_limiter.to_numpy()[0] == 7

    cloud.vx.from_numpy(np.array([1.0e10]))
    pusher.prepare_dt_candidates(
        cloud,
        0.0,
        0.5e-12,
        0.0,
        0.0,
        0.0,
        0.0,
        1.380649e-23,
        0.1,
        0.0,
        0.05,
        0.2,
        1.0e-3,
        1.0e-12,
        1.0e-3,
        0,
        0.0,
        0.0,
        0.05,
        1.0e-9,
    )
    assert cloud.dt_candidate.to_numpy()[0] == pytest.approx(0.05e-12)
    assert cloud.dt_limiter.to_numpy()[0] == 7

    cloud.vx.from_numpy(np.array([np.nan]))
    pusher.prepare_dt_candidates(
        cloud,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.380649e-23,
        0.1,
        0.0,
        0.05,
        0.2,
        1.0e-3,
        1.0e-12,
        1.0e-3,
        0,
        0.0,
        0.0,
        0.05,
        1.0e-9,
    )
    assert cloud.dt_candidate.to_numpy()[0] == 0.0

    rho_c_per_m3 = np.zeros((grid.nr, grid.nz), dtype=np.float64)
    rho_c_per_m3[0, 1] = EPSILON_0
    sor = SORSolver(
        sor_omega=1.0,
        tolerance=1.0e-12,
        max_iters=1,
        warm_start=False,
    )
    sor.setup(grid)
    sor_result = sor.solve(rho_c_per_m3)

    assert not sor_result.converged
    assert sor_result.relative_residual > sor.tolerance
    np.testing.assert_array_equal(sor_result.E_r[0, :], 0.0)
