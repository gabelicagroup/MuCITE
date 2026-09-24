"""Architecture and numerical contracts for environment-owned particle transport."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from src.agents.taichi_cloud import IonCloud3D as EntityIonCloud
from src.config import (
    PARTICLE_ACTIVE,
    PARTICLE_DOMAIN_OUT,
    PARTICLE_ELECTRODE_HIT,
    PARTICLE_RADIAL_OUT,
    PARTICLE_Z_EXIT,
)
from src.env.fields.axisymmetric_grid import UnifiedGrid2D
from src.env.fields.cartesian3d import CartesianField3D
from src.env.boundaries.mask3d import ElectrodeMask3D
from src.env.transport import ParticlePusher as EnvironmentPusher
from src.core.simulation import _ensure_pic_taichi_initialized


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_DIR = PROJECT_ROOT / "src" / "env" / "transport"
PRODUCTION_MODULES = (
    PROJECT_ROOT / "src" / "agents" / "taichi_cloud.py",
    *TRANSPORT_DIR.glob("*.py"),
)
def _initialize_cloud(
    cloud: EntityIonCloud,
    *,
    position_m: tuple[float, float, float],
    velocity_m_per_s: tuple[float, float, float],
    charge_c: float,
    mass_kg: float,
) -> None:
    cloud.x.from_numpy(np.array([position_m[0]], dtype=np.float64))
    cloud.y.from_numpy(np.array([position_m[1]], dtype=np.float64))
    cloud.z.from_numpy(np.array([position_m[2]], dtype=np.float64))
    cloud.vx.from_numpy(np.array([velocity_m_per_s[0]], dtype=np.float64))
    cloud.vy.from_numpy(np.array([velocity_m_per_s[1]], dtype=np.float64))
    cloud.vz.from_numpy(np.array([velocity_m_per_s[2]], dtype=np.float64))
    cloud.q.from_numpy(np.array([charge_c], dtype=np.float64))
    cloud.m.from_numpy(np.array([mass_kg], dtype=np.float64))
    cloud.active.from_numpy(np.array([1], dtype=np.int32))


def _motion_state(cloud: EntityIonCloud) -> tuple[np.ndarray, np.ndarray]:
    position = np.array(
        [cloud.x.to_numpy()[0], cloud.y.to_numpy()[0], cloud.z.to_numpy()[0]]
    )
    velocity = np.array(
        [cloud.vx.to_numpy()[0], cloud.vy.to_numpy()[0], cloud.vz.to_numpy()[0]]
    )
    return position, velocity


def _zero_grid() -> UnifiedGrid2D:
    grid = UnifiedGrid2D(3, 3, 1.0, 0.0, 1.0)
    grid.bake_zero_static_fields(
        background_pressure_pa=0.0,
        background_temperature_k=300.0,
    )
    return grid


def test_transport_modules_meet_granularity_limits() -> None:
    for path in PRODUCTION_MODULES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert len(source.splitlines()) <= 500, path
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.end_lineno is not None
                assert node.end_lineno - node.lineno + 1 <= 300, (path, node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno is not None
                assert node.end_lineno - node.lineno + 1 <= 50, (path, node.name)


def test_transport_modules_do_not_import_removed_physics_package() -> None:
    for path in PRODUCTION_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "physics.particle_pusher" not in imported_modules, path
        assert "src.physics.particle_pusher" not in imported_modules, path


@pytest.mark.physics
def test_axisymmetric_constant_field_rk4_matches_analytic_motion() -> None:
    _ensure_pic_taichi_initialized("cpu")
    grid = _zero_grid()
    grid.E_dc_r.from_numpy(np.full((3, 3), 4.0, dtype=np.float64))
    grid.E_dc_z.from_numpy(np.full((3, 3), 3.0, dtype=np.float64))
    cloud = EntityIonCloud(1)
    _initialize_cloud(
        cloud,
        position_m=(0.2, 0.0, 0.25),
        velocity_m_per_s=(0.0, 0.0, 0.0),
        charge_c=2.0,
        mass_kg=1.0,
    )

    dt_s = 0.02
    EnvironmentPusher(grid).push_rk4(cloud, 0.0, dt_s, 0.0, 0.0, 0.0)
    position, velocity = _motion_state(cloud)

    acceleration = np.array([8.0, 0.0, 6.0])
    np.testing.assert_allclose(position, [0.2, 0.0, 0.25] + 0.5 * acceleration * dt_s**2)
    np.testing.assert_allclose(velocity, acceleration * dt_s)


@pytest.mark.physics
def test_cartesian_rf_overlay_is_sampled_at_all_rk4_substages() -> None:
    _ensure_pic_taichi_initialized("cpu")
    grid = _zero_grid()
    shape = (2, 2, 2)
    zeros = np.zeros(shape, dtype=np.float64)
    rf_x_v_per_m = np.full(shape, 2.0, dtype=np.float64)
    overlay = CartesianField3D(
        x_coords_m=np.array([-1.0, 1.0]),
        y_coords_m=np.array([-1.0, 1.0]),
        z_coords_m=np.array([0.0, 1.0]),
        e_dc_x_v_per_m=zeros,
        e_dc_y_v_per_m=zeros,
        e_dc_z_v_per_m=zeros,
        e_rf_x_v_per_m=rf_x_v_per_m,
        e_rf_y_v_per_m=zeros,
        e_rf_z_v_per_m=zeros,
        rf_reference_peak_voltage_v=1.0,
        metadata={},
    )
    cloud = EntityIonCloud(1)
    initial_position = np.array([0.1, 0.2, 0.3])
    initial_velocity = np.array([0.4, 0.0, 0.0])
    _initialize_cloud(
        cloud,
        position_m=tuple(initial_position),
        velocity_m_per_s=tuple(initial_velocity),
        charge_c=1.0,
        mass_kg=1.0,
    )

    time_s, dt_s = 0.2, 0.05
    omega, phase_rad, rf_scale = 17.0, 0.3, 3.0
    EnvironmentPusher(grid, cartesian_field3d=overlay).push_rk4(
        cloud,
        time_s,
        dt_s,
        omega,
        phase_rad,
        rf_scale,
    )
    position, velocity = _motion_state(cloud)

    acceleration_scale = 2.0 * rf_scale
    acceleration_0 = acceleration_scale * math.cos(omega * time_s + phase_rad)
    acceleration_half = acceleration_scale * math.cos(
        omega * (time_s + 0.5 * dt_s) + phase_rad
    )
    acceleration_1 = acceleration_scale * math.cos(
        omega * (time_s + dt_s) + phase_rad
    )
    expected_vx = initial_velocity[0] + dt_s / 6.0 * (
        acceleration_0 + 4.0 * acceleration_half + acceleration_1
    )
    expected_x = initial_position[0] + dt_s * initial_velocity[0] + dt_s**2 / 6.0 * (
        acceleration_0 + 2.0 * acceleration_half
    )
    np.testing.assert_allclose(position, [expected_x, 0.2, 0.3], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(velocity, [expected_vx, 0.0, 0.0], rtol=0.0, atol=1.0e-14)


@pytest.mark.physics
def test_cartesian_electrode_terminal_kernel_preserves_id_and_priority() -> None:
    _ensure_pic_taichi_initialized("cpu")
    shape = (2, 2, 2)
    metal = np.zeros(shape, dtype=bool)
    electrode_id = np.full(shape, -1, dtype=np.int32)
    surface_distance_m = np.full(shape, 1.0, dtype=np.float64)
    metal[1, 1, 1] = True
    electrode_id[1, 1, 1] = 9
    surface_distance_m[1, 1, 1] = -0.1
    mask = ElectrodeMask3D(
        x_coords_m=np.array([-1.0, 1.0]),
        y_coords_m=np.array([-1.0, 1.0]),
        z_coords_m=np.array([0.0, 1.0]),
        metal_mask=metal,
        electrode_id=electrode_id,
        surface_distance_m=surface_distance_m,
        metadata={},
    )
    cloud = EntityIonCloud(1)
    _initialize_cloud(
        cloud,
        position_m=(1.0, 1.0, 1.0),
        velocity_m_per_s=(0.0, 0.0, 0.0),
        charge_c=1.0,
        mass_kg=1.0,
    )

    EnvironmentPusher(_zero_grid(), electrode_mask3d=mask).select_terminal_boundary_hits(
        cloud,
        5.0,
        5.0,
        5.0,
        5.0,
        2.0,
        0.0,
        0,
        PARTICLE_ACTIVE,
        PARTICLE_Z_EXIT,
        PARTICLE_RADIAL_OUT,
        PARTICLE_ELECTRODE_HIT,
        PARTICLE_DOMAIN_OUT,
    )

    assert cloud.terminal_count.to_numpy().item() == 1
    assert cloud.terminal_status.to_numpy()[0] == PARTICLE_ELECTRODE_HIT
    assert cloud.terminal_electrode_id.to_numpy()[0] == 9
    assert cloud.terminal_surface_distance.to_numpy()[0] == pytest.approx(-0.1)
    assert cloud.active.to_numpy()[0] == 0
