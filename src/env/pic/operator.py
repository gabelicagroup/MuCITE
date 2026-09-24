"""Environment-owned PIC charge-deposition and Poisson operator."""

import taichi as ti
from typing import Any, Optional

from .poisson import PoissonSolveResult, create_poisson_solver
from ..fields.axisymmetric_grid import UnifiedGrid2D


class PoissonConvergenceError(RuntimeError):
    """Raised when a PIC Poisson solve fails its declared residual tolerance."""

    def __init__(self, result: PoissonSolveResult) -> None:
        self.result = result
        super().__init__(
            "PIC Poisson solve did not converge: "
            f"backend={result.backend_name}, iterations={result.iterations}, "
            f"relative_residual={result.relative_residual:.6e}."
        )


@ti.data_oriented
class PICSolver:
    """Standard PIC operator set on a 2D axisymmetric ``(r, z)`` mesh."""

    def __init__(
        self,
        grid: UnifiedGrid2D,
        *,
        sor_omega: float = 1.7,
        max_sor_iters: int = 200,
        sor_tolerance: float = 1.0e-6,
        poisson_backend: str = "sor",
        poisson_preconditioner: str = "none",
        poisson_warm_start: bool = True,
        poisson_amg_mode: str = "solve",
        poisson_amg_solver: str = "ruge_stuben",
        poisson_amg_tolerance: float = 1.0e-3,
        poisson_amg_max_iters: int = 50,
        poisson_amg_fallback_backend: str = "sparse_bicgstab",
        electrode_mask: Any = None,
    ) -> None:
        self._require_pic_storage(grid)
        self.grid = grid
        self.sor_omega = float(sor_omega)
        self.max_sor_iters = int(max_sor_iters)
        self.sor_tolerance = float(sor_tolerance)
        self.poisson_backend = str(poisson_backend)
        self.poisson_preconditioner = str(poisson_preconditioner)
        self.poisson_warm_start = bool(poisson_warm_start)
        self.poisson_amg_mode = str(poisson_amg_mode)
        self.poisson_amg_solver = str(poisson_amg_solver)
        self.poisson_amg_tolerance = float(poisson_amg_tolerance)
        self.poisson_amg_max_iters = int(poisson_amg_max_iters)
        self.poisson_amg_fallback_backend = str(poisson_amg_fallback_backend)
        self.electrode_mask = electrode_mask
        self.poisson_solver = create_poisson_solver(
            self.poisson_backend,
            sor_omega=self.sor_omega,
            tolerance=self.sor_tolerance,
            max_iters=self.max_sor_iters,
            warm_start=self.poisson_warm_start,
            preconditioner=self.poisson_preconditioner,
            amg_mode=self.poisson_amg_mode,
            amg_solver=self.poisson_amg_solver,
            amg_tolerance=self.poisson_amg_tolerance,
            amg_max_iters=self.poisson_amg_max_iters,
            amg_fallback_backend=self.poisson_amg_fallback_backend,
        )
        self.poisson_solver.setup(self.grid, mask=self.electrode_mask)
        self.last_poisson_result: Optional[PoissonSolveResult] = None

    @staticmethod
    def _require_pic_storage(grid: UnifiedGrid2D) -> None:
        if not grid.has_pic_fields:
            raise ValueError(
                "PICSolver requires a UnifiedGrid2D with PIC storage "
                "(storage_mode='pic' or 'unified')."
            )

    @ti.kernel
    def _scatter_charge_kernel(
        self,
        x: ti.template(),
        y: ti.template(),
        z: ti.template(),
        q: ti.template(),
        weight: ti.template(),
        active: ti.template(),
        n_particles: ti.i32,
    ):
        for p in range(n_particles):
            if active[p] != 0:
                r = ti.sqrt(x[p] * x[p] + y[p] * y[p])
                z_pos = z[p]
                if 0.0 <= r <= self.grid.r_max_m and self.grid.z_min_m <= z_pos <= self.grid.z_max_m:
                    fr = r / self.grid.dr
                    fz = (z_pos - self.grid.z_min_m) / self.grid.dz
                    i0 = ti.max(0, ti.min(self.grid.nr - 2, int(ti.floor(fr))))
                    k0 = ti.max(0, ti.min(self.grid.nz - 2, int(ti.floor(fz))))
                    i1 = i0 + 1
                    k1 = k0 + 1

                    wr = fr - float(i0)
                    wz = fz - float(k0)

                    # - Axisymmetric CIC deposition:
                    #   the particle ring charge is distributed to the 4 surrounding
                    #   nodes with bilinear area weights in the logical ``(r, z)``
                    #   cell:
                    #
                    #   w00 = (1-wr)(1-wz)
                    #   w10 = wr(1-wz)
                    #   w01 = (1-wr)wz
                    #   w11 = wr wz
                    #
                    # - Because this is an axisymmetric mesh, each node represents a
                    #   3D annular control volume, so the deposited charge must be
                    #   normalized by the ring volume ``V_i`` to obtain ``rho [C/m^3]``.
                    w00 = (1.0 - wr) * (1.0 - wz)
                    w10 = wr * (1.0 - wz)
                    w01 = (1.0 - wr) * wz
                    w11 = wr * wz

                    q_macro = q[p] * weight[p]

                    ti.atomic_add(self.grid.rho_charge[i0, k0], q_macro * w00 / self.grid.node_volume(i0))
                    ti.atomic_add(self.grid.rho_charge[i1, k0], q_macro * w10 / self.grid.node_volume(i1))
                    ti.atomic_add(self.grid.rho_charge[i0, k1], q_macro * w01 / self.grid.node_volume(i0))
                    ti.atomic_add(self.grid.rho_charge[i1, k1], q_macro * w11 / self.grid.node_volume(i1))

    def scatter_charge(self, ions: object, grid: Optional[UnifiedGrid2D] = None) -> None:
        """Deposit the macro-particle charge onto the axisymmetric mesh."""

        if grid is not None:
            self._require_pic_storage(grid)
            self.grid = grid

        self._scatter_charge_kernel(ions.x, ions.y, ions.z, ions.q, ions.weight, ions.active, ions.n_particles)

    def setup_poisson_solver(self, grid: Optional[UnifiedGrid2D] = None, *, electrode_mask: Any = None) -> None:
        """Rebuild backend state when the PIC grid or electrode mask changes."""

        if grid is not None:
            self._require_pic_storage(grid)
            self.grid = grid
        if electrode_mask is not None or self.electrode_mask is None:
            self.electrode_mask = electrode_mask
        self.poisson_solver.setup(self.grid, mask=self.electrode_mask)

    def solve_poisson(self, grid: Optional[UnifiedGrid2D] = None) -> PoissonSolveResult:
        """Solve the PIC Poisson equation with the configured backend."""

        if grid is not None and grid is not self.grid:
            self._require_pic_storage(grid)
            self.setup_poisson_solver(grid)
        rho = self.grid.rho_charge.to_numpy()
        result = self.poisson_solver.solve(rho)
        self.last_poisson_result = result
        if not result.converged:
            raise PoissonConvergenceError(result)
        self.grid.phi_sce.from_numpy(result.phi)
        self.grid.E_sce_r.from_numpy(result.E_r)
        self.grid.E_sce_z.from_numpy(result.E_z)
        return result

    @ti.kernel
    def _scale_space_charge_solution_kernel(self, scale: ti.f64):
        for i, k in self.grid.E_sce_r:
            self.grid.phi_sce[i, k] *= scale
            self.grid.E_sce_r[i, k] *= scale
            self.grid.E_sce_z[i, k] *= scale

    def scale_space_charge_solution(self, grid: Optional[UnifiedGrid2D] = None, *, scale: float = 1.0) -> None:
        """Scale the solved PIC potential and electric field without changing particle weights."""

        if grid is not None:
            self._require_pic_storage(grid)
            self.grid = grid
        self._scale_space_charge_solution_kernel(float(scale))
