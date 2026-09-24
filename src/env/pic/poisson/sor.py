"""Legacy red-black SOR reference backend."""

import time
from typing import Any, Optional

import numpy as np
import taichi as ti

from ...fields.axisymmetric_grid import UnifiedGrid2D
from .base import PoissonSolverBase, make_dirichlet_mask, norm, relative_residual
from .types import EPSILON_0, PoissonSolveResult


@ti.data_oriented
class SORSolver(PoissonSolverBase):
    """Legacy red-black SOR backend retained as a reference/debug solver."""

    backend_name = "sor"

    def __init__(
        self,
        *,
        sor_omega: float = 1.7,
        tolerance: float = 1.0e-6,
        max_iters: int = 200,
        warm_start: bool = True,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            max_iters=max_iters,
            warm_start=warm_start,
        )
        self.sor_omega = float(sor_omega)
        if not np.isfinite(self.sor_omega) or not 0.0 < self.sor_omega < 2.0:
            raise ValueError(
                "SOR relaxation factor must satisfy "
                f"0 < omega < 2, got {sor_omega}."
            )
        self.max_update = ti.field(dtype=ti.f64, shape=())
        self._dirichlet_mask_ti: Optional[Any] = None

    def setup(
        self,
        grid: UnifiedGrid2D,
        boundary: Optional[dict[str, Any]] = None,
        mask: Any = None,
    ) -> None:
        started = time.perf_counter()
        self.grid = grid
        self.dirichlet_mask = make_dirichlet_mask(grid, mask)
        self._dirichlet_mask_ti = ti.field(dtype=ti.i32, shape=(grid.nr, grid.nz))
        self._dirichlet_mask_ti.from_numpy(
            self.dirichlet_mask.astype(np.int32)
        )
        self.setup_time_ms = (time.perf_counter() - started) * 1.0e3
        self._diagnostics = {
            "sor_omega": float(self.sor_omega),
            "max_iters": int(self.max_iters),
            "tolerance": float(self.tolerance),
            "dirichlet_nodes": int(np.count_nonzero(self.dirichlet_mask)),
            "mask_dirichlet_applied": bool(mask is not None),
        }

    @ti.kernel
    def _reset_max_update_kernel(self):
        self.max_update[None] = 0.0

    @ti.kernel
    def _apply_dirichlet_kernel(self):
        for i, k in self.grid.phi_sce:
            if self._dirichlet_mask_ti[i, k] != 0:
                self.grid.phi_sce[i, k] = 0.0

    @ti.kernel
    def _sor_sweep_kernel(self, parity: ti.i32):
        for i, k in ti.ndrange((0, self.grid.nr), (0, self.grid.nz)):
            if self._dirichlet_mask_ti[i, k] == 0 and (i + k) % 2 == parity:
                old_phi = self.grid.phi_sce[i, k]
                rho = self.grid.rho_charge[i, k]
                ap = 1.0
                rhs = 0.0

                if i == 0:
                    ap = 4.0 / (self.grid.dr * self.grid.dr) + 2.0 / (self.grid.dz * self.grid.dz)
                    rhs = (
                        4.0 * self.grid.phi_sce[1, k] / (self.grid.dr * self.grid.dr)
                        + (self.grid.phi_sce[0, k + 1] + self.grid.phi_sce[0, k - 1]) / (self.grid.dz * self.grid.dz)
                        + rho / EPSILON_0
                    )
                else:
                    r_i = self.grid.radial_coordinate(i)
                    ap = 2.0 / (self.grid.dr * self.grid.dr) + 2.0 / (self.grid.dz * self.grid.dz)
                    rhs = (
                        (self.grid.phi_sce[i + 1, k] + self.grid.phi_sce[i - 1, k]) / (self.grid.dr * self.grid.dr)
                        + (self.grid.phi_sce[i + 1, k] - self.grid.phi_sce[i - 1, k]) / (2.0 * r_i * self.grid.dr)
                        + (self.grid.phi_sce[i, k + 1] + self.grid.phi_sce[i, k - 1]) / (self.grid.dz * self.grid.dz)
                        + rho / EPSILON_0
                    )

                phi_star = rhs / ap
                phi_new = (1.0 - self.sor_omega) * old_phi + self.sor_omega * phi_star
                self.grid.phi_sce[i, k] = phi_new
                ti.atomic_max(self.max_update[None], ti.abs(phi_new - old_phi))

    @ti.kernel
    def _update_sce_field_kernel(self):
        for i, k in self.grid.E_sce_r:
            if i == 0:
                self.grid.E_sce_r[i, k] = 0.0
            elif i == self.grid.nr - 1:
                self.grid.E_sce_r[i, k] = -(
                    self.grid.phi_sce[i, k]
                    - self.grid.phi_sce[i - 1, k]
                ) / self.grid.dr
            else:
                self.grid.E_sce_r[i, k] = -(
                    self.grid.phi_sce[i + 1, k]
                    - self.grid.phi_sce[i - 1, k]
                ) / (2.0 * self.grid.dr)

            if k == 0:
                self.grid.E_sce_z[i, k] = -(
                    self.grid.phi_sce[i, k + 1]
                    - self.grid.phi_sce[i, k]
                ) / self.grid.dz
            elif k == self.grid.nz - 1:
                self.grid.E_sce_z[i, k] = -(
                    self.grid.phi_sce[i, k]
                    - self.grid.phi_sce[i, k - 1]
                ) / self.grid.dz
            else:
                self.grid.E_sce_z[i, k] = -(
                    self.grid.phi_sce[i, k + 1]
                    - self.grid.phi_sce[i, k - 1]
                ) / (2.0 * self.grid.dz)

    def solve(
        self,
        rho: np.ndarray,
        phi_initial: Optional[np.ndarray] = None,
    ) -> PoissonSolveResult:
        rho, residual_initial, rhs_norm = _prepare_solve(
            self,
            rho,
            phi_initial,
        )
        started = time.perf_counter()
        iterations = _run_iterations(
            self,
            rho,
            residual_initial,
            rhs_norm,
        )
        runtime_ms = (time.perf_counter() - started) * 1.0e3
        return _build_result(
            self,
            rho,
            residual_initial,
            rhs_norm,
            iterations,
            runtime_ms,
        )

    def compute_residual(self, phi: np.ndarray, rho: np.ndarray) -> np.ndarray:
        if self.grid is None or self.dirichlet_mask is None:
            raise RuntimeError(
                "SORSolver.setup() must be called before residual evaluation."
            )
        phi = np.asarray(phi, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        residual = np.zeros_like(phi, dtype=np.float64)
        dr2 = float(self.grid.dr) ** 2
        dz2 = float(self.grid.dz) ** 2
        for i in range(self.grid.nr - 1):
            for k in range(1, self.grid.nz - 1):
                if self.dirichlet_mask[i, k]:
                    continue
                if i == 0:
                    laplace = 4.0 * (phi[1, k] - phi[0, k]) / dr2
                else:
                    r_i = float(i) * float(self.grid.dr)
                    laplace = (
                        (
                            phi[i + 1, k]
                            - 2.0 * phi[i, k]
                            + phi[i - 1, k]
                        )
                        / dr2
                        + (phi[i + 1, k] - phi[i - 1, k])
                        / (2.0 * r_i * float(self.grid.dr))
                    )
                laplace += (
                    phi[i, k + 1]
                    - 2.0 * phi[i, k]
                    + phi[i, k - 1]
                ) / dz2
                residual[i, k] = rho[i, k] / EPSILON_0 - (-laplace)
        return residual


def _install_initial_state(
    solver: SORSolver,
    rho: np.ndarray,
    phi_initial: Optional[np.ndarray],
) -> None:
    if phi_initial is not None:
        initial_phi = np.asarray(phi_initial, dtype=np.float64)
        if initial_phi.shape != rho.shape:
            raise ValueError(
                f"phi_initial shape={initial_phi.shape}, expected={rho.shape}."
            )
        if not np.all(np.isfinite(initial_phi)):
            raise ValueError(
                "Poisson initial potential contains non-finite values."
            )
        solver.grid.phi_sce.from_numpy(initial_phi)
    elif solver.warm_start and solver.previous_phi is not None:
        solver.grid.phi_sce.from_numpy(solver.previous_phi)
    solver.grid.rho_charge.from_numpy(rho)
    solver._apply_dirichlet_kernel()


def _prepare_solve(
    solver: SORSolver,
    rho: np.ndarray,
    phi_initial: Optional[np.ndarray],
) -> tuple[np.ndarray, float, float]:
    if solver.grid is None or solver.dirichlet_mask is None:
        raise RuntimeError("SORSolver.setup() must be called before solve().")
    rho = np.asarray(rho, dtype=np.float64)
    if rho.shape != (solver.grid.nr, solver.grid.nz):
        raise ValueError(
            f"rho shape={rho.shape}, expected={(solver.grid.nr, solver.grid.nz)}."
        )
    if not np.all(np.isfinite(rho)):
        raise ValueError(
            "Poisson charge-density array contains non-finite values."
        )
    _install_initial_state(solver, rho, phi_initial)
    phi = solver.grid.phi_sce.to_numpy()
    residual_initial = norm(solver.compute_residual(phi, rho))
    rhs = np.where(solver.dirichlet_mask, 0.0, rho / EPSILON_0)
    return rho, residual_initial, norm(rhs)


def _run_iterations(
    solver: SORSolver,
    rho: np.ndarray,
    residual_initial: float,
    rhs_norm: float,
) -> int:
    performed_iters = 0
    solver._reset_max_update_kernel()
    converged = (
        relative_residual(residual_initial, rhs_norm, residual_initial)
        <= solver.tolerance
    )
    if converged:
        return performed_iters
    for iteration in range(solver.max_iters):
        solver._reset_max_update_kernel()
        solver._sor_sweep_kernel(0)
        solver._sor_sweep_kernel(1)
        solver._apply_dirichlet_kernel()
        performed_iters = iteration + 1
        if float(solver.max_update[None]) < solver.tolerance:
            candidate = norm(
                solver.compute_residual(
                    solver.grid.phi_sce.to_numpy(),
                    rho,
                )
            )
            relative = relative_residual(
                candidate,
                rhs_norm,
                residual_initial,
            )
            if relative <= solver.tolerance:
                break
    return performed_iters


def _build_result(
    solver: SORSolver,
    rho: np.ndarray,
    residual_initial: float,
    rhs_norm: float,
    iterations: int,
    runtime_ms: float,
) -> PoissonSolveResult:
    solver._update_sce_field_kernel()
    phi = solver.grid.phi_sce.to_numpy()
    e_r = solver.grid.E_sce_r.to_numpy()
    e_z = solver.grid.E_sce_z.to_numpy()
    residual_final = norm(solver.compute_residual(phi, rho))
    relative = relative_residual(
        residual_final,
        rhs_norm,
        residual_initial,
    )
    result = PoissonSolveResult(
        phi=phi,
        E_r=e_r,
        E_z=e_z,
        residual_initial=residual_initial,
        residual_final=residual_final,
        relative_residual=relative,
        iterations=iterations,
        converged=_finite_converged(solver, phi, e_r, e_z, relative),
        runtime_ms=runtime_ms,
        backend_name=solver.backend_name,
        setup_time_ms=solver.setup_time_ms,
        diagnostics=_result_diagnostics(solver),
    )
    solver.previous_phi = phi.copy()
    solver.last_result = result
    return result


def _finite_converged(
    solver: SORSolver,
    phi: np.ndarray,
    e_r: np.ndarray,
    e_z: np.ndarray,
    relative: float,
) -> bool:
    return bool(
        np.all(np.isfinite(phi))
        and np.all(np.isfinite(e_r))
        and np.all(np.isfinite(e_z))
        and relative <= solver.tolerance
    )


def _result_diagnostics(solver: SORSolver) -> dict[str, float | str]:
    return {
        "max_update": float(solver.max_update[None]),
        "tolerance": float(solver.tolerance),
        "sor_omega": float(solver.sor_omega),
        "convergence_criterion": "relative_residual",
    }
