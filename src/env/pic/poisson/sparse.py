"""Finite-volume sparse backends for the axisymmetric PIC Poisson equation."""

import math
import time
from typing import Any, Optional

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, splu, spsolve

from ...fields.axisymmetric_grid import UnifiedGrid2D
from .base import (
    PoissonSolverBase,
    call_iterative_solver,
    make_dirichlet_mask,
    norm,
    relative_residual,
)
from .types import EPSILON_0, PoissonSolveResult, SMALL_RESIDUAL_NORM


class SparseSystemMixin:
    """Matrix assembly and active/full-grid transformations."""

    def _assemble_matrix(self) -> None:
        if self.grid is None or self.dirichlet_mask is None:
            raise RuntimeError(
                "Grid and Dirichlet mask must be initialized before assembly."
            )
        grid = self.grid
        active_index = np.full((grid.nr, grid.nz), -1, dtype=np.int64)
        active_cells: list[tuple[int, int]] = []
        for i in range(grid.nr):
            for k in range(grid.nz):
                if not self.dirichlet_mask[i, k]:
                    active_index[i, k] = len(active_cells)
                    active_cells.append((i, k))
        matrix = lil_matrix(
            (len(active_cells), len(active_cells)),
            dtype=np.float64,
        )
        volumes = np.zeros((grid.nr, grid.nz), dtype=np.float64)
        for row, (i, k) in enumerate(active_cells):
            r_i = float(i) * grid.dr
            r_w = max(0.0, r_i - 0.5 * grid.dr)
            r_e = min(grid.r_max_m, r_i + 0.5 * grid.dr)
            annulus_area_m2 = math.pi * (r_e * r_e - r_w * r_w)
            volumes[i, k] = annulus_area_m2 * grid.dz
            diag = 0.0
            neighbors = (
                (i + 1, k, 2.0 * math.pi * r_e * grid.dz / grid.dr),
                (i - 1, k, 2.0 * math.pi * r_w * grid.dz / grid.dr),
                (i, k + 1, annulus_area_m2 / grid.dz),
                (i, k - 1, annulus_area_m2 / grid.dz),
            )
            for ni, nk, conductance in neighbors:
                if conductance == 0.0:
                    continue
                if not (0 <= ni < grid.nr and 0 <= nk < grid.nz):
                    continue
                diag -= conductance
                col = int(active_index[ni, nk])
                if col >= 0:
                    matrix[row, col] += conductance
            matrix[row, row] += diag
        self._active_index = active_index
        self._active_cells = active_cells
        self._volumes_m3 = volumes
        self._A = matrix.tocsr()

    def _setup_amg(self) -> None:
        self._amg_solver = None
        self._amg_import_error = None
        try:
            import pyamg  # type: ignore
        except Exception as exc:  # pragma: no cover - optional package
            self._amg_import_error = repr(exc)
            return
        try:
            self._amg_solver = pyamg.ruge_stuben_solver(-self._A)
        except Exception as exc:  # pragma: no cover - environment-specific
            self._amg_import_error = repr(exc)
            self._amg_solver = None

    def _build_preconditioner(
        self,
        effective_mode: str,
    ) -> Optional[LinearOperator]:
        if self._A is None or self.preconditioner == "none":
            return None
        if self.preconditioner != "jacobi":
            raise ValueError(
                "Supported sparse PIC Poisson preconditioners "
                "are 'none' and 'jacobi'."
            )
        operator = (
            -self._A
            if effective_mode in {"sparse_cg", "amg"}
            else self._A
        )
        diag = np.asarray(operator.diagonal(), dtype=np.float64)
        inv_diag = np.zeros_like(diag)
        valid = np.abs(diag) > 0.0
        inv_diag[valid] = 1.0 / diag[valid]
        return LinearOperator(
            operator.shape,
            matvec=lambda x: inv_diag * x,
            dtype=np.float64,
        )

    def _rhs_from_rho(self, rho: np.ndarray) -> np.ndarray:
        if self.grid is None or self._volumes_m3 is None:
            raise RuntimeError(
                "Sparse solver must be set up before building RHS."
            )
        rho = np.asarray(rho, dtype=np.float64)
        if rho.shape != (self.grid.nr, self.grid.nz):
            raise ValueError(
                f"rho shape={rho.shape}, expected={(self.grid.nr, self.grid.nz)}."
            )
        rhs = np.empty(len(self._active_cells), dtype=np.float64)
        for row, (i, k) in enumerate(self._active_cells):
            rhs[row] = (
                -rho[i, k] * self._volumes_m3[i, k] / EPSILON_0
            )
        return rhs

    def _active_phi(self, phi: np.ndarray) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError(
                "Sparse solver must be set up before using active phi."
            )
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape != (self.grid.nr, self.grid.nz):
            raise ValueError(
                f"phi shape={phi.shape}, expected={(self.grid.nr, self.grid.nz)}."
            )
        values = np.empty(len(self._active_cells), dtype=np.float64)
        for row, (i, k) in enumerate(self._active_cells):
            values[row] = phi[i, k]
        return values

    def _full_phi(self, active_phi: np.ndarray) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError(
                "Sparse solver must be set up before expanding phi."
            )
        phi = np.zeros((self.grid.nr, self.grid.nz), dtype=np.float64)
        for row, (i, k) in enumerate(self._active_cells):
            phi[i, k] = active_phi[row]
        return phi


class SparseKrylovMixin:
    """Krylov and optional legacy-AMG solve implementations."""

    def _solve_bicgstab(
        self,
        b: np.ndarray,
        x0: np.ndarray,
    ) -> tuple[int, np.ndarray, int]:
        iterations = 0
        preconditioner = self._M
        if self.mode == "amg" and self.preconditioner != "none":
            preconditioner = self._build_preconditioner("sparse_bicgstab")

        def callback(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = call_iterative_solver(
            bicgstab,
            self._A,
            b,
            x0=x0,
            rtol=self.tolerance,
            atol=0.0,
            maxiter=self.max_iters,
            M=preconditioner,
            callback=callback,
        )
        return iterations, np.asarray(solution, dtype=np.float64), int(info)

    def _solve_cg(
        self,
        b: np.ndarray,
        x0: np.ndarray,
    ) -> tuple[int, np.ndarray, int]:
        iterations = 0
        positive_operator = -self._A
        positive_rhs = -b
        preconditioner = (
            self._M
            if self.preconditioner == "jacobi"
            else self._build_preconditioner("sparse_cg")
        )

        def callback(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = call_iterative_solver(
            cg,
            positive_operator,
            positive_rhs,
            x0=x0,
            rtol=self.tolerance,
            atol=0.0,
            maxiter=self.max_iters,
            M=preconditioner,
            callback=callback,
        )
        return iterations, np.asarray(solution, dtype=np.float64), int(info)

    def _solve_amg(
        self,
        b: np.ndarray,
        x0: np.ndarray,
    ) -> tuple[int, np.ndarray, int]:
        residuals: list[float] = []
        solution = self._amg_solver.solve(
            -b,
            x0=x0,
            tol=self.tolerance,
            maxiter=self.max_iters,
            residuals=residuals,
            accel="cg",
        )
        iterations = max(0, len(residuals) - 1)
        final_residual = (
            residuals[-1]
            if residuals
            else norm((-self._A) @ solution - (-b))
        )
        initial_residual = (
            residuals[0]
            if residuals
            else max(norm(-b), SMALL_RESIDUAL_NORM)
        )
        info = (
            0
            if final_residual
            <= self.tolerance * max(initial_residual, SMALL_RESIDUAL_NORM)
            else 1
        )
        return iterations, np.asarray(solution, dtype=np.float64), info

    def _solve_with_fallback(
        self,
        b: np.ndarray,
        x0: np.ndarray,
    ) -> tuple[int, np.ndarray, int]:
        if self.amg_fallback_mode == "sparse_cg":
            return self._solve_cg(b, x0)
        if self.amg_fallback_mode == "sparse_spsolve":
            return 1, np.asarray(spsolve(self._A, b), dtype=np.float64), 0
        if self.amg_fallback_mode == "sparse_direct":
            if self._lu is None:
                self._lu = splu(csc_matrix(self._A))
            return 1, np.asarray(self._lu.solve(b), dtype=np.float64), 0
        return self._solve_bicgstab(b, x0)


class AxisymmetricSparsePoissonSolver(
    SparseSystemMixin,
    SparseKrylovMixin,
    PoissonSolverBase,
):
    """Finite-volume sparse axisymmetric PIC Poisson solver."""

    def __init__(
        self,
        *,
        mode: str = "sparse_direct",
        tolerance: float = 1.0e-8,
        max_iters: int = 1000,
        warm_start: bool = True,
        preconditioner: str = "none",
        amg_fallback_mode: str = "sparse_bicgstab",
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            max_iters=max_iters,
            warm_start=warm_start,
        )
        self.mode = str(mode).lower()
        self.backend_name = self.mode
        self.preconditioner = str(preconditioner).lower()
        self.amg_fallback_mode = str(amg_fallback_mode).lower()
        self._active_index: Optional[np.ndarray] = None
        self._active_cells: list[tuple[int, int]] = []
        self._volumes_m3: Optional[np.ndarray] = None
        self._A: Optional[csr_matrix] = None
        self._lu: Any = None
        self._M: Optional[LinearOperator] = None
        self._amg_solver: Any = None
        self._amg_import_error: Optional[str] = None

    def setup(
        self,
        grid: UnifiedGrid2D,
        boundary: Optional[dict[str, Any]] = None,
        mask: Any = None,
    ) -> None:
        started = time.perf_counter()
        self.grid = grid
        self.dirichlet_mask = make_dirichlet_mask(grid, mask)
        self._assemble_matrix()
        if self.mode == "sparse_direct":
            self._lu = splu(csc_matrix(self._A))
        elif self.mode == "amg":
            self._setup_amg()
        if self.preconditioner != "none":
            self._M = self._build_preconditioner(self.mode)
        self.setup_time_ms = (time.perf_counter() - started) * 1.0e3
        self._diagnostics = {
            "mode": self.mode,
            "active_unknowns": int(len(self._active_cells)),
            "dirichlet_nodes": int(np.count_nonzero(self.dirichlet_mask)),
            "matrix_nnz": 0 if self._A is None else int(self._A.nnz),
            "tolerance": float(self.tolerance),
            "max_iters": int(self.max_iters),
            "warm_start": bool(self.warm_start),
            "preconditioner": self.preconditioner,
            "amg_available": self._amg_solver is not None,
            "amg_import_error": self._amg_import_error,
        }

    def _prepare_solve(
        self,
        rho: np.ndarray,
        phi_initial: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if self.grid is None or self._A is None:
            raise RuntimeError(
                "AxisymmetricSparsePoissonSolver.setup() "
                "must be called before solve()."
            )
        rho = np.asarray(rho, dtype=np.float64)
        if not np.all(np.isfinite(rho)):
            raise ValueError(
                "Poisson charge-density array contains non-finite values."
            )
        b = self._rhs_from_rho(rho)
        if phi_initial is not None:
            if not np.all(np.isfinite(phi_initial)):
                raise ValueError(
                    "Poisson initial potential contains non-finite values."
                )
            x0 = self._active_phi(phi_initial)
        elif self.warm_start and self.previous_phi is not None:
            x0 = self._active_phi(self.previous_phi)
        else:
            x0 = np.zeros_like(b)
        return b, x0, norm(b - self._A @ x0)

    def _dispatch_solve(
        self,
        b: np.ndarray,
        x0: np.ndarray,
    ) -> tuple[int, np.ndarray, int, str]:
        effective_backend = self.mode
        if self.mode == "sparse_direct":
            return 1, np.asarray(self._lu.solve(b), dtype=np.float64), 0, self.mode
        if self.mode == "sparse_spsolve":
            solution = np.asarray(spsolve(self._A, b), dtype=np.float64)
            return 1, solution, 0, self.mode
        if self.mode == "sparse_bicgstab":
            iterations, solution, info = self._solve_bicgstab(b, x0)
        elif self.mode == "sparse_cg":
            iterations, solution, info = self._solve_cg(b, x0)
        elif self.mode == "amg" and self._amg_solver is not None:
            iterations, solution, info = self._solve_amg(b, x0)
            effective_backend = "pyamg"
        elif self.mode == "amg":
            effective_backend = (
                "pyamg_unavailable_fallback_"
                f"{self.amg_fallback_mode}"
            )
            iterations, solution, info = self._solve_with_fallback(b, x0)
        else:
            raise ValueError(f"Unsupported sparse Poisson mode: {self.mode}")
        return iterations, solution, info, effective_backend

    def _build_result(
        self,
        b: np.ndarray,
        solution: np.ndarray,
        info: int,
        effective_backend: str,
        residual_initial: float,
        iterations: int,
        runtime_ms: float,
    ) -> PoissonSolveResult:
        phi = self._full_phi(solution)
        e_r, e_z = self.compute_e_field(phi)
        residual_final = norm(b - self._A @ solution)
        relative = relative_residual(
            residual_final,
            norm(b),
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
            converged=bool(
                np.all(np.isfinite(solution))
                and info == 0
                and relative <= self.tolerance
            ),
            runtime_ms=runtime_ms,
            backend_name=effective_backend,
            setup_time_ms=self.setup_time_ms,
            diagnostics=self._result_diagnostics(
                effective_backend,
                info,
            ),
        )
        self.previous_phi = phi.copy()
        self.last_result = result
        return result

    def _result_diagnostics(
        self,
        effective_backend: str,
        info: int,
    ) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "effective_backend": effective_backend,
            "solver_info": int(info),
            "matrix_nnz": int(self._A.nnz),
            "active_unknowns": int(len(self._active_cells)),
            "preconditioner": self.preconditioner,
            "amg_import_error": self._amg_import_error,
            "convergence_criterion": "relative_residual",
        }

    def solve(
        self,
        rho: np.ndarray,
        phi_initial: Optional[np.ndarray] = None,
    ) -> PoissonSolveResult:
        b, x0, residual_initial = self._prepare_solve(rho, phi_initial)
        started = time.perf_counter()
        iterations, solution, info, backend = self._dispatch_solve(b, x0)
        runtime_ms = (time.perf_counter() - started) * 1.0e3
        return self._build_result(
            b,
            solution,
            info,
            backend,
            residual_initial,
            iterations,
            runtime_ms,
        )

    def compute_residual(self, phi: np.ndarray, rho: np.ndarray) -> np.ndarray:
        if self._A is None:
            raise RuntimeError(
                "Sparse solver must be set up before residual evaluation."
            )
        active_phi = self._active_phi(phi)
        b = self._rhs_from_rho(rho)
        return b - self._A @ active_phi
