"""Optional pyamg-backed axisymmetric Poisson solver."""

import time
import warnings
from typing import Any, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab, cg

from ...fields.axisymmetric_grid import UnifiedGrid2D
from .base import call_iterative_solver, norm, relative_residual
from .sparse import AxisymmetricSparsePoissonSolver
from .types import PoissonSolveResult, SMALL_RESIDUAL_NORM


class AxisymmetricAMGPoissonSolver(AxisymmetricSparsePoissonSolver):
    """Optional pyamg-backed production-candidate Poisson solver."""

    backend_name = "amg"

    def __init__(
        self,
        *,
        amg_mode: str = "solve",
        amg_solver: str = "ruge_stuben",
        fallback_backend: str = "sparse_bicgstab",
        tolerance: float = 1.0e-3,
        max_iters: int = 50,
        warm_start: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("mode", None)
        kwargs.pop("preconditioner", None)
        kwargs.pop("amg_fallback_mode", None)
        super().__init__(
            mode="amg",
            tolerance=tolerance,
            max_iters=max_iters,
            warm_start=warm_start,
            preconditioner="none",
            amg_fallback_mode=fallback_backend,
            **kwargs,
        )
        self.amg_mode = str(amg_mode).lower()
        self.amg_solver = str(amg_solver).lower()
        self.fallback_backend = str(fallback_backend).lower()
        self._amg_A: Optional[csr_matrix] = None
        self._amg_warning: Optional[str] = None

    def setup(
        self,
        grid: UnifiedGrid2D,
        boundary: Optional[dict[str, Any]] = None,
        mask: Any = None,
    ) -> None:
        super().setup(grid, boundary=boundary, mask=mask)
        self._diagnostics.update(
            {
                "amg_mode": self.amg_mode,
                "amg_solver": self.amg_solver,
                "amg_fallback_backend": self.fallback_backend,
                "amg_warning": self._amg_warning,
                "amg_operator": "-finite_volume_laplacian",
            }
        )

    def _setup_amg(self) -> None:
        self._amg_solver = None
        self._amg_import_error = None
        self._amg_warning = None
        if self._A is None:
            raise RuntimeError(
                "Sparse matrix must be assembled before AMG setup."
            )
        self._amg_A = csr_matrix(-self._A)
        try:
            import pyamg  # type: ignore
        except Exception as exc:  # pragma: no cover - optional package
            self._amg_import_error = repr(exc)
            self._warn_and_fallback(
                f"pyamg is not installed ({exc!r}); "
                f"falling back to {self.fallback_backend}."
            )
            return
        try:
            if self.amg_solver in {"ruge_stuben", "rs"}:
                self._amg_solver = pyamg.ruge_stuben_solver(self._amg_A)
            elif self.amg_solver in {"smoothed_aggregation", "sa"}:
                self._amg_solver = pyamg.smoothed_aggregation_solver(
                    self._amg_A
                )
            else:
                raise ValueError(
                    "amg_solver must be 'ruge_stuben' "
                    "or 'smoothed_aggregation'."
                )
        except Exception as exc:  # pragma: no cover - environment-specific
            self._amg_import_error = repr(exc)
            self._amg_solver = None
            self._warn_and_fallback(
                f"pyamg setup failed ({exc!r}); "
                f"falling back to {self.fallback_backend}."
            )

    def _warn_and_fallback(self, message: str) -> None:
        self._amg_warning = message
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _dispatch_amg_solve(
        self,
        b: np.ndarray,
        x0: np.ndarray,
    ) -> tuple[int, np.ndarray, int, str]:
        if self._amg_solver is None:
            backend = (
                "pyamg_unavailable_fallback_"
                f"{self.fallback_backend}"
            )
            iterations, solution, info = self._solve_with_fallback(b, x0)
        elif self.amg_mode == "solve":
            backend = f"pyamg_{self.amg_solver}_solve"
            iterations, solution, info = self._solve_amg_direct(b, x0)
        elif self.amg_mode == "preconditioned_cg":
            backend = f"pyamg_{self.amg_solver}_preconditioned_cg"
            iterations, solution, info = self._solve_amg_preconditioned(
                b,
                x0,
                krylov="cg",
            )
        elif self.amg_mode == "preconditioned_bicgstab":
            backend = (
                f"pyamg_{self.amg_solver}_preconditioned_bicgstab"
            )
            iterations, solution, info = self._solve_amg_preconditioned(
                b,
                x0,
                krylov="bicgstab",
            )
        else:
            raise ValueError(
                "amg_mode must be 'solve', 'preconditioned_cg', "
                "or 'preconditioned_bicgstab'."
            )
        return iterations, solution, info, backend

    def _amg_diagnostics(
        self,
        effective_backend: str,
        info: int,
        iterations: int,
    ) -> dict[str, Any]:
        return {
            "mode": "amg",
            "effective_backend": effective_backend,
            "solver_info": int(info),
            "amg_iterations": int(iterations),
            "amg_mode": self.amg_mode,
            "amg_solver": self.amg_solver,
            "amg_available": self._amg_solver is not None,
            "amg_import_error": self._amg_import_error,
            "amg_warning": self._amg_warning,
            "amg_fallback_backend": self.fallback_backend,
            "matrix_nnz": int(self._A.nnz),
            "active_unknowns": int(len(self._active_cells)),
            "tolerance": float(self.tolerance),
            "max_iters": int(self.max_iters),
            "convergence_criterion": "relative_residual",
        }

    def _build_amg_result(
        self,
        b: np.ndarray,
        solution: np.ndarray,
        info: int,
        backend: str,
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
            backend_name=backend,
            setup_time_ms=self.setup_time_ms,
            diagnostics=self._amg_diagnostics(
                backend,
                info,
                iterations,
            ),
        )
        self.previous_phi = phi.copy()
        self.last_result = result
        return result

    def solve(
        self,
        rho: np.ndarray,
        phi_initial: Optional[np.ndarray] = None,
    ) -> PoissonSolveResult:
        b, x0, residual_initial = self._prepare_solve(rho, phi_initial)
        started = time.perf_counter()
        iterations, solution, info, backend = self._dispatch_amg_solve(
            b,
            x0,
        )
        runtime_ms = (time.perf_counter() - started) * 1.0e3
        return self._build_amg_result(
            b,
            solution,
            info,
            backend,
            residual_initial,
            iterations,
            runtime_ms,
        )

    def _solve_amg_direct(
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
        )
        iterations = max(0, len(residuals) - 1)
        residual_final = norm(self._A @ solution - b)
        info = (
            0
            if residual_final
            <= self.tolerance * max(norm(b), SMALL_RESIDUAL_NORM)
            else 1
        )
        return iterations, np.asarray(solution, dtype=np.float64), info

    def _solve_amg_preconditioned(
        self,
        b: np.ndarray,
        x0: np.ndarray,
        *,
        krylov: str,
    ) -> tuple[int, np.ndarray, int]:
        if self._amg_A is None:
            raise RuntimeError("AMG matrix was not initialized.")
        iterations = 0
        b_amg = -b
        preconditioner = self._amg_solver.aspreconditioner()

        def callback(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        solver = cg if krylov == "cg" else bicgstab
        if krylov not in {"cg", "bicgstab"}:
            raise ValueError("krylov must be 'cg' or 'bicgstab'.")
        solution, info = call_iterative_solver(
            solver,
            self._amg_A,
            b_amg,
            x0=x0,
            rtol=self.tolerance,
            atol=0.0,
            maxiter=self.max_iters,
            M=preconditioner,
            callback=callback,
        )
        return iterations, np.asarray(solution, dtype=np.float64), int(info)
