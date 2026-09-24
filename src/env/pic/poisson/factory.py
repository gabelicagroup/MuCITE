"""Factory for axisymmetric PIC Poisson backends."""

from .amg import AxisymmetricAMGPoissonSolver
from .base import PoissonSolverBase
from .sor import SORSolver
from .sparse import AxisymmetricSparsePoissonSolver


def create_poisson_solver(
    backend: str,
    *,
    sor_omega: float = 1.7,
    tolerance: float = 1.0e-6,
    max_iters: int = 200,
    warm_start: bool = True,
    preconditioner: str = "none",
    amg_mode: str = "solve",
    amg_solver: str = "ruge_stuben",
    amg_tolerance: float = 1.0e-3,
    amg_max_iters: int = 50,
    amg_fallback_backend: str = "sparse_bicgstab",
) -> PoissonSolverBase:
    """Build one solver while preserving established backend aliases."""

    backend_name = str(backend).lower()
    if backend_name in {"sor", "legacy_sor"}:
        return SORSolver(
            sor_omega=sor_omega,
            tolerance=tolerance,
            max_iters=max_iters,
            warm_start=warm_start,
        )
    sparse_names = {
        "sparse_direct",
        "sparse_spsolve",
        "sparse_bicgstab",
        "sparse_cg",
    }
    if backend_name in sparse_names:
        return AxisymmetricSparsePoissonSolver(
            mode=backend_name,
            tolerance=tolerance,
            max_iters=max_iters,
            warm_start=warm_start,
            preconditioner=preconditioner,
        )
    if backend_name in {"amg", "pyamg", "sparse_amg"}:
        return AxisymmetricAMGPoissonSolver(
            amg_mode=amg_mode,
            amg_solver=amg_solver,
            fallback_backend=amg_fallback_backend,
            tolerance=amg_tolerance,
            max_iters=amg_max_iters,
            warm_start=warm_start,
        )
    raise ValueError(f"Unsupported PIC Poisson backend: {backend}")
