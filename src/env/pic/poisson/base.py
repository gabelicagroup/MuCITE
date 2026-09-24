"""Common geometry, residual, and solver interfaces for PIC Poisson backends."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from ...fields.axisymmetric_grid import UnifiedGrid2D
from .types import PoissonSolveResult, SMALL_RESIDUAL_NORM


class PoissonSolverBase(ABC):
    """Common interface for axisymmetric PIC Poisson solvers."""

    backend_name = "base"

    def __init__(
        self,
        *,
        tolerance: float = 1.0e-6,
        max_iters: int = 200,
        warm_start: bool = True,
    ) -> None:
        self.tolerance = float(tolerance)
        self.max_iters = int(max_iters)
        self.warm_start = bool(warm_start)
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError(
                f"Poisson tolerance must be finite and positive, got {tolerance}."
            )
        if self.max_iters <= 0:
            raise ValueError(f"Poisson max_iters must be positive, got {max_iters}.")
        self.grid: Optional[UnifiedGrid2D] = None
        self.dirichlet_mask: Optional[np.ndarray] = None
        self.setup_time_ms = 0.0
        self.last_result: Optional[PoissonSolveResult] = None
        self.previous_phi: Optional[np.ndarray] = None
        self._diagnostics: dict[str, Any] = {}

    @abstractmethod
    def setup(
        self,
        grid: UnifiedGrid2D,
        boundary: Optional[dict[str, Any]] = None,
        mask: Any = None,
    ) -> None:
        """Prepare backend state for a fixed grid and optional electrode mask."""

    @abstractmethod
    def solve(
        self,
        rho: np.ndarray,
        phi_initial: Optional[np.ndarray] = None,
    ) -> PoissonSolveResult:
        """Solve for the space-charge perturbation potential and field."""

    @abstractmethod
    def compute_residual(self, phi: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Return the backend-native residual vector or residual grid."""

    def compute_e_field(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Differentiate ``phi`` into ``E_r`` and ``E_z`` in SI units."""

        if self.grid is None:
            raise RuntimeError(
                "Poisson solver must be set up before computing E field."
            )
        return compute_axisymmetric_e_field(phi, self.grid)

    def get_diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(self._diagnostics)
        diagnostics["backend_name"] = self.backend_name
        diagnostics["setup_time_ms"] = float(self.setup_time_ms)
        if self.last_result is not None:
            diagnostics["last_runtime_ms"] = float(self.last_result.runtime_ms)
            diagnostics["last_iterations"] = int(self.last_result.iterations)
            diagnostics["last_relative_residual"] = float(
                self.last_result.relative_residual
            )
            diagnostics["last_converged"] = bool(self.last_result.converged)
        return diagnostics


def compute_axisymmetric_e_field(
    phi: np.ndarray,
    grid: UnifiedGrid2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``E = -grad(phi)`` on the same ``(nr, nz)`` grid."""

    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != (grid.nr, grid.nz):
        raise ValueError(f"phi shape={phi.shape}, expected={(grid.nr, grid.nz)}.")
    e_r = np.zeros_like(phi, dtype=np.float64)
    e_z = np.zeros_like(phi, dtype=np.float64)
    if grid.nr > 1:
        e_r[0, :] = 0.0
        e_r[-1, :] = -(phi[-1, :] - phi[-2, :]) / grid.dr
    if grid.nr > 2:
        e_r[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * grid.dr)
    if grid.nz > 1:
        e_z[:, 0] = -(phi[:, 1] - phi[:, 0]) / grid.dz
        e_z[:, -1] = -(phi[:, -1] - phi[:, -2]) / grid.dz
    if grid.nz > 2:
        e_z[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * grid.dz)
    return e_r, e_z


def grid_r_coords(grid: UnifiedGrid2D) -> np.ndarray:
    return np.arange(grid.nr, dtype=np.float64) * float(grid.dr)


def grid_z_coords(grid: UnifiedGrid2D) -> np.ndarray:
    return float(grid.z_min_m) + np.arange(grid.nz, dtype=np.float64) * float(
        grid.dz
    )


def nearest_sample_mask(
    values: np.ndarray,
    r_coords_m: np.ndarray,
    z_coords_m: np.ndarray,
    target_r_m: np.ndarray,
    target_z_m: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    r_coords_m = np.asarray(r_coords_m, dtype=np.float64)
    z_coords_m = np.asarray(z_coords_m, dtype=np.float64)
    inside = (
        (target_r_m >= r_coords_m[0])
        & (target_r_m <= r_coords_m[-1])
        & (target_z_m >= z_coords_m[0])
        & (target_z_m <= z_coords_m[-1])
    )
    sampled = np.zeros(target_r_m.shape, dtype=bool)
    if not np.any(inside):
        return sampled
    r_inside = target_r_m[inside]
    z_inside = target_z_m[inside]
    ir_hi = np.clip(
        np.searchsorted(r_coords_m, r_inside, side="left"),
        1,
        len(r_coords_m) - 1,
    )
    iz_hi = np.clip(
        np.searchsorted(z_coords_m, z_inside, side="left"),
        1,
        len(z_coords_m) - 1,
    )
    ir_lo = ir_hi - 1
    iz_lo = iz_hi - 1
    ir = np.where(
        np.abs(r_coords_m[ir_hi] - r_inside)
        < np.abs(r_inside - r_coords_m[ir_lo]),
        ir_hi,
        ir_lo,
    )
    iz = np.where(
        np.abs(z_coords_m[iz_hi] - z_inside)
        < np.abs(z_inside - z_coords_m[iz_lo]),
        iz_hi,
        iz_lo,
    )
    sampled[inside] = values[ir, iz]
    return sampled


def make_dirichlet_mask(grid: UnifiedGrid2D, mask: Any = None) -> np.ndarray:
    """Build the PIC perturbation-potential Dirichlet mask for a grid."""

    dirichlet = np.zeros((grid.nr, grid.nz), dtype=bool)
    dirichlet[:, 0] = True
    dirichlet[:, -1] = True
    dirichlet[-1, :] = True
    if mask is None:
        return dirichlet
    if isinstance(mask, np.ndarray):
        metal = np.asarray(mask, dtype=bool)
        if metal.shape != dirichlet.shape:
            raise ValueError(
                f"mask shape={metal.shape}, expected={dirichlet.shape}."
            )
        dirichlet |= metal
        return dirichlet
    required = ("metal_mask", "r_coords_m", "z_coords_m")
    if all(hasattr(mask, name) for name in required):
        rr, zz = np.meshgrid(
            grid_r_coords(grid),
            grid_z_coords(grid),
            indexing="ij",
        )
        dirichlet |= nearest_sample_mask(
            np.asarray(mask.metal_mask, dtype=bool),
            np.asarray(mask.r_coords_m, dtype=np.float64),
            np.asarray(mask.z_coords_m, dtype=np.float64),
            rr,
            zz,
        )
        return dirichlet
    raise TypeError(
        "mask must be None, a bool ndarray, or an electrode-mask object."
    )


def norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64).ravel()))


def relative_residual(
    residual_norm: float,
    rhs_norm: float,
    initial_residual_norm: float,
) -> float:
    """Return a stable relative residual, including homogeneous RHS problems."""

    denominator = rhs_norm
    if denominator <= SMALL_RESIDUAL_NORM:
        denominator = max(initial_residual_norm, SMALL_RESIDUAL_NORM)
    return float(residual_norm / denominator)


def call_iterative_solver(
    solver: Any,
    matrix: Any,
    rhs: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, int]:
    """Call SciPy iterative solvers across old/new tolerance signatures."""

    try:
        return solver(matrix, rhs, **kwargs)
    except TypeError as exc:
        if "rtol" not in kwargs:
            raise
        legacy_kwargs = dict(kwargs)
        legacy_kwargs["tol"] = legacy_kwargs.pop("rtol")
        legacy_kwargs.pop("atol", None)
        try:
            return solver(matrix, rhs, **legacy_kwargs)
        except TypeError:
            raise exc
