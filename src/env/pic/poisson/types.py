"""Shared result types and constants for axisymmetric Poisson solvers."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


EPSILON_0 = 8.8541878128e-12
SMALL_RESIDUAL_NORM = 1.0e-300


@dataclass(frozen=True)
class PoissonSolveResult:
    """Uniform result object returned by every PIC Poisson backend."""

    phi: np.ndarray
    E_r: np.ndarray
    E_z: np.ndarray
    residual_initial: float
    residual_final: float
    relative_residual: float
    iterations: int
    converged: bool
    runtime_ms: float
    backend_name: str
    setup_time_ms: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
