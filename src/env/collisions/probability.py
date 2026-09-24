"""Collision probability formulas."""

from __future__ import annotations

import numpy as np


def collision_probability_from_lambda_dt(lambda_dt: np.ndarray) -> np.ndarray:
    """Return the production Bernoulli probability ``1 - exp(-lambda*dt)``."""

    values = np.asarray(lambda_dt, dtype=np.float64)
    return np.clip(1.0 - np.exp(-values), 0.0, 1.0)


__all__ = ["collision_probability_from_lambda_dt"]
