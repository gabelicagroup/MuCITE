"""Pure Cartesian segment geometry used by terminal-event policies."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _segment_endpoints(
    start_m: np.ndarray,
    end_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start_m, dtype=np.float64)
    end = np.asarray(end_m, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("Segment endpoints must have shape (3,).")
    return start, end


def segment_plane_crossing_fraction(
    start_m: np.ndarray,
    end_m: np.ndarray,
    plane_z_m: float,
    *,
    direction: int = 1,
) -> Optional[float]:
    """Return the first forward/backward axial-plane crossing fraction."""

    start, end = _segment_endpoints(start_m, end_m)
    if direction not in {-1, 1}:
        raise ValueError("direction must be +1 or -1.")
    z0 = float(start[2])
    z1 = float(end[2])
    crossed = z0 < plane_z_m <= z1 if direction > 0 else z0 > plane_z_m >= z1
    if not crossed or z1 == z0:
        return None
    return float(np.clip((plane_z_m - z0) / (z1 - z0), 0.0, 1.0))


def segment_cylinder_exit_fraction(
    start_m: np.ndarray,
    end_m: np.ndarray,
    radius_m: float,
) -> Optional[float]:
    """Return where a segment first exits ``x²+y² <= radius²``."""

    start, end = _segment_endpoints(start_m, end_m)
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive.")
    xy0 = start[:2]
    dxy = end[:2] - xy0
    radius_sq = float(radius_m) ** 2
    if float(np.dot(xy0, xy0)) > radius_sq:
        return None
    if float(np.dot(end[:2], end[:2])) <= radius_sq:
        return None
    a = float(np.dot(dxy, dxy))
    if a <= 0.0:
        return None
    b = 2.0 * float(np.dot(xy0, dxy))
    c = float(np.dot(xy0, xy0)) - radius_sq
    discriminant = max(0.0, b * b - 4.0 * a * c)
    roots = (
        (-b - np.sqrt(discriminant)) / (2.0 * a),
        (-b + np.sqrt(discriminant)) / (2.0 * a),
    )
    admissible = [root for root in roots if 0.0 <= root <= 1.0]
    return None if not admissible else float(min(admissible))


def locate_first_segment_hit(
    start_m: np.ndarray,
    end_m: np.ndarray,
    is_hit: Callable[[np.ndarray], bool],
    *,
    max_sample_spacing_m: float,
    bisection_iterations: int = 24,
) -> Optional[float]:
    """Locate the first sampled false-to-true entry along a segment."""

    start, end = _segment_endpoints(start_m, end_m)
    if not np.isfinite(max_sample_spacing_m) or max_sample_spacing_m <= 0.0:
        raise ValueError("max_sample_spacing_m must be finite and positive.")
    if bisection_iterations < 0:
        raise ValueError("bisection_iterations must be non-negative.")
    if bool(is_hit(start)):
        return 0.0
    delta = end - start
    length_m = float(np.linalg.norm(delta))
    if length_m == 0.0:
        return None
    sample_count = max(1, int(np.ceil(length_m / max_sample_spacing_m)))
    lower_fraction = 0.0
    for sample_index in range(1, sample_count + 1):
        upper_fraction = sample_index / sample_count
        if bool(is_hit(start + upper_fraction * delta)):
            return _bisect_hit(
                start,
                delta,
                is_hit,
                lower_fraction,
                upper_fraction,
                bisection_iterations,
            )
        lower_fraction = upper_fraction
    return None


def _bisect_hit(
    start: np.ndarray,
    delta: np.ndarray,
    is_hit: Callable[[np.ndarray], bool],
    lower: float,
    upper: float,
    iterations: int,
) -> float:
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if bool(is_hit(start + midpoint * delta)):
            upper = midpoint
        else:
            lower = midpoint
    return float(upper)


def interpolate_segment_state(
    start_position_m: np.ndarray,
    end_position_m: np.ndarray,
    start_velocity_m_per_s: np.ndarray,
    end_velocity_m_per_s: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate position and velocity at an event fraction."""

    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1].")
    x0, x1 = _segment_endpoints(start_position_m, end_position_m)
    v0, v1 = _segment_endpoints(
        start_velocity_m_per_s,
        end_velocity_m_per_s,
    )
    return x0 + fraction * (x1 - x0), v0 + fraction * (v1 - v0)
