"""Source-side gas velocity sampling from raw axisymmetric Fluent exports."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", header.strip().lower())


def _read_csv_columns(csv_path: Path) -> dict[str, np.ndarray]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        raw_fieldnames = next(reader, None)
        if raw_fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        normalized_names: list[str] = []
        seen: dict[str, int] = {}
        for raw_name in raw_fieldnames:
            base_name = _normalize_header(raw_name)
            count = seen.get(base_name, 0) + 1
            seen[base_name] = count
            normalized_names.append(base_name if count == 1 else f"{base_name}{count}")

        buckets: dict[str, list[float]] = {key: [] for key in normalized_names}
        for row in reader:
            for index, normalized_name in enumerate(normalized_names):
                raw_value = row[index] if index < len(row) else ""
                buckets[normalized_name].append(np.nan if raw_value is None or raw_value.strip() == "" else float(raw_value))
    return {key: np.asarray(values, dtype=float) for key, values in buckets.items()}


def _pick_column(columns: dict[str, np.ndarray], aliases: Sequence[str], *, csv_path: Path) -> np.ndarray:
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    available = ", ".join(sorted(columns.keys()))
    alias_text = ", ".join(aliases)
    raise KeyError(f"Could not find column [{alias_text}] in {csv_path}; available columns: {available}")


def _validate_sampling_window(
    z_min_m: float,
    z_max_m: float,
    r_min_m: Optional[float],
    r_max_m: Optional[float],
) -> None:
    if z_max_m < z_min_m:
        raise ValueError(
            "source birth velocity z_max must be >= z_min "
            f"(got {z_max_m * 1.0e3:g} mm < {z_min_m * 1.0e3:g} mm)."
        )
    if r_min_m is not None and r_min_m < 0.0:
        raise ValueError("source birth velocity r_min must be non-negative.")
    if r_max_m is not None and r_max_m < 0.0:
        raise ValueError("source birth velocity r_max must be non-negative.")
    if r_min_m is not None and r_max_m is not None and r_max_m < r_min_m:
        raise ValueError("source birth velocity r_max must be >= r_min.")


def _gas_velocity_columns(
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = _read_csv_columns(source_path)
    z_m = _pick_column(
        columns, ("zm", "z", "xcoordinate", "xcoordinatem"),
        csv_path=source_path,
    )
    r_m = _pick_column(
        columns, ("rm", "r", "ycoordinate", "ycoordinatem"),
        csv_path=source_path,
    )
    vz_m_per_s = _pick_column(
        columns,
        ("vzms", "velocityzms", "vzmps", "axialvelocity", "xvelocity", "velocityx"),
        csv_path=source_path,
    )
    vr_m_per_s = _pick_column(
        columns,
        ("vrms", "velocityrms", "vrmps", "radialvelocity", "yvelocity", "velocityy"),
        csv_path=source_path,
    )
    return z_m, r_m, vz_m_per_s, vr_m_per_s


def _sampling_window_mask(
    samples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bounds_m: tuple[float, float, Optional[float], Optional[float]],
) -> np.ndarray:
    z_m, r_m, vz_m_per_s, vr_m_per_s = samples
    z_min_m, z_max_m, r_min_m, r_max_m = bounds_m
    window = (
        np.isfinite(z_m) & np.isfinite(r_m)
        & np.isfinite(vz_m_per_s) & np.isfinite(vr_m_per_s)
        & (z_m >= z_min_m) & (z_m <= z_max_m)
    )
    if r_min_m is not None:
        window &= r_m >= r_min_m
    if r_max_m is not None:
        window &= r_m <= r_max_m
    return window


def _require_sampling_points(
    source_path: Path,
    window: np.ndarray,
    bounds_m: tuple[float, float, Optional[float], Optional[float]],
) -> None:
    if int(np.count_nonzero(window)) >= 3:
        return
    z_min_m, z_max_m, r_min_m, r_max_m = bounds_m
    r_text = ""
    if r_min_m is not None or r_max_m is not None:
        lower = float("-inf") if r_min_m is None else r_min_m * 1.0e3
        upper = float("inf") if r_max_m is None else r_max_m * 1.0e3
        r_text = f", r=[{lower:g}, {upper:g}] mm"
    raise ValueError(
        "Source birth velocity gas CSV has fewer than 3 finite points in "
        f"z=[{z_min_m * 1.0e3:g}, {z_max_m * 1.0e3:g}] mm{r_text}: "
        f"{source_path}"
    )


@dataclass(frozen=True)
class AxisymmetricGasVelocitySampler:
    """Sample gas velocity vectors from raw Fluent ``(z, r)`` data.

    The source CSV convention follows ``field_baker.py``:
    Fluent ``x-coordinate`` is global/source axial ``z`` and Fluent
    ``y-coordinate`` is radius ``r``. This sampler is intentionally source-only;
    it does not replace the baked external gas field used by collisions.
    """

    path: Path
    z_min_m: float
    z_max_m: float
    r_min_m: Optional[float]
    r_max_m: Optional[float]
    source_z_min_m: float
    source_z_max_m: float
    source_r_min_m: float
    source_r_max_m: float
    raw_point_count: int
    retained_point_count: int
    _linear_vz: Any = field(repr=False)
    _linear_vr: Any = field(repr=False)
    _nearest_vz: Any = field(repr=False)
    _nearest_vr: Any = field(repr=False)
    reference_sample_count: int = 101

    @classmethod
    def from_fluent_csv(
        cls,
        path: Path,
        *,
        z_min_m: float,
        z_max_m: float,
        r_min_m: Optional[float] = None,
        r_max_m: Optional[float] = None,
        reference_sample_count: int = 101,
    ) -> "AxisymmetricGasVelocitySampler":
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source birth velocity gas CSV not found: {source_path}")
        bounds_m = (float(z_min_m), float(z_max_m), r_min_m, r_max_m)
        _validate_sampling_window(*bounds_m)
        samples = _gas_velocity_columns(source_path)
        window = _sampling_window_mask(samples, bounds_m)
        _require_sampling_points(source_path, window, bounds_m)
        z_m, r_m, vz_m_per_s, vr_m_per_s = samples
        z_used = z_m[window]
        r_used = r_m[window]
        vz_used = vz_m_per_s[window]
        vr_used = vr_m_per_s[window]
        points = np.column_stack((z_used, r_used))

        return cls(
            path=source_path,
            z_min_m=float(z_min_m),
            z_max_m=float(z_max_m),
            r_min_m=None if r_min_m is None else float(r_min_m),
            r_max_m=None if r_max_m is None else float(r_max_m),
            source_z_min_m=float(np.nanmin(z_used)),
            source_z_max_m=float(np.nanmax(z_used)),
            source_r_min_m=float(np.nanmin(r_used)),
            source_r_max_m=float(np.nanmax(r_used)),
            raw_point_count=int(z_m.size),
            retained_point_count=int(z_used.size),
            _linear_vz=LinearNDInterpolator(points, vz_used, fill_value=np.nan),
            _linear_vr=LinearNDInterpolator(points, vr_used, fill_value=np.nan),
            _nearest_vz=NearestNDInterpolator(points, vz_used),
            _nearest_vr=NearestNDInterpolator(points, vr_used),
            reference_sample_count=int(reference_sample_count),
        )

    def _sample_component(self, linear: Any, nearest: Any, z_m: np.ndarray, r_m: np.ndarray) -> np.ndarray:
        values = np.asarray(linear(z_m, r_m), dtype=np.float64)
        missing = ~np.isfinite(values)
        if np.any(missing):
            values[missing] = nearest(np.asarray(z_m)[missing], np.asarray(r_m)[missing])
        return np.asarray(values, dtype=np.float64)

    def sample_velocity_vectors(self, positions_m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        positions = np.asarray(positions_m, dtype=np.float64)
        velocities = np.zeros((positions.shape[0], 3), dtype=np.float64)
        if positions.size == 0:
            return velocities

        x_m = positions[:, 0]
        y_m = positions[:, 1]
        r_m = np.sqrt(x_m * x_m + y_m * y_m)
        if self.z_max_m > self.z_min_m:
            z_birth_m = rng.uniform(self.z_min_m, self.z_max_m, size=positions.shape[0])
        else:
            z_birth_m = np.full(positions.shape[0], self.z_min_m, dtype=np.float64)

        vr_m_per_s = self._sample_component(self._linear_vr, self._nearest_vr, z_birth_m, r_m)
        vz_m_per_s = self._sample_component(self._linear_vz, self._nearest_vz, z_birth_m, r_m)
        velocities[:, 2] = vz_m_per_s

        radial = r_m > 1.0e-16
        if np.any(radial):
            velocities[radial, 0] = vr_m_per_s[radial] * x_m[radial] / r_m[radial]
            velocities[radial, 1] = vr_m_per_s[radial] * y_m[radial] / r_m[radial]
        return velocities

    def reference_axial_velocity_m_per_s(self) -> float:
        sample_count = (
            int(self.reference_sample_count)
            if self.z_max_m > self.z_min_m
            else 1
        )
        z_m = np.linspace(self.z_min_m, self.z_max_m, sample_count, dtype=np.float64)
        r_m = np.zeros(sample_count, dtype=np.float64)
        vz_m_per_s = self._sample_component(self._linear_vz, self._nearest_vz, z_m, r_m)
        return float(max(np.nanmedian(vz_m_per_s), 0.0))

    def metadata(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "source_path": str(self.path),
            "coordinate_mapping": "fluent_x_to_source_z__fluent_y_to_radius",
            "z_min_m": float(self.z_min_m),
            "z_max_m": float(self.z_max_m),
            "r_min_m": None if self.r_min_m is None else float(self.r_min_m),
            "r_max_m": None if self.r_max_m is None else float(self.r_max_m),
            "source_z_min_m": float(self.source_z_min_m),
            "source_z_max_m": float(self.source_z_max_m),
            "source_r_min_m": float(self.source_r_min_m),
            "source_r_max_m": float(self.source_r_max_m),
            "raw_point_count": int(self.raw_point_count),
            "retained_point_count": int(self.retained_point_count),
            "reference_axial_velocity_m_per_s": float(self.reference_axial_velocity_m_per_s()),
        }
