"""Fixed-memory online statistics owned by the control layer."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_DT_QUANTILE_RESERVOIR_SIZE = 8_192
DEFAULT_MACRO_HISTORY_SAMPLE_SIZE = 4_096
DEFAULT_MACRO_SUMMARY_FIELDS = (
    "time_s",
    "ion_count",
    "mean_internal_temperature_k",
    "mean_axial_position_m",
    "collision_count",
    "poisson_iterations",
    "poisson_relative_residual",
    "poisson_runtime_ms",
    "terminal_count",
    "sce_external_ratio_p95",
)


class OnlineScalarStatistics:
    """Exact count/min/max/mean using constant memory."""

    def __init__(self) -> None:
        self.count = 0
        self.minimum = float("nan")
        self.maximum = float("nan")
        self.mean = float("nan")

    def clear(self) -> None:
        self.count = 0
        self.minimum = float("nan")
        self.maximum = float("nan")
        self.mean = float("nan")

    def update(self, value: float) -> None:
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError("Online statistics require finite values.")
        if self.count == 0:
            self.count = 1
            self.minimum = scalar
            self.maximum = scalar
            self.mean = scalar
            return
        self.count += 1
        self.minimum = min(self.minimum, scalar)
        self.maximum = max(self.maximum, scalar)
        self.mean += (scalar - self.mean) / float(self.count)

    def update_many(self, values: Sequence[float] | np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        if not np.all(np.isfinite(array)):
            raise ValueError("Online statistics require finite values.")
        batch_count = int(array.size)
        batch_mean = float(np.mean(array))
        batch_min = float(np.min(array))
        batch_max = float(np.max(array))
        if self.count == 0:
            self.count = batch_count
            self.minimum = batch_min
            self.maximum = batch_max
            self.mean = batch_mean
            return
        combined_count = self.count + batch_count
        self.mean += (batch_mean - self.mean) * (batch_count / float(combined_count))
        self.minimum = min(self.minimum, batch_min)
        self.maximum = max(self.maximum, batch_max)
        self.count = combined_count

    def summary(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "mean": float(self.mean),
        }


class FixedQuantileReservoir:
    """Uniform Algorithm-R reservoir for approximate fixed-memory quantiles."""

    def __init__(self, capacity: int = DEFAULT_DT_QUANTILE_RESERVOIR_SIZE, *, seed: int = 0) -> None:
        bounded_capacity = int(capacity)
        if bounded_capacity < 0:
            raise ValueError("capacity must be non-negative.")
        self.capacity = bounded_capacity
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._values = np.empty(self.capacity, dtype=np.float64)
        self.seen_count = 0
        self.sample_count = 0

    def clear(self) -> None:
        self._rng = random.Random(self._seed)
        self.seen_count = 0
        self.sample_count = 0

    def update(self, value: float) -> None:
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError("Quantile reservoir requires finite values.")
        self.seen_count += 1
        if self.capacity == 0:
            return
        if self.sample_count < self.capacity:
            self._values[self.sample_count] = scalar
            self.sample_count += 1
            return
        replacement_index = self._rng.randrange(self.seen_count)
        if replacement_index < self.capacity:
            self._values[replacement_index] = scalar

    def update_many(self, values: Sequence[float] | np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError("Quantile reservoir requires finite values.")
        for value in array:
            self.update(float(value))

    def values(self) -> np.ndarray:
        return self._values[: self.sample_count].copy()

    def quantile(self, quantile: float) -> float:
        if self.sample_count == 0:
            return float("nan")
        bounded_quantile = float(quantile)
        if not 0.0 <= bounded_quantile <= 1.0:
            raise ValueError("quantile must be in the closed interval [0, 1].")
        return float(np.quantile(self._values[: self.sample_count], bounded_quantile))


class OnlineTimeStepStatistics:
    """Exact dt moments and limiter counts plus bounded approximate quantiles."""

    def __init__(
        self,
        *,
        quantile_reservoir_size: int = DEFAULT_DT_QUANTILE_RESERVOIR_SIZE,
        seed: int = 0,
        limiter_names: Iterable[str] = (),
    ) -> None:
        self._moments = OnlineScalarStatistics()
        self._quantiles = FixedQuantileReservoir(quantile_reservoir_size, seed=seed)
        self._initial_limiter_names = tuple(str(name) for name in limiter_names)
        self._limiter_counts = {name: 0 for name in self._initial_limiter_names}
        self._raw_limiter_counts = {name: 0 for name in self._initial_limiter_names}

    @property
    def count(self) -> int:
        return self._moments.count

    @property
    def quantile_sample_count(self) -> int:
        return self._quantiles.sample_count

    @property
    def quantile_reservoir_size(self) -> int:
        return self._quantiles.capacity

    @property
    def limiter_counts(self) -> dict[str, int]:
        return dict(self._limiter_counts)

    @property
    def raw_limiter_counts(self) -> dict[str, int]:
        return dict(self._raw_limiter_counts)

    def clear(self) -> None:
        self._moments.clear()
        self._quantiles.clear()
        self._limiter_counts = {name: 0 for name in self._initial_limiter_names}
        self._raw_limiter_counts = {name: 0 for name in self._initial_limiter_names}

    @staticmethod
    def _validated_dt(dt_s: float) -> float:
        value = float(dt_s)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"dt_s must be finite and positive, got {value!r}.")
        return value

    @staticmethod
    def _increment(counter: dict[str, int], name: str | None) -> None:
        if name is None:
            return
        key = str(name)
        counter[key] = counter.get(key, 0) + 1

    def update(
        self,
        dt_s: float,
        *,
        limiter: str | None = None,
        raw_limiter: str | None = None,
    ) -> None:
        value = self._validated_dt(dt_s)
        self._moments.update(value)
        self._quantiles.update(value)
        self._increment(self._limiter_counts, limiter)
        self._increment(self._raw_limiter_counts, raw_limiter)

    def update_many(
        self,
        dt_values_s: Sequence[float] | np.ndarray,
        *,
        limiters: Sequence[str | None] | None = None,
        raw_limiters: Sequence[str | None] | None = None,
    ) -> None:
        values = np.asarray(dt_values_s, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("All dt values must be finite and positive.")
        if limiters is not None and len(limiters) != values.size:
            raise ValueError("limiters must have the same length as dt_values_s.")
        if raw_limiters is not None and len(raw_limiters) != values.size:
            raise ValueError("raw_limiters must have the same length as dt_values_s.")
        self._moments.update_many(values)
        self._quantiles.update_many(values)
        if limiters is not None:
            for limiter in limiters:
                self._increment(self._limiter_counts, limiter)
        if raw_limiters is not None:
            for limiter in raw_limiters:
                self._increment(self._raw_limiter_counts, limiter)

    def summary(self) -> dict[str, float | int | bool]:
        return {
            "micro_step_count": int(self._moments.count),
            "mean_dt_s": float(self._moments.mean),
            "p50_dt_s": self._quantiles.quantile(0.50),
            "p95_dt_s": self._quantiles.quantile(0.95),
            "min_dt_s": float(self._moments.minimum),
            "max_dt_s": float(self._moments.maximum),
            "quantile_sample_count": int(self._quantiles.sample_count),
            "quantile_reservoir_size": int(self._quantiles.capacity),
            "quantiles_approximate": bool(self._moments.count > self._quantiles.sample_count),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "dt_statistics": self.summary(),
            "dt_limiter_counts": self.limiter_counts,
            "dt_raw_limiter_counts": self.raw_limiter_counts,
        }


class BoundedMacroHistory:
    """Keep first/last rows, a uniform interior sample, and exact scalar summaries."""

    def __init__(
        self,
        max_sample_rows: int = DEFAULT_MACRO_HISTORY_SAMPLE_SIZE,
        *,
        summary_fields: Iterable[str] = DEFAULT_MACRO_SUMMARY_FIELDS,
        seed: int = 0,
    ) -> None:
        bounded_size = int(max_sample_rows)
        if bounded_size < 0:
            raise ValueError("max_sample_rows must be non-negative.")
        self.max_sample_rows = bounded_size
        self.summary_fields = tuple(dict.fromkeys(str(field) for field in summary_fields))
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._field_statistics = {field: OnlineScalarStatistics() for field in self.summary_fields}
        self._count = 0
        self._first: tuple[int, dict[str, Any]] | None = None
        self._last: tuple[int, dict[str, Any]] | None = None
        self._interior_seen_count = 0
        self._interior: list[tuple[int, dict[str, Any]]] = []

    @property
    def count(self) -> int:
        return self._count

    @property
    def retained_row_count(self) -> int:
        return len(self.sample_rows())

    @property
    def sample_is_complete(self) -> bool:
        return self._count <= self.max_sample_rows

    def clear(self) -> None:
        self._rng = random.Random(self._seed)
        for statistics in self._field_statistics.values():
            statistics.clear()
        self._count = 0
        self._first = None
        self._last = None
        self._interior_seen_count = 0
        self._interior.clear()

    def _interior_capacity(self) -> int:
        return max(self.max_sample_rows - 2, 0)

    def _consider_interior(self, item: tuple[int, dict[str, Any]]) -> None:
        self._interior_seen_count += 1
        capacity = self._interior_capacity()
        if capacity == 0:
            return
        if len(self._interior) < capacity:
            self._interior.append(item)
            return
        replacement_index = self._rng.randrange(self._interior_seen_count)
        if replacement_index < capacity:
            self._interior[replacement_index] = item

    def record(self, row: Mapping[str, Any]) -> None:
        copied_row = dict(row)
        sequence_id = self._count
        if self._last is None:
            self._first = (sequence_id, copied_row)
        else:
            previous_last = self._last
            if previous_last[0] != 0:
                self._consider_interior(previous_last)
        self._last = (sequence_id, copied_row)
        self._count += 1

        for field, statistics in self._field_statistics.items():
            if field not in copied_row:
                continue
            try:
                value = float(copied_row[field])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                statistics.update(value)

    def sample_rows(self) -> list[dict[str, Any]]:
        if self.max_sample_rows == 0 or self._last is None:
            return []
        if self.max_sample_rows == 1:
            return [dict(self._last[1])]
        retained: list[tuple[int, dict[str, Any]]] = []
        if self._first is not None:
            retained.append(self._first)
        retained.extend(self._interior)
        if self._last != self._first:
            retained.append(self._last)
        retained.sort(key=lambda item: item[0])
        return [dict(row) for _, row in retained]

    def summary(self) -> dict[str, Any]:
        sample = self.sample_rows()
        return {
            "macro_step_count": int(self._count),
            "sample_row_count": int(len(sample)),
            "max_sample_rows": int(self.max_sample_rows),
            "sample_is_complete": bool(self.sample_is_complete),
            "sampling": "complete" if self.sample_is_complete else "first_last_uniform_interior_reservoir",
            "first": None if self._first is None else dict(self._first[1]),
            "last": None if self._last is None else dict(self._last[1]),
            "numeric_fields": {
                field: statistics.summary()
                for field, statistics in self._field_statistics.items()
            },
        }
