"""Non-physical cadence and bounded-memory policy for one engine run."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

from .metrics import (
    DEFAULT_DT_QUANTILE_RESERVOIR_SIZE,
    DEFAULT_MACRO_HISTORY_SAMPLE_SIZE,
)


@dataclass(frozen=True)
class EnginePolicy:
    """Control observability cost without changing the physical request."""

    progress_interval_s: float = 0.25
    dt_quantile_reservoir_size: int = DEFAULT_DT_QUANTILE_RESERVOIR_SIZE
    macro_history_max_rows: int = DEFAULT_MACRO_HISTORY_SAMPLE_SIZE

    def __post_init__(self) -> None:
        interval = self.progress_interval_s
        if (
            isinstance(interval, bool)
            or not isinstance(interval, Real)
            or not math.isfinite(float(interval))
            or float(interval) <= 0.0
        ):
            raise ValueError("progress_interval_s must be finite and positive.")
        for field_name in (
            "dt_quantile_reservoir_size",
            "macro_history_max_rows",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer."
                )
