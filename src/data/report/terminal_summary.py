"""Terminal-event accounting and bounded-distribution report summaries."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...config import PARTICLE_STATUS_NAMES
from .values import _weighted_mean, _weighted_quantile


def _summary_header(
    rows: list[dict[str, Any]],
    total_row_count: int | None,
    distribution_sample_is_complete: bool,
) -> dict[str, Any]:
    row_count = int(len(rows) if total_row_count is None else total_row_count)
    sample_complete = bool(
        distribution_sample_is_complete and len(rows) == row_count
    )
    return {
        "row_count": row_count,
        "distribution_sample_row_count": int(len(rows)),
        "distribution_sample_is_complete": sample_complete,
        "distribution_statistics": (
            "exact"
            if sample_complete
            else "bounded_uniform_reservoir_approximation"
        ),
        "tof_reference": (
            "tof_s is measured from capillary-exit activation into the external "
            "transport region. Capillary dwell time is excluded."
        ),
        "by_status": {},
    }


def _ordered_status_names(
    rows: list[dict[str, Any]],
    exact_totals: Mapping[str, Mapping[str, float | int]],
) -> list[str]:
    status_order = {
        name: index for index, name in enumerate(PARTICLE_STATUS_NAMES.values())
    }
    names = (
        {str(row.get("status", "")) for row in rows}
        | {str(name) for name in exact_totals}
    )
    return sorted(names, key=lambda name: status_order.get(name, 999))


def _status_arrays(
    status_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, ...]:
    weights = np.asarray(
        [
            float(row.get("represented_real_ions", 0.0))
            for row in status_rows
        ],
        dtype=np.float64,
    )
    parent_weights = np.asarray(
        [
            float(
                row.get(
                    "parent_real_ions_remaining",
                    row.get("represented_real_ions", 0.0),
                )
            )
            for row in status_rows
        ],
        dtype=np.float64,
    )
    fragment_weights = np.asarray(
        [
            float(row.get("fragmented_real_ions_represented", 0.0))
            for row in status_rows
        ],
        dtype=np.float64,
    )
    tof_us = 1.0e6 * np.asarray(
        [float(row.get("tof_s", float("nan"))) for row in status_rows],
        dtype=np.float64,
    )
    ke_ev = np.asarray(
        [float(row.get("ke_ev", float("nan"))) for row in status_rows],
        dtype=np.float64,
    )
    z_mm = 1.0e3 * np.asarray(
        [float(row.get("z_m", float("nan"))) for row in status_rows],
        dtype=np.float64,
    )
    r_mm = 1.0e3 * np.asarray(
        [float(row.get("r_m", float("nan"))) for row in status_rows],
        dtype=np.float64,
    )
    return weights, parent_weights, fragment_weights, tof_us, ke_ev, z_mm, r_mm


def _exact_or_sum(
    exact_status: Mapping[str, float | int],
    key: str,
    values: np.ndarray,
) -> float:
    return float(
        exact_status.get(key, np.sum(values[np.isfinite(values)]))
    )


def _status_summary(
    status_rows: list[dict[str, Any]],
    exact_status: Mapping[str, float | int],
) -> dict[str, Any]:
    weights, parent, fragments, tof_us, ke_ev, z_mm, r_mm = _status_arrays(
        status_rows
    )
    return {
        "macro_event_count": int(
            exact_status.get("macro_event_count", len(status_rows))
        ),
        "represented_real_ions": _exact_or_sum(
            exact_status,
            "represented_real_ions",
            weights,
        ),
        "parent_real_ions_remaining": _exact_or_sum(
            exact_status,
            "parent_real_ions_remaining",
            parent,
        ),
        "fragmented_real_ions_represented": _exact_or_sum(
            exact_status,
            "fragmented_real_ions_represented",
            fragments,
        ),
        "distribution_sample_macro_event_count": int(len(status_rows)),
        "tof_us_mean_weighted": _weighted_mean(tof_us, weights),
        "tof_us_p05_weighted": _weighted_quantile(tof_us, weights, 0.05),
        "tof_us_p50_weighted": _weighted_quantile(tof_us, weights, 0.50),
        "tof_us_p95_weighted": _weighted_quantile(tof_us, weights, 0.95),
        "ke_ev_mean_weighted": _weighted_mean(ke_ev, weights),
        "ke_ev_p50_weighted": _weighted_quantile(ke_ev, weights, 0.50),
        "ke_ev_p95_weighted": _weighted_quantile(ke_ev, weights, 0.95),
        "z_mm_p50_weighted": _weighted_quantile(z_mm, weights, 0.50),
        "z_mm_p95_weighted": _weighted_quantile(z_mm, weights, 0.95),
        "r_mm_p50_weighted": _weighted_quantile(r_mm, weights, 0.50),
        "r_mm_p95_weighted": _weighted_quantile(r_mm, weights, 0.95),
    }


def _terminal_event_summary(
    rows: list[dict[str, Any]],
    *,
    total_row_count: int | None = None,
    exact_by_status: Mapping[str, Mapping[str, float | int]] | None = None,
    distribution_sample_is_complete: bool = True,
) -> dict[str, Any]:
    """Summarize complete terminal rows or a bounded distribution sample."""

    exact_totals = {} if exact_by_status is None else dict(exact_by_status)
    summary = _summary_header(
        rows,
        total_row_count,
        distribution_sample_is_complete,
    )
    if not rows and not exact_totals:
        return summary
    for status in _ordered_status_names(rows, exact_totals):
        status_rows = [
            row for row in rows if str(row.get("status", "")) == status
        ]
        summary["by_status"][status] = _status_summary(
            status_rows,
            exact_totals.get(status, {}),
        )
    return summary
