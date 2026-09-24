"""Time-binned terminal current and ion-count presentation helpers."""

from __future__ import annotations

import csv
import heapq
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from scipy.constants import elementary_charge

from .plotting import make_empty_figure


MAX_TERMINAL_UI_BINS = 25_000


def iter_terminal_event_csv(path: Path) -> Iterator[dict[str, str]]:
    """Yield terminal rows without loading a potentially large CSV into memory."""

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _new_bin(index: int, bin_ms: float) -> dict[str, float]:
    start_ms = index * bin_ms
    return {
        "start_ms": start_ms,
        "end_ms": start_ms + bin_ms,
        "active_macro": float("nan"),
        "terminal_macro": 0.0,
        "z_exit_macro": 0.0,
        "z_exit_real_ions": 0.0,
        "z_exit_parent_real_ions": 0.0,
        "z_exit_fragment_real_ions": 0.0,
        "electrode_hit_macro": 0.0,
        "radial_out_macro": 0.0,
        "domain_out_macro": 0.0,
        "fragmented_macro": 0.0,
    }


def _as_float(row: Mapping[str, Any], key: str, fallback: float = 0.0) -> float:
    try:
        value = float(row.get(key, fallback))
    except (TypeError, ValueError):
        return fallback
    return value if np.isfinite(value) else fallback


def _accumulate_event(target: dict[str, float], row: Mapping[str, Any]) -> None:
    status = str(row.get("status", "")).strip()
    target["terminal_macro"] += 1.0
    status_key = f"{status}_macro"
    if status_key in target:
        target[status_key] += 1.0
    if status != "z_exit":
        return
    represented = _as_float(row, "represented_real_ions")
    target["z_exit_real_ions"] += represented
    target["z_exit_parent_real_ions"] += _as_float(
        row, "parent_real_ions_remaining", represented
    )
    target["z_exit_fragment_real_ions"] += _as_float(
        row, "fragmented_real_ions_represented"
    )


def _accumulate_bounded_event(
    by_index: dict[int, dict[str, float]],
    index_heap: list[int],
    row: Mapping[str, Any],
    *,
    index: int,
    bin_ms: float,
    max_bins: int | None,
    latest_index: int,
) -> int:
    latest = max(latest_index, index)
    cutoff = 0 if max_bins is None else max(0, latest - max_bins + 1)
    if index < cutoff:
        return latest
    if index not in by_index:
        by_index[index] = _new_bin(index, bin_ms)
        heapq.heappush(index_heap, index)
    _accumulate_event(by_index[index], row)
    while index_heap and index_heap[0] < cutoff:
        by_index.pop(heapq.heappop(index_heap), None)
    return latest


def _attach_active_counts(
    bins: list[dict[str, float]],
    macro_history: Sequence[Mapping[str, Any]],
) -> None:
    history = sorted(macro_history, key=lambda row: _as_float(row, "time_s"))
    history_index = 0
    active = float("nan")
    for target in bins:
        end_s = target["end_ms"] * 1.0e-3
        while history_index < len(history):
            row = history[history_index]
            if _as_float(row, "time_s") > end_s:
                break
            active = _as_float(row, "ion_count", active)
            history_index += 1
        target["active_macro"] = active


def _attach_currents(
    bins: list[dict[str, float]],
    bin_ms: float,
    charge_state: int,
) -> None:
    scale_na = abs(int(charge_state)) * elementary_charge / (bin_ms * 1.0e-3) * 1.0e9
    for target in bins:
        target["z_exit_current_na"] = target["z_exit_real_ions"] * scale_na
        target["z_exit_parent_current_na"] = (
            target["z_exit_parent_real_ions"] * scale_na
        )
        target["z_exit_fragment_current_na"] = (
            target["z_exit_fragment_real_ions"] * scale_na
        )
        target["loss_macro"] = sum(
            target[key]
            for key in (
                "electrode_hit_macro",
                "radial_out_macro",
                "domain_out_macro",
                "fragmented_macro",
            )
        )


def _finalize_bins(
    by_index: dict[int, dict[str, float]],
    *,
    bin_ms: float,
    charge_state: int,
    macro_history: Sequence[Mapping[str, Any]],
    max_bins: int | None = None,
) -> list[dict[str, float]]:
    bin_s = bin_ms * 1.0e-3
    history_end = max(
        (_as_float(row, "time_s") for row in macro_history), default=0.0
    )
    history_last_index = max(0, int(np.ceil(history_end / bin_s)) - 1)
    last_index = max(max(by_index, default=0), history_last_index)
    first_index = 0 if max_bins is None else max(0, last_index - max_bins + 1)
    bins = [
        by_index.get(index, _new_bin(index, bin_ms))
        for index in range(first_index, last_index + 1)
    ]
    _attach_active_counts(bins, macro_history)
    _attach_currents(bins, bin_ms, charge_state)
    return bins


def aggregate_terminal_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    bin_ms: float,
    charge_state: int,
    macro_history: Sequence[Mapping[str, Any]] = (),
    max_bins: int | None = None,
) -> list[dict[str, float]]:
    """Aggregate terminal events into physical current and count time bins."""

    if not np.isfinite(bin_ms) or bin_ms <= 0.0:
        raise ValueError("Terminal time bin must be positive.")
    if max_bins is not None and max_bins <= 0:
        raise ValueError("Maximum terminal bins must be positive.")
    bin_s = bin_ms * 1.0e-3
    by_index: dict[int, dict[str, float]] = {}
    index_heap: list[int] = []
    latest_index = 0
    for row in rows:
        index = max(0, int(np.floor(_as_float(row, "event_time_s") / bin_s)))
        latest_index = _accumulate_bounded_event(
            by_index,
            index_heap,
            row,
            index=index,
            bin_ms=bin_ms,
            max_bins=max_bins,
            latest_index=latest_index,
        )
    return _finalize_bins(
        by_index,
        bin_ms=bin_ms,
        charge_state=charge_state,
        macro_history=macro_history,
        max_bins=max_bins,
    )


def aggregate_terminal_resolutions(
    rows: Iterable[Mapping[str, Any]],
    *,
    bin_sizes_ms: Sequence[float],
    charge_state: int,
    macro_history: Sequence[Mapping[str, Any]] = (),
    max_bins: int | None = None,
) -> dict[float, list[dict[str, float]]]:
    """Build several terminal resolutions in one streaming pass."""

    sizes = tuple(sorted({float(value) for value in bin_sizes_ms}))
    if not sizes or any(not np.isfinite(value) or value <= 0.0 for value in sizes):
        raise ValueError("Terminal time bins must be positive.")
    if max_bins is not None and max_bins <= 0:
        raise ValueError("Maximum terminal bins must be positive.")
    grouped: dict[float, dict[int, dict[str, float]]] = {
        value: {} for value in sizes
    }
    heaps = {value: [] for value in sizes}
    latest = {value: 0 for value in sizes}
    for row in rows:
        event_time_s = _as_float(row, "event_time_s")
        for bin_ms, by_index in grouped.items():
            index = max(0, int(np.floor(event_time_s / (bin_ms * 1.0e-3))))
            latest[bin_ms] = _accumulate_bounded_event(
                by_index,
                heaps[bin_ms],
                row,
                index=index,
                bin_ms=bin_ms,
                max_bins=max_bins,
                latest_index=latest[bin_ms],
            )
    return {
        bin_ms: _finalize_bins(
            by_index,
            bin_ms=bin_ms,
            charge_state=charge_state,
            macro_history=macro_history,
            max_bins=max_bins,
        )
        for bin_ms, by_index in grouped.items()
    }


def make_terminal_current_figure(
    bins: Sequence[Mapping[str, float]],
    *,
    source_current_a: float,
) -> Figure:
    """Plot source/exit currents and useful weighted-ion-pack counts."""

    if not bins:
        return make_empty_figure("Terminal Current", "No terminal events are available.")
    plotted = bins
    if len(bins) > 25000:
        indices = np.linspace(0, len(bins) - 1, 25000, dtype=np.int64)
        plotted = [bins[int(index)] for index in indices]
    time_ms = np.asarray([row["end_ms"] for row in plotted], dtype=np.float64)
    figure = Figure(figsize=(8.2, 6.0), dpi=100)
    current_axis, count_axis = figure.subplots(2, 1, sharex=True)
    current_axis.axhline(source_current_a * 1.0e9, color="black", linestyle="--", label="source")
    for key, label in (
        ("z_exit_current_na", "exit total"),
        ("z_exit_parent_current_na", "exit parent"),
        ("z_exit_fragment_current_na", "exit fragments"),
    ):
        current_axis.plot(time_ms, [row[key] for row in plotted], label=label)
    current_axis.set_ylabel("current [nA]")
    current_axis.legend(loc="best")
    current_axis.grid(True, alpha=0.22)
    count_axis.plot(time_ms, [row["active_macro"] for row in plotted], label="active")
    count_axis.plot(time_ms, [row["z_exit_macro"] for row in plotted], label="exit/bin")
    count_axis.plot(time_ms, [row["loss_macro"] for row in plotted], label="loss/bin")
    count_axis.set_xlabel("simulation time [ms]")
    count_axis.set_ylabel("weighted ion packs")
    count_axis.legend(loc="best")
    count_axis.grid(True, alpha=0.22)
    figure.tight_layout()
    return figure
