"""Exact additive accounting for recorded and omitted terminal events."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..config import PARTICLE_ACTIVE, PARTICLE_STATUS_NAMES

StatisticKind = Literal["recorded", "omitted"]


class TerminalEventStatistics:
    """Own additive terminal-event counters independently of row retention."""

    def __init__(self) -> None:
        self.recorded_rows_by_status: dict[str, int] = {}
        self.recorded_real_ions_by_status: dict[str, float] = {}
        self.recorded_parent_real_ions_by_status: dict[str, float] = {}
        self.recorded_fragment_real_ions_by_status: dict[str, float] = {}
        self.omitted_rows_by_status: dict[str, int] = {}
        self.omitted_real_ions_by_status: dict[str, float] = {}
        self.omitted_parent_real_ions_by_status: dict[str, float] = {}
        self.omitted_fragment_real_ions_by_status: dict[str, float] = {}

    def clear(self) -> None:
        for mapping in self._all_mappings():
            mapping.clear()

    def _all_mappings(self) -> tuple[dict[str, int | float], ...]:
        return (
            self.recorded_rows_by_status,
            self.recorded_real_ions_by_status,
            self.recorded_parent_real_ions_by_status,
            self.recorded_fragment_real_ions_by_status,
            self.omitted_rows_by_status,
            self.omitted_real_ions_by_status,
            self.omitted_parent_real_ions_by_status,
            self.omitted_fragment_real_ions_by_status,
        )

    def recorded_summary_by_status(
        self,
    ) -> dict[str, dict[str, float | int]]:
        statuses = set(self.recorded_rows_by_status)
        statuses.update(self.recorded_real_ions_by_status)
        statuses.update(self.recorded_parent_real_ions_by_status)
        statuses.update(self.recorded_fragment_real_ions_by_status)
        return {
            status: self._recorded_summary(status)
            for status in statuses
        }

    def _recorded_summary(self, status: str) -> dict[str, float | int]:
        return {
            "macro_event_count": int(
                self.recorded_rows_by_status.get(status, 0)
            ),
            "represented_real_ions": float(
                self.recorded_real_ions_by_status.get(status, 0.0)
            ),
            "parent_real_ions_remaining": float(
                self.recorded_parent_real_ions_by_status.get(status, 0.0)
            ),
            "fragmented_real_ions_represented": float(
                self.recorded_fragment_real_ions_by_status.get(status, 0.0)
            ),
        }

    def accumulate(
        self,
        kind: StatisticKind,
        status_codes: np.ndarray,
        represented_real_ions: np.ndarray,
        offsets: np.ndarray,
        parent_real_ions_remaining: np.ndarray | None = None,
        fragmented_real_ions_represented: np.ndarray | None = None,
    ) -> None:
        if offsets.size == 0:
            return
        codes = np.asarray(status_codes[offsets], dtype=np.int16)
        weights = np.asarray(represented_real_ions[offsets], dtype=np.float64)
        parent = self._selected(parent_real_ions_remaining, offsets)
        fragments = self._selected(
            fragmented_real_ions_represented,
            offsets,
        )
        for code in np.unique(codes):
            self._accumulate_status(kind, int(code), codes, weights, parent, fragments)

    @staticmethod
    def _selected(
        values: np.ndarray | None,
        offsets: np.ndarray,
    ) -> np.ndarray | None:
        if values is None:
            return None
        return np.asarray(values[offsets], dtype=np.float64)

    def _accumulate_status(
        self,
        kind: StatisticKind,
        code: int,
        codes: np.ndarray,
        weights: np.ndarray,
        parent: np.ndarray | None,
        fragments: np.ndarray | None,
    ) -> None:
        if code == PARTICLE_ACTIVE:
            return
        status = PARTICLE_STATUS_NAMES.get(code, str(code))
        mask = codes == code
        rows, real_ions, parent_ions, fragment_ions = self._kind_maps(kind)
        rows[status] = rows.get(status, 0) + int(np.count_nonzero(mask))
        real_ions[status] = real_ions.get(status, 0.0) + float(
            np.sum(weights[mask])
        )
        if parent is not None:
            parent_ions[status] = parent_ions.get(status, 0.0) + float(
                np.sum(parent[mask])
            )
        if fragments is not None:
            fragment_ions[status] = fragment_ions.get(status, 0.0) + float(
                np.sum(fragments[mask])
            )

    def _kind_maps(
        self,
        kind: StatisticKind,
    ) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, float]]:
        if kind == "recorded":
            return (
                self.recorded_rows_by_status,
                self.recorded_real_ions_by_status,
                self.recorded_parent_real_ions_by_status,
                self.recorded_fragment_real_ions_by_status,
            )
        return (
            self.omitted_rows_by_status,
            self.omitted_real_ions_by_status,
            self.omitted_parent_real_ions_by_status,
            self.omitted_fragment_real_ions_by_status,
        )

    def omit_summary(
        self,
        status_code: int,
        macro_event_count: int,
        represented_real_ions: float,
        parent_real_ions_remaining: float | None = None,
        fragmented_real_ions_represented: float | None = None,
    ) -> None:
        macro_count = int(macro_event_count)
        if macro_count <= 0:
            return
        status = PARTICLE_STATUS_NAMES.get(
            int(status_code),
            str(status_code),
        )
        self.omitted_rows_by_status[status] = (
            self.omitted_rows_by_status.get(status, 0) + macro_count
        )
        self.omitted_real_ions_by_status[status] = (
            self.omitted_real_ions_by_status.get(status, 0.0)
            + float(represented_real_ions)
        )
        self._add_optional_omitted(
            status,
            parent_real_ions_remaining,
            fragmented_real_ions_represented,
        )

    def _add_optional_omitted(
        self,
        status: str,
        parent_real_ions: float | None,
        fragmented_real_ions: float | None,
    ) -> None:
        if parent_real_ions is not None:
            self.omitted_parent_real_ions_by_status[status] = (
                self.omitted_parent_real_ions_by_status.get(status, 0.0)
                + float(parent_real_ions)
            )
        if fragmented_real_ions is not None:
            self.omitted_fragment_real_ions_by_status[status] = (
                self.omitted_fragment_real_ions_by_status.get(status, 0.0)
                + float(fragmented_real_ions)
            )
