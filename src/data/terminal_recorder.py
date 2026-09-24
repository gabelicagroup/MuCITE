"""Bounded terminal-event collection and optional CSV persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..agents.events import (
    TRANSPORT_TERMINAL_STATUS_CODES,
    TerminalAccounting,
    TerminalEventBatch,
)
from ..config import PARTICLE_ACTIVE, PARTICLE_STATUS_NAMES
from .terminal_rows import (
    NormalizedTerminalBatch,
    build_terminal_rows,
    normalize_terminal_batch,
)
from .terminal_reservoir import TerminalEventReservoir
from .terminal_schema import (
    DEFAULT_TERMINAL_EVENT_SAMPLE_SIZE,
    TERMINAL_EVENT_FIELDNAMES,
)
from .terminal_statistics import TerminalEventStatistics
from .terminal_stream import TerminalCsvStream


def _recorder_diagnostics(
    recorder: "TerminalEventRecorder",
) -> dict[str, Any]:
    return {
        "recorded_rows": int(recorder.recorded_row_count),
        "streamed_rows": int(recorder.streamed_row_count),
        "retained_sample_rows": int(len(recorder.rows)),
        "stream_only": bool(recorder.stream_only),
        "sample_size": int(recorder.sample_size),
        "sample_is_complete": bool(recorder.sample_is_complete),
        "next_event_id": int(recorder.next_event_id),
        "max_rows_reached": bool(recorder.max_rows_reached),
        "recorded_summary_by_status": recorder.recorded_summary_by_status(),
        "omitted_rows_by_status": dict(recorder.omitted_rows_by_status),
        "omitted_real_ions_by_status": dict(
            recorder.omitted_real_ions_by_status
        ),
        "omitted_parent_real_ions_by_status": dict(
            recorder.omitted_parent_real_ions_by_status
        ),
        "omitted_fragment_real_ions_by_status": dict(
            recorder.omitted_fragment_real_ions_by_status
        ),
    }


class TerminalEventRecorder:
    """Coordinate exact accounting, bounded retention, and optional streaming."""

    def __init__(self, *, sample_seed: int = 0) -> None:
        self._reservoir = TerminalEventReservoir(sample_seed=sample_seed)
        self.rows = self._reservoir.rows
        self.statistics = TerminalEventStatistics()
        self._publish_statistics_views()
        self.max_rows_reached = False
        self._recorded_row_count = 0
        self._streamed_row_count = 0
        self._next_event_id = 0
        self._stream_only = False
        self._sample_size = 0
        self._stream = TerminalCsvStream()

    def _publish_statistics_views(self) -> None:
        """Keep historical public dictionaries as identity-stable views."""

        stats = self.statistics
        self.recorded_rows_by_status = stats.recorded_rows_by_status
        self.recorded_real_ions_by_status = stats.recorded_real_ions_by_status
        self.recorded_parent_real_ions_by_status = (
            stats.recorded_parent_real_ions_by_status
        )
        self.recorded_fragment_real_ions_by_status = (
            stats.recorded_fragment_real_ions_by_status
        )
        self.omitted_rows_by_status = stats.omitted_rows_by_status
        self.omitted_real_ions_by_status = stats.omitted_real_ions_by_status
        self.omitted_parent_real_ions_by_status = (
            stats.omitted_parent_real_ions_by_status
        )
        self.omitted_fragment_real_ions_by_status = (
            stats.omitted_fragment_real_ions_by_status
        )

    @property
    def stream_path(self) -> Path | None:
        return self._stream.path

    @property
    def stream_only(self) -> bool:
        return self._stream_only

    @property
    def sample_size(self) -> int:
        return self._sample_size

    @property
    def recorded_row_count(self) -> int:
        return self._recorded_row_count

    @property
    def streamed_row_count(self) -> int:
        return self._streamed_row_count

    @property
    def next_event_id(self) -> int:
        return self._next_event_id

    @property
    def retained_row_count(self) -> int:
        return len(self.rows)

    @property
    def sample_is_complete(self) -> bool:
        return (
            not self._stream_only
            or self._recorded_row_count <= self._sample_size
        )

    def report_rows(self) -> list[dict[str, float | int | str]]:
        """Return retained rows in event order for reports and plots."""

        if not self._stream_only:
            return self._reservoir.report_rows(stream_only=False)
        return self._reservoir.report_rows(stream_only=True)

    def recorded_summary_by_status(
        self,
    ) -> dict[str, dict[str, float | int]]:
        return self.statistics.recorded_summary_by_status()

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded-recorder metadata for JSON/report schemas."""

        return _recorder_diagnostics(self)

    def configure_sample_seed(self, sample_seed: int) -> None:
        """Configure deterministic report sampling before recording begins."""

        if self._recorded_row_count > 0:
            raise RuntimeError(
                "Cannot change terminal sample seed after recording rows."
            )
        self._reservoir.configure_seed(sample_seed)

    def open_csv_stream(
        self,
        path: Path,
        *,
        stream_only: bool = False,
        sample_size: int = DEFAULT_TERMINAL_EVENT_SAMPLE_SIZE,
    ) -> None:
        """Open an incremental terminal CSV with optional reservoir retention."""

        stream_path = Path(path)
        bounded_size = int(sample_size)
        if bounded_size < 0:
            raise ValueError("sample_size must be non-negative.")
        requested_only = bool(stream_only)
        requested_size = bounded_size if requested_only else 0
        if self._same_open_stream(stream_path, requested_only, requested_size):
            return
        if self._recorded_row_count > 0:
            raise RuntimeError(
                "Cannot open or reconfigure a terminal-event stream after "
                "recording rows; call clear()."
            )
        self._stream_only = requested_only
        self._sample_size = requested_size
        self._stream.configure(stream_path)

    def _same_open_stream(
        self,
        path: Path,
        stream_only: bool,
        sample_size: int,
    ) -> bool:
        return (
            self._stream.path == path
            and self._stream.is_open
            and self._stream_only == stream_only
            and self._sample_size == sample_size
        )

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def clear(self) -> None:
        """Reset counters and truncate a previously configured stream."""

        self._reservoir.clear()
        self.statistics.clear()
        self.max_rows_reached = False
        self._recorded_row_count = 0
        self._streamed_row_count = 0
        self._next_event_id = 0
        if self._stream.path is not None:
            self._stream.close()
            self._stream.open()

    def omit_summary(
        self,
        status_code: int,
        macro_event_count: int,
        represented_real_ions: float,
        parent_real_ions_remaining: float | None = None,
        fragmented_real_ions_represented: float | None = None,
    ) -> None:
        """Account for intentionally summarized terminal rows."""

        self.statistics.omit_summary(
            status_code,
            macro_event_count,
            represented_real_ions,
            parent_real_ions_remaining,
            fragmented_real_ions_represented,
        )

    def record(
        self,
        batch: TerminalEventBatch,
        *,
        source_mode: str,
        source_terminal_accounting: TerminalAccounting | None = None,
        terminal_event_mode: str = "transport-only",
        max_terminal_event_rows: int = 0,
    ) -> None:
        """Persist one event batch and update exact additive accounting."""

        normalized = normalize_terminal_batch(batch)
        if normalized.particle_indices.size == 0:
            return
        max_rows = int(max_terminal_event_rows)
        if max_rows < 0:
            raise ValueError(
                "max_terminal_event_rows must be non-negative."
            )
        self._account_source(
            normalized,
            source_mode,
            source_terminal_accounting,
        )
        writable_offsets = self._filter_writable(
            normalized,
            terminal_event_mode,
        )
        writable_offsets = self._apply_row_limit(
            normalized,
            writable_offsets,
            max_rows,
        )
        if writable_offsets.size == 0:
            return
        self._store_rows(normalized, writable_offsets)

    @staticmethod
    def _account_source(
        batch: NormalizedTerminalBatch,
        source_mode: str,
        accounting: TerminalAccounting | None,
    ) -> None:
        if source_mode != "continuous-current" or accounting is None:
            return
        accounting.add_batch(
            batch.status_codes,
            batch.represented_real_ions,
            batch.parent_real_ions_remaining,
            batch.fragmented_real_ions_represented,
        )

    def _filter_writable(
        self,
        batch: NormalizedTerminalBatch,
        mode: str,
    ) -> np.ndarray:
        writable_codes = self._writable_status_codes(mode)
        nonactive = batch.status_codes != PARTICLE_ACTIVE
        writable = nonactive & np.isin(
            batch.status_codes,
            list(writable_codes),
        )
        self._accumulate("omitted", batch, np.flatnonzero(nonactive & ~writable))
        return np.flatnonzero(writable)

    @staticmethod
    def _writable_status_codes(mode: str) -> set[int]:
        normalized = str(mode).strip().lower()
        if normalized == "all":
            return set(PARTICLE_STATUS_NAMES) - {PARTICLE_ACTIVE}
        if normalized == "none":
            return set()
        return set(TRANSPORT_TERMINAL_STATUS_CODES)

    def _apply_row_limit(
        self,
        batch: NormalizedTerminalBatch,
        offsets: np.ndarray,
        max_rows: int,
    ) -> np.ndarray:
        if offsets.size == 0 or max_rows <= 0:
            return offsets
        available = max(max_rows - self._recorded_row_count, 0)
        if available >= offsets.size:
            return offsets
        self.max_rows_reached = True
        self._accumulate("omitted", batch, offsets[available:])
        return offsets[:available]

    def _accumulate(
        self,
        kind: str,
        batch: NormalizedTerminalBatch,
        offsets: np.ndarray,
    ) -> None:
        self.statistics.accumulate(
            kind,  # type: ignore[arg-type]
            batch.status_codes,
            batch.represented_real_ions,
            offsets,
            batch.parent_real_ions_remaining,
            batch.fragmented_real_ions_represented,
        )

    def _store_rows(
        self,
        batch: NormalizedTerminalBatch,
        offsets: np.ndarray,
    ) -> None:
        rows = build_terminal_rows(batch, offsets, self._next_event_id)
        self._streamed_row_count += self._stream.write(rows)
        for row in rows:
            self._reservoir.retain(
                row,
                stream_only=self._stream_only,
                sample_size=self._sample_size,
            )
        self._accumulate("recorded", batch, offsets)
        self._recorded_row_count += len(rows)
        self._next_event_id += len(rows)


__all__ = [
    "DEFAULT_TERMINAL_EVENT_SAMPLE_SIZE",
    "TERMINAL_EVENT_FIELDNAMES",
    "TerminalEventRecorder",
]
