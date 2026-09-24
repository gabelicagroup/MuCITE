"""Bounded in-memory retention for streamed terminal events."""

from __future__ import annotations

import random


TerminalRow = dict[str, float | int | str]


class TerminalEventReservoir:
    """Retain all rows or a deterministic uniform reservoir."""

    def __init__(self, *, sample_seed: int = 0) -> None:
        self.rows: list[TerminalRow] = []
        self._sample_seed = int(sample_seed)
        self._sample_rng = random.Random(self._sample_seed)
        self._seen_count = 0

    def clear(self) -> None:
        self.rows.clear()
        self._sample_rng = random.Random(self._sample_seed)
        self._seen_count = 0

    def configure_seed(self, sample_seed: int) -> None:
        """Set the non-physical sampling seed before retaining any rows."""

        if self._seen_count > 0 or self.rows:
            raise RuntimeError(
                "Cannot change terminal sample seed after retaining rows."
            )
        self._sample_seed = int(sample_seed)
        self.clear()

    def retain(
        self,
        row: TerminalRow,
        *,
        stream_only: bool,
        sample_size: int,
    ) -> None:
        if not stream_only:
            self.rows.append(row)
            return
        self._seen_count += 1
        if sample_size <= 0:
            return
        if len(self.rows) < sample_size:
            self.rows.append(row)
            return
        replacement = self._sample_rng.randrange(self._seen_count)
        if replacement < sample_size:
            self.rows[replacement] = row

    def report_rows(self, *, stream_only: bool) -> list[TerminalRow]:
        if not stream_only:
            return list(self.rows)
        return sorted(self.rows, key=lambda row: int(row["event_id"]))
