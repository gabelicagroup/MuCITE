"""Incremental CSV stream owned by the terminal-event output layer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from .terminal_schema import TERMINAL_EVENT_FIELDNAMES


class TerminalCsvStream:
    """Own one optional terminal-event CSV handle."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.handle: TextIO | None = None
        self.writer: csv.DictWriter[TextIO] | None = None

    @property
    def is_open(self) -> bool:
        return self.handle is not None

    def configure(self, path: Path) -> None:
        self.close()
        self.path = Path(path)
        self.open()

    def open(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=TERMINAL_EVENT_FIELDNAMES,
        )
        self.writer.writeheader()
        self.handle.flush()

    def write(self, rows: list[dict[str, float | int | str]]) -> int:
        if self.writer is None or not rows:
            return 0
        self.writer.writerows(rows)
        self.flush()
        return len(rows)

    def flush(self) -> None:
        if self.handle is not None:
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
        self.handle = None
        self.writer = None
