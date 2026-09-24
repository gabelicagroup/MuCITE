"""Thread-safe cooperative cancellation owned by the control layer."""

from __future__ import annotations

import threading


class CancellationToken:
    """Small reusable cancellation state with no GUI or worker dependency."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    @property
    def requested(self) -> bool:
        return self._event.is_set()
