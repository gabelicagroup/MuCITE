"""Process-local guard against concurrent reuse of one simulation runtime."""

from __future__ import annotations

import threading
from types import TracebackType


class ConcurrentRunError(RuntimeError):
    """Raised when one runtime is advanced by two engine calls at once."""


class RunGuard:
    """Fail fast instead of allowing concurrent mutation of model state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> "RunGuard":
        if not self._lock.acquire(blocking=False):
            raise ConcurrentRunError(
                "This simulation runtime is already running in another thread."
            )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._lock.release()
