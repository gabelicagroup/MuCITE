"""Concurrency ownership contracts for one mutable simulation runtime."""

from __future__ import annotations

import threading

from src.core import ConcurrentRunError, RunGuard


def test_run_guard_rejects_overlapping_entry_and_is_reusable() -> None:
    guard = RunGuard()
    entered = threading.Event()
    release = threading.Event()

    def hold_guard() -> None:
        with guard:
            entered.set()
            release.wait(timeout=2.0)

    worker = threading.Thread(target=hold_guard)
    worker.start()
    assert entered.wait(timeout=2.0)
    try:
        with guard:
            raise AssertionError("overlapping entry unexpectedly succeeded")
    except ConcurrentRunError:
        pass
    finally:
        release.set()
        worker.join(timeout=2.0)
    assert not worker.is_alive()
    with guard:
        assert guard.active
    assert not guard.active
