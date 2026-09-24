"""Thread lifecycle and message transport for GUI background tasks."""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable


class TaskStopped(RuntimeError):
    """Raised by progress callbacks when the user requests cancellation."""


@dataclass
class WorkerHandle:
    kind: str
    thread: threading.Thread
    stop_event: threading.Event


def _post(
    message_queue: "queue.Queue[dict[str, Any]]",
    message_type: str,
    **payload: Any,
) -> None:
    payload["type"] = message_type
    message_queue.put(payload)


def start_worker(
    message_queue: "queue.Queue[dict[str, Any]]",
    kind: str,
    target: Callable[..., None],
    *args: Any,
) -> WorkerHandle:
    stop_event = threading.Event()

    def runner() -> None:
        _post(message_queue, "task_started", kind=kind)
        try:
            target(message_queue, stop_event, *args)
        except TaskStopped:
            _post(message_queue, "task_stopped", kind=kind)
        except Exception:
            _post(
                message_queue,
                "task_failed",
                kind=kind,
                error=traceback.format_exc(),
            )
        finally:
            _post(message_queue, "task_done", kind=kind)

    thread = threading.Thread(
        target=runner,
        name=f"mucite-gui-{kind}",
        daemon=False,
    )
    thread.start()
    return WorkerHandle(kind=kind, thread=thread, stop_event=stop_event)
