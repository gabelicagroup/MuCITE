"""Graphical presentation front-end for the MuCITE ion-source runtime."""

from __future__ import annotations

from typing import Optional


def launch_main_window(*, force_backend: Optional[str] = None) -> None:
    """Launch the best available GUI backend.

    The project virtualenv used on some Windows installs does not include
    ``tkinter``. In that case, fall back to the stdlib browser-based workbench
    instead of failing at import time.
    """

    backend = (force_backend or "").strip().lower()
    if backend not in {"", "tk", "web"}:
        raise ValueError("force_backend must be one of: 'tk', 'web', or ''.")

    if backend != "web":
        try:
            from .main_window import launch_main_window as launch_tk_window
        except ModuleNotFoundError as exc:
            if backend == "tk" or exc.name != "tkinter":
                raise
        else:
            launch_tk_window()
            return

    from .web_window import launch_web_window

    launch_web_window()

__all__ = ["launch_main_window"]
