"""Composed terminal-boundary runtime policy."""

from .accounting import TerminalAccountingMixin
from .localization import TerminalLocalizationMixin
from .runtime import TerminalBoundaryRuntimeMixin


class BoundaryRuntimeMixin(
    TerminalAccountingMixin,
    TerminalLocalizationMixin,
    TerminalBoundaryRuntimeMixin,
):
    """Composed accounting, localization, and terminal runtime behavior."""


__all__ = ["BoundaryRuntimeMixin"]
