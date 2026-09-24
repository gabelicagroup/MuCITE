"""Aggregate source behavior exposed by the coupled model runtime."""

from __future__ import annotations

from .access import SourceAccessMixin
from .continuous_runtime import ContinuousSourceRuntimeMixin
from .packet_runtime import PacketSourceRuntimeMixin
from .slot_runtime import SourceSlotRuntimeMixin


class SourceRuntimeMixin(
    PacketSourceRuntimeMixin,
    ContinuousSourceRuntimeMixin,
    SourceSlotRuntimeMixin,
    SourceAccessMixin,
):
    """Behavior-only source model mixed into the composition root."""


__all__ = ["SourceRuntimeMixin"]
