"""Synchronous lifecycle events emitted by the simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias


@dataclass(frozen=True)
class RunStarted:
    started_at: str


@dataclass(frozen=True)
class MacroStepCompleted:
    time_s: float
    macro_step_index: int
    micro_step_index: int
    active_particle_count: int
    collision_count: int


@dataclass(frozen=True)
class RunFinished:
    ended_at: str
    final_time_s: float
    termination_reason: str


SimulationEvent: TypeAlias = RunStarted | MacroStepCompleted | RunFinished
EventSubscriber: TypeAlias = Callable[[SimulationEvent], None]


class EventBus:
    """In-process observer port; publishing is a no-op without subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[EventSubscriber] = []

    def subscribe(self, subscriber: EventSubscriber) -> Callable[[], None]:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

        return unsubscribe

    def publish(self, event: SimulationEvent) -> None:
        for subscriber in tuple(self._subscribers):
            subscriber(event)
