"""Control-layer primitives for deterministic simulation execution."""

from .cancellation import CancellationToken
from .concurrency import ConcurrentRunError, RunGuard
from .events import (
    EventBus,
    MacroStepCompleted,
    RunFinished,
    RunStarted,
    SimulationEvent,
)
from .engine import SimulationEngine
from .policy import EnginePolicy
from .randomness import SeedManager


def release_pic_taichi_runtime() -> None:
    """Release the process-global Taichi runtime on its owner thread."""

    from .simulation import release_pic_taichi_runtime as release_runtime

    release_runtime()

__all__ = [
    "CancellationToken",
    "ConcurrentRunError",
    "EventBus",
    "EnginePolicy",
    "MacroStepCompleted",
    "RunFinished",
    "RunGuard",
    "RunStarted",
    "SeedManager",
    "SimulationEngine",
    "SimulationEvent",
    "release_pic_taichi_runtime",
]
