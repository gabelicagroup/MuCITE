"""Deterministic random-seed ownership for one simulation runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeedManager:
    """Own physical randomness without changing the established RNG stream.

    Source sampling, collisions, and fragmentation intentionally share one
    ``numpy.random.Generator``. Splitting that stream would change fixed-seed
    trajectories because it changes random-number consumption order.
    """

    root_seed: int
    _physical: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_seed = int(self.root_seed)
        self._physical = np.random.default_rng(self.root_seed)

    @property
    def physical(self) -> np.random.Generator:
        return self._physical

    def reset_physical(self) -> np.random.Generator:
        """Reset explicitly; normal engine reuse never resets implicitly."""

        self._physical = np.random.default_rng(self.root_seed)
        return self._physical

    def derived_seed(self, namespace: str) -> int:
        """Return a stable independent seed for non-physical sampling."""

        payload = f"{self.root_seed}:{namespace}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)
