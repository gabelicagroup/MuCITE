"""Cache deterministic k(T) evaluations without approximating temperatures."""

from __future__ import annotations

from typing import Any

import numpy as np


class DissociationRateCache:
    """A slot is reusable only for the same ion, temperature and provider."""

    def __init__(self, runtime: Any, state: dict[str, np.ndarray]) -> None:
        self.adapter = runtime.adapter
        self.template = runtime.template
        method = runtime.adapter.fragmentation_rate
        self.provider = getattr(method, "__func__", method)
        self.temperature_k = np.full(state["temperature"].size, np.nan)
        self.track_ids = np.full(self.temperature_k.size, -1, dtype=np.int64)
        self.rates_s_inverse = np.empty_like(self.temperature_k)

    def matches(self, runtime: Any, state: dict[str, np.ndarray]) -> bool:
        method = runtime.adapter.fragmentation_rate
        return (self.adapter is runtime.adapter and self.template is runtime.template
                and self.provider is getattr(method, "__func__", method)
                and self.temperature_k.size == state["temperature"].size)

    def rates(
        self, runtime: Any, state: dict[str, np.ndarray], indices: np.ndarray,
    ) -> np.ndarray:
        temperatures = state["temperature"][indices]
        track_ids = runtime._particle_track_id[indices]
        changed = ((self.temperature_k[indices] != temperatures)
                   | (self.track_ids[indices] != track_ids))
        selected = indices[changed]
        if selected.size:
            rates = runtime.adapter.fragmentation_rate(temperatures[changed], runtime.template)
            self.rates_s_inverse[selected] = np.broadcast_to(
                np.asarray(rates, dtype=np.float64), selected.shape,
            )
            self.temperature_k[selected] = temperatures[changed]
            self.track_ids[selected] = track_ids[changed]
        diagnostics = runtime.dissociation_runtime_info
        diagnostics["dissociation_rate_evaluated_particles"] += int(selected.size)
        diagnostics["dissociation_rate_cache_hits"] += int(indices.size - selected.size)
        return self.rates_s_inverse[indices]


def cached_fragmentation_rates(
    runtime: Any, state: dict[str, np.ndarray], indices: np.ndarray,
) -> np.ndarray:
    cache = getattr(runtime, "_dissociation_rate_cache", None)
    if cache is None or not cache.matches(runtime, state):
        cache = runtime._dissociation_rate_cache = DissociationRateCache(runtime, state)
    return cache.rates(runtime, state, indices)
