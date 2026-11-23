"""Range checks around IonSPA tables, without modifying the provider."""

from __future__ import annotations

from typing import Any

import numpy as np


class HeatCapacityRangeError(ValueError):
    """A conversion requested values outside the selected heat-capacity table."""

    def __init__(self, profile: str, quantity: str, values: np.ndarray,
                 lower: float, upper: float, invalid: np.ndarray) -> None:
        self.profile = profile
        self.quantity = quantity
        self.lower = lower
        self.upper = upper
        self.invalid_indices = np.flatnonzero(invalid.reshape(-1))
        self.invalid_values = values.reshape(-1)[self.invalid_indices]
        super().__init__(
            f"Heat-capacity profile {profile!r}: {quantity} outside the finite "
            f"table range [{lower:.17g}, {upper:.17g}]; "
            f"invalid flat indices {self.invalid_indices[:8].tolist()}, "
            f"values {self.invalid_values[:8].tolist()}. "
            "Conversion stopped; no endpoint clipping or extrapolation was applied."
        )


class CheckedIonModel:
    """Preserve provider conversions inside their tabulated T/U range."""

    def __init__(self, model: Any, profile: str) -> None:
        self._model = model
        self.profile = profile
        table = getattr(model, "hcprofileobject", None)
        temperatures = np.asarray(getattr(table, "Tsamples", None), dtype=float)
        energies = np.asarray(getattr(table, "UfromTsamples", None), dtype=float)
        if (temperatures.ndim != 1 or temperatures.size < 2
                or energies.shape != temperatures.shape
                or not np.all(np.isfinite(temperatures))
                or not np.all(np.isfinite(energies))
                or np.any(np.diff(temperatures) <= 0)
                or np.any(np.diff(energies) < 0)
                or energies[-1] <= energies[0]):
            raise ValueError(
                f"IonSPA profile {profile!r} must expose finite ordered "
                "hcprofileobject.Tsamples and UfromTsamples for range checking."
            )
        self.temperature_range_k = (float(temperatures[0]), float(temperatures[-1]))
        self.energy_range_j = (float(energies[0]), float(energies[-1]))
        self.range_error_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _check(self, values: Any, quantity: str, bounds: tuple[float, float]) -> None:
        array = np.asarray(values, dtype=np.float64)
        low, high = bounds
        invalid = ~np.isfinite(array) | (array < low) | (array > high)
        if np.any(invalid):
            self.range_error_count += int(np.count_nonzero(invalid))
            raise HeatCapacityRangeError(self.profile, quantity, array, low, high, invalid)

    def UfromT(self, temperature_k: Any) -> Any:
        self._check(temperature_k, "temperature_k [K]", self.temperature_range_k)
        return self._model.UfromT(temperature_k)

    def TfromU(self, internal_energy_j: Any) -> Any:
        self._check(internal_energy_j, "internal_energy_j [J/ion]", self.energy_range_j)
        return self._model.TfromU(internal_energy_j)

    @property
    def range_metadata(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "policy": "reject_out_of_range",
            "temperature_min_k": self.temperature_range_k[0],
            "temperature_max_k": self.temperature_range_k[1],
            "internal_energy_min_j": self.energy_range_j[0],
            "internal_energy_max_j": self.energy_range_j[1],
            "out_of_range_values": self.range_error_count,
        }
