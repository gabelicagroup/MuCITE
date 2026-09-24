"""Field cache, PIC macro-update, and sampling delegates."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ...config import LocalStateBatch


class FieldRuntimeMixin:
    """Expose environment field operations required by the control engine."""

    def _refresh_static_field_cache(self) -> None:
        self.field_sampler.refresh_static()

    def _refresh_pic_field_cache(self) -> None:
        self.field_sampler.refresh_pic()

    def replace_static_fields(self, updater: Callable[[Any], None]) -> None:
        """Apply a static-grid update and refresh CPU field caches."""

        self.field_sampler.replace_static_fields(updater)

    def _refresh_field_cache(self) -> None:
        self.field_sampler.refresh_all()

    def _update_space_charge_fields(self) -> Optional[Any]:
        self.pic_grid.clear_charge()
        self.pic_grid.clear_space_charge_solution()
        if self.pic_space_charge_scale <= 0.0:
            self.pic_solver.last_poisson_result = None
            self._refresh_pic_field_cache()
            return None
        self.pic_solver.scatter_charge(self.cloud, self.pic_grid)
        result = self.pic_solver.solve_poisson(self.pic_grid)
        if not np.isclose(
            self.pic_space_charge_scale,
            1.0,
            rtol=0.0,
            atol=1.0e-15,
        ):
            self.pic_solver.scale_space_charge_solution(
                self.pic_grid,
                scale=self.pic_space_charge_scale,
            )
        self._refresh_pic_field_cache()
        return result

    def _scalar_bilinear(
        self,
        field_name: str,
        r_m: float,
        z_m: float,
    ) -> float:
        return self.field_sampler.scalar_bilinear(field_name, r_m, z_m)

    def _scalar_bilinear_many(
        self,
        field_name: str,
        r_m: np.ndarray,
        z_m: np.ndarray,
    ) -> np.ndarray:
        return self.field_sampler.scalar_bilinear_many(field_name, r_m, z_m)

    def _rf_modulation(self, time_s: float) -> float:
        return self.field_sampler.rf_modulation(time_s)

    def _sample_local_state_batch(
        self,
        positions_m: np.ndarray,
        time_s: float,
    ) -> LocalStateBatch:
        return self.field_sampler.sample_local_state_batch(positions_m, time_s)

    def _space_charge_external_ratio_p95(
        self,
        state: dict[str, np.ndarray],
        time_s: float,
    ) -> float:
        return self.field_sampler.space_charge_external_ratio_p95(
            state,
            time_s,
        )
