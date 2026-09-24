"""Static-field stage scheduling glue for the model runtime."""

from __future__ import annotations

from typing import Any


class StageRuntimeMixin:
    """Advance and install scheduled static-field stages."""

    def _current_stage_name(self) -> str:
        if not self.stage_schedule:
            return "static"
        return self.stage_schedule[self.current_stage_index].name

    def _record_stage_activation(
        self,
        stage: Any,
        *,
        event_time_s: float,
        loaded_dc: bool,
    ) -> None:
        self.stage_history.append(
            {
                "name": stage.name,
                "event_time_s": float(event_time_s),
                "start_s": float(stage.start_s),
                "end_s": float(stage.end_s),
                "duration_s": float(stage.duration_s),
                "static_field_path": str(stage.static_field_path),
                "source_enabled": bool(stage.source_enabled),
                "loaded_dc": bool(loaded_dc),
            }
        )

    def _next_stage_boundary_s(self, time_s: float) -> float:
        if not self.stage_schedule:
            return float(self.config.total_time_s)
        return float(self.stage_schedule[self.current_stage_index].end_s)

    def _load_stage_dc_field(self, stage: Any) -> None:
        baked_fields = self._stage_services.load_baked_field_file(
            stage.static_field_path
        )
        self._stage_services.validate_baked_grid_matches_runtime(
            self.static_grid,
            baked_fields,
            stage.static_field_path,
        )
        self._stage_services.load_baked_dc_fields_into_grid(
            self.static_grid,
            baked_fields,
        )
        self.static_field_path = stage.static_field_path
        self._refresh_static_field_cache()

    def _apply_stage_for_time(self, time_s: float) -> None:
        if not self.stage_schedule:
            return
        eps_s = 1.0e-15
        while (
            self.current_stage_index + 1 < len(self.stage_schedule)
            and float(time_s)
            >= self.stage_schedule[self.current_stage_index].end_s - eps_s
        ):
            self.current_stage_index += 1
            stage = self.stage_schedule[self.current_stage_index]
            self._load_stage_dc_field(stage)
            self._current_stage_source_enabled = bool(stage.source_enabled)
            self._record_stage_activation(
                stage,
                event_time_s=float(time_s),
                loaded_dc=True,
            )
