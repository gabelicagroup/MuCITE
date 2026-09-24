"""Construct the established backend/configuration status message."""

from __future__ import annotations

from typing import Any

from .artifacts import RuntimeArtifacts
from .services import BootstrapServices


def _source_birth_velocity_message(owner: Any) -> str:
    metadata = owner.source_birth_velocity_metadata
    if not metadata.get("loaded"):
        return ""
    r_max_m = metadata.get("r_max_m")
    r_message = (
        ""
        if r_max_m is None
        else f", r<= {float(r_max_m) * 1.0e3:.3f} mm"
    )
    return (
        " Source birth velocity samples raw Fluent gas over "
        f"z=[{metadata['z_min_m'] * 1.0e3:.3f}, "
        f"{metadata['z_max_m'] * 1.0e3:.3f}] mm"
        f"{r_message}."
    )


def _stage_message(artifacts: RuntimeArtifacts) -> str:
    if not artifacts.stage_schedule:
        return ""
    stage_names = ", ".join(
        stage.name for stage in artifacts.stage_schedule
    )
    return (
        f" Stage schedule loaded from {artifacts.stage_schedule_path} "
        f"({len(artifacts.stage_schedule)} stages: {stage_names}); "
        "total_time follows schedule. "
    )


def _pic_grid_message(owner: Any) -> str:
    return (
        f" PIC mesh={owner.pic_grid.nr}x{owner.pic_grid.nz} "
        f"(dr={owner.pic_grid.dr * 1.0e3:.4g} mm, "
        f"dz={owner.pic_grid.dz * 1.0e3:.4g} mm, "
        f"selection={owner.pic_grid_resolution_source}). "
    )


def install_backend_message(
    owner: Any,
    artifacts: RuntimeArtifacts,
    static_field_message: str,
    services: BootstrapServices,
) -> None:
    config = owner.config
    rf_message = services.rf_convention_message(
        enabled=owner.rf_enabled,
        peak_v=owner.rf_peak_voltage_v,
        reference_peak_v=owner.rf_reference_peak_voltage_v,
        frequency_hz=config.rf_frequency_hz,
        phase_rad=config.rf_phase_rad,
        peak_to_reference_scale=owner.rf_peak_to_reference_scale,
    )
    cartesian_message = services.cartesian_field_message(
        artifacts.static_field_3d_path,
        artifacts.cartesian_field3d,
    )
    owner.backend_message = (
        "Using flowchart-aligned axisymmetric PIC workflow on Taichi "
        f"({owner.arch_label}); collision handling now runs before RK4 "
        f"pushing. {static_field_message} "
        f"{_stage_message(artifacts)}"
        f"{cartesian_message} "
        f"{rf_message}"
        f"{_pic_grid_message(owner)}"
        f" PIC space-charge scale={owner.pic_space_charge_scale:.6g}. "
        f"{_source_birth_velocity_message(owner)}"
    )
