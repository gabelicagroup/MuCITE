"""Validate and install PIC/RF scalar runtime state."""

from __future__ import annotations

from typing import Any

import numpy as np

from .artifacts import RuntimeArtifacts
from .services import BootstrapServices


def _validate_field_scales(config: Any) -> None:
    pic_scale = float(config.pic_space_charge_scale)
    if not np.isfinite(pic_scale) or pic_scale < 0.0:
        raise ValueError(
            "pic_space_charge_scale must be finite and non-negative."
        )
    if config.rf_peak_voltage_v < 0.0:
        raise ValueError("rf_peak_voltage_v must be non-negative.")
    if (
        config.rf_reference_peak_voltage_v is not None
        and config.rf_reference_peak_voltage_v <= 0.0
    ):
        raise ValueError(
            "rf_reference_peak_voltage_v must be positive when provided."
        )


def _validate_mixed_rf_references(
    config: Any,
    artifacts: RuntimeArtifacts,
    services: BootstrapServices,
    baked_reference_peak_v: float,
    cartesian_reference_peak_v: Any,
) -> None:
    if (
        config.rf_reference_peak_voltage_v is None
        and cartesian_reference_peak_v is not None
        and services.has_baked_rf(artifacts.baked_fields)
        and not np.isclose(
            cartesian_reference_peak_v,
            baked_reference_peak_v,
            rtol=1.0e-9,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            "2D baked RF and Cartesian 3D RF use different reference peak "
            f"voltages ({baked_reference_peak_v} V vs "
            f"{cartesian_reference_peak_v} V). Pass "
            "--rf-reference-peak-voltage explicitly or rebake one field so "
            "both share the same Vref."
        )


def install_rf_state(
    owner: Any,
    artifacts: RuntimeArtifacts,
    services: BootstrapServices,
) -> None:
    config = artifacts.config
    _validate_field_scales(config)
    owner.pic_space_charge_scale = float(config.pic_space_charge_scale)
    baked_reference = services.infer_rf_reference(artifacts.baked_fields)
    cartesian_reference = services.infer_cartesian_rf_reference(
        artifacts.cartesian_field3d
    )
    _validate_mixed_rf_references(
        config,
        artifacts,
        services,
        baked_reference,
        cartesian_reference,
    )
    inferred_reference = cartesian_reference or baked_reference
    owner.rf_reference_peak_voltage_v = float(
        config.rf_reference_peak_voltage_v or inferred_reference
    )
    owner.rf_peak_voltage_v = float(config.rf_peak_voltage_v)
    owner.rf_enabled = (
        (
            services.has_baked_rf(artifacts.baked_fields)
            or services.has_cartesian_rf(artifacts.cartesian_field3d)
        )
        and owner.rf_peak_voltage_v > 0.0
        and config.rf_frequency_hz > 0.0
        and owner.rf_reference_peak_voltage_v > 0.0
    )
    owner.rf_peak_to_reference_scale = (
        owner.rf_peak_voltage_v / owner.rf_reference_peak_voltage_v
        if owner.rf_enabled
        else 0.0
    )
    owner.rf_angular_frequency_rad_s = (
        2.0 * np.pi * float(config.rf_frequency_hz)
        if owner.rf_enabled
        else 0.0
    )
