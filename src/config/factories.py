"""Canonical factories for normalized and validated request models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from .models import ExecutionConfig, IonTemplate, OutputConfig, SimulationConfig
from .validation import (
    normalize_execution_config,
    normalize_ion_template,
    normalize_output_config,
    normalize_simulation_config,
    validate_execution_config,
    validate_ion_template,
    validate_output_config,
    validate_simulation_config,
)


def make_execution_config(
    base: Optional[ExecutionConfig] = None,
    **overrides: Any,
) -> ExecutionConfig:
    """Build the canonical non-physical execution policy."""

    if base is None:
        base = ExecutionConfig()
    if not isinstance(base, ExecutionConfig):
        raise TypeError("base must be an ExecutionConfig instance or None.")
    config = normalize_execution_config(replace(base, **overrides))
    validate_execution_config(config)
    return config


def make_output_config(
    base: Optional[OutputConfig] = None,
    **overrides: Any,
) -> OutputConfig:
    """Build the canonical output policy."""

    if base is None:
        base = OutputConfig()
    if not isinstance(base, OutputConfig):
        raise TypeError("base must be an OutputConfig instance or None.")
    config = normalize_output_config(replace(base, **overrides))
    validate_output_config(config)
    return config


def make_simulation_config(
    base: Optional[SimulationConfig] = None,
    **overrides: Any,
) -> SimulationConfig:
    """Build, normalize, and validate the canonical runtime request."""

    if base is None:
        base = SimulationConfig()
    if not isinstance(base, SimulationConfig):
        raise TypeError("base must be a SimulationConfig instance or None.")
    config = normalize_simulation_config(replace(base, **overrides))
    validate_simulation_config(config)
    return config


def make_ion_template(
    base: Optional[IonTemplate] = None,
    **overrides: Any,
) -> IonTemplate:
    """Build, normalize, and validate the canonical ion request."""

    if base is None:
        base = IonTemplate()
    if not isinstance(base, IonTemplate):
        raise TypeError("base must be an IonTemplate instance or None.")
    template = normalize_ion_template(replace(base, **overrides))
    validate_ion_template(template)
    return template
