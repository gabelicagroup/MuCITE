"""Strict, presentation-layer validation for editable IICT overrides."""

from __future__ import annotations

import math

from .models import RuntimeConfig


_MODEL_CHOICES = {
    "iict_heat_capacity_model": {"classical", "constant_cv", "tabulated"},
    "iict_pseudoatom_model": {"constant", "tabulated"},
    "iict_fragmentation_model": {"none", "eyring"},
}
_POSITIVE_FIELDS = (
    "iict_constant_cv_j_per_k_per_ion",
    "iict_pseudoatom_mass_da",
    "iict_pseudoatom_min_mass_da",
    "iict_pseudoatom_max_mass_da",
    "iict_delta_h_kj_per_mol",
)


def _positive_number(value: object, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"runtime.{field_name} must be finite and positive.")
    return numeric


def _validate_model_choices(config: RuntimeConfig) -> None:
    for field_name, allowed in _MODEL_CHOICES.items():
        value = getattr(config, field_name)
        if value is not None and value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(
                f"runtime.{field_name} must be one of {{{choices}}}; "
                f"got {value!r}."
            )


def _validate_numeric_overrides(config: RuntimeConfig) -> None:
    if config.iict_num_atoms is not None and config.iict_num_atoms <= 2:
        raise ValueError("runtime.iict_num_atoms must be greater than 2.")
    for field_name in _POSITIVE_FIELDS:
        value = getattr(config, field_name)
        if value is not None:
            _positive_number(value, field_name)
    entropy = config.iict_delta_s_j_per_mol_k
    if entropy is not None and not math.isfinite(float(entropy)):
        raise ValueError(
            "runtime.iict_delta_s_j_per_mol_k must be finite."
        )
    minimum = config.iict_pseudoatom_min_mass_da
    maximum = config.iict_pseudoatom_max_mass_da
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError(
            "runtime iict pseudo-atom minimum mass must be below maximum."
        )
    mass = config.iict_pseudoatom_mass_da
    if mass is not None and minimum is not None and mass < minimum:
        raise ValueError("runtime iict pseudo-atom mass is below its minimum.")
    if mass is not None and maximum is not None and mass > maximum:
        raise ValueError("runtime iict pseudo-atom mass exceeds its maximum.")


def _validate_direct_parameter_set(config: RuntimeConfig) -> None:
    if config.iict_parameter_config_path.strip():
        return
    missing = [
        name
        for name in _MODEL_CHOICES
        if getattr(config, name) is None
    ]
    if missing:
        raise ValueError(
            "iict-lite requires a parameter JSON or explicit model selections: "
            + ", ".join(missing)
        )
    if (
        config.iict_heat_capacity_model == "constant_cv"
        and config.iict_constant_cv_j_per_k_per_ion is None
    ):
        raise ValueError("iict-lite constant_cv requires a per-ion Cv value.")
    if (
        config.iict_heat_capacity_model == "tabulated"
        and not config.iict_heat_capacity_csv_path.strip()
    ):
        raise ValueError("iict-lite tabulated heat capacity requires a CSV path.")
    if (
        config.iict_pseudoatom_model == "constant"
        and config.iict_pseudoatom_mass_da is None
    ):
        raise ValueError("iict-lite constant pseudo-atom model requires mass [Da].")
    if (
        config.iict_pseudoatom_model == "tabulated"
        and not config.iict_pseudoatom_csv_path.strip()
    ):
        raise ValueError("iict-lite tabulated pseudo-atom model requires a CSV path.")
    if config.iict_fragmentation_model == "eyring" and (
        config.iict_delta_h_kj_per_mol is None
        or config.iict_delta_s_j_per_mol_k is None
    ):
        raise ValueError("iict-lite Eyring fragmentation requires delta H and delta S.")


def validate_iict_runtime_config(config: RuntimeConfig) -> None:
    """Validate present overrides and direct, JSON-free IICT requests."""

    _validate_model_choices(config)
    _validate_numeric_overrides(config)
    for field_name in (
        "iict_parameter_config_path",
        "iict_heat_capacity_csv_path",
        "iict_pseudoatom_csv_path",
    ):
        if "\x00" in getattr(config, field_name):
            raise ValueError(f"runtime.{field_name} must not contain NUL.")
    if config.collision_physics_backend == "iict-lite":
        _validate_direct_parameter_set(config)


__all__ = ["validate_iict_runtime_config"]
