"""Focused validation for optional iict-lite configuration fields."""

from __future__ import annotations

import numpy as np

from .models import SimulationConfig


_OPTIONAL_MODELS = {
    "iict_heat_capacity_model": {"classical", "constant_cv", "tabulated"},
    "iict_pseudoatom_model": {"constant", "tabulated"},
    "iict_fragmentation_model": {"none", "eyring"},
}
_OVERRIDE_FIELDS = {
    "iict_heat_capacity_model",
    "iict_num_atoms",
    "iict_constant_cv_j_per_k_per_ion",
    "iict_heat_capacity_csv_path",
    "iict_pseudoatom_model",
    "iict_pseudoatom_mass_da",
    "iict_pseudoatom_csv_path",
    "iict_pseudoatom_min_mass_da",
    "iict_pseudoatom_max_mass_da",
    "iict_fragmentation_model",
    "iict_delta_h_kj_per_mol",
    "iict_delta_s_j_per_mol_k",
}


def _validate_model_choices(config: SimulationConfig) -> None:
    for field_name, allowed in _OPTIONAL_MODELS.items():
        value = getattr(config, field_name)
        if value is not None and value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"{field_name} must be one of {{{choices}}}.")


def _validate_optional_number(
    value: float | None,
    field_name: str,
    *,
    positive: bool,
) -> None:
    if value is None:
        return
    numeric = float(value)
    if not np.isfinite(numeric) or (positive and numeric <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{field_name} must be {qualifier}.")


def _validate_numeric_parameters(config: SimulationConfig) -> None:
    if config.iict_num_atoms is not None and int(config.iict_num_atoms) <= 2:
        raise ValueError("iict_num_atoms must be greater than 2.")
    for field_name in (
        "iict_constant_cv_j_per_k_per_ion",
        "iict_pseudoatom_mass_da",
        "iict_pseudoatom_min_mass_da",
        "iict_pseudoatom_max_mass_da",
        "iict_delta_h_kj_per_mol",
    ):
        _validate_optional_number(
            getattr(config, field_name),
            field_name,
            positive=True,
        )
    _validate_optional_number(
        config.iict_delta_s_j_per_mol_k,
        "iict_delta_s_j_per_mol_k",
        positive=False,
    )


def validate_iict_request(config: SimulationConfig) -> None:
    """Validate model selectors, optional overrides, and safety bounds."""

    _validate_model_choices(config)
    _validate_numeric_parameters(config)
    minimum = config.iict_pseudoatom_min_mass_da
    maximum = config.iict_pseudoatom_max_mass_da
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError(
            "iict_pseudoatom_min_mass_da must be below "
            "iict_pseudoatom_max_mass_da."
        )
    unknown = sorted(set(config.iict_cli_override_fields) - _OVERRIDE_FIELDS)
    if unknown:
        raise ValueError(
            "Unknown iict_cli_override_fields value(s): " + ", ".join(unknown)
        )


__all__ = ["validate_iict_request"]
