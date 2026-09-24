"""Independent Improved Impulsive Collision Theory backend."""

from .backend import IictLiteBackend
from .fragmentation import (
    EyringFragmentationModel,
    FragmentationModel,
    NoFragmentationModel,
)
from .heat_capacity import (
    ClassicalHeatCapacityModel,
    ConstantCvHeatCapacityModel,
    HeatCapacityModel,
    TabulatedHeatCapacityModel,
)
from .parameters import (
    IICT_PARAMETER_SCHEMA_VERSION,
    ResolvedIictParameters,
    resolve_iict_parameters,
)
from .physics import (
    IictCollisionCoreResult,
    apply_iict_collision_core,
    collision_frequency_s_inverse,
    expected_internal_energy_change_j,
    expected_kinetic_energy_change_j,
    iict_transfer_efficiency,
    mean_collision_waiting_time_s,
    mean_free_path_m,
    mean_relative_speed_m_per_s,
    steady_drift_temperature_k,
)
from .pseudoatom import (
    ConstantPseudoAtomMassModel,
    PseudoAtomMassModel,
    TabulatedPseudoAtomMassModel,
)

__all__ = [
    "ClassicalHeatCapacityModel",
    "ConstantCvHeatCapacityModel",
    "ConstantPseudoAtomMassModel",
    "EyringFragmentationModel",
    "FragmentationModel",
    "HeatCapacityModel",
    "IICT_PARAMETER_SCHEMA_VERSION",
    "IictCollisionCoreResult",
    "IictLiteBackend",
    "NoFragmentationModel",
    "PseudoAtomMassModel",
    "ResolvedIictParameters",
    "TabulatedHeatCapacityModel",
    "TabulatedPseudoAtomMassModel",
    "apply_iict_collision_core",
    "collision_frequency_s_inverse",
    "expected_internal_energy_change_j",
    "expected_kinetic_energy_change_j",
    "iict_transfer_efficiency",
    "mean_collision_waiting_time_s",
    "mean_free_path_m",
    "mean_relative_speed_m_per_s",
    "resolve_iict_parameters",
    "steady_drift_temperature_k",
]
