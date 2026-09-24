"""Public report-schema API assembled from focused helper modules."""

# Keep one stable package-level schema API while implementations remain split
# into focused modules below this package.
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from scipy.constants import elementary_charge

from ...config import (
    ATOMIC_MASS_CONSTANT,
    PARTICLE_ACTIVE,
    PARTICLE_CAPILLARY_BUFFER,
    PARTICLE_STATUS_NAMES,
    SLOT_CAPILLARY,
    IonTemplate,
    SimulationConfig,
    SimulationResult,
)
from ...agents.ion import IonState
from .dto import _config_summary, _template_summary
from .final_particles import (
    _final_particle_table,
    _final_particle_transport_summary,
)
from .runtime_summary import (
    _performance_summary,
    _runtime_field_summary,
    _source_runtime_summary,
    _survivor_summary,
)
from .terminal_summary import _terminal_event_summary
from .values import (
    _array_stats,
    _json_safe,
    _weighted_mean,
    _weighted_quantile,
)
