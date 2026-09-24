"""Compatibility facade for split GUI background-task adapters."""

from __future__ import annotations

# Preserve the historical import surface of this module. Canonical
# implementations live in the focused modules imported at the end.
import queue
import threading
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.constants import elementary_charge

from ...config import (
    ATOMIC_MASS_CONSTANT,
    PROJECT_ROOT,
    IonTemplate,
    SimulationConfig,
    make_ion_template,
    make_simulation_config,
)
from ...data.logger import DataLogger
from ...data.report import (
    ReportSnapshot,
    _write_simulation_report,
    capture_report_snapshot,
)
from ...env.sources import sample_source_xy_offsets
from ..cli.runner import build_demo_simulation
from .beam_tasks import run_beam_smoke_task, sample_beam_phase_space
from .cli_preview import gui_run_arguments
from .config_adapter import (
    build_ion_template,
    build_simulation_config,
    direction_vector,
)
from .field_tasks import run_field_bake_task, run_field_diagnostics_task
from .models import AppConfig, BeamConfig
from .simulation_tasks import (
    _emit_latest_snapshot,
    run_report_task,
    run_simulation_task,
)
from .worker_lifecycle import (
    TaskStopped,
    WorkerHandle,
    _post,
    start_worker,
)
