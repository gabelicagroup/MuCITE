"""Static/RF field and electrode-mask CLI arguments."""

from __future__ import annotations

import argparse

import numpy as np

from ....config import DEFAULT_STATIC_FIELD_PATH, SimulationConfig

_defaults = SimulationConfig()

ELECTRODE_ARGUMENTS = (
    (("--electrode-mask",), dict(default=None, help="Electrode mask path or 'default'.")),
    (("--no-electrode-mask",), dict(action="store_true", help="Disable 2D electrode-mask collisions.")),
    (("--electrode-mask-cache",), dict(default="", help="Optional electrode-mask cache.")),
    (("--electrode-mask-z-offset-mm",), dict(type=float, default=None, help="Optional electrode-mask z offset [mm].")),
    (("--electrode-mask-3d",), dict(default="", help="Optional Cartesian 3D electrode mask.")),
    (("--electrode-hit-distance-mm",), dict(type=float, default=_defaults.electrode_hit_distance_m * 1.0e3, help="Electrode hit threshold [mm].")),
)

FIELD_ARGUMENTS = (
    (("--rf-frequency",), dict(type=float, default=_defaults.rf_frequency_hz, help="RF frequency [Hz].")),
    (("--rf-peak-voltage",), dict(type=float, default=_defaults.rf_peak_voltage_v, help="Single-phase RF Vpeak [V].")),
    (("--rf-phase-deg",), dict(type=float, default=np.degrees(_defaults.rf_phase_rad), help="RF phase [deg].")),
    (("--rf-reference-peak-voltage",), dict(type=float, default=None, help="Baked RF reference Vpeak [V].")),
    (("--rf-vpp",), dict(dest="deprecated_rf_vpp", type=float, default=None, help=argparse.SUPPRESS)),
    (("--static-field",), dict(default=str(DEFAULT_STATIC_FIELD_PATH), help="Baked static field path.")),
    (("--stage-schedule", "--trap-stage-config"), dict(dest="stage_schedule", default="", help="Optional field stage schedule JSON.")),
    (("--static-field-3d",), dict(default="", help="Optional Cartesian 3D field.")),
    (("--dummy-static-field",), dict(action="store_true", help="Use analytic placeholder fields.")),
)


def add_field_arguments(parser: argparse.ArgumentParser) -> None:
    for flags, kwargs in ELECTRODE_ARGUMENTS + FIELD_ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
