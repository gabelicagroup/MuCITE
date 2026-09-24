"""Detector and early-termination CLI arguments."""

from __future__ import annotations

from argparse import ArgumentParser

from ....config import SimulationConfig

_defaults = SimulationConfig()

ARGUMENTS = (
    (("--detector-z-mm",), dict(type=float, default=None, help="Detector axial plane [mm].")),
    (("--detector-radius-mm",), dict(type=float, default=None, help="Detector accepted radius [mm].")),
    (("--radial-limit-mm",), dict(type=float, default=None, help="Radial loss boundary [mm].")),
    (("--stop-active-fraction-below",), dict(type=float, default=_defaults.stop_active_fraction_below, help="Active-fraction stop threshold.")),
    (("--stop-stable-window",), dict(type=int, default=_defaults.stop_stable_window_steps, help="Stable status-count window.")),
    (("--stop-stable-fraction-tol",), dict(type=float, default=_defaults.stop_stable_fraction_tol, help="Stable-window fractional tolerance.")),
    (("--max-wall-time-s",), dict(type=float, default=_defaults.max_wall_time_s, help="Wall-clock cap [s].")),
    (("--terminal-event-mode",), dict(choices=["transport-only", "all", "none"], default=_defaults.terminal_event_mode, help="Terminal event row policy.")),
    (("--max-terminal-event-rows",), dict(type=int, default=_defaults.max_terminal_event_rows, help="Maximum terminal event rows; 0 is unlimited.")),
)


def add_boundary_arguments(parser: ArgumentParser) -> None:
    for flags, kwargs in ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
