"""Process and macro-runtime CLI arguments."""

from __future__ import annotations

from argparse import ArgumentParser

from ....config import ExecutionConfig, SimulationConfig

_execution = ExecutionConfig()
_simulation = SimulationConfig()

ARGUMENTS = (
    (("--config",), dict(default="", help="Load a strict schema-v1 configuration JSON document.")),
    (("--write-config",), dict(default="", help="Write the final canonical configuration and exit.")),
    (("--backend",), dict(choices=["cpu", "taichi"], default=_execution.backend, help="Particle execution backend.")),
    (("--gui",), dict(action="store_true", default=_execution.gui, help="Open the MuCITE graphical workbench.")),
    (("--live-window",), dict(action="store_true", default=_execution.live_window, help="Open the runtime monitor.")),
    (("--no-window",), dict(action="store_true", default=_execution.no_window, help="Force command-line mode.")),
    (("--progress",), dict(action="store_true", default=_execution.progress, help="Print macro-step progress.")),
    (("--ion-count",), dict(type=int, default=_simulation.ion_count, help="Number of simulation particle slots.")),
    (("--total-time",), dict(type=float, default=_simulation.total_time_s, help="Total simulation time [s].")),
    (("--macro-time-step",), dict(type=float, default=_simulation.macro_time_step_s, help="Macro-step duration [s].")),
    (("--random-seed",), dict(type=int, default=_simulation.random_seed, help="Random seed.")),
    (("--macro-particle-weight",), dict(type=float, default=_simulation.macro_particle_weight, help="Physical ions represented by each weighted ion pack.")),
    (("--initial-position-jitter-mm",), dict(type=float, default=_simulation.initial_position_jitter_m * 1.0e3, help="Initial position jitter [mm].")),
    (("--initial-velocity-jitter-m-per-s",), dict(type=float, default=_simulation.initial_velocity_jitter_m_per_s, help="Initial velocity jitter [m/s].")),
)


def add_runtime_arguments(parser: ArgumentParser) -> None:
    for flags, kwargs in ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
