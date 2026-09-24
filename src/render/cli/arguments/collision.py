"""Collision, IonSPA, and fragmentation CLI arguments."""

from __future__ import annotations

from argparse import ArgumentParser

from ....config import SimulationConfig

_defaults = SimulationConfig()

ARGUMENTS = (
    (("--collision-batch-size",), dict(type=int, default=_defaults.collision_batch_size, help="Collision processing chunk size.")),
    (("--legacy-collision-gather",), dict(action="store_true", help="Use the legacy full CPU collision gather.")),
    (("--collision-physics-backend",), dict(choices=["iict-lite", "ionspa"], default=_defaults.collision_physics_backend, help="Collision thermodynamics and single-event physics backend.")),
    (("--ionspa-backend",), dict(choices=["bundled", "approximate", "local"], default=_defaults.ionspa_backend, help="Deprecated compatibility selector for the optional IonSPA provider.")),
    (("--iict-parameter-config",), dict(default="", help="Strict iict-lite schema-v1 parameter JSON.")),
    (("--iict-heat-capacity-model",), dict(choices=["classical", "constant_cv", "tabulated"], default=None, help="Override the iict-lite heat-capacity model.")),
    (("--iict-num-atoms",), dict(type=int, default=None, help="Atom count for classical iict-lite heat capacity.")),
    (("--iict-constant-cv-j-per-k-per-ion",), dict(type=float, default=None, help="Constant heat capacity [J/K/ion].")),
    (("--iict-heat-capacity-csv",), dict(default="", help="Tabulated heat-capacity CSV.")),
    (("--iict-pseudoatom-model",), dict(choices=["constant", "tabulated"], default=None, help="Override the iict-lite pseudo-atom model.")),
    (("--iict-pseudoatom-mass-da",), dict(type=float, default=None, help="Constant pseudo-atom mass [Da].")),
    (("--iict-pseudoatom-csv",), dict(default="", help="Tabulated pseudo-atom mass CSV.")),
    (("--iict-pseudoatom-min-mass-da",), dict(type=float, default=None, help="Lower pseudo-atom mass validation bound [Da].")),
    (("--iict-pseudoatom-max-mass-da",), dict(type=float, default=None, help="Upper pseudo-atom mass validation bound [Da].")),
    (("--iict-fragmentation-model",), dict(choices=["none", "eyring"], default=None, help="Override iict-lite fragmentation model.")),
    (("--iict-delta-h-kj-per-mol",), dict(type=float, default=None, help="Eyring activation enthalpy [kJ/mol].")),
    (("--iict-delta-s-j-per-mol-k",), dict(type=float, default=None, help="Eyring activation entropy [J/mol/K].")),
    (("--collision-model",), dict(choices=["explicit", "hybrid-langevin"], default=_defaults.collision_model, help="Collision update model.")),
    (("--fragmentation-mode",), dict(choices=["transport", "loss", "off"], default=_defaults.fragmentation_mode, help="Fragmentation handling.")),
    (("--langevin-z-start-mm",), dict(type=float, default=_defaults.langevin_z_start_m * 1.0e3, help="Langevin window start [mm].")),
    (("--langevin-z-end-mm",), dict(type=float, default=_defaults.langevin_z_end_m * 1.0e3, help="Langevin window end [mm].")),
    (("--langevin-switch-prob",), dict(type=float, default=_defaults.langevin_switch_probability, help="Langevin switch threshold.")),
    (("--langevin-max-dt-s",), dict(type=float, default=_defaults.langevin_max_dt_s, help="Maximum Langevin step [s].")),
)


def add_collision_arguments(parser: ArgumentParser) -> None:
    for flags, kwargs in ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
