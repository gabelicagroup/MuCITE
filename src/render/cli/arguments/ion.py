"""Ion-template CLI arguments."""

from __future__ import annotations

from argparse import ArgumentParser

from ....config import ATOMIC_MASS_CONSTANT, IonTemplate

_ion = IonTemplate()

ARGUMENTS = (
    (("--ion-name",), dict(default=_ion.name, help="Ion species label.")),
    (("--ion-mass-amu",), dict(type=float, default=_ion.mass_kg / ATOMIC_MASS_CONSTANT, help="Ion mass [amu].")),
    (("--charge-state",), dict(type=int, default=_ion.charge_state, help="Positive ion charge state.")),
    (("--collision-cross-section-m2",), dict(type=float, default=_ion.collision_cross_section_m2, help="Ion-neutral collision cross section [m^2].")),
    (("--num-atoms",), dict(type=int, default=_ion.num_atoms, help="Atom count for IonSPA heat capacity.")),
    (("--heat-capacity-profile",), dict(default=_ion.heat_capacity_profile, help="IonSPA heat-capacity profile.")),
    (("--delta-h-kj-per-mol",), dict(type=float, default=_ion.delta_h_kj_per_mol, help="Fragmentation enthalpy [kJ/mol].")),
    (("--delta-s-j-per-mol-k",), dict(type=float, default=_ion.delta_s_j_per_mol_k, help="Fragmentation entropy [J/mol/K].")),
    (("--initial-internal-temperature-k",), dict(type=float, default=_ion.initial_internal_temperature_k, help="Initial internal temperature [K].")),
    (("--initial-kinetic-energy-ev",), dict(type=float, default=None, help="Initial translational kinetic energy [eV].")),
    (("--initial-direction-axis",), dict(choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="+z", help="Initial mean velocity axis.")),
    (("--initial-x-mm",), dict(type=float, default=0.0, help="Initial mean x [mm].")),
    (("--initial-y-mm",), dict(type=float, default=0.0, help="Initial mean y [mm].")),
    (("--initial-z-mm",), dict(type=float, default=0.0, help="Initial mean z [mm].")),
)


def add_ion_arguments(parser: ArgumentParser) -> None:
    for flags, kwargs in ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
