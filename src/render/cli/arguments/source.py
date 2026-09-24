"""Ion-source and capillary CLI arguments."""

from __future__ import annotations

from argparse import ArgumentParser

from ....config import SimulationConfig

_defaults = SimulationConfig()

SOURCE_ARGUMENTS = (
    (("--source-radius-mm",), dict(type=float, default=None, help="Source transverse cutoff radius [mm].")),
    (("--source-profile",), dict(choices=["uniform-disk", "gaussian"], default=_defaults.source_profile, help="Transverse source profile.")),
    (("--source-gaussian-sigma-mm",), dict(type=float, default=None, help="Gaussian source sigma [mm].")),
    (("--source-mode",), dict(choices=["packet", "continuous-current"], default=_defaults.source_mode, help="Ion source mode.")),
    (("--ion-current-a",), dict(type=float, default=_defaults.ion_current_a, help="Continuous source current [A].")),
    (("--source-mach-number",), dict(type=float, default=_defaults.source_mach_number, help="Source Mach number.")),
    (("--source-gas-gamma",), dict(type=float, default=_defaults.source_gas_gamma, help="Source gas heat-capacity ratio.")),
    (("--source-gas-molar-mass-kg-per-mol",), dict(type=float, default=_defaults.source_gas_molar_mass_kg_per_mol, help="Source gas molar mass [kg/mol].")),
    (("--source-temperature-k",), dict(type=float, default=None, help="Optional source temperature [K].")),
    (("--source-axial-velocity-m-per-s",), dict(type=float, default=None, help="Optional source axial velocity [m/s].")),
    (("--source-velocity-from-gas-field",), dict(action="store_true", default=_defaults.source_velocity_from_gas_field, help="Initialize source velocity from gas data.")),
    (("--source-birth-velocity-gas-csv",), dict(default="", help="Raw gas CSV for birth velocity.")),
    (("--source-birth-velocity-z-min-mm",), dict(type=float, default=_defaults.source_birth_velocity_z_min_m * 1.0e3, help="Birth gas minimum z [mm].")),
    (("--source-birth-velocity-z-max-mm",), dict(type=float, default=None, help="Birth gas maximum z [mm].")),
    (("--source-birth-velocity-radius-mm",), dict(type=float, default=None, help="Birth gas radial cutoff [mm].")),
    (("--source-radial-velocity-scale",), dict(type=float, default=_defaults.source_radial_velocity_scale, help="Source radial gas-velocity scale.")),
    (("--source-velocity-delta-m-per-s",), dict(type=float, default=_defaults.source_velocity_delta_m_per_s, help="Source axial velocity increment [m/s].")),
)

CAPILLARY_ARGUMENTS = (
    (("--capillary-exit-z-mm",), dict(type=float, default=None, help="Capillary exit z [mm].")),
    (("--capillary-voltage-v",), dict(type=float, default=_defaults.capillary_voltage_v, help="Capillary voltage [V].")),
    (("--capillary-prefill-length-mm",), dict(type=float, default=_defaults.capillary_prefill_length_m * 1.0e3, help="Capillary prefill length [mm].")),
    (("--capillary-prefill-macro-particles",), dict(type=int, default=_defaults.capillary_prefill_macro_particles, help="Capillary prefill weighted ion pack count.")),
    (("--macro-particles-per-injection",), dict(type=int, default=_defaults.macro_particles_per_injection, help="Target weighted ion packs per source injection.")),
    (("--max-macro-particle-weight",), dict(type=float, default=_defaults.max_macro_particle_weight, help="Maximum real ions per emitted weighted ion pack.")),
    (("--cone-half-angle-deg",), dict(type=float, default=0.0, help="Initial cone half-angle [deg].")),
)


def add_source_arguments(parser: ArgumentParser) -> None:
    for flags, kwargs in SOURCE_ARGUMENTS + CAPILLARY_ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
