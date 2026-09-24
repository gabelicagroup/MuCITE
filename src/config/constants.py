"""Physical constants, repository paths, and stable runtime codes."""

from __future__ import annotations

from pathlib import Path

from scipy.constants import Avogadro, physical_constants

ATOMIC_MASS_CONSTANT = physical_constants["atomic mass constant"][0]
N2_MOLECULAR_MASS_KG = 28.0134e-3 / Avogadro
N2_MOLAR_MASS_KG_PER_MOL = 28.0134e-3
UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324

# constants.py lives at src/config/constants.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_FIELD_PATH = (
    Path("outputs")
    / "slens100x_bake_0p01_zmax65"
    / "baked_fields.npy"
)

PARTICLE_ACTIVE = 0
PARTICLE_Z_EXIT = 1
PARTICLE_RADIAL_OUT = 2
PARTICLE_ELECTRODE_HIT = 3
PARTICLE_DOMAIN_OUT = 4
PARTICLE_FRAGMENTED = 5
PARTICLE_CAPILLARY_BUFFER = 6
PARTICLE_STATUS_NAMES = {
    PARTICLE_ACTIVE: "active",
    PARTICLE_Z_EXIT: "z_exit",
    PARTICLE_RADIAL_OUT: "radial_out",
    PARTICLE_ELECTRODE_HIT: "electrode_hit",
    PARTICLE_DOMAIN_OUT: "domain_out",
    PARTICLE_FRAGMENTED: "fragmented",
    PARTICLE_CAPILLARY_BUFFER: "capillary_buffer",
}

SLOT_FREE = 0
SLOT_CAPILLARY = 1
SLOT_ACTIVE = 2

DT_LIMITER_NAMES = {
    0: "max_dt",
    1: "macro_end",
    2: "collision",
    3: "rf",
    4: "acceleration",
    5: "min_dt_floor",
    6: "langevin_max",
    7: "advection",
}
