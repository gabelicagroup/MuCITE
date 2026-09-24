"""Preserve omitted historical JSON fields despite new-project demo defaults."""

from copy import deepcopy
from typing import Any


def preserve_import_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply only the old defaults changed for the public new-project demo."""
    result = deepcopy(payload)
    legacy = {
        "beam": {
            "ion_name": "protein_ion", "particle_count": 50000,
            "source_mode": "continuous-current", "num_atoms": 1200,
        },
        "runtime": {
            "total_time_s": 3.0e-2, "collision_physics_backend": "ionspa",
            "ionspa_backend": "bundled", "iict_parameter_config_path": "",
            "collision_mode": "hybrid-langevin", "fragmentation_mode": "transport",
            "electrode_mask_path": "default", "dummy_static_field": False,
            "electrode_mask_cache_path": (
                "outputs/fluent_current_ccs_charge_sweep_20260430_144536/"
                "I10nA_CCS2e18_z5/electrode_mask_cache.npz"
            ),
        },
    }
    for group, defaults in legacy.items():
        values = result.setdefault(group, {})
        for key, value in defaults.items():
            values.setdefault(key, value)
    result.setdefault("loaded_baked_field_path", (
        "outputs/slens100x_bake_0p01_zmax65_fluent_interior_clipped_capexit4p5/"
        "baked_fields.npy"
    ))
    return result
