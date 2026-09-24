"""Strict JSON save/load helpers for GUI configuration documents."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Type, TypeVar, get_args, get_type_hints

from .models import (
    AppConfig,
    BeamConfig,
    FieldBakeConfig,
    OutputConfig,
    RuntimeConfig,
    SessionState,
)
from .iict_config_validation import validate_iict_runtime_config

T = TypeVar("T")

SCHEMA_VERSION = 5
_PRIOR_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
_MIGRATABLE_LEGACY_VERSIONS = frozenset({1, 2, 3})
_DOCUMENT_FIELDS = frozenset({"schema_version", "app_config"})
_CONFIG_GROUPS = frozenset({"beam", "field_bake", "runtime", "output"})
_LEGACY_RUNTIME_TYPES: dict[str, Any] = {
    "enable_space_charge": bool,
    "pic_coupling_mode": str,
    "capillary_exit_z_mm": float,
    "capillary_prefill_length_mm": float,
    "sor_omega": float,
    "sor_max_iters": int,
    "sor_tolerance": float,
}

CURRENT_FIELD_CAPILLARY_GEOMETRY = {
    "capillary_exit_z_mm": 6.0,
    "fluent_capillary_total_length_mm": 101.0,
    "fluent_capillary_radius_mm": 0.25,
    "fluent_capillary_external_start_z_mm": 4.5,
}


def _reject_unknown_fields(
    payload: dict[str, Any],
    allowed: set[str] | frozenset[str],
    context: str,
) -> None:
    unknown = sorted((key for key in payload if key not in allowed), key=str)
    if unknown:
        names = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"Unknown {context} field(s): {names}.")


def _matches_json_type(value: Any, annotation: Any) -> bool:
    options = get_args(annotation)
    if value is None:
        return type(None) in options
    if options and type(None) in options:
        return any(
            _matches_json_type(value, option)
            for option in options
            if option is not type(None)
        )
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int
    if annotation is float:
        return type(value) in {int, float}
    if annotation is str:
        return type(value) is str
    return True


def _validate_field_types(
    payload: dict[str, Any],
    model_type: Type[Any],
    context: str,
    extra_types: dict[str, Any] | None = None,
) -> None:
    annotations = get_type_hints(model_type)
    annotations.update(extra_types or {})
    for name, value in payload.items():
        if not _matches_json_type(value, annotations[name]):
            raise ValueError(f"{context}.{name} has an invalid JSON type.")


def _checked_section(
    payload: Any,
    model_type: Type[T],
    context: str,
    extra_types: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object.")
    allowed = {item.name for item in fields(model_type)}
    allowed.update((extra_types or {}).keys())
    _reject_unknown_fields(payload, allowed, context)
    _validate_field_types(payload, model_type, context, extra_types)
    return dict(payload)


def _validate_envelope(payload: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("GUI configuration document must be a JSON object.")
    _reject_unknown_fields(payload, _DOCUMENT_FIELDS, "document")
    missing = sorted(_DOCUMENT_FIELDS - set(payload))
    if missing:
        raise ValueError(
            "Missing GUI configuration document field(s): " + ", ".join(missing)
        )
    version = payload["schema_version"]
    supported = _PRIOR_SCHEMA_VERSIONS | {SCHEMA_VERSION}
    if type(version) is not int or version not in supported:
        raise ValueError(f"Unsupported GUI schema_version: {version!r}.")
    raw_config = payload["app_config"]
    if not isinstance(raw_config, dict):
        raise ValueError("app_config must be a JSON object.")
    return version, raw_config


def _checked_config_payload(
    version: int,
    raw_config: dict[str, Any],
) -> dict[str, Any]:
    app_fields = {item.name for item in fields(AppConfig)} - {"state"}
    allowed_app_fields = set(app_fields)
    if version in _MIGRATABLE_LEGACY_VERSIONS:
        allowed_app_fields.add("state")
    _reject_unknown_fields(raw_config, allowed_app_fields, "app_config")

    scalar_values = {
        key: value
        for key, value in raw_config.items()
        if key not in _CONFIG_GROUPS and key != "state"
    }
    _validate_field_types(scalar_values, AppConfig, "app_config")
    checked = dict(scalar_values)
    extra_runtime = (
        _LEGACY_RUNTIME_TYPES
        if version in _MIGRATABLE_LEGACY_VERSIONS
        else None
    )
    for name, model_type in (
        ("beam", BeamConfig),
        ("field_bake", FieldBakeConfig),
        ("runtime", RuntimeConfig),
        ("output", OutputConfig),
    ):
        if name in raw_config:
            checked[name] = _checked_section(
                raw_config[name],
                model_type,
                f"app_config.{name}",
                extra_runtime if name == "runtime" else None,
            )
    if "state" in raw_config:
        _checked_section(raw_config["state"], SessionState, "app_config.state")
    return checked


def _migrate_legacy_payload(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Migrate documented v1-v3 fields without accepting arbitrary input."""

    migrated = deepcopy(raw_config)
    beam = migrated.setdefault("beam", {})
    runtime = migrated.setdefault("runtime", {})
    field_bake = migrated.setdefault("field_bake", {})
    if str(beam.get("gas_velocity_init_mode", "")).lower() == "off":
        beam["gas_velocity_init_mode"] = "static"
    for key in ("capillary_exit_z_mm", "capillary_prefill_length_mm"):
        legacy_value = runtime.pop(key, None)
        if key not in beam and legacy_value is not None:
            beam[key] = legacy_value

    enabled = runtime.pop("enable_space_charge", None)
    if "pic_space_charge_scale" not in runtime and enabled is not None:
        runtime["pic_space_charge_scale"] = 1.0 if enabled else 0.0
    coupling = runtime.pop("pic_coupling_mode", None)
    if coupling not in {None, "coupled", "decoupled"}:
        raise ValueError(
            "app_config.runtime.pic_coupling_mode must be 'coupled' or "
            "'decoupled'."
        )
    if coupling == "coupled":
        runtime["pic_grid_nr"] = 0
        runtime["pic_grid_nz"] = 0
    for key in ("sor_omega", "sor_max_iters", "sor_tolerance"):
        runtime.pop(key, None)
    if str(runtime.get("pic_poisson_backend", "")).lower() == "sor":
        runtime["pic_poisson_backend"] = "amg"

    if "gas_field_mode" not in field_bake:
        field_bake["gas_field_mode"] = (
            "import" if str(field_bake.get("fluent_path", "")).strip() else "static"
        )
    for key, value in CURRENT_FIELD_CAPILLARY_GEOMETRY.items():
        field_bake.setdefault(key, value)
    return migrated


def _require_number(value: Any, path: str) -> float:
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be finite.")
    return numeric


def _require_positive(value: Any, path: str) -> float:
    numeric = _require_number(value, path)
    if numeric <= 0.0:
        raise ValueError(f"{path} must be positive.")
    return numeric


def _require_nonnegative(value: Any, path: str) -> float:
    numeric = _require_number(value, path)
    if numeric < 0.0:
        raise ValueError(f"{path} must be non-negative.")
    return numeric


def _require_choice(value: str, allowed: set[str], path: str) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{path} must be one of {{{choices}}}; got {value!r}.")


def _validate_beam_ranges(beam: BeamConfig) -> None:
    _require_choice(beam.source_mode, {"packet", "continuous-current"}, "beam.source_mode")
    _require_choice(beam.source_profile, {"uniform-disk", "gaussian"}, "beam.source_profile")
    _require_choice(beam.gas_velocity_init_mode, {"static", "from-gas-field"}, "beam.gas_velocity_init_mode")
    _require_choice(beam.direction_axis, {"+x", "-x", "+y", "-y", "+z", "-z"}, "beam.direction_axis")
    if not beam.ion_name.strip() or not beam.heat_capacity_profile.strip():
        raise ValueError("beam ion_name and heat_capacity_profile must not be empty.")
    for name in (
        "mass_amu", "macro_particle_weight", "collision_cross_section_m2",
        "initial_internal_temperature_k", "delta_h_kj_per_mol",
    ):
        _require_positive(getattr(beam, name), f"beam.{name}")
    if beam.charge_e <= 0 or beam.particle_count <= 0 or beam.num_atoms <= 2:
        raise ValueError("beam charge/count must be positive and num_atoms must exceed 2.")
    for name in (
        "current_a", "beam_radius_mm", "initial_position_jitter_mm",
        "velocity_jitter_m_per_s", "source_gaussian_sigma_mm",
        "source_birth_velocity_z_min_mm", "source_radial_velocity_scale",
        "capillary_exit_z_mm", "capillary_prefill_length_mm",
    ):
        _require_nonnegative(getattr(beam, name), f"beam.{name}")
    for name in ("kinetic_energy_ev", "source_temperature_k", "source_birth_velocity_radius_mm"):
        value = getattr(beam, name)
        if value is not None:
            _require_positive(value, f"beam.{name}")
    if beam.source_axial_velocity_m_per_s is not None:
        _require_nonnegative(beam.source_axial_velocity_m_per_s, "beam.source_axial_velocity_m_per_s")
    if not 0.0 <= beam.cone_half_angle_deg < 90.0:
        raise ValueError("beam.cone_half_angle_deg must be in [0, 90).")
    if beam.source_mode == "continuous-current" and beam.current_a <= 0.0:
        raise ValueError("beam.current_a must be positive for continuous-current mode.")
    if beam.source_profile == "gaussian" and beam.source_gaussian_sigma_mm <= 0.0:
        raise ValueError("beam.source_gaussian_sigma_mm must be positive for gaussian mode.")
    if beam.source_birth_velocity_z_max_mm is not None:
        _require_nonnegative(beam.source_birth_velocity_z_max_mm, "beam.source_birth_velocity_z_max_mm")
        if beam.source_birth_velocity_z_max_mm < beam.source_birth_velocity_z_min_mm:
            raise ValueError("beam birth gas z max must be greater than or equal to z min.")


def _validate_field_ranges(config: FieldBakeConfig) -> None:
    _require_choice(config.gas_field_mode, {"static", "import"}, "field_bake.gas_field_mode")
    if not config.coordinate_mapping.strip() or not config.phi_key_for_plot.strip():
        raise ValueError("field_bake mapping and plot key must not be empty.")
    if config.gas_field_mode == "import" and not config.fluent_path.strip():
        raise ValueError("field_bake.fluent_path is required in import mode.")
    if config.z_max_mm <= config.z_min_mm or config.r_max_mm <= config.r_min_mm:
        raise ValueError("field_bake grid maxima must be greater than grid minima.")
    for name in ("dz_mm", "dr_mm", "pa_grids_per_mm", "background_temperature_k", "fluent_capillary_total_length_mm", "fluent_capillary_radius_mm"):
        _require_positive(getattr(config, name), f"field_bake.{name}")
    _require_nonnegative(config.background_pressure_pa, "field_bake.background_pressure_pa")
    if config.fluent_capillary_external_start_z_mm >= config.capillary_exit_z_mm:
        raise ValueError("field_bake external gas start z must be smaller than capillary exit z.")


def _validate_runtime_ranges(config: RuntimeConfig) -> None:
    choices = {
        "backend": {"cpu", "taichi"},
        "collision_physics_backend": {"iict-lite", "ionspa"},
        "ionspa_backend": {"bundled", "approximate", "local"},
        "collision_mode": {"explicit", "hybrid-langevin"},
        "fragmentation_mode": {"transport", "loss", "off"},
        "pic_poisson_backend": {"amg", "pyamg", "sparse_direct", "sparse_spsolve", "sparse_cg", "sparse_bicgstab"},
        "pic_poisson_preconditioner": {"none", "jacobi"},
        "pic_amg_mode": {"solve", "preconditioned_cg", "preconditioned_bicgstab"},
        "pic_amg_solver": {"ruge_stuben", "smoothed_aggregation"},
        "pic_amg_fallback_backend": {"sparse_direct", "sparse_spsolve", "sparse_cg", "sparse_bicgstab"},
        "terminal_event_mode": {"transport-only", "all", "none"},
    }
    for name, allowed in choices.items():
        _require_choice(getattr(config, name), allowed, f"runtime.{name}")
    for name in ("total_time_s", "macro_dt_s", "pic_amg_tolerance"):
        _require_positive(getattr(config, name), f"runtime.{name}")
    for name in (
        "snapshot_every", "pic_space_charge_scale", "rf_frequency_hz",
        "rf_peak_voltage_v", "detector_z_mm", "detector_radius_mm",
        "radial_limit_mm", "capillary_prefill_macro_particles",
        "max_macro_particle_weight", "stop_stable_window_steps",
        "stop_stable_fraction_tol", "max_wall_time_s", "max_terminal_event_rows",
        "collision_batch_size", "langevin_z_start_mm", "langevin_z_end_mm",
        "langevin_max_dt_s", "electrode_hit_distance_mm",
    ):
        _require_nonnegative(getattr(config, name), f"runtime.{name}")
    if config.pic_amg_max_iters <= 0 or config.macro_particles_per_injection <= 0:
        raise ValueError("runtime AMG iterations and particles per injection must be positive.")
    pair = (config.pic_grid_nr, config.pic_grid_nz)
    if pair != (0, 0) and (pair[0] < 3 or pair[1] < 3):
        raise ValueError("runtime PIC grids must both be 0 or both be at least 3.")
    if not 0.0 <= config.stop_active_fraction_below <= 1.0:
        raise ValueError("runtime.stop_active_fraction_below must be in [0, 1].")
    if not 0.0 <= config.langevin_switch_prob <= 1.0:
        raise ValueError("runtime.langevin_switch_prob must be in [0, 1].")
    if config.collision_mode == "hybrid-langevin":
        if config.langevin_z_end_mm < config.langevin_z_start_mm:
            raise ValueError("runtime Langevin z end must be greater than or equal to z start.")
        _require_positive(config.langevin_max_dt_s, "runtime.langevin_max_dt_s")
    validate_iict_runtime_config(config)


def _validate_output_ranges(config: OutputConfig) -> None:
    if config.trajectory_sample_count < 0:
        raise ValueError("output.trajectory_sample_count must be non-negative.")
    for name in ("trajectory_record_every", "snapshot_plot_max_points"):
        if getattr(config, name) <= 0:
            raise ValueError(f"output.{name} must be positive.")
    _require_positive(config.terminal_time_bin_ms, "output.terminal_time_bin_ms")
    if not config.report_format.strip():
        raise ValueError("output.report_format must not be empty.")


def _validate_app_config(config: AppConfig) -> None:
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig instance.")
    _validate_field_types(
        {
            "session_name": config.session_name,
            "output_dir": config.output_dir,
            "loaded_baked_field_path": config.loaded_baked_field_path,
        },
        AppConfig,
        "app_config",
    )
    if not config.session_name.strip() or not config.output_dir.strip():
        raise ValueError("session_name and output_dir must not be empty.")
    for name, model_type in (
        ("beam", BeamConfig), ("field_bake", FieldBakeConfig),
        ("runtime", RuntimeConfig), ("output", OutputConfig),
    ):
        value = getattr(config, name)
        if not isinstance(value, model_type):
            raise TypeError(f"config.{name} must be a {model_type.__name__} instance.")
        _validate_field_types(asdict(value), model_type, name)
        for item in fields(model_type):
            numeric = getattr(value, item.name)
            if type(numeric) in {int, float}:
                _require_number(numeric, f"{name}.{item.name}")
    _validate_beam_ranges(config.beam)
    _validate_field_ranges(config.field_bake)
    _validate_runtime_ranges(config.runtime)
    _validate_output_ranges(config.output)


def _build_config(raw_config: dict[str, Any]) -> AppConfig:
    scalar_values = {
        key: value for key, value in raw_config.items() if key not in _CONFIG_GROUPS
    }
    config = AppConfig(**scalar_values)
    for name, model_type in (
        ("beam", BeamConfig),
        ("field_bake", FieldBakeConfig),
        ("runtime", RuntimeConfig),
        ("output", OutputConfig),
    ):
        setattr(config, name, model_type(**raw_config.get(name, {})))
    config.state = SessionState(
        loaded_baked_field_path=config.loaded_baked_field_path,
    )
    _validate_app_config(config)
    return config


def app_config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Return one validated v5 document without transient session state."""

    _validate_app_config(config)
    raw_config = asdict(config)
    raw_config.pop("state", None)
    return {"schema_version": SCHEMA_VERSION, "app_config": raw_config}


def app_config_from_dict(payload: dict[str, Any]) -> AppConfig:
    """Load a strict current or explicitly supported legacy GUI document."""

    version, raw_config = _validate_envelope(payload)
    checked = _checked_config_payload(version, raw_config)
    if version in _MIGRATABLE_LEGACY_VERSIONS:
        checked = _migrate_legacy_payload(checked)
    from .release_migration import preserve_import_defaults

    checked = preserve_import_defaults(checked)
    return _build_config(checked)


def save_app_config(path: Path, config: AppConfig) -> None:
    output_path = Path(path)
    document = json.dumps(
        app_config_to_dict(config), indent=2, allow_nan=False
    ) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        document,
        encoding="utf-8",
    )


def load_app_config(path: Path) -> AppConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return app_config_from_dict(payload)
