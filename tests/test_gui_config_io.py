"""Strict persistence contracts for GUI JSON configuration documents."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest

from src.render.gui.config_io import (
    SCHEMA_VERSION,
    app_config_from_dict,
    app_config_to_dict,
    load_app_config,
    save_app_config,
)
from src.render.gui.models import AppConfig


def _current_payload() -> dict[str, object]:
    return app_config_to_dict(AppConfig())


def test_gui_json_excludes_session_state_and_load_starts_fresh(tmp_path) -> None:
    config = AppConfig(loaded_baked_field_path="outputs/custom.npy")
    config.state.session_id = "stale-session"
    config.state.current_status = "RUNNING"
    config.state.current_macro_step = 123
    config.state.alive_count = 99

    payload = app_config_to_dict(config)
    assert payload["schema_version"] == SCHEMA_VERSION == 5
    assert "state" not in payload["app_config"]

    path = tmp_path / "gui.json"
    save_app_config(path, config)
    loaded = load_app_config(path)
    assert loaded.state.session_id != "stale-session"
    assert loaded.state.current_status == "CONFIGURED"
    assert loaded.state.current_macro_step == 0
    assert loaded.state.alive_count == 0
    assert loaded.state.loaded_baked_field_path == "outputs/custom.npy"


def test_gui_json_file_round_trip_preserves_all_reusable_config(tmp_path) -> None:
    config = AppConfig()
    config.session_name = "strict-round-trip"
    config.beam.source_birth_velocity_z_max_mm = None
    config.beam.source_birth_velocity_radius_mm = 0.25
    config.runtime.pic_space_charge_scale = 0.125
    config.runtime.pic_grid_nr = 201
    config.runtime.pic_grid_nz = 651
    config.runtime.collision_physics_backend = "iict-lite"
    config.runtime.ionspa_backend = "local"
    config.runtime.iict_parameter_config_path = "configs/iict.json"
    config.runtime.iict_pseudoatom_mass_da = 31.0
    config.output.save_figures = False

    path = tmp_path / "complete-gui-config.json"
    save_app_config(path, config)
    loaded = load_app_config(path)

    expected = asdict(config)
    actual = asdict(loaded)
    expected.pop("state")
    actual.pop("state")
    assert actual == expected


def test_current_gui_json_omitted_birth_limits_default_to_none() -> None:
    payload = _current_payload()
    beam = payload["app_config"]["beam"]  # type: ignore[index]
    beam.pop("source_birth_velocity_z_max_mm")
    beam.pop("source_birth_velocity_radius_mm")

    loaded = app_config_from_dict(payload)

    assert loaded.beam.source_birth_velocity_z_max_mm is None
    assert loaded.beam.source_birth_velocity_radius_mm is None


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"app_config": {}}, "Missing"),
        ({"schema_version": True, "app_config": {}}, "schema_version"),
        ({"schema_version": 6, "app_config": {}}, "schema_version"),
        ({"schema_version": 5, "app_config": {}, "extra": 1}, "Unknown document"),
        ({"schema_version": 5, "app_config": []}, "app_config must"),
    ],
)
def test_gui_json_rejects_invalid_document_envelopes(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        app_config_from_dict(payload)


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("app_config", "unknown"), 1, "Unknown app_config"),
        (("app_config", "state"), {}, "Unknown app_config"),
        (("app_config", "beam", "unknown"), 1, "Unknown app_config.beam"),
        (("app_config", "beam", "particle_count"), True, "invalid JSON type"),
        (("app_config", "beam", "mass_amu"), "2000", "invalid JSON type"),
        (("app_config", "runtime", "pic_poisson_warm_start"), 1, "invalid JSON type"),
        (("app_config", "runtime", "iict_num_atoms"), True, "invalid JSON type"),
        (("app_config", "runtime", "iict_pseudoatom_mass_da"), "30", "invalid JSON type"),
        (("app_config", "output"), [], "must be a JSON object"),
    ],
)
def test_current_gui_json_rejects_unknown_fields_and_wrong_types(
    path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    payload = deepcopy(_current_payload())
    target = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=match):
        app_config_from_dict(payload)


@pytest.mark.parametrize(
    "group, field, value, match",
    [
        ("beam", "mass_amu", 0.0, "must be positive"),
        ("beam", "source_birth_velocity_radius_mm", -0.1, "must be positive"),
        ("field_bake", "dz_mm", 0.0, "must be positive"),
        ("runtime", "pic_space_charge_scale", -0.1, "must be non-negative"),
        ("runtime", "pic_grid_nr", 2, "PIC grids"),
        ("runtime", "langevin_switch_prob", 1.1, "must be in"),
        ("runtime", "collision_physics_backend", "automatic", "must be one of"),
        ("runtime", "ionspa_backend", "automatic", "must be one of"),
        ("runtime", "iict_heat_capacity_model", "profile", "must be one of"),
        ("runtime", "iict_num_atoms", 2, "greater than 2"),
        ("runtime", "iict_pseudoatom_mass_da", 0.0, "positive"),
        ("runtime", "iict_delta_s_j_per_mol_k", float("nan"), "finite"),
        ("output", "terminal_time_bin_ms", 0.0, "must be positive"),
        ("beam", "delta_s_j_per_mol_k", float("nan"), "must be finite"),
    ],
)
def test_gui_json_rejects_invalid_ranges(
    group: str,
    field: str,
    value: object,
    match: str,
) -> None:
    payload = deepcopy(_current_payload())
    payload["app_config"][group][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=match):
        app_config_from_dict(payload)


def test_legacy_gui_json_migrates_only_documented_fields_and_resets_state() -> None:
    payload = {
        "schema_version": 3,
        "app_config": {
            "beam": {"gas_velocity_init_mode": "off"},
            "runtime": {
                "enable_space_charge": False,
                "pic_coupling_mode": "coupled",
                "pic_grid_nr": 101,
                "pic_grid_nz": 301,
                "sor_omega": 1.5,
                "pic_poisson_backend": "sor",
            },
            "field_bake": {},
            "state": {
                "session_id": "old-session",
                "current_status": "RUNNING",
                "current_macro_step": 20,
            },
        },
    }
    migrated = app_config_from_dict(payload)
    assert migrated.beam.gas_velocity_init_mode == "static"
    assert migrated.runtime.pic_space_charge_scale == 0.0
    assert (migrated.runtime.pic_grid_nr, migrated.runtime.pic_grid_nz) == (0, 0)
    assert migrated.runtime.pic_poisson_backend == "amg"
    assert migrated.state.session_id != "old-session"
    assert migrated.state.current_status == "CONFIGURED"
    assert migrated.state.current_macro_step == 0


def test_schema_v4_loads_strictly_and_adds_new_backend_defaults() -> None:
    payload = {
        "schema_version": 4,
        "app_config": {
            "session_name": "v4-project",
            "runtime": {"ionspa_backend": "approximate"},
        },
    }

    loaded = app_config_from_dict(payload)

    assert loaded.runtime.collision_physics_backend == "ionspa"
    assert loaded.runtime.ionspa_backend == "approximate"
    assert loaded.runtime.iict_parameter_config_path == ""
    assert loaded.runtime.iict_heat_capacity_model is None
    assert app_config_to_dict(loaded)["schema_version"] == 5


@pytest.mark.parametrize(
    "runtime",
    [
        {"sor_omega": 1.5},
        {"state": {}},
    ],
)
def test_schema_v4_does_not_reaccept_v1_to_v3_legacy_fields(
    runtime: dict[str, object],
) -> None:
    app_config: dict[str, object]
    if "state" in runtime:
        app_config = runtime
    else:
        app_config = {"runtime": runtime}
    with pytest.raises(ValueError, match="Unknown"):
        app_config_from_dict(
            {"schema_version": 4, "app_config": app_config}
        )


def test_iict_lite_accepts_json_source_or_complete_direct_models() -> None:
    json_payload = deepcopy(_current_payload())
    json_runtime = json_payload["app_config"]["runtime"]  # type: ignore[index]
    json_runtime["collision_physics_backend"] = "iict-lite"
    json_runtime["iict_parameter_config_path"] = "configs/portable-iict.json"
    loaded_json = app_config_from_dict(json_payload)
    assert loaded_json.runtime.iict_heat_capacity_model is None

    direct_payload = deepcopy(_current_payload())
    direct_runtime = direct_payload["app_config"]["runtime"]  # type: ignore[index]
    direct_runtime.update({
        "collision_physics_backend": "iict-lite",
        "iict_heat_capacity_model": "classical",
        "iict_pseudoatom_model": "constant",
        "iict_pseudoatom_mass_da": 30.0,
        "iict_fragmentation_model": "none",
    })
    loaded_direct = app_config_from_dict(direct_payload)
    assert loaded_direct.runtime.iict_pseudoatom_mass_da == pytest.approx(30.0)


def test_iict_lite_rejects_incomplete_direct_models_but_ionspa_keeps_draft() -> None:
    payload = deepcopy(_current_payload())
    runtime = payload["app_config"]["runtime"]  # type: ignore[index]
    runtime["collision_physics_backend"] = "iict-lite"
    runtime["iict_parameter_config_path"] = ""
    with pytest.raises(ValueError, match="parameter JSON or explicit"):
        app_config_from_dict(payload)

    runtime["collision_physics_backend"] = "ionspa"
    runtime["iict_pseudoatom_model"] = "constant"
    loaded = app_config_from_dict(payload)
    assert loaded.runtime.iict_pseudoatom_model == "constant"


def test_iict_direct_model_requirements_and_mass_bounds_are_strict() -> None:
    payload = deepcopy(_current_payload())
    runtime = payload["app_config"]["runtime"]  # type: ignore[index]
    runtime.update({
        "iict_parameter_config_path": "",
        "collision_physics_backend": "iict-lite",
        "iict_heat_capacity_model": "constant_cv",
        "iict_pseudoatom_model": "constant",
        "iict_pseudoatom_mass_da": 30.0,
        "iict_fragmentation_model": "eyring",
    })
    with pytest.raises(ValueError, match="constant_cv"):
        app_config_from_dict(payload)
    runtime["iict_constant_cv_j_per_k_per_ion"] = 1.0e-20
    with pytest.raises(ValueError, match="Eyring"):
        app_config_from_dict(payload)
    runtime["iict_delta_h_kj_per_mol"] = 80.0
    runtime["iict_delta_s_j_per_mol_k"] = -120.0
    runtime["iict_pseudoatom_min_mass_da"] = 40.0
    with pytest.raises(ValueError, match="below its minimum"):
        app_config_from_dict(payload)


def test_legacy_gui_json_rejects_undocumented_fields() -> None:
    payload = {
        "schema_version": 3,
        "app_config": {"runtime": {"obsolete_magic": 1}},
    }
    with pytest.raises(ValueError, match="Unknown app_config.runtime"):
        app_config_from_dict(payload)
