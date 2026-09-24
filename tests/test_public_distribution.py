"""Public demos, migration, package resources, and provider isolation."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from src.config import PROJECT_ROOT, load_config_document
from src.env.collisions import backend
from src.env.collisions.adapter import IonSpaCollisionAdapter
from src.env.collisions.models import IonSpaLoadError
from src.render.gui.config_io import app_config_from_dict, app_config_to_dict
from src.render.gui.models import AppConfig


def test_distribution_identity_and_license_metadata() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "mucite"
    assert metadata["project"]["scripts"] == {
        "mucite": "src.render.cli.runner:run_cli",
        "mucite-gui": "src.render.gui:launch_main_window",
    }
    assert metadata["project"]["license"] == {"file": "LICENSE.md"}
    assert (PROJECT_ROOT / "LICENSE.md").is_file()
    assert "license: GPL-3.0-only" in (PROJECT_ROOT / "CITATION.cff").read_text(
        encoding="utf-8"
    )


def test_default_demo_runs_without_importing_or_opening_ionspa() -> None:
    script = r'''
import importlib.abc
import pathlib
import sys
class RejectIonSpa(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if "ionspa" in fullname.split("."):
            raise AssertionError("Unexpected IonSPA import: " + fullname)
def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = str(args[0]).replace(chr(92), "/").lower()
        if "/ionspa/" in name or "hcprofiles" in name:
            raise AssertionError("Unexpected IonSPA resource access")
sys.meta_path.insert(0, RejectIonSpa())
sys.addaudithook(audit)
import src
assert pathlib.Path(src.__file__).resolve().parent.parent == pathlib.Path.cwd()
from src.render.cli.runner import run_cli
result = run_cli(["--config", "configs/headless_smoke.json"])
assert abs(result.final_time_s - 1e-9) < 1e-15
assert result.collision_count > 0
assert not any("ionspa" in name.split(".") for name in sys.modules)
print("PUBLIC_ISOLATED_SMOKE_OK", result.collision_count)
'''
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env,
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PUBLIC_ISOLATED_SMOKE_OK" in result.stdout


def test_bundled_provider_fails_before_import(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("Provider loading must not be attempted")
    monkeypatch.setattr(backend, "import_module", unexpected)
    with pytest.raises(IonSpaLoadError, match="not bundled"):
        IonSpaCollisionAdapter(backend="bundled")


def test_external_provider_is_lazy_and_reports_origin(monkeypatch) -> None:
    calls = []
    provider = SimpleNamespace(
        ionclass=lambda *a, **k: None, cellclass=lambda *a, **k: None,
        fracloss=lambda *a, **k: 0.0, get_version=lambda: "external-test-stub",
        __name__="ionspa", __file__=str(PROJECT_ROOT / "stub-provider.py"),
    )
    def load(name):
        calls.append(name)
        return provider
    monkeypatch.setattr(backend, "import_module", load)
    adapter = IonSpaCollisionAdapter(backend="local")
    assert calls == ["ionspa"]
    assert adapter.runtime_info["version"] == "external-test-stub"
    assert adapter.runtime_info["canonical_package_dir"] is None


def test_missing_local_provider_has_explicit_error(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("ionspa")
    monkeypatch.setattr(backend, "import_module", missing)
    with pytest.raises(IonSpaLoadError, match="lawful local copy"):
        IonSpaCollisionAdapter(backend="local")


def test_new_gui_demo_is_complete_and_roundtrips() -> None:
    config = AppConfig()
    assert config.runtime.dummy_static_field
    assert config.runtime.collision_physics_backend == "iict-lite"
    assert config.runtime.fragmentation_mode == "off"
    assert Path(config.runtime.iict_parameter_config_path).is_file()
    assert config.runtime.pic_grid_nr == config.runtime.pic_grid_nz == 0
    restored = app_config_from_dict(app_config_to_dict(config))
    assert app_config_to_dict(restored) == app_config_to_dict(config)


@pytest.mark.parametrize("version", [3, 4, 5])
def test_omitted_legacy_backend_never_switches_to_demo(version: int) -> None:
    config = app_config_from_dict({"schema_version": version, "app_config": {}})
    assert config.runtime.collision_physics_backend == "ionspa"
    assert config.runtime.ionspa_backend == "bundled"
    assert config.runtime.iict_parameter_config_path == ""
    assert not config.runtime.dummy_static_field
    assert config.beam.particle_count == 50000


def test_packaged_default_survives_missing_checkout_configs(monkeypatch, tmp_path) -> None:
    from src.config import release_defaults
    from src.env.collisions.factory import build_collision_physics_backend

    monkeypatch.setattr(release_defaults, "PROJECT_ROOT", tmp_path)
    document = release_defaults.load_release_default()
    adapter = build_collision_physics_backend(
        config=document.simulation, template=document.ion, project_root=tmp_path,
    )
    assert adapter.runtime_info["effective_backend"] == "iict-lite"
    assert document.simulation.iict_parameter_config_path.is_absolute()


def test_packaged_and_checkout_demos_match() -> None:
    preset = PROJECT_ROOT / "src/config/presets"
    for name in ["default.json", "iict_lite_parameters.example.json"]:
        assert json.loads((preset / name).read_text()) == json.loads(
            (PROJECT_ROOT / "configs" / name).read_text()
        )
    load_config_document(PROJECT_ROOT / "configs/headless_smoke.json")


def test_public_source_boundary_audit() -> None:
    path = PROJECT_ROOT / "tools/check_release.py"
    spec = importlib.util.spec_from_file_location("public_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.audit_source()["errors"]
