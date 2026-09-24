"""Behavior contracts for the mode-aware GUI beam workflow."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from src.render.gui.beam_dialog import (
    BeamSetupDialog,
    beam_velocity_visibility,
    normalize_gas_velocity_mode,
)
from src.render.gui import beam_dialog
from src.render.gui.beam_tasks import (
    beam_velocity_source,
    run_beam_smoke_task,
    sample_beam_phase_space,
)
from src.render.gui.models import AppConfig, BeamConfig


class _Var:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _Top:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


def _headless_dialog(
    config: BeamConfig,
) -> tuple[BeamSetupDialog, list[BeamConfig]]:
    dialog = BeamSetupDialog.__new__(BeamSetupDialog)
    dialog.config = config
    dialog.vars = {
        name: _Var("" if value is None else str(value))
        for name, value in vars(config).items()
    }
    dialog.top = _Top()
    applied: list[BeamConfig] = []
    dialog.on_apply = applied.append
    return dialog, applied


def _write_constant_gas_field(path: Path) -> None:
    path.write_text(
        "z_m,r_m,vz_m_s,vr_m_s\n"
        "0.0,0.0,100.0,10.0\n"
        "0.0,0.001,100.0,10.0\n"
        "0.001,0.0,100.0,10.0\n"
        "0.001,0.001,100.0,10.0\n",
        encoding="utf-8",
    )


def test_velocity_visibility_is_canonical_and_legacy_compatible() -> None:
    assert normalize_gas_velocity_mode("off") == "static"
    assert beam_velocity_visibility("static") == {
        "static_motion": True,
        "gas_field": False,
    }
    assert beam_velocity_visibility("from-gas-field") == {
        "static_motion": False,
        "gas_field": True,
    }
    with pytest.raises(ValueError, match="static.*from-gas-field"):
        beam_velocity_visibility("automatic")


def test_gas_mode_apply_ignores_and_preserves_hidden_static_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BeamConfig(
        gas_velocity_init_mode="static",
        kinetic_energy_ev=4.0,
        source_temperature_k=320.0,
        source_axial_velocity_m_per_s=275.0,
        direction_axis="+z",
        cone_half_angle_deg=3.0,
        velocity_jitter_m_per_s=2.0,
        source_birth_velocity_gas_csv="gas/original.csv",
    )
    dialog, applied = _headless_dialog(original)
    dialog.vars["gas_velocity_init_mode"].set("from-gas-field")
    dialog.vars["source_birth_velocity_gas_csv"].set("")
    dialog.vars["source_birth_velocity_z_max_mm"].set("")
    dialog.vars["source_birth_velocity_radius_mm"].set("0.25")
    for key in (
        "kinetic_energy_ev",
        "source_temperature_k",
        "source_axial_velocity_m_per_s",
        "cone_half_angle_deg",
        "velocity_jitter_m_per_s",
    ):
        dialog.vars[key].set("invalid-hidden-static")
    dialog.vars["direction_axis"].set("invalid-hidden-direction")
    errors: list[str] = []
    monkeypatch.setattr(
        beam_dialog.messagebox,
        "showerror",
        lambda _title, message, **_kwargs: errors.append(str(message)),
    )

    dialog._apply()

    assert not errors
    assert len(applied) == 1
    updated = applied[0]
    assert updated.gas_velocity_init_mode == "from-gas-field"
    assert updated.source_birth_velocity_gas_csv == ""
    assert updated.source_birth_velocity_z_max_mm is None
    assert updated.source_birth_velocity_radius_mm == pytest.approx(0.25)
    for key in (
        "kinetic_energy_ev",
        "source_temperature_k",
        "source_axial_velocity_m_per_s",
        "direction_axis",
        "cone_half_angle_deg",
        "velocity_jitter_m_per_s",
    ):
        assert getattr(updated, key) == getattr(original, key)
    assert dialog.top.destroyed


def test_static_mode_apply_ignores_and_preserves_hidden_gas_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BeamConfig(
        gas_velocity_init_mode="from-gas-field",
        source_birth_velocity_gas_csv="gas/original.csv",
        source_birth_velocity_z_min_mm=0.2,
        source_birth_velocity_z_max_mm=4.1,
        source_birth_velocity_radius_mm=0.25,
        source_radial_velocity_scale=0.8,
        source_velocity_delta_m_per_s=12.0,
    )
    dialog, applied = _headless_dialog(original)
    dialog.vars["gas_velocity_init_mode"].set("static")
    dialog.vars["kinetic_energy_ev"].set("6.0")
    for key in (
        "source_birth_velocity_z_min_mm",
        "source_birth_velocity_z_max_mm",
        "source_birth_velocity_radius_mm",
        "source_radial_velocity_scale",
        "source_velocity_delta_m_per_s",
    ):
        dialog.vars[key].set("invalid-hidden-gas")
    dialog.vars["source_birth_velocity_gas_csv"].set("hidden/replacement.csv")
    errors: list[str] = []
    monkeypatch.setattr(
        beam_dialog.messagebox,
        "showerror",
        lambda _title, message, **_kwargs: errors.append(str(message)),
    )

    dialog._apply()

    assert not errors
    assert len(applied) == 1
    updated = applied[0]
    assert updated.gas_velocity_init_mode == "static"
    assert updated.kinetic_energy_ev == pytest.approx(6.0)
    for key in (
        "source_birth_velocity_gas_csv",
        "source_birth_velocity_z_min_mm",
        "source_birth_velocity_z_max_mm",
        "source_birth_velocity_radius_mm",
        "source_radial_velocity_scale",
        "source_velocity_delta_m_per_s",
    ):
        assert getattr(updated, key) == getattr(original, key)
    assert dialog.top.destroyed


def test_static_beam_preview_remains_seeded_and_repeatable() -> None:
    beam = BeamConfig(
        gas_velocity_init_mode="static",
        particle_count=32,
        kinetic_energy_ev=8.0,
        beam_radius_mm=0.2,
        initial_position_jitter_mm=0.01,
        direction_axis="-y",
        cone_half_angle_deg=12.0,
        velocity_jitter_m_per_s=3.0,
    )

    first = sample_beam_phase_space(beam)
    second = sample_beam_phase_space(beam)

    assert beam_velocity_source(beam) == "static"
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_gas_field_preview_ignores_hidden_static_motion_values(
    tmp_path: Path,
) -> None:
    gas_path = tmp_path / "gas.csv"
    _write_constant_gas_field(gas_path)
    common = {
        "gas_velocity_init_mode": "from-gas-field",
        "source_birth_velocity_gas_csv": str(gas_path),
        "source_birth_velocity_z_min_mm": 0.0,
        "source_birth_velocity_z_max_mm": 1.0,
        "source_radial_velocity_scale": 0.5,
        "source_velocity_delta_m_per_s": 7.0,
        "particle_count": 32,
        "beam_radius_mm": 0.1,
    }
    first_beam = BeamConfig(
        **common,
        kinetic_energy_ev=1.0,
        direction_axis="+x",
        cone_half_angle_deg=5.0,
        velocity_jitter_m_per_s=2.0,
    )
    second_beam = BeamConfig(
        **common,
        kinetic_energy_ev=100.0,
        direction_axis="-y",
        cone_half_angle_deg=80.0,
        velocity_jitter_m_per_s=500.0,
    )

    first = sample_beam_phase_space(first_beam)
    second = sample_beam_phase_space(second_beam)

    assert beam_velocity_source(first_beam) == "gas-field-csv"
    for name in first:
        np.testing.assert_allclose(first[name], second[name], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first["velocities_m_per_s"][:, 2], 107.0)


def test_gas_field_preview_fails_instead_of_faking_static_velocity(
    tmp_path: Path,
) -> None:
    beam = BeamConfig(
        gas_velocity_init_mode="from-gas-field",
        source_birth_velocity_gas_csv=str(tmp_path / "missing.csv"),
        particle_count=4,
        kinetic_energy_ev=25.0,
        cone_half_angle_deg=30.0,
    )

    with pytest.raises(FileNotFoundError, match="gas CSV not found"):
        sample_beam_phase_space(beam)


def test_beam_smoke_stats_identify_static_velocity_source(
    tmp_path: Path,
) -> None:
    app = AppConfig()
    app.output_dir = str(tmp_path)
    app.beam = BeamConfig(
        gas_velocity_init_mode="static",
        particle_count=5,
        kinetic_energy_ev=2.0,
    )
    messages: "queue.Queue[dict[str, object]]" = queue.Queue()

    run_beam_smoke_task(messages, threading.Event(), app)

    message = messages.get_nowait()
    assert message["type"] == "beam_smoke_finished"
    stats = message["stats"]
    assert isinstance(stats, dict)
    assert stats["velocity_source"] == "static"
