"""Behavior contracts for the GUI field/gas/mask workflow."""

from __future__ import annotations

import inspect
import json
import queue
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.diagnostics.plot_field_flow_mask_alignment import _plot_indices
from src.data.tools.field_baker import (
    FieldBaker,
    FieldBakeConfig as BackendFieldBakeConfig,
    GlobalGrid,
)
from src.render.gui import field_dialog, field_tasks, workers
from src.render.gui.field_dialog import (
    FIELD_NUMERIC_LABELS,
    FieldBakerDialog,
    _initial_gas_mode,
    _normalized_file_suffix,
    _validate_field_config,
    _validate_mask_config,
)
from src.render.gui.models import AppConfig, FieldBakeConfig, RuntimeConfig


class _Var:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _GridGroup:
    def __init__(self) -> None:
        self.visible = False
        self.calls: list[str] = []

    def grid_remove(self) -> None:
        self.visible = False
        self.calls.append("remove")

    def grid(self) -> None:
        self.visible = True
        self.calls.append("grid")


def test_field_dialog_accepts_runtime_mask_config_and_explicit_gas_mode() -> None:
    parameters = inspect.signature(FieldBakerDialog.__init__).parameters
    assert tuple(parameters) == (
        "self",
        "parent",
        "config",
        "runtime_config",
        "on_apply",
    )
    assert _initial_gas_mode(FieldBakeConfig(gas_field_mode="static")) == "static"
    legacy = SimpleNamespace(fluent_path="gas.csv")
    assert _initial_gas_mode(legacy) == "import"


def test_field_baker_uses_current_geometry_and_explicit_fluent_offset() -> None:
    assert FieldBakeConfig().offset_z_fluent_mm == pytest.approx(-95.0)
    config = FieldBakeConfig(offset_z_fluent_mm=-87.25)
    assert config.capillary_exit_z_mm == pytest.approx(6.0)
    assert config.fluent_capillary_total_length_mm == pytest.approx(101.0)
    assert config.fluent_capillary_radius_mm == pytest.approx(0.25)
    assert config.fluent_capillary_external_start_z_mm == pytest.approx(4.5)

    kwargs = field_tasks._field_bake_kwargs(config)
    assert kwargs["capillary_exit_z_mm"] == pytest.approx(6.0)
    assert kwargs["fluent_capillary_total_length_mm"] == pytest.approx(101.0)
    assert kwargs["fluent_capillary_radius_mm"] == pytest.approx(0.25)
    assert kwargs["fluent_capillary_external_start_z_mm"] == pytest.approx(4.5)
    backend = BackendFieldBakeConfig(**kwargs)
    assert backend.resolved_fluent_z_offset_mm() == pytest.approx(-87.25)
    assert backend.fluent_z_offset_mode() == "manual_offset_with_capillary_metadata"


def test_current_capillary_geometry_masks_only_the_internal_fluent_region(
    tmp_path: Path,
) -> None:
    gas_path = tmp_path / "gas.csv"
    rows = ["z(m),r(m),Pressure(Pa),Temperature(K),V_z(m/s),V_r(m/s)"]
    for raw_z_m in (0.099, 0.100, 0.101, 0.102):
        for radius_m in (0.0, 0.00025, 0.0005):
            rows.append(f"{raw_z_m},{radius_m},1000,400,10,2")
    gas_path.write_text("\n".join(rows), encoding="utf-8")
    grid = GlobalGrid(
        z_min_m=0.004,
        z_max_m=0.007,
        r_min_m=0.0,
        r_max_m=0.0005,
        dz_m=0.001,
        dr_m=0.00025,
        capillary_exit_z_m=0.006,
    )

    baked = FieldBaker(grid).bake_fluent_field(
        gas_path,
        -95.0,
        background_pressure_pa=100.0,
        background_temperature_k=300.0,
        capillary_total_length_mm=101.0,
        capillary_radius_mm=0.25,
        capillary_external_start_z_mm=4.5,
    )

    pressure = baked["pressure_pa"]
    assert pressure[0, 1] == pytest.approx(100.0)  # z=5 mm, r=0
    assert pressure[1, 1] == pytest.approx(100.0)  # radius boundary is internal
    assert pressure[2, 1] == pytest.approx(1000.0)  # z=5 mm, r=0.5 mm
    assert pressure[0, 2] == pytest.approx(1000.0)  # z=6 mm keeps all radii


def test_global_diagnostic_sampling_preserves_axis_endpoints() -> None:
    indices = np.arange(6501, dtype=np.int64)
    sampled = _plot_indices(indices, 1200)
    assert sampled.size == 1200
    assert sampled[0] == 0
    assert sampled[-1] == 6500
    assert np.all(np.diff(sampled) > 0)


@pytest.mark.parametrize("mode", ("static", "import"))
def test_gas_mode_switch_uses_one_grid_geometry_manager(mode: str) -> None:
    dialog = FieldBakerDialog.__new__(FieldBakerDialog)
    dialog.vars = {"gas_field_mode": _Var(mode)}
    dialog.static_gas_group = _GridGroup()
    dialog.import_gas_group = _GridGroup()

    dialog._sync_gas_mode()

    expected = (
        dialog.import_gas_group if mode == "import" else dialog.static_gas_group
    )
    hidden = (
        dialog.static_gas_group if mode == "import" else dialog.import_gas_group
    )
    assert expected.visible is True
    assert expected.calls == ["remove", "grid"]
    assert hidden.visible is False
    assert hidden.calls == ["remove"]


def test_field_save_paths_use_canonical_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = FieldBakerDialog.__new__(FieldBakerDialog)
    dialog.top = object()
    responses = iter((tmp_path / "baked.txt", tmp_path / "mask.npy"))
    calls: list[dict[str, object]] = []

    def fake_save(**options: object) -> str:
        calls.append(options)
        return str(next(responses))

    monkeypatch.setattr(
        field_dialog.filedialog,
        "asksaveasfilename",
        fake_save,
    )
    output_var = _Var()
    cache_var = _Var()
    dialog._browse_file(
        output_var,
        key="output_npy",
        save=True,
        mask=False,
    )
    dialog._browse_file(
        cache_var,
        key="electrode_mask_cache_path",
        save=True,
        mask=True,
    )

    assert output_var.get() == str(tmp_path / "baked.npy")
    assert cache_var.get() == str(tmp_path / "mask.npz")
    assert calls[0]["defaultextension"] == ".npy"
    assert calls[0]["filetypes"] == [
        ("NumPy field", "*.npy"),
        ("All files", "*.*"),
    ]
    assert calls[1]["defaultextension"] == ".npz"
    assert calls[1]["filetypes"] == [
        ("Electrode mask cache", "*.npz"),
        ("All files", "*.*"),
    ]


@pytest.mark.parametrize("suffix", (".npy", ".npz"))
def test_file_suffix_normalization_replaces_wrong_or_uppercase_suffix(
    suffix: str,
) -> None:
    assert _normalized_file_suffix("result.bad", suffix) == f"result{suffix}"
    assert _normalized_file_suffix("result" + suffix.upper(), suffix) == f"result{suffix}"
    assert _normalized_file_suffix("", suffix) == ""


@pytest.mark.parametrize("field_name", tuple(FIELD_NUMERIC_LABELS))
@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), float("-inf")))
def test_field_dialog_rejects_nonfinite_field_numbers(
    field_name: str,
    invalid: float,
) -> None:
    config = replace(FieldBakeConfig(), **{field_name: invalid})
    with pytest.raises(ValueError, match="must be finite"):
        _validate_field_config(config)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"fluent_capillary_total_length_mm": 0.0}, "total length"),
        ({"fluent_capillary_radius_mm": 0.0}, "radius"),
        (
            {
                "capillary_exit_z_mm": 6.0,
                "fluent_capillary_external_start_z_mm": 6.0,
            },
            "external gas start z",
        ),
    ),
)
def test_field_dialog_rejects_invalid_capillary_geometry(
    changes: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_field_config(replace(FieldBakeConfig(), **changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"electrode_mask_z_offset_mm": float("nan")}, "Mask z offset"),
        ({"electrode_mask_z_offset_mm": float("inf")}, "Mask z offset"),
        ({"electrode_hit_distance_mm": float("nan")}, "Electrode hit distance"),
        ({"electrode_hit_distance_mm": float("inf")}, "Electrode hit distance"),
    ),
)
def test_field_dialog_rejects_nonfinite_mask_numbers(
    changes: dict[str, float],
    message: str,
) -> None:
    runtime = replace(RuntimeConfig(), **changes)
    with pytest.raises(ValueError, match=message):
        _validate_mask_config(runtime)


def test_field_apply_normalizes_manually_entered_save_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = FieldBakeConfig()
    runtime = RuntimeConfig()
    dialog = FieldBakerDialog.__new__(FieldBakerDialog)
    dialog.config = config
    dialog.runtime_config = runtime
    dialog.vars = {
        key: _Var(value)
        for key, value in vars(config).items()
    }
    dialog.vars.update(
        electrode_mask_path=_Var(runtime.electrode_mask_path),
        electrode_mask_cache_path=_Var("cache/wrong.npy"),
        electrode_mask_z_offset_mm=_Var(""),
        electrode_hit_distance_mm=_Var(runtime.electrode_hit_distance_mm),
    )
    dialog.vars["output_npy"] = _Var("fields/wrong.txt")
    applied: list[tuple[FieldBakeConfig, RuntimeConfig]] = []
    destroyed: list[bool] = []
    dialog.on_apply = lambda field, mask: applied.append((field, mask))
    dialog.top = SimpleNamespace(destroy=lambda: destroyed.append(True))
    monkeypatch.setattr(
        field_dialog.messagebox,
        "showerror",
        lambda *_args, **_kwargs: pytest.fail("valid config was rejected"),
    )

    dialog._apply()

    assert applied[0][0].output_npy == str(Path("fields") / "wrong.npy")
    assert applied[0][1].electrode_mask_cache_path == str(
        Path("cache") / "wrong.npz"
    )
    assert destroyed == [True]


def test_static_gas_mode_ignores_fluent_path_and_keeps_uniform_state() -> None:
    config = FieldBakeConfig(
        gas_field_mode="static",
        fluent_path="missing-gas.csv",
        background_pressure_pa=321.0,
        background_temperature_k=456.0,
    )
    kwargs = field_tasks._field_bake_kwargs(config)
    assert kwargs["fluent_csv"] is None
    assert kwargs["background_pressure_pa"] == pytest.approx(321.0)
    assert kwargs["background_temperature_k"] == pytest.approx(456.0)


def test_import_gas_mode_requires_an_existing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a Fluent gas-field path"):
        field_tasks._field_bake_kwargs(
            FieldBakeConfig(gas_field_mode="import", fluent_path="")
        )

    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Imported gas field"):
        field_tasks._field_bake_kwargs(
            FieldBakeConfig(gas_field_mode="import", fluent_path=str(missing))
        )

    gas_path = tmp_path / "gas.csv"
    gas_path.write_text("z,r,p,t,vz,vr\n", encoding="utf-8")
    kwargs = field_tasks._field_bake_kwargs(
        FieldBakeConfig(gas_field_mode="import", fluent_path=str(gas_path))
    )
    assert kwargs["fluent_csv"] == gas_path.resolve()


def test_diagnostic_command_calls_requested_module_with_field_mask_and_rf(
    tmp_path: Path,
) -> None:
    config = AppConfig()
    config.loaded_baked_field_path = str(tmp_path / "baked.npy")
    config.runtime.electrode_mask_path = "default"
    config.runtime.electrode_mask_cache_path = str(tmp_path / "mask.npz")
    config.runtime.electrode_mask_z_offset_mm = 1.25
    config.runtime.rf_peak_voltage_v = 77.0
    config.runtime.rf_phase_deg = -45.0
    output_path = tmp_path / "alignment.png"

    command, resolved_output = field_tasks.field_diagnostics_command(
        config,
        output_path,
    )

    assert command[:3] == [
        sys.executable,
        "-m",
        "src.data.diagnostics.plot_field_flow_mask_alignment",
    ]
    assert command[command.index("--field") + 1] == str(
        (tmp_path / "baked.npy").resolve()
    )
    assert command[command.index("--electrode-mask") + 1] == "default"
    assert command[command.index("--electrode-mask-cache") + 1] == str(
        (tmp_path / "mask.npz").resolve()
    )
    assert command[command.index("--electrode-mask-z-offset-mm") + 1] == "1.25"
    assert command[command.index("--rf-peak-voltage-v") + 1] == "77.0"
    assert command[command.index("--rf-phase-deg") + 1] == "-45.0"
    assert "--global-window" in command
    assert resolved_output == output_path.resolve()


def test_diagnostic_worker_posts_generated_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field_path = tmp_path / "baked.npy"
    field_path.write_bytes(b"placeholder")
    config = AppConfig(
        output_dir=str(tmp_path / "results"),
        loaded_baked_field_path=str(field_path),
    )
    config.runtime.electrode_mask_cache_path = ""

    def fake_process(
        command: list[str],
        _stop_event: threading.Event,
    ) -> tuple[int, str, str]:
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"png")
        output.with_suffix(".summary.json").write_text("{}", encoding="utf-8")
        return 0, f"Saved: {output}", ""

    monkeypatch.setattr(field_tasks, "_run_diagnostic_process", fake_process)
    messages: "queue.Queue[dict[str, object]]" = queue.Queue()
    field_tasks.run_field_diagnostics_task(messages, threading.Event(), config)
    payloads = list(messages.queue)
    finished = next(
        payload for payload in payloads
        if payload.get("type") == "field_diagnostics_finished"
    )
    assert Path(str(finished["path"])).read_bytes() == b"png"
    assert Path(str(finished["summary_path"])).is_file()


def test_workers_reexports_field_diagnostics_task() -> None:
    assert workers.run_field_diagnostics_task is field_tasks.run_field_diagnostics_task


def test_diagnostic_worker_runs_real_alignment_module(tmp_path: Path) -> None:
    z_coords_m = np.array([0.0, 32.5e-3, 65.0e-3])
    r_coords_m = np.array([0.0, 10.0e-3, 20.0e-3])
    shape = (r_coords_m.size, z_coords_m.size)
    field_path = tmp_path / "synthetic_baked.npy"
    np.save(
        field_path,
        {
            "grid": {
                "z_coords_m": z_coords_m,
                "r_coords_m": r_coords_m,
                "capillary_exit_z_m": 5.0e-3,
            },
            "simion": {
                "e_dc_r_v_per_m": np.ones(shape),
                "e_dc_z_v_per_m": np.full(shape, 2.0),
                "e_rf_r_v_per_m": np.full(shape, 0.5),
                "e_rf_z_v_per_m": np.full(shape, 0.25),
                "rf_reference_peak_voltage_v": 1.0,
            },
            "fluent": {
                "v_r_m_per_s": np.zeros(shape),
                "v_z_m_per_s": np.full(shape, 10.0),
            },
        },
        allow_pickle=True,
    )
    config = AppConfig(
        output_dir=str(tmp_path / "diagnostics"),
        loaded_baked_field_path=str(field_path),
    )
    config.runtime.electrode_mask_path = "none"
    config.runtime.electrode_mask_cache_path = ""
    messages: "queue.Queue[dict[str, object]]" = queue.Queue()

    field_tasks.run_field_diagnostics_task(messages, threading.Event(), config)

    finished = next(
        payload for payload in list(messages.queue)
        if payload.get("type") == "field_diagnostics_finished"
    )
    assert Path(str(finished["path"])).stat().st_size > 0
    summary_path = Path(str(finished["summary_path"]))
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["z_window_mm"] == pytest.approx([0.0, 65.0])
    assert summary["r_window_mm"] == pytest.approx([0.0, 20.0])
