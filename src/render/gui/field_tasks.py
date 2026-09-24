"""Offline field-baking task adapter for the GUI."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ...config import PROJECT_ROOT
from .models import AppConfig, FieldBakeConfig
from .worker_lifecycle import TaskStopped, _post


def _gas_field_mode(gui_config: FieldBakeConfig) -> str:
    raw_mode = str(getattr(gui_config, "gas_field_mode", "")).strip().lower()
    if not raw_mode:
        raw_mode = "import" if str(gui_config.fluent_path).strip() else "static"
    if raw_mode not in {"static", "import"}:
        raise ValueError("Gas field mode must be 'static' or 'import'.")
    return raw_mode


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    return path.resolve()


def _fluent_path(gui_config: FieldBakeConfig) -> Path | None:
    if _gas_field_mode(gui_config) == "static":
        return None
    path_text = str(gui_config.fluent_path).strip()
    if not path_text:
        raise ValueError("Import gas mode requires a Fluent gas-field path.")
    path = _project_path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Imported gas field does not exist: {path}")
    return path


def _field_bake_kwargs(gui_config: FieldBakeConfig) -> dict[str, Any]:
    return {
        "simion_dc_csv": (
            Path(gui_config.simion_dc_path)
            if str(gui_config.simion_dc_path).strip()
            else None
        ),
        "simion_rf_csv": (
            Path(gui_config.simion_rf_path)
            if str(gui_config.simion_rf_path).strip()
            else None
        ),
        "fluent_csv": _fluent_path(gui_config),
        "offset_z_simion_mm": float(gui_config.offset_z_simion_mm),
        "offset_z_fluent_mm": float(gui_config.offset_z_fluent_mm),
        "output_npy": Path(gui_config.output_npy),
        "output_plot_dir": Path(gui_config.output_plot_dir),
        "background_pressure_pa": float(gui_config.background_pressure_pa),
        "background_temperature_k": float(
            gui_config.background_temperature_k
        ),
        "phi_key_for_plot": str(gui_config.phi_key_for_plot),
        "z_min_mm": float(gui_config.z_min_mm),
        "z_max_mm": float(gui_config.z_max_mm),
        "r_min_mm": float(gui_config.r_min_mm),
        "r_max_mm": float(gui_config.r_max_mm),
        "dz_mm": float(gui_config.dz_mm),
        "dr_mm": float(gui_config.dr_mm),
        "capillary_exit_z_mm": float(gui_config.capillary_exit_z_mm),
        "fluent_capillary_total_length_mm": float(
            gui_config.fluent_capillary_total_length_mm
        ),
        "fluent_capillary_radius_mm": float(
            gui_config.fluent_capillary_radius_mm
        ),
        "fluent_capillary_external_start_z_mm": float(
            gui_config.fluent_capillary_external_start_z_mm
        ),
        "simion_pa_effective_grids_per_mm": float(
            gui_config.pa_grids_per_mm
        ),
        "simion_dc_voltage_scale": float(gui_config.dc_voltage_scale),
    }


def _diagnostic_mask_argument(raw_path: str) -> str:
    text = str(raw_path).strip()
    if text.lower() in {"", "none", "off", "default", "auto"}:
        return text or "none"
    return str(_project_path(text))


def field_diagnostics_command(
    app_config: AppConfig,
    output_path: Path | None = None,
) -> tuple[list[str], Path]:
    """Build the direct diagnostic-module command used by both GUI backends."""
    field_path = _project_path(app_config.loaded_baked_field_path)
    if output_path is None:
        output_path = _project_path(app_config.output_dir) / "field_flow_mask_alignment.png"
    else:
        output_path = _project_path(output_path)
    runtime = app_config.runtime
    command = [
        sys.executable,
        "-m",
        "src.data.diagnostics.plot_field_flow_mask_alignment",
        "--field",
        str(field_path),
        "--electrode-mask",
        _diagnostic_mask_argument(runtime.electrode_mask_path),
        "--output",
        str(output_path),
        "--global-window",
        "--rf-peak-voltage-v",
        str(float(runtime.rf_peak_voltage_v)),
        "--rf-phase-deg",
        str(float(runtime.rf_phase_deg)),
    ]
    if str(runtime.electrode_mask_cache_path).strip():
        command.extend(
            ["--electrode-mask-cache", str(_project_path(runtime.electrode_mask_cache_path))]
        )
    offset = runtime.electrode_mask_z_offset_mm
    if offset is not None and str(offset).strip():
        command.extend(["--electrode-mask-z-offset-mm", str(float(offset))])
    return command, output_path


def _run_diagnostic_process(
    command: list[str],
    stop_event: threading.Event,
) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while True:
        if stop_event.is_set():
            process.terminate()
            try:
                process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise TaskStopped()
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            return int(process.returncode or 0), stdout, stderr
        except subprocess.TimeoutExpired:
            continue


def run_field_bake_task(
    message_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    app_config: AppConfig,
) -> None:
    if stop_event.is_set():
        raise TaskStopped()
    from ...data.tools.field_baker import (
        FieldBakeConfig as BackendFieldBakeConfig,
    )
    from ...data.tools.field_baker import run_field_bake

    backend_config = BackendFieldBakeConfig(
        **_field_bake_kwargs(app_config.field_bake)
    )
    _post(
        message_queue,
        "log",
        channel="logs",
        text="Starting field bake.",
    )
    result = run_field_bake(backend_config)
    _post(message_queue, "field_bake_finished", result=result)


def run_field_diagnostics_task(
    message_queue: "queue.Queue[dict[str, Any]]",
    stop_event: threading.Event,
    app_config: AppConfig,
) -> None:
    """Render field/gas/mask alignment in an isolated Matplotlib process."""
    if stop_event.is_set():
        raise TaskStopped()
    command, output_path = field_diagnostics_command(app_config)
    field_path = _project_path(app_config.loaded_baked_field_path)
    if not field_path.is_file():
        raise FileNotFoundError(f"Baked field does not exist: {field_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _post(message_queue, "log", channel="logs", text="Starting field alignment diagnostic.")
    return_code, stdout, stderr = _run_diagnostic_process(command, stop_event)
    if stdout.strip():
        _post(message_queue, "log", channel="logs", text=stdout.strip())
    if return_code != 0:
        detail = stderr.strip() or f"diagnostic exited with code {return_code}"
        raise RuntimeError(f"Field alignment diagnostic failed: {detail}")
    if not output_path.is_file():
        raise FileNotFoundError(f"Diagnostic did not create its PNG: {output_path}")
    _post(
        message_queue,
        "field_diagnostics_finished",
        path=str(output_path),
        summary_path=str(output_path.with_suffix(".summary.json")),
    )


__all__ = [
    "field_diagnostics_command",
    "run_field_bake_task",
    "run_field_diagnostics_task",
]
