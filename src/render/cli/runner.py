"""CLI execution orchestration over canonical request documents."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Optional

from ...config import (
    DEFAULT_STATIC_FIELD_PATH,
    ConfigDocument,
    ExecutionConfig,
    IonTemplate,
    OutputConfig,
    SimulationConfig,
    SimulationResult,
    dump_config_document,
)
from ...data.report import (
    _stdout_progress,
    _write_simulation_report,
    capture_report_snapshot,
)
from ...core import EnginePolicy
from .config_adapter import resolve_cli_document
from .parser import parse_cli


def build_demo_simulation(
    *,
    config: Optional[SimulationConfig] = None,
    template: Optional[IonTemplate] = None,
    particle_backend: str = "cpu",
    data_logger: Optional[Any] = None,
    engine_policy: Optional[EnginePolicy] = None,
) -> Any:
    """Construct the production simulation while keeping CLI import lightweight."""

    from ...core.simulation import FlowchartPicSimulation

    template = template or IonTemplate()
    if config is None:
        config = replace(
            SimulationConfig(),
            static_field_path=DEFAULT_STATIC_FIELD_PATH,
        )
    return FlowchartPicSimulation(
        template,
        config,
        particle_backend=particle_backend,
        data_logger=data_logger,
        engine_policy=engine_policy,
    )


def _build_engine_policy(
    execution: ExecutionConfig,
    output: OutputConfig,
) -> EnginePolicy:
    return EnginePolicy(
        progress_interval_s=execution.progress_interval_s,
        dt_quantile_reservoir_size=output.dt_quantile_reservoir_size,
        macro_history_max_rows=output.macro_history_max_rows,
    )


def _report_dir_hint(output: OutputConfig) -> Optional[Path]:
    if output.report_dir is not None:
        return Path(output.report_dir)
    if output.export_dir is not None:
        return Path(output.export_dir)
    if output.report:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return Path("outputs") / f"{stamp}_src_main_report"
    return None


def _build_data_logger(
    output: OutputConfig,
    report_dir_hint: Optional[Path],
) -> tuple[Optional[Any], Optional[Path]]:
    if output.export_every <= 0:
        return None, report_dir_hint
    from ...data.logger import DataLogger

    output_dir = output.export_dir or report_dir_hint
    if output_dir is None:
        output_dir = Path("outputs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir_hint = report_dir_hint or Path(output_dir)
    logger = DataLogger(
        output_dir,
        export_every_macro_steps=output.export_every,
        file_format=output.export_format,
        representative_trajectory_count=output.trajectory_sample_count,
        trajectory_record_every_snapshots=output.trajectory_record_every,
        make_plots=output.snapshot_plots_enabled,
        plot_max_points=output.snapshot_plot_max_points,
        trajectory_plot_max_tracks=output.trajectory_plot_max_tracks,
    )
    return logger, report_dir_hint


def _run_simulation(
    simulation: Any,
    execution: ExecutionConfig,
) -> SimulationResult:
    result: Optional[SimulationResult] = None
    if execution.live_window or not execution.no_window:
        try:
            from ..runtime_window import launch_runtime_window
        except ModuleNotFoundError as exc:
            if exc.name != "tkinter":
                raise
            print("Tkinter is unavailable; falling back to command-line mode.")
        else:
            result = launch_runtime_window(simulation)
    if result is None:
        callback = _stdout_progress if execution.progress else None
        result = simulation.run(progress_callback=callback)
    return result


def _print_result(result: SimulationResult) -> None:
    print(f"Backend: {result.particle_backend}")
    print(f"Termination: {result.termination_reason or 'completed'}")
    print(f"Survivors: {len(result.survivors)}")
    print(f"Collisions: {result.collision_count}")
    if result.export_directory:
        print(f"Exports: {result.export_directory}")
        export_dir = Path(result.export_directory)
        artifact_names = (
            "ion_spatial_distribution_rz.png",
            "ion_spatial_distribution_xyz.png",
            "representative_trajectories.csv",
            "representative_trajectories_rz.png",
        )
        for artifact_name in artifact_names:
            artifact_path = export_dir / artifact_name
            if artifact_path.exists():
                print(f"Export artifact: {artifact_path}")
    if result.macro_history:
        print(result.macro_history[-1])


def _final_report_dir(
    output: OutputConfig,
    result: SimulationResult,
    report_dir_hint: Optional[Path],
) -> Path:
    if output.report_dir is not None:
        return Path(output.report_dir)
    if result.export_directory:
        return Path(result.export_directory)
    if output.export_dir is not None:
        return Path(output.export_dir)
    if report_dir_hint is None:
        raise RuntimeError("Report output was requested without a directory.")
    return Path(report_dir_hint)


def _write_report(
    document: ConfigDocument,
    simulation: Any,
    result: SimulationResult,
    argv: tuple[str, ...],
    report_dir_hint: Optional[Path],
) -> None:
    output = document.output
    if not output.report and output.report_dir is None:
        return
    snapshot = capture_report_snapshot(
        argv=list(argv),
        requested_config=document.simulation,
        template=document.ion,
        simulation=simulation,
        result=result,
    )
    paths = _write_simulation_report(
        report_dir=_final_report_dir(output, result, report_dir_hint),
        snapshot=snapshot,
        report_plot_max_points=output.report_plot_max_points,
    )
    for label, path in paths.items():
        print(f"{label.capitalize()}: {path}")


def _launch_gui() -> None:
    from ..gui import launch_main_window

    launch_main_window()


def _configure_terminal_recorder(
    simulation: Any,
    output: OutputConfig,
    report_hint: Optional[Path],
) -> None:
    recorder = simulation.terminal_event_recorder
    configure_seed = getattr(recorder, "configure_sample_seed", None)
    if callable(configure_seed):
        configure_seed(output.terminal_event_sample_seed)
    if report_hint is None:
        return
    open_stream = recorder.open_csv_stream
    parameters = signature(open_stream).parameters.values()
    supports_sample_size = any(
        parameter.name == "sample_size"
        or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    options: dict[str, object] = {"stream_only": True}
    if supports_sample_size:
        options["sample_size"] = output.terminal_event_sample_rows
    open_stream(Path(report_hint) / "terminal_events.csv", **options)


def run_cli(argv: Optional[list[str]] = None) -> Optional[SimulationResult]:
    """Parse, resolve, optionally persist, and execute one CLI request."""

    parsed = parse_cli(argv)
    document = resolve_cli_document(
        parsed.namespace,
        parsed.explicit_dests,
    )
    write_path = str(parsed.namespace.write_config).strip()
    if write_path:
        dump_config_document(Path(write_path), document)
        print(f"Configuration: {Path(write_path)}")
        return None
    if document.execution.gui:
        _launch_gui()
        return None
    report_hint = _report_dir_hint(document.output)
    data_logger, report_hint = _build_data_logger(document.output, report_hint)
    simulation = build_demo_simulation(
        config=document.simulation,
        template=document.ion,
        particle_backend=document.execution.backend,
        data_logger=data_logger,
        engine_policy=_build_engine_policy(
            document.execution,
            document.output,
        ),
    )
    _configure_terminal_recorder(
        simulation,
        document.output,
        report_hint,
    )
    if simulation.backend_message:
        print(simulation.backend_message)
    result = _run_simulation(simulation, document.execution)
    _print_result(result)
    _write_report(document, simulation, result, parsed.argv, report_hint)
    return result
