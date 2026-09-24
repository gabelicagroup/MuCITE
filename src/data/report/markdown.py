"""Markdown rendering for the stable simulation report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import schema as _schema
from .snapshot import ReportSnapshot


def _format_float(value: float, digits: int = 6) -> str:
    value = float(value)
    return "nan" if not np.isfinite(value) else f"{value:.{digits}g}"

def _continuous_source_overview(source: dict[str, Any]) -> list[str]:
    return [
        f"- requested ion current: `{_format_float(source.get('ion_current_a', float('nan')))} A`",
        f"- emission rate: `{_format_float(source.get('ion_emission_rate_per_s', float('nan')))} ions/s`",
        f"- represented boundary current: `{_format_float(source.get('represented_boundary_current_a', float('nan')))} A`",
        f"- represented emitted current: `{_format_float(source.get('represented_emitted_current_a', float('nan')))} A`",
        f"- source Mach number: `{_format_float(source.get('source_mach_number', float('nan')))}`",
        f"- source gas temperature: `{_format_float(source.get('source_temperature_k', float('nan')))} K`",
        f"- source velocity from gas field: `{source.get('source_velocity_from_gas_field', False)}`",
        f"- source radial velocity scale: `{_format_float(source.get('source_radial_velocity_scale', 1.0))}`",
        f"- source velocity delta: `{_format_float(source.get('source_velocity_delta_m_per_s', 0.0))} m/s`",
        f"- capillary transport velocity: `{_format_float(source.get('source_axial_velocity_m_per_s', float('nan')))} m/s`",
        f"- external exit speed: `{_format_float(source.get('source_exit_speed_m_per_s', float('nan')))} m/s`",
        f"- external exit KE: `{_format_float(source.get('source_exit_kinetic_energy_ev', float('nan')))} eV`",
        f"- capillary exit z: `{_format_float(source.get('capillary_exit_z_m', float('nan')) * 1.0e3)} mm`",
        f"- capillary inlet z: `{_format_float(source.get('capillary_inlet_z_m', float('nan')) * 1.0e3)} mm`",
        f"- capillary buffer length: `{_format_float(source.get('capillary_prefill_length_m', 0.0) * 1.0e3)} mm`",
        f"- capillary voltage: `{_format_float(source.get('capillary_voltage_v', float('nan')))} V`",
        f"- capillary ion number density: `{_format_float(source.get('capillary_number_density_m3', float('nan')))} 1/m^3`",
        f"- target real ions per weighted ion pack: `{_format_float(source.get('source_target_macro_weight', float('nan')))}`",
        f"- pending source accumulator: `{_format_float(source.get('source_pending_real_ion_accumulator', 0.0))} real ions`",
    ]

def _continuous_source_accounting(source: dict[str, Any]) -> list[str]:
    current = source.get("terminal_current_a_by_status", {})
    parent_current = source.get("terminal_parent_current_a_by_status", {})
    fragment_current = source.get("terminal_fragment_current_a_by_status", {})
    real = source.get("terminal_real_ions_by_status", {})
    parent_real = source.get("terminal_parent_real_ions_by_status", {})
    fragment_real = source.get("terminal_fragment_real_ions_by_status", {})
    return [
        f"- prefill real ions represented: `{_format_float(source.get('prefill_real_ions', 0.0))}`",
        f"- prefill weighted ion packs: `{source.get('prefill_macro_particles', 0)}`",
        f"- boundary-injected real ions represented: `{_format_float(source.get('boundary_injected_real_ions', 0.0))}`",
        f"- boundary-injected weighted ion packs: `{source.get('boundary_injected_macro_particles', 0)}`",
        f"- emitted real ions represented: `{_format_float(source.get('emitted_real_ions', 0.0))}`",
        f"- emitted weighted ion packs: `{source.get('emitted_macro_particles', 0)}`",
        f"- total real ions represented: `{_format_float(source.get('injected_real_ions', 0.0))}`",
        f"- total weighted ion packs injected/filled: `{source.get('injected_macro_particles', 0)}`",
        f"- active real ions represented: `{_format_float(source.get('active_real_ions', 0.0))}`",
        f"- active parent real ions remaining: `{_format_float(source.get('active_parent_real_ions', 0.0))}`",
        f"- active charged fragments represented: `{_format_float(source.get('active_fragment_real_ions', 0.0))}`",
        f"- capillary-buffer real ions represented: `{_format_float(source.get('capillary_buffer_real_ions', 0.0))}`",
        f"- blocked real ions due to full weighted ion pack pool: `{_format_float(source.get('blocked_real_ions', 0.0))}`",
        f"- z_exit current: `{_format_float(current.get('z_exit', 0.0))} A`",
        f"- z_exit parent current: `{_format_float(parent_current.get('z_exit', 0.0))} A`",
        f"- z_exit charged-fragment current: `{_format_float(fragment_current.get('z_exit', 0.0))} A`",
        f"- electrode loss current: `{_format_float(current.get('electrode_hit', 0.0))} A`",
        f"- current transmission: `{_format_float(source.get('current_transmission', float('nan')))}`",
        f"- eta_exit: `{_format_float(source.get('eta_exit', float('nan')))}`",
        f"- I_exit_cycle: `{_format_float(source.get('i_exit_cycle_a', float('nan')))} A`",
        f"- z_exit real ions represented: `{_format_float(real.get('z_exit', 0.0))}`",
        f"- z_exit parent real ions represented: `{_format_float(parent_real.get('z_exit', 0.0))}`",
        f"- z_exit charged fragments represented: `{_format_float(fragment_real.get('z_exit', 0.0))}`",
        f"- electrode-hit real ions represented: `{_format_float(real.get('electrode_hit', 0.0))}`",
    ]

def _append_continuous_source_section(
    lines: list[str],
    source: dict[str, Any],
) -> None:
    lines.extend(
        ["## Continuous Current Source", "", *_continuous_source_overview(source),
         *_continuous_source_accounting(source), ""]
    )

def _append_terminal_status(
    lines: list[str],
    status_name: str,
    stats: dict[str, Any],
) -> None:
    lines.extend(
        [
            "",
            f"### {status_name}",
            "",
            f"- weighted ion pack events: `{stats.get('macro_event_count', 0)}`",
            f"- represented real ions: `{_format_float(stats.get('represented_real_ions', float('nan')))} `",
            f"- parent real ions remaining: `{_format_float(stats.get('parent_real_ions_remaining', float('nan')))} `",
            f"- charged fragments represented: `{_format_float(stats.get('fragmented_real_ions_represented', float('nan')))} `",
            f"- weighted TOF p50/p95: `{_format_float(stats.get('tof_us_p50_weighted', float('nan')))} / {_format_float(stats.get('tof_us_p95_weighted', float('nan')))} us`",
            f"- weighted KE p50/p95: `{_format_float(stats.get('ke_ev_p50_weighted', float('nan')))} / {_format_float(stats.get('ke_ev_p95_weighted', float('nan')))} eV`",
            f"- weighted z p50/p95: `{_format_float(stats.get('z_mm_p50_weighted', float('nan')))} / {_format_float(stats.get('z_mm_p95_weighted', float('nan')))} mm`",
            f"- weighted r p50/p95: `{_format_float(stats.get('r_mm_p50_weighted', float('nan')))} / {_format_float(stats.get('r_mm_p95_weighted', float('nan')))} mm`",
        ]
    )

def _append_terminal_event_section(
    lines: list[str],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines.extend(
        [
            "",
            "## Historical Terminal Events",
            "",
            f"- event rows: `{summary.get('rows', 0)}`",
            f"- retained report/plot sample rows: `{summary.get('sample_rows', 0)}`",
            f"- distribution statistics: `{summary.get('distribution_statistics', 'exact')}`",
            f"- event CSV: `{path}`",
            f"- recording mode: `{summary.get('recording_mode', '')}`",
            f"- stream-only memory mode: `{summary.get('stream_only', False)}`",
            f"- max rows reached: `{summary.get('max_rows_reached', False)}`",
            f"- omitted rows by status: `{summary.get('omitted_rows_by_status', {})}`",
            f"- omitted parent real ions by status: `{summary.get('omitted_parent_real_ions_by_status', {})}`",
            f"- omitted charged fragments by status: `{summary.get('omitted_fragment_real_ions_by_status', {})}`",
            f"- TOF reference: `{summary.get('tof_reference', '')}`",
            "- use this section for cumulative transport; `Final Particle Status` is only the slot end-state at stop time.",
        ]
    )
    for status_name, stats in summary.get("summary_by_status", {}).items():
        if int(stats.get("macro_event_count", 0)) > 0:
            _append_terminal_status(lines, status_name, stats)

def _purpose_and_command(summary: dict[str, Any]) -> list[str]:
    return [
        "# Coupled PIC/IICT simulation report",
        "",
        "## Purpose",
        "",
        "Validate the runtime coupling chain with baked electric field, optional space-charge coupling, and sampled gas state:",
        "",
        "- baked SIMION DC/RF electric field",
        "- PIC space-charge deposition and Poisson solve",
        "- Monte Carlo neutral collisions with a selectable IICT physics backend",
        "- baked gas pressure, temperature, and macroscopic velocity field when present",
        "",
        "## Command",
        "",
        "```powershell",
        " ".join(summary["command"]),
        "```",
        "",
    ]

def _source_configuration_lines(summary: dict[str, Any]) -> list[str]:
    config = summary["config"]
    template = summary["template"]
    capillary_z = config["capillary_exit_z_m"]
    if capillary_z is None:
        capillary_z = template["initial_position_m"][2]
    return [
        f"- ions: `{config['ion_count']}`",
        f"- real ions per weighted ion pack: `{_format_float(config['macro_particle_weight'])}`",
        f"- source mode: `{config['source_mode']}`",
        f"- collision operator: `{config['collision_operator']}`",
        f"- ion mass: `{_format_float(template['mass_amu'])} amu`",
        f"- charge state: `+{template['charge_state']}`",
        f"- collision cross section: `{_format_float(template['collision_cross_section_m2'])} m^2` (`{_format_float(template['collision_cross_section_nm2'])} nm^2`)",
        f"- initial kinetic energy: `{_format_float(template['initial_kinetic_energy_ev'])} eV`",
        f"- initial speed: `{_format_float(template['initial_speed_m_per_s'])} m/s`",
        f"- source radius: `{_format_float((config['source_radius_m'] or 0.0) * 1.0e3)} mm`",
        f"- cone half angle: `{_format_float(np.degrees(config['cone_half_angle_rad']))} deg`",
        f"- capillary exit z: `{_format_float(capillary_z * 1.0e3)} mm`",
        f"- source velocity from gas field: `{config.get('source_velocity_from_gas_field', False)}`",
        f"- source radial velocity scale: `{_format_float(config.get('source_radial_velocity_scale', 1.0))}`",
        f"- source velocity delta: `{_format_float(config.get('source_velocity_delta_m_per_s', 0.0))} m/s`",
        f"- capillary voltage: `{_format_float(config['capillary_voltage_v'])} V`",
        f"- RF peak voltage: `{_format_float(config['rf_peak_voltage_v'])} V`",
        f"- RF frequency: `{_format_float(config['rf_frequency_hz'])} Hz`",
        f"- RF phase: `{_format_float(np.degrees(config['rf_phase_rad']))} deg`",
        f"- total time: `{_format_float(config['total_time_s'])} s`",
        f"- macro step: `{_format_float(config['macro_time_step_s'])} s`",
    ]

def _solver_configuration_lines(config: dict[str, Any]) -> list[str]:
    return [
        f"- PIC Poisson backend: `{config.get('pic_poisson_backend', 'sor')}`",
        f"- PIC Poisson preconditioner: `{config.get('pic_poisson_preconditioner', 'none')}`",
        f"- PIC Poisson warm start: `{config.get('pic_poisson_warm_start', True)}`",
        f"- PIC AMG mode: `{config.get('pic_amg_mode', 'solve')}`",
        f"- PIC AMG solver: `{config.get('pic_amg_solver', 'ruge_stuben')}`",
        f"- PIC AMG tolerance/max iters: `{_format_float(config.get('pic_amg_tolerance', 1.0e-3))}` / `{config.get('pic_amg_max_iters', 50)}`",
        f"- PIC AMG fallback backend: `{config.get('pic_amg_fallback_backend', 'sparse_bicgstab')}`",
        f"- SOR max iterations: `{config['sor_max_iters']}`",
        f"- detector z: `{_format_float((config['detector_z_m'] or config['domain_length_m']) * 1.0e3)} mm`",
        f"- detector radius: `{_format_float((config['detector_radius_m'] or config['radial_limit_m'] or config['domain_radius_m']) * 1.0e3)} mm`",
        f"- max wall time: `{_format_float(config['max_wall_time_s'])} s`",
        f"- terminal event mode: `{config['terminal_event_mode']}`",
        f"- max terminal event rows: `{config['max_terminal_event_rows']}`",
        f"- collision batch size: `{config['collision_batch_size']}`",
        f"- collision physics backend: `{config.get('collision_physics_backend', 'ionspa')}`",
        f"- optional IonSPA provider: `{config.get('ionspa_backend', 'bundled')}`",
        f"- collision model: `{config['collision_model']}`",
        f"- fragmentation mode: `{config['fragmentation_mode']}`",
        f"- Langevin z window: `{_format_float(config['langevin_z_start_m'] * 1.0e3)}..{_format_float(config['langevin_z_end_m'] * 1.0e3)} mm`",
        f"- Langevin switch probability: `{_format_float(config['langevin_switch_probability'])}`",
        f"- Langevin max dt: `{_format_float(config['langevin_max_dt_s'])} s`",
    ]


def _configuration_lines(summary: dict[str, Any]) -> list[str]:
    return [
        "## Configuration", "", *_source_configuration_lines(summary),
        *_solver_configuration_lines(summary["config"]), "",
    ]


def _field_lines(summary: dict[str, Any]) -> list[str]:
    config = summary["config"]
    grid_runtime = summary["grid_runtime"]
    poisson = summary.get("poisson_solver", {})
    field_metadata = summary["field_metadata"]
    fluent = field_metadata.get("fluent", {}) if isinstance(field_metadata, dict) else {}
    mask = summary["electrode_mask"]
    mask3d = summary.get("electrode_mask_3d", {})
    return [
        "## Field And Gas",
        "",
        f"- static field path: `{config['static_field_path']}`",
        f"- stage schedule path: `{config.get('stage_schedule_path')}`",
        f"- Cartesian 3D electric overlay path: `{config.get('static_field_3d_path')}`",
        f"- baked/static grid: `{grid_runtime['static_grid_nr']} x {grid_runtime['static_grid_nz']}`",
        f"- PIC grid: `{grid_runtime['pic_grid_nr']} x {grid_runtime['pic_grid_nz']}`",
        f"- PIC grid decoupled: `{grid_runtime['pic_grid_decoupled']}`",
        f"- PIC grid selection: `{grid_runtime.get('pic_grid_resolution_source', 'unknown')}`",
        f"- PIC space-charge scale: `{_format_float(grid_runtime.get('pic_space_charge_scale', float('nan')))}`",
        f"- PIC Poisson active unknowns: `{poisson.get('active_unknowns', 'n/a')}`",
        f"- PIC Poisson matrix nnz: `{poisson.get('matrix_nnz', 'n/a')}`",
        f"- gas pressure background: `{_format_float(float(fluent.get('background_pressure_pa', float('nan'))))} Pa`",
        f"- gas temperature background: `{_format_float(float(fluent.get('background_temperature_k', float('nan'))))} K`",
        f"- gas axial velocity background: `{_format_float(float(fluent.get('background_vz_m_per_s', float('nan'))))} m/s`",
        f"- gas radial velocity background: `{_format_float(float(fluent.get('background_vr_m_per_s', float('nan'))))} m/s`",
        f"- electrode mask loaded: `{bool(mask.get('loaded', False))}`",
        f"- 3D electrode mask path: `{config.get('electrode_mask_3d_path')}`",
        f"- Cartesian 3D electrode mask loaded: `{bool(mask3d.get('loaded', False))}`",
        "",
    ]


def _configuration_difference_lines(summary: dict[str, Any]) -> list[str]:
    differences = summary["config_differences"]
    lines = ["## Requested Versus Effective Configuration", ""]
    if not differences:
        return [*lines, "- No runtime configuration adjustments.", ""]
    lines.extend(
        [
            "The runtime adjusted the following requested values after loading schedules or field metadata:",
            "",
        ]
    )
    for key, values in differences.items():
        lines.append(
            f"- `{key}`: requested `{values['requested']}`; effective `{values['effective']}`"
        )
    lines.append("")
    return lines


def _basic_result_lines(summary: dict[str, Any], result: Any) -> list[str]:
    result_summary = summary["result"]
    physics = summary.get("collision_physics", summary["ionspa"])
    collision = result_summary.get("collision_statistics", {})
    fields = result_summary["runtime_field_statistics"]
    return [
        "## Result",
        "",
        f"- backend: `{result.particle_backend}`",
        f"- backend requested: `{result_summary['particle_backend_requested']}`",
        f"- backend effective arch: `{result_summary['particle_backend_effective']}`",
        f"- collision physics backend: `{physics.get('collision_physics_backend', physics.get('effective_backend', 'unknown'))}`",
        f"- collision implementation: `{physics.get('implementation')}`",
        f"- optional IonSPA loaded: `{physics.get('ionspa_loaded', physics.get('loaded', False))}`",
        f"- collision parameter config: `{physics.get('parameter_config_path')}`",
        f"- collision parameter sources: `{_schema._json_safe(physics.get('parameter_sources', {}))}`",
        f"- termination reason: `{result.termination_reason}`",
        f"- survivors: `{len(result.survivors)}`",
        f"- collisions: `{result.collision_count}`",
        f"- collision_real_ions_represented: `{_format_float(collision.get('collision_real_ions_represented', float('nan')))} `",
        f"- fragmented parent real ions represented: `{_format_float(collision.get('fragmented_real_ions_represented', float('nan')))} `",
        f"- hybrid Langevin weighted ion pack events: `{_format_float(collision.get('hybrid_langevin_macro_events', 0.0))}`",
        f"- hybrid_langevin_real_ions_represented: `{_format_float(collision.get('hybrid_langevin_real_ions_represented', 0.0))}`",
        f"- hybrid_langevin_estimated_binary_collisions: `{_format_float(collision.get('hybrid_langevin_estimated_binary_collisions', 0.0))}`",
        f"- macro steps: `{result_summary['macro_step_count']}`",
        f"- retained macro-history sample rows: `{len(result.macro_history)}`",
        f"- micro steps: `{result.micro_step_count}`",
        f"- max |E_sce_r|: `{_format_float(fields.get('E_sce_r', {}).get('max_abs', float('nan')))} V/m`",
        f"- max |E_sce_z|: `{_format_float(fields.get('E_sce_z', {}).get('max_abs', float('nan')))} V/m`",
    ]


def _append_failure_and_stages(
    lines: list[str],
    summary: dict[str, Any],
    result: Any,
) -> None:
    if result.numerical_failure:
        lines.extend(
            [
                "",
                "## Numerical Failure",
                "",
                f"- details: `{_schema._json_safe(result.numerical_failure)}`",
            ]
        )
    stages = summary["result"].get("stage_history", [])
    if stages:
        lines.extend(["", "## Stage Schedule", ""])
    for stage in stages:
        lines.append(
            f"- `{stage.get('name')}`: "
            f"{_format_float(stage.get('start_s', float('nan')))}.."
            f"{_format_float(stage.get('end_s', float('nan')))} s, "
            f"source_enabled=`{stage.get('source_enabled')}`, "
            f"field=`{stage.get('static_field_path')}`"
        )


def _append_final_macro(
    lines: list[str],
    summary: dict[str, Any],
) -> None:
    final = summary["result"].get("final_macro", {})
    if not final:
        return
    config = summary["config"]
    lines.extend(
        [
            f"- final time: `{_format_float(final.get('time_s', float('nan')))} s`",
            f"- final mean z: `{_format_float(final.get('mean_axial_position_m', float('nan')))} m`",
            f"- final mean internal temperature: `{_format_float(final.get('mean_internal_temperature_k', float('nan')))} K`",
            f"- final PIC Poisson backend: `{final.get('poisson_backend', config.get('pic_poisson_backend', 'sor'))}`",
            f"- final PIC Poisson iterations: `{_format_float(final.get('poisson_iterations', final.get('sor_iters', float('nan'))))}`",
            f"- final PIC Poisson relative residual: `{_format_float(final.get('poisson_relative_residual', float('nan')))}`",
            f"- final PIC Poisson runtime: `{_format_float(final.get('poisson_runtime_ms', float('nan')))} ms`",
            f"- final p95 |E_sce|/|E_external|: `{_format_float(final.get('sce_external_ratio_p95', float('nan')))} `",
        ]
    )


def _final_status_lines(summary: dict[str, Any]) -> list[str]:
    counts = summary["result"]["final_status_counts"]
    return [
        "",
        "## Final Particle Status",
        "",
        f"- active: `{counts.get('active', 0)}`",
        f"- capillary_buffer: `{counts.get('capillary_buffer', 0)}`",
        f"- z_exit: `{counts.get('z_exit', 0)}`",
        f"- radial_out: `{counts.get('radial_out', 0)}`",
        f"- domain_out: `{counts.get('domain_out', 0)}`",
        f"- electrode_hit: `{counts.get('electrode_hit', 0)}`",
        f"- fragmented: `{counts.get('fragmented', 0)}`",
        "- electrode_hit is populated when `--electrode-mask` or `--electrode-mask-3d` is enabled.",
    ]


def _append_survivor_lines(lines: list[str], summary: dict[str, Any]) -> None:
    survivor = summary["result"]["survivor_statistics"]
    if survivor.get("count", 0):
        lines.extend(
            [
                f"- survivor z p50/p95: `{_format_float(survivor['z_m']['p50'])}` / `{_format_float(survivor['z_m']['p95'])} m`",
                f"- survivor KE p50/p95: `{_format_float(survivor['kinetic_energy_ev']['p50'])}` / `{_format_float(survivor['kinetic_energy_ev']['p95'])} eV`",
                f"- survivor temperature p50/p95: `{_format_float(survivor['internal_temperature_k']['p50'])}` / `{_format_float(survivor['internal_temperature_k']['p95'])} K`",
            ]
        )
    transport = summary["result"].get("final_particle_transport", {})
    if int(transport.get("active_count", 0)) <= 0:
        return
    particle_vz = transport.get("particle_vz_m_per_s", {})
    gas_vz = transport.get("local_gas_vz_m_per_s", {})
    relative = transport.get("relative_speed_m_per_s", {})
    lines.extend(
        [
            f"- survivor axial vz p50/p95: `{_format_float(particle_vz.get('p50', float('nan')))} / {_format_float(particle_vz.get('p95', float('nan')))} m/s`",
            f"- local gas axial vz at survivors p50/p95: `{_format_float(gas_vz.get('p50', float('nan')))} / {_format_float(gas_vz.get('p95', float('nan')))} m/s`",
            f"- survivor-gas relative speed p50/p95: `{_format_float(relative.get('p50', float('nan')))} / {_format_float(relative.get('p95', float('nan')))} m/s`",
        ]
    )


def _dt_performance_lines(summary: dict[str, Any], result: Any) -> list[str]:
    performance = summary["performance"]
    dt = performance.get("dt_statistics", {})
    limits = performance.get("dt_limiter_counts", {})
    raw = performance.get("dt_raw_limiter_counts", {})
    return [
        f"- run started: `{performance['run_started_at']}`",
        f"- run ended: `{performance['run_ended_at']}`",
        f"- wall time: `{_format_float(performance['run_wall_time_s'])} s`",
        f"- simulated time: `{_format_float(performance['simulated_us'])} us`",
        f"- simulated us / wall s: `{_format_float(performance['simulated_us_per_wall_s'])}`",
        f"- wall s / simulated us: `{_format_float(performance['wall_s_per_simulated_us'])}`",
        f"- micro_step_count: `{dt.get('micro_step_count', result.micro_step_count)}`",
        f"- mean_dt: `{_format_float(dt.get('mean_dt_s', float('nan')))} s`",
        f"- p50_dt: `{_format_float(dt.get('p50_dt_s', float('nan')))} s`",
        f"- p95_dt: `{_format_float(dt.get('p95_dt_s', float('nan')))} s`",
        f"- min_dt: `{_format_float(dt.get('min_dt_s', float('nan')))} s`",
        f"- max_dt: `{_format_float(dt.get('max_dt_s', float('nan')))} s`",
        f"- dt quantile sample count/capacity: `{dt.get('quantile_sample_count', 'n/a')} / {dt.get('quantile_reservoir_size', 'n/a')}`",
        f"- dt quantiles approximate: `{dt.get('quantiles_approximate', False)}`",
        f"- dt_limit_max_dt: `{limits.get('max_dt', 0)}`",
        f"- dt_limit_macro_end: `{limits.get('macro_end', 0)}`",
        f"- dt_limit_collision: `{limits.get('collision', 0)}`",
        f"- dt_limit_rf: `{limits.get('rf', 0)}`",
        f"- dt_limit_acceleration: `{limits.get('acceleration', 0)}`",
        f"- dt_limit_advection: `{limits.get('advection', 0)}`",
        f"- dt_limit_min_dt_floor: `{limits.get('min_dt_floor', 0)}`",
        f"- dt_limit_langevin_max: `{limits.get('langevin_max', 0)}`",
        f"- dt_raw_limit_collision: `{raw.get('collision', 0)}`",
        f"- dt_raw_limit_rf: `{raw.get('rf', 0)}`",
        f"- dt_raw_limit_acceleration: `{raw.get('acceleration', 0)}`",
        f"- dt_raw_limit_advection: `{raw.get('advection', 0)}`",
        f"- dt_raw_limit_langevin_max: `{raw.get('langevin_max', 0)}`",
    ]


def _timing_performance_lines(summary: dict[str, Any]) -> list[str]:
    timing = summary["performance"].get("breakdown", {})
    return [
        f"- time_pic_poisson: `{_format_float(timing.get('time_pic_poisson', timing.get('time_pic_sor', float('nan'))))} s`",
        f"- time_download_state: `{_format_float(timing.get('time_download_state', float('nan')))} s`",
        f"- time_sample_fields: `{_format_float(timing.get('time_sample_fields', float('nan')))} s`",
        f"- time_dt_estimate: `{_format_float(timing.get('time_dt_estimate', float('nan')))} s`",
        f"- time_collision_loop: `{_format_float(timing.get('time_collision_loop', float('nan')))} s`",
        f"- time_upload_state: `{_format_float(timing.get('time_upload_state', float('nan')))} s`",
        f"- time_rk4_push: `{_format_float(timing.get('time_rk4_push', float('nan')))} s`",
        f"- time_terminal_events: `{_format_float(timing.get('time_terminal_events', float('nan')))} s`",
        f"- time_source_injection: `{_format_float(timing.get('time_source_injection', float('nan')))} s`",
        f"- time_unaccounted: `{_format_float(timing.get('time_unaccounted', float('nan')))} s`",
    ]


def _performance_lines(summary: dict[str, Any], result: Any) -> list[str]:
    return [
        "", "## Performance", "", *_dt_performance_lines(summary, result),
        *_timing_performance_lines(summary),
    ]


def _output_lines(
    summary: dict[str, Any],
    paths: dict[str, Path],
    result: Any,
) -> list[str]:
    lines = [
        "",
        "## Output Files",
        "",
        f"- summary JSON: `{paths['summary']}`",
        f"- report: `{paths['report']}`",
        f"- final particles CSV: `{paths['final_particles']}`",
        f"- terminal events CSV: `{paths['terminal_events']}`",
    ]
    for name, path in summary["result"].get("diagnostic_plots", {}).items():
        lines.append(f"- {name} plot: `{path}`")
    if result.export_directory:
        lines.append(f"- snapshots: `{result.export_directory}`")
    lines.append("")
    return lines


def build_markdown(
    summary: dict[str, Any],
    snapshot: ReportSnapshot,
    paths: dict[str, Path],
) -> str:
    """Render the established human-readable report."""

    result = snapshot.result
    config = summary["config"]
    lines = _purpose_and_command(summary)
    lines.extend(_configuration_lines(summary))
    lines.extend(_field_lines(summary))
    lines.extend(_configuration_difference_lines(summary))
    if config["source_mode"] == "continuous-current":
        _append_continuous_source_section(lines, summary["source_runtime"])
    lines.extend(_basic_result_lines(summary, result))
    _append_failure_and_stages(lines, summary, result)
    _append_final_macro(lines, summary)
    lines.extend(_final_status_lines(summary))
    _append_survivor_lines(lines, summary)
    if config["source_mode"] == "continuous-current":
        _append_terminal_event_section(
            lines,
            summary["result"]["terminal_event_diagnostics"],
            paths["terminal_events"],
        )
    lines.extend(_performance_lines(summary, result))
    lines.extend(_output_lines(summary, paths, result))
    return "\n".join(lines)
