"""Contracts for Phase 3 externalized runtime tuning."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import (
    ExecutionConfig,
    OutputConfig,
    SimulationConfig,
    config_document_from_dict,
    config_document_to_dict,
    make_config_document,
    make_execution_config,
    make_output_config,
    make_simulation_config,
)


SIMULATION_TUNING_DEFAULTS = {
    "collision_safety_factor": 0.1,
    "rf_safety_factor": 0.05,
    "acceleration_safety_factor": 0.2,
    "spatial_tolerance_m": 5.0e-5,
    "min_time_step_s": 1.0e-12,
    "max_time_step_s": 5.0e-8,
    "advection_cfl_fraction": 0.5,
    "gas_species": "n2",
    "gas_gamma": 1.4,
    "gas_molar_mass_kg_per_mol": 0.0280134,
    "electrode_sweep_spacing_fraction": 0.25,
    "electrode_hit_bisection_iterations": 24,
    "source_reference_velocity_sample_count": 101,
    "source_gaussian_max_attempts": 1_000,
    "source_gaussian_min_batch": 1_024,
    "source_gaussian_oversample_factor": 2,
}

OUTPUT_TUNING_DEFAULTS = {
    "dt_quantile_reservoir_size": 8_192,
    "macro_history_max_rows": 4_096,
    "terminal_event_sample_rows": 50_000,
    "terminal_event_sample_seed": 0,
    "report_plot_max_points": 200_000,
    "trajectory_plot_max_tracks": 50,
}


def test_externalized_tuning_defaults_preserve_previous_literals() -> None:
    simulation = SimulationConfig()
    output = OutputConfig()

    for field_name, expected in SIMULATION_TUNING_DEFAULTS.items():
        assert getattr(simulation, field_name) == expected
    assert ExecutionConfig().progress_interval_s == pytest.approx(0.25)
    for field_name, expected in OUTPUT_TUNING_DEFAULTS.items():
        assert getattr(output, field_name) == expected


def test_early_v1_document_upgrades_every_externalized_tuning_field() -> None:
    document = config_document_from_dict(
        {"schema_version": 1, "simulation": {}, "ion": {}}
    )
    encoded = config_document_to_dict(document)

    for field_name, expected in SIMULATION_TUNING_DEFAULTS.items():
        assert encoded["simulation"][field_name] == expected
    assert encoded["execution"]["progress_interval_s"] == pytest.approx(0.25)
    for field_name, expected in OUTPUT_TUNING_DEFAULTS.items():
        assert encoded["output"][field_name] == expected


def test_custom_tuning_values_survive_canonical_json_round_trip() -> None:
    simulation = replace(
        SimulationConfig(),
        collision_safety_factor=0.08,
        min_time_step_s=2.0e-12,
        max_time_step_s=4.0e-8,
        source_gaussian_max_attempts=500,
        electrode_hit_bisection_iterations=12,
    )
    execution = ExecutionConfig(progress_interval_s=0.5)
    output = replace(
        OutputConfig(),
        dt_quantile_reservoir_size=256,
        terminal_event_sample_seed=19,
        trajectory_plot_max_tracks=0,
    )
    document = make_config_document(
        simulation=simulation,
        execution=execution,
        output=output,
    )

    restored = config_document_from_dict(config_document_to_dict(document))

    assert restored.simulation == simulation
    assert restored.execution == execution
    assert restored.output == output


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("collision_safety_factor", 0.0),
        ("rf_safety_factor", float("nan")),
        ("acceleration_safety_factor", 1.01),
        ("spatial_tolerance_m", "5e-5"),
        ("min_time_step_s", 0.0),
        ("max_time_step_s", float("inf")),
        ("advection_cfl_fraction", 1.01),
        ("gas_species", "argon"),
        ("gas_gamma", 1.0),
        ("gas_molar_mass_kg_per_mol", 0.0),
        ("electrode_sweep_spacing_fraction", 0.0),
        ("electrode_hit_bisection_iterations", -1),
        ("source_reference_velocity_sample_count", 0),
        ("source_gaussian_max_attempts", True),
        ("source_gaussian_min_batch", 1.5),
        ("source_gaussian_oversample_factor", 0),
    ],
)
def test_invalid_simulation_tuning_fails_closed(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        make_simulation_config(**{field_name: value})


def test_inverted_timestep_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="min_time_step_s"):
        make_simulation_config(
            min_time_step_s=2.0e-8,
            max_time_step_s=1.0e-8,
        )


@pytest.mark.parametrize("value", [0.0, float("inf"), "0.25", True])
def test_invalid_progress_interval_fails_closed(value: object) -> None:
    with pytest.raises(ValueError, match="progress_interval_s"):
        make_execution_config(progress_interval_s=value)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("dt_quantile_reservoir_size", -1),
        ("macro_history_max_rows", 1.5),
        ("terminal_event_sample_rows", True),
        ("terminal_event_sample_seed", 2.5),
        ("report_plot_max_points", 0),
        ("trajectory_plot_max_tracks", -1),
    ],
)
def test_invalid_output_tuning_fails_closed(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        make_output_config(**{field_name: value})
