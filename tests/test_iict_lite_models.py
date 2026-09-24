"""Independent IICT parameter, thermodynamic, and API contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import atomic_mass

from src.agents.ion import IonState
from src.config import IonTemplate, PROJECT_ROOT, SimulationConfig
from src.env.collisions.factory import build_collision_physics_backend
from src.env.collisions.iict_lite import (
    ClassicalHeatCapacityModel,
    ConstantCvHeatCapacityModel,
    EyringFragmentationModel,
    NoFragmentationModel,
    TabulatedPseudoAtomMassModel,
)
from src.env.collisions.iict_lite.heat_capacity import (
    tabulated_cv_model,
    tabulated_energy_model,
)
from src.env.gas.layered import GasState
from src.render.cli.config_adapter import resolve_cli_document
from src.render.cli.parser import parse_cli


def _lite_config(**overrides: object) -> SimulationConfig:
    values = {
        "collision_physics_backend": "iict-lite",
        "iict_heat_capacity_model": "classical",
        "iict_num_atoms": 72,
        "iict_pseudoatom_model": "constant",
        "iict_pseudoatom_mass_da": 30.0,
        "iict_fragmentation_model": "none",
    }
    values.update(overrides)
    return SimulationConfig(**values)


def _template() -> IonTemplate:
    return IonTemplate(mass_kg=5000.0 * atomic_mass, num_atoms=72)


def _gas(velocity: np.ndarray, temperature_k: float = 300.0) -> GasState:
    number_density = 2.0e20
    return GasState(
        pressure_pa=number_density * 1.380649e-23 * temperature_k,
        temperature_k=temperature_k,
        velocity_m_per_s=np.asarray(velocity, dtype=np.float64),
        number_density_m3=number_density,
        mass_density_kg_per_m3=number_density * 28.0 * atomic_mass,
        mach_number=0.0,
    )


def _ion(
    velocity: np.ndarray,
    temperature_k: float,
    mass_kg: float,
) -> IonState:
    return IonState(
        position_m=np.zeros(3),
        velocity_m_per_s=np.asarray(velocity, dtype=np.float64),
        mass_kg=mass_kg,
        charge_c=1.602176634e-19,
        collision_cross_section_m2=1.0e-18,
        internal_temperature_k=temperature_k,
    )


def test_iict_lite_factory_neither_imports_nor_reads_ionspa() -> None:
    code = r"""
import json
import pathlib
import sys
accessed = []
def audit(event, args):
    if event == "open" and args:
        text = str(args[0]).replace("\\", "/").lower()
        if "/collisions/ionspa/" in text or "hcprofiles2.json" in text:
            accessed.append(text)
sys.addaudithook(audit)
from src.config import IonTemplate, SimulationConfig
from src.env.collisions.factory import build_collision_physics_backend
config = SimulationConfig(
    collision_physics_backend="iict-lite",
    iict_heat_capacity_model="classical",
    iict_num_atoms=72,
    iict_pseudoatom_model="constant",
    iict_pseudoatom_mass_da=30.0,
    iict_fragmentation_model="none",
)
backend = build_collision_physics_backend(config=config, template=IonTemplate())
modules = [
    name for name in sys.modules
    if name == "ionspa" or name.startswith("src.env.collisions.ionspa")
]
print(json.dumps({"modules": modules, "accessed": accessed,
                  "runtime": backend.runtime_info}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["modules"] == []
    assert payload["accessed"] == []
    assert payload["runtime"]["ionspa_loaded"] is False
    assert payload["runtime"]["ionspa_accessed"] is False


def test_scalar_and_batch_use_identical_seeded_physical_core() -> None:
    template = _template()
    backend = build_collision_physics_backend(
        config=_lite_config(),
        template=template,
    )
    velocities = np.array([[100.0, -20.0, 30.0], [-40.0, 50.0, 60.0]])
    masses = np.array([template.mass_kg, 1.1 * template.mass_kg])
    temperatures = np.array([310.0, 450.0])
    gas_velocities = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    gases = [_gas(gas_velocities[0], 295.0), _gas(gas_velocities[1], 305.0)]
    scalar_rng = np.random.default_rng(1701)
    scalar = [
        backend.apply_collision(
            _ion(velocities[index], temperatures[index], masses[index]),
            template,
            gases[index],
            gas_name="n2",
            dt_s=2.0e-9,
            rng=scalar_rng,
        )
        for index in range(2)
    ]
    batch_rng = np.random.default_rng(1701)
    batch = backend.apply_collision_batch(
        velocities_m_per_s=velocities,
        masses_kg=masses,
        internal_temperatures_k=temperatures,
        gas_velocities_m_per_s=gas_velocities,
        gas_temperatures_k=np.array([295.0, 305.0]),
        gas_number_density_m3=np.array([gas.number_density_m3 for gas in gases]),
        gas_mass_density_kg_per_m3=np.array(
            [gas.mass_density_kg_per_m3 for gas in gases]
        ),
        template=template,
        gas_name="n2",
        dt_s=2.0e-9,
        rng=batch_rng,
    )
    np.testing.assert_array_equal(
        batch[0],
        np.stack([outcome.velocity_m_per_s for outcome in scalar]),
    )
    np.testing.assert_array_equal(
        batch[1],
        [outcome.internal_temperature_k for outcome in scalar],
    )
    np.testing.assert_array_equal(
        batch[2],
        [outcome.fragmentation_probability for outcome in scalar],
    )
    np.testing.assert_array_equal(batch_rng.random(8), scalar_rng.random(8))


@pytest.mark.parametrize(
    "model",
    [
        ClassicalHeatCapacityModel(72),
        ConstantCvHeatCapacityModel(3.5e-21),
        tabulated_energy_model(
            np.array([100.0, 300.0, 800.0]),
            np.array([1.0e-19, 4.0e-19, 1.4e-18]),
            source_path="energy.csv",
        ),
        tabulated_cv_model(
            np.array([100.0, 300.0, 800.0]),
            np.array([1.0e-21, 2.0e-21, 3.0e-21]),
            source_path="cv.csv",
        ),
    ],
)
def test_energy_temperature_round_trip(model: object) -> None:
    temperatures = np.array([0.0, 50.0, 250.0, 600.0, 1000.0])
    energy = model.energy_from_temperature(temperatures)
    recovered = model.temperature_from_energy(energy)
    np.testing.assert_allclose(recovered, temperatures, rtol=2.0e-15, atol=1.0e-12)


def test_tabulated_pseudoatom_interpolation_and_bounds() -> None:
    temperatures = np.array([200.0, 200.0, 400.0, 400.0])
    speeds = np.array([0.0, 1000.0, 0.0, 1000.0])
    masses = np.array([20.0, 30.0, 40.0, 50.0])
    model = TabulatedPseudoAtomMassModel(
        temperatures,
        speeds,
        masses,
        min_mass_da=10.0,
        max_mass_da=60.0,
        source_path="pseudo.csv",
        gas_name="n2",
    )
    actual_da = model.mass_kg(300.0, np.array([500.0, 2000.0]), "n2")
    np.testing.assert_allclose(actual_da / atomic_mass, [35.0, 40.0])
    with pytest.raises(ValueError, match="not 'ar'"):
        model.mass_kg(300.0, 500.0, "ar")
    with pytest.raises(ValueError, match="duplicate"):
        TabulatedPseudoAtomMassModel(
            np.array([200.0, 200.0, 400.0, 400.0]),
            np.array([0.0, 0.0, 1000.0, 1000.0]),
            masses,
            min_mass_da=1.0,
            max_mass_da=60.0,
            source_path="bad.csv",
        )


def test_eyring_probability_boundaries_and_disabled_fragmentation() -> None:
    model = EyringFragmentationModel(80.0, -120.0)
    rate = model.rate_s_inverse(600.0)
    assert rate > 0.0
    assert model.probability(600.0, 0.0) == 0.0
    assert model.probability(600.0, 1.0e100) == 1.0
    expected = -np.expm1(-rate * 2.0e-6)
    assert model.probability(600.0, 2.0e-6) == pytest.approx(expected)
    disabled = NoFragmentationModel()
    np.testing.assert_array_equal(
        disabled.probability(np.array([200.0, 2000.0]), 1.0e9),
        [0.0, 0.0],
    )


def test_strict_parameter_json_and_cli_source_metadata(tmp_path: Path) -> None:
    parameter_path = tmp_path / "iict.json"
    parameter_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "heat_capacity": {"type": "classical", "num_atoms": 72},
                "pseudoatom_mass": {"type": "constant", "mass_da": 30.0},
                "fragmentation": {"type": "none"},
            }
        ),
        encoding="utf-8",
    )
    config = _lite_config(
        iict_parameter_config_path=parameter_path,
        iict_pseudoatom_mass_da=31.0,
        iict_cli_override_fields=("iict_pseudoatom_mass_da",),
    )
    info = build_collision_physics_backend(
        config=config,
        template=_template(),
    ).runtime_info
    assert info["effective_parameters"]["pseudoatom_mass"]["mass_da"] == 31.0
    assert info["parameter_sources"]["pseudoatom_mass.mass_da"] == "cli"
    assert info["parameter_config_sha256"]
    bad = json.loads(parameter_path.read_text(encoding="utf-8"))
    bad["pseudoatom_mass"]["unknown"] = 1
    parameter_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown pseudoatom_mass"):
        build_collision_physics_backend(
            config=_lite_config(iict_parameter_config_path=parameter_path),
            template=_template(),
        )


def test_missing_required_iict_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="pseudoatom_mass.type is required"):
        build_collision_physics_backend(
            config=SimulationConfig(
                collision_physics_backend="iict-lite",
                iict_heat_capacity_model="classical",
                iict_num_atoms=72,
                iict_fragmentation_model="none",
            ),
            template=_template(),
        )
    with pytest.raises(ValueError, match="delta_h_kj_per_mol"):
        build_collision_physics_backend(
            config=_lite_config(iict_fragmentation_model="eyring"),
            template=_template(),
        )


def test_cli_overrides_parameter_json_and_records_source(tmp_path: Path) -> None:
    parameter_path = tmp_path / "iict.json"
    parameter_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "heat_capacity": {"type": "classical", "num_atoms": 72},
                "pseudoatom_mass": {"type": "constant", "mass_da": 30.0},
                "fragmentation": {"type": "none"},
            }
        ),
        encoding="utf-8",
    )
    parsed = parse_cli(
        [
            "--collision-physics-backend",
            "iict-lite",
            "--iict-parameter-config",
            str(parameter_path),
            "--iict-pseudoatom-mass-da",
            "32",
        ]
    )
    document = resolve_cli_document(parsed.namespace, parsed.explicit_dests)
    info = build_collision_physics_backend(
        config=document.simulation,
        template=document.ion,
    ).runtime_info
    assert document.simulation.collision_physics_backend == "iict-lite"
    assert info["effective_parameters"]["pseudoatom_mass"]["mass_da"] == 32.0
    assert info["parameter_sources"]["pseudoatom_mass.mass_da"] == "cli"
    assert "iict_pseudoatom_mass_da" in (
        document.simulation.iict_cli_override_fields
    )


def test_legacy_ionspa_cli_selector_warns_and_selects_compatibility() -> None:
    parsed = parse_cli(["--ionspa-backend", "approximate"])
    with pytest.warns(FutureWarning, match="deprecated"):
        document = resolve_cli_document(
            parsed.namespace,
            parsed.explicit_dests,
        )
    assert document.simulation.collision_physics_backend == "ionspa"
    assert document.simulation.ionspa_backend == "approximate"
