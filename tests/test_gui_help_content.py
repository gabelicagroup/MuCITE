"""Coverage and disclosure contracts for reusable GUI help text."""

from __future__ import annotations

from dataclasses import fields

from src.render.gui.help_content import (
    PARAMETER_HELP_BY_SECTION,
    about_text,
    parameter_reference_text,
    simulation_workflow_text,
)
from src.render.gui.models import (
    AppConfig,
    BeamConfig,
    FieldBakeConfig,
    OutputConfig,
    RuntimeConfig,
)


def _field_names(model_type: type[object]) -> set[str]:
    return {item.name for item in fields(model_type)}


def test_parameter_help_has_exact_persistent_app_config_field_coverage() -> None:
    app_groups = {"beam", "field_bake", "runtime", "output", "state"}
    expected_project = _field_names(AppConfig) - app_groups
    expected = {
        "Project": expected_project,
        "Beam / Ion": _field_names(BeamConfig),
        "Field Baker": _field_names(FieldBakeConfig),
        "Runtime": _field_names(RuntimeConfig),
        "Output": _field_names(OutputConfig),
    }
    actual = {
        section: set(entries)
        for section, entries in PARAMETER_HELP_BY_SECTION.items()
    }
    assert actual == expected
    assert all(
        description.strip()
        for entries in PARAMETER_HELP_BY_SECTION.values()
        for description in entries.values()
    )


def test_parameter_reference_renders_every_field_and_nullable_semantics() -> None:
    reference = parameter_reference_text()
    for entries in PARAMETER_HELP_BY_SECTION.values():
        for name in entries:
            assert f"{name}\n    " in reference
    assert "source_birth_velocity_z_max_mm" in reference
    assert "Blank/None means no explicit upper bound" in reference
    assert "pic_grid_nz=0" in reference
    assert "single-phase RF peak voltage" in reference
    assert "iict_parameter_config_path" in reference
    assert "Blank/None reads the type from the parameter JSON" in reference
    assert "No protein/N2-derived mass" in reference
    assert "weighted ion pack" in reference
    assert "macro ions" not in reference


def test_workflow_documents_execution_order_and_validated_conventions() -> None:
    workflow = simulation_workflow_text()
    required = (
        "∇²φ_sc = -ρ/ε₀",
        "P = 1 - exp(-λ Δt)",
        "m dv/dt = q E_total",
        "r = sqrt(x²+y²)",
        "Vpeak is a single-phase peak voltage, not Vpp",
        "SIMION x→Python z",
        "SIMION export +90° to Python rf_phase_deg=-90°",
        "Rendering and reporting are output-layer services",
        "10.1016/j.ijms.2024.117290",
        "does not claim IonSPA numerical equivalence",
    )
    assert all(text in workflow for text in required)


def test_about_is_transparent_about_authorship_license_and_warranty() -> None:
    notice = about_text()
    assert "MuCITE" in notice
    assert "Dr. Yihui Yan" in notice
    assert "GNU General Public License version 3" in notice
    assert "See LICENSE.md" in notice
    assert "Third-party notice" in notice
    assert "each component" in notice.lower()
    assert "neither imports IonSPA nor reads IonSPA data files" in notice
    assert "lawful provider/source" in notice
    assert "provided as-is, without warranty" in notice
