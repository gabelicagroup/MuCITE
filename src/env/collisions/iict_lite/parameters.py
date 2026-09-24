"""Strict JSON/CSV parameter resolution for the independent IICT backend."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ....config import IonTemplate, SimulationConfig
from .fragmentation import (
    EyringFragmentationModel,
    FragmentationModel,
    NoFragmentationModel,
)
from .heat_capacity import (
    ClassicalHeatCapacityModel,
    ConstantCvHeatCapacityModel,
    HeatCapacityModel,
    tabulated_cv_model,
    tabulated_energy_model,
)
from .pseudoatom import (
    ConstantPseudoAtomMassModel,
    PseudoAtomMassModel,
    TabulatedPseudoAtomMassModel,
)


IICT_PARAMETER_SCHEMA_VERSION = 1
DEFAULT_MIN_PSEUDOATOM_MASS_DA = 0.01
DEFAULT_MAX_PSEUDOATOM_MASS_DA = 1.0e6

_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "heat_capacity", "pseudoatom_mass", "fragmentation"}
)
_REQUIRED_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS
_SECTION_FIELDS = {
    "heat_capacity": frozenset(
        {"type", "num_atoms", "cv_j_per_k_per_ion", "csv_path"}
    ),
    "pseudoatom_mass": frozenset(
        {
            "type",
            "mass_da",
            "csv_path",
            "min_mass_da",
            "max_mass_da",
            "gas_name",
        }
    ),
    "fragmentation": frozenset(
        {"type", "delta_h_kj_per_mol", "delta_s_j_per_mol_k"}
    ),
}
_CLI_OVERRIDE_FIELDS = {
    "iict_heat_capacity_model": ("heat_capacity", "type"),
    "iict_num_atoms": ("heat_capacity", "num_atoms"),
    "iict_constant_cv_j_per_k_per_ion": (
        "heat_capacity",
        "cv_j_per_k_per_ion",
    ),
    "iict_heat_capacity_csv_path": ("heat_capacity", "csv_path"),
    "iict_pseudoatom_model": ("pseudoatom_mass", "type"),
    "iict_pseudoatom_mass_da": ("pseudoatom_mass", "mass_da"),
    "iict_pseudoatom_csv_path": ("pseudoatom_mass", "csv_path"),
    "iict_pseudoatom_min_mass_da": ("pseudoatom_mass", "min_mass_da"),
    "iict_pseudoatom_max_mass_da": ("pseudoatom_mass", "max_mass_da"),
    "iict_fragmentation_model": ("fragmentation", "type"),
    "iict_delta_h_kj_per_mol": ("fragmentation", "delta_h_kj_per_mol"),
    "iict_delta_s_j_per_mol_k": ("fragmentation", "delta_s_j_per_mol_k"),
}

_TEMPERATURE_HEADERS = ("temperature_k", "T[K]", "T_k")
_ENERGY_HEADERS = (
    "internal_energy_j_per_ion",
    "U[J/ion]",
    "U_j_per_ion",
)
_CV_HEADERS = (
    "heat_capacity_j_per_k_per_ion",
    "Cv[J/K/ion]",
    "Cv_j_per_k_per_ion",
)
_SPEED_HEADERS = (
    "relative_speed_m_per_s",
    "relative_speed[m/s]",
)
_MASS_HEADERS = ("mass_da", "mass[Da]")


@dataclass(frozen=True)
class ResolvedIictParameters:
    """Validated models plus complete effective-value/source metadata."""

    heat_capacity: HeatCapacityModel
    pseudoatom_mass: PseudoAtomMassModel
    fragmentation: FragmentationModel
    effective_parameters: dict[str, Any]
    parameter_sources: dict[str, str]
    parameter_config_path: str | None
    parameter_config_sha256: str | None


def _strict_object(
    value: Any,
    *,
    name: str,
    allowed: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}")
    return dict(value)


def _load_parameter_json(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], str]:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    envelope = _strict_object(
        payload,
        name="iict parameter document",
        allowed=_TOP_LEVEL_FIELDS,
    )
    missing = sorted(_REQUIRED_TOP_LEVEL_FIELDS - set(envelope))
    if missing:
        raise ValueError(f"Missing iict parameter field(s): {', '.join(missing)}")
    version = envelope["schema_version"]
    if type(version) is not int or version != IICT_PARAMETER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported iict schema_version: {version!r}.")
    sections = {
        name: _strict_object(
            envelope[name],
            name=name,
            allowed=_SECTION_FIELDS[name],
        )
        for name in ("heat_capacity", "pseudoatom_mass", "fragmentation")
    }
    return sections, hashlib.sha256(raw_bytes).hexdigest()


def _resolve_project_path(path: Path, project_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _resolve_table_path(value: Any, base_dir: Path, field_name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field_name} requires a non-empty csv_path.")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} CSV not found: {path}")
    return path


def _parameter_sections(
    config: SimulationConfig,
    project_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], str | None, str | None]:
    path = config.iict_parameter_config_path
    sections = {name: {} for name in _SECTION_FIELDS}
    sources: dict[str, str] = {}
    config_path: str | None = None
    digest: str | None = None
    if path is not None:
        resolved_path = _resolve_project_path(path, project_root)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"iict parameter JSON not found: {resolved_path}")
        sections, digest = _load_parameter_json(resolved_path)
        config_path = str(resolved_path)
        for section, values in sections.items():
            for key in values:
                sources[f"{section}.{key}"] = f"iict_parameter_json:{resolved_path}"
        base_dir = resolved_path.parent
    else:
        base_dir = project_root
    _apply_simulation_overrides(config, sections, sources)
    for section in ("heat_capacity", "pseudoatom_mass"):
        if "csv_path" in sections[section]:
            source = sources.get(f"{section}.csv_path", "")
            table_base = (
                project_root
                if source in {"cli", "simulation_config"}
                else base_dir
            )
            sections[section]["csv_path"] = _resolve_table_path(
                sections[section]["csv_path"],
                table_base,
                section,
            )
    return sections, sources, config_path, digest


def _apply_simulation_overrides(
    config: SimulationConfig,
    sections: dict[str, dict[str, Any]],
    sources: dict[str, str],
) -> None:
    cli_fields = set(config.iict_cli_override_fields)
    for config_field, (section, key) in _CLI_OVERRIDE_FIELDS.items():
        value = getattr(config, config_field)
        if value is None:
            continue
        sections[section][key] = value
        source = "cli" if config_field in cli_fields else "simulation_config"
        sources[f"{section}.{key}"] = source


def _validate_present_parameter_types(
    sections: Mapping[str, Mapping[str, Any]],
) -> None:
    heat = sections["heat_capacity"]
    pseudoatom = sections["pseudoatom_mass"]
    fragmentation = sections["fragmentation"]
    if "num_atoms" in heat and _integer(heat, "num_atoms") <= 2:
        raise ValueError("heat_capacity.num_atoms must be greater than 2.")
    if "cv_j_per_k_per_ion" in heat:
        _positive_float(heat, "cv_j_per_k_per_ion")
    for section, key in (
        (pseudoatom, "mass_da"),
        (pseudoatom, "min_mass_da"),
        (pseudoatom, "max_mass_da"),
        (fragmentation, "delta_h_kj_per_mol"),
    ):
        if key in section:
            _positive_float(section, key)
    if "delta_s_j_per_mol_k" in fragmentation:
        _finite_float(fragmentation, "delta_s_j_per_mol_k")
    gas_name = pseudoatom.get("gas_name")
    if gas_name is not None and (
        not isinstance(gas_name, str) or not gas_name.strip()
    ):
        raise ValueError("pseudoatom_mass.gas_name must be a non-empty string.")
    for section, section_name in (
        (heat, "heat_capacity"),
        (pseudoatom, "pseudoatom_mass"),
    ):
        path = section.get("csv_path")
        if path is not None and (
            not isinstance(path, (str, Path)) or not str(path).strip()
        ):
            raise ValueError(f"{section_name}.csv_path must be a non-empty path.")


def _required_type(
    section: Mapping[str, Any],
    section_name: str,
    allowed: set[str],
) -> str:
    raw = section.get("type")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{section_name}.type is required for iict-lite.")
    value = raw.strip().lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{section_name}.type must be one of {{{choices}}}.")
    return value


def _finite_float(section: Mapping[str, Any], key: str) -> float:
    raw = section.get(key)
    if type(raw) not in {int, float}:
        raise ValueError(f"{key} must be a JSON number.")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite.")
    return value


def _positive_float(section: Mapping[str, Any], key: str) -> float:
    value = _finite_float(section, key)
    if value <= 0.0:
        raise ValueError(f"{key} must be positive.")
    return value


def _integer(section: Mapping[str, Any], key: str) -> int:
    raw = section.get(key)
    if type(raw) is not int:
        raise ValueError(f"{key} must be an integer.")
    return int(raw)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if not headers or not rows:
        raise ValueError(f"CSV must contain a header and data rows: {path}")
    return headers, rows


def _find_header(headers: list[str], aliases: tuple[str, ...], name: str) -> str:
    matches = [header for header in headers if header.strip() in aliases]
    if len(matches) != 1:
        expected = ", ".join(aliases)
        raise ValueError(f"CSV requires exactly one {name} column: {expected}.")
    return matches[0]


def _numeric_column(
    rows: list[dict[str, str]],
    header: str,
    path: Path,
) -> np.ndarray:
    try:
        values = np.asarray([float(row[header]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CSV column {header!r} contains non-numeric data: {path}") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"CSV column {header!r} must contain finite values.")
    return values


def _heat_table(path: Path) -> HeatCapacityModel:
    headers, rows = _read_csv(path)
    temperature_header = _find_header(
        headers,
        _TEMPERATURE_HEADERS,
        "temperature [K]",
    )
    energy_matches = [header for header in headers if header.strip() in _ENERGY_HEADERS]
    cv_matches = [header for header in headers if header.strip() in _CV_HEADERS]
    if (len(energy_matches), len(cv_matches)) not in {(1, 0), (0, 1)}:
        raise ValueError("Heat CSV requires exactly one U[J/ion] or Cv[J/K/ion] column.")
    temperatures = _numeric_column(rows, temperature_header, path)
    if energy_matches:
        energy = _numeric_column(rows, energy_matches[0], path)
        return tabulated_energy_model(temperatures, energy, source_path=str(path))
    cv = _numeric_column(rows, cv_matches[0], path)
    return tabulated_cv_model(temperatures, cv, source_path=str(path))


def _pseudoatom_table(
    path: Path,
    section: Mapping[str, Any],
    min_mass_da: float,
    max_mass_da: float,
) -> PseudoAtomMassModel:
    headers, rows = _read_csv(path)
    t_header = _find_header(headers, _TEMPERATURE_HEADERS, "temperature [K]")
    v_header = _find_header(headers, _SPEED_HEADERS, "relative speed [m/s]")
    m_header = _find_header(headers, _MASS_HEADERS, "mass [Da]")
    gas_name = section.get("gas_name")
    if gas_name is not None and not isinstance(gas_name, str):
        raise ValueError("pseudoatom_mass.gas_name must be a string.")
    return TabulatedPseudoAtomMassModel(
        _numeric_column(rows, t_header, path),
        _numeric_column(rows, v_header, path),
        _numeric_column(rows, m_header, path),
        min_mass_da=min_mass_da,
        max_mass_da=max_mass_da,
        source_path=str(path),
        gas_name=gas_name,
    )


def _build_heat_model(
    section: Mapping[str, Any],
    template: IonTemplate,
    sources: dict[str, str],
) -> HeatCapacityModel:
    model_type = _required_type(
        section,
        "heat_capacity",
        {"classical", "constant_cv", "tabulated"},
    )
    if model_type == "classical":
        if "num_atoms" in section:
            num_atoms = _integer(section, "num_atoms")
        else:
            num_atoms = int(template.num_atoms)
            sources["heat_capacity.num_atoms"] = "ion_template"
        return ClassicalHeatCapacityModel(num_atoms)
    if model_type == "constant_cv":
        return ConstantCvHeatCapacityModel(
            _positive_float(section, "cv_j_per_k_per_ion")
        )
    if "csv_path" not in section:
        raise ValueError("heat_capacity.csv_path is required for tabulated.")
    return _heat_table(Path(section["csv_path"]))


def _mass_bounds(
    section: Mapping[str, Any],
    sources: dict[str, str],
) -> tuple[float, float]:
    if "min_mass_da" in section:
        minimum = _positive_float(section, "min_mass_da")
    else:
        minimum = DEFAULT_MIN_PSEUDOATOM_MASS_DA
        sources["pseudoatom_mass.min_mass_da"] = "iict_lite_safety_default"
    if "max_mass_da" in section:
        maximum = _positive_float(section, "max_mass_da")
    else:
        maximum = DEFAULT_MAX_PSEUDOATOM_MASS_DA
        sources["pseudoatom_mass.max_mass_da"] = "iict_lite_safety_default"
    return minimum, maximum


def _build_pseudoatom_model(
    section: Mapping[str, Any],
    sources: dict[str, str],
) -> PseudoAtomMassModel:
    model_type = _required_type(
        section,
        "pseudoatom_mass",
        {"constant", "tabulated"},
    )
    minimum, maximum = _mass_bounds(section, sources)
    gas_name = section.get("gas_name")
    if gas_name is not None and not isinstance(gas_name, str):
        raise ValueError("pseudoatom_mass.gas_name must be a string.")
    if model_type == "constant":
        return ConstantPseudoAtomMassModel(
            _positive_float(section, "mass_da"),
            minimum,
            maximum,
            gas_name,
        )
    if "csv_path" not in section:
        raise ValueError("pseudoatom_mass.csv_path is required for tabulated.")
    return _pseudoatom_table(
        Path(section["csv_path"]),
        section,
        minimum,
        maximum,
    )


def _build_fragmentation_model(
    section: Mapping[str, Any],
) -> FragmentationModel:
    model_type = _required_type(
        section,
        "fragmentation",
        {"none", "eyring"},
    )
    if model_type == "none":
        return NoFragmentationModel()
    return EyringFragmentationModel(
        _positive_float(section, "delta_h_kj_per_mol"),
        _finite_float(section, "delta_s_j_per_mol_k"),
    )


def resolve_iict_parameters(
    config: SimulationConfig,
    template: IonTemplate,
    *,
    project_root: Path,
) -> ResolvedIictParameters:
    """Resolve JSON first, then explicit simulation/CLI overrides."""

    sections, sources, config_path, digest = _parameter_sections(
        config,
        project_root,
    )
    _validate_present_parameter_types(sections)
    heat = _build_heat_model(sections["heat_capacity"], template, sources)
    pseudoatom = _build_pseudoatom_model(sections["pseudoatom_mass"], sources)
    fragmentation = _build_fragmentation_model(sections["fragmentation"])
    effective = {
        "schema_version": IICT_PARAMETER_SCHEMA_VERSION,
        "heat_capacity": heat.metadata,
        "pseudoatom_mass": pseudoatom.metadata,
        "fragmentation": fragmentation.metadata,
    }
    return ResolvedIictParameters(
        heat,
        pseudoatom,
        fragmentation,
        effective,
        sources,
        config_path,
        digest,
    )


__all__ = [
    "IICT_PARAMETER_SCHEMA_VERSION",
    "ResolvedIictParameters",
    "resolve_iict_parameters",
]
