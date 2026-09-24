"""Versioned, fail-closed JSON documents for simulation and ion requests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, get_args, get_origin, get_type_hints

import numpy as np

from .factories import (
    make_execution_config,
    make_ion_template,
    make_output_config,
    make_simulation_config,
)
from .models import (
    ExecutionConfig,
    IonTemplate,
    OutputConfig,
    SimulationConfig,
)
from .validation import SIMULATION_CONFIG_PATH_FIELDS

CONFIG_SCHEMA_VERSION = 1
_DOCUMENT_FIELDS = frozenset(
    {"schema_version", "simulation", "ion", "execution", "output"}
)
_REQUIRED_DOCUMENT_FIELDS = frozenset({"schema_version", "simulation", "ion"})


@dataclass(frozen=True)
class ConfigDocument:
    """A validated configuration document independent of runtime state."""

    simulation: SimulationConfig
    ion: IonTemplate
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    schema_version: int = CONFIG_SCHEMA_VERSION


def make_config_document(
    simulation: Optional[SimulationConfig] = None,
    ion: Optional[IonTemplate] = None,
    execution: Optional[ExecutionConfig] = None,
    output: Optional[OutputConfig] = None,
) -> ConfigDocument:
    """Create a canonical in-memory version-1 document."""

    return ConfigDocument(
        simulation=make_simulation_config(simulation),
        ion=make_ion_template(ion),
        execution=make_execution_config(execution),
        output=make_output_config(output),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported configuration JSON value: {type(value).__name__}.")


def config_document_to_dict(document: ConfigDocument) -> dict[str, Any]:
    """Serialize a validated document to JSON-compatible builtins."""

    if not isinstance(document, ConfigDocument):
        raise TypeError("document must be a ConfigDocument instance.")
    if document.schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config schema_version: {document.schema_version!r}."
        )
    canonical = make_config_document(
        document.simulation,
        document.ion,
        document.execution,
        document.output,
    )
    simulation = asdict(canonical.simulation)
    # CLI provenance is invocation-local; persisted values become config values.
    simulation["iict_cli_override_fields"] = []
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "simulation": _json_safe(simulation),
        "ion": _json_safe(asdict(canonical.ion)),
        "execution": _json_safe(asdict(canonical.execution)),
        "output": _json_safe(asdict(canonical.output)),
    }


def _checked_section(
    payload: Any,
    *,
    section_name: str,
    model_type: type[Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{section_name} must be a JSON object.")
    known = {item.name for item in fields(model_type)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"Unknown {section_name} configuration field(s): "
            + ", ".join(unknown)
        )
    values = dict(payload)
    _validate_json_section_types(
        values,
        section_name=section_name,
        model_type=model_type,
    )
    return values


def _matches_json_annotation(value: Any, annotation: Any) -> bool:
    options = get_args(annotation)
    if get_origin(annotation) is tuple:
        if not isinstance(value, list):
            return False
        item_type = options[0] if options else Any
        return all(
            _matches_json_annotation(item, item_type)
            for item in value
        )
    if value is None:
        return type(None) in options
    if options and type(None) in options:
        return any(
            _matches_json_annotation(value, option)
            for option in options
            if option is not type(None)
        )
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int
    if annotation is float:
        return type(value) in {int, float}
    if annotation is str:
        return type(value) is str
    if annotation is Path:
        return type(value) is str
    if annotation is np.ndarray:
        return isinstance(value, list)
    return True


def _validate_json_section_types(
    values: dict[str, Any],
    *,
    section_name: str,
    model_type: type[Any],
) -> None:
    annotations = get_type_hints(model_type)
    for field_name, value in values.items():
        annotation = annotations[field_name]
        if not _matches_json_annotation(value, annotation):
            raise ValueError(
                f"{section_name}.{field_name} has an invalid JSON type."
            )


def _validate_document_envelope(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Configuration document must be a JSON object.")
    unknown = sorted(set(payload) - _DOCUMENT_FIELDS)
    if unknown:
        raise ValueError(
            "Unknown configuration document field(s): " + ", ".join(unknown)
        )
    missing = sorted(_REQUIRED_DOCUMENT_FIELDS - set(payload))
    if missing:
        raise ValueError(
            "Missing configuration document field(s): " + ", ".join(missing)
        )
    version = payload["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported config schema_version: {version!r}.")
    return payload


def _coerce_optional_paths(
    values: dict[str, Any],
    field_names: tuple[str, ...],
    section_name: str,
) -> None:
    for field_name in field_names:
        if field_name not in values:
            continue
        value = values[field_name]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{section_name}.{field_name} must be a string or null.")
        if value is not None:
            values[field_name] = Path(value)


def config_document_from_dict(payload: Any) -> ConfigDocument:
    """Load one strict version-1 document from decoded JSON data."""

    payload = _validate_document_envelope(payload)
    simulation_values = _checked_section(
        payload["simulation"],
        section_name="simulation",
        model_type=SimulationConfig,
    )
    simulation_values["iict_cli_override_fields"] = ()
    ion_values = _checked_section(
        payload["ion"],
        section_name="ion",
        model_type=IonTemplate,
    )
    execution_values = _checked_section(
        payload.get("execution", {}),
        section_name="execution",
        model_type=ExecutionConfig,
    )
    output_values = _checked_section(
        payload.get("output", {}),
        section_name="output",
        model_type=OutputConfig,
    )
    _coerce_optional_paths(
        simulation_values,
        SIMULATION_CONFIG_PATH_FIELDS,
        "simulation",
    )
    _coerce_optional_paths(
        output_values,
        ("export_dir", "report_dir"),
        "output",
    )
    return make_config_document(
        make_simulation_config(**simulation_values),
        make_ion_template(**ion_values),
        make_execution_config(**execution_values),
        make_output_config(**output_values),
    )


def dump_config_document(path: Path, document: ConfigDocument) -> None:
    """Write a canonical configuration document as UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            config_document_to_dict(document),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_config_document(path: Path) -> ConfigDocument:
    """Read and validate a canonical configuration document."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return config_document_from_dict(payload)


dump_config_json = dump_config_document
load_config_json = load_config_document
