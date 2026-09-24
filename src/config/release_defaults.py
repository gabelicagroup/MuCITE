"""Resolve the explicit public demo from a checkout or installed package."""

from dataclasses import replace
from pathlib import Path

from .constants import PROJECT_ROOT
from .json_config import ConfigDocument, load_config_document


def parameter_example_path() -> Path:
    """Locate only this distribution's parameter example."""
    checkout = PROJECT_ROOT / "configs" / "iict_lite_parameters.example.json"
    if checkout.is_file():
        return checkout
    return Path(__file__).parent / "presets" / "iict_lite_parameters.example.json"


def load_release_default() -> ConfigDocument:
    """Keep explicit --config requests strict; fall back only for the demo."""
    checkout = PROJECT_ROOT / "configs" / "default.json"
    if checkout.is_file():
        return load_config_document(checkout)
    preset = Path(__file__).parent / "presets" / "default.json"
    document = load_config_document(preset)
    return replace(document, simulation=replace(
        document.simulation, iict_parameter_config_path=parameter_example_path(),
    ))
