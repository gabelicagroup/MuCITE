"""Command-line presentation and request-adaptation layer."""

from .parser import parse_args
from .runner import build_demo_simulation, run_cli

__all__ = ["build_demo_simulation", "parse_args", "run_cli"]
