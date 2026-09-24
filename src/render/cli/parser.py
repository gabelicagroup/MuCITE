"""CLI parser construction and explicit-option tracking."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

from .arguments import (
    add_boundary_arguments,
    add_collision_arguments,
    add_field_arguments,
    add_ion_arguments,
    add_output_arguments,
    add_pic_arguments,
    add_runtime_arguments,
    add_source_arguments,
)


@dataclass(frozen=True)
class ParsedCli:
    namespace: argparse.Namespace
    argv: tuple[str, ...]
    explicit_dests: frozenset[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Simu_IonSource coupled simulation."
    )
    add_runtime_arguments(parser)
    add_ion_arguments(parser)
    add_source_arguments(parser)
    add_field_arguments(parser)
    add_pic_arguments(parser)
    add_collision_arguments(parser)
    add_boundary_arguments(parser)
    add_output_arguments(parser)
    return parser


def explicit_argument_destinations(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> frozenset[str]:
    destinations: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            destinations.add(action.dest)
    return frozenset(destinations)


def parse_cli(argv: Optional[list[str]] = None) -> ParsedCli:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    namespace = parser.parse_args(raw_argv)
    if namespace.deprecated_rf_vpp is not None:
        parser.error(
            "--rf-vpp has been removed. Convert Vpp to single-phase Vpeak "
            "and pass --rf-peak-voltage."
        )
    return ParsedCli(
        namespace=namespace,
        argv=tuple(raw_argv),
        explicit_dests=explicit_argument_destinations(parser, raw_argv),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Compatibility parser returning the legacy argparse Namespace."""

    return parse_cli(argv).namespace
