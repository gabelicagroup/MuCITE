"""Package entry point for `python -m src`.

This dispatches directly to the decoupled CLI presentation layer.
"""

from __future__ import annotations

from .render.cli.runner import run_cli


if __name__ == "__main__":
    run_cli()
