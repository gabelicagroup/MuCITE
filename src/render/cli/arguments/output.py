"""Snapshot and report CLI arguments."""

from __future__ import annotations

from argparse import ArgumentParser

from ....config import OutputConfig

_defaults = OutputConfig()

ARGUMENTS = (
    (("--export-every",), dict(type=int, default=_defaults.export_every, help="Export snapshots every N macro steps.")),
    (("--export-dir",), dict(default="", help="Snapshot output directory.")),
    (("--export-format",), dict(choices=["npy", "h5"], default=_defaults.export_format, help="Snapshot storage format.")),
    (("--trajectory-sample-count",), dict(type=int, default=_defaults.trajectory_sample_count, help="Representative trajectory count.")),
    (("--trajectory-record-every",), dict(type=int, default=_defaults.trajectory_record_every, help="Trajectory record stride in snapshots.")),
    (("--snapshot-plot-max-points",), dict(type=int, default=_defaults.snapshot_plot_max_points, help="Maximum plotted snapshot points.")),
    (("--no-snapshot-plots",), dict(action="store_true", help="Disable automatic snapshot plots.")),
    (("--report",), dict(action="store_true", default=_defaults.report, help="Write JSON/Markdown reports.")),
    (("--report-dir",), dict(default="", help="Report output directory.")),
)


def add_output_arguments(parser: ArgumentParser) -> None:
    for flags, kwargs in ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
