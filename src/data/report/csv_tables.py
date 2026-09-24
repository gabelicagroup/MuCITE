"""CSV artifact writers for immutable simulation report snapshots."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from ..terminal_recorder import TERMINAL_EVENT_FIELDNAMES


FINAL_PARTICLE_FIELDNAMES = (
    "ion_id",
    "final_status",
    "final_x_m",
    "final_y_m",
    "final_z_m",
    "final_r_m",
    "final_vx_m_per_s",
    "final_vy_m_per_s",
    "final_vz_m_per_s",
    "final_speed_m_per_s",
    "local_gas_vx_m_per_s",
    "local_gas_vy_m_per_s",
    "local_gas_vz_m_per_s",
    "local_gas_speed_m_per_s",
    "relative_speed_m_per_s",
    "tof_s",
    "represented_real_ions",
    "parent_real_ions_remaining",
    "fragmented_real_ions_represented",
    "collision_count_per_ion",
    "final_ke_ev",
    "final_internal_temperature_k",
    "electrode_id",
    "surface_distance_m",
)


def write_final_particle_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write the stable final-particle table schema."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_PARTICLE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_terminal_event_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write retained terminal-event rows when no stream already exists."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TERMINAL_EVENT_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def materialize_terminal_event_csv(
    destination: Path,
    rows: list[dict[str, Any]],
    stream_path: Path | None,
) -> None:
    """Preserve stream-copy semantics while producing the report artifact."""

    if stream_path is None:
        write_terminal_event_csv(destination, rows)
        return
    if not _same_path(stream_path, destination):
        if not stream_path.exists():
            raise FileNotFoundError(
                "Terminal-event stream is missing and cannot be materialized "
                f"at {destination}: {stream_path}"
            )
        shutil.copyfile(stream_path, destination)
        return
    if not destination.exists():
        raise FileNotFoundError(
            f"Terminal-event stream is missing: {destination}"
        )
