"""Stable column schema for terminal-event persistence."""

from __future__ import annotations


TERMINAL_EVENT_FIELDNAMES = [
    "event_id",
    "track_id",
    "slot_id",
    "status_code",
    "status",
    "event_time_s",
    "birth_time_s",
    "tof_s",
    "x_m",
    "y_m",
    "z_m",
    "r_m",
    "vx_m_per_s",
    "vy_m_per_s",
    "vz_m_per_s",
    "speed_m_per_s",
    "ke_ev",
    "internal_temperature_k",
    "represented_real_ions",
    "parent_real_ions_remaining",
    "fragmented_real_ions_represented",
    "collision_count_per_ion",
    "electrode_id",
    "surface_distance_m",
]

DEFAULT_TERMINAL_EVENT_SAMPLE_SIZE = 50_000


__all__ = [
    "DEFAULT_TERMINAL_EVENT_SAMPLE_SIZE",
    "TERMINAL_EVENT_FIELDNAMES",
]
