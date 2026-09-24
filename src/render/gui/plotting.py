"""Matplotlib figure builders for GUI presentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib import image as mpl_image
from matplotlib.figure import Figure


SNAPSHOT_COLUMNS = {
    "r_m": 0,
    "z_m": 1,
    "teff_k": 2,
    "kinetic_energy_j": 3,
    "x_m": 4,
    "y_m": 5,
    "vx_m_per_s": 6,
    "vy_m_per_s": 7,
    "vz_m_per_s": 8,
    "speed_m_per_s": 9,
    "kinetic_energy_ev": 10,
    "represented_real_ions": 11,
    "track_id": 14,
    "slot_id": 15,
    "status_code": 16,
    "collision_count_per_ion": 17,
}


def make_empty_figure(title: str, message: str = "No data yet.") -> Figure:
    fig = Figure(figsize=(7.5, 5.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.text(0.5, 0.56, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.46, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    return fig


def _sample_rows(data: np.ndarray, max_points: int = 25000) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] <= max_points:
        return data
    offsets = np.linspace(0, data.shape[0] - 1, max_points, dtype=np.int64)
    return data[offsets]


def make_beam_preview_figure(sample: dict[str, np.ndarray]) -> Figure:
    positions_m = np.asarray(sample["positions_m"], dtype=np.float64)
    velocities_m_per_s = np.asarray(sample["velocities_m_per_s"], dtype=np.float64)
    energy_ev = np.asarray(sample["kinetic_energy_ev"], dtype=np.float64)
    angle_deg = np.asarray(sample["angle_deg"], dtype=np.float64)
    r_m = np.linalg.norm(positions_m[:, :2], axis=1)

    fig = Figure(figsize=(8.0, 6.0), dpi=100)
    axes = fig.subplots(2, 2)

    ax = axes[0][0]
    ax.scatter(positions_m[:, 0] * 1.0e3, positions_m[:, 1] * 1.0e3, s=5, alpha=0.45, linewidths=0)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Initial XY")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0][1]
    ax.scatter(positions_m[:, 2] * 1.0e3, r_m * 1.0e3, s=5, alpha=0.45, linewidths=0)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title("Initial RZ")
    ax.grid(True, alpha=0.25)

    ax = axes[1][0]
    ax.hist(angle_deg[np.isfinite(angle_deg)], bins=36, color="#4C78A8", alpha=0.82)
    ax.set_xlabel("angle to axis [deg]")
    ax.set_ylabel("count")
    ax.set_title("Angular Distribution")
    ax.grid(True, alpha=0.2)

    ax = axes[1][1]
    ax.hist(energy_ev[np.isfinite(energy_ev)], bins=36, color="#F58518", alpha=0.82)
    ax.set_xlabel("kinetic energy [eV]")
    ax.set_ylabel("count")
    ax.set_title("Energy Distribution")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


def make_snapshot_figure(
    snapshot: np.ndarray,
    mode: str,
    *,
    max_points: int = 25000,
) -> Figure:
    data = _sample_rows(np.asarray(snapshot, dtype=np.float64), max_points)
    if data.ndim != 2 or data.shape[0] == 0:
        return make_empty_figure(mode, "Snapshot contains no active particles.")

    fig = Figure(figsize=(7.5, 5.0), dpi=100)
    ax = fig.add_subplot(111)
    mode_key = mode.lower().replace(" ", "_")
    _draw_snapshot_mode(fig, ax, data, mode_key, mode)
    ax.set_title(mode)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def _draw_snapshot_mode(
    fig: Figure,
    ax: Any,
    data: np.ndarray,
    mode_key: str,
    display_mode: str,
) -> None:
    x_mm = data[:, SNAPSHOT_COLUMNS["x_m"]] * 1.0e3
    y_mm = data[:, SNAPSHOT_COLUMNS["y_m"]] * 1.0e3
    z_mm = data[:, SNAPSHOT_COLUMNS["z_m"]] * 1.0e3
    r_mm = data[:, SNAPSHOT_COLUMNS["r_m"]] * 1.0e3
    energy_ev = data[:, SNAPSHOT_COLUMNS["kinetic_energy_ev"]]
    temperature_k = data[:, SNAPSHOT_COLUMNS["teff_k"]]
    if mode_key == "xy":
        scatter = ax.scatter(x_mm, y_mm, c=z_mm, s=5, alpha=0.5, cmap="viridis", linewidths=0)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("z [mm]")
    elif mode_key in {"rz", "snapshot"}:
        scatter = ax.scatter(z_mm, r_mm, c=energy_ev, s=5, alpha=0.5, cmap="plasma", linewidths=0)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("r [mm]")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("kinetic energy [eV]")
    elif mode_key == "phase_space":
        ax.scatter(z_mm, data[:, SNAPSHOT_COLUMNS["vz_m_per_s"]], s=5, alpha=0.5, linewidths=0)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("vz [m/s]")
    elif mode_key == "temperature":
        ax.hist(temperature_k[np.isfinite(temperature_k)], bins=48, color="#54A24B", alpha=0.82)
        ax.set_xlabel("effective temperature [K]")
        ax.set_ylabel("count")
    elif mode_key == "energy":
        ax.hist(energy_ev[np.isfinite(energy_ev)], bins=48, color="#E45756", alpha=0.82)
        ax.set_xlabel("kinetic energy [eV]")
        ax.set_ylabel("count")
    elif mode_key == "loss_map":
        status = data[:, SNAPSHOT_COLUMNS["status_code"]]
        scatter = ax.scatter(z_mm, r_mm, c=status, s=5, alpha=0.5, cmap="tab10", linewidths=0)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("r [mm]")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("status code")
    else:
        ax.text(0.5, 0.5, f"No renderer for {display_mode}.", ha="center", va="center")
        ax.set_axis_off()


def _load_baked_field(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        item = payload.item()
        if isinstance(item, dict):
            return item
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported baked field format: {path}")


def make_field_preview_figure(path: Path, *, phi_key: str = "phi_dc_v") -> Figure:
    path = Path(path)
    if not path.exists():
        return make_empty_figure("Field", f"File not found: {path}")

    fields = _load_baked_field(path)
    grid = fields.get("grid", {})
    simion = fields.get("simion", {})
    z_coords_mm = np.asarray(grid.get("z_coords_m", []), dtype=np.float64) * 1.0e3
    r_coords_mm = np.asarray(grid.get("r_coords_m", []), dtype=np.float64) * 1.0e3
    if z_coords_mm.size == 0 or r_coords_mm.size == 0:
        return make_empty_figure("Field", "Baked field metadata does not contain grid coordinates.")

    phi = simion.get(phi_key)
    if phi is None:
        phi = simion.get("phi_dc_v")
    e_r = simion.get("e_dc_r_v_per_m")
    e_z = simion.get("e_dc_z_v_per_m")
    if phi is None or e_r is None or e_z is None:
        return make_empty_figure("Field", "Baked field does not contain SIMION DC arrays.")

    phi_array = np.asarray(phi, dtype=np.float64)
    e_mag = np.hypot(np.asarray(e_r, dtype=np.float64), np.asarray(e_z, dtype=np.float64))
    extent = [float(z_coords_mm[0]), float(z_coords_mm[-1]), float(r_coords_mm[0]), float(r_coords_mm[-1])]

    fig = Figure(figsize=(8.0, 5.7), dpi=100)
    axes = fig.subplots(1, 2)

    im0 = axes[0].imshow(phi_array, extent=extent, origin="lower", aspect="auto", cmap="coolwarm")
    axes[0].set_xlabel("z [mm]")
    axes[0].set_ylabel("r [mm]")
    axes[0].set_title(phi_key)
    fig.colorbar(im0, ax=axes[0], label="V")

    im1 = axes[1].imshow(e_mag, extent=extent, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("r [mm]")
    axes[1].set_title("|E_dc|")
    fig.colorbar(im1, ax=axes[1], label="V/m")

    fig.suptitle(str(path))
    fig.tight_layout()
    return fig


def make_diagnostic_image_figure(path: Path) -> Figure:
    """Wrap a subprocess-rendered diagnostic PNG in a Tk-safe Figure."""

    path = Path(path)
    if not path.is_file():
        return make_empty_figure("Field Diagnostics", f"File not found: {path}")
    image = mpl_image.imread(path)
    figure = Figure(figsize=(9.0, 6.0), dpi=100)
    axis = figure.add_subplot(111)
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_title(path.name)
    figure.tight_layout()
    return figure


def make_summary_figure(macro_history: list[dict[str, float]]) -> Figure:
    if not macro_history:
        return make_empty_figure("Run Summary", "No macro history is available.")

    time_ms = np.asarray([float(row.get("time_s", 0.0)) for row in macro_history], dtype=np.float64) * 1.0e3
    ion_count = np.asarray([float(row.get("ion_count", np.nan)) for row in macro_history], dtype=np.float64)
    collisions = np.asarray([float(row.get("collision_count", np.nan)) for row in macro_history], dtype=np.float64)
    mean_t = np.asarray([float(row.get("mean_internal_temperature_k", np.nan)) for row in macro_history], dtype=np.float64)
    mean_z_mm = np.asarray([float(row.get("mean_axial_position_m", np.nan)) for row in macro_history], dtype=np.float64) * 1.0e3

    fig = Figure(figsize=(8.0, 6.0), dpi=100)
    axes = fig.subplots(2, 2)
    series = [
        (axes[0][0], ion_count, "Alive particles", "count"),
        (axes[0][1], collisions, "Collisions", "count"),
        (axes[1][0], mean_t, "Mean internal temperature", "K"),
        (axes[1][1], mean_z_mm, "Mean axial position", "mm"),
    ]
    for ax, values, title, ylabel in series:
        ax.plot(time_ms, values, linewidth=1.5)
        ax.set_xlabel("time [ms]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig
