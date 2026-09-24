"""Plot electric field, gas flow, and electrode mask alignment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm, Normalize

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from ..boundary_masks import load_electrode_mask
    from .plot_style import add_subplot_labels, apply_paper_style, scaled
except ImportError:  # pragma: no cover - supports direct script execution.
    from src.data.boundary_masks import load_electrode_mask  # type: ignore
    from src.data.diagnostics.plot_style import (  # type: ignore
        add_subplot_labels,
        apply_paper_style,
        scaled,
    )


DEFAULT_FIELD_PATH = (
    Path("outputs") / "slens100x_bake_0p01_zmax65_fluent_interior_clipped_capexit4p5" / "baked_fields.npy"
)
DEFAULT_MASK_CANDIDATES = (
    Path("E_field") / "slens" / "slens100x_gem.patxt",
    Path("E_field") / "slens100x_gem.patxt",
)


def _default_mask_path() -> Path:
    for candidate in DEFAULT_MASK_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_MASK_CANDIDATES[0]


def _load_baked(path: Path) -> dict[str, object]:
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a baked field dictionary.")
    for group in ("grid", "simion", "fluent"):
        if group not in payload:
            raise KeyError(f"{path} is missing required group {group!r}.")
    return payload


def _window_indices(coords_mm: np.ndarray, lower_mm: float, upper_mm: float, axis_name: str) -> np.ndarray:
    if upper_mm <= lower_mm:
        raise ValueError(f"{axis_name} max must be greater than min.")
    indices = np.flatnonzero((coords_mm >= lower_mm) & (coords_mm <= upper_mm))
    if indices.size == 0:
        raise ValueError(f"No {axis_name} grid points in [{lower_mm:g}, {upper_mm:g}] mm.")
    return indices


def _plot_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    """Retain full axis extent while bounding display-only array sizes."""

    limit = int(max_points)
    if limit <= 0 or indices.size <= limit:
        return indices
    offsets = np.linspace(0, indices.size - 1, limit, dtype=np.int64)
    return indices[np.unique(offsets)]


def _log_norm(values: np.ndarray, floor: float) -> LogNorm:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return LogNorm(vmin=floor, vmax=floor * 10.0)
    vmin = max(float(np.nanpercentile(positive, 1.0)), floor)
    vmax = max(float(np.nanpercentile(positive, 99.5)), vmin * 10.0)
    return LogNorm(vmin=vmin, vmax=vmax)


def _linear_norm(values: np.ndarray) -> Normalize:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.nanpercentile(finite, 1.0))
    vmax = float(np.nanpercentile(finite, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def _mesh_for_mask(
    mask,
    z_min_mm: float,
    z_max_mm: float,
    r_min_mm: float,
    r_max_mm: float,
    *,
    max_z_points: int,
    max_r_points: int,
):
    mask_z_mm = mask.z_coords_m * 1.0e3
    mask_r_mm = mask.r_coords_m * 1.0e3
    z_idx = _window_indices(mask_z_mm, z_min_mm, z_max_mm, "mask z")
    r_idx = _window_indices(mask_r_mm, r_min_mm, r_max_mm, "mask r")
    z_idx = _plot_indices(z_idx, max_z_points)
    r_idx = _plot_indices(r_idx, max_r_points)
    mask_local = mask.metal_mask[np.ix_(r_idx, z_idx)].astype(np.float64)
    zg, rg = np.meshgrid(mask_z_mm[z_idx], mask_r_mm[r_idx])
    return zg, rg, mask_local


def _draw_mask_overlay(ax, mask_zg, mask_rg, mask_local, *, alpha: float, contour_width: float) -> int:
    metal_count = int(np.count_nonzero(mask_local))
    if metal_count <= 0:
        return 0
    ax.contourf(mask_zg, mask_rg, mask_local, levels=[0.5, 1.5], colors=["#d9d9d9"], alpha=alpha)
    ax.contour(mask_zg, mask_rg, mask_local, levels=[0.5], colors="black", linewidths=contour_width)
    return metal_count


def _draw_vector_direction_arrows(
    ax,
    zg: np.ndarray,
    rg: np.ndarray,
    z_component: np.ndarray,
    r_component: np.ndarray,
    *,
    stride_z: int,
    stride_r: int,
    length_mm: float,
    color: str,
    alpha: float,
    width: float,
) -> int:
    stride_z = max(1, int(stride_z))
    stride_r = max(1, int(stride_r))
    z_points = zg[::stride_r, ::stride_z]
    r_points = rg[::stride_r, ::stride_z]
    z_values = z_component[::stride_r, ::stride_z]
    r_values = r_component[::stride_r, ::stride_z]
    magnitude = np.hypot(z_values, r_values)
    valid = np.isfinite(z_points) & np.isfinite(r_points) & np.isfinite(magnitude) & (magnitude > 0.0)
    if not np.any(valid):
        return 0
    u_mm = np.zeros_like(z_values, dtype=np.float64)
    v_mm = np.zeros_like(r_values, dtype=np.float64)
    u_mm[valid] = z_values[valid] / magnitude[valid] * float(length_mm)
    v_mm[valid] = r_values[valid] / magnitude[valid] * float(length_mm)
    ax.quiver(
        z_points[valid],
        r_points[valid],
        u_mm[valid],
        v_mm[valid],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        alpha=float(alpha),
        width=float(width),
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.5,
        zorder=5,
    )
    return int(np.count_nonzero(valid))


def _resolve_mask_path(raw_value: str) -> Path:
    text = str(raw_value).strip()
    if text.lower() in {"default", "auto"}:
        return _default_mask_path()
    path = Path(text)
    if path.exists() or path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    # Current workspaces may keep legacy S-lens PA exports under E_field/slens.
    slens_path = Path("E_field") / "slens" / path.name
    if slens_path.exists():
        return slens_path
    return path


def _parse_panel_keys(panel_text: str, *, include_total_field_vector_panel: bool) -> list[str]:
    aliases = {
        "electrode": "mask",
        "electrode_mask": "mask",
        "metal": "mask",
        "edc": "dc",
        "e_dc": "dc",
        "erf": "rf",
        "e_rf": "rf",
        "rf_actual": "rf",
        "flow": "gas",
        "vgas": "gas",
        "v_gas": "gas",
        "total_field": "total",
        "rf_dc": "total",
        "dc_rf": "total",
    }
    valid = {"mask", "dc", "rf", "gas", "total"}
    if str(panel_text).strip():
        raw_keys = [item.strip().lower() for item in str(panel_text).split(",") if item.strip()]
    else:
        raw_keys = ["mask", "dc", "rf", "gas"]
        if include_total_field_vector_panel:
            raw_keys.append("total")

    keys: list[str] = []
    for raw_key in raw_keys:
        key = aliases.get(raw_key, raw_key)
        if key not in valid:
            raise ValueError(f"Unknown panel {raw_key!r}. Valid panels: mask, dc, rf, gas, total.")
        if key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError("--panels must select at least one panel.")
    return keys


def _layout_shape(layout: str, panel_count: int) -> tuple[int, int]:
    layout_text = str(layout).strip().lower()
    if layout_text == "auto":
        if panel_count <= 1:
            return 1, 1
        if panel_count == 2:
            return 1, 2
        if panel_count <= 4:
            return 2, 2
        if panel_count <= 6:
            return 2, 3
        return int(math.ceil(panel_count / 3.0)), 3

    parts = layout_text.split("x", 1)
    if len(parts) != 2:
        raise ValueError("--layout must be auto or rowsxcols, for example 1x1, 1x2, 2x2, 2x3.")
    try:
        rows = int(parts[0])
        cols = int(parts[1])
    except ValueError as exc:
        raise ValueError("--layout rows and columns must be integers.") from exc
    if rows <= 0 or cols <= 0:
        raise ValueError("--layout rows and columns must be positive.")
    if rows * cols < panel_count:
        raise ValueError(f"--layout {layout_text!r} has only {rows * cols} slots for {panel_count} panels.")
    return rows, cols


def _figure_size(nrows: int, ncols: int) -> tuple[float, float]:
    if nrows == 1 and ncols == 1:
        return 7.4, 5.8
    return max(6.2, 5.0 * ncols), max(4.8, 4.25 * nrows)


def make_plot(args: argparse.Namespace) -> Path:
    apply_paper_style(font_scale=args.paper_font_scale, line_scale=args.paper_line_scale)
    field_path = Path(args.field)
    baked = _load_baked(field_path)
    grid = baked["grid"]
    simion = baked["simion"]
    fluent = baked["fluent"]
    if not isinstance(grid, dict) or not isinstance(simion, dict) or not isinstance(fluent, dict):
        raise TypeError("baked field groups must be dictionaries.")

    z_mm = np.asarray(grid["z_coords_m"], dtype=np.float64) * 1.0e3
    r_mm = np.asarray(grid["r_coords_m"], dtype=np.float64) * 1.0e3
    if args.global_window:
        args.z_min_mm = float(np.nanmin(z_mm))
        args.z_max_mm = float(np.nanmax(z_mm))
        args.r_min_mm = float(np.nanmin(r_mm))
        args.r_max_mm = float(np.nanmax(r_mm))
    z_idx = _window_indices(z_mm, args.z_min_mm, args.z_max_mm, "z")
    r_idx = _window_indices(r_mm, args.r_min_mm, args.r_max_mm, "r")
    z_idx = _plot_indices(z_idx, args.plot_max_z_points)
    r_idx = _plot_indices(r_idx, args.plot_max_r_points)
    zg, rg = np.meshgrid(z_mm[z_idx], r_mm[r_idx])

    def subset(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)[np.ix_(r_idx, z_idx)]

    e_dc_r = subset(simion["e_dc_r_v_per_m"])
    e_dc_z = subset(simion["e_dc_z_v_per_m"])
    e_rf_r = subset(simion["e_rf_r_v_per_m"])
    e_rf_z = subset(simion["e_rf_z_v_per_m"])
    rf_reference_peak_voltage_v = float(simion.get("rf_reference_peak_voltage_v", 1.0))
    if not np.isfinite(rf_reference_peak_voltage_v) or rf_reference_peak_voltage_v <= 0.0:
        raise ValueError(f"Invalid rf_reference_peak_voltage_v in baked field: {rf_reference_peak_voltage_v!r}")
    rf_scale = float(args.rf_peak_voltage_v) / rf_reference_peak_voltage_v
    rf_phase_factor = math.cos(math.radians(float(args.rf_phase_deg)))
    e_rf_actual_r = e_rf_r * rf_scale
    e_rf_actual_z = e_rf_z * rf_scale
    e_total_r = e_dc_r + e_rf_actual_r * rf_phase_factor
    e_total_z = e_dc_z + e_rf_actual_z * rf_phase_factor
    e_dc = np.hypot(e_dc_r, e_dc_z)
    e_rf_actual = np.hypot(e_rf_actual_r, e_rf_actual_z)
    e_total = np.hypot(e_total_r, e_total_z)
    v_gas = np.hypot(subset(fluent["v_r_m_per_s"]), subset(fluent["v_z_m_per_s"]))

    mask_path: Path | None
    if str(args.electrode_mask).strip().lower() in {"", "none", "off"}:
        mask_path = None
        mask = None
        mask_zg = mask_rg = mask_local = None
    else:
        mask_path = _resolve_mask_path(args.electrode_mask)
        mask_cache = Path(args.electrode_mask_cache) if args.electrode_mask_cache else None
        mask_offset_m = None if args.electrode_mask_z_offset_mm is None else float(args.electrode_mask_z_offset_mm) * 1.0e-3
        mask = load_electrode_mask(
            source_path=mask_path,
            cache_path=mask_cache,
            z_offset_m=mask_offset_m,
            field_metadata=baked,
            project_root=Path.cwd(),
        )
        if mask is None:
            raise RuntimeError("Electrode mask was requested but did not load.")
        mask_zg, mask_rg, mask_local = _mesh_for_mask(
            mask,
            args.z_min_mm,
            args.z_max_mm,
            args.r_min_mm,
            args.r_max_mm,
            max_z_points=args.plot_max_z_points,
            max_r_points=args.plot_max_r_points,
        )

    output = Path(args.output) if args.output else Path(args.output_dir) / "field_flow_mask_alignment.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    panel_defs = {
        "mask": {"key": "mask", "title": "Electrode metal mask", "data": None, "cmap": None, "norm": None, "vector": None},
        "dc": {
            "key": "dc",
            "title": "|E_DC| [V/m]",
            "data": np.maximum(e_dc, 1.0e-30),
            "cmap": args.dc_cmap,
            "norm": _log_norm(e_dc, 1.0e-8),
            "vector": None,
        },
        "rf": {
            "key": "rf",
            "title": f"|E_RF| peak @ Vpeak={args.rf_peak_voltage_v:g} V [V/m]",
            "data": np.maximum(e_rf_actual, 1.0e-30),
            "cmap": args.rf_cmap,
            "norm": _log_norm(e_rf_actual, 1.0e-4),
            "vector": None,
        },
        "gas": {"key": "gas", "title": "|v_gas| [m/s]", "data": v_gas, "cmap": args.gas_cmap, "norm": _linear_norm(v_gas), "vector": None},
        "total": {
            "key": "total",
            "title": f"E_DC + E_RF(t) direction, phase={args.rf_phase_deg:g} deg",
            "data": np.maximum(e_total, 1.0e-30),
            "cmap": args.rf_cmap,
            "norm": _log_norm(e_total, 1.0e-8),
            "vector": (e_total_z, e_total_r),
        },
    }
    panel_keys = _parse_panel_keys(args.panels, include_total_field_vector_panel=bool(args.include_total_field_vector_panel))
    panels = [panel_defs[key] for key in panel_keys]

    nrows, ncols = _layout_shape(args.layout, len(panels))
    width, height = _figure_size(nrows, ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), dpi=args.dpi, sharex=True, sharey=True, constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    metal_count = 0
    rf_vector_arrow_count = 0
    for ax, panel in zip(axes_flat, panels):
        title = panel["title"]
        data = panel["data"]
        cmap = panel["cmap"]
        norm = panel["norm"]
        vector = panel["vector"]
        if data is None:
            if mask is not None and mask_zg is not None:
                metal_count = _draw_mask_overlay(
                    ax,
                    mask_zg,
                    mask_rg,
                    mask_local,
                    alpha=0.88,
                    contour_width=scaled(0.85, args.paper_line_scale),
                )
            ax.text(
                0.02,
                0.05,
                f"metal cells in window = {metal_count}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9.5 * float(args.paper_font_scale),
                bbox={"facecolor": "white", "alpha": 0.74, "edgecolor": "none", "pad": 3},
            )
        else:
            image = ax.pcolormesh(zg, rg, data, shading="auto", cmap=cmap, norm=norm)
            cbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.01)
            cbar.ax.tick_params(labelsize=9.0 * float(args.paper_font_scale), width=scaled(0.8, args.paper_line_scale))
            if mask is not None and mask_zg is not None:
                _draw_mask_overlay(
                    ax,
                    mask_zg,
                    mask_rg,
                    mask_local,
                    alpha=float(args.mask_alpha),
                    contour_width=scaled(0.65, args.paper_line_scale),
                )
            if vector is not None:
                rf_vector_arrow_count = _draw_vector_direction_arrows(
                    ax,
                    zg,
                    rg,
                    vector[0],
                    vector[1],
                    stride_z=args.rf_vector_stride_z,
                    stride_r=args.rf_vector_stride_r,
                    length_mm=args.rf_vector_length_mm,
                    color=args.rf_vector_color,
                    alpha=args.rf_vector_alpha,
                    width=scaled(args.rf_vector_width, args.paper_line_scale),
                )

        capillary_exit_m = grid.get("capillary_exit_z_m")
        if capillary_exit_m is not None and bool(args.show_capillary_exit):
            ax.axvline(
                float(capillary_exit_m) * 1.0e3,
                color="white",
                linestyle="--",
                linewidth=scaled(1.25, args.paper_line_scale),
                alpha=0.95,
            )
        for reference_z_mm in args.reference_z_mm:
            ax.axvline(float(reference_z_mm), color="cyan", linestyle=":", linewidth=scaled(1.15, args.paper_line_scale), alpha=0.95)
        ax.set_title(title)
        ax.set_xlim(args.z_min_mm, args.z_max_mm)
        ax.set_ylim(args.r_min_mm, args.r_max_mm)
        ax.grid(color="white", alpha=0.17, linewidth=scaled(0.5, args.paper_line_scale))

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    add_subplot_labels(
        axes_flat[: len(panels)],
        enabled=bool(args.subplot_labels) and len(panels) > 1,
        font_scale=args.paper_font_scale,
        line_scale=args.paper_line_scale,
    )
    axes_grid = np.asarray(axes).reshape(nrows, ncols)
    for ax in axes_grid[:, 0]:
        ax.set_ylabel("r [mm]")
    for ax in axes_grid[-1, :]:
        ax.set_xlabel("z [mm]")

    title = args.title or "Electric field, gas flow, and electrode mask alignment"
    fig.suptitle(title)
    fig.savefig(output)
    plt.close(fig)

    summary = {
        "field": str(field_path),
        "electrode_mask": "" if mask_path is None else str(mask_path),
        "output": str(output),
        "z_window_mm": [float(args.z_min_mm), float(args.z_max_mm)],
        "r_window_mm": [float(args.r_min_mm), float(args.r_max_mm)],
        "plot_grid_shape": [int(r_idx.size), int(z_idx.size)],
        "capillary_exit_z_mm": None if grid.get("capillary_exit_z_m") is None else float(grid["capillary_exit_z_m"]) * 1.0e3,
        "reference_z_mm": [float(value) for value in args.reference_z_mm],
        "mask_loaded": mask is not None,
        "mask_metal_count_in_window": int(metal_count),
        "panels": panel_keys,
        "layout": str(args.layout),
        "layout_shape": [int(nrows), int(ncols)],
        "rf_peak_voltage_v": float(args.rf_peak_voltage_v),
        "rf_reference_peak_voltage_v": float(rf_reference_peak_voltage_v),
        "rf_scale": float(rf_scale),
        "rf_phase_deg": float(args.rf_phase_deg),
        "rf_phase_factor_cos": float(rf_phase_factor),
        "include_total_field_vector_panel": bool(args.include_total_field_vector_panel),
        "rf_vector_arrow_count": int(rf_vector_arrow_count),
        "mask_metadata": {} if mask is None else mask.metadata,
        "simion_dc_z_offset_mm": float(simion.get("dc_z_offset_m", 0.0)) * 1.0e3,
        "simion_rf_z_offset_mm": float(simion.get("rf_z_offset_m", 0.0)) * 1.0e3,
        "fluent_z_offset_mm": float(fluent.get("fluent_z_offset_m", 0.0)) * 1.0e3,
        "fluent_clipped_before_capillary_exit": bool(fluent.get("fluent_clipped_before_capillary_exit", False)),
        "fluent_clipped_after_z": bool(fluent.get("fluent_clipped_after_z", False)),
        "fluent_clip_z_max_mm": None
        if fluent.get("fluent_clip_z_max_m") is None
        else float(fluent["fluent_clip_z_max_m"]) * 1.0e3,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    print(f"Saved: {summary_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baked electric field, gas flow, and 2D electrode mask alignment.")
    parser.add_argument("--field", default=str(DEFAULT_FIELD_PATH), help="baked_fields.npy path.")
    parser.add_argument(
        "--electrode-mask",
        default="default",
        help="2D electrode mask .patxt/.npz path; use default/auto for the S-lens mask, none to disable.",
    )
    parser.add_argument("--electrode-mask-cache", default="", help="Optional .npz cache for the 2D electrode mask.")
    parser.add_argument(
        "--electrode-mask-z-offset-mm",
        type=float,
        default=None,
        help="Override mask z offset [mm]. Default uses baked field SIMION metadata.",
    )
    parser.add_argument("--output-dir", default="outputs/paper_figures", help="Output directory when --output is omitted.")
    parser.add_argument("--output", default="", help="Output PNG path.")
    parser.add_argument(
        "--panels",
        default="",
        help="Comma-separated panels to draw: mask,dc,rf,gas,total. Default is mask,dc,rf,gas; total is appended by --include-rf-vector-panel.",
    )
    parser.add_argument("--layout", default="auto", help="Figure layout: auto or rowsxcols such as 1x1, 1x2, 2x2, 2x3, 3x1.")
    parser.add_argument("--z-min-mm", type=float, default=3.8, help="Left edge of plotted z window [mm].")
    parser.add_argument("--z-max-mm", type=float, default=12.0, help="Right edge of plotted z window [mm].")
    parser.add_argument("--r-min-mm", type=float, default=0.0, help="Lower edge of plotted r window [mm].")
    parser.add_argument("--r-max-mm", type=float, default=8.0, help="Upper edge of plotted r window [mm].")
    parser.add_argument(
        "--global-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the complete baked-grid z/r coordinate range, overriding explicit window bounds.",
    )
    parser.add_argument(
        "--plot-max-z-points",
        type=int,
        default=1200,
        help="Maximum axial grid nodes used for display; 0 keeps all nodes.",
    )
    parser.add_argument(
        "--plot-max-r-points",
        type=int,
        default=600,
        help="Maximum radial grid nodes used for display; 0 keeps all nodes.",
    )
    parser.add_argument("--reference-z-mm", type=float, nargs="*", default=[5.0], help="Optional vertical reference lines [mm].")
    parser.add_argument("--show-capillary-exit", action=argparse.BooleanOptionalAction, default=True, help="Draw capillary_exit_z metadata line.")
    parser.add_argument("--mask-alpha", type=float, default=0.30, help="Mask overlay alpha in field/flow panels.")
    parser.add_argument("--rf-peak-voltage-v", type=float, default=100.0, help="Runtime RF peak voltage Vpeak used to scale the RF basis field [V].")
    parser.add_argument("--rf-phase-deg", type=float, default=0.0, help="RF phase used for the total-field direction panel [deg].")
    parser.add_argument(
        "--include-rf-vector-panel",
        "--include-total-field-vector-panel",
        dest="include_total_field_vector_panel",
        action="store_true",
        help="Add a panel with E_DC + E_RF_peak*cos(phase) direction arrows.",
    )
    parser.add_argument("--rf-vector-stride-z", type=int, default=35, help="RF vector arrow stride along z grid nodes.")
    parser.add_argument("--rf-vector-stride-r", type=int, default=18, help="RF vector arrow stride along r grid nodes.")
    parser.add_argument("--rf-vector-length-mm", type=float, default=0.25, help="Displayed RF vector arrow length [mm].")
    parser.add_argument("--rf-vector-color", default="white", help="RF vector arrow color.")
    parser.add_argument("--rf-vector-alpha", type=float, default=0.82, help="RF vector arrow alpha.")
    parser.add_argument("--rf-vector-width", type=float, default=0.0022, help="RF vector arrow shaft width.")
    parser.add_argument("--dc-cmap", default="magma", help="Matplotlib colormap for |E_DC|.")
    parser.add_argument("--rf-cmap", default="viridis", help="Matplotlib colormap for |E_RF|.")
    parser.add_argument("--gas-cmap", default="plasma", help="Matplotlib colormap for |v_gas|.")
    parser.add_argument(
        "--subplot-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label grouped subplots as (a), (b), ... for manuscript figures.",
    )
    parser.add_argument("--paper-font-scale", type=float, default=1.18, help="Global font-size multiplier for paper figures.")
    parser.add_argument("--paper-line-scale", type=float, default=1.25, help="Line-width multiplier for paper figures.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    parser.add_argument("--title", default="", help="Optional plot title.")
    return parser.parse_args()


def main() -> None:
    make_plot(parse_args())


if __name__ == "__main__":
    main()
