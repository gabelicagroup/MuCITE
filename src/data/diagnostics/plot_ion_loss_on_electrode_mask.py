"""Plot ion-loss terminal events on top of a 2D electrode mask.

The script reads one or more simulation case directories containing
``terminal_events.csv`` and overlays weighted loss markers on the axisymmetric
electrode mask. Loss events include ``electrode_hit``, ``radial_out``, and
``domain_out`` by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from ...env.boundaries.mask2d import ElectrodeMask
    from ..boundary_masks import load_electrode_mask
    from .plot_style import apply_paper_style, scaled, subplot_label
except ImportError:  # pragma: no cover - supports direct script execution.
    from src.data.boundary_masks import load_electrode_mask  # type: ignore
    from src.data.diagnostics.plot_style import (  # type: ignore
        apply_paper_style,
        scaled,
        subplot_label,
    )
    from src.env.boundaries.mask2d import ElectrodeMask  # type: ignore


DEFAULT_MASK_CANDIDATES = (
    Path("manuscript") / "slens_Efield_Fluent_baker" / "slens100x_gem.patxt",
    Path("E_field") / "slens100x_gem.patxt",
    Path("E_field") / "slens" / "slens100x_gem.patxt",
)

LOSS_STATUS_STYLES = {
    "electrode_hit": {"color": "#d7191c", "marker": "o", "label": "electrode hit"},
    "radial_out": {"color": "#d7191c", "marker": "o", "label": "radial out"},
    "domain_out": {"color": "#d7191c", "marker": "o", "label": "domain out"},
}
LOSS_STATUS_SEED_OFFSETS = {
    "electrode_hit": 11,
    "radial_out": 37,
    "domain_out": 71,
}
ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass(frozen=True)
class LossPoints:
    status: str
    z_mm: np.ndarray
    r_mm: np.ndarray
    weight: np.ndarray
    raw_count: int

    @property
    def total_weight(self) -> float:
        return float(np.nansum(self.weight))


@dataclass(frozen=True)
class CaseLossData:
    case_dir: Path
    label: str
    losses: dict[str, LossPoints]
    status_weights: dict[str, float]
    ion_current_a: float
    elapsed_s: float
    injected_real_ions: float
    electrode_loss_pct: float
    electrode_hit_z_p50_mm: float
    electrode_hit_r_p50_mm: float

    @property
    def total_loss_weight(self) -> float:
        return float(sum(points.total_weight for points in self.losses.values()))


def _default_mask_path() -> Path:
    for candidate in DEFAULT_MASK_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_MASK_CANDIDATES[0]


def _resolve_mask_path(raw_value: str) -> Optional[Path]:
    text = str(raw_value).strip()
    if text.lower() in {"", "none", "off"}:
        return None
    if text.lower() in {"default", "auto"}:
        return _default_mask_path()
    path = Path(text)
    if path.exists() or path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return path


def _parse_statuses(text: str) -> tuple[str, ...]:
    statuses = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not statuses:
        raise ValueError("At least one loss status must be requested.")
    return statuses


def _selected_statuses(args: argparse.Namespace) -> tuple[str, ...]:
    """Return loss statuses from explicit text or the show/hide switches."""

    if str(args.loss_statuses).strip():
        return _parse_statuses(args.loss_statuses)
    statuses: list[str] = []
    if args.show_electrode_hit:
        statuses.append("electrode_hit")
    if args.show_radial_out:
        statuses.append("radial_out")
    if args.show_domain_out:
        statuses.append("domain_out")
    if not statuses:
        raise ValueError("At least one loss channel must be enabled.")
    return tuple(statuses)


def _case_label(case_dir: Path) -> str:
    name = case_dir.name
    compact = (
        name.replace("G1_", "")
        .replace("G2_", "")
        .replace("G3_", "")
        .replace("_CCS1p7e-18_P1mbar_5ms_z61", "")
        .replace("_I2nA", "")
        .replace("_AMGpcg1e-6", "")
    )
    return compact.replace("_", "\n")


def _discover_case_dirs(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    for value in args.paths + args.case_dir:
        path = Path(value)
        if (path / "terminal_events.csv").exists():
            candidates.append(path)
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_dir() and (child / "terminal_events.csv").exists():
                    candidates.append(child)
        else:
            raise FileNotFoundError(f"Case path does not exist: {path}")
    for value in args.case_root:
        root = Path(value)
        if not root.is_dir():
            raise FileNotFoundError(f"Case root does not exist: {root}")
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "terminal_events.csv").exists():
                candidates.append(child)
    seen: set[Path] = set()
    cases: list[Path] = []
    for case in candidates:
        resolved = case.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        cases.append(resolved)
    if not cases:
        raise FileNotFoundError("No case directories with terminal_events.csv were found.")
    return cases


def _float_field(row: dict[str, str], field: str, default: float = math.nan) -> float:
    try:
        return float(row.get(field, default))
    except (TypeError, ValueError):
        return default


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return math.nan
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return float(np.interp(float(q) * cumulative[-1], cumulative, values))


def _read_command_text(case_dir: Path) -> str:
    command_path = case_dir / "command.ps1"
    if not command_path.exists():
        return ""
    return command_path.read_text(encoding="utf-8", errors="ignore")


def _command_arg_float(command_text: str, name: str) -> float:
    escaped = re.escape(name)
    patterns = (
        rf"'{escaped}'\s+'([^']+)'",
        rf'"{escaped}"\s+"([^"]+)"',
        rf"{escaped}\s+([^\s`]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, command_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return math.nan
    return math.nan


def _case_name_current_a(case_dir: Path) -> float:
    match = re.search(r"I(?P<value>\d+(?:p\d+)?)nA", case_dir.name, flags=re.IGNORECASE)
    if not match:
        return math.nan
    return float(match.group("value").replace("p", ".")) * 1.0e-9


def _snapshot_elapsed_s(case_dir: Path, macro_dt_s: float) -> float:
    index_path = case_dir / "snapshots_index.csv"
    if index_path.exists():
        last_time = math.nan
        with index_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                last_time = _float_field(row, "time_s", last_time)
        if math.isfinite(last_time):
            return last_time

    max_step = -1
    for snapshot_path in case_dir.glob("snapshot_macro_*.npy"):
        match = re.search(r"snapshot_macro_(\d+)\.npy$", snapshot_path.name)
        if match:
            max_step = max(max_step, int(match.group(1)))
    if max_step >= 0 and math.isfinite(macro_dt_s):
        return float(max_step) * float(macro_dt_s)
    return math.nan


def _format_float_compact(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "NA"
    if abs(value - round(value)) < 1.0e-9:
        return str(int(round(value)))
    text = f"{value:.{digits}g}"
    if "e" not in text.lower() and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _format_current_label(current_a: float, fallback_name: str) -> str:
    if not math.isfinite(current_a) or current_a <= 0.0:
        return fallback_name
    return f"{_format_float_compact(current_a * 1.0e9)} nA"


def _case_name_rf_v(case_dir: Path) -> float:
    match = re.search(r"RF(?P<value>\d+(?:p\d+)?)V", case_dir.name, flags=re.IGNORECASE)
    if not match:
        return math.nan
    return float(match.group("value").replace("p", "."))


def _case_name_model(case_dir: Path) -> str:
    name = case_dir.name.lower()
    if "explicit" in name:
        return "explicit"
    if "hybrid" in name:
        return "hybrid"
    return ""


def _case_variable_label(case: CaseLossData, mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "current":
        return _format_current_label(case.ion_current_a, _case_label(case.case_dir))
    if mode == "rf":
        rf_v = _case_name_rf_v(case.case_dir)
        return f"RF {_format_float_compact(rf_v)} V" if math.isfinite(rf_v) else _case_label(case.case_dir)
    if mode == "model":
        model = _case_name_model(case.case_dir)
        return model or _case_label(case.case_dir)
    return _case_label(case.case_dir).replace("\n", " ")


def _resolve_case_label_mode(cases: list[CaseLossData], requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested != "auto":
        return requested
    currents = {round(case.ion_current_a * 1.0e12, 6) for case in cases if math.isfinite(case.ion_current_a)}
    rf_values = {round(_case_name_rf_v(case.case_dir), 6) for case in cases if math.isfinite(_case_name_rf_v(case.case_dir))}
    models = {_case_name_model(case.case_dir) for case in cases if _case_name_model(case.case_dir)}
    if len(currents) > 1:
        return "current"
    if len(rf_values) > 1:
        return "rf"
    if len(models) > 1:
        return "model"
    return "name"


def _panel_title(case: CaseLossData, args: argparse.Namespace, panel_index: int) -> str:
    if getattr(args, "_manual_case_labels", False):
        variation = case.label.replace("\n", " ")
    else:
        variation = _case_variable_label(case, getattr(args, "_resolved_case_label_mode", args.case_label_mode))
    prefix = f"{subplot_label(panel_index)} " if args.subplot_labels else ""
    if math.isfinite(case.electrode_loss_pct):
        return f"{prefix}{variation}: electrode loss {case.electrode_loss_pct:.1f}%"
    return f"{prefix}{variation}: electrode loss NA"


def _panel_detail_text(case: CaseLossData) -> str:
    if not (math.isfinite(case.electrode_hit_z_p50_mm) and math.isfinite(case.electrode_hit_r_p50_mm)):
        return ""
    return f"hit z P50={case.electrode_hit_z_p50_mm:.2f} mm, r P50={case.electrode_hit_r_p50_mm:.2f} mm"


def _load_case_loss_data(
    case_dir: Path,
    *,
    statuses: tuple[str, ...],
    weight_column: str,
    label: Optional[str],
) -> CaseLossData:
    terminal_path = case_dir / "terminal_events.csv"
    if not terminal_path.exists():
        raise FileNotFoundError(f"Missing terminal_events.csv: {terminal_path}")

    command_text = _read_command_text(case_dir)
    ion_current_a = _command_arg_float(command_text, "--ion-current-a")
    if not math.isfinite(ion_current_a):
        ion_current_a = _case_name_current_a(case_dir)
    macro_dt_s = _command_arg_float(command_text, "--macro-time-step")
    if not math.isfinite(macro_dt_s):
        macro_dt_s = 5.0e-8
    total_time_s = _command_arg_float(command_text, "--total-time")

    buffers: dict[str, dict[str, list[float]]] = {
        status: {"z_mm": [], "r_mm": [], "weight": []} for status in statuses
    }
    raw_counts = {status: 0 for status in statuses}
    status_weights: dict[str, float] = {}
    hit_z_mm: list[float] = []
    hit_r_mm: list[float] = []
    hit_weight: list[float] = []
    max_event_time_s = math.nan
    with terminal_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"status", "z_m", "r_m"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{terminal_path} is missing required columns: {sorted(missing)}")
        for row in reader:
            status = row.get("status", "")
            z_mm = _float_field(row, "z_m") * 1.0e3
            r_mm = _float_field(row, "r_m") * 1.0e3
            weight = _float_field(row, weight_column, 1.0)
            if not math.isfinite(weight) or weight <= 0.0:
                weight = 1.0
            event_time_s = _float_field(row, "event_time_s")
            if math.isfinite(event_time_s):
                max_event_time_s = event_time_s if not math.isfinite(max_event_time_s) else max(max_event_time_s, event_time_s)
            status_weights[status] = status_weights.get(status, 0.0) + weight
            if not (math.isfinite(z_mm) and math.isfinite(r_mm)):
                continue
            if status == "electrode_hit":
                hit_z_mm.append(z_mm)
                hit_r_mm.append(r_mm)
                hit_weight.append(weight)
            if status not in buffers:
                continue
            raw_counts[status] += 1
            buffers[status]["z_mm"].append(z_mm)
            buffers[status]["r_mm"].append(r_mm)
            buffers[status]["weight"].append(weight)

    losses: dict[str, LossPoints] = {}
    for status, values in buffers.items():
        losses[status] = LossPoints(
            status=status,
            z_mm=np.asarray(values["z_mm"], dtype=float),
            r_mm=np.asarray(values["r_mm"], dtype=float),
            weight=np.asarray(values["weight"], dtype=float),
            raw_count=raw_counts[status],
        )
    snapshot_elapsed_s = _snapshot_elapsed_s(case_dir, macro_dt_s)
    elapsed_candidates = [value for value in (snapshot_elapsed_s, max_event_time_s) if math.isfinite(value)]
    if not elapsed_candidates and math.isfinite(total_time_s):
        elapsed_candidates.append(total_time_s)
    elapsed_s = max(elapsed_candidates) if elapsed_candidates else math.nan
    injected_real_ions = (
        ion_current_a * elapsed_s / ELEMENTARY_CHARGE_C
        if math.isfinite(ion_current_a) and ion_current_a > 0.0 and math.isfinite(elapsed_s) and elapsed_s > 0.0
        else math.nan
    )
    electrode_hit_weight = status_weights.get("electrode_hit", 0.0)
    denominator = injected_real_ions if math.isfinite(injected_real_ions) and injected_real_ions > 0.0 else sum(status_weights.values())
    electrode_loss_pct = electrode_hit_weight / denominator * 100.0 if denominator > 0.0 else math.nan
    hit_z_array = np.asarray(hit_z_mm, dtype=float)
    hit_r_array = np.asarray(hit_r_mm, dtype=float)
    hit_weight_array = np.asarray(hit_weight, dtype=float)
    return CaseLossData(
        case_dir=case_dir,
        label=label or _case_label(case_dir),
        losses=losses,
        status_weights=status_weights,
        ion_current_a=ion_current_a,
        elapsed_s=elapsed_s,
        injected_real_ions=injected_real_ions,
        electrode_loss_pct=electrode_loss_pct,
        electrode_hit_z_p50_mm=_weighted_quantile(hit_z_array, hit_weight_array, 0.5),
        electrode_hit_r_p50_mm=_weighted_quantile(hit_r_array, hit_weight_array, 0.5),
    )


def _parse_layout(layout: str, count: int) -> tuple[int, int]:
    text = str(layout).strip().lower()
    if text in {"", "auto"}:
        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        return rows, cols
    parts = text.split("x")
    if len(parts) != 2:
        raise ValueError("--layout must be auto or ROWSxCOLS, for example 2x3.")
    rows = int(parts[0])
    cols = int(parts[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("--layout rows and columns must be positive.")
    if rows * cols < count:
        raise ValueError(f"--layout {layout} has only {rows * cols} panels for {count} cases.")
    return rows, cols


def _mask_local_window(
    mask: ElectrodeMask,
    *,
    z_min_mm: float,
    z_max_mm: float,
    r_min_mm: float,
    r_max_mm: float,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    z_mm = mask.z_coords_m * 1.0e3
    r_mm = mask.r_coords_m * 1.0e3
    z_indices = np.flatnonzero((z_mm >= z_min_mm) & (z_mm <= z_max_mm))
    r_indices = np.flatnonzero((r_mm >= r_min_mm) & (r_mm <= r_max_mm))
    if z_indices.size == 0 or r_indices.size == 0:
        return None
    local = mask.metal_mask[np.ix_(r_indices, z_indices)].astype(float)
    zg, rg = np.meshgrid(z_mm[z_indices], r_mm[r_indices])
    return zg, rg, local


def _axis_limits(
    cases: list[CaseLossData],
    mask: ElectrodeMask,
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    z_values: list[np.ndarray] = [mask.z_coords_m * 1.0e3]
    r_values: list[np.ndarray] = [mask.r_coords_m * 1.0e3]
    for case in cases:
        for points in case.losses.values():
            if points.z_mm.size:
                z_values.append(points.z_mm)
                r_values.append(points.r_mm)
    z_all = np.concatenate(z_values)
    r_all = np.concatenate(r_values)
    z_min = float(args.z_min_mm) if args.z_min_mm is not None else float(np.nanmin(z_all))
    z_max = float(args.z_max_mm) if args.z_max_mm is not None else float(np.nanmax(z_all))
    r_min = float(args.r_min_mm) if args.r_min_mm is not None else max(0.0, float(np.nanmin(r_all)))
    r_max = float(args.r_max_mm) if args.r_max_mm is not None else float(np.nanmax(r_all))
    pad = max(float(args.auto_window_pad_mm), 0.0)
    if args.z_min_mm is None:
        z_min -= pad
    if args.z_max_mm is None:
        z_max += pad
    if args.r_min_mm is None:
        r_min = max(0.0, r_min - pad)
    if args.r_max_mm is None:
        r_max += pad
    if z_max <= z_min or r_max <= r_min:
        raise ValueError("Invalid axis limits.")
    return z_min, z_max, r_min, r_max


def _weighted_sample_indices(points: LossPoints, max_points: int, seed: int) -> np.ndarray:
    count = points.z_mm.size
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=int)
    weights = np.asarray(points.weight, dtype=float)
    if np.all(np.isfinite(weights)) and np.nansum(weights) > 0.0:
        prob = np.clip(weights, 0.0, None)
        prob = prob / float(np.sum(prob))
    else:
        prob = None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_points, replace=False, p=prob))


def _weighted_sample_arrays(
    z_mm: np.ndarray,
    r_mm: np.ndarray,
    weight: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(z_mm.size)
    if max_points <= 0 or count <= max_points:
        return z_mm, r_mm, weight
    weights = np.asarray(weight, dtype=float)
    if np.all(np.isfinite(weights)) and np.nansum(weights) > 0.0:
        prob = np.clip(weights, 0.0, None)
        prob = prob / float(np.sum(prob))
    else:
        prob = None
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(count, size=int(max_points), replace=False, p=prob))
    return z_mm[indices], r_mm[indices], weight[indices]


def _aggregate_loss_points(
    points: LossPoints,
    *,
    z_origin_mm: float,
    r_origin_mm: float,
    z_bin_mm: float,
    r_bin_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate loss events into local z-r bins using represented-real-ion weights."""

    if points.z_mm.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, 0
    z_bin_mm = float(z_bin_mm)
    r_bin_mm = float(r_bin_mm)
    if z_bin_mm <= 0.0 or r_bin_mm <= 0.0:
        return points.z_mm, points.r_mm, points.weight, int(points.z_mm.size)

    weights = np.asarray(points.weight, dtype=float)
    valid = (
        np.isfinite(points.z_mm)
        & np.isfinite(points.r_mm)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, 0

    z_valid = points.z_mm[valid]
    r_valid = points.r_mm[valid]
    w_valid = weights[valid]
    iz = np.floor((z_valid - float(z_origin_mm)) / z_bin_mm).astype(np.int64)
    ir = np.floor((r_valid - float(r_origin_mm)) / r_bin_mm).astype(np.int64)
    nz = int(np.max(iz)) + 1
    linear = ir * nz + iz
    unique, inverse = np.unique(linear, return_inverse=True)
    weight_sum = np.bincount(inverse, weights=w_valid)
    z_sum = np.bincount(inverse, weights=z_valid * w_valid)
    r_sum = np.bincount(inverse, weights=r_valid * w_valid)
    nonzero = weight_sum > 0.0
    z_agg = z_sum[nonzero] / weight_sum[nonzero]
    r_agg = r_sum[nonzero] / weight_sum[nonzero]
    w_agg = weight_sum[nonzero]
    order = np.lexsort((z_agg, r_agg))
    return z_agg[order], r_agg[order], w_agg[order], int(unique.size)


def _marker_sizes(weights: np.ndarray, *, min_size: float, max_size: float) -> np.ndarray:
    if weights.size == 0:
        return np.asarray([], dtype=float)
    finite = weights[np.isfinite(weights) & (weights > 0.0)]
    if finite.size == 0:
        return np.full(weights.shape, min_size, dtype=float)
    low = float(np.nanpercentile(finite, 5.0))
    high = float(np.nanpercentile(finite, 99.0))
    if high <= low:
        high = low + 1.0
    scaled_weight = np.clip((weights - low) / (high - low), 0.0, 1.0)
    return min_size + scaled_weight * (max_size - min_size)


def _draw_mask(ax, mask: ElectrodeMask, limits: tuple[float, float, float, float], args: argparse.Namespace) -> None:
    z_min, z_max, r_min, r_max = limits
    mesh = _mask_local_window(mask, z_min_mm=z_min, z_max_mm=z_max, r_min_mm=r_min, r_max_mm=r_max)
    if mesh is None:
        return
    zg, rg, local = mesh
    if np.count_nonzero(local) == 0:
        return
    ax.contourf(zg, rg, local, levels=[0.5, 1.5], colors=[args.mask_color], alpha=float(args.mask_alpha))
    ax.contour(
        zg,
        rg,
        local,
        levels=[0.5],
        colors=args.mask_edge_color,
        linewidths=scaled(args.mask_edge_width, args.line_scale),
        alpha=float(args.mask_edge_alpha),
    )


def _draw_case(
    ax,
    case: CaseLossData,
    mask: ElectrodeMask,
    limits: tuple[float, float, float, float],
    args: argparse.Namespace,
    *,
    panel_index: int,
) -> dict[str, dict[str, float]]:
    _draw_mask(ax, mask, limits, args)
    z_min, z_max, r_min, r_max = limits
    plotted: dict[str, dict[str, float]] = {}
    total_loss = case.total_loss_weight
    for status, points in case.losses.items():
        if points.z_mm.size == 0:
            plotted[status] = {"plotted_points": 0, "window_weight": 0.0}
            continue
        in_window = (
            (points.z_mm >= z_min)
            & (points.z_mm <= z_max)
            & (points.r_mm >= r_min)
            & (points.r_mm <= r_max)
        )
        window_points = LossPoints(
            status=status,
            z_mm=points.z_mm[in_window],
            r_mm=points.r_mm[in_window],
            weight=points.weight[in_window],
            raw_count=int(np.count_nonzero(in_window)),
        )
        if window_points.z_mm.size == 0:
            plotted[status] = {"plotted_points": 0, "window_weight": 0.0}
            continue
        seed = int(args.random_seed) + panel_index * 1009 + LOSS_STATUS_SEED_OFFSETS.get(status, 509)
        if args.aggregate_loss_bins:
            plot_z_mm, plot_r_mm, plot_weight, aggregated_bins = _aggregate_loss_points(
                window_points,
                z_origin_mm=z_min,
                r_origin_mm=r_min,
                z_bin_mm=float(args.loss_bin_z_mm),
                r_bin_mm=float(args.loss_bin_r_mm),
            )
            plot_z_mm, plot_r_mm, plot_weight = _weighted_sample_arrays(
                plot_z_mm,
                plot_r_mm,
                plot_weight,
                int(args.max_points_per_status),
                seed=seed,
            )
        else:
            sample = _weighted_sample_indices(window_points, int(args.max_points_per_status), seed=seed)
            plot_z_mm = window_points.z_mm[sample]
            plot_r_mm = window_points.r_mm[sample]
            plot_weight = window_points.weight[sample]
            aggregated_bins = int(plot_z_mm.size)
        style = LOSS_STATUS_STYLES.get(status, {"color": "#333333", "marker": "x", "label": status})
        sizes = _marker_sizes(
            plot_weight,
            min_size=float(args.marker_min_size),
            max_size=float(args.marker_max_size),
        )
        ax.scatter(
            plot_z_mm,
            plot_r_mm,
            s=sizes,
            marker=style["marker"],
            c=style["color"] if args.color_by_status else args.loss_marker_color,
            alpha=float(args.marker_alpha),
            linewidths=scaled(args.marker_edge_width, args.line_scale),
            edgecolors=args.marker_edge_color,
            label=style["label"],
            zorder=10,
        )
        plotted[status] = {
            "plotted_points": float(plot_z_mm.size),
            "aggregated_bins": float(aggregated_bins),
            "window_weight": float(np.nansum(window_points.weight)),
            "status_weight": points.total_weight,
            "loss_fraction_pct": points.total_weight / total_loss * 100.0 if total_loss > 0.0 else math.nan,
        }
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(r_min, r_max)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.grid(True, color="#d9d9d9", alpha=0.35, linewidth=scaled(0.45, args.line_scale))
    if args.panel_titles:
        ax.set_title(
            _panel_title(case, args, panel_index),
            loc="left",
            pad=float(args.panel_title_pad),
            fontweight="bold",
        )
    if args.panel_detail_line:
        detail_text = _panel_detail_text(case)
        if detail_text:
            ax.text(
                float(args.panel_detail_x),
                float(args.panel_detail_y),
                detail_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0 * float(args.font_scale),
                color="#202020",
                zorder=20,
            )
    if args.annotate_counts:
        text_lines = []
        for status, points in case.losses.items():
            if points.total_weight <= 0.0:
                continue
            label = LOSS_STATUS_STYLES.get(status, {}).get("label", status)
            fraction = points.total_weight / total_loss * 100.0 if total_loss > 0.0 else math.nan
            text_lines.append(f"{label}: {points.total_weight:.3g} ({fraction:.1f}%)")
        if text_lines:
            ax.text(
                0.985,
                0.965,
                "\n".join(text_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.0 * float(args.font_scale),
                color="#202020",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bdbdbd", alpha=0.82),
                zorder=20,
            )
    return plotted


def _write_summary(
    path: Path,
    cases: list[CaseLossData],
    plotted: list[dict[str, dict[str, float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "status",
                "raw_events",
                "status_weight",
                "total_loss_weight",
                "loss_fraction_pct",
                "electrode_loss_pct",
                "ion_current_a",
                "elapsed_s",
                "injected_real_ions",
                "electrode_hit_z_p50_mm",
                "electrode_hit_r_p50_mm",
                "window_weight",
                "aggregated_bins",
                "plotted_points",
            ]
        )
        for case, plotted_case in zip(cases, plotted):
            total_loss = case.total_loss_weight
            for status, points in case.losses.items():
                info = plotted_case.get(status, {})
                writer.writerow(
                    [
                        str(case.case_dir),
                        status,
                        points.raw_count,
                        points.total_weight,
                        total_loss,
                        points.total_weight / total_loss * 100.0 if total_loss > 0.0 else math.nan,
                        case.electrode_loss_pct,
                        case.ion_current_a,
                        case.elapsed_s,
                        case.injected_real_ions,
                        case.electrode_hit_z_p50_mm,
                        case.electrode_hit_r_p50_mm,
                        info.get("window_weight", 0.0),
                        int(info.get("aggregated_bins", 0)),
                        int(info.get("plotted_points", 0)),
                    ]
                )


def make_plot(args: argparse.Namespace) -> dict[str, Path]:
    statuses = _selected_statuses(args)
    case_dirs = _discover_case_dirs(args)
    labels = list(args.case_label)
    if labels and len(labels) != len(case_dirs):
        raise ValueError("--case-label count must match the number of discovered cases.")
    cases = [
        _load_case_loss_data(
            case_dir,
            statuses=statuses,
            weight_column=args.weight_column,
            label=labels[index] if labels else None,
        )
        for index, case_dir in enumerate(case_dirs)
    ]
    args._manual_case_labels = bool(labels)
    args._resolved_case_label_mode = _resolve_case_label_mode(cases, args.case_label_mode)

    mask_path = _resolve_mask_path(args.electrode_mask)
    if mask_path is None:
        raise ValueError("--electrode-mask cannot be none for this plot.")
    mask_cache = Path(args.electrode_mask_cache) if args.electrode_mask_cache else None
    z_offset_m = None if args.electrode_mask_z_offset_mm is None else float(args.electrode_mask_z_offset_mm) * 1.0e-3
    mask = load_electrode_mask(source_path=mask_path, cache_path=mask_cache, z_offset_m=z_offset_m, project_root=Path.cwd())
    if mask is None:
        raise RuntimeError("Electrode mask did not load.")

    limits = _axis_limits(cases, mask, args)
    rows, cols = _parse_layout(args.layout, len(cases))
    apply_paper_style(font_scale=float(args.font_scale), line_scale=float(args.line_scale))
    fig_width = float(args.panel_width_in) * cols
    fig_height = float(args.panel_height_in) * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False, sharex=args.share_axes, sharey=args.share_axes)
    plotted: list[dict[str, dict[str, float]]] = []
    for index, case in enumerate(cases):
        ax = axes.ravel()[index]
        plotted.append(_draw_case(ax, case, mask, limits, args, panel_index=index))
        if args.hide_inner_xlabels and index // cols < rows - 1:
            ax.set_xlabel("")
    for ax in axes.ravel()[len(cases) :]:
        ax.set_visible(False)

    if args.legend:
        if args.color_by_status:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker=LOSS_STATUS_STYLES.get(status, {}).get("marker", "x"),
                    color="none",
                    markerfacecolor=LOSS_STATUS_STYLES.get(status, {}).get("color", "#333333"),
                    markeredgecolor=args.marker_edge_color,
                    markersize=10.5,
                    label=LOSS_STATUS_STYLES.get(status, {}).get("label", status),
                )
                for status in statuses
            ]
        else:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=args.loss_marker_color,
                    markeredgecolor=args.marker_edge_color,
                    markersize=6.5,
                    label=args.small_marker_label,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=args.loss_marker_color,
                    markeredgecolor=args.marker_edge_color,
                    markersize=13.5,
                    label=args.large_marker_label,
                ),
            ]
        handles.insert(
            0,
            Line2D(
                [0],
                [0],
                color=args.mask_edge_color,
                linewidth=scaled(args.mask_edge_width, args.line_scale),
                label="electrode mask",
            ),
        )
        legend_columns = int(args.legend_columns) if int(args.legend_columns) > 0 else min(len(handles), 4)
        fig.legend(handles=handles, loc=args.legend_loc, frameon=False, ncols=legend_columns)
    if args.title:
        fig.suptitle(args.title)
    bottom = float(args.bottom_margin)
    top = float(args.top_margin)
    if args.legend and "lower" in str(args.legend_loc).lower():
        bottom = max(bottom, 0.08)
    if args.legend and "upper" in str(args.legend_loc).lower():
        top = min(top, 0.93)
    if args.title:
        top = min(top, 0.93)
    fig.tight_layout(rect=(float(args.left_margin), bottom, float(args.right_margin), top))
    fig.subplots_adjust(wspace=float(args.subplot_wspace), hspace=float(args.subplot_hspace))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(args.dpi))
    plt.close(fig)

    summary_output = Path(args.summary_output) if args.summary_output else output.with_name(output.stem + "_summary.csv")
    _write_summary(summary_output, cases, plotted)
    return {"figure": output, "summary": summary_output}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay ion-loss terminal events from one or more cases on a 2D electrode mask."
    )
    parser.add_argument("paths", nargs="*", help="Case directories or roots containing case subdirectories.")
    parser.add_argument("--case-dir", action="append", default=[], help="Case directory. May be repeated.")
    parser.add_argument("--case-root", action="append", default=[], help="Root directory whose child cases contain terminal_events.csv.")
    parser.add_argument("--case-label", action="append", default=[], help="Panel label. Repeat once per discovered case.")
    parser.add_argument(
        "--case-label-mode",
        choices=("auto", "current", "rf", "model", "name"),
        default="auto",
        help="Case variation shown in the upper-left panel title.",
    )
    parser.add_argument("--electrode-mask", default="default", help="2D electrode mask .patxt/.npz path, or default/auto.")
    parser.add_argument("--electrode-mask-cache", default="", help="Optional .npz electrode mask cache.")
    parser.add_argument("--electrode-mask-z-offset-mm", type=float, default=None, help="Override mask z offset in mm.")
    parser.add_argument(
        "--loss-statuses",
        default="",
        help=(
            "Comma-separated terminal statuses to plot. If set, this overrides "
            "--show-electrode-hit/--show-radial-out/--show-domain-out."
        ),
    )
    parser.add_argument("--show-electrode-hit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-radial-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-domain-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-column", default="represented_real_ions", help="Terminal-event weight column.")
    parser.add_argument(
        "--aggregate-loss-bins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggregate terminal events in local z-r bins before drawing weighted markers.",
    )
    parser.add_argument("--loss-bin-z-mm", type=float, default=0.02, help="z bin size for aggregated weighted loss markers.")
    parser.add_argument("--loss-bin-r-mm", type=float, default=0.05, help="r bin size for aggregated weighted loss markers.")
    parser.add_argument("--output", required=True, help="Output PNG/PDF/SVG path.")
    parser.add_argument("--summary-output", default="", help="Optional summary CSV path. Defaults next to the figure.")
    parser.add_argument("--layout", default="auto", help="Subplot layout: auto or ROWSxCOLS, for example 2x3.")
    parser.add_argument("--z-min-mm", type=float, default=None)
    parser.add_argument("--z-max-mm", type=float, default=None)
    parser.add_argument("--r-min-mm", type=float, default=None)
    parser.add_argument("--r-max-mm", type=float, default=None)
    parser.add_argument("--auto-window-pad-mm", type=float, default=0.2)
    parser.add_argument("--max-points-per-status", type=int, default=8000, help="Weighted downsample per status per case; <=0 disables.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--loss-marker-color", default="#d7191c", help="Default marker color for all loss channels.")
    parser.add_argument("--color-by-status", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--marker-min-size", type=float, default=42.0)
    parser.add_argument("--marker-max-size", type=float, default=260.0)
    parser.add_argument("--marker-alpha", type=float, default=0.82)
    parser.add_argument("--marker-edge-color", default="#7f0000")
    parser.add_argument("--marker-edge-width", type=float, default=0.45)
    parser.add_argument("--mask-color", default="#bdbdbd")
    parser.add_argument("--mask-alpha", type=float, default=0.55)
    parser.add_argument("--mask-edge-color", default="#4d4d4d")
    parser.add_argument("--mask-edge-alpha", type=float, default=0.85)
    parser.add_argument("--mask-edge-width", type=float, default=0.8)
    parser.add_argument("--panel-width-in", type=float, default=5.8)
    parser.add_argument("--panel-height-in", type=float, default=3.5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-scale", type=float, default=1.25)
    parser.add_argument("--line-scale", type=float, default=1.35)
    parser.add_argument("--title", default="", help="Optional figure-level title. Default: no title.")
    parser.add_argument("--legend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--legend-loc", default="lower center")
    parser.add_argument("--legend-columns", type=int, default=0, help="Legend column count; <=0 uses automatic layout.")
    parser.add_argument("--small-marker-label", default="smaller weighted loss")
    parser.add_argument("--large-marker-label", default="larger weighted loss")
    parser.add_argument("--panel-title-pad", type=float, default=6.0)
    parser.add_argument("--panel-detail-line", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--panel-detail-x", type=float, default=0.01)
    parser.add_argument("--panel-detail-y", type=float, default=0.965)
    parser.add_argument(
        "--hide-inner-xlabels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide x-axis labels on rows above the bottom row.",
    )
    parser.add_argument("--left-margin", type=float, default=0.035)
    parser.add_argument("--right-margin", type=float, default=0.995)
    parser.add_argument("--top-margin", type=float, default=0.985)
    parser.add_argument("--bottom-margin", type=float, default=0.035)
    parser.add_argument("--subplot-wspace", type=float, default=0.12)
    parser.add_argument("--subplot-hspace", type=float, default=0.20)
    parser.add_argument("--annotate-counts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--panel-titles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--share-axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subplot-labels", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    outputs = make_plot(parse_args())
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
