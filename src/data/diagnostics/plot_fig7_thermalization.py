"""Build manuscript Fig. 7 from partial thermalization snapshot data."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from .plot_style import add_subplot_labels, apply_paper_style, scaled
except ImportError:  # pragma: no cover - supports direct script execution.
    from plot_style import add_subplot_labels, apply_paper_style, scaled  # type: ignore


DEFAULT_ROOT = Path("manuscript") / "unfinisheddata" / "gaussian_tube_source_65mm_current_sweep_20260518_093601"
DEFAULT_CASES = ("I1nA_hybrid_30ms", "I2nA_hybrid_30ms", "I5nA_hybrid_30ms", "I2nA_explicit_15ms")


@dataclass(frozen=True)
class CaseData:
    name: str
    path: Path
    current_na: float
    model: str
    macro_step: int
    time_ms: float
    rows: int
    active_real_ions: float
    r_p50_mm: float
    r_p95_mm: float
    z_p50_mm: float
    z_p95_mm: float
    ke_p50_ev: float
    ke_p95_ev: float
    teff_p50_k: float
    teff_p95_k: float
    collision_p50: float
    collision_p95: float
    data: np.ndarray


def _load_columns(case_dir: Path) -> list[str]:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing snapshot metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return list(metadata["columns"])


def _parse_ps_args(command_path: Path) -> dict[str, str]:
    if not command_path.exists():
        return {}
    text = command_path.read_text(encoding="utf-8")
    tokens = shlex.split(text.replace("`", ""), posix=False)
    parsed: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index].strip("'")
        if token.startswith("--"):
            if index + 1 < len(tokens) and not tokens[index + 1].strip("'").startswith("--"):
                parsed[token] = tokens[index + 1].strip("'")
                index += 2
            else:
                parsed[token] = "true"
                index += 1
        else:
            index += 1
    return parsed


def _current_from_name(name: str) -> float:
    match = re.search(r"I([0-9]+(?:p[0-9]+)?)nA", name)
    if match is None:
        return float("nan")
    return float(match.group(1).replace("p", "."))


def _model_from_name(name: str) -> str:
    lower = name.lower()
    if "explicit" in lower:
        return "explicit"
    if "hybrid" in lower:
        return "hybrid-langevin"
    return "unknown"


def _snapshot_macros(case_dir: Path) -> dict[int, Path]:
    snapshots: dict[int, Path] = {}
    for path in case_dir.glob("snapshot_macro_*.npy"):
        match = re.match(r"snapshot_macro_(\d+)\.npy$", path.name)
        if match is not None:
            snapshots[int(match.group(1))] = path
    return dict(sorted(snapshots.items()))


def _common_macro(case_dirs: list[Path], requested_macro: int | None) -> int:
    macro_sets = [set(_snapshot_macros(case_dir)) for case_dir in case_dirs]
    if not macro_sets or any(not macros for macros in macro_sets):
        raise FileNotFoundError("At least one case has no snapshot_macro_*.npy files.")
    common = set.intersection(*macro_sets)
    if not common:
        latest = min(max(macros) for macros in macro_sets)
        raise ValueError(f"No exact common snapshot macro across cases. Latest shared upper bound is {latest}.")
    if requested_macro is None:
        return max(common)
    return min(common, key=lambda value: abs(value - requested_macro))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(percentile) / 100.0 * cumulative[-1]
    return float(values[np.searchsorted(cumulative, target, side="left")])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _load_case(case_dir: Path, macro_step: int, macro_dt_s: float) -> CaseData:
    columns = _load_columns(case_dir)
    index = {column: i for i, column in enumerate(columns)}
    required = ("r_m", "z_m", "teff_k", "kinetic_energy_ev", "represented_real_ions", "collision_count_per_ion")
    missing = [column for column in required if column not in index]
    if missing:
        raise KeyError(f"{case_dir} metadata is missing columns: {missing}")

    ps_args = _parse_ps_args(case_dir / "command.ps1")
    current_na = float(ps_args.get("--ion-current-a", "nan")) * 1.0e9
    if not np.isfinite(current_na):
        current_na = _current_from_name(case_dir.name)
    model = ps_args.get("--collision-model", _model_from_name(case_dir.name))
    snapshot = case_dir / f"snapshot_macro_{macro_step:05d}.npy"
    if not snapshot.exists():
        snapshot = case_dir / f"snapshot_macro_{macro_step}.npy"
    if not snapshot.exists():
        raise FileNotFoundError(f"Missing selected snapshot: {snapshot}")

    data = np.asarray(np.load(snapshot), dtype=np.float64)
    rows = int(data.shape[0])
    weights = data[:, index["represented_real_ions"]] if rows else np.empty(0)
    return CaseData(
        name=case_dir.name,
        path=case_dir,
        current_na=float(current_na),
        model=str(model),
        macro_step=int(macro_step),
        time_ms=float(macro_step) * float(macro_dt_s) * 1.0e3,
        rows=rows,
        active_real_ions=float(np.nansum(weights)) if rows else 0.0,
        r_p50_mm=_weighted_percentile(data[:, index["r_m"]] * 1.0e3, weights, 50.0),
        r_p95_mm=_weighted_percentile(data[:, index["r_m"]] * 1.0e3, weights, 95.0),
        z_p50_mm=_weighted_percentile(data[:, index["z_m"]] * 1.0e3, weights, 50.0),
        z_p95_mm=_weighted_percentile(data[:, index["z_m"]] * 1.0e3, weights, 95.0),
        ke_p50_ev=_weighted_percentile(data[:, index["kinetic_energy_ev"]], weights, 50.0),
        ke_p95_ev=_weighted_percentile(data[:, index["kinetic_energy_ev"]], weights, 95.0),
        teff_p50_k=_weighted_percentile(data[:, index["teff_k"]], weights, 50.0),
        teff_p95_k=_weighted_percentile(data[:, index["teff_k"]], weights, 95.0),
        collision_p50=_weighted_percentile(data[:, index["collision_count_per_ion"]], weights, 50.0),
        collision_p95=_weighted_percentile(data[:, index["collision_count_per_ion"]], weights, 95.0),
        data=data,
    )


def _case_rows(cases: list[CaseData]) -> list[dict[str, object]]:
    return [
        {
            "case": case.name,
            "current_na": case.current_na,
            "collision_model": case.model,
            "macro_step": case.macro_step,
            "time_ms": case.time_ms,
            "snapshot_rows": case.rows,
            "active_real_ions": case.active_real_ions,
            "r_p50_mm": case.r_p50_mm,
            "r_p95_mm": case.r_p95_mm,
            "z_p50_mm": case.z_p50_mm,
            "z_p95_mm": case.z_p95_mm,
            "ke_p50_ev": case.ke_p50_ev,
            "ke_p95_ev": case.ke_p95_ev,
            "teff_p50_k": case.teff_p50_k,
            "teff_p95_k": case.teff_p95_k,
            "collision_p50": case.collision_p50,
            "collision_p95": case.collision_p95,
        }
        for case in cases
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _hybrid_cases(cases: list[CaseData]) -> list[CaseData]:
    return sorted([case for case in cases if "hybrid" in case.model.lower()], key=lambda item: item.current_na)


def _case_at(cases: list[CaseData], current_na: float, model_key: str) -> CaseData | None:
    for case in cases:
        if abs(case.current_na - current_na) < 1.0e-9 and model_key.lower() in case.model.lower():
            return case
    return None


def _draw_temperature_vs_current(ax, cases: list[CaseData], args: argparse.Namespace) -> None:
    hybrid = _hybrid_cases(cases)
    current = np.asarray([case.current_na for case in hybrid], dtype=np.float64)
    p50 = np.asarray([case.teff_p50_k for case in hybrid], dtype=np.float64)
    p95 = np.asarray([case.teff_p95_k for case in hybrid], dtype=np.float64)
    ax.plot(current, p50, marker="o", color="#0b559f", linewidth=scaled(1.8, args.paper_line_scale), label="hybrid p50")
    ax.plot(current, p95, marker="s", color="#b2182b", linewidth=scaled(1.8, args.paper_line_scale), label="hybrid p95")
    ax.fill_between(current, p50, p95, color="#f4a582", alpha=0.22, linewidth=0)
    explicit = _case_at(cases, 2.0, "explicit")
    if explicit is not None:
        ax.scatter(
            [explicit.current_na],
            [explicit.teff_p95_k],
            marker="D",
            s=54 * float(args.paper_font_scale),
            facecolor="white",
            edgecolor="#b2182b",
            linewidth=scaled(1.1, args.paper_line_scale),
            label="explicit p95 @ 2 nA",
            zorder=5,
        )
    ax.set_xlabel("ion current [nA]")
    ax.set_ylabel("effective internal temperature [K]")
    ax.set_title("current-dependent thermalization")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best", frameon=False)


def _draw_spread_and_energy(ax, cases: list[CaseData], args: argparse.Namespace) -> None:
    hybrid = _hybrid_cases(cases)
    current = np.asarray([case.current_na for case in hybrid], dtype=np.float64)
    r95 = np.asarray([case.r_p95_mm for case in hybrid], dtype=np.float64)
    ke95 = np.asarray([case.ke_p95_ev for case in hybrid], dtype=np.float64)
    color_r = "#2166ac"
    color_ke = "#b2182b"
    ax.plot(current, r95, marker="o", color=color_r, linewidth=scaled(1.8, args.paper_line_scale), label="r p95")
    ax.set_xlabel("ion current [nA]")
    ax.set_ylabel("active-ion r p95 [mm]", color=color_r)
    ax.tick_params(axis="y", labelcolor=color_r)
    ax.grid(True, alpha=0.24)
    twin = ax.twinx()
    twin.plot(current, ke95, marker="s", color=color_ke, linewidth=scaled(1.8, args.paper_line_scale), label="KE p95")
    twin.set_ylabel("kinetic energy p95 [eV]", color=color_ke)
    twin.tick_params(axis="y", labelcolor=color_ke)
    ax.set_title("beam broadening and kinetic spread")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", frameon=False)


def _draw_model_sensitivity(ax, cases: list[CaseData], args: argparse.Namespace) -> None:
    hybrid = _case_at(cases, 2.0, "hybrid")
    explicit = _case_at(cases, 2.0, "explicit")
    if hybrid is None or explicit is None:
        ax.text(0.5, 0.5, "2 nA hybrid/explicit pair not available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    metrics = [
        ("T_eff p95", explicit.teff_p95_k / hybrid.teff_p95_k),
        ("KE p95", explicit.ke_p95_ev / hybrid.ke_p95_ev),
        ("collision p95", explicit.collision_p95 / hybrid.collision_p95),
        ("r p95", explicit.r_p95_mm / hybrid.r_p95_mm),
    ]
    labels = [item[0] for item in metrics]
    values = np.asarray([item[1] for item in metrics], dtype=np.float64)
    colors = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, alpha=0.86)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=scaled(1.0, args.paper_line_scale))
    for xpos, value in zip(x, values):
        ax.text(xpos, value + 0.025, f"{value:.2f}x", ha="center", va="bottom", fontsize=9.0 * float(args.paper_font_scale))
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("explicit / hybrid at 2 nA")
    ax.set_ylim(0.82, max(1.22, float(np.nanmax(values)) + 0.12))
    ax.set_title("collision-model sensitivity")
    ax.grid(True, axis="y", alpha=0.24)


def _axial_profiles(case: CaseData, columns: list[str], z_bins_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = {column: i for i, column in enumerate(columns)}
    z_mm = case.data[:, index["z_m"]] * 1.0e3
    teff = case.data[:, index["teff_k"]]
    weights = case.data[:, index["represented_real_ions"]]
    centers: list[float] = []
    p50: list[float] = []
    p95: list[float] = []
    mean: list[float] = []
    for lo, hi in zip(z_bins_mm[:-1], z_bins_mm[1:]):
        mask = (z_mm >= lo) & (z_mm < hi) & np.isfinite(teff) & np.isfinite(weights) & (weights > 0.0)
        if np.count_nonzero(mask) < 8:
            continue
        centers.append(0.5 * (lo + hi))
        p50.append(_weighted_percentile(teff[mask], weights[mask], 50.0))
        p95.append(_weighted_percentile(teff[mask], weights[mask], 95.0))
        mean.append(_weighted_mean(teff[mask], weights[mask]))
    return np.asarray(centers), np.asarray(p50), np.asarray(p95), np.asarray(mean)


def _draw_axial_profile(ax, cases: list[CaseData], columns: list[str], args: argparse.Namespace) -> None:
    hybrid = _case_at(cases, 2.0, "hybrid")
    explicit = _case_at(cases, 2.0, "explicit")
    z_bins_mm = np.linspace(float(args.profile_z_min_mm), float(args.profile_z_max_mm), int(args.profile_bins) + 1)
    if hybrid is not None:
        zc, p50, p95, _ = _axial_profiles(hybrid, columns, z_bins_mm)
        ax.plot(zc, p50, color="#0b559f", linewidth=scaled(1.6, args.paper_line_scale), label="hybrid p50")
        ax.plot(zc, p95, color="#0b559f", linestyle="--", linewidth=scaled(1.4, args.paper_line_scale), label="hybrid p95")
    if explicit is not None:
        zc, p50, p95, _ = _axial_profiles(explicit, columns, z_bins_mm)
        ax.plot(zc, p50, color="#b2182b", linewidth=scaled(1.6, args.paper_line_scale), label="explicit p50")
        ax.plot(zc, p95, color="#b2182b", linestyle="--", linewidth=scaled(1.4, args.paper_line_scale), label="explicit p95")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("effective internal temperature [K]")
    ax.set_xlim(float(args.profile_z_min_mm), float(args.profile_z_max_mm))
    ax.set_title("axial thermalization profile at 2 nA")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best", frameon=False, ncols=2)


def build_figure(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.data_root)
    case_names = [item.strip() for item in str(args.cases).split(",") if item.strip()]
    case_dirs = [root / case_name for case_name in case_names]
    missing = [str(path) for path in case_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing case directories: {missing}")
    macro_dt_s = float(args.macro_time_step_s)
    requested_macro = None if args.snapshot_macro is None else int(args.snapshot_macro)
    if requested_macro is None and args.snapshot_time_ms is not None:
        requested_macro = int(round(float(args.snapshot_time_ms) * 1.0e-3 / macro_dt_s))
    macro_step = _common_macro(case_dirs, requested_macro)
    cases = [_load_case(case_dir, macro_step, macro_dt_s) for case_dir in case_dirs]
    columns = _load_columns(case_dirs[0])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_csv = output.with_suffix(".stats.csv")
    summary_json = output.with_suffix(".summary.json")
    _write_csv(stats_csv, _case_rows(cases))
    summary = {
        "data_root": str(root),
        "cases": [case.name for case in cases],
        "macro_step": int(macro_step),
        "time_ms": float(cases[0].time_ms) if cases else None,
        "basis": "active-ion snapshots at common macro step; not terminal-event transmission statistics",
        "stats_csv": str(stats_csv),
        "output": str(output),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    apply_paper_style(font_scale=args.paper_font_scale, line_scale=args.paper_line_scale)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6), dpi=int(args.dpi), constrained_layout=True)
    _draw_temperature_vs_current(axes[0, 0], cases, args)
    _draw_spread_and_energy(axes[0, 1], cases, args)
    _draw_model_sensitivity(axes[1, 0], cases, args)
    _draw_axial_profile(axes[1, 1], cases, columns, args)
    add_subplot_labels(
        axes,
        enabled=bool(args.subplot_labels),
        font_scale=args.paper_font_scale,
        line_scale=args.paper_line_scale,
    )
    fig.suptitle(f"Partial-run IICT/IonSPA thermalization diagnostics at t={cases[0].time_ms:.2f} ms")
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Saved: {stats_csv}")
    print(f"Saved: {summary_json}")
    return {"figure": output, "stats_csv": stats_csv, "summary_json": summary_json}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create manuscript Fig. 7 thermalization diagnostics from partial snapshots.")
    parser.add_argument("--data-root", default=str(DEFAULT_ROOT), help="Root directory containing current-sweep case folders.")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated case directory names.")
    parser.add_argument("--output", default=str(Path("manuscript") / "Figs" / "Fig7_thermalization_map.png"), help="Output PNG path.")
    parser.add_argument("--macro-time-step-s", type=float, default=5.0e-8, help="Macro time step used by snapshot macro indices [s].")
    parser.add_argument("--snapshot-macro", type=int, default=None, help="Requested common snapshot macro step; nearest common step is used.")
    parser.add_argument("--snapshot-time-ms", type=float, default=None, help="Requested common snapshot time [ms]; nearest common step is used.")
    parser.add_argument("--profile-z-min-mm", type=float, default=4.5, help="Lower z bound for axial T_eff profiles [mm].")
    parser.add_argument("--profile-z-max-mm", type=float, default=65.0, help="Upper z bound for axial T_eff profiles [mm].")
    parser.add_argument("--profile-bins", type=int, default=28, help="Number of z bins for axial T_eff profiles.")
    parser.add_argument(
        "--subplot-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label subplots as (a), (b), ...",
    )
    parser.add_argument("--paper-font-scale", type=float, default=1.16, help="Global font-size multiplier for paper figures.")
    parser.add_argument("--paper-line-scale", type=float, default=1.25, help="Line-width multiplier for paper figures.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    build_figure(parse_args())


if __name__ == "__main__":
    main()
