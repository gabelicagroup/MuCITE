"""Generate a compact paper workflow diagram for the ion-source simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(
    ax,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#1f2937",
    fontsize: float = 9.5,
    weight: str = "normal",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#111827",
        linespacing=1.22,
    )
    return patch


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *, rad: float = 0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color="#374151",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def _stage_label(ax, x: float, text: str) -> None:
    ax.text(
        x,
        0.94,
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
    )


def make_workflow_figure(output_base: Path, *, dpi: int = 300) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.8, 3.0), dpi=dpi)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.78)
    ax.axis("off")

    colors = ["#eef2ff", "#ecfdf5", "#f0fdfa", "#fff7ed", "#f8fafc"]
    boxes = [
        _box(
            ax,
            xy=(0.035, 0.29),
            width=0.15,
            height=0.32,
            text="Input fields\nSIMION DC/RF\nFluent gas\nElectrode mask",
            facecolor=colors[0],
            fontsize=8.2,
            weight="bold",
        ),
        _box(
            ax,
            xy=(0.235, 0.29),
            width=0.15,
            height=0.32,
            text="Field baker\naligned r-z grid\nE_DC, E_RF\nv_gas, T, p",
            facecolor=colors[1],
            fontsize=8.2,
            weight="bold",
        ),
        _box(
            ax,
            xy=(0.435, 0.29),
            width=0.15,
            height=0.32,
            text="Ion source\ncontinuous current\nGaussian profile\nbirth velocity",
            facecolor=colors[2],
            fontsize=8.2,
            weight="bold",
        ),
        _box(
            ax,
            xy=(0.635, 0.25),
            width=0.16,
            height=0.40,
            text="3D transport\nRF(t) + DC\nPIC space charge\ncollisions\nsurface hits",
            facecolor=colors[3],
            fontsize=8.2,
            weight="bold",
        ),
        _box(
            ax,
            xy=(0.835, 0.29),
            width=0.14,
            height=0.32,
            text="Analysis\nz_exit events\nsnapshots\nbeam metrics",
            facecolor=colors[4],
            fontsize=8.2,
            weight="bold",
        ),
    ]

    for left, right in zip(boxes[:-1], boxes[1:]):
        _arrow(
            ax,
            (left.get_x() + left.get_width(), left.get_y() + left.get_height() / 2.0),
            (right.get_x(), right.get_y() + right.get_height() / 2.0),
        )

    ax.text(
        0.5,
        0.11,
        "Particles are propagated in 3D; the baked electric field, gas field and PIC mesh use the aligned axisymmetric r-z representation.",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#4b5563",
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact simulation workflow figure.")
    parser.add_argument("--output", default=str(Path("outputs") / "paper_figures" / "simulation_workflow"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = make_workflow_figure(Path(args.output), dpi=args.dpi)
    print("Saved workflow figure:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
