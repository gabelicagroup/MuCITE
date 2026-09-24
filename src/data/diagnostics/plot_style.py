"""Shared paper-figure styling helpers for diagnostic plots."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import matplotlib
import numpy as np


def apply_paper_style(*, font_scale: float = 1.18, line_scale: float = 1.25) -> None:
    """Set readable defaults for figures intended for manuscript placement."""

    font_scale = max(float(font_scale), 0.2)
    line_scale = max(float(line_scale), 0.2)
    matplotlib.rcParams.update(
        {
            "font.size": 10.5 * font_scale,
            "axes.titlesize": 12.0 * font_scale,
            "axes.labelsize": 11.2 * font_scale,
            "xtick.labelsize": 9.8 * font_scale,
            "ytick.labelsize": 9.8 * font_scale,
            "legend.fontsize": 9.0 * font_scale,
            "figure.titlesize": 13.2 * font_scale,
            "axes.linewidth": 0.95 * line_scale,
            "lines.linewidth": 1.45 * line_scale,
            "patch.linewidth": 0.95 * line_scale,
            "xtick.major.width": 0.9 * line_scale,
            "ytick.major.width": 0.9 * line_scale,
            "xtick.minor.width": 0.7 * line_scale,
            "ytick.minor.width": 0.7 * line_scale,
            "grid.linewidth": 0.55 * line_scale,
            "savefig.bbox": "tight",
        }
    )


def scaled(value: float, line_scale: float) -> float:
    return float(value) * max(float(line_scale), 0.2)


def subplot_label(index: int) -> str:
    """Return a lower-case subplot label: (a), (b), ..., (aa), ..."""

    if index < 0:
        raise ValueError("subplot index must be non-negative.")
    letters: list[str] = []
    value = int(index)
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(ord("a") + remainder))
        if value == 0:
            break
        value -= 1
    return f"({''.join(reversed(letters))})"


def _flatten_axes(axes: Any) -> list[Any]:
    if isinstance(axes, np.ndarray):
        return [ax for ax in axes.ravel().tolist() if ax is not None]
    if isinstance(axes, Iterable) and not hasattr(axes, "text"):
        return [ax for ax in axes if ax is not None]
    return [axes]


def add_subplot_labels(
    axes: Any,
    *,
    enabled: bool = True,
    x: float = 0.025,
    y: float = 0.965,
    font_scale: float = 1.18,
    line_scale: float = 1.25,
) -> None:
    """Add bold manuscript-style subplot labels in axes coordinates."""

    if not enabled:
        return
    for index, ax in enumerate(_flatten_axes(axes)):
        if hasattr(ax, "get_visible") and not ax.get_visible():
            continue
        ax.text(
            x,
            y,
            subplot_label(index),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.5 * max(float(font_scale), 0.2),
            fontweight="bold",
            color="black",
            zorder=30,
        )
