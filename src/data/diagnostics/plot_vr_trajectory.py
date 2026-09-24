"""Compatibility entry point for local v_r trajectory diagnostic plots."""

from __future__ import annotations

if __package__:
    from .plot_vr_trajectory_overlay import main
else:  # pragma: no cover - supports direct script execution.
    from plot_vr_trajectory_overlay import main  # type: ignore


if __name__ == "__main__":
    main()
