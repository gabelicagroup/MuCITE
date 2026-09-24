"""PIC and Poisson-solver CLI arguments."""

from __future__ import annotations

import argparse

from ....config import SimulationConfig

_defaults = SimulationConfig()

ARGUMENTS = (
    (("--sor-omega",), dict(type=float, default=_defaults.sor_omega, help="SOR relaxation factor.")),
    (("--sor-max-iters",), dict(type=int, default=_defaults.sor_max_iters, help="Maximum SOR iterations.")),
    (("--sor-tolerance",), dict(type=float, default=_defaults.sor_tolerance, help="SOR convergence tolerance.")),
    (("--pic-poisson-backend",), dict(choices=["sor", "sparse_direct", "sparse_spsolve", "sparse_bicgstab", "sparse_cg", "amg", "pyamg"], default=_defaults.pic_poisson_backend, help="PIC Poisson backend.")),
    (("--pic-poisson-preconditioner",), dict(choices=["none", "jacobi"], default=_defaults.pic_poisson_preconditioner, help="Sparse solver preconditioner.")),
    (("--pic-poisson-warm-start",), dict(action=argparse.BooleanOptionalAction, default=_defaults.pic_poisson_warm_start, help="Reuse the prior PIC potential.")),
    (("--pic-amg-mode",), dict(choices=["solve", "preconditioned_cg", "preconditioned_bicgstab"], default=_defaults.pic_amg_mode, help="AMG solve mode.")),
    (("--pic-amg-solver",), dict(choices=["ruge_stuben", "smoothed_aggregation"], default=_defaults.pic_amg_solver, help="AMG hierarchy builder.")),
    (("--pic-amg-tolerance",), dict(type=float, default=_defaults.pic_amg_tolerance, help="AMG relative tolerance.")),
    (("--pic-amg-max-iters",), dict(type=int, default=_defaults.pic_amg_max_iters, help="Maximum AMG iterations.")),
    (("--pic-amg-fallback-backend",), dict(choices=["sparse_bicgstab", "sparse_cg", "sparse_spsolve", "sparse_direct"], default=_defaults.pic_amg_fallback_backend, help="AMG fallback backend.")),
    (("--pic-grid-nr",), dict(type=int, default=None, help="PIC radial nodes; omit or use 0 with nz for shared mode.")),
    (("--pic-grid-nz",), dict(type=int, default=None, help="PIC axial nodes; omit or use 0 with nr for shared mode.")),
    (("--pic-space-charge-scale",), dict(type=float, default=_defaults.pic_space_charge_scale, help="PIC field multiplier.")),
)


def add_pic_arguments(parser: argparse.ArgumentParser) -> None:
    for flags, kwargs in ARGUMENTS:
        parser.add_argument(*flags, **kwargs)
