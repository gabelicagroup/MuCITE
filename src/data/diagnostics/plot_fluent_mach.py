"""Plot a mirrored meridional Mach-number contour from Fluent node data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .plot_fluent_thermodynamics import _load_fluent_points, _crop_points


def _overlay_electric_field(ax, args, fluid_interpolator) -> dict:
    field_path = Path(args.electric_field).resolve()
    payload = np.load(field_path, allow_pickle=True).item()
    grid, simion = payload['grid'], payload['simion']
    z_all = np.asarray(grid['z_coords_m']) * 1000.0
    r_all = np.asarray(grid['r_coords_m']) * 1000.0
    zi = np.flatnonzero((z_all >= ax.get_xlim()[0]) & (z_all <= ax.get_xlim()[1]))[::args.electric_grid_stride]
    ri = np.flatnonzero(r_all <= ax.get_ylim()[1])[::args.electric_grid_stride]
    z, r = z_all[zi], r_all[ri]
    reference = float(simion.get('rf_reference_peak_voltage_v', 1.0))
    if reference <= 0 or not np.isfinite(reference):
        raise ValueError('Invalid baked RF reference voltage.')
    rf_factor = args.rf_peak_voltage_v / reference * np.cos(np.deg2rad(args.rf_phase_deg))
    selection = np.ix_(ri, zi)
    er = np.asarray(simion['e_dc_r_v_per_m'])[selection] + rf_factor * np.asarray(simion['e_rf_r_v_per_m'])[selection]
    ez = np.asarray(simion['e_dc_z_v_per_m'])[selection] + rf_factor * np.asarray(simion['e_rf_z_v_per_m'])[selection]
    er[0] = 0.0
    geometry = {
        'dc_source': simion.get('dc_source_path'),
        'rf_source': simion.get('rf_source_path'),
        'dc_z_offset_m': simion.get('dc_z_offset_m'),
        'rf_z_offset_m': simion.get('rf_z_offset_m'),
    }
    del payload, grid, simion
    zg, rg = np.meshgrid(z, r)
    fluid = fluid_interpolator(zg, rg)
    magnitude = np.hypot(er, ez)
    excluded = np.ma.getmaskarray(fluid) | ~np.isfinite(magnitude) | (magnitude < args.electric_min_v_per_m)
    er = np.ma.masked_where(excluded, er)
    ez = np.ma.masked_where(excluded, ez)
    patch_count = len(ax.patches)
    lines = ax.streamplot(
        z, r, ez, er, density=(args.electric_density * 1.5, args.electric_density),
        color=args.electric_line_color, linewidth=.5, arrowsize=.65,
        minlength=.08, maxlength=4.0, integration_direction='both', zorder=4,
    )
    segments = lines.lines.get_segments()
    if not segments:
        raise ValueError('No electric field lines were generated in the fluid domain.')
    reflected = [segment * np.array([1.0, -1.0]) for segment in segments]
    ax.add_collection(LineCollection(reflected, colors=args.electric_line_color, linewidths=.5, zorder=4))
    upper_arrows = list(ax.patches[patch_count:])
    for arrow in upper_arrows:
        mirrored_arrow = copy.copy(arrow)
        mirrored_arrow.set_transform(Affine2D().scale(1.0, -1.0) + ax.transData)
        ax.add_patch(mirrored_arrow)
    return {
        'field_file': str(field_path), 'mode': 'instantaneous_external_dc_plus_rf',
        'rf_peak_voltage_v': args.rf_peak_voltage_v,
        'rf_reference_peak_voltage_v': reference,
        'rf_phase_deg': args.rf_phase_deg,
        'rf_cosine_factor': float(np.cos(np.deg2rad(args.rf_phase_deg))),
        'electric_units': 'V/m', 'includes_pic_space_charge': False,
        'sampled_z_bounds_mm': [float(z[0]), float(z[-1])],
        'sampled_r_bounds_mm': [float(r[0]), float(r[-1])],
        'maximum_sampled_electric_magnitude_v_per_m': float(magnitude.max()),
        'display_min_electric_magnitude_v_per_m': args.electric_min_v_per_m,
        'upper_line_segment_count': len(segments),
        'mirrored_line_segment_count': len(reflected),
        'upper_arrow_count': len(upper_arrows), 'mirrored_arrow_count': len(upper_arrows),
        'reflection': 'Ez even and Er odd; lower line paths and arrows reflected from upper half',
        'line_color': args.electric_line_color,
        'geometry': geometry,
    }


def plot_mach(args: argparse.Namespace) -> Path:
    source = Path(args.gas_csv).resolve()
    points = _load_fluent_points(
        source,
        z_offset_mm=args.z_offset_mm,
        gamma=args.gamma,
        specific_gas_constant_j_per_kg_k=args.gas_constant,
    )
    points, bounds = _crop_points(
        points, z_min_mm=args.z_min_mm, z_max_mm=args.z_max_mm,
        r_min_mm=0.0, r_max_mm=args.r_max_mm,
    )
    nodes = pd.DataFrame({
        'z_mm': points['z_mm'], 'r_mm': points['r_mm'],
        'mach': points['mach_number'],
    }).groupby(['z_mm', 'r_mm'], as_index=False, sort=True).mean()
    z, r, mach = (nodes[name].to_numpy() for name in ['z_mm', 'r_mm', 'mach'])
    coords = np.column_stack((z, r))
    tree = cKDTree(coords)
    spacing = tree.query(coords, k=2, workers=-1)[0][:, 1]
    tri = mtri.Triangulation(z, r)
    vertices = tri.triangles
    a, b, c = (coords[vertices[:, i]] for i in range(3))
    longest_edge = np.maximum.reduce([
        np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1),
        np.linalg.norm(c - a, axis=1),
    ])
    # Disconnected fluid boundaries must not be bridged by long display triangles.
    local_spacing = np.max(spacing[vertices], axis=1)
    mask = longest_edge > np.maximum(args.gap_floor_mm, args.gap_factor * local_spacing)
    tri.set_mask(mask)
    interpolate = mtri.LinearTriInterpolator(tri, mach)
    z_grid = np.linspace(bounds[0], bounds[1], args.grid_nz)
    r_grid = np.linspace(0.0, bounds[3], args.grid_nr)
    z_mesh, r_mesh = np.meshgrid(z_grid, r_grid)
    upper = np.ma.masked_invalid(interpolate(z_mesh, r_mesh))
    mirrored = np.ma.concatenate([upper[:0:-1], upper], axis=0)
    transverse = np.r_[-r_grid[:0:-1], r_grid]
    assert np.array_equal(np.ma.getmaskarray(mirrored), np.ma.getmaskarray(mirrored)[::-1])
    assert np.ma.max(np.ma.abs(mirrored - mirrored[::-1])) == 0.0

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.linewidth': 0.85, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'savefig.facecolor': 'white',
    })
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    fig.subplots_adjust(left=.085, right=.88, bottom=.12, top=.98)
    levels = np.linspace(0.0, args.mach_max, args.bands + 1)
    cmap = plt.get_cmap('jet', args.bands).copy()
    cmap.set_bad('#E6E6E6')
    ax.set_facecolor('#E6E6E6')
    image = ax.imshow(
        mirrored, origin='lower',
        extent=[z_grid[0], z_grid[-1], transverse[0], transverse[-1]],
        cmap=cmap, norm=BoundaryNorm(levels, cmap.N),
        interpolation='nearest', aspect='equal',
    )
    ax.set_xlabel(r'$z$ / mm')
    ax.set_ylabel(r'$y$ / mm')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(-bounds[3], bounds[3])
    ax.tick_params(length=4, width=.8, pad=5)
    electric = _overlay_electric_field(ax, args, interpolate) if args.electric_field else None
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes('right', size='3.2%', pad=.16)
    colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=np.linspace(0, args.mach_max, 6))
    colorbar.set_label('Mach number', labelpad=9)
    colorbar.outline.set_linewidth(.7)
    colorbar.ax.tick_params(length=3, width=.7)
    assert not ax.get_title()
    target = Path(args.output).resolve() if args.output else source.parent / 'plots' / f'{source.stem}_mach_symmetric.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=args.dpi, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    summary = {
        'input_file': str(source), 'output_file': str(target),
        'mach_definition': 'hypot(axial_velocity, radial_velocity) / sqrt(gamma * R_specific * static_temperature)',
        'gamma': args.gamma, 'specific_gas_constant_j_per_kg_k': args.gas_constant,
        'z_offset_mm': args.z_offset_mm,
        'plot_z_bounds_mm': list(bounds[:2]),
        'plot_transverse_bounds_mm': [-bounds[3], bounds[3]],
        'display_coordinate': 'y = +/- r; reflection about the axis is exact',
        'retained_unique_nodes': len(nodes), 'raw_mach_range': [float(mach.min()), float(mach.max())],
        'display_mach_range': [0, args.mach_max], 'color_bands': args.bands,
        'interpolation': 'piecewise linear on node Delaunay triangles, with long-gap triangles masked',
        'masked_triangle_count': int(mask.sum()), 'total_triangle_count': len(vertices),
        'gap_factor': args.gap_factor, 'gap_floor_mm': args.gap_floor_mm,
        'mask_note': 'Gray is unsampled or display-masked space; not an independently supplied electrode mask.',
        'grid_shape': list(mirrored.shape),
        'reflection_max_abs_error': 0.0, 'title': None, 'dpi': args.dpi,
        'electric_overlay': electric,
    }
    target.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gas_csv')
    parser.add_argument('--output', default='')
    parser.add_argument('--z-offset-mm', type=float, default=0.0)
    parser.add_argument('--z-min-mm', type=float, default=None)
    parser.add_argument('--z-max-mm', type=float, default=None)
    parser.add_argument('--r-max-mm', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=1.4)
    parser.add_argument('--gas-constant', type=float, default=296.80)
    parser.add_argument('--mach-max', type=float, default=10.0)
    parser.add_argument('--bands', type=int, default=20)
    parser.add_argument('--grid-nz', type=int, default=2401)
    parser.add_argument('--grid-nr', type=int, default=681)
    parser.add_argument('--gap-factor', type=float, default=4.0)
    parser.add_argument('--gap-floor-mm', type=float, default=.05)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--electric-field', default='', help='Optional baked NPY field for electric field-line overlay.')
    parser.add_argument('--rf-peak-voltage-v', type=float, default=100.0)
    parser.add_argument('--rf-phase-deg', type=float, default=0.0)
    parser.add_argument('--electric-density', type=float, default=1.3)
    parser.add_argument('--electric-line-color', default='white')
    parser.add_argument('--electric-min-v-per-m', type=float, default=5.0)
    parser.add_argument('--electric-grid-stride', type=int, default=10)
    args = parser.parse_args()
    if min(args.gamma, args.gas_constant, args.mach_max, args.gap_factor, args.gap_floor_mm) <= 0:
        parser.error('Gas properties and display limits must be positive.')
    if min(args.grid_nz, args.grid_nr, args.bands, args.dpi) < 2:
        parser.error('Grid dimensions, bands, and dpi must be at least 2.')
    if args.electric_grid_stride < 1 or args.electric_density <= 0:
        parser.error('Electric grid stride and line density must be positive.')
    plot_mach(args)


if __name__ == '__main__':
    main()
