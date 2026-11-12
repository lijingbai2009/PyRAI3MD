#!/usr/bin/env python3
"""
======================================================================
This module recomputes the cell history of a qualification run from the
products it left behind.

An anisotropic variable-cell NPT run is judged by numbers that are not
in any single product: the volume has to stay near its target, the cell
must not turn out from under the state it carries, and the barostat
tensors must stay bounded. All of it is derivable from two files the
workflow already writes -- the per-frame Lattice="..." comment of
md.xyz and the column table of md.ensemble -- and none of it should
rest on a number retold from a failure message.

The reading is the runner's own: the trajectory cells, the ensemble
columns, the frame rotation and the volume drift come from
`pyrai3md.qualification.runner` (`read_md_lattice`, `read_md_ensemble`,
`cell_frame_metrics`), because a second implementation of a judged
quantity is how a tool and a round disagree. Measured on three synthetic
frames before this was fixed, the tool's own determinant gave
volume_drift 1.00000000000076614e-04 against the runner's
9.9999999999988987e-05, and its SVD polar factor sat 8.795e-15 rad away
from the runner's Newton-Schulz one. One function now answers both, so
what this report prints is bit-for-bit what the round decided with.

The thresholds are the round's too: `BOX_TOLERANCES['volume_drift']`
(0.25) decides, and the accumulated turn is reported against
`BOX_NOTES['cell_rotation_rad']` (0.15 rad, 8.594 degrees). The literal
this file carried, 17.2 degrees, was an accumulated turn seen in one
round rather than the constant the round notes with, which is why it
went away. The per-pair maximum turn (`rotation_max_deg`) left with it:
the runner sums the turn and notes the sum, and reaching across the
boundary for the private per-pair helper would only move the
duplication somewhere else.

For every example in a downloaded run tree this tool reports the frame
count, the volume excursion against the first frame, the accumulated
rotation of the simulation frame, the extremes of the strain, its
momentum and its velocity, the kinetic / virial / total pressure, and the
instantaneous temperature. It reads nothing but the products, writes
nothing, and names the file it took each number from.

Usage:
    python3 tools/analyze_npt_frames.py --root <tree>
    python3 tools/analyze_npt_frames.py --root <tree> --detail

This tool is located in tools/ and is run directly. It does not
participate in test planning or execution logic, and it writes nothing.

Sep 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

from pyrai3md.qualification.runner import (
    BOX_NOTES,
    BOX_TOLERANCES,
    cell_frame_metrics,
    read_md_ensemble,
    read_md_lattice,
)

MIN_JUDGED_FRAMES = 50
TENSOR_PARTS = ('xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz')
ENSEMBLE_COLUMNS = tuple(
    ['time', 'T_inst(K)', 'Pint_bar', 'Pkin_bar', 'Pvir_bar',
     'scale_cell_pre']
    + ['peps_' + part for part in TENSOR_PARTS]
    + ['eps_' + part for part in TENSOR_PARTS])


def cell_matrices(rows):
    """The runner's flat nine-element rows as an (n, 3, 3) array."""
    if not rows:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3, 3)


def title_of(xyz_path):
    """Example title from the file name, the way the run wrote it."""
    title = os.path.basename(xyz_path)
    for suffix in ('_example.md.xyz', '.md.xyz'):
        if title.endswith(suffix):
            return title[:-len(suffix)]
    return title


def tensor_column(ensemble, prefix):
    """Rebuild a (n, 3, 3) tensor series from nine _xx .. _zz columns."""
    names = ['%s_%s' % (prefix, part) for part in TENSOR_PARTS]
    if not all(name in ensemble for name in names):
        return np.zeros((0, 3, 3))
    rows = len(ensemble[names[0]])
    out = np.zeros((rows, 3, 3), dtype=np.float64)
    for index, name in enumerate(names):
        out[:, index // 3, index % 3] = np.asarray(ensemble[name],
                                                  dtype=np.float64)
    return out


def analyze(xyz_path, ensemble_path, temperature=0.0):
    """Collect the cell-history metrics of one example run.

    `volume_drift`, `cell_rotation_rad` and `temperature_drift` are the
    values the round judged with, taken from `cell_frame_metrics`; the
    rest are the extremes this report adds on top of the same two files.

    Parameters
    ----------
    xyz_path : str
        Path to the md.xyz product.
    ensemble_path : str
        Path to the md.ensemble product.
    temperature : float, optional
        Target temperature of the example, for the drift the round uses.

    Returns
    -------
    dict
        Metrics, with the frame count and the name of the example.
    """
    rows = read_md_lattice(xyz_path)
    ensemble = (read_md_ensemble(ensemble_path, ENSEMBLE_COLUMNS)
                if ensemble_path else {})
    result = {'title': title_of(xyz_path), 'frames': len(rows),
              'ensemble_frames': len(ensemble.get('time', []))}
    result.update(cell_frame_metrics(rows, ensemble, temperature))
    lattices = cell_matrices(rows)
    if lattices.shape[0] < 2:
        return result
    result['rotation_deg'] = float(
        math.degrees(result.get('cell_rotation_rad', 0.0)))
    volumes = np.abs(np.linalg.det(lattices))
    result['volume_ratio_min'] = float(np.min(volumes) / volumes[0])
    result['volume_ratio_max'] = float(np.max(volumes) / volumes[0])
    axes = np.array([np.linalg.norm(cell, axis=1) for cell in lattices])
    result['axis_min'] = float(axes.min())
    result['axis_max'] = float(axes.max())
    p_eps = tensor_column(ensemble, 'peps')
    eps = tensor_column(ensemble, 'eps')
    if p_eps.shape[0]:
        result['peps_max'] = float(np.nanmax(
            np.linalg.norm(p_eps, 'fro', axis=(1, 2))))
        result['peps_skew_max'] = float(np.nanmax(np.linalg.norm(
            p_eps - np.transpose(p_eps, (0, 2, 1)), 'fro', axis=(1, 2))))
    if eps.shape[0]:
        result['eps_max'] = float(np.nanmax(np.abs(eps)))
    temperatures = ensemble.get('T_inst(K)')
    if temperatures:
        array = np.asarray(temperatures, dtype=np.float64)
        result['t_inst_min'] = float(np.nanmin(array))
        result['t_inst_max'] = float(np.nanmax(array))
    for short, column in (('pkin', 'Pkin_bar'), ('pvir', 'Pvir_bar'),
                          ('pint', 'Pint_bar')):
        values = ensemble.get(column)
        if values:
            array = np.asarray(values, dtype=np.float64)
            result['%s_extreme' % short] = float(
                array[np.nanargmax(np.abs(array))])
    result['scale_cell_pre_nan'] = bool(
        'scale_cell_pre' in ensemble
        and np.isnan(np.asarray(ensemble['scale_cell_pre'],
                               dtype=np.float64)).any())
    return result


def find_runs(root):
    """Yield (md.xyz, md.ensemble) pairs under a run tree."""
    for xyz_path in sorted(glob.glob(os.path.join(root, '**', '*.md.xyz'),
                                     recursive=True)):
        base = xyz_path[:-len('.md.xyz')]
        ensemble_path = '%s.md.ensemble' % base
        yield xyz_path, (ensemble_path
                         if os.path.exists(ensemble_path) else '')


def diverged(rows):
    """Examples past the thresholds the round itself decides with."""
    volume_bound = BOX_TOLERANCES['volume_drift']
    rotation_bound = math.degrees(BOX_NOTES['cell_rotation_rad'])
    return [row['title'] for row in rows
            if row.get('volume_drift', 0.0) > volume_bound
            or row.get('rotation_deg', 0.0) > rotation_bound]


def report(root, detail=False):
    """Print the per-example cell history of a run tree."""
    rows = []
    for xyz_path, ensemble_path in find_runs(root):
        metrics = analyze(xyz_path, ensemble_path)
        metrics['source'] = os.path.relpath(xyz_path, root)
        rows.append(metrics)
    if not rows:
        print('no md.xyz products under %s' % root)
        return rows
    npt = [row for row in rows if 'npt' in row['title']]

    def show(row, key, spec, span=None):
        """Format one metric, or '-' when the products do not carry it."""
        if key not in row:
            return '-'
        if span:
            return '%s-%s' % (spec % row[key], spec % row[span])
        return spec % row[key]

    print('%-32s %6s %13s %9s %9s %9s %9s'
          % ('example', 'frames', 'V/V0', 'rot(deg)', 'axis(A)',
             'T(K)', 'Pint(bar)'))
    for row in rows:
        verdict = ('unjudged, %d frames' % row['frames']
                   if 0 < row['frames'] < MIN_JUDGED_FRAMES else '')
        if not row.get('ensemble_frames'):
            verdict = (verdict + ' (no md.ensemble)').strip()
        print('%-32s %6d %13s %9s %9s %9s %9s  %s'
              % (row['title'], row['frames'],
                 show(row, 'volume_ratio_min', '%.3f', 'volume_ratio_max'),
                 show(row, 'rotation_deg', '%.2f'),
                 show(row, 'axis_min', '%.2f', 'axis_max'),
                 show(row, 't_inst_min', '%.0f', 't_inst_max'),
                 show(row, 'pint_extreme', '%.3g'),
                 verdict))
    print()
    print('%d trajectories read, %d of them NPT' % (len(rows), len(npt)))
    if detail:
        for row in npt:
            print()
            print('-- %s (%s)' % (row['title'], row['source']))
            for key in sorted(row):
                if key in ('title', 'source'):
                    continue
                value = row[key]
                print('   %-20s %s'
                      % (key, '%.6g' % value if isinstance(value, float)
                         else value))
    return rows


def main(argv=None):
    """Parse arguments and run the report."""
    parser = argparse.ArgumentParser(
        description='Recompute the cell history of a qualification run '
                    'from its md.xyz and md.ensemble products.')
    parser.add_argument('--root', required=True,
                        help='run tree to read (a tmp_* qualification dir)')
    parser.add_argument('--detail', action='store_true',
                        help='print every metric for each NPT example')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.root):
        print('%s is not a directory' % args.root)
        return 1
    rows = report(args.root, detail=args.detail)
    over = diverged(rows)
    if over:
        print('over threshold (volume drift > %.2f, rotation > %.3f deg): %s'
              % (BOX_TOLERANCES['volume_drift'],
                 math.degrees(BOX_NOTES['cell_rotation_rad']),
                 ', '.join(over)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
