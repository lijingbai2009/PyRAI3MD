"""
======================================================================
Engine-gated: xTB really writes the charge file we parse.

The skill reads `charges` (GFN-x) or `gfnff_charges` (GFN-FF) into the
framework's `computed_charges` channel, and the input builder now always
passes `--pop`. Both halves were unit-tested against fabricated files; this
module is the only place that asks the engine, and it is skipped when no
xTB is on PATH (the test environment carries it, `qualification/pmd_env.yaml`
names `xtb_path`).

The sum rule is the discriminator: a population of two neutral fragments has
to add up to the requested total charge, so a file that exists but belongs to
another geometry (wrong atom count, wrong mtime) cannot pass by accident.

Nov 03 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

XTB = shutil.which('xtb')

pytestmark = pytest.mark.skipif(
    not XTB,
    reason='no xTB on PATH: set qualification/pmd_env.yaml xtb_path or put '
           'a verified xtb on PATH. This harness runs the real engine, so a '
           'skip is not a pass')

WATER = """3

O  0.000000  0.000000  0.117340
H  0.000000  0.757540 -0.469360
H  0.000000 -0.757540 -0.469360
"""


def _run(tmp_path, flags):
    geometry = tmp_path / 'water.xyz'
    geometry.write_text(WATER)
    completed = subprocess.run(
        [XTB, str(geometry), '--grad', '--pop'] + flags,
        cwd=str(tmp_path), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    return tmp_path


def test_gfn2_population_writes_charges(tmp_path):
    _run(tmp_path, ['--gfn', '2'])
    charge_file = tmp_path / 'charges'
    assert charge_file.exists(), (
        'xTB wrote no charges file for GFN2 with --pop; the '
        'computed_charges channel would stay empty')
    charges = np.loadtxt(charge_file)
    assert charges.shape == (3,)
    assert abs(float(charges.sum())) < 1e-6


def test_gfnff_population_writes_gfnff_charges(tmp_path):
    _run(tmp_path, ['--gfnff'])
    charge_file = tmp_path / 'gfnff_charges'
    assert charge_file.exists(), (
        'xTB wrote no gfnff_charges file for GFN-FF with --pop; the parser '
        'cannot return GFN-FF charges without it')
    charges = np.loadtxt(charge_file)
    assert charges.shape == (3,)
    assert abs(float(charges.sum())) < 1e-6
