"""
Unit tests for GFN-FF topology continuation in xtb_skill.

The backend keeps one calculation directory for its whole life, and xTB reads
`gfnff_topo` back from it, so an evaluation is scored against the bond list the
*previous* geometry produced. A dynamics run wants exactly that; a geometry
search does not, and the framework now says which by injecting
`keep_topology` from the jobtype. Measured in the test environment: carrying a
clashed run's topology onto a sane geometry moved its energy by +0.3508098
Hartree, and round tmp_20261017_234656's variable-cell leg reported +0.20 at
rows 6-21 while those very frames evaluate fresh to -0.929.

These tests do NOT require an xTB installation: the runner is replaced by a
recorder that stops the call where the engine would have started.

Run: pytest xtb_skill/tests/test_keep_topology.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os

import pytest

from ..backend import XtbBackend

ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]


class _StoppedAtTheEngine(Exception):
    """Raised by the recorder instead of launching xTB."""


def _backend(tmp_path, keep_topology):
    """Build a backend whose run() reports the directory and stops.

    Returns (backend, seen) where `seen` collects the directory listing taken
    at the moment the engine would have been invoked - the only point from which
    "was the previous topology visible to this evaluation" can be answered.
    """
    calcdir = os.path.join(str(tmp_path), 'tmp_XTB')
    # The backend validates that xtb_path exists, so a placeholder file stands
    # in for the installation: nothing here runs it, the recorder takes over
    # where the engine would have been invoked.
    fake_xtb = os.path.join(str(tmp_path), 'xtb')
    with open(fake_xtb, 'w') as handler:
        handler.write('#!/bin/sh\nexit 0\n')
    flat = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'profile': 'gfn_ff_gradient',
        'method': 'gfn_ff',
        'xtb_path': fake_xtb,
        'execution_mode': 'local',
    }
    if keep_topology:
        flat['keep_topology'] = True
    backend = XtbBackend(flat, calcdir=calcdir)
    seen = []

    def recorder(config, directory):
        seen.append(sorted(os.listdir(directory)))
        # Emulate the engine leaving its topology behind for the next call.
        with open(os.path.join(directory, 'gfnff_topo'), 'w') as handler:
            handler.write('fake bond list\n')
        raise _StoppedAtTheEngine()

    backend.run = recorder
    return backend, seen


def _calculate(backend, seen):
    with pytest.raises(_StoppedAtTheEngine):
        backend.calculate(ATOMS, COORDS)
    assert seen, 'the recorder never saw the call'


def test_a_geometry_search_rebuilds_the_topology_every_evaluation(tmp_path):
    """No carried bond list on the search path."""
    backend, seen = _backend(tmp_path, keep_topology=False)

    _calculate(backend, seen)
    assert 'gfnff_topo' not in seen[0], (
        'a fresh backend showed the engine a topology it never earned: %s'
        % seen[0])

    # Second call: the first left gfnff_topo in the fixed directory. It has to
    # be gone by the time the engine is invoked.
    _calculate(backend, seen)
    assert 'gfnff_topo' not in seen[1], (
        'the second evaluation was scored against the bond list the first '
        'geometry produced: %s' % seen[1])


def test_a_dynamics_run_keeps_the_topology_of_its_first_step(tmp_path):
    """The continuation the framework asks for when the jobtype is md."""
    backend, seen = _backend(tmp_path, keep_topology=True)

    _calculate(backend, seen)
    _calculate(backend, seen)
    assert 'gfnff_topo' in seen[1], (
        'a dynamics continuation lost the bond list its first step built: %s'
        % seen[1])


def test_the_default_is_the_safe_reading(tmp_path):
    """An unset key must not silently select the continuation."""
    from ..schema.adapter import build_config
    from ..schema.normalize import normalize

    config = normalize(build_config({'atoms': ATOMS, 'coordinates': COORDS}))
    assert getattr(config, 'keep_topology', None) is False, (
        'keep_topology is not a declared default, so a run that asks for '
        'neither route inherits whatever the previous evaluation left')


# The framework side of the split - deriving keep_topology from the jobtype -
# is checked in qualification/verification/test_workflow.py, which owns task
# planning; a skill test reaching into pyrai3md.data would invert the decoupling
# this key exists to preserve.

