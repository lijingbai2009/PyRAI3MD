"""
======================================================================
Test suite for the periodicity the shipped examples declare.

Every LAMMPS topology in this package carries a simulation box, so an
example that names one and stays silent about ``pbc`` is not a molecular
run: the data template still declares the box and the settings tail still
re-issues ``kspace_style``, and the engine evaluates the periodic system
while the config, the planned artifacts and the run log all claim the
opposite. Nothing crashes. What is lost is everything keyed on the
declaration - for a QM/MM field probe that is the pinned Ewald split and
the kspace channel of its two tallies, which is the A/B mismatch measured
in the test round tmp_20261031_005246.

The LAMMPS skill tests its own fixtures for this
(skills/lammps_skill/tests/test_pbc_declaration.py). These tests hold the
shipped examples to the same rule, and add the half the fixtures cannot
see: a declaration is only truthful while the cell it states is the box
the topology was built for.

Oct 31 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import glob
import os
import unittest

import yaml

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_EXAMPLES_DIR = os.path.join(_PACKAGE_DIR, 'examples')
# A topology ships either beside the examples that run it or with the
# skill's own fixture set; both are searched, and a name found in neither
# is a failure rather than a skip, so the scan cannot pass by reading
# nothing at all.
_TOPOLOGY_DIRS = (_EXAMPLES_DIR,
                  os.path.join(_PACKAGE_DIR, 'skills', 'lammps_skill',
                               'tests'))


def _box_lengths(project):
    """The three axis lengths a LAMMPS topology file declares.

    The data file states the *bounding* planes, so a tilted or offset box
    reads as ``hi - lo`` on each axis - the same arithmetic the skill's
    own fixture test uses.
    """
    for directory in _TOPOLOGY_DIRS:
        path = os.path.join(directory, '%s.data' % project)
        if not os.path.isfile(path):
            continue
        lengths = []
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle.read().splitlines():
                tokens = line.split()
                if len(tokens) == 4 and tokens[2:] in (['xlo', 'xhi'],
                                                       ['ylo', 'yhi'],
                                                       ['zlo', 'zhi']):
                    lengths.append(float(tokens[1]) - float(tokens[0]))
        if len(lengths) == 3:
            return lengths
        break
    raise AssertionError(
        'topology %s.data declares no complete box in %s'
        % (project, ' or '.join(_TOPOLOGY_DIRS)))


def _lammps_examples():
    """The shipped examples that name a LAMMPS topology.

    Returns
    -------
    list of tuple
        ``(file name, topology project, molecule section)`` for every
        example whose ``lammps`` block carries ``lammps_project``.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(_EXAMPLES_DIR, '*.yaml'))):
        with open(path, 'r', encoding='utf-8') as handle:
            document = yaml.safe_load(handle)
        if not isinstance(document, dict):
            continue
        section = document.get('input') or document
        lammps = section.get('lammps')
        if not isinstance(lammps, dict):
            continue
        project = lammps.get('lammps_project')
        if not project:
            continue
        molecule = section.get('molecule') or {}
        system = section.get('system') or {}
        found.append((os.path.basename(path), project, molecule, system))
    return found


class TestExamplePeriodicityDeclaration(unittest.TestCase):
    """An example running a boxed topology says whether it is periodic."""

    def setUp(self):
        self.cases = _lammps_examples()

    def test_the_scan_finds_both_branches(self):
        # Non-identity witness on the scan itself: a rule asserted over an
        # empty or one-sided set passes without ever being exercised, so
        # the population is counted before any per-case claim is read.
        self.assertGreater(len(self.cases), 20,
                           'the scan found %d LAMMPS examples'
                           % len(self.cases))
        periodic = [case for case in self.cases
                    if any(int(flag) for flag in
                           (case[2].get('pbc') or case[3].get('pbc') or []))]
        molecular = [case for case in self.cases if case not in periodic]
        self.assertGreater(len(periodic), 10,
                           'only %d periodic examples' % len(periodic))
        self.assertGreater(len(molecular), 2,
                           'only %d molecular examples' % len(molecular))

    def test_every_example_declares_periodicity(self):
        undeclared = []
        for name, project, molecule, system in self.cases:
            if 'pbc' not in molecule and 'pbc' not in system:
                undeclared.append('%s (%s)' % (name, project))
        self.assertEqual(undeclared, [],
                         'these examples run a boxed LAMMPS topology '
                         'without declaring pbc, so the engine evaluates '
                         'the periodic system while everything keyed on '
                         'the declaration reads it as molecular: %s'
                         % ', '.join(undeclared))

    def test_a_declared_cell_is_the_topology_box(self):
        # The declaration is only truthful while it agrees with the box
        # the force field was built for: a mismatch silently resizes the
        # system every case runs on, because pbc on makes the framework
        # cell - not the template - the source of the simulation box.
        #
        # Compared relatively, not exactly: the examples state the box to
        # six decimals (25.237532 for a topology holding 25.2375321966),
        # so the rounding is at most 5e-7 absolute, ~2e-8 relative on a
        # 25 A box. 1e-5 leaves that two orders of margin, while a cell
        # taken from the wrong topology differs by ~1%.
        wrong = []
        for name, project, molecule, system in self.cases:
            flags = molecule.get('pbc') or system.get('pbc') or []
            if not any(int(flag) for flag in flags):
                continue
            cell = molecule.get('cell') or system.get('cell')
            if not cell:
                wrong.append('%s declares pbc but no cell' % name)
                continue
            lengths = _box_lengths(project)
            for axis, (declared, held) in enumerate(zip(cell[:3], lengths)):
                if abs(float(declared) - held) > 1e-5 * held:
                    wrong.append('%s axis %d declares %.9g, %s.data holds '
                                 '%.9g' % (name, axis, float(declared),
                                           project, held))
            if [float(value) for value in cell[3:]] != [90.0, 90.0, 90.0]:
                wrong.append('%s declares a tilted cell %s'
                             % (name, cell[3:]))
        self.assertEqual(wrong, [], '; '.join(wrong))

    def test_a_molecular_example_names_a_topology_that_has_a_box(self):
        # The other half of the rule, and the reason the molecular cases
        # declare pbc rather than omit it: the topology really does carry
        # a box, so `[0, 0, 0]` is a decision about that box and not an
        # absence of one.
        molecular = [(name, project)
                     for name, project, molecule, system in self.cases
                     if not any(int(flag) for flag in
                                (molecule.get('pbc') or system.get('pbc')
                                 or []))]
        self.assertTrue(molecular, 'no molecular LAMMPS example to check')
        for name, project in molecular:
            self.assertEqual(len(_box_lengths(project)), 3,
                             '%s runs %s, which declares no box'
                             % (name, project))


if __name__ == '__main__':
    unittest.main()
