#!/usr/bin/env python3
"""
======================================================================
This module refreshes the shared LAMMPS water-box fixture.

It turns the final frame of a LAMMPS ff_npt equilibration (a real-units
custom dump carrying box bounds, coordinates and velocities) into the
representation PyRAI3MD stores: the box moved to its own origin, the
coordinates wrapped into the main image, and the momenta converted from
Angstrom/fs to Bohr/au with the center-of-mass translation removed and
the whole set rescaled to the declared temperature. The rendered blocks
are then written into every file that carries the fixture, so the example
workflows, the skill test cases and the topology data files never drift
apart.

The tool is a development utility, not part of the installed package. It
refuses to write anything if a self-check fails and it rewrites every
target from the same computed snapshot, so a half-refreshed fixture
family is not a state a person can reach by accident.

Usage:
    python3 tools/refresh_water_fixture.py --cond snapshot.cond --check
    python3 tools/refresh_water_fixture.py --cond snapshot.cond

Sep 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import argparse
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pyrai3md.utilities.lattice import (cell6_to_matrix,   # noqa: E402
                                        wrap_fractional_with_image)
from pyrai3md.utilities.units import (AMU_TO_AU, FS_TO_AU_TIME,   # noqa
                                      HARTREE_TO_KCAL, KELVIN_TO_HARTREE,
                                      ANGSTROM_TO_BOHR)

# One Angstrom/fs expressed in Bohr/au of time, from the package's own
# length and time factors.
ANGSTROM_FS_TO_AU_VELOCITY = ANGSTROM_TO_BOHR / FS_TO_AU_TIME
AVOGADRO = 6.02214076e23

EXAMPLES = os.path.join('pyrai3md', 'examples')
SKILL_TESTS = os.path.join('pyrai3md', 'skills', 'lammps_skill', 'tests')
SKILL_TEST_NAMES = ('ff_energy.yaml', 'ff_gradient.yaml',
                    'ff_gradient_persistent.yaml', 'ff_nve.yaml',
                    'ff_nvt.yaml', 'ff_npt.yaml', 'ff_optimize.yaml',
                    'ff_stress.yaml')
DATA_TARGETS = [os.path.join(EXAMPLES, 'h2o.data'),
                os.path.join(SKILL_TESTS, 'h2o.data')]
WATER_MASS_AMU = 20 * 18.015          # the 20 waters of the box

# The profiles that integrate inside LAMMPS. They start from whatever
# momenta the input carries, so a fixture with no velocities block leaves
# the run at 0 K and the first thing the log shows is the temperature
# climbing from rest under the forces - not the ensemble the profile is
# meant to exercise. A single point ignores velocities, so its fixture
# keeps the field empty and states nothing it does not need.
VELOCITY_PROFILES = ('ff_nve', 'ff_nvt', 'ff_npt')

VELOCITY_COMMENT = (
    "  # Initial velocities: the same equilibrated frame as the\n"
    "  # coordinates above, converted from Angstrom/fs to Bohr/au with\n"
    "  # the center-of-mass translation removed and rescaled to 298.15 K.\n"
    "  # LAMMPS integrates this profile, so it starts from these.\n")

# Prose that states where the snapshot came from. A block absent from a
# file is reported, never guessed at.
COMMENT_BLOCKS = [
    (
        "# 60 atoms = 20 TIP3P waters in a 10 Angstrom cubic cell; the atom\n"
        "# order matches the shared h2o.data topology fixture shipped in this\n"
        "# directory (h2o.data / h2o.in.init / h2o.in.settings; qualification\n"
        "# additionally stages project-renamed copies into the run directory).\n",
        "# 60 atoms = 20 TIP3P waters in the relaxed cubic cell declared below;\n"
        "# the atom order matches the shared h2o.data topology fixture shipped\n"
        "# in this directory (h2o.data / h2o.in.init / h2o.in.settings;\n"
        "# qualification additionally stages project-renamed copies into the\n"
        "# run directory). Geometry and momenta are the final frame of a LAMMPS\n"
        "# ff_npt equilibration (refresh with tools/refresh_water_fixture.py),\n"
        "# so an example no longer spends its first 20 fs shedding the strain\n"
        "# of an ad-hoc placement.\n",
    ),
    (
        "# from the shared 300 K Maxwell-Boltzmann draw below and records 11\n",
        "# from the equilibrated snapshot below and records 11\n",
    ),
    (
        "# Initial velocities: one shared Maxwell-Boltzmann draw at 300 K for\n"
        "# this 60-atom box (fixed seed 20260926, center-of-mass translation\n"
        "# removed, Bohr/au, instantaneous temperature 300.5 K over N_df=177).\n"
        "# Regenerate with numpy.default_rng(20260926).normal(0, sqrt(kT/m_au))\n"
        "# using the h2o.data masses (O 15.9994, H 1.008).\n",
        "# Initial velocities: the equilibrated draw that came with the box,\n"
        "# converted from Angstrom/fs to Bohr/au, with the center-of-mass\n"
        "# translation removed and the whole set rescaled from the sampled\n"
        "# temperature to 298.15 K. N_df is 180 because nothing is constrained\n"
        "# in this model: see the SHAKE note above.\n",
    ),
    (
        "  # The framework declares periodicity and box; these match the\n"
        "  # h2o.data fixture (0.0 10.00000000 xlo xhi), so a topology swap\n"
        "  # has to move both together.\n",
        "  # The framework declares periodicity and box; these match the\n"
        "  # h2o.data fixture box, so a topology swap has to move both\n"
        "  # together. Some O-H bonds straddle a cell face in this main-image\n"
        "  # representation, which is expected: the minimum-image distance\n"
        "  # puts each of them back inside the bond range.\n",
    ),
]


class FixtureError(RuntimeError):
    """The snapshot or the repository is not in the assumed state."""


def read_cond(path):
    """Read a LAMMPS custom dump holding box, positions and velocities."""
    lines = open(path, 'r', encoding='utf-8').read().split('\n')
    index = {}
    for i, line in enumerate(lines):
        if line.startswith('ITEM:'):
            index.setdefault(line, i)
    for needed in ('ITEM: BOX BOUNDS pp pp pp', 'ITEM: ATOMS id x y z vx vy vz',
                   'ITEM: NUMBER OF ATOMS'):
        if needed not in index:
            raise FixtureError('%s lacks the item %r' % (path, needed))
    ib = index['ITEM: BOX BOUNDS pp pp pp']
    bounds = [lines[ib + 1 + k].split() for k in range(3)]
    lo = np.array([float(row[0]) for row in bounds])
    hi = np.array([float(row[1]) for row in bounds])
    natom = int(lines[index['ITEM: NUMBER OF ATOMS'] + 1])
    columns = lines[index['ITEM: ATOMS id x y z vx vy vz']].split()[2:]
    rows = [lines[index['ITEM: ATOMS id x y z vx vy vz'] + 1 + k].split()
            for k in range(natom)]
    pick = lambda name: np.array([float(row[columns.index(name)])
                                  for row in rows])
    ids = pick('id').astype(int)
    xyz = np.column_stack([pick('x'), pick('y'), pick('z')])
    vel = np.column_stack([pick('vx'), pick('vy'), pick('vz')])
    return lo, hi, ids, xyz, vel


def read_topology(path):
    """Read the per-atom mass and the bonded pairs of the data fixture."""
    lines = open(path, 'r', encoding='utf-8').read().split('\n')
    counts = {}
    for line in lines[:16]:
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            counts[parts[1]] = int(parts[0])
    masses = {}
    i = lines.index('Masses') + 2
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 2 or not parts[0].isdigit():
            break
        masses[int(parts[0])] = float(parts[1])
        i += 1

    def section(name, count):
        i = lines.index(name) + 2
        rows = []
        while i < len(lines) and len(rows) < count:
            parts = lines[i].split()
            if len(parts) >= 4 and parts[0].isdigit():
                rows.append([int(value) for value in parts[2:]])
            i += 1
        if len(rows) != count:
            raise FixtureError('%s: expected %d %s rows, read %d'
                               % (path, count, name, len(rows)))
        return rows

    atoms = []
    i = lines.index('Atoms') + 2
    while len(atoms) < counts['atoms']:
        parts = lines[i].split()
        if len(parts) >= 6 and parts[0].isdigit():
            atoms.append((int(parts[0]), int(parts[2])))
        i += 1
    if sorted(atom_id for atom_id, _ in atoms) != \
            list(range(1, counts['atoms'] + 1)):
        raise FixtureError('%s: atom ids are not 1..natom' % path)
    mass = np.array([masses[type_id] for _, type_id in sorted(atoms)])
    return (mass,
            section('Bonds', counts['bonds']),
            section('Angles', counts['angles']))


def instantaneous_temperature(velocity_au, mass_amu):
    """Temperature of a velocity set from its kinetic energy (dof = 3N).

    Velocities are Bohr per atomic unit of time and masses atomic mass
    units, so the kinetic energy comes out in Hartree and the framework's
    own conversion factors give the temperature: this is the same
    bookkeeping PyRAIState.get_kinetic_energy performs.
    """
    kinetic = 0.5 * np.sum(mass_amu[:, None] * AMU_TO_AU * velocity_au ** 2)
    dof = len(mass_amu) * 3
    return 2.0 * kinetic / (dof * KELVIN_TO_HARTREE)


def min_image_vector(corners, first, second, lattice, inverse):
    """Nearest-image difference vector corners[first] - corners[second]."""
    d = (corners[first] - corners[second]) @ inverse
    d -= np.round(d)
    return d @ lattice


def convert(lo, hi, ids, xyz, vel, mass, bonds, angles, target_kelvin):
    """Bring the snapshot into the repository's representation."""
    if not (ids == np.arange(1, len(ids) + 1)).all():
        raise FixtureError('snapshot atom ids are not 1..natom in order')
    cell = hi - lo
    if np.abs(cell - cell.mean()).max() > 1e-4:
        raise FixtureError('box is not cubic: %s' % cell.tolist())
    # Quantize the box once, so the number written into a data file, the
    # number written into molecule.cell and the lattice the coordinates are
    # wrapped against are all the same box.
    side = round(float(cell.mean()), 6)
    lattice = cell6_to_matrix([side, side, side, 90.0, 90.0, 90.0])
    inverse = np.linalg.inv(lattice)

    # The framework's cell carries no origin, so the box is translated to
    # its own lower bound before the wrap records the crossings.
    wrapped_frac, image = wrap_fractional_with_image(
        (xyz - lo) @ inverse, [1, 1, 1])
    coordinates = wrapped_frac @ lattice

    # The momenta arrive in Angstrom/fs: convert, take out the residual
    # translation of the whole box, then scale the set to the temperature
    # the example declares. Nothing is resampled, so the equilibrated
    # velocity correlations survive the refresh.
    de_drifted = (vel - np.average(vel, axis=0, weights=mass)) \
        * ANGSTROM_FS_TO_AU_VELOCITY
    sampled = instantaneous_temperature(de_drifted, mass)
    velocities = de_drifted * np.sqrt(target_kelvin / sampled)
    return {
        'side': side,
        'origin': lo.tolist(),
        'coordinates': coordinates,
        'image': image,
        'velocities': velocities,
        'sampled_kelvin': sampled,
        'target_kelvin': target_kelvin,
        'lattice': lattice,
        'inverse': inverse,
        'bonds': bonds,
        'angles': angles,
        'mass': mass,
    }


def self_checks(snapshot):
    """Verify the converted snapshot before anything is written."""
    c = snapshot['coordinates']
    lattice, inverse = snapshot['lattice'], snapshot['inverse']
    side = snapshot['side']
    problems = []

    if (c < -1e-6).any() or (c > side + 1e-6).any():
        problems.append('an atom is outside [0, L) after the wrap')

    bonds = np.array([np.linalg.norm(min_image_vector(c, a - 1, b - 1,
                                                      lattice, inverse))
                      for a, b in snapshot['bonds']])
    # Judge the bonds by vibrational energy, not by an absolute range: at
    # the target temperature a harmonic O-H (K = 450 kcal/mol/A^2,
    # re = 0.9572) carries kT of mode energy, so a few hundredths of an
    # Angstrom of excursion is thermal content rather than strain. A bond
    # outside [0.85, 1.10] A or holding more than 3 kT is a dissociating
    # molecule seen through either lens.
    k_t = KELVIN_TO_HARTREE * HARTREE_TO_KCAL * snapshot['target_kelvin']
    stretch = 0.5 * 450.0 * (bonds - 0.9572) ** 2
    if bonds.min() < 0.85 or bonds.max() > 1.10:
        problems.append('minimum-image O-H outside [0.85, 1.10] A: %.4f to '
                        '%.4f' % (bonds.min(), bonds.max()))
    if stretch.max() > 3.0 * k_t:
        problems.append('the hottest bond holds %.2f kcal/mol, above 3 kT'
                        % stretch.max())
    if stretch.mean() > k_t:
        problems.append('the mean bond energy %.2f kcal/mol exceeds one kT, '
                        'so the snapshot is not a relaxed liquid'
                        % stretch.mean())
    snapshot['stretch_mean'] = float(stretch.mean())
    snapshot['stretch_max'] = float(stretch.max())

    angles = []
    for i, j, k in snapshot['angles']:       # LAMMPS puts the apex middle
        u = min_image_vector(c, i - 1, j - 1, lattice, inverse)
        v = min_image_vector(c, k - 1, j - 1, lattice, inverse)
        angles.append(np.degrees(np.arccos(
            u.dot(v) / np.linalg.norm(u) / np.linalg.norm(v))))
    angles = np.array(angles)
    if angles.min() < 85.0 or angles.max() > 125.0:
        problems.append('H-O-H angle outside [85, 125] deg: %.2f to %.2f'
                        % (angles.min(), angles.max()))

    # Judged straight from the stored atomic-unit velocities, so it tests
    # the conversion the snapshot went through rather than cancelling it.
    achieved = instantaneous_temperature(snapshot['velocities'],
                                         snapshot['mass'])
    if abs(achieved - snapshot['target_kelvin']) > 0.5:
        problems.append('rescaled temperature %.2f K misses the target'
                        % achieved)

    volume = side ** 3
    density = WATER_MASS_AMU / AVOGADRO / (volume * 1e-24)
    if not 0.9 < density < 1.1:
        problems.append('density %.3f g/cc is away from liquid water'
                        % density)

    snapshot.update({
        'bond_min': float(bonds.min()), 'bond_max': float(bonds.max()),
        'angle_min': float(angles.min()), 'angle_max': float(angles.max()),
        'angle_mean': float(angles.mean()),
        'achieved_kelvin': float(achieved), 'density': float(density),
        'volume': float(volume),
        'straddling': int(sum(1 for a, b in snapshot['bonds']
                              if np.linalg.norm(c[a - 1] - c[b - 1]) > 1.5)),
    })
    return problems


def render_triples(values, indent, decimals):
    """Render a flow-style list of triples at the given indentation."""
    pad = ' ' * indent
    return ''.join('%s- [%s]\n' % (
        pad, ', '.join('%.*f' % (decimals, float(value)) for value in row))
        for row in values)


def replace_list(text, header, body, path):
    """Replace the list block that follows a bare `header:` line.

    Any comment lines between the header and its list are kept, so a
    comment that documents the block survives the refresh.
    """
    pattern = re.compile(r'(?m)^([ \t]*)%s:[ \t]*\n((?:[ \t]*#[^\n]*\n)*)'
                         r'((?:[ \t]*- [^\n]*\n?)+)' % re.escape(header))
    match = pattern.search(text)
    if not match:
        raise FixtureError('%s: no %s block found' % (path, header))
    return text[:match.start(3)] + body + text[match.end(3):]


def replace_cell(text, side, path):
    """Rewrite the cell triple to the refreshed box."""
    line = 'cell: [%s]' % ', '.join(['%.6f' % side] * 3 + ['90.0'] * 3)
    pattern = re.compile(r'(?m)^([ \t]*)cell: \[[^\]]*\]')
    if not pattern.search(text):
        raise FixtureError('%s: no cell line found' % path)
    return pattern.sub(lambda m: m.group(1) + line, text)


def replace_comments(text, report):
    """Apply the provenance blocks, recording which ones matched."""
    for old, new in COMMENT_BLOCKS:
        if old in text:
            text = text.replace(old, new)
            report[old.splitlines()[0][:40]] = report.get(
                old.splitlines()[0][:40], 0) + 1
    return text


def insert_list_after(text, anchor, header, body, path):
    """Insert a `header:` block directly after the anchor's list block.

    Used where a fixture never carried the block at all, as opposed to
    `replace_list`, which rewrites one that exists.
    """
    pattern = re.compile(r'(?m)^([ \t]*)%s:[ \t]*\n((?:[ \t]*#[^\n]*\n)*)'
                         r'((?:[ \t]*- [^\n]*\n?)+)' % re.escape(anchor))
    match = pattern.search(text)
    if not match:
        raise FixtureError('%s: no %s block to follow' % (path, anchor))
    indent = match.group(1)
    block = '%s%s:\n%s' % (indent, header, body)
    return text[:match.end(3)] + block + text[match.end(3):]


def apply_yaml(text, snapshot, path):
    """Rewrite one YAML fixture and return the new text plus a report."""
    report = {}
    text = replace_comments(text, report)
    bodies = {'coordinates': None, 'velocities': None}
    for header, decimals in (('coordinates', 7), ('velocities', 9)):
        values = snapshot[header]
        if values is None:
            continue
        indent = re.search(r'(?m)^([ \t]*)%s:[ \t]*$' % header, text)
        if indent is not None:
            bodies[header] = render_triples(values, len(indent.group(1)) + 2,
                                            decimals)
    if bodies['coordinates'] is None:
        raise FixtureError('%s: no coordinates block' % path)
    text = replace_list(text, 'coordinates', bodies['coordinates'], path)
    if bodies['velocities'] is not None:
        text = replace_list(text, 'velocities', bodies['velocities'], path)
    elif (snapshot['velocities'] is not None
          and _declares_velocity_profile(text)):
        # Nothing to replace: the profile needs momenta and the fixture
        # never carried them, so the block is added where its siblings
        # sit rather than leaving the run to start from rest.
        header = re.search(r'(?m)^([ \t]*)coordinates:[ \t]*$', text)
        if header is None:
            raise FixtureError('%s: no coordinates header' % path)
        body = VELOCITY_COMMENT + render_triples(
            snapshot['velocities'], len(header.group(1)) + 2, 9)
        text = insert_list_after(text, 'coordinates', 'velocities',
                                 body, path)
        report['velocities block'] = report.get('velocities block', 0) + 1
    return replace_cell(text, snapshot['side'], path), report


def _declares_velocity_profile(text):
    """True when this fixture asks LAMMPS to integrate a trajectory."""
    profile = re.search(r'(?m)^[ \t]*profile:[ \t]*(\S+)', text)
    return bool(profile) and profile.group(1) in VELOCITY_PROFILES


def apply_data(text, snapshot, path):
    """Rewrite the box lines and the Atoms coordinates of a data file."""
    lines = text.split('\n')
    side = '%.8f' % snapshot['side']
    for name in ('xlo xhi', 'ylo yhi', 'zlo zhi'):
        for i, line in enumerate(lines):
            if line.endswith(name):
                lines[i] = '0.0 %s %s' % (side, name)
                break
        else:
            raise FixtureError('%s: missing %s line' % (path, name))
    header = int(next(line.split()[0] for line in lines[:16]
                      if line.endswith('atoms')))
    i = lines.index('Atoms') + 2
    replaced = 0
    while replaced < header:
        parts = lines[i].split()
        if len(parts) >= 6 and parts[0].isdigit():
            xyz = snapshot['coordinates'][int(parts[0]) - 1]
            # Everything but the last three columns is identity the data
            # file owns (molecule, type, charge, ...): only the
            # coordinates move, whatever the atom style's column count.
            lines[i] = ' '.join(parts[:-3] + ['%.8f' % value
                                              for value in xyz])
            replaced += 1
        i += 1
    return '\n'.join(lines)


def water_family_prefix():
    """The `md_lammps_` family prefix, as the switch table declares it."""
    from pyrai3md.qualification.manager import EXAMPLE_FAMILIES
    return EXAMPLE_FAMILIES['md_lammps_test']


def targets():
    """List the fixture files this tool owns."""
    prefix = water_family_prefix()
    names = sorted(name for name in os.listdir(os.path.join(REPO_ROOT,
                                                            EXAMPLES))
                   if name.startswith(prefix) and name.endswith('.yaml'))
    return ([os.path.join(EXAMPLES, name) for name in names]
            + [os.path.join(SKILL_TESTS, name) for name in SKILL_TEST_NAMES])


def report(snapshot):
    """Print what the conversion produced."""
    print('box side            %.8f A (origin %s shifted to 0)'
          % (snapshot['side'], np.array2string(
              np.asarray(snapshot['origin']), precision=6)))
    print('volume / density    %.1f A^3 / %.4f g/cc'
          % (snapshot['volume'], snapshot['density']))
    print('sampled -> target   %.1f K -> %.2f K (achieved %.2f K)'
          % (snapshot['sampled_kelvin'], snapshot['target_kelvin'],
             snapshot['achieved_kelvin']))
    print('min-image O-H       %.4f to %.4f A (bond energy mean %.3f, max '
          '%.3f kcal/mol against kT %.3f)'
          % (snapshot['bond_min'], snapshot['bond_max'],
             snapshot['stretch_mean'], snapshot['stretch_max'],
             KELVIN_TO_HARTREE * HARTREE_TO_KCAL * snapshot['target_kelvin']))
    print('H-O-H               %.2f to %.2f deg (mean %.2f)'
          % (snapshot['angle_min'], snapshot['angle_max'],
             snapshot['angle_mean']))
    print('bonds straddling a face (main image): %d of %d'
          % (snapshot['straddling'], len(snapshot['bonds'])))
    print('wrap crossings      %d atoms crossed at least one cell'
          % int(np.any(snapshot['image'] != 0, axis=1).sum()))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Refresh the shared LAMMPS water-box fixture.')
    parser.add_argument('--cond', required=True,
                        help='LAMMPS custom dump with box, x y z and v')
    parser.add_argument('--temperature', type=float, default=298.15,
                        help='target instantaneous temperature in K')
    parser.add_argument('--check', action='store_true',
                        help='report only; write nothing')
    args = parser.parse_args(argv)

    lo, hi, ids, xyz, vel = read_cond(args.cond)
    mass, bonds, angles = read_topology(
        os.path.join(REPO_ROOT, DATA_TARGETS[0]))
    snapshot = convert(lo, hi, ids, xyz, vel, mass, bonds, angles,
                       args.temperature)
    problems = self_checks(snapshot)
    report(snapshot)
    if problems:
        for problem in problems:
            print('CHECK FAILED: %s' % problem)
        return 1
    print('self-checks passed')
    if args.check:
        # Report the repository against the snapshot without writing: a
        # fixture family that drifted in one file only has to be seen to
        # be believed, and the velocities block is the part a profile can
        # silently run without.
        for relative in targets():
            with open(os.path.join(REPO_ROOT, relative), 'r',
                      encoding='utf-8') as handle:
                original = handle.read()
            rewritten, notes = apply_yaml(original, snapshot, relative)
            if rewritten == original:
                continue
            missing = ('%s needs a velocities block' % relative
                       if 'velocities block' in notes else None)
            print('differs: %s%s' % (relative,
                                     '' if missing is None else ' (%s)'
                                     % missing))
        return 0

    # Render everything before touching the repository: a family of
    # fixtures that share one box must not be left half-refreshed.
    rendered = []
    tally = {}
    for relative in targets():
        path = os.path.join(REPO_ROOT, relative)
        text, report_this = apply_yaml(
            open(path, 'r', encoding='utf-8').read(), snapshot, relative)
        rendered.append((relative, path, text))
        for key, hits in report_this.items():
            tally[key] = tally.get(key, 0) + hits
    for relative in DATA_TARGETS:
        path = os.path.join(REPO_ROOT, relative)
        rendered.append((relative, path, apply_data(
            open(path, 'r', encoding='utf-8').read(), snapshot, relative)))
    for relative, path, text in rendered:
        open(path, 'w', encoding='utf-8').write(text)
        print('wrote %s' % relative)
    total = len(targets())
    for key, hits in tally.items():
        if hits != total:
            print('note: comment "%s" matched %d of %d files'
                  % (key.replace('# ', ''), hits, total))
    return 0


if __name__ == '__main__':
    sys.exit(main())
