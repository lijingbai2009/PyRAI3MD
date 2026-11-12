"""
Byte-identity guard for the LAMMPS input generation.

The resident ('python_enabled') work has to hand the very same command
sequence to a live LAMMPS instance that the subprocess path writes to
disk, so the text builders of service/builder are shared by both paths.
Sharing means refactoring, and refactoring a generator whose output a
physics engine consumes is exactly the change a substring test cannot
catch: a dropped space or a reordered line still satisfies every
'assert x in text'.

This module therefore freezes the whole generated text of every output
file, for each profile, with and without a compressed force-field
model, and with the stress request on and off. The digests below were
recorded against the pre-refactor tree; any drift in any character of
any generated file fails here.

The internal-iteration profiles ignore the stress flag (it only adds a
pressure compute to the single-point tail), so their model/verbatim
digests repeat across the two stress values on purpose.

These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_input_builder_golden.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import hashlib
import os

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from ..service.builder import write_input_file
from ..service.builder.ff_model import build_ff_model
from ..service.workflow import compile_workflow

# A single water molecule with the full set of counts, so the compressor
# has something to prove and the verbatim branch has something to copy.
DATA = """LAMMPS Description

3  atoms
2  bonds
1  angles
0  dihedrals
0  impropers

2  atom types
1  bond types
1  angle types
0  dihedral types
0  improper types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 15.999
2 1.008

Atoms

1 1 1 -0.834 0.0000000000 0.0000000000 0.1173000000
2 1 2 0.417 0.0000000000 0.7572000000 -0.4692000000
3 1 2 0.417 0.0000000000 -0.7572000000 -0.4692000000

Bonds

1 1 1 2
2 1 1 3

Angles

1 1 2 1 3
"""

INIT = """units real
atom_style full
bond_style harmonic
angle_style harmonic
pair_style lj/cut 9.0
kspace_style pppm 1e-4
"""

SETTINGS = """pair_coeff 1 1 lj/cut 0.1 3.0
pair_coeff 1 2 lj/cut 0.05 2.5
pair_coeff 2 2 lj/cut 0.02 2.0
bond_coeff 1 harmonic 450.0 0.9572
angle_coeff 1 harmonic 55.0 104.52
group wat type 1 2
fix fShakeTIP3P wat shake 0.0001 10 100 b 1 a 1
"""

# Execution commands the internal-iteration profiles drop and the
# single-point profile keeps, so both branches of the strip logic are
# inside the frozen text.
MAIN = """include "LAMMPS.in.init"
read_data "LAMMPS.data"
include "LAMMPS.in.settings"
thermo 100
thermo_style multi
velocity all create 300 12345
fix old_name all nvt temp 300 300 100
run 999
unfix old_name
"""

ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]

# {profile|stress|model}: sha256 over every generated file, recorded
# against the pre-refactor generator.
#
# Re-recorded for #234, with the drift accounted for rather than absorbed:
# the eight ff_energy/ff_gradient digests move because their tails now name
# {title}.energy/.force/.stress/.coords instead of {project}_force.xyz and
# {project}_stress.dat, and the eight MD/optimize 'model' digests move only
# because {project}.ff.json became a member of the package. The text of
# those eight inputs is byte-identical to the recorded generation, checked
# by re-digesting each case with the model file excluded; the four
# MD/optimize 'verbatim' digests are unchanged outright, which is what
# proves no execution text moved outside the single-point profiles.
# Re-recorded again on 2026-10-01, and for a narrower reason than the
# digest can show: only the twelve '|model' cases moved, because the
# constraint audit sentence inside <project>.ff.json was reworded. The
# ownership rule did not change, only the text explaining a dropped fix.
# The twelve '|verbatim' digests are untouched, and that no execution text
# moved was measured rather than assumed: hashing the four LAMMPS files
# alone, excluding <project>.ff.json, gives byte-identical digests for all
# twelve model cases before and after the rewording.
GOLDEN = {
    # Re-recorded when a periodic run began stating its image flags in the
    # trailing Atoms columns (dev/methods/PBC_DEVELOPMENT.md stage C). The
    # drift was shown to be confined to those columns rather than accepted
    # on trust: with the column writer stubbed out, all 24 cases below
    # reproduce the digests recorded before the change, so no execution
    # text moved along with the three integers. The four ff_npt digests
    # moved again with it, for a different and intended reason: the npt fix
    # now honours barostat_mode and couples the cell isotropically the way
    # the framework default says, where it used to scale three independent
    # axes whatever the input declared.
    #
    # The twelve `model` digests moved again when `ewald_accuracy` joined
    # `config_hash`, and that move was bounded rather than trusted, the same
    # way as the one above: run with the key left inert, all twelve reproduce
    # these pre-change digests (12/12), and run with it declared in the
    # spelling the template itself uses, the only file that differs from the
    # inert run is `LAMMPS.ff.json`, whose `config_hash` field is the whole
    # difference - every generated input script is byte-for-byte what it was.
    # The `verbatim` digests do not move at all: that path writes no manifest,
    # so nothing keyed on the config identity is in the directory.
    'ff_energy|0|model': 'a134f047dff260b8',
    'ff_energy|0|verbatim': 'f85aa4bf059fcfec',
    'ff_energy|1|model': '3fe0c8355803c57a',
    'ff_energy|1|verbatim': '30e259ddc31eeb25',
    'ff_gradient|0|model': '050afc630d677840',
    'ff_gradient|0|verbatim': 'af69e8cd3d476a43',
    'ff_gradient|1|model': '1a83196765ab882b',
    'ff_gradient|1|verbatim': '846281503c1a9105',
    'ff_optimize|0|model': 'f9e9934d0bf4675e',
    'ff_optimize|0|verbatim': 'b7e1817d49044c88',
    'ff_optimize|1|model': 'f9e9934d0bf4675e',
    'ff_optimize|1|verbatim': 'b7e1817d49044c88',
    'ff_nve|0|model': '93badaabe3975173',
    'ff_nve|0|verbatim': '5ac2f6c09125592d',
    'ff_nve|1|model': '93badaabe3975173',
    'ff_nve|1|verbatim': '5ac2f6c09125592d',
    'ff_nvt|0|model': '10ae62545ef494ce',
    'ff_nvt|0|verbatim': '5715eb56189af274',
    'ff_nvt|1|model': '10ae62545ef494ce',
    'ff_nvt|1|verbatim': '5715eb56189af274',
    'ff_npt|0|model': 'b9f994885e48f768',
    'ff_npt|0|verbatim': 'd09d14267644e74c',
    'ff_npt|1|model': 'b9f994885e48f768',
    'ff_npt|1|verbatim': 'd09d14267644e74c',
}

CASES = sorted(GOLDEN)


def _make_config(profile: str, stress: bool):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'profile': profile,
        'method': 'ff',
        'lammps_project': 'LAMMPS',
        'lammps_units': 'real',
        # These goldens stand for the classical cases, which keep the
        # accuracy their template declares instead of the skill default.
        # Spelled exactly as INIT spells it, so declaring it costs no text:
        # a case that wants the template's number writes that number, and the
        # only thing left to move is the manifest digest, which gains the key
        # on purpose.
        'ewald_accuracy': '1e-4',
        'pbc': [1, 1, 1],
        'lattice_vector': [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]],
    }
    if stress:
        raw['stress'] = True
    return validate(normalize(build_config(raw)))


def _digest(calcdir: str) -> str:
    """Fold every generated file into one digest, name included."""
    hasher = hashlib.sha256()
    names = sorted(os.listdir(calcdir))
    assert names, 'write_input_file produced no output'
    for name in names:
        hasher.update(name.encode('utf-8'))
        hasher.update(b'\0')
        with open(os.path.join(calcdir, name), 'rb') as handle:
            hasher.update(handle.read())
    return hasher.hexdigest()[:16]


@pytest.fixture
def templates(tmp_path, monkeypatch):
    """Launch directory carrying all four user templates."""
    for name, text in (('LAMMPS.data', DATA), ('LAMMPS.in.init', INIT),
                       ('LAMMPS.in.settings', SETTINGS),
                       ('LAMMPS.in', MAIN)):
        (tmp_path / name).write_text(text)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize('case', CASES)
def test_generated_text_is_byte_identical(templates, case):
    profile, stress, kind = case.split('|')
    config = _make_config(profile, stress == '1')
    model = (build_ff_model(config, profile) if kind == 'model'
             else None)

    calcdir = str(templates / kind / stress / profile)
    os.makedirs(calcdir)
    write_input_file(config, compile_workflow(profile), calcdir, model)

    assert _digest(calcdir) == GOLDEN[case], (
        f'generated input text drifted for {case}')
