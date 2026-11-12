"""
The two execution paths must load one identical input package.

The resident service is only a fast way of evaluating the same physics,
so the data file, the ``*_style`` declarations and the coefficient table
it loads have to be the very bytes the subprocess path writes. That
identity used to be a convention rather than a structure: each path
listed the three writers itself, and the resident one passed the model to
two of them. The consequence was not a crash in our code but a different
simulation: the pruned settings drop the coefficient rows of a hybrid
sub-style the system never uses, while a verbatim ``in.init`` kept
declaring that sub-style, and LAMMPS aborted at run time with "Pair
hybrid sub-style is not used". A missing keyword argument silently
changed the force field.

These tests do NOT require a LAMMPS installation.

Run: pytest lammps_skill/tests/test_input_builder_package.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import os

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate
from ..service.builder import write_input_file, write_static_input_files
from ..service.builder.ff_model import FFModelUnsupported, build_ff_model, \
    _validate
from ..service.workflow import compile_workflow

PACKAGE_FILES = ('PKG.data', 'PKG.in.init', 'PKG.in.settings')

# The declaration lists a second sub-style whose only rows belong to atom
# type 3, which this system does not contain. Trimming therefore starves
# it, and a declaration that kept listing it would abort LAMMPS' run.
INIT = """units real
atom_style full
bond_style harmonic
angle_style harmonic
pair_style hybrid lj/charmm/coul/long 9.0 10.0 10.0 buck/coul/long 10.0
kspace_style pppm 1e-4
"""

SETTINGS = """pair_coeff 1 1 lj/charmm/coul/long 0.1 3.0 3.2
pair_coeff 1 2 lj/charmm/coul/long 0.05 2.5 2.8
pair_coeff 2 2 lj/charmm/coul/long 0.02 2.0 2.2
pair_coeff 3 3 buck/coul/long 0.5 3.0 0.1
bond_coeff 1 harmonic 450.0 0.9572
angle_coeff 1 harmonic 55.0 104.52
"""

DATA = """LAMMPS Description

3  atoms
2  bonds
1  angles
0  dihedrals
0  impropers

3  atom types
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
3 1.008

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

ATOMS = ['O', 'H', 'H']
COORDS = [
    [0.000000000, 0.000000000, 0.117300000],
    [0.000000000, 0.757200000, -0.469200000],
    [0.000000000, -0.757200000, -0.469200000],
]


def _config():
    return validate(normalize(build_config({
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'profile': 'ff_gradient',
        'method': 'ff',
        'lammps_project': 'PKG',
        'lammps_units': 'real',
        'pbc': [1, 1, 1],
        'lattice_vector': [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]],
    })))


@pytest.fixture
def templates(tmp_path, monkeypatch):
    for name, text in (('PKG.data', DATA), ('PKG.in.init', INIT),
                       ('PKG.in.settings', SETTINGS)):
        (tmp_path / name).write_text(text)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _read(directory, name):
    with open(os.path.join(str(directory), name), 'rb') as handle:
        return handle.read()


class TestPackageIdentity:
    """Cold and warm load the same three files, byte for byte."""

    def test_the_warm_package_equals_the_cold_package(self, templates):
        config = _config()
        model = build_ff_model(config, 'ff_gradient')
        workflow = compile_workflow('ff_gradient')

        cold = str(templates / 'cold')
        warm = str(templates / 'warm')
        os.makedirs(cold)
        write_input_file(config, workflow, cold, model)
        write_static_input_files(config, workflow, warm, model)

        # The negative control is the shape of the bug: a verbatim copy
        # of the template still lists the sub-style no coefficient uses,
        # so identical bytes here is a claim about the trimmed render.
        assert b'buck/coul/long' in _read(templates, 'PKG.in.init')
        for name in PACKAGE_FILES:
            assert _read(warm, name) == _read(cold, name), name
        assert b'buck/coul/long' not in _read(cold, 'PKG.in.init')

    def test_the_warm_script_loads_that_package(self, templates):
        config = _config()
        model = build_ff_model(config, 'ff_gradient')
        workflow = compile_workflow('ff_gradient')
        warm = str(templates / 'warm')
        package = write_static_input_files(config, workflow, warm, model)

        script = _read(warm, f'{package["script"]}.in').decode()
        assert 'include "PKG.in.init"' in script
        assert 'read_data "PKG.data"' in script
        assert 'include "PKG.in.settings"' in script
        assert 'write_dump' not in script and 'dump ' not in script


class TestDeclarationFollowsTheCoefficients:
    """The model refuses a declaration its coefficients do not support."""

    def test_a_starved_substyle_left_in_the_declaration_is_refused(
            self, templates):
        config = _config()
        model = build_ff_model(config, 'ff_gradient')
        # Build keeps the two in step; claim the opposite and the model
        # has to say so before any writer can put it on disk.
        model['init']['styles']['pair']['segments'].append(
            ['buck/coul/long', []])
        with pytest.raises(FFModelUnsupported, match='buck/coul/long'):
            _validate(model, config)
