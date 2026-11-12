"""
Unit tests for the open-shell reference the CP2K skill cannot yet write.

`MULTIPLICITY` is emitted into `&DFT` while the `&SCF` block carries no
unrestricted keyword, so a multiplicity other than 1 describes an open-shell
reference that the written input does not solve. CP2K says so itself, several
seconds after the file is submitted; the tests below check that the skill says
so at the request instead, and that it says nothing when the reference is the
one its input does write.

These tests do NOT require a CP2K installation.

Run: pytest cp2k_skill/tests/test_open_shell.py -v

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..schema.validate import validate

LATTICE = [[6.3, 0.0, 0.0], [0.0, 6.3, 0.0], [0.0, 0.0, 6.3]]
CELL = [6.3, 6.3, 6.3, 90.0, 90.0, 90.0]
ATOMS = ['O', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [0.0, 0.75, -0.47], [0.0, -0.75, -0.47]]


def _config(**overrides):
    """Run the real schema pipeline (build -> normalize -> validate)."""
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': 1,
        'state_spin': [0],
        'profile': 'dft_energy',
        'method': 'dft',
        'functional': 'pbe',
        'project': 'open_shell',
        'pbc': [1, 1, 1],
        'cell': CELL,
        'lattice_vector': LATTICE,
    }
    raw.update(overrides)
    return validate(normalize(build_config(raw)))


class TestATripletRowNeedsAManifoldThatSolvesIt:
    """A row labelled triplet is only meaningful if something computes one."""

    def test_a_triplet_row_under_plain_dft_is_refused_by_name(self):
        # Method 'dft' writes no &TDDFPT block at all, so a triplet row is a
        # label on nothing: the crossing search reading it would step on a
        # surface the engine never produced.
        with pytest.raises(ValueError) as caught:
            _config(method='dft', profile='dft_energy', nstate=2,
                    state_spin=[0, 2])
        message = str(caught.value)
        assert 'triplet' in message
        assert 'TDDFPT' in message

    def test_the_same_row_under_tddft_is_accepted(self):
        # The positive arm: the refusal is about the method solving no excited
        # states, not about the spin keyword.
        config = _config(method='tddft', profile='tddft_gradient', nstate=2,
                         state_spin=[0, 2], target_states=[0, 1])
        assert config.ntrip_root == 1
        assert config.nsing_root == 0


class TestOpenShellReference:
    """A request for a spin-polarized reference is answered before the file."""

    def test_a_triplet_reference_is_refused_by_name(self):
        with pytest.raises(ValueError) as caught:
            _config(multiplicity=3, state_spin=[2])
        message = str(caught.value)
        assert 'multiplicity' in message
        assert 'UKS' in message

    def test_a_doublet_reference_is_refused_too(self):
        # The guard is on the reference, not on one spin: a doublet needs the
        # same unrestricted block. This arm carries an odd electron count on
        # purpose (O plus H, nine electrons), because the electron-parity check
        # in `verify_electronic_structure` runs first and would otherwise be the
        # thing refusing the request - a doublet on a ten-electron water is
        # refused for its electron count, which is a different defect with a
        # different message, and this test is about the missing UKS section.
        with pytest.raises(ValueError) as caught:
            _config(atoms=['O', 'H'],
                    coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
                    charges=0,
                    multiplicity=2, state_spin=[1], nstate=1)
        message = str(caught.value)
        assert 'multiplicity' in message
        assert 'UKS' in message

    def test_the_singlet_reference_is_not_refused(self):
        # Without this arm the two refusals above could be a ban on the keyword,
        # and every shipped case would be the one that noticed.
        config = _config()
        assert int(config.system.multiplicity) == 1
