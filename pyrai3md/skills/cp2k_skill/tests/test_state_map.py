"""
Unit tests for the numbering rules in `service/parser/state_map.py`.

`state_map` is the one place that says which framework row a CP2K state is, so
the rules it holds are the ones worth pinning: what a declared spin list means
for row order, how a row is numbered under each of the two readings of the
tdfrc header, and what happens when the engine's own labels cannot separate two
surfaces. None of it needs CP2K.

The measured part comes from `cp2k_demo/sp/bulk.log`: two excitation tables,
`multiplicity 1` then `multiplicity 3`, each numbered 1..N. What is not measured
locally is how `&PRINT &FORCES LIST` counts the same states, so both readings are
kept and the tests below say which one the parser picks for each shape of output.

Nov 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
"""

import pytest

from ..schema.adapter import build_config
from ..schema.normalize import normalize
from ..service.parser import state_map

ATOMS = ['O', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [0.0, 0.75, -0.47], [0.0, -0.75, -0.47]]


def _config(state_spin, target_states, profile='tddft_gradient'):
    raw = {
        'atoms': ATOMS,
        'coordinates': COORDS,
        'charges': 0,
        'multiplicity': 1,
        'nstate': len(state_spin),
        'state_spin': list(state_spin),
        'target_states': list(target_states),
        'profile': profile,
        'method': 'tddft',
        'functional': 'pbe',
        'project': 'numbering',
    }
    return normalize(build_config(raw))


class TestRowOrder:
    """A declared spin list fixes the rows; the ground state is always row 0."""

    def test_the_ground_row_is_never_an_excited_root(self):
        config = _config([0, 0, 2], [0, 1, 2])
        groups = state_map.excited_rows(config)
        assert groups[0] == [1]
        assert groups[2] == [2]
        assert 0 not in groups[0] and 0 not in groups[2]

    def test_requested_rows_follow_the_target_states_not_the_declared_states(
            self):
        # Three states declared, forces asked for one of them: the run asks for
        # one gradient and must be answerable for one, not three.
        config = _config([0, 0, 0], [0, 1])
        assert state_map.requested_rows(config) == [1]

    def test_a_target_naming_the_ground_state_adds_no_row(self):
        config = _config([0, 0, 0], [0])
        assert state_map.requested_rows(config) == []

    def test_spin_of_row(self):
        config = _config([0, 0, 2], [0, 1, 2])
        assert state_map.spin_of_row(config, 0) == 0
        assert state_map.spin_of_row(config, 2) == 2
        with pytest.raises(ValueError, match='outside'):
            state_map.spin_of_row(config, 7)


class TestNumbering:
    """The same row is two different integers under the two readings."""

    def test_a_single_manifold_leaves_nothing_to_disambiguate(self):
        config = _config([0, 0, 0], [0, 1, 2])
        assert state_map.state_number(config, 1, state_map.MANIFOLD) == 1
        assert state_map.state_number(config, 2, state_map.MANIFOLD) == 2
        assert state_map.state_number(config, 2, state_map.GLOBAL) == 2

    def test_the_triplet_root_is_numbered_both_ways(self):
        config = _config([0, 0, 2], [0, 1, 2])
        # NSTATES is per manifold, so a continuing count places the first
        # triplet after the singlet roots the engine solved.
        assert config.tddft_nstates == 2
        assert state_map.state_number(config, 2, state_map.MANIFOLD) == 1
        assert state_map.state_number(config, 2, state_map.GLOBAL) == 3

    def test_the_written_list_numbers_roots_inside_the_manifold(self):
        # Measured in the test environment: the continuing numbering (LIST 1 3) made
        # CP2K abort with `TDDFPT Print Forces: Invalid State`, so what the writer
        # emits is the count inside the manifold, and a request that spans two
        # manifolds is refused rather than numbered twice.
        config = _config([0, 0, 0], [0, 1, 2])
        assert state_map.list_numbers(config) == [1, 2]
        assert state_map.LIST_ORDER == state_map.MANIFOLD

    def test_the_global_reading_still_numbers_for_the_parse_side(self):
        # The reading the engine rejected for LIST is still tried when interpreting
        # a printed header, because that is a separate question from what the
        # keyword accepts - and it is how `attribute_blocks` decides which counting
        # a given run used.
        config = _config([0, 0, 2], [0, 1, 2])
        assert state_map.state_number(config, 2, state_map.MANIFOLD) == 1
        assert state_map.state_number(config, 2, state_map.GLOBAL) == 3

    def test_a_two_manifold_force_request_is_refused(self):
        config = _config([0, 0, 2], [0, 1, 2])
        with pytest.raises(ValueError, match='no spin argument'):
            state_map.require_addressable(config)

    def test_a_single_manifold_request_is_not_refused(self):
        # The positive arm: the refusal is about two manifolds sharing one number
        # space, not about the triplet row itself.
        config = _config([0, 0, 0], [0, 1, 2])
        state_map.require_addressable(config)
        config_triplets_only = _config([0, 2, 2], [0, 1, 2])
        state_map.require_addressable(config_triplets_only)

    def test_the_written_list_is_the_row_itself_for_one_manifold(self):
        # The byte-identity claim behind the builder change: with no triplet the
        # new writer emits exactly what `sorted(s for s in target_states if s>0)`
        # emitted before, so every existing input is unchanged.
        config = _config([0, 0, 0], [0, 1, 2])
        assert state_map.list_numbers(config) == [1, 2]

    def test_the_ground_row_has_no_state_number(self):
        config = _config([0, 0, 2], [0, 1, 2])
        with pytest.raises(ValueError, match='ground state'):
            state_map.state_number(config, 1 - 1)

    def test_spin_and_manifold_are_inverse(self):
        for two_s in (0, 1, 2):
            assert state_map.spin_of_manifold(
                state_map.manifold_of_spin(two_s)) == two_s

    def test_an_unknown_order_is_refused(self):
        config = _config([0, 0, 2], [0, 1, 2])
        with pytest.raises(ValueError, match='unknown state numbering'):
            state_map.state_number(config, 1, 'whatever')


class TestLogResolution:
    """Extra roots the engine solves for are not rows the run follows."""

    def test_rows_inside_the_request_resolve_and_others_do_not(self):
        config = _config([0, 0, 2], [0, 1, 2])
        assert state_map.resolve_log_row(config, 1, 1) == 1
        assert state_map.resolve_log_row(config, 3, 1) == 2
        assert state_map.resolve_log_row(config, 1, 2) is None
        assert state_map.resolve_log_row(config, 3, 2) is None

    def test_a_spin_the_request_never_declared_resolves_to_nothing(self):
        config = _config([0, 0, 0], [0, 1, 2])
        assert state_map.resolve_log_row(config, 3, 1) is None


def _row(multiplicity, state, energy):
    return {'multiplicity': multiplicity, 'state': state,
            'excitation': energy, 'transition_dipole': [0.0, 0.0, 0.0],
            'osc_strength': 0.0}


def _block(state, energy, forces):
    return {'state': state, 'excitation': energy, 'osc_strength': 0.0,
            'forces': forces}


SINGLET = _row(1, 1, 3.94485)
TRIPLET = _row(3, 1, 3.76173)
FORCES_S = [[0.01, 0.02, 0.03]] * 3
FORCES_T = [[-0.04, 0.05, -0.06]] * 3


class TestBlockAttribution:
    """Which reading of the header the blocks turn out to follow."""

    def test_continuing_numbers_are_recognised(self):
        config = _config([0, 0, 2], [0, 1, 2])
        blocks = [_block(1, 3.94485, FORCES_S), _block(3, 3.76173, FORCES_T)]
        assignment, by_order, order = state_map.attribute_blocks(
            config, [SINGLET, TRIPLET], blocks)
        assert assignment == {1: 0, 2: 1}
        assert by_order == []
        assert order == state_map.GLOBAL

    def test_restarting_numbers_are_recognised_too(self):
        # Both blocks print 1, which is what the measured excitation tables do.
        # The energies carry the attribution, and the caller is told which
        # counting the engine turned out to use.
        config = _config([0, 0, 2], [0, 1, 2])
        blocks = [_block(1, 3.94485, FORCES_S), _block(1, 3.76173, FORCES_T)]
        assignment, by_order, order = state_map.attribute_blocks(
            config, [SINGLET, TRIPLET], blocks)
        assert assignment == {1: 0, 2: 1}
        assert order == state_map.MANIFOLD

    def test_one_manifold_is_ambiguous_by_design_and_keeps_the_written_order(
            self):
        config = _config([0, 0, 0], [0, 1])
        blocks = [_block(1, 3.94485, FORCES_S)]
        assignment, _by_order, order = state_map.attribute_blocks(
            config, [SINGLET], blocks)
        assert assignment == {1: 0}
        assert order == state_map.LIST_ORDER

    def test_a_degenerate_pair_falls_to_request_order_and_says_so(self):
        # At an MECP the two surfaces agree in energy by definition, and under
        # the per-manifold reading they agree on the printed number too, so
        # nothing in the engine's output separates them. The tie-break is the
        # order the roots were asked for, and the caller is told it was used.
        config = _config([0, 0, 2], [0, 1, 2])
        shared = _row(3, 1, 3.94485)
        blocks = [_block(1, 3.94485, FORCES_S), _block(1, 3.94485, FORCES_T)]
        assignment, by_order, order = state_map.attribute_blocks(
            config, [SINGLET, shared], blocks)
        assert order == state_map.MANIFOLD
        assert sorted(assignment) == [1, 2]
        assert assignment[1] == 0 and assignment[2] == 1
        assert by_order == [1]

    def test_a_number_neither_reading_explains_is_refused(self):
        config = _config([0, 0, 2], [0, 1, 2])
        blocks = [_block(1, 3.94485, FORCES_S), _block(9, 3.76173, FORCES_T)]
        with pytest.raises(RuntimeError, match='neither reading'):
            state_map.attribute_blocks(config, [SINGLET, TRIPLET], blocks)

    def test_two_blocks_naming_one_row_are_refused(self):
        # Letting the later block win would drop a surface while the block count,
        # the row count and every shape still agreed.
        config = _config([0, 0, 2], [0, 1, 2])
        blocks = [_block(1, 3.94485, FORCES_S), _block(1, 3.94485, FORCES_T)]
        with pytest.raises(RuntimeError, match='neither reading'):
            state_map.attribute_blocks(config, [SINGLET, TRIPLET], blocks)

    def test_a_missing_block_is_refused_instead_of_left_as_a_zero_row(self):
        config = _config([0, 0, 2], [0, 1, 2])
        blocks = [_block(1, 3.94485, FORCES_S)]
        with pytest.raises(RuntimeError, match='neither reading'):
            state_map.attribute_blocks(config, [SINGLET, TRIPLET], blocks)
