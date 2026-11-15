"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Nov 15 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Tests of the phase table behind the phaseless coupling loss.

The constraint is that one state phase holds across every pair that state
appears in, which is what the complete table encodes and what selecting
columns from it must preserve. The expectations are therefore the legal
2**nstate family itself (enumerated here from the state sign products),
not a stored copy of the module's output. The four-state table is also
spelled out literally, because it is the one the design document prints.

The table tests need no TensorFlow. The loss tests need the neural network
stack and skip with a named reason when it is absent.

"""

import itertools
import unittest

import numpy as np

from pyrai3md.skills.nn_pes_skill.service.nn_pes.utils.phase import (
    full_phase_map, reduced_phase_map)


def _legal_rows(nstate, pairs):
    """Return every legal sign row of one pair set as a set of tuples."""
    rows = set()
    for signs in itertools.product((0, 1), repeat=nstate):
        rows.add(tuple(signs[i] ^ signs[j] for i, j in pairs))
    return rows


class TestCompletePhaseMap(unittest.TestCase):

    def test_four_state_table_is_the_documented_one(self):
        # Rows are the 2**3 assignments with state 0 fixed at +; columns
        # are 01, 02, 03, 12, 13, 23. 1 marks a flipped pair.
        pairs, table = full_phase_map(4)
        self.assertEqual(pairs, ((0, 1), (0, 2), (0, 3),
                                 (1, 2), (1, 3), (2, 3)))
        expected = [[0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 0, 1],
                    [0, 1, 1, 1, 1, 0],
                    [1, 0, 0, 1, 1, 0],
                    [1, 0, 1, 1, 0, 1],
                    [1, 1, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0, 0]]
        self.assertEqual(table.tolist(), expected)

    def test_rows_are_exactly_the_legal_state_sign_family(self):
        for nstate in range(2, 6):
            pairs, table = full_phase_map(nstate)
            self.assertEqual(len(table), 2 ** (nstate - 1))
            self.assertEqual({tuple(int(v) for v in row) for row in table},
                             _legal_rows(nstate, pairs))

    def test_every_row_satisfies_the_ring_product(self):
        # The product of pair signs around any cycle must be +1; on a
        # triangle that is the product of its three columns, with 0/1 read
        # as +1/-1.
        pairs, table = full_phase_map(4)
        column = {pair: index for index, pair in enumerate(pairs)}
        triangle = [column[(0, 1)], column[(1, 2)], column[(0, 2)]]
        signs = 1 - 2 * table[:, triangle]
        self.assertTrue(np.all(np.prod(signs, axis=1) == 1))


class TestReducedPhaseMap(unittest.TestCase):

    def test_row_counts_of_the_documented_cases(self):
        all_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        cases = {
            all_pairs: (8, 6),
            ((0, 1), (1, 2), (2, 3)): (8, 3),
            ((0, 1), (1, 2)): (4, 2),
            ((0, 1),): (2, 1),
            ((0, 1), (0, 2), (1, 2)): (4, 3),
        }
        for pairs, shape in cases.items():
            self.assertEqual(reduced_phase_map(4, pairs).shape, shape, pairs)

    def test_disconnected_states_do_not_multiply_the_candidates(self):
        # Pairs 01 and 12 touch states 0, 1, 2; state 3 is isolated, so its
        # phase cannot create a distinct candidate.
        table = reduced_phase_map(5, ((0, 1), (1, 2)))
        self.assertEqual(table.shape[0], 4)

    def test_a_partial_map_keeps_exactly_the_legal_rows(self):
        pairs = ((0, 1), (1, 2))
        rows = {tuple(int(v) for v in row)
                for row in reduced_phase_map(4, pairs)}
        self.assertEqual(rows, _legal_rows(4, pairs))

    def test_column_order_follows_the_declared_order(self):
        forward = reduced_phase_map(4, ((0, 1), (1, 2), (2, 3)))
        backward = reduced_phase_map(4, ((2, 3), (1, 2), (0, 1)))
        self.assertEqual(backward.shape, (8, 3))
        # The same candidate set, with its columns mirrored.
        self.assertEqual({tuple(row) for row in backward},
                         {tuple(reversed(row)) for row in forward})

    def test_a_reversed_pair_is_the_same_pair(self):
        a = reduced_phase_map(4, ((0, 1), (1, 2)))
        b = reduced_phase_map(4, ((1, 0), (2, 1)))
        self.assertEqual(a.tolist(), b.tolist())

    def test_a_bad_declaration_is_refused(self):
        bad = [((), 'no pair'),
               (((0, 1), (0, 1)), 'repeated'),
               (((0, 4),), 'outside'),
               (((1, 1),), 'itself'),
               (((0,),), 'two indices')]
        for pairs, reason in bad:
            with self.assertRaises(ValueError, msg=reason):
                reduced_phase_map(4, pairs)

    def test_a_single_state_has_no_pair_to_phase(self):
        with self.assertRaises(ValueError):
            reduced_phase_map(1, ((0, 1),))

    def test_the_cache_returns_the_same_read_only_table(self):
        first = reduced_phase_map(4, ((0, 1), (1, 2)))
        second = reduced_phase_map(4, ((0, 1), (1, 2)))
        self.assertIs(first, second)


class TestPhaselessLoss(unittest.TestCase):
    """The loss is the legal-phase minimum of the per-pair squared error."""

    def setUp(self):
        try:
            import keras  # noqa: F401
            from pyrai3md.skills.nn_pes_skill.service.nn_pes.utils.loss import (
                NACphaselessLoss)
        except ImportError as error:
            raise unittest.SkipTest('neural network stack not installed: %s'
                                    % error)
        self.loss_class = NACphaselessLoss

    @staticmethod
    def _brute_force(y_true, y_pred, nstate, pairs):
        """Minimum over every assignment of state signs, by definition."""
        values = []
        for sample in range(y_true.shape[0]):
            per_sample = []
            for signs in itertools.product((1.0, -1.0), repeat=nstate):
                errors = [np.mean((y_pred[sample, e]
                                   - signs[i] * signs[j]
                                   * y_true[sample, e]) ** 2)
                          for e, (i, j) in enumerate(pairs)]
                per_sample.append(np.mean(errors))
            values.append(min(per_sample))
        return float(np.mean(values))

    def test_matches_brute_force_enumeration(self):
        import keras
        rng = np.random.default_rng(17)
        for nstate, pairs in (
                (2, ((0, 1),)),
                (3, ((0, 1), (0, 2), (1, 2))),
                (4, ((0, 1), (0, 2), (0, 3),
                     (1, 2), (1, 3), (2, 3))),
                (4, ((0, 1), (1, 2))),
                (4, ((0, 1), (0, 2), (1, 2)))):
            channels = len(pairs)
            y_true = rng.normal(size=(5, channels, 3, 3)).astype('float32')
            y_pred = rng.normal(size=(5, channels, 3, 3)).astype('float32')
            loss = self.loss_class(number_state=nstate, pairs=pairs)
            got = float(keras.ops.convert_to_numpy(loss(y_true, y_pred)))
            self.assertAlmostEqual(
                got, self._brute_force(y_true, y_pred, nstate, pairs),
                places=5, msg=str(pairs))

    def test_a_consistent_state_sign_change_does_not_change_the_loss(self):
        import keras
        rng = np.random.default_rng(29)
        nstate = 4
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        y_true = rng.normal(size=(3, len(pairs), 3, 3)).astype('float32')
        y_pred = rng.normal(size=(3, len(pairs), 3, 3)).astype('float32')
        loss = self.loss_class(number_state=nstate, pairs=pairs)
        base = float(keras.ops.convert_to_numpy(loss(y_true, y_pred)))
        # Flip state 2: every pair carrying it changes sign, every other
        # pair keeps its own.
        flipped = y_true.copy()
        for e, (i, j) in enumerate(pairs):
            if 2 in (i, j):
                flipped[:, e] *= -1.0
        moved = float(keras.ops.convert_to_numpy(loss(flipped, y_pred)))
        self.assertAlmostEqual(base, moved, places=6)

    def test_zero_couplings_give_a_finite_loss(self):
        import keras
        nstate = 2
        zero = np.zeros((2, 1, 2, 3), dtype='float32')
        loss = self.loss_class(number_state=nstate, pairs=((0, 1),))
        value = float(keras.ops.convert_to_numpy(loss(zero, zero)))
        self.assertEqual(value, 0.0)

    def test_a_fit_updates_the_weights_through_the_loss(self):
        import keras
        rng = np.random.default_rng(31)
        atoms = 3
        for nstate, pairs in ((2, ((0, 1),)),
                              (4, ((0, 1), (1, 2)))):
            channels = len(pairs)
            x = rng.normal(size=(8, atoms, 3)).astype('float32')
            y = rng.normal(size=(8, channels, atoms, 3)).astype('float32')
            inputs = keras.Input(shape=(atoms, 3))
            hidden = keras.layers.Dense(channels * atoms * 3)(
                keras.layers.Flatten()(inputs))
            model = keras.Model(
                inputs,
                keras.layers.Reshape((channels, atoms, 3))(hidden))
            model.compile(
                loss=self.loss_class(number_state=nstate, pairs=pairs),
                optimizer=keras.optimizers.Adam(1e-3))
            history = model.fit(x, y, epochs=2, batch_size=4, verbose=0)
            self.assertTrue(all(np.isfinite(history.history['loss'])))

    def test_the_declaration_is_carried_in_the_config(self):
        loss = self.loss_class(number_state=4,
                               pairs=((0, 1), (1, 2)))
        config = loss.get_config()
        self.assertEqual(config['number_state'], 4)
        self.assertEqual(config['pairs'], [[0, 1], [1, 2]])


class TestCouplingDeclaration(unittest.TestCase):
    """The ensemble hyper carries the ladder and the pairs the loss reads."""

    def setUp(self):
        try:
            from pyrai3md.skills.nn_pes_skill.service import trainer
        except ImportError as error:
            raise unittest.SkipTest('neural network stack not installed: %s'
                                    % error)
        self.trainer = trainer

    @staticmethod
    def _config(nstate, pairs):
        import types
        return types.SimpleNamespace(
            system=types.SimpleNamespace(
                nstate=nstate, nnac=len(pairs), nac_pair_map=pairs),
            calculation=types.SimpleNamespace(mult_nn=1, nac={}, soc={},
                                              eg={}))

    def _hyper(self, nstate, pairs):
        return self.trainer.ensemble_hyper(
            self._config(nstate, pairs), 'nac', 'mlp_nac2', natom=3,
            indices={'invd': [], 'angle': [], 'dihed': []})[0]

    def test_state_count_is_the_ladder_not_the_channel_count(self):
        for nstate, pairs in ((2, [[0, 1]]),
                              (4, [[0, 1], [1, 2]]),
                              (4, [[0, 1], [0, 2], [0, 3],
                                   [1, 2], [1, 3], [2, 3]])):
            hyper = self._hyper(nstate, pairs)
            self.assertEqual(hyper['model']['states'], nstate)
            self.assertEqual(hyper['model']['nnac'], len(pairs))
            self.assertEqual(hyper['model']['nac_pairs'], pairs)

    def test_the_loss_is_refused_without_a_pair_declaration(self):
        hyper = self._hyper(2, [])
        with self.assertRaises(ValueError):
            self.trainer.check_phaseless('nac', [hyper], 'training')

    def test_the_loss_is_refused_for_a_repeated_pair(self):
        hyper = self._hyper(3, [[0, 1], [0, 1]])
        with self.assertRaises(ValueError):
            self.trainer.check_phaseless('nac', [hyper], 'training')

    def test_a_declared_partial_pair_set_is_accepted(self):
        # The fit is allowed to cover only some pairs of the ladder.
        hyper = self._hyper(4, [[0, 1], [1, 2]])
        self.trainer.check_phaseless('nac', [hyper], 'training')

    def test_the_switch_off_needs_no_declaration(self):
        hyper = self._hyper(2, [])
        hyper['training']['phase_less_loss'] = False
        self.trainer.check_phaseless('nac', [hyper], 'training')


if __name__ == '__main__':
    unittest.main()
