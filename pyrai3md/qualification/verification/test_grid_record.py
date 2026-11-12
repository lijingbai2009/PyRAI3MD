"""
======================================================================
Test suite for the read-back side of a hyperparameter grid scan:
- pyrai3md/workflow/grid/record.py

Tests cover:
- point_record over the three carriers a dispatched point comes back in
  (live state, state dump, plain result)
- point_failure over the same three carriers: a dispatched child's runtime
  arrives as the dumped dict, not as a RuntimeState
- the structured folder read: which products mark a point finished, how
  a stored fit error is put back into a physical unit, and what a
  half-written folder reports
- ranking: the worst member by default, the average on request, points
  that report nothing last, ties broken by the key
- the two tables and the two files a scan leaves behind

The folder fixture is written here rather than shipped: a scan reads
whatever a training wrote, and a test that trains to prove a reader
works would be measuring the trainer. The unit factors are asserted
against the skill's own reference table, not against a literal.

Oct 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from pyrai3md.workflow.grid.plan import build_context, point_key, skill_surface
from pyrai3md.workflow.grid.record import (build_row, column_stat, disk_errors,
                                           folder_complete, format_plan,
                                           format_summary, point_failure,
                                           point_record,
                                           rank_points, record_errors,
                                           write_csv, write_crashed)

from pyrai3md.qualification.verification.test_grid_plan import _nn_task

_SKILL = 'nn_pes_skill'


def _write_folder(path, members=1, model_type='mlp_eg', products=None,
                  stored=None, spreads=None):
    """Write one ensemble folder the way a finished fit leaves it."""
    os.makedirs(path, exist_ok=True)
    wanted = products if products is not None else range(members)
    for index in wanted:
        hyper = {'general': {'model_type': model_type},
                 'model': {'atoms': 2, 'states': 1}}
        with open(os.path.join(path, 'hyper_v%d.json' % index), 'w') as handle:
            json.dump(hyper, handle)
        scaler = (spreads or {}).get(index, {'energy_std': [[1.0]],
                                             'gradient_std': [[[1.0]]]})
        with open(os.path.join(path, 'scaler_v%d.json' % index), 'w') as handle:
            json.dump(scaler, handle)
        np.save(os.path.join(path, 'fiterr_valid_v%d.npy' % index),
                (stored or {}).get(index, [0.1, 0.2]))
        with open(os.path.join(path, 'weights_v%d.weights.h5' % index),
                  'w') as handle:
            handle.write('placeholder')


class TestPointRecord(unittest.TestCase):
    """The three shapes a dispatched point arrives in."""

    def test_live_state_reads_pmd_results(self):
        from pyrai3md.data.runtime.result import Result

        state = _nn_task(mult_nn=1)
        state.pmd.results = {'train_errors': {'energy_grad': {'energy': [0.1]}}}
        self.assertEqual({'energy_grad': {'energy': [0.1]}},
                         point_record(state)['train_errors'])
        # A live state is the carrier that is not a mapping at all.
        self.assertFalse(isinstance(state, dict))
        # A plain property Result carries no training record at all.
        self.assertEqual({'energy': -1.0},
                         dict(point_record(Result({'energy': -1.0}))))

    def test_state_dump_reads_nested_results(self):
        dumped = {'pmd': {'results': {'train_errors': {'nac': {'nac': [0.3]}}}}}
        self.assertEqual({'nac': {'nac': [0.3]}},
                         point_record(dumped)['train_errors'])

    def test_plain_result_reads_itself(self):
        record = {'train_errors': {'soc': {'soc': [1.0]}}}
        self.assertEqual(record['train_errors'],
                         point_record(record)['train_errors'])

    def test_nothing_returns_nothing(self):
        for carrier in (None, 7, 'text'):
            self.assertEqual({}, point_record(carrier))


class TestPointFailure(unittest.TestCase):
    """The child's own failure sentence, over the same three carriers.

    The carrier that was never exercised is the dispatched one: `Result`
    answers any attribute question out of its payload, so a Slurm child's
    runtime reaches `point_failure` as the dumped dict, and the first
    version of this reader asked it for `to_dict()`. That killed the whole
    scan at the first outcome, in both carriers of
    `examples/nn_grid_search.yaml`, in the test environment (round
    `tmp_20261010_133424`) and nowhere else -- the local-topology sibling
    `nn_grid_search_local` passed with the same tree.
    """

    _MESSAGE = ('channel "nac" came back empty: the fit wrote no '
                'validation error')

    def _runtime(self, status='failed'):
        return {'status': status, 'error_message': self._MESSAGE}

    def test_a_live_state_carries_a_runtime_object(self):
        class _State:
            def __init__(self, runtime):
                self.runtime = runtime

        from pyrai3md.data.runtime.state import RuntimeState

        state = _State(RuntimeState(status='failed',
                                    error_message=self._MESSAGE))
        self.assertEqual(self._MESSAGE, point_failure(state))
        # This carrier is not a mapping at all, which is why the reader
        # cannot tell the shapes apart with `isinstance(x, dict)` alone.
        self.assertFalse(isinstance(state, dict))

    def test_a_dispatched_child_carries_the_dumped_dict(self):
        from pyrai3md.data.runtime.result import Result

        for carrier in (Result(self._runtime_payload()),
                        self._runtime_payload(),
                        {'runtime': self._runtime()}):
            self.assertEqual(self._MESSAGE, point_failure(carrier), carrier)

    def test_the_message_is_read_whatever_the_status_claims(self):
        # `build_row` consults this text only for a row that came back with
        # no numbers, so a point that finished is not talked out of its
        # status by a sentence its runtime happened to keep.
        from pyrai3md.data.runtime.result import Result

        payload = {'runtime': self._runtime(status='completed')}
        self.assertEqual(self._MESSAGE, point_failure(Result(payload)))

    def test_a_carrier_that_said_nothing_gives_no_message(self):
        from pyrai3md.data.runtime.result import Result

        for carrier in (None, Result({'energy': -1.0}), {}, 7,
                        {'runtime': {'status': 'completed'}}):
            self.assertEqual('', point_failure(carrier), carrier)

    def _runtime_payload(self):
        return {'runtime': self._runtime(),
                'pmd': {'results': {'train_errors': {}}}}


class TestRecordErrors(unittest.TestCase):
    """What counts as a training record."""

    def test_absent_and_empty_both_report_no_errors(self):
        for record in ({}, {'train_errors': {}}, {'train_errors': None}):
            errors, units, _wall = record_errors(record)
            self.assertEqual({}, errors)
            self.assertEqual({}, units)

    def test_a_populated_record_carries_its_units(self):
        record = {'train_errors': {'energy_grad': {'energy': [0.1]}},
                  'train_metric_units': {'energy': 'eV'},
                  'train_wall_seconds': 3.5}
        errors, units, wall = record_errors(record)
        self.assertEqual([0.1], errors['energy_grad']['energy'])
        self.assertEqual('eV', units['energy'])
        self.assertEqual(3.5, wall)


class _FolderCase(unittest.TestCase):
    """Shared fixture plumbing for the disk reads."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='grid_record_')
        self.surface = skill_surface(_SKILL)
        self.state = _nn_task(train_data=self._make_data(), mult_nn=1)
        self.context = build_context(self.state, self.surface,
                                     {'grid_root': self.dir})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _make_data(self):
        path = os.path.join(self.dir, 'train.json')
        with open(path, 'w') as handle:
            json.dump({'natom': 2, 'nstate': 1,
                       'xyz': [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                       'energy': [[-1.0]],
                       'grad': [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]}, handle)
        return path

    def _point(self, **values):
        return {'values': values, 'key': point_key(values)}

    def _modeldir(self, point):
        return os.path.join(self.dir, 'NN-grid_unit-%s' % point['key'])


class TestFolderReading(_FolderCase):
    """The structured read that replaced the log scrape."""

    def test_a_complete_folder_ranks_and_reports_physical_units(self):
        from pyrai3md.skills.nn_pes_skill.schema.reference import HARTREE_TO_EV

        point = self._point(depth=2)
        directory = self._modeldir(point)
        _write_folder(os.path.join(directory, 'energy_grad'))
        self.assertTrue(folder_complete(self.context, directory))

        errors, units, reason = disk_errors(self.context, directory)
        self.assertEqual('', reason)
        # The stored error is in the fit's scaled units; the scaler spread
        # (1.0 here) and the Hartree-to-eV factor put it back.
        self.assertAlmostEqual(0.1 * HARTREE_TO_EV,
                               errors['energy_grad']['energy'][0])
        self.assertAlmostEqual(0.2 * HARTREE_TO_EV,
                               errors['energy_grad']['gradient'][0])
        self.assertEqual('eV', units['energy'])

    def test_a_missing_product_is_named_not_assumed_finished(self):
        point = self._point(depth=3)
        directory = self._modeldir(point)
        _write_folder(os.path.join(directory, 'energy_grad'), products=[])
        self.assertFalse(folder_complete(self.context, directory))
        errors, _units, reason = disk_errors(self.context, directory)
        self.assertEqual({}, errors)
        self.assertIn('fiterr_valid_v0.npy', reason)

    def test_an_absent_folder_reports_the_path(self):
        point = self._point(depth=4)
        errors, _units, reason = disk_errors(self.context,
                                             self._modeldir(point))
        self.assertIn('no model directory', reason)

    def test_a_spread_is_applied_per_member(self):
        from pyrai3md.skills.nn_pes_skill.schema.reference import HARTREE_TO_EV

        point = self._point(depth=5)
        directory = self._modeldir(point)
        self.context['members'] = 2
        _write_folder(os.path.join(directory, 'energy_grad'), members=2,
                      stored={0: [0.1, 0.2], 1: [0.2, 0.4]},
                      spreads={0: {'energy_std': [[2.0]],
                                   'gradient_std': [[[1.0]]]},
                               1: {'energy_std': [[0.5]],
                                   'gradient_std': [[[1.0]]]}})
        errors, _units, _reason = disk_errors(self.context, directory)
        # Each member is unscaled by its own stored spread (2.0 and 0.5)
        # and then into eV: the same arithmetic the fit report uses.
        values = errors['energy_grad']['energy']
        self.assertAlmostEqual(0.1 * 2.0 * HARTREE_TO_EV, values[0], places=9)
        self.assertAlmostEqual(0.2 * 0.5 * HARTREE_TO_EV, values[1], places=9)


class TestRanking(_FolderCase):
    """The comparison a scan reports."""

    def _rows(self, specifications):
        rows = []
        for values, stored in specifications:
            point = self._point(**values)
            directory = self._modeldir(point)
            _write_folder(os.path.join(directory, 'energy_grad'),
                          stored={0: stored})
            rows.append(build_row(point, self.context, None, directory))
        return rows

    def test_max_is_the_worst_member_and_mean_the_average(self):
        columns = {('energy_grad', 'energy', 0): 0.4,
                   ('energy_grad', 'energy', 1): 0.1}
        self.assertEqual(0.4, column_stat(columns, 'energy', 'max'))
        self.assertAlmostEqual(0.25, column_stat(columns, 'energy', 'mean'))

    def test_a_point_that_reports_nothing_ranks_last(self):
        rows = self._rows([({'depth': 1}, [0.1, 0.1]),
                           ({'depth': 2}, [0.3, 0.3])])
        missing = build_row(self._point(depth=3), self.context, None,
                            self._modeldir(self._point(depth=3)))
        ranked = rank_points(rows + [missing], 'energy')
        self.assertEqual(['depth=1', 'depth=2', 'depth=3'],
                         [row['key'] for row in ranked])
        self.assertIsNone(ranked[-1]['stat'])

    def test_ranking_descending_flips_the_order(self):
        rows = self._rows([({'depth': 1}, [0.1, 0.1]),
                           ({'depth': 2}, [0.3, 0.3])])
        ranked = rank_points(rows, 'energy', sort_order='desc')
        self.assertEqual(['depth=2', 'depth=1'], [row['key'] for row in ranked])

    def test_a_tie_breaks_on_the_key_so_the_table_is_deterministic(self):
        rows = self._rows([({'depth': 2, 'nn_size': 8}, [0.1, 0.1]),
                           ({'depth': 1, 'nn_size': 16}, [0.1, 0.1])])
        first = [row['key'] for row in rank_points(rows, 'energy')]
        second = [row['key'] for row in rank_points(list(reversed(rows)),
                                                   'energy')]
        self.assertEqual(first, second)


class TestTables(_FolderCase):
    """What a scan leaves for a reader."""

    def test_plan_block_names_the_number_of_points(self):
        points = [{'values': {'depth': 1}, 'key': 'depth=1'},
                  {'values': {'depth': 2}, 'key': 'depth=2'}]
        block = format_plan(points, ['depth'])
        self.assertIn('&grid search', block)
        self.assertIn('Number of search: 2', block)
        self.assertIn('depth=2', block)

    def test_summary_carries_units_and_blanks_for_absent_channels(self):
        point = self._point(depth=1)
        directory = self._modeldir(point)
        _write_folder(os.path.join(directory, 'energy_grad'))
        row = build_row(point, self.context, None, directory)
        other = build_row(self._point(depth=2), self.context, None,
                          self._modeldir(self._point(depth=2)))
        ranked = rank_points([row, other], 'energy')
        block = format_summary(ranked, ['depth'], 'energy')
        self.assertIn('&grid search summary', block)
        self.assertIn('eV', block)
        self.assertIn('1', block)
        self.assertIn('--', block, 'an unread folder must not print a zero')

    def test_csv_rows_are_ranked_and_the_detail_block_names_roles(self):
        point = self._point(depth=1)
        directory = self._modeldir(point)
        _write_folder(os.path.join(directory, 'energy_grad'))
        ranked = rank_points([build_row(point, self.context, None, directory)],
                             'energy')
        path = write_csv(os.path.join(self.dir, 'summary.csv'), ranked,
                         ['depth'], 'energy')
        with open(path) as handle:
            text = handle.read()
        self.assertIn('rank,key,status,sort_by,sort_value', text)
        self.assertIn('rank,key,role,channel,member', text)
        self.assertIn('energy_grad,energy,0', text)
        self.assertIn('eV', text)

    def test_no_crash_file_is_written_when_nothing_crashed(self):
        point = self._point(depth=1)
        directory = self._modeldir(point)
        _write_folder(os.path.join(directory, 'energy_grad'))
        rows = [build_row(point, self.context, None, directory)]
        self.assertIsNone(write_crashed(os.path.join(self.dir, 'c.txt'), rows))
        self.assertFalse(os.path.exists(os.path.join(self.dir, 'c.txt')))

    def test_a_failed_point_is_written_with_its_reason(self):
        point = self._point(depth=2)
        rows = [build_row(point, self.context, None,
                          self._modeldir(point))]
        path = write_crashed(os.path.join(self.dir, 'sub', 'c.txt'), rows)
        self.assertTrue(path.endswith('c.txt'))
        with open(path) as handle:
            text = handle.read()
        self.assertIn('no model directory', text)

    def test_the_dispatch_death_outranks_the_absent_folder(self):
        # Two sentences describe the same point, and the parent's names the
        # cause: a job that never reached the cluster leaves no folder, so
        # 'no model directory' is what the absence looks like from disk.
        point = self._point(depth=5)
        row = build_row(point, self.context, None, self._modeldir(point),
                        failure='RuntimeError: sbatch submission failed')
        self.assertEqual('RuntimeError: sbatch submission failed',
                         row['reason'])
        self.assertEqual('failed', row['status'])

    def test_a_completed_folder_is_not_judged_by_the_batch_death(self):
        # A point whose fit landed keeps its numbers even when the batch it
        # belonged to died: the death sentence is only looked for once the
        # row has no columns to report.
        point = self._point(depth=6)
        directory = self._modeldir(point)
        _write_folder(os.path.join(directory, 'energy_grad'))
        self.assertTrue(folder_complete(self.context, directory))
        row = build_row(point, self.context, None, directory,
                        failure='RuntimeError: sbatch submission failed')
        self.assertEqual('ok', row['status'])
        self.assertEqual('', row['reason'])


if __name__ == '__main__':
    unittest.main()
