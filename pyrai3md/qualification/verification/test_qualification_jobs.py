"""
======================================================================
Test suite for the grouped Slurm execution layer
(`qualification/jobs.py`): how a plan becomes independent jobs, how a
job's missing rows become errors, and how the runner orders the report.

The grouping is a plan reading (each planner stamps `job_group` on its
tasks), so the job count and the plan order are pinned here without a
cluster: a local round and a slurm round must report the same rows, and
the only difference is where the tasks ran.

Nov 11 2026
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

from pyrai3md.qualification import jobs, runner


def _task(name, group, **extra):
    task = {'name': name, 'mode': 'pyrai3md_yaml', 'job_group': group,
            'input_file': '/tmp/%s.yaml' % name,
            'run_dir': '/tmp/%s' % name, 'script': '/tmp/%s.sh' % name}
    task.update(extra)
    return task


class TestSchedulerMode(unittest.TestCase):
    """environment.scheduler is the round's mode, not the skill's."""

    def test_slurm_is_case_insensitive(self):
        for value in ('slurm', 'SLURM', ' slurm '):
            self.assertEqual(
                jobs.scheduler_mode({'environment': {'scheduler': value}}),
                'slurm')

    def test_everything_else_is_local(self):
        for value in ('', 'local', 'LOCAL', None):
            self.assertEqual(
                jobs.scheduler_mode({'environment': {'scheduler': value}}),
                'local')
        self.assertEqual(jobs.scheduler_mode({}), 'local')


class TestInFlightLimit(unittest.TestCase):
    """environment.max_in_flight: 0 = submit the round at once, N =
    keep N unfinished jobs in flight."""

    def test_absent_means_no_cap(self):
        self.assertEqual(jobs.in_flight_limit({}), 0)
        self.assertEqual(jobs.in_flight_limit({'environment': {}}), 0)
        self.assertEqual(jobs.in_flight_limit(None), 0)

    def test_zero_means_no_cap(self):
        self.assertEqual(
            jobs.in_flight_limit(
                {'environment': {'max_in_flight': 0}}), 0)

    def test_positive_value_is_the_cap(self):
        self.assertEqual(
            jobs.in_flight_limit(
                {'environment': {'max_in_flight': 7}}), 7)
        self.assertEqual(
            jobs.in_flight_limit(
                {'environment': {'max_in_flight': '7'}}), 7)

    def test_negative_and_non_integer_values_are_refused(self):
        for value in (-1, 'many', None):
            with self.assertRaises(ValueError):
                jobs.in_flight_limit(
                    {'environment': {'max_in_flight': value}})


class TestBuildGroups(unittest.TestCase):
    """One job per job_group, in plan order, skips excluded."""

    def setUp(self):
        self.plan = {
            'work_dir': '/tmp/round',
            'main_tests': [
                _task('unit_a', 'main_unit'),
                _task('unit_b', 'main_unit'),
                {'name': 'unit_closed', 'mode': 'local', 'skip': True,
                 'skip_reason': 'disabled by _runall',
                 'job_group': 'main_unit'},
            ],
            'example_tests': [
                _task('nn_train_yaml', 'example_nn'),
                _task('nn_predict_yaml', 'example_nn'),
                _task('energy_yaml', 'example_energy'),
            ],
            'skill_tests': [
                _task('case_a_skill', 'skill__g00_nn_eg'),
                _task('case_a_agent', 'skill__g00_nn_eg'),
            ],
            'benchmark_tests': [
                _task('benchmark_fssh', 'benchmark__fssh'),
            ],
        }

    def test_groups_follow_first_appearance_and_keep_plan_indices(self):
        groups = jobs.build_groups(self.plan)
        self.assertEqual(
            [group['gid'] for group in groups],
            ['main_unit', 'example_nn', 'example_energy',
             'skill__g00_nn_eg', 'benchmark__fssh'])
        self.assertEqual(
            [group['short'] for group in groups],
            ['g000', 'g001', 'g002', 'g003', 'g004'])
        self.assertEqual(groups[0]['indices'], [0, 1])
        # The closed task is index 2 and is not a job.
        self.assertEqual(groups[1]['indices'], [3, 4])
        self.assertEqual(groups[2]['indices'], [5])

    def test_a_task_without_a_group_is_a_job_of_one(self):
        plan = {'work_dir': '/tmp/round',
                'main_tests': [{'name': 'solo', 'mode': 'local'}]}
        groups = jobs.build_groups(plan)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['gid'], 'solo')

    def test_indices_match_the_runner_walk(self):
        plan = {'work_dir': '/tmp/round',
                'main_tests': [_task('one', 'g')],
                'example_tests': [{'name': 'closed', 'skip': True,
                                   'skip_reason': 'disabled by x'}],
                'skill_tests': [_task('two', 'g')]}
        walked = [(index, task['name'])
                  for index, _bucket, task in jobs.iter_plan(plan)]
        self.assertEqual(walked, [(0, 'one'), (1, 'closed'), (2, 'two')])
        groups = jobs.build_groups(plan)
        self.assertEqual(groups[0]['indices'], [0, 2])


class TestCollect(unittest.TestCase):
    """A finished job's rows are read; a missing row is an error."""

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.jobs_dir = os.path.join(self.work, 'jobs')
        self.group = {
            'gid': 'example_x', 'short': 'g007', 'job_id': '12345',
            'tasks': [(0, _task('first', 'example_x')),
                      (1, _task('second', 'example_x'))],
        }
        rows = os.path.join(self.jobs_dir, 'example_x', 'rows')
        os.makedirs(rows)
        with open(os.path.join(rows, '0.json'), 'w', encoding='utf-8') as f:
            json.dump({'name': 'first', 'mode': 'pyrai3md_yaml',
                       'status': 'pass'}, f)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def test_the_written_row_survives_and_the_gap_becomes_an_error(self):
        err = os.path.join(self.jobs_dir, 'g007.12345.err')
        with open(err, 'w', encoding='utf-8') as handle:
            handle.write('slurmstepd: error: *** JOB 12345 CANCELLED ***\n')
        rows = dict(jobs.collect(self.work, self.group, self.jobs_dir,
                                 'CANCELLED'))
        self.assertEqual(rows[0]['status'], 'pass')
        self.assertEqual(rows[0]['job_id'], '12345')
        self.assertEqual(rows[1]['status'], 'error')
        self.assertIn('example_x', rows[1]['details'])
        self.assertIn('12345', rows[1]['details'])
        self.assertIn('CANCELLED', rows[1]['details'])
        self.assertIn('slurmstepd', rows[1]['details'])


class TestBenchmarkSubmissionPolicy(unittest.TestCase):
    """Physics calibrations share the cluster; the timing case does not."""

    def test_the_perf_case_is_the_alone_one(self):
        perf = {'gid': 'benchmark__perf',
                'tasks': [(0, {'benchmark_task': 'perf'})]}
        fssh = {'gid': 'benchmark__fssh',
                'tasks': [(0, {'benchmark_task': 'fssh'})]}
        self.assertTrue(jobs._is_timing_benchmark(perf))
        self.assertFalse(jobs._is_timing_benchmark(fssh))

    def test_a_group_named_after_perf_is_the_alone_one(self):
        group = {'gid': 'benchmark__perf', 'tasks': [(0, {})]}
        self.assertTrue(jobs._is_timing_benchmark(group))


class TestReportSequence(unittest.TestCase):
    """The grouped report prints in plan order with one fold per bucket."""

    def test_closed_tasks_fold_at_the_bucket_end(self):
        plan = {
            'work_dir': '/tmp/round',
            'main_tests': [
                _task('unit_a', 'main_unit'),
                {'name': 'unit_closed', 'mode': 'local', 'skip': True,
                 'skip_reason': 'disabled by _runall'},
                _task('unit_b', 'main_unit'),
            ],
            'example_tests': [],
            'skill_tests': [],
            'benchmark_tests': [],
        }
        sequence = runner._build_sequence(plan, {})
        self.assertEqual(
            [item['kind'] for item in sequence],
            ['task', 'task', 'fold'])
        self.assertEqual([item.get('index') for item in sequence[:2]],
                         [0, 2])
        self.assertEqual(sequence[2]['rows'][0]['skip_reason'],
                         'disabled by _runall')

    def test_a_planned_skip_keeps_its_own_row(self):
        plan = {
            'work_dir': '/tmp/round',
            'main_tests': [],
            'example_tests': [
                {'name': 'postponed', 'mode': 'pyrai3md_yaml',
                 'skip': True, 'skip_reason': 'Qualification postponed'},
            ],
            'skill_tests': [],
            'benchmark_tests': [],
        }
        sequence = runner._build_sequence(plan, {})
        self.assertEqual(len(sequence), 1)
        self.assertEqual(sequence[0]['row']['status'], 'skip')


if __name__ == '__main__':
    unittest.main()
