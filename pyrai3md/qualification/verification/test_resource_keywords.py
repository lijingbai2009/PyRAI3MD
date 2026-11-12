"""
======================================================================
Test suite for the compute-resource keywords: who reads them, how the
two spellings combine, and what the framework does with the answer
(dev/SKILL_APP_DEVELOPMENT.md section 5.6, CHANGELOG #306).

Three keywords own three different things. multitask/multipool size the
process pool, remotetask decides who submits, and `thread`/`nproc` say
how many OpenMP threads and how many MPI ranks one calculation may
occupy. The framework-side spellings are `<skill>_thread` and
`<skill>_nproc`, so a run can steer the one skill it names; a skill also
reads the plain `thread`/`nproc`, so a standalone input works without a
framework in front of it. When both spellings arrive and disagree the
skill keeps the larger, and both default to 1. The rank count is the
exception one skill chose to close: openqp_skill merged `nproc` into
`openqp_nproc` (#427) and refuses the bare name.

The run-wide floor #303 built on `environment.num_thread` is cancelled,
and the keyword with it: a thread count is a property of the
calculation, and only the owner of the calculation knows what the engine
will do with it. What replaces it is a read-only check -- the framework
asks each skill's manifest which keywords carry its counts, multiplies
them by the units it is about to start, and warns when that exceeds the
CPUs this run can occupy. Warning only: oversubscription is allowed.

Tests cover:
- the retirement: `num_thread` is refused in &environment and in every
  skill block, with the replacement spelled out;
- the resolution: the two spellings fold to the maximum through the real
  loading chain, and either one alone reaches the engine;
- the declaration: every skill's manifest names the keywords it reads
  and the default it assumes, and the names it declares are the ones its
  own schema defines;
- the arithmetic: units x CPUs per unit against this run's CPU count --
  the local cores or the Slurm allocation the snapshot carries;
- the refusal: a framework workflow section cannot carry them, because
  it consumes processes rather than threads.

Oct 05 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

from pyrai3md.data.build import build_state
from pyrai3md.interface.parser import parse_input
from pyrai3md.kernel.environment import Environment
from pyrai3md.qualification.verification._source_paths import (
    source_read, source_tree)
from pyrai3md.schema import normalize, validate
from pyrai3md.schema.adapter import build_config
from pyrai3md.schema.config import PyRAIConfig
from pyrai3md.skills.query import discover_all

_TEMP_INPUT = """control:
  title: t306_resources
  jobtype: sp
  theory: [pyscf]
atoms: [H, H]
coordinates:
  - [0.0, 0.0, 0.0]
  - [0.0, 0.0, 0.74]
pyscf:
  method: hf
  basis: sto3g
"""

_ORCA_INPUT = _TEMP_INPUT.replace('theory: [pyscf]',
                                  'theory: [orca]').replace(
    'pyscf:\n  method: hf\n  basis: sto3g\n',
    'orca:\n  method: dft\n  basis: def2-svp\n')

_SKILL_FOLDERS = sorted(os.path.basename(path) for path in
                        source_tree('pyrai3md/skills/*_skill'))


class _TempInput(unittest.TestCase):
    """Shared helper: write one yaml into a temp dir and load it."""

    @classmethod
    def setUpClass(cls):
        Environment().initialize()

    def _write(self, text):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'case.yaml')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        return path

    def _load(self, text):
        raw = parse_input(self._write(text))
        return raw, validate(normalize(build_config(raw)))

    def tearDown(self):
        # An `&environment` section writes through to the kernel
        # Environment singleton (schema.adapter._apply_environment_overrides),
        # so a sentinel written below would otherwise travel to every test
        # that runs after this one in the same process.
        Environment().initialize()

    def _skill_configs(self, text):
        """The per-skill config dicts the task queue will hand over."""
        raw, config = self._load(text)
        state = build_state(config, raw_config=raw)
        configs = [task['config']['skill_config']
                   for _identity, task in state.iter_tasks()]
        self.assertTrue(configs, 'the run staged no task')
        return configs


class TestRetiredFrameworkKeyword(_TempInput):
    """`num_thread` no longer means anything to the framework."""

    def test_environment_section_is_refused(self):
        text = _TEMP_INPUT + 'environment:\n  num_thread: 4\n'
        with self.assertRaises(ValueError) as caught:
            self._load(text)
        message = str(caught.exception)
        self.assertIn('`num_thread`', message)
        self.assertIn('_thread', message,
                      'the refusal has to name the replacement')

    def test_the_model_carries_no_such_field(self):
        self.assertFalse(hasattr(PyRAIConfig().environment, 'num_thread'),
                         'a field nobody writes is a second source of a '
                         'decision the skill already owns')

    def test_scratch_path_is_still_readable(self):
        # &environment keeps its one live job: saying where the run log
        # goes. The rest of the section stays available to the schema.
        text = _TEMP_INPUT.replace(
            'pyscf:', 'environment:\n  scratch_path: run_dir\npyscf:')
        _raw, config = self._load(text)
        self.assertEqual(config.environment.scratch_path, 'run_dir')

    def test_the_snapshot_carries_scratch_path_to_the_state(self):
        text = _TEMP_INPUT.replace(
            'pyscf:', 'environment:\n  scratch_path: run_dir\npyscf:')
        raw, config = self._load(text)
        state = build_state(config, raw_config=raw)
        self.assertEqual(state.pmd.environment.scratch_path, 'run_dir')


class TestTwoSpellingsFoldToTheMaximum(_TempInput):
    """The skill reads one name; the framework writes the other."""

    def test_plain_spelling_reaches_the_engine(self):
        text = _TEMP_INPUT.replace(
            '  basis: sto3g\n', '  basis: sto3g\n  thread: 4\n')
        self.assertEqual([c['thread'] for c in self._skill_configs(text)],
                         [4])

    def test_framework_spelling_survives_the_queue(self):
        # The framework does not run the skill schema pipeline, so the
        # queue carries both spellings verbatim and the skill folds them
        # when it builds its own config.
        text = _TEMP_INPUT.replace(
            '  basis: sto3g\n', '  basis: sto3g\n  pyscf_thread: 5\n')
        queued = self._skill_configs(text)[0]
        self.assertEqual(queued['pyscf_thread'], 5)
        self.assertEqual(queued['thread'], 1)

    def test_the_skill_keeps_the_larger(self):
        from pyrai3md.skills.pyscf_skill.schema.adapter import \
            build_config as skill_build_config
        folded = skill_build_config({'skill_name': 'pyscf_skill',
                                     'thread': 4, 'pyscf_thread': 7})
        self.assertEqual(folded.environment.thread, 7)
        other = skill_build_config({'skill_name': 'pyscf_skill',
                                    'thread': 7, 'pyscf_thread': 4})
        self.assertEqual(other.environment.thread, 7,
                         'the fold is a maximum, not last-write-wins')

    def test_nothing_written_stays_one(self):
        from pyrai3md.skills.pyscf_skill.schema.adapter import \
            build_config as skill_build_config
        self.assertEqual(skill_build_config(
            {'skill_name': 'pyscf_skill'}).environment.thread, 1)

    def test_the_ranks_pair_folds_the_same_way(self):
        from pyrai3md.skills.orca_skill.schema.adapter import \
            build_config as skill_build_config
        folded = skill_build_config({'skill_name': 'orca_skill',
                                     'nproc': 2, 'orca_nproc': 5})
        self.assertEqual(folded.environment.nproc, 5)


class TestEverySkillRetiresTheOldNames(unittest.TestCase):
    """Each skill refuses the keywords it used to read."""

    def test_num_thread_is_refused_by_every_skill(self):
        import importlib
        offenders = []
        for folder in _SKILL_FOLDERS:
            adapter = importlib.import_module(
                'pyrai3md.skills.%s.schema.adapter' % folder)
            self.assertIn('num_thread', adapter.RETIRED_RESOURCES, folder)
            with self.assertRaises(ValueError, msg=folder) as caught:
                adapter.build_config({'skill_name': folder,
                                      'num_thread': 4})
            if '_thread' not in str(caught.exception):
                offenders.append(folder)
        self.assertEqual([], offenders)

    def test_the_process_count_names_are_refused_where_they_looked(self):
        from pyrai3md.skills.cp2k_skill.schema.adapter import \
            build_config as cp2k_build_config
        from pyrai3md.skills.openqp_skill.schema.adapter import \
            build_config as openqp_build_config
        from pyrai3md.skills.xtb_skill.schema.adapter import \
            build_config as xtb_build_config
        with self.assertRaises(ValueError):
            cp2k_build_config({'skill_name': 'cp2k_skill', 'mpi': 4})
        with self.assertRaises(ValueError):
            openqp_build_config({'skill_name': 'openqp_skill',
                                 'mpi_nproc': 4})
        with self.assertRaises(ValueError) as caught:
            # #427: openqp_skill merged the two rank spellings into one, so
            # a bare `nproc` cannot be set and then ignored by the launcher.
            openqp_build_config({'skill_name': 'openqp_skill', 'nproc': 4})
        self.assertIn('openqp_nproc', str(caught.exception))
        with self.assertRaises(ValueError):
            # xtb has no MPI arm: `xtb_nproc` counted threads under a
            # process name, which is the confusion #306 removes.
            xtb_build_config({'skill_name': 'xtb_skill', 'xtb_nproc': 4})


class TestManifestDeclarations(unittest.TestCase):
    """The framework asks each skill how to read its own counts."""

    def test_every_skill_declares_its_threads(self):
        for label, meta in sorted(discover_all().items()):
            declaration = (meta.get('resources') or {}).get('threads')
            self.assertIsInstance(declaration, dict, label)
            self.assertEqual(declaration.get('keyword'), 'thread', label)

    def test_only_mpi_skills_declare_ranks(self):
        rank_skills = {'bagel', 'cp2k', 'lammps', 'molcas', 'openqp',
                       'orca'}
        for label, meta in sorted(discover_all().items()):
            resources = meta.get('resources') or {}
            self.assertEqual('ranks' in resources, label in rank_skills,
                             label)
            if 'ranks' in resources:
                # The rank count is the one resource a skill may own under
                # its own name only: openqp_skill merged `nproc` into
                # `openqp_nproc` (#427) so a run cannot set a rank count the
                # launcher then ignores.
                keyword = resources['ranks'].get('keyword')
                prefix = meta['skill_name'][:-len('_skill')]
                self.assertIn(keyword, ('nproc', '%s_nproc' % prefix), label)

    def test_declared_names_exist_in_the_skill_schema(self):
        # Drift guard: a manifest that names a keyword the skill does
        # not define would let the framework compute a demand the skill
        # never applies.
        import importlib
        for label, meta in sorted(discover_all().items()):
            folder = meta['skill_name']
            module = importlib.import_module(
                'pyrai3md.skills.%s.schema.config' % folder)
            fields = set(module.Environment.model_fields)
            prefix = folder[:-len('_skill')]
            for entry in (meta.get('resources') or {}).values():
                keyword = entry['keyword']
                self.assertIn(keyword, fields, '%s %s' % (label, keyword))
                if not keyword.startswith('%s_' % prefix):
                    # A bare declaration means the pair convention, so the
                    # prefixed spelling must exist to be folded with it.
                    self.assertIn('%s_%s' % (prefix, keyword), fields,
                                  '%s %s' % (label, prefix))
                self.assertEqual(module.DEFAULT.get(keyword),
                                 entry.get('default'),
                                 '%s default drifts from its schema'
                                 % label)

    def test_the_merged_ranks_keyword_is_the_one_the_launcher_reads(self):
        # The declaration is only honest if the surviving name is what the
        # run command actually uses: #427 merged `nproc` into
        # `openqp_nproc`, and a launcher left reading the old attribute would
        # start one rank no matter what the run asked for.
        templates = _read('pyrai3md/skills/openqp_skill/service/templates/'
                          'run_templates.py')
        self.assertIn('mpirun -n {config.openqp_nproc}', templates)
        self.assertNotIn('config.nproc', templates)


def _scheduler(state, multipool=1):
    from pyrai3md.workflow.dispatch.scheduler import Scheduler
    return Scheduler(state=state, multipool=multipool)


class TestSchedulerResourceArithmetic(_TempInput):
    """Units times CPUs per unit, read from the queue that will run."""

    def _orca_state(self, extra=''):
        raw, config = self._load(_ORCA_INPUT + extra)
        return build_state(config, raw_config=raw)

    def test_threads_only_skill_demands_its_threads(self):
        raw, config = self._load(_TEMP_INPUT.replace(
            '  basis: sto3g\n', '  basis: sto3g\n  thread: 4\n'))
        state = build_state(config, raw_config=raw)
        self.assertEqual(_scheduler(state)._unit_demand(),
                         (4, 'pyscf_skill'))

    def test_hybrid_skill_demands_ranks_times_threads(self):
        state = self._orca_state('  thread: 3\n  nproc: 2\n')
        self.assertEqual(_scheduler(state)._unit_demand(),
                         (6, 'orca_skill'))

    def test_a_missing_keyword_falls_back_to_the_declared_default(self):
        state = self._orca_state('  nproc: 4\n')
        self.assertEqual(_scheduler(state)._unit_demand(),
                         (4, 'orca_skill'),
                         'thread is unwritten, so its manifest default '
                         'counts, and it counts as 1')

    def test_the_framework_spelling_counts(self):
        state = self._orca_state('  orca_thread: 3\n  nproc: 2\n')
        self.assertEqual(_scheduler(state)._unit_demand(),
                         (6, 'orca_skill'))

    def test_a_one_rank_one_thread_run_demands_one_cpu(self):
        state = self._orca_state('  thread: 1\n  nproc: 1\n'
                                 '  orca_nproc: 1\n')
        self.assertEqual(_scheduler(state)._unit_demand(), (1, ''),
                         'the retired framework field used to be the only '
                         'source and could report nothing but 1 (#303)')

    def test_an_omitted_keyword_reads_what_the_run_already_merged(self):
        # Nothing written in the skill section: the environment layer
        # still says how many ranks orca runs (`orca_nproc` of
        # qualification/pmd_env.yaml, unless the user layer overrides it
        # in ~/.pyrai3md/pmd_env.yaml), and the demand is that number --
        # taken from the queue, not guessed here. The name is only asked
        # for when the demand exceeds one CPU: `_unit_demand` reports
        # `(1, '')` for a run that declared 1 (#303), and a machine whose
        # user layer pins `orca_nproc: 1` is exactly that run, so the
        # assertion follows the demand it read instead of a literal 2.
        state = self._orca_state()
        demand, skill = _scheduler(state)._unit_demand()
        queued = next(task['config']['skill_config']
                      for _identity, task in state.iter_tasks())
        self.assertIn('orca_nproc', queued,
                      'the environment layer did not reach the queue, so '
                      'this case no longer tests the omitted keyword')
        self.assertEqual(skill, 'orca_skill' if demand > 1 else '')
        self.assertEqual(demand, max(1, int(queued['orca_nproc'])))

    def test_the_term_pool_multiplies_the_units(self):
        state = self._orca_state()
        self.assertEqual(_scheduler(state)._term_workers(), 1)
        config = PyRAIConfig()
        config.control.jobtype = 'sp'
        config.control.theory = ['orca']
        config.system.atoms = ['H', 'H']
        config.system.coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        config.system.natom = 2
        config.composite.executor = 'thread'
        config.composite.max_workers = 3
        threaded = _scheduler(build_state(validate(normalize(config))))
        self.assertEqual(threaded._term_workers(), 3)

    def test_local_reference_is_this_machine(self):
        state = self._orca_state()
        self.assertEqual(_scheduler(state)._available_cpus('local'),
                         max(1, os.cpu_count() or 1))

    def test_remote_reference_is_the_slurm_allocation(self):
        state = self._orca_state()
        state.pmd.environment.slurm = {'ntasks': 2, 'cpus_per_task': 4}
        self.assertEqual(_scheduler(state)._available_cpus('slurm_run'), 8,
                         'a submitted run lands on its allocation, not on '
                         'the machine that parsed the input')

    def test_oversubscription_warns_and_does_not_stop(self):
        state = self._orca_state('  thread: 3\n  nproc: 2\n')
        scheduler = _scheduler(state, multipool=4)
        recorded = []
        original = scheduler.logger.warning
        scheduler.logger.warning = (
            lambda message, *args: recorded.append(
                message % args if args else message))
        try:
            # 4 units x 6 CPUs = 24 against a machine with 10 cores.
            scheduler.validate_resources('local', 4)
        finally:
            scheduler.logger.warning = original
        demand = 4 * 6
        if demand > scheduler._available_cpus('local'):
            self.assertTrue(
                any('4 unit(s) x 6 CPU(s) per unit' in line
                    for line in recorded), recorded)
        else:  # pragma: no cover - a machine with 24+ cores
            self.assertEqual(recorded, [])

    def test_a_request_that_fits_is_silent(self):
        state = self._orca_state()
        scheduler = _scheduler(state)
        recorded = []
        original = scheduler.logger.warning
        scheduler.logger.warning = (
            lambda message, *args: recorded.append(
                message % args if args else message))
        try:
            scheduler.validate_resources('local', 1)
        finally:
            scheduler.logger.warning = original
        self.assertEqual(recorded, [])


class TestRunLogDirectory(_TempInput):
    """The scratch root the run names is created before the log opens."""

    def _state_with_scratch(self, scratch):
        text = _TEMP_INPUT.replace(
            'pyscf:', 'environment:\n  scratch_path: %s\npyscf:' % scratch)
        raw, config = self._load(text)
        return build_state(config, raw_config=raw)

    def test_a_missing_scratch_root_is_created(self):
        from pyrai3md.kernel.logger import RunLogger
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base, True)
        target = os.path.join(base, 'not_yet_made')
        state = self._state_with_scratch(target)
        self.assertEqual(state.pmd.environment.scratch_path, target)
        logger = RunLogger(state, log_dir=state.pmd.environment.scratch_path)
        logger.write('probe')
        logger.finish('completed')
        self.assertTrue(os.path.isfile(
            os.path.join(target, 't306_resources.log')),
            'a dispatched child that cannot open its log only shows up to '
            'the parent as a missing result file')

    def test_an_unset_root_writes_beside_the_launch_directory(self):
        from pyrai3md.kernel.logger import RunLogger
        previous = os.getcwd()
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base, True)
        state = self._state_with_scratch('')
        self.assertEqual(state.pmd.environment.scratch_path, '')
        os.chdir(base)
        try:
            logger = RunLogger(state, log_dir=None)
            logger.write('probe')
            logger.finish('completed')
            self.assertTrue(os.path.isfile(
                os.path.join(base, 't306_resources.log')))
        finally:
            os.chdir(previous)


class TestWorkflowSectionsDeclineTheKeywords(_TempInput):
    """A workflow section consumes processes, not threads or ranks."""

    def test_every_framework_section_is_refused(self):
        from pyrai3md.schema.adapter import SECTION_TYPES
        for section in sorted(SECTION_TYPES):
            for keyword in ('thread', 'nproc', 'num_thread',
                            'orca_thread'):
                text = _ORCA_INPUT + '%s:\n  %s: 5\n' % (section, keyword)
                with self.assertRaises(ValueError,
                                       msg='%s %s' % (section, keyword)):
                    self._load(text)

    def test_the_skill_section_is_still_the_right_place(self):
        text = _ORCA_INPUT + '  thread: 3\n'
        self.assertEqual(self._skill_configs(text)[0]['thread'], 3)


class TestNoSkillPinsTheThreadCount(unittest.TestCase):
    """Every skill forwards the keyword; none hard-codes a number."""

    def _sources(self):
        import re
        paths = source_tree('pyrai3md/skills/*_skill/**/*.py')
        offenders = []
        pattern = re.compile(r'OMP_NUM_THREADS=\d')
        for path in paths:
            if os.sep + 'tests' + os.sep in path:
                continue
            with open(path, encoding='utf-8') as handle:
                for number, line in enumerate(handle, start=1):
                    if pattern.search(line):
                        offenders.append('%s:%d' % (path, number))
        return offenders

    def test_no_run_script_pins_a_literal_thread_count(self):
        self.assertEqual([], self._sources(),
                         'the thread count belongs to the keyword, not to '
                         'a literal in a template')

    def test_the_three_lammps_arms_all_read_the_keyword(self):
        """Subprocess script, slurm allocation, resident service script."""
        templates = _read('pyrai3md/skills/lammps_skill/service/templates/'
                          'run_templates.py')
        launcher = _read('pyrai3md/skills/lammps_skill/service/persistent/'
                         'launcher.py')
        self.assertIn('num_thread = config.thread', templates)
        self.assertIn('export OMP_NUM_THREADS={num_thread}', templates)
        self.assertIn('cpus_per_task = nproc * num_thread', templates)
        self.assertIn('export OMP_NUM_THREADS={config.thread}', launcher)


def _read(path):
    return source_read(path)


if __name__ == '__main__':
    unittest.main()
