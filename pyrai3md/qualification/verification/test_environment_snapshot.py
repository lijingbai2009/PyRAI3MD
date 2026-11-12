"""
======================================================================
Test suite for the framework environment snapshot carried by
`PyRAIConfig.environment` (state.pmd.environment).

The snapshot is frozen onto the config by schema.adapter.build_config
from the merged pmd_env.yaml and the run input, and the dispatch layer
documents it as its single data source (workflow/dispatch/slurm_run.py
builds the child runtime submission script from it;
workflow/dispatch/runtime.py flattens its export section).

schema.normalize and schema.validate rebuild the config through a flat
namespace (flatten_config_dump -> partition_to_submodels ->
Config.model_validate). That round trip cannot represent the
environment section: FIELD_TO_SUBMODEL routes none of its keys, and
names like shell/program/device share the flat namespace with the
workflow sections -- the same collision the dispatch keywords are
restored for. Without the RESTORED_NESTED_SECTIONS restore step the
rebuild silently replaces every machine setting of the run with its
default, and the child runtime scripts come out bare.

Tests cover:
- preservation: normalize and validate each keep the environment
  section field for field (sentinel values, no machine dependency);
- the mechanism: the flat round trip alone loses the section, so the
  restore step is load-bearing and not decoration;
- no cross-section leak: an environment key never lands in a workflow
  or system section;
- artifact: the generated Slurm child-runtime script carries the
  #SBATCH directives, the shell commands and the export lines;
- carriers: the allocation a dispatched child submits with is the same
  whether a YAML block, a classic `&SLURM` section or an agent `.config`
  JSON stated it, is applied with or without an `&environment` section
  beside it, and is not inherited by the next run in the process.

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from pyrai3md.schema.config import (flatten_config_dump,
                                partition_to_submodels)
import importlib
import os
import tempfile
import unittest

# pyrai3md.schema re-exports the normalize/validate functions, which
# shadow the submodule attributes, so the modules are taken from
# importlib rather than by attribute access.
normalize_module = importlib.import_module('pyrai3md.schema.normalize')
validate_module = importlib.import_module('pyrai3md.schema.validate')

EXAMPLES_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'examples'))

# Sentinel machine settings. They are written onto the config directly
# so the suite never depends on what the local pmd_env.yaml happens to
# declare.
_SENTINEL = {
    'shell': 'module load sentinel/gcc\nmodule load sentinel/mpi',
    'environment': {'MPI_NP': '4', 'OMP_NUM_THREADS': '2'},
    'slurm': {'partition': 'sentinel-q', 'mem': '7G', 'time': '2:00:00'},
    'scratch_path': '/sentinel/scratch',
    'device': 'sentinel-device',
    'program_path': '/sentinel/bin',
}


def _load_example(name='md_batch.yaml'):
    """Run the real load chain up to (but excluding) normalize.

    Parameters
    ----------
    name : str
        Shipped example file name.

    Returns
    -------
    PyRAIConfig
        The config as build_config freezes it.
    """
    from pyrai3md.interface.parser import parse_input
    from pyrai3md.schema.adapter import build_config

    previous = os.getcwd()
    workdir = tempfile.mkdtemp(prefix='env_snapshot_chain_')
    try:
        os.chdir(workdir)
        raw = parse_input(os.path.join(EXAMPLES_DIR, name))
        return build_config(raw)
    finally:
        os.chdir(previous)


def _with_sentinels(config):
    """Stamp the sentinel machine settings onto a config.

    Parameters
    ----------
    config : PyRAIConfig
        Config whose environment section is overwritten.

    Returns
    -------
    PyRAIConfig
        The same config, with the sentinel values applied.
    """
    env = config.environment
    for key, value in _SENTINEL.items():
        setattr(env, key, value)
    return config


class TestEnvironmentSurvivesRebuild(unittest.TestCase):
    """normalize/validate must not flatten the environment section."""

    def test_normalize_preserves_every_environment_field(self):
        config = _with_sentinels(_load_example())
        before = config.environment.model_dump()
        after = normalize_module.normalize(config).environment.model_dump()
        self.assertEqual(before, after)

    def test_validate_preserves_every_environment_field(self):
        config = _with_sentinels(_load_example())
        before = config.environment.model_dump()
        after = validate_module.validate(config).environment.model_dump()
        self.assertEqual(before, after)

    def test_full_chain_preserves_the_snapshot(self):
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate

        config = _with_sentinels(_load_example())
        env = validate(normalize(config)).environment
        self.assertEqual(env.shell, _SENTINEL['shell'])
        self.assertEqual(env.environment, _SENTINEL['environment'])
        self.assertEqual(env.slurm, _SENTINEL['slurm'])
        self.assertEqual(env.scratch_path, _SENTINEL['scratch_path'])
        self.assertEqual(env.device, _SENTINEL['device'])
        self.assertEqual(env.program_path, _SENTINEL['program_path'])

    def test_flat_round_trip_alone_loses_the_section(self):
        # Negative control for why the restore step exists: the flat
        # namespace cannot route environment back, so without
        # RESTORED_NESTED_SECTIONS the rebuild yields the defaults.
        config = _with_sentinels(_load_example())
        nested = config.model_dump()
        flat = flatten_config_dump(nested)
        partitioned = partition_to_submodels(flat)
        self.assertNotIn('environment', partitioned)
        self.assertNotIn('shell', partitioned.get('system', {}))
        self.assertNotIn('shell', partitioned.get('optimize', {}))

    def test_restore_step_is_what_keeps_the_snapshot(self):
        # Causal control: put the restore list back to what it was
        # before the fix (the workflow sections only) and the same
        # rebuild really does drop the snapshot. Without this, the
        # preservation tests above would only prove today's behavior,
        # not which line is responsible for it.
        config = _with_sentinels(_load_example())
        saved = normalize_module.RESTORED_NESTED_SECTIONS
        try:
            normalize_module.RESTORED_NESTED_SECTIONS = tuple(
                key for key in saved if key != 'environment')
            dropped = normalize_module.normalize(config).environment
        finally:
            normalize_module.RESTORED_NESTED_SECTIONS = saved
        self.assertNotEqual(dropped.shell, _SENTINEL['shell'])
        self.assertEqual(dropped.slurm, {})
        self.assertEqual(dropped.environment, {})

    def test_no_environment_key_leaks_into_another_section(self):
        from pyrai3md.schema.normalize import normalize

        config = _with_sentinels(_load_example())
        result = normalize(config)
        dumped = result.model_dump()
        for section, values in dumped.items():
            if section == 'environment' or not isinstance(values, dict):
                continue
            leaked = set(_SENTINEL) & set(values)
            self.assertFalse(
                leaked,
                'environment keys leaked into %s: %s' % (section, leaked))


class TestChildRuntimeScriptContent(unittest.TestCase):
    """The dispatch layer builds the submission script from the state."""

    def test_script_carries_sbatch_shell_and_exports(self):
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager)

        slurm_run = importlib.import_module(
            'pyrai3md.workflow.dispatch.slurm_run')

        config = _with_sentinels(_load_example())
        config.control.jobtype = 'energy'
        state = build_state(validate(normalize(config)),
                            raw_config={'control': {}})

        previous = os.getcwd()
        workdir = tempfile.mkdtemp(prefix='env_snapshot_script_')
        try:
            os.chdir(workdir)
            manager = RuntimeDirectoryManager(skill_name='pyscf_skill',
                                              root=workdir)
            script = slurm_run._build_script(state, manager, 1)
        finally:
            os.chdir(previous)

        self.assertIn('#SBATCH --partition=sentinel-q', script)
        self.assertIn('#SBATCH --mem=7G', script)
        self.assertIn('module load sentinel/gcc', script)
        self.assertIn('module load sentinel/mpi', script)
        self.assertIn('export MPI_NP=4', script)
        self.assertIn('export OMP_NUM_THREADS=2', script)
        self.assertIn('python3 -m pyrai3md', script)


# The allocation one run asks its dispatched children to submit with. Each
# value differs from what the shipped pmd_env.yaml states for that key, so an
# assertion below cannot be satisfied by the defaults the host would read if
# the input's section were dropped on the way.
# `exclusive` is the one key the two agree on: its failure mode is a text
# 'false' read as truth, which would add a directive the disk never writes,
# and `test_the_one_boolean_directive_is_read_as_a_boolean` judges that
# without a baseline to compare against.
_SLURM_CASE = {
    'partition': 'carriertesting-q', 'time': '0:37:00', 'mem': '3G',
    'cpus_per_task': 2, 'ntasks': 3, 'exclusive': False,
}

_CARRIER_CASE = {
    'control': {'title': 'slurm_carrier', 'jobtype': 'energy',
                'theory': ['nn']},
    'atoms': ['C', 'H', 'H', 'H', 'H'],
    'coordinates': [
        [0.0000000, 0.0000000, 0.0000000],
        [1.0890000, 0.0000000, 0.0000000],
        [-0.3630000, 1.0280000, 0.0000000],
        [-0.3630000, -0.5140000, 0.8890000],
        [-0.3630000, -0.5140000, -0.8890000],
    ],
    'molecule': {'charges': 0, 'multiplicity': 1},
    'nn': {'method': 'mlp_eg', 'modeldir': './models'},
    'slurm': dict(_SLURM_CASE),
}


def _disk_slurm():
    """What the pmd_env.yaml files state, with no input merged in.

    `reset_input_overrides` is called because the kernel environment is a
    process singleton: by the time a test reads a baseline, an earlier test
    has already merged an input into it.
    """
    from pyrai3md.kernel.environment import Environment

    env = Environment()
    env.initialize()
    env.reset_input_overrides()
    return env.get_slurm()


class TestSlurmSectionCarriers(unittest.TestCase):
    """The same allocation, whichever file stated it.

    Three carriers write a run's `slurm` section: the YAML block, the
    classic `&SLURM` text section, and the agent `.config` JSON. The route
    they must share is the one the job-script builder reads --
    `Environment.get_slurm` -> the `state.pmd.environment.slurm` snapshot
    -> `dispatch/slurm_run` -- and each of the three had a way to miss it:
    the merge sat behind the `&environment` guard (so an input that
    declares no `&environment` block, which is the normal case, lost every
    directive), and the kernel environment is a process singleton whose
    merged config outlives the run that wrote into it.

    The classic text states every value as a string, so the numeric
    directives arrive as '2' where the YAML gives 2. What is compared is
    what the consumer does with them: `slurm_run` writes each value into
    its `#SBATCH` line, and the lines are what the cluster reads.
    """

    @classmethod
    def setUpClass(cls):
        import yaml

        cls.work = tempfile.mkdtemp(prefix='slurm_carrier_')
        yaml_path = os.path.join(cls.work, 'case.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as handle:
            handle.write(yaml.safe_dump(_CARRIER_CASE))

        from pyrai3md.qualification.certification.converter import (
            convert, convert_to_pyrai_config, yaml_to_config)

        flat_path = os.path.join(cls.work, 'case.json')
        yaml_to_config(yaml_path, flat_path)
        input_path = os.path.join(cls.work, 'case.input')
        convert(flat_path, input_path)

        from pyrai3md.interface.parser import parse_input

        previous = os.getcwd()
        try:
            os.chdir(cls.work)
            cls.carriers = {
                'yaml': parse_input(yaml_path),
                'input': parse_input(input_path),
                'agent': convert_to_pyrai_config(flat_path),
            }
            # The classic text is the one place a false value arrives as a
            # word; render it once here so a test can name it.
            with open(input_path, encoding='utf-8') as handle:
                cls.text = handle.read()
        finally:
            os.chdir(previous)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.work, ignore_errors=True)

    def _slurm_of(self, label):
        """Load one carrier through the framework and read the snapshot."""
        from pyrai3md.schema.adapter import build_config

        previous = os.getcwd()
        run = tempfile.mkdtemp(prefix='slurm_carrier_run_')
        try:
            os.chdir(run)
            return build_config(self.carriers[label]).environment.slurm
        finally:
            os.chdir(previous)

    def _sbatch_lines_of(self, label):
        """The directives as the cluster receives them, in build order."""
        from pyrai3md.data.build import build_state
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        from pyrai3md.workflow.dispatch.runtime import (
            RuntimeDirectoryManager)

        slurm_run = importlib.import_module(
            'pyrai3md.workflow.dispatch.slurm_run')

        previous = os.getcwd()
        run = tempfile.mkdtemp(prefix='slurm_carrier_script_')
        try:
            os.chdir(run)
            config = build_config(self.carriers[label])
            state = build_state(validate(normalize(config)),
                                raw_config={'control': {}})
            manager = RuntimeDirectoryManager(skill_name='nn_pes_skill',
                                              root=run)
            script = slurm_run._build_script(state, manager, 1)
        finally:
            os.chdir(previous)
        return [line for line in script.splitlines()
                if line.startswith('#SBATCH')]

    def test_the_classic_carrier_renders_the_section(self):
        # Without this the equality below could be vacuous in either
        # direction: a converter that dropped the section would leave the
        # text carrier reading the host defaults, and the test would pass
        # by comparing two omissions.
        text = self.text.upper()
        start = text.find('&SLURM')
        self.assertGreaterEqual(start, 0, 'no &SLURM block was rendered')
        block = self.text[start:].splitlines()
        stated = {}
        for line in block[1:]:
            if line.startswith('&'):
                break
            tokens = line.split()
            if len(tokens) == 2:
                stated[tokens[0]] = tokens[1]
        self.assertEqual(stated, {key: ('false' if value is False
                                        else str(value))
                                  for key, value in _SLURM_CASE.items()})

    def test_every_directive_arrives_from_every_carrier(self):
        for label in sorted(self.carriers):
            slurm = self._slurm_of(label)
            for key, value in _SLURM_CASE.items():
                self.assertEqual(str(slurm.get(key)), str(value),
                                 '%s carrier lost %s' % (label, key))

    def test_the_cluster_is_asked_the_same_whatever_wrote_it(self):
        lines = {label: self._sbatch_lines_of(label)
                 for label in sorted(self.carriers)}
        for label in sorted(lines):
            block = lines[label]
            for directive in ('#SBATCH --partition=carriertesting-q',
                              '#SBATCH --time=0:37:00',
                              '#SBATCH --mem=3G',
                              '#SBATCH --cpus-per-task=2',
                              '#SBATCH --ntasks=3'):
                self.assertIn(directive, block, label)
            # A boolean false writes no directive at all: `exclusive false`
            # read as text would have reserved the whole node for a fit that
            # asks to share it.
            self.assertFalse([line for line in block
                              if line.startswith('#SBATCH --exclusive')],
                             label)
        # The builder emits in its own order, so the sets are what must
        # agree: the YAML gives an integer and the classic text the string
        # '2' for the same allocation, and both become one directive line.
        self.assertEqual(set(lines['yaml']), set(lines['input']),
                         'the classic text and the YAML block asked the '
                         'cluster for different allocations')
        self.assertEqual(set(lines['yaml']), set(lines['agent']),
                         'the agent config asked for a third allocation')

    def test_the_directives_apply_with_or_without_an_environment_section(self):
        # The merge used to sit behind the guard that returns early when the
        # input declares no `&environment` section, which is how a scan
        # example is written. Both arms must now reach the snapshot, and the
        # shell has to arrive in the arm that states it, or the comparison
        # would be two runs that both skipped the section.
        from pyrai3md.schema.adapter import build_config

        bare = self._slurm_of('yaml')
        with_env = dict(self.carriers['yaml'],
                        environment={'shell': 'module load carrier/gcc'})
        previous = os.getcwd()
        run = tempfile.mkdtemp(prefix='slurm_carrier_guard_')
        try:
            os.chdir(run)
            stated = build_config(with_env)
        finally:
            os.chdir(previous)
        self.assertEqual({key: str(stated.environment.slurm.get(key))
                          for key in _SLURM_CASE},
                         {key: str(bare.get(key)) for key in _SLURM_CASE})
        self.assertEqual(stated.environment.shell, 'module load carrier/gcc')
        # The host's own section is what is extended, not replaced: a
        # directive the input never mentions keeps the value the disk states.
        disk = _disk_slurm()
        unmentioned = [key for key in disk if key not in _SLURM_CASE]
        self.assertTrue(unmentioned, 'the disk states no extra directive')
        for key in unmentioned:
            self.assertEqual(str(bare.get(key)), str(disk.get(key)), key)

    def test_the_one_boolean_directive_is_read_as_a_boolean(self):
        from pyrai3md.kernel.environment import Environment

        env = Environment()
        env.initialize()
        try:
            # The classic text has no boolean type, so `exclusive false`
            # arrives as the word 'false'. Taken as truth it would add
            # '#SBATCH --exclusive' and reserve a whole node per grid point
            # for a scan that asks to share one.
            env.reset_input_overrides()
            env.apply_slurm_overrides({'exclusive': 'false'})
            self.assertIs(env.get_slurm()['exclusive'], False)
            env.reset_input_overrides()
            env.apply_slurm_overrides({'exclusive': 'true'})
            self.assertIs(env.get_slurm()['exclusive'], True)
            env.reset_input_overrides()
            env.apply_slurm_overrides({'exclusive': 'no'})
            self.assertIs(env.get_slurm()['exclusive'], False)
        finally:
            env.reset_input_overrides()

    def test_a_run_does_not_inherit_the_previous_run_s_directives(self):
        # `Environment` is a process singleton and its merged config is not
        # rebuilt per input, so the run that declares `partition: x` leaves
        # x in the dictionary the next run reads as its host default. An
        # input must be able to override the disk, not the last caller.
        from pyrai3md.schema.adapter import build_config

        disk = _disk_slurm()
        previous = os.getcwd()
        run = tempfile.mkdtemp(prefix='slurm_carrier_scope_')
        try:
            os.chdir(run)
            build_config(self.carriers['yaml'])
            quiet = build_config(
                {key: value for key, value in self.carriers['yaml'].items()
                 if key != 'slurm'}).environment.slurm
        finally:
            os.chdir(previous)
        self.assertEqual({key: str(value) for key, value in quiet.items()},
                         {key: str(value) for key, value in disk.items()},
                         'the second run read the first run\'s directives')


if __name__ == '__main__':
    unittest.main()
