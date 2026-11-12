"""
======================================================================
Test suite for the dispatch keywords of the shipped MD fan-out
examples (pyrai3md/examples/md_parallel.yaml and md_batch.yaml).

The trajectory fan-out is a workflow-node decision, so the examples must
spell it in the channel the execution-graph planner actually reads
(&md multitask / remotetask / multipool). The retired
&environment workflow_execution / workflow_workers pair is not read by
any component: it reaches neither the planner nor the state, so an
example carrying it silently runs its trajectories one after another.

These cases walk the real load chain (parse_input -> build_config ->
normalize -> validate -> build_state -> ExecutionGraphBuilder) and read
the frozen graph. No backend is instantiated and no program is executed.

Oct 04 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

try:
    import pytest
except ImportError:  # unittest-only hosts do not need pytest
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.integration

import os
import tempfile
import unittest

EXAMPLES_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'examples'))


def _resolve_workflow_node(name):
    """Resolve the workflow node of one shipped example.

    Parameters
    ----------
    name : str
        Example file name inside the shipped examples directory.

    Returns
    -------
    dict
        The frozen workflow node descriptor.
    """
    from pyrai3md.interface.parser import parse_input
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.schema.normalize import normalize
    from pyrai3md.schema.validate import validate
    from pyrai3md.data.build import build_state
    from pyrai3md.workflow import resolve_workflow, WORKFLOW_REGISTRY
    from pyrai3md.workflow.dispatch.planner import ExecutionGraphBuilder

    path = os.path.join(EXAMPLES_DIR, name)
    previous = os.getcwd()
    workdir = tempfile.mkdtemp(prefix='example_dispatch_')
    try:
        os.chdir(workdir)
        raw_config = parse_input(path)
        config = validate(normalize(build_config(raw_config)))
        state = build_state(config, raw_config=raw_config)
        graph = ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
            state, resolve_workflow(state))
    finally:
        os.chdir(previous)
        from shutil import rmtree
        rmtree(workdir, ignore_errors=True)
    return graph[0]


class TestShippedMdFanOutKeywords(unittest.TestCase):
    """The fan-out keywords of the md examples reach the graph node."""

    def test_md_parallel_requests_the_local_pool(self):
        node = _resolve_workflow_node('md_parallel.yaml')
        self.assertEqual(node['name'], 'md')
        self.assertTrue(node['multitask'])
        self.assertFalse(node['remotetask'])
        self.assertEqual(node['multipool'], 3)

    def test_md_batch_requests_the_remote_pool(self):
        node = _resolve_workflow_node('md_batch.yaml')
        self.assertEqual(node['name'], 'md')
        self.assertTrue(node['multitask'])
        self.assertTrue(node['remotetask'])
        self.assertEqual(node['multipool'], 3)

    def test_environment_sections_carry_no_retired_key(self):
        # Repo-wide guard: the retired keys read as a setting while
        # nothing consumed them (#66 retired workflow_execution /
        # workflow_workers, #300 retires the framework-level
        # execution_mode and refuses all three at the input). The scan
        # covers the shipped YAML inputs and the Python writers that
        # generate case dicts.
        import glob as _glob

        repo = os.path.abspath(os.path.join(EXAMPLES_DIR, os.pardir,
                                            os.pardir))
        retired = ('execution_mode', 'workflow_execution',
                   'workflow_workers')

        offenders = []

        # YAML inputs: the keys are illegal inside a top-level
        # `environment:` block, legal anywhere else (a skill section
        # declaring its own execution_mode is the live source).
        for path in sorted(_glob.glob(os.path.join(repo, '**/examples/*.yaml'))):
            in_env = False
            with open(path, encoding='utf-8') as handle:
                for number, line in enumerate(handle, 1):
                    body = line.rstrip('\n')
                    if body and not body[0].isspace():
                        in_env = body.rstrip() == 'environment:'
                    if not in_env:
                        continue
                    key = body.strip().split(':', 1)[0].strip()
                    if key in retired:
                        offenders.append('%s:%d %s' % (
                            os.path.relpath(path, repo), number, key))

        # Python writers: workflow_execution/workflow_workers are gone
        # outright; execution_mode may not sit in an 'environment' dict.
        scanned = (_glob.glob(os.path.join(repo, 'qualification/benchmark/*.py'))
                   + _glob.glob(os.path.join(repo, '..', 'tools/*.py')))
        for path in sorted(set(scanned)):
            with open(path, encoding='utf-8') as handle:
                lines = handle.readlines()
            inside_env = False
            for number, line in enumerate(lines, 1):
                if "'environment': {" in line:
                    inside_env = True
                elif inside_env and line.strip().startswith('}'):
                    inside_env = False
                key = line.strip().split(':')[0].strip().strip("'\"")
                if key in ('workflow_execution', 'workflow_workers'):
                    offenders.append('%s:%d %s' % (
                        os.path.relpath(path, repo), number, key))
                elif key == 'execution_mode' and inside_env:
                    offenders.append('%s:%d execution_mode in environment'
                                     % (os.path.relpath(path, repo), number))

        self.assertFalse(offenders,
                         'retired dispatch keys are still written: %s'
                         % offenders)


class TestRetiredEnvironmentKeysRefused(unittest.TestCase):
    """The retired &environment keys are refused, not absorbed."""

    def _build(self, environment):
        from pyrai3md.schema.adapter import build_config
        raw = {
            'control': {'title': 'retired_key', 'jobtype': 'energy',
                        'theory': ['xtb']},
            'atoms': ['H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            'molecule': {'charges': 0, 'multiplicity': 1},
            'environment': dict(environment),
        }
        return build_config(raw)

    def test_each_retired_key_is_refused_with_a_pointer(self):
        for key in ('execution_mode', 'workflow_execution',
                    'workflow_workers'):
            with self.assertRaises(ValueError) as caught:
                self._build({key: 'local' if key != 'workflow_workers'
                             else 2})
            message = str(caught.exception)
            self.assertIn('`%s`' % key, message)
            # The refusal is only useful if it says where to write it.
            self.assertIn('->', message)

    def test_a_live_environment_key_still_loads(self):
        config = self._build({'shell': 'module load sentinel/toolchain'})
        self.assertEqual(config.environment.shell,
                         'module load sentinel/toolchain')

    def test_skill_section_execution_mode_is_not_refused(self):
        # The live source is the skill section; refusing it there would
        # break the skills that declare it.
        from pyrai3md.schema.adapter import build_config
        raw = {
            'control': {'title': 'skill_mode_key', 'jobtype': 'energy',
                        'theory': ['xtb']},
            'atoms': ['H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            'molecule': {'charges': 0, 'multiplicity': 1},
            'xtb': {'execution_mode': 'slurm'},
        }
        config = build_config(raw)
        self.assertEqual(config.control.title, 'skill_mode_key')


class TestShippedInitialEnsembles(unittest.TestCase):
    """Every shipped `<title>.init.xyz` must load as a real ensemble.

    The file is an input like any other, and the fan-out depends on it: a run
    with `initcond: read` takes its first `ninitcond` frames from it. Round 7
    measured the alternative this replaces — three trajectories whose energy
    records were byte-identical, because the default `initcond: 0` copies one
    state instead of distributing states. These gates load the shipped files
    through the runtime's own reader rather than trusting the text.
    """

    def _ensembles(self):
        import glob as _glob
        return sorted(_glob.glob(os.path.join(EXAMPLES_DIR, '*.init.xyz')))

    def test_a_titled_ensemble_is_shipped_for_every_read_case(self):
        import glob as _glob
        shipped = {os.path.basename(path)[:-len('.init.xyz')]
                   for path in self._ensembles()}
        readers = set()
        for path in sorted(_glob.glob(os.path.join(EXAMPLES_DIR, '*.yaml'))):
            with open(path, encoding='utf-8') as handle:
                text = handle.read()
            for line in text.splitlines():
                if line.strip().startswith('title:'):
                    title = line.split(':', 1)[1].strip()
            if 'initcond: read' in text:
                readers.add(title)
        self.assertTrue(readers,
                        'no shipped example uses initcond: read any more')
        self.assertEqual(readers - shipped, set(),
                         'initcond: read without a shipped ensemble file')

    def test_each_ensemble_loads_with_distinct_frames(self):
        import numpy as np

        from pyrai3md.workflow.initcond import read_init_xyz

        declared = self._declared_atom_counts()
        for path in self._ensembles():
            frames = read_init_xyz(path)
            self.assertGreaterEqual(len(frames), 3,
                                    '%s carries %d frames' % (path,
                                                              len(frames)))
            # The atom count is not a constant of this suite: `md_*` ships a
            # six-atom ensemble and `adaptive_c2h4water` ships the 1581-atom
            # box, so the count is checked against the example that titles
            # itself after the file. A per-frame shape invariant was tried
            # first and dropped as an identity: `read_init_xyz` refuses a row
            # count that disagrees with its own header ('Atom line needs
            # element + 6 columns', 'Malformed frame header'), so no file
            # reaches this test with mixed shapes. The declared count is not
            # an identity - measured with a stub example that names five atoms
            # against the six-atom shipped ensemble, which this branch fails.
            coordinates, velocities = frames[0]
            self.assertEqual(coordinates.shape[1], 3, path)
            stem = os.path.basename(path)[:-len('.init.xyz')]
            if stem in declared:
                self.assertEqual(
                    coordinates.shape[0], declared[stem],
                    '%s carries %d atoms while the example titled %s '
                    'declares %d' % (path, coordinates.shape[0], stem,
                                     declared[stem]))
            # Distinctness is the whole point of the file: identical frames
            # would reproduce the byte-for-byte triplicate this replaced.
            seen = {tuple(np.round(velocity.ravel(), 12))
                    for _, velocity in frames}
            self.assertEqual(len(seen), len(frames),
                             '%s repeats a frame' % path)
            positions = {tuple(np.round(coord.ravel(), 12))
                         for coord, _ in frames}
            self.assertGreater(len(positions), 1,
                               '%s never moves the nuclei' % path)

    def _declared_atom_counts(self):
        """Map each shipped example title to its declared atom count."""
        import glob as _glob

        import yaml as _yaml

        counts = {}
        for path in _glob.glob(os.path.join(EXAMPLES_DIR, '*.yaml')):
            try:
                spec = _yaml.safe_load(open(path, encoding='utf-8'))
            except _yaml.YAMLError:
                continue
            if not isinstance(spec, dict):
                continue
            title = (spec.get('control') or {}).get('title')
            atoms = spec.get('atoms')
            if title and isinstance(atoms, list):
                counts[str(title)] = len(atoms)
        return counts

    def test_the_reader_resolves_a_shipped_case_through_the_real_path(self):
        """`initcond: read` on a shipped case yields the first N frames."""
        import yaml as _yaml

        from pyrai3md.data.state import PyRAIState
        from pyrai3md.data.runtime.state import RuntimeState
        from pyrai3md.schema.config import PyRAIConfig
        from pyrai3md.workflow.initcond import resolve_initial_conditions

        source = os.path.join(EXAMPLES_DIR, 'md_parallel.yaml')
        spec = _yaml.safe_load(open(source, encoding='utf-8'))
        title = spec['control']['title']
        ensemble = os.path.join(EXAMPLES_DIR, '%s.init.xyz' % title)
        self.assertTrue(os.path.isfile(ensemble), ensemble)

        work = tempfile.mkdtemp(prefix='ensemble_gate_')
        cwd = os.getcwd()
        try:
            os.chdir(work)
            import shutil
            shutil.copy(ensemble, os.path.join(work,
                                               '%s.init.xyz' % title))
            config = PyRAIConfig()
            config.control.title = title
            config.control.jobtype = 'md'
            config.system.atoms = spec['atoms']
            config.system.coordinates = spec['coordinates']
            config.system.natom = len(spec['atoms'])
            config.dynamics.initcond = spec['md']['initcond']
            config.dynamics.ninitcond = spec['md']['ninitcond']
            state = PyRAIState(runtime=RuntimeState(total_steps=1))
            state.pmd = config
            batch = resolve_initial_conditions(state)
        finally:
            os.chdir(cwd)
            import shutil
            shutil.rmtree(work, ignore_errors=True)

        self.assertEqual(len(batch), int(spec['md']['ninitcond']))


class TestTheSelectedStateReachesEveryCarrier(unittest.TestCase):
    """An example that names a state must drive that state on both carriers.

    `examples/optimize_tda.yaml` asks for S1 and its two carriers ran two
    different physics in round `tmp_20261015_211219`: the YAML leg reported
    `-77.17527079585400` Ha with the backend's own header
    `TDA gradients for state 1`, the classic leg `-77.47568494347603` Ha with
    `RKS gradients` and `grid_response = False`, 20 steps against 12. Nothing
    failed, because the writer's list of optimization keywords carried the
    convergence thresholds and the box switch but not the state, so the text
    carrier silently optimized the ground state.
    """

    def setUp(self):
        import tempfile
        self.root = tempfile.mkdtemp(prefix='state_carrier_')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def _effective_state(raw):
        from pyrai3md.schema.adapter import build_config
        from pyrai3md.schema.normalize import normalize
        from pyrai3md.schema.validate import validate
        return validate(normalize(build_config(raw))).optimize.opt_state_i

    def _state_examples(self):
        import glob
        import yaml
        found = []
        for path in sorted(glob.glob(os.path.join(EXAMPLES_DIR, '*.yaml'))):
            with open(path) as handle:
                spec = yaml.safe_load(handle)
            if (spec or {}).get('optimize', {}).get('state_i'):
                found.append((path, spec))
        return found

    def _classic_raw(self, example):
        from pyrai3md.interface.parser import parse_input
        from pyrai3md.qualification.certification.converter import (
            convert, yaml_to_config,
        )
        stem = os.path.splitext(os.path.basename(example))[0]
        config_path = os.path.join(self.root, stem + '.config')
        input_path = os.path.join(self.root, stem + '.input')
        yaml_to_config(example, config_path)
        convert(config_path, input_path)
        return input_path, parse_input(input_path)

    def test_both_carriers_select_the_same_state(self):
        cases = self._state_examples()
        self.assertTrue(cases,
                        'no shipped example selects an excited state, so this '
                        'walk would pass on an empty set')
        for example, spec in cases:
            name = os.path.basename(example)
            from_yaml = self._effective_state(spec)
            _input_path, classic_raw = self._classic_raw(example)
            from_classic = self._effective_state(classic_raw)
            self.assertEqual(
                from_classic, from_yaml,
                msg='%s: the text carrier selects state %s while the YAML '
                    'carrier selects %s, so the case runs on two different '
                    'surfaces under one name'
                    % (name, from_classic, from_yaml))

    def test_the_written_line_is_what_carries_the_selection(self):
        # The negative control: take the classic input the generator wrote and
        # delete only that line. The read then lands on the schema default,
        # which is exactly the round's silent ground-state run - so the
        # assertion above is reading a line the file really depends on.
        example, spec = self._state_examples()[0]
        input_path, classic_raw = self._classic_raw(example)
        self.assertEqual(self._effective_state(classic_raw),
                         spec['optimize']['state_i'])
        with open(input_path) as handle:
            text = handle.read()
        self.assertIn('state_i', text)
        stripped = '\n'.join(line for line in text.splitlines()
                             if not line.strip().startswith('state_i'))
        variant = os.path.join(self.root, 'no_state.input')
        with open(variant, 'w') as handle:
            handle.write(stripped)
        sibling = os.path.splitext(input_path)[0] + '.xyz'
        if os.path.exists(sibling):
            import shutil
            shutil.copy2(sibling, os.path.join(self.root, 'no_state.xyz'))
        from pyrai3md.interface.parser import parse_input
        self.assertEqual(self._effective_state(parse_input(variant)), 0,
                         'a classic input with the state line removed still '
                         'selects an excited state, so the line is not what '
                         'carries it and the parity test proves nothing')


def _resolve_skill_environment(name):
    """Resolve one shipped example down to what its skill actually runs on.

    Two chains, because the two layers ask different questions and the
    readings differ: the framework's task queue (where a dropped skill
    section keyword would first go missing) and then the skill's own
    schema pipeline, which is the object
    `service/stress/num_stress.py::build_payloads`'s caller reads its
    tri-state dispatch keys from. Constructing the skill's config class
    directly is NOT this path and gives a different answer - measured:
    `SkillConfig(**flat)` leaves `environment` None, so a probe built
    that way reads the strain pool as serial while the run is asking for
    two workers.

    Parameters
    ----------
    name : str
        Example file name inside the shipped examples directory.

    Returns
    -------
    tuple
        (flat skill config, skill environment config, framework node).
    """
    from pyrai3md.interface.parser import parse_input
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.schema.normalize import normalize
    from pyrai3md.schema.validate import validate
    from pyrai3md.data.build import build_state
    from pyrai3md.skills.cp2k_skill.schema.adapter import (
        build_config as skill_build_config)
    from pyrai3md.skills.cp2k_skill.schema.normalize import (
        normalize as skill_normalize)

    previous = os.getcwd()
    workdir = tempfile.mkdtemp(prefix='example_skill_env_')
    try:
        os.chdir(workdir)
        raw = parse_input(os.path.join(EXAMPLES_DIR, name))
        state = build_state(validate(normalize(build_config(raw))),
                            raw_config=raw)
        flat = dict((state.theory[0].get('config') or {})
                    ['skill_config'])
        env = skill_normalize(skill_build_config(flat)).environment
    finally:
        os.chdir(previous)
        from shutil import rmtree
        rmtree(workdir, ignore_errors=True)
    return flat, env, _resolve_workflow_node(name)


class TestTheStrainedBoxExamplesAskTheSkillsOwnStrainPool(unittest.TestCase):
    """The two num-stress cp2k examples fan their twelve strain points out.

    Neither declared `multitask`/`multipool` until #393, and the skill's
    own tri-state read treats undeclared as serial, so the twelve CP2K
    calls that differentiate one box were being walked one after
    another. Round 18 is what that cost: `optimize_cell_cp2k`'s two
    carriers took 13 m 41 s and 13 m 37 s - 821 s, 68% of the shared
    1200 s backstop - and the round spent 42 minutes to reach row 104 of
    251.
    """

    NAMES = ('optimize_cell_cp2k.yaml', 'optimize_cell_gap_cp2k.yaml')

    def test_the_pool_pair_reaches_the_task_through_the_framework_chain(self):
        for name in self.NAMES:
            flat, _env, _node = _resolve_skill_environment(name)
            self.assertIs(flat.get('multitask'), True,
                          '%s: the pair left the framework queue before it '
                          'reached the skill (read %r)' % (name, flat.get(
                              'multitask')))
            self.assertEqual(flat.get('multipool'), 2, name)

    def test_the_skill_reads_two_workers_for_the_twelve_points(self):
        # The number `num_stress.py` decides its pool from, with the same
        # expression it uses: `workers` only exists when multitask is
        # declared true, and a declared pool of 2 is 2.
        for name in self.NAMES:
            _flat, env, _node = _resolve_skill_environment(name)
            self.assertIs(env.multitask, True, name)
            self.assertEqual(env.multipool, 2, name)
            self.assertEqual(env.cp2k_nproc, 4, name)
            workers = (max(1, int(env.multipool or 1)) if env.multitask
                       else 1)
            self.assertEqual(workers, 2,
                             '%s: the strain route would run %d worker(s), '
                             'not the 2 the example declares' % (name,
                                                                 workers))

    def test_the_framework_node_stays_serial_for_these_two(self):
        # A reading, not a design claim: the fan-out the ruling bought is
        # the skill's over its own perturbed points, and this layer says
        # nothing about it. It is pinned because a second fan-out here
        # would multiply against the one above (2 x 2 workers, each 4
        # ranks), and because the two layers disagreeing is the shape a
        # dropped key also produces - so the pair of readings is what
        # makes the first criterion above mean anything.
        for name in self.NAMES:
            _flat, _env, node = _resolve_skill_environment(name)
            self.assertEqual(node['name'], 'optimize', name)
            self.assertFalse(node['multitask'], name)
            self.assertEqual(node['multipool'], 1, name)


_TRIO = ('multitask', 'remotetask', 'multipool')


def _render_classic(name):
    """Render one shipped example's classic input the harness renders it."""
    from pyrai3md.qualification.certification.converter import (
        yaml_to_config, convert)
    workdir = tempfile.mkdtemp(prefix='classic_carrier_')
    config_path = os.path.join(workdir, name + '.json')
    input_path = os.path.join(workdir, name + '.input')
    yaml_to_config(os.path.join(EXAMPLES_DIR, name + '.yaml'), config_path)
    convert(config_path, input_path)
    return input_path, workdir


def _classic_sections(name):
    """The rendered classic input, split into {SECTION: [keyword, ...]}."""
    import re
    input_path, _workdir = _render_classic(name)
    with open(input_path, 'r', encoding='utf-8') as handle:
        text = handle.read()
    sections = {}
    current = None
    for line in text.splitlines():
        header = re.match(r'^&(\w+)\s*$', line)
        if header:
            current = header.group(1).upper()
            sections.setdefault(current, [])
            continue
        if current is not None and line.strip():
            sections[current].append(line.split()[0])
    return sections


def _stage_carriers(workdir, name):
    """Copy an example's own coordinate carriers beside a rendered input."""
    import glob as _glob
    import shutil
    for path in _glob.glob(os.path.join(EXAMPLES_DIR, name + '*')):
        if not path.endswith('.yaml'):
            shutil.copy(path, workdir)


def _state_from_classic(name, input_path, workdir):
    """Build the state one classic carrier produces, from its own directory."""
    from pyrai3md.interface.parser import parse_input
    from pyrai3md.schema.adapter import build_config
    from pyrai3md.schema.normalize import normalize
    from pyrai3md.schema.validate import validate
    from pyrai3md.data.build import build_state
    _stage_carriers(workdir, name)
    previous = os.getcwd()
    os.chdir(workdir)
    try:
        raw = parse_input(os.path.basename(input_path))
        return build_state(validate(normalize(build_config(raw))),
                           raw_config=raw)
    finally:
        os.chdir(previous)


def _node_from_classic(name, input_path, workdir):
    """Resolve the workflow graph node one classic carrier builds."""
    from pyrai3md.workflow import resolve_workflow, WORKFLOW_REGISTRY
    from pyrai3md.workflow.dispatch.planner import ExecutionGraphBuilder
    state = _state_from_classic(name, input_path, workdir)
    return ExecutionGraphBuilder(WORKFLOW_REGISTRY).build(
        state, resolve_workflow(state))[0]


def _cp2k_environment_from_classic(name, input_path, workdir):
    """The cp2k skill's own environment node as the classic carrier resolves it.

    The skill's schema pipeline is the object its strain route reads its
    tri-state pool from; the framework queue is only the middle hand-off, so
    both are checked here rather than one of them.
    """
    from pyrai3md.skills.cp2k_skill.schema.adapter import (
        build_config as skill_build_config)
    from pyrai3md.skills.cp2k_skill.schema.normalize import (
        normalize as skill_normalize)
    state = _state_from_classic(name, input_path, workdir)
    flat = dict((state.theory[0].get('config') or {})['skill_config'])
    return skill_normalize(skill_build_config(flat)).environment


class TestTheDispatchPoolReachesTheClassicCarrier(unittest.TestCase):
    """`multitask` / `multipool` / `remotetask` are stated in the `.input`.

    Nothing wrote them, so a declared pool existed in the yaml, survived into
    the agent config, and was absent from the classic carrier - which left the
    `.input` leg of one example walking its own fan-out one point at a time.
    Round `tmp_20261019_163711` prices it in the test environment: `optimize_cell_cp2k`
    took 589 s in yaml mode against its own 821 s serial reading from round 18,
    while the `.input` mode of the same file took 815 s against 817 s - no
    saving, because its carrier stated no pool. The same class measured locally
    on the framework side: `md_parallel.yaml` gives the workflow node
    `multitask=True multipool=3` from yaml and `multitask=False multipool=1`
    from its own generated classic input. The readers were never the gap:
    inserting the two lines by hand into `&MD` returns `True/3` and into
    `&CP2K` returns two workers, so this is a writer-side omission.
    """

    def test_the_md_carrier_states_its_own_pool(self):
        expectations = {'md_parallel': (True, 3, False),
                        'md_batch': (True, 3, True)}
        for name, expected in sorted(expectations.items()):
            keys = _classic_sections(name).get('MD', [])
            self.assertIn('multitask', keys, name)
            self.assertIn('multipool', keys, name)
            self.assertEqual('remotetask' in keys, expected[2], name)
            # The line is not what carries the decision until the node built
            # from the text agrees with the yaml mode - a keyword the reader
            # ignores would satisfy the three assertions above.
            input_path, workdir = _render_classic(name)
            node = _node_from_classic(name, input_path, workdir)
            read = (bool(node['multitask']), node['multipool'],
                    bool(node['remotetask']))
            self.assertEqual(
                read, expected,
                '%s: the classic carrier hands the workflow node %r where the '
                'yaml mode hands it %r' % (name, read, expected))

    def test_the_strained_box_carrier_states_the_skills_pool(self):
        # This pool belongs to the SKILL (its 'num' strain route fans the
        # twelve perturbed points out), so it renders in `&CP2K`, and the
        # framework node stays serial in both modes - the input asks for one
        # fan-out, not two.
        for name in ('optimize_cell_cp2k', 'optimize_cell_gap_cp2k'):
            sections = _classic_sections(name)
            self.assertIn('multitask', sections.get('CP2K', []), name)
            self.assertIn('multipool', sections.get('CP2K', []), name)
            self.assertNotIn('multitask', sections.get('MD', []),
                             '%s: the skill pool leaked into &MD, where the '
                             'framework would open a second fan-out' % name)
            _flat, yaml_env, yaml_node = _resolve_skill_environment(
                name + '.yaml')
            input_path, workdir = _render_classic(name)
            classic_env = _cp2k_environment_from_classic(
                name, input_path, workdir)
            self.assertEqual(
                (yaml_env.multitask, yaml_env.multipool),
                (classic_env.multitask, classic_env.multipool),
                '%s: the two carriers resolve the strain pool differently'
                % name)
            self.assertEqual(classic_env.multipool, 2, name)
            self.assertFalse(yaml_node['multitask'], name)
            self.assertFalse(
                _node_from_classic(name, input_path, workdir)['multitask'],
                '%s: the framework node opened a pool the input never asked '
                'for' % name)

    def test_the_two_owners_of_a_pool_keep_their_own_section(self):
        # One flat namespace, two legitimate claimants: `&md` opens the
        # trajectory fan-out, a skill opens its own perturbed-point pool, and
        # both spell the keyword `multipool`. So a skill's trio stays out of
        # the flat top level and lives in its block, and each classic section
        # states the number its own input declared. Carried here as a synthetic
        # config because no shipped example asks for a dynamics pool AND a skill
        # pool at once - which is exactly why the rule needs its own carrier:
        # measured with the skill trio flattened to the top level, the `&MD`
        # below reads `multipool 2`, which is cp2k's number, and the framework
        # would fan out two workers its input never asked for.
        import json
        from pyrai3md.qualification.certification.converter import convert
        config = {
            'title': 'synth_pool_carrier', 'jobtype': 'md',
            'theory': ['cp2k'], 'skill_name': 'cp2k_skill',
            'atoms': ['O', 'H', 'H'],
            'coordinates': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]],
            'charges': 0, 'multiplicity': 1, 'nstate': 1,
            'multitask': True, 'multipool': 3,
            'ensemble': 'nvt', 'thermostat': 'berendsen', 'temperature': 300,
            'skill_sections': {'cp2k': {'method': 'dft',
                                        'profile': 'dft_gradient',
                                        'multitask': True, 'multipool': 2}},
        }
        workdir = tempfile.mkdtemp(prefix='pool_carrier_')
        config_path = os.path.join(workdir, 'synth_pool_carrier.json')
        input_path = os.path.join(workdir, 'synth_pool_carrier.input')
        with open(config_path, 'w', encoding='utf-8') as handle:
            json.dump(config, handle)
        text = convert(config_path, input_path)

        def stated(header):
            body = text.split('&' + header)[1].split('&')[0]
            found = {}
            for line in body.splitlines():
                parts = line.split()
                if parts and parts[0] in _TRIO:
                    found[parts[0]] = parts[1] if len(parts) > 1 else ''
            return found

        self.assertEqual(stated('MD'), {'multitask': 'true', 'multipool': '3'},
                         'the dynamics pool did not reach &MD as declared')
        self.assertEqual(stated('CP2K'),
                         {'multitask': 'true', 'multipool': '2'},
                         'the skill pool did not reach &CP2K as declared')

        # The render above is the destination; the route is the other half of
        # the rule. Read the same pair of owners through `yaml_to_config`, which
        # is where a skill's keywords are copied to the top level: the trio must
        # arrive there from `&md` only, or every reader of the flat namespace -
        # `&MD`'s echo among them - sees the engine's number.
        yaml_text = (
            'control:\n'
            '  title: pool_two_owners\n'
            '  jobtype: md\n'
            '  theory: [cp2k]\n'
            'atoms: [O, H, H]\n'
            'coordinates:\n'
            '  - [0.0, 0.0, 0.0]\n'
            '  - [0.0, 0.0, 1.0]\n'
            '  - [0.0, 1.0, 0.0]\n'
            'molecule:\n'
            '  charges: 0\n'
            '  multiplicity: 1\n'
            'md:\n'
            '  ensemble: nvt\n'
            '  temperature: 300\n'
            '  multitask: true\n'
            '  multipool: 3\n'
            'cp2k:\n'
            '  method: dft\n'
            '  profile: dft_gradient\n'
            '  multitask: true\n'
            '  multipool: 2\n'
        )
        carrier_dir = tempfile.mkdtemp(prefix='pool_yaml_carrier_')
        carrier_path = os.path.join(carrier_dir, 'pool_two_owners.yaml')
        with open(carrier_path, 'w', encoding='utf-8') as handle:
            handle.write(yaml_text)
        from pyrai3md.qualification.certification.converter import (
            yaml_to_config)
        carrier_json = os.path.join(carrier_dir, 'pool_two_owners.json')
        flattened = yaml_to_config(carrier_path, carrier_json)
        self.assertEqual(flattened.get('multipool'), 3,
                         'the flat namespace does not carry the dynamics pool')
        self.assertEqual(flattened['skill_sections']['cp2k'].get('multipool'), 2,
                         'the skill block lost its own pool on the way in')
        rendered = os.path.join(carrier_dir, 'pool_two_owners.input')
        text2 = convert(carrier_json, rendered)
        body = text2.split('&MD')[1].split('&')[0]
        self.assertIn('multipool     3', body,
                      'the &MD carrier states the engine\'s pool, not the '
                      'dynamics one it was given')

    def test_a_pool_line_appears_only_where_a_section_declares_one(self):
        # Placement, measured over the shipped set: the pool a `&md` declares
        # renders in `&MD`, the pool a skill section declares renders in that
        # skill's block, and nowhere else. The first cut of this fix filled a
        # skill block from the shared flat namespace to recover the optimizer's
        # cell thresholds, and that put `ensemble`, `thermostat`, `barostat`,
        # `temperature`, `pressure` and `debug` - &MD's keywords - inside &XTB
        # and &PYSCF of the multicenter carriers; the fill was dropped because
        # a block must not be fed by a namespace whose trio has another owner.
        from pyrai3md.interface.parser import parse_input
        stating = []
        for path in sorted(os.listdir(EXAMPLES_DIR)):
            if not path.endswith('.yaml'):
                continue
            name = path[:-len('.yaml')]
            raw = parse_input(os.path.join(EXAMPLES_DIR, path))
            declared = {}
            for section, body in raw.items():
                if isinstance(body, dict):
                    for key in _TRIO:
                        if body.get(key) is not None:
                            declared.setdefault(section.upper(), set()).add(key)
            for header, keys in _classic_sections(name).items():
                rendered = {key for key in keys if key in _TRIO}
                if not rendered:
                    continue
                self.assertTrue(
                    rendered <= declared.get(header, set()),
                    '%s: &%s states %s, which its own input section never '
                    'declared (declared there: %s)'
                    % (name, header, sorted(rendered),
                       sorted(declared.get(header, set()))))
                stating.append((name, header))
        # Measured 2026-10-19 over all 96 shipped examples: eight sections
        # state a pool. Four already did, because `&search` / `&train` /
        # `&prediction` are carried nested and their renderer echoes the trio
        # (nn_grid_search, nn_grid_search_local, nn_train, nn_predict); the
        # other four are this fix - the two `&MD` carriers, which flatten their
        # keywords to the top level where nothing echoed them, and the two
        # `&CP2K` strain legs, whose pool belongs to the skill and reached
        # neither table. Before the fix those last four stated nothing and ran
        # their fan-out serially.
        #
        # Measured 2026-11-06 over all 201 shipped examples: eleven. The three
        # new rows are the `adaptive_*_slurm` carriers, whose `&adaptive`
        # declares the generic `multitask`/`remotetask` next to the per-stage
        # `md_*`/`reference_*` flags, so the classic carrier states the trio
        # and the renderer echoes it. They are the same pair `AdaptiveSampling.
        # _stage_flags` reads (`_pick` falls back to the generic keyword when a
        # stage names none), so the `.input` leg of an adaptive run dispatches
        # what its YAML leg did - which is what round `tmp_20261106_102745`
        # read as three slurm cases in both carriers with equal structural
        # counts. The three `_local` carriers are absent because they declare
        # no generic key at all.
        self.assertEqual(sorted(stating), sorted([
            ('adaptive_oniom2_slurm', 'ADAPTIVE'),
            ('adaptive_qmmm_slurm', 'ADAPTIVE'),
            ('adaptive_single_slurm', 'ADAPTIVE'),
            ('md_batch', 'MD'), ('md_parallel', 'MD'),
            ('nn_grid_search', 'SEARCH'), ('nn_grid_search_local', 'SEARCH'),
            ('nn_predict', 'PREDICTION'), ('nn_train', 'TRAIN'),
            ('optimize_cell_cp2k', 'CP2K'),
            ('optimize_cell_gap_cp2k', 'CP2K')]), stating)


if __name__ == '__main__':
    unittest.main()
