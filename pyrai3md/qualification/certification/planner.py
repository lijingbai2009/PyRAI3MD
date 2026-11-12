"""
======================================================================
This module plans the skill certification tests (split out of the
qualification manager).

For every skill the switches certify it plans, in this order:
1. The runtime test scripts (test_*.py from the skill tests/ dir,
   executed through pytest) so skill-internal regressions surface
   before any end-to-end mode runs;
2. The three-mode YAML task tests:
    - skill_agent:     pyrai3md_skill <config>  (always local)
    - pyrai3md_agent:  pyrai3md_agent <config>  (always local)
    - pyrai3md_input:  pyrai3md <input>         (always local)

A slurm round wraps each skill case group in one job; the legs run
locally inside that job. The skill's own `execution_mode: slurm`
still submits its own batch job from inside the leg -- that is the
skill feature under test, not this module's scheduling decision.

Two switch levels decide a skill, above the top-level `_runall` master:
`skills.<name>.test` (three-valued: true certifies every case of the skill,
false certifies none, `condition` leaves it to the per-case switches) and the
per-case booleans `skills.<name>.tests.<case>` (keyed by the test YAML's file
stem) plus `skills.<name>.runtime_tests` for the pytest scripts. A case the
switches close is still listed in the plan, as a skipped task naming the
switch that closed it.

The planning chain is fully dynamic: skills.query discovers the
registry, the handler locates each skill's tests directory, the
converter parses the YAML cases into flat configs (no per-skill
dev_tool) and renders the .input/.config outputs.

The qualification manager aggregates the returned task list into
the unified test plan and the runner executes it unchanged; this
module only owns the skill-side planning decisions (registry
validation, test file discovery, environment injection, driver
forcing, slurm policy).

Reference direction: planner -> qualification.manager (shared script
writers), planner -> certification.handler (test file discovery),
planner -> certification.converter (test parsing and input
conversion), planner -> skills.query (skill registry).

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import importlib.util
import json
import os
import shutil
import warnings

from ..manager import (QUALIFICATION_ONLY_KEYS, _coerce_flag,
                       _coerce_switch, _get_shell_commands, _merge_exports,
                       _resolve_chain, _resolve_timeout, _runall_mode,
                       SWITCH_CONDITION, _write_shell_script)
from ...skills.query import discover_all, list_skills, skill_module_path
from . import handler
from .converter import convert as convert_to_classic_input
from .converter import convert_to_pyrai_config, generate_test_configs


def _pytest_available():
    """Check whether pytest is importable at plan time.

    Returns
    -------
    bool
        True when the runtime test scripts can be executed.
    """
    try:
        importlib.import_module('pytest')
        return True
    except ImportError:
        return False


def _scheduler_is_slurm(config):
    """Whether this round is executed as grouped slurm jobs."""
    from ..jobs import scheduler_mode
    return scheduler_mode(config) == 'slurm'


def _skipped_case_tasks(skill_name, closer):
    """One skipped task per test YAML of a skill the switches closed.

    A closed case still appears in the plan and in results.json, because a
    suite that quietly runs 3 of 16 certified cases cannot be told apart from
    one that certified 3. The entries carry the discovered YAML as their
    anchor and no script: nothing is generated for a case that will not run,
    and `skip_reason` names the switch to edit to open it.

    Parameters
    ----------
    skill_name : str
        Skill whose cases are listed.
    closer : str
        Switch key that closed them, as the operator wrote it.

    Returns
    -------
    list of dict
        Skipped task descriptors, one per test YAML.
    """
    tasks = []
    for yaml_path in handler.list_skill_test_yamls(skill_name):
        case = os.path.splitext(os.path.basename(yaml_path))[0]
        tasks.append({
            'name': f'{skill_name}_{case}',
            'skill': skill_name,
            'mode': 'skill_case',
            'input_file': yaml_path,
            'skip': True,
            'skip_reason': f'disabled by {closer}',
        })
    return tasks


def _skipped_runtime_tasks(skill_name, closer):
    """One skipped task per runtime `test_*.py` the switches closed.

    Same listing rule as :func:`_skipped_case_tasks`: closing a switch hides
    nothing from the report.
    """
    tasks = []
    for test_path in handler.list_skill_runtime_tests(skill_name):
        stem = os.path.splitext(os.path.basename(test_path))[0]
        tasks.append({
            'name': f'{skill_name}_{stem}',
            'skill': skill_name,
            'mode': 'skill_runtime',
            'input_file': test_path,
            'skip': True,
            'skip_reason': f'disabled by {closer}',
        })
    return tasks


def _skill_case_switch(skill_name, case_flags, profile_name):
    """Whether one test case of a skill is switched on.

    A case the `tests:` map does not mention keeps running: the map is how an
    operator closes a single case, not an allow-list that defaults to nothing.
    """
    closer = f'skills.{skill_name}.tests.{profile_name}'
    return _coerce_flag(case_flags.get(profile_name, True), closer, True,
                        'case'), closer


def stage_skill_fixtures(skill_name, run_dir):
    """Lay the skill's shipped data fixtures into its case run directory.

    A skill case names its input by file name (`train_data:
    ./egs_small.json`), and the case runs with the *run* directory as its
    working directory, not the installed package. So the data a skill ships
    beside its cases has to be where the case can resolve it. The example
    flow already stages the topology files a backend resolves from the
    launch directory; this is the same policy for the skill-case flow, and a
    skill that ships no data directory is a no-op.

    Parameters
    ----------
    skill_name : str
        Skill package name (e.g. 'nn_pes_skill').
    run_dir : str
        The skill's case run directory inside the test work directory.

    Returns
    -------
    list of str
        Names of the staged files, for the caller's reporting.
    """
    try:
        skill_pkg = importlib.import_module(
            skill_module_path(skill_name, ''))
    except Exception:
        return []
    data_dir = os.path.join(os.path.dirname(skill_pkg.__file__),
                            'tests', 'data')
    if not os.path.isdir(data_dir):
        return []
    staged = []
    for fname in sorted(os.listdir(data_dir)):
        src = os.path.join(data_dir, fname)
        if not os.path.isfile(src):
            continue
        shutil.copy(src, os.path.join(run_dir, fname))
        staged.append(fname)
    return staged


def _write_leg_config(config_path, leg_dir, profile_name):
    """Copy one generated config into a leg with a leg-private scratch.

    The three carrier legs of a case must not share the skill's
    calculation directory: a later leg would reuse the orbitals the
    earlier one converged, so the same input starts from a different
    guess and the legs are no longer comparable (a state-averaged
    CASSCF can even land on a different solution). The config the leg
    runs therefore points `scratch_path` at its own directory.

    Parameters
    ----------
    config_path : str
        Shared flat `.json` config written by `generate_test_configs`.
    leg_dir : str
        This leg's working directory.
    profile_name : str
        Case stem, also the config file name.

    Returns
    -------
    str
        Path to the leg-private `.json` config.
    """
    with open(config_path, 'r', encoding='utf-8') as fh:
        config = json.load(fh)
    config['scratch_path'] = os.path.join(leg_dir, 'tmp')
    os.makedirs(config['scratch_path'], exist_ok=True)
    path = os.path.join(leg_dir, f'{profile_name}.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    return path


def _case_run_title(config_path, profile_name):
    """The run title one generated skill-case config declares for itself.

    `certification/converter.py` writes the case YAML's `name` into the flat
    config as `title` (qualification project/title naming, #159), and the
    framework run log is named after that title rather than the file stem.
    An unreadable or untitled config falls back to the stem, which is what
    the log check used before this existed.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as handler:
            title = json.load(handler).get('title')
    except (OSError, ValueError):
        return profile_name
    return str(title or profile_name)


# The skills layer's own test directory is not a skill, so it gets a row label
# where a skill name would otherwise stand. No skill is called 'skills', so no
# row of this group can collide with a skill's own.
SHARED_SKILL_TESTS_LABEL = 'skills'


def _runtime_row(skill_name, test_path, skill_dir, config):
    """Plan one runtime `test_*.py` as a pytest row.

    The command cd's into the file's own directory so package-relative
    fixtures resolve, and turns pytest's cache provider off so the round
    writes nothing into the installed package. The launch script goes to the
    caller's `skill_dir`, which is where a round keeps one directory per group.
    """
    test_stem = os.path.splitext(os.path.basename(test_path))[0]
    tests_dir = os.path.dirname(test_path)
    runtime_cmd = (f'cd {tests_dir} && python3 -m pytest '
                   f'{os.path.basename(test_path)} -q '
                   f'-p no:cacheprovider')
    sh_runtime = _write_shell_script(
        skill_dir, f'{skill_name}_{test_stem}',
        runtime_cmd, _merge_exports(config), _get_shell_commands(config),
        timeout=_resolve_timeout(config),
    )
    return {
        'name': f'{skill_name}_{test_stem}',
        'skill': skill_name,
        'mode': 'skill_runtime',
        'input_file': test_path,
        'script': sh_runtime,
        'use_slurm': False,
        # One job per skill's pytest phase; the shared layer's own label
        # ('skills') makes its rows a group of their own.
        'job_group': f'{skill_name}__runtime',
    }


def _skipped_shared_runtime_tasks(tests, closer):
    """One skipped row per shared runtime file the switches closed."""
    rows = []
    for path in tests:
        stem = os.path.splitext(os.path.basename(path))[0]
        rows.append({
            'name': f'{SHARED_SKILL_TESTS_LABEL}_{stem}',
            'skill': SHARED_SKILL_TESTS_LABEL,
            'mode': 'skill_runtime',
            'input_file': path,
            'skip': True,
            'skip_reason': f'disabled by {closer}',
        })
    return rows


def _shared_runtime_tasks(config, work_dir, have_pytest, runtime_open=True,
                          closer='skills.*.runtime_tests'):
    """Rows for the test scripts of the skills layer itself.

    pyrai3md/skills/tests/ holds the checks that span every skill -- the agent
    `calculate()` forwarding contract and the wrapper's config parsing. They
    belong to no skill package, so no skill's directory discovery reached them
    and an engine round planned none of them, while the same rule #359 states
    for a skill's own pytest files applies: nothing here needs an engine.

    The group follows the skills' pytest switches: it runs when at least one
    skill's `runtime_tests` phase is open (or the master/`test` switch forces
    it), and it is listed as skipped when every skill's phase is closed. It
    has no switch of its own; the group is the shared half of the same phase.

    They arrive as `skill_runtime` rows and not through the main-test phase for
    a reason worth keeping: that phase discovers with
    `unittest.TestLoader().discover`, a pytest-style module gives it zero cases,
    and zero cases printed as a pass is the fake green this whole file exists
    to prevent.
    """
    tests = handler.list_shared_skill_tests()
    if not tests:
        return []
    if not runtime_open:
        return _skipped_shared_runtime_tasks(tests, closer)
    if not have_pytest:
        return [_make_diagnostic_task(
            SHARED_SKILL_TESTS_LABEL,
            f'{len(tests)} runtime test scripts found '
            f'but pytest is not installed',
        )]
    skill_dir = os.path.join(work_dir, SHARED_SKILL_TESTS_LABEL)
    os.makedirs(skill_dir, exist_ok=True)
    return [_runtime_row(SHARED_SKILL_TESTS_LABEL, path, skill_dir, config)
            for path in tests]


def _runtime_switch(skill_name, skill_env, force_all):
    """Whether one skill's pytest phase is open, and the switch that decided."""
    if force_all:
        return True, '_runall'
    closer = f'skills.{skill_name}.runtime_tests'
    return _coerce_flag(
        skill_env.get('runtime_tests', True), closer, True, 'case'), closer


def _runtime_tasks(skill_name, skill_env, work_dir, config, force_all,
                   have_pytest):
    """The skill's own `tests/test_*.py` phase, as planned rows.

    Extracted because two different switch states reach it. A skill whose
    certification is open runs it ahead of its YAML cases, as before; a skill
    whose certification is closed runs it too, because these files are
    engine-free code checks and `skills.<name>.test` was never about them -
    `skills.<name>.runtime_tests` is. Before #359 the closed branch returned
    early, so an optimize round with the ten skills switched off executed none
    of the 62 pytest files the package ships, and "the framework's code is only
    tested on my laptop" stayed true against the rule that a local run is just
    a stand-in for an unreachable server.

    Parameters
    ----------
    skill_name : str
        Skill directory name ('bagel_skill').
    skill_env : dict
        The skill's pmd_env.yaml section.
    work_dir : str
        Round working directory; the script goes into <work_dir>/<skill>/.
    config : dict
        The loaded configuration (environment shell, exports, _timeout).
    force_all : bool
        True when the master or `skills.<name>.test` opened everything.
    have_pytest : bool
        Whether pytest is importable on this host.

    Returns
    -------
    list of dict
        Planned rows, or skipped rows naming `skills.<name>.runtime_tests`.
    """
    skill_dir = os.path.join(work_dir, skill_name)
    os.makedirs(skill_dir, exist_ok=True)
    tasks = []

    # `runtime_tests` is the switch of this whole phase; a master or
    # skill-level `true` overrides it, because that setting promises every
    # case of the skill runs.
    runtime_on, runtime_closer = _runtime_switch(
        skill_name, skill_env, force_all)
    runtime_tests = (handler.list_skill_runtime_tests(skill_name)
                     if runtime_on else [])
    if not runtime_on:
        tasks.extend(_skipped_runtime_tasks(skill_name, runtime_closer))
    if runtime_tests and not have_pytest:
        tasks.append(_make_diagnostic_task(
            skill_name,
            f'{len(runtime_tests)} runtime test scripts found '
            f'but pytest is not installed',
        ))
    for test_path in runtime_tests:
        if not have_pytest:
            break
        tasks.append(_runtime_row(skill_name, test_path, skill_dir, config))
    return tasks


def _stage_leg_directory(skill_dir, group_dir, mode, exclude_names=()):
    """Create (or reuse) one carrier leg's working directory.

    A leg starts where the run used to start: everything the converter
    staged in the skill directory -- the data files a skill ships beside
    its cases, and every case's project-named topology templates, which a
    backend resolves from the *launch directory* by name -- is copied in,
    so moving the working directory moves no lookup and adds no second
    resolution rule. What is *not* copied is another case's generated
    config: the per-case loop writes this case's own copy into its leg,
    and copying every case's artifacts into every group would grow the
    round quadratically. Launchers stay out too: a `.sh` / `.sbatch`
    belongs to the parent that drives the leg, not to the run inside it.

    Parameters
    ----------
    skill_dir : str
        The skill's own qualification directory (fixtures live here).
    group_dir : str
        The directory of this case group (`<skill>/<gid>`).
    mode : str
        Carrier mode ('skill_agent', 'pyrai3md_agent', 'pyrai3md_input').
    exclude_names : iterable of str
        Generated file names of every case; `{case}.json/.config/.input`.

    Returns
    -------
    str
        The leg's directory.
    """
    path = os.path.join(group_dir, mode)
    os.makedirs(path, exist_ok=True)
    excluded = set(exclude_names)
    for fname in sorted(os.listdir(skill_dir)):
        if fname.endswith(('.sh', '.sbatch')) or fname in excluded:
            continue
        src = os.path.join(skill_dir, fname)
        if not os.path.isfile(src):
            continue
        # A file the leg already owns wins: the per-case loop rewrites the
        # leg-private config (and its scratch path) after this copy.
        dst = os.path.join(path, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    return path


def _case_groups(test_configs):
    """Partition a skill's cases into independent directory groups.

    A case that declares a `modeldir` reads or writes that directory,
    so cases naming the same one are a chain (the training case writes
    what the serving cases read) and must share one directory per
    carrier leg. Every other case is independent and gets a directory
    of its own. The key is the declared path, normalized, so
    `./nn_eg` and `nn_eg` are one group; `./tmp` is the scratch every
    case declares and `_write_leg_config` replaces, so it never groups.

    Parameters
    ----------
    test_configs : dict
        Mapping {case stem: generated config path}.

    Returns
    -------
    tuple of (list, dict)
        `[(gid, [case, ...]), ...]` in plan order, and
        `{case: gid}`.
    """
    groups = {}
    for profile_name, config_path in test_configs.items():
        try:
            with open(config_path, 'r', encoding='utf-8') as handle:
                modeldir = json.load(handle).get('modeldir')
        except (OSError, ValueError):
            modeldir = None
        key = None
        if isinstance(modeldir, str) and modeldir.strip():
            normalized = os.path.normpath(modeldir.strip())
            if normalized not in ('.', 'tmp'):
                key = 'model:%s' % normalized
        if key is None:
            key = 'case:%s' % profile_name
        groups.setdefault(key, []).append(profile_name)

    grouped = []
    by_case = {}
    for ordinal, (key, members) in enumerate(groups.items()):
        slug = (os.path.basename(key.split(':', 1)[1])
                if key.startswith('model:') else members[0])
        slug = ''.join(
            char if char.isalnum() or char in '_.-' else '_'
            for char in slug)[:40] or 'case'
        gid = 'g%02d_%s' % (ordinal, slug)
        grouped.append((gid, members))
        for profile_name in members:
            by_case[profile_name] = gid
    return grouped, by_case


def plan_skill_tests(config, work_dir):
    """Plan skill tests based on registered skills.

    For each skill in the configuration:
    1. Locate the skill's tests directory through the handler
    2. Create a skill subfolder under work_dir
    3. Call converter.generate_test_configs to write the flat .json
       configs (the unified replacement of the former dev_tool)
    4. Create test tasks for each generated config

    Environment validation (e.g. molcas_path, bagel_path) is the
    responsibility of each skill's backend __init__, not the planner.
    This ensures runtime errors are captured as skill failures rather
    than being silently skipped.

    If the generator cannot produce test configs (e.g. missing YAML
    files, import failure), a diagnostic skip task is created so the
    user sees a clear message in the test report.

    File naming: flat .json configs match the test YAML names
    (e.g. casscf_energy.yaml -> casscf_energy.json).
    The certification/converter.py module later converts each .json into
    the PyRAI2MD classic text input (.input) and the pyrai3md
    agent config (.config) from one shared sectioned dict.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.
    work_dir : str
        Working directory for generated files.

    Returns
    -------
    list of dict
        List of skill test task descriptors.
    """
    # Discover skill registry via query modules (sole source of truth)
    skill_registry = discover_all()

    skills = config.get('skills', {})
    # Support both list (old) and dict (new) format
    if isinstance(skills, list):
        skills = {name: {} for name in skills}
    tasks = []
    have_pytest = _pytest_available()

    # Top-level `_runall` master, shared with the example and benchmark
    # planners: true certifies every listed skill whatever its own switches
    # say, false closes them all (each case still listed as skipped), and
    # 'condition' - the setting when the key is absent - leaves
    # skills.<name>.test to answer. Skills absent from pmd_env.yaml are still
    # never enumerated: the config section is the sole source.
    master = _runall_mode(config)

    # Per-test execution timeout (pmd_env.yaml _timeout): the runner's
    # subprocess backstop uses it directly, and each generated .sh
    # embeds `timeout (_timeout - margin)s` to kill the engine group
    # first so a timed-out test leaves no residual processes.
    test_timeout = _resolve_timeout(config)

    # The registry is read once: it is the same list for every skill in
    # the loop, and listing it per entry made the plan cost quadratic in
    # the number of skills.
    known_dirs = list_skills()
    # The shared pytest group follows the skills' pytest phase: open when at
    # least one skill's phase is open, closed with the switch that closed the
    # first one otherwise. It has no switch of its own.
    shared_runtime_open = False
    # The shared group closes when every skill's pytest phase is closed; the
    # reason names the family, not whichever skill happened to be last.
    shared_runtime_closer = 'skills.*.runtime_tests'

    for skill_name, skill_env in skills.items():
        # Validate skill exists in registry via query. This comes before the
        # switch gate on purpose: a name that matches no skill is an editing
        # error rather than a closed test, and it stays a diagnostic even when
        # the skill it names is switched off.
        if skill_name not in known_dirs:
            tasks.append(_make_error_task(
                skill_name,
                f"skill '{skill_name}' not found in skill registry. "
                f"Available: {sorted(known_dirs)}",
            ))
            continue

        # Two levels below the master decide this skill: its own `test`
        # switch, then the per-case `tests:` map and `runtime_tests`. A
        # `true` from the master or from `test` certifies every case and the
        # per-case switches are not read at all; `false` closes the YAML cases
        # and each is still listed, as a skipped task naming the switch.
        # `false` does NOT close the pytest phase: those files test code that
        # no engine is needed to exercise, and #359 made
        # `skills.<name>.runtime_tests` the only switch that speaks for them.
        state, closer = _resolve_chain([
            (master, '_runall'),
            (_coerce_switch(skill_env.get('test', False),
                            f'skills.{skill_name}.test', False),
             f'skills.{skill_name}.test'),
        ])
        force_all = state is True
        runtime_on, _runtime_closer = _runtime_switch(
            skill_name, skill_env, force_all)
        if runtime_on:
            shared_runtime_open = True
        if state is False:
            tasks.extend(_skipped_case_tasks(skill_name, closer))
            tasks.extend(_runtime_tasks(skill_name, skill_env, work_dir,
                                         config, False, have_pytest))
            continue

        case_flags = skill_env.get('tests') or {}
        if not isinstance(case_flags, dict):
            warnings.warn(
                f"skills.{skill_name}.tests is a mapping of test case names to "
                f"true/false, got {case_flags!r}; the per-case switches are "
                f"ignored and every case follows skills.{skill_name}.test."
            )
            case_flags = {}

        # Skill self-submission switch: skills.<name>.execution_mode
        # = 'slurm' makes the skill backend submit its own sbatch job
        # (sbatch -W) while qualification drives the test. Combined
        # with environment.scheduler this spans the scheduler x
        # execution_mode test matrix. The gate checks sbatch at plan
        # time so the missing binary surfaces as a diagnostic skip
        # instead of a runtime failure.
        gate, skill_slurm = _host_gate(skill_registry, skill_name, skill_env)
        if gate:
            tasks.append(_make_diagnostic_task(skill_name, gate))
            continue

        # Create skill subfolder: work_dir/<skill_name>/
        skill_dir = os.path.join(work_dir, skill_name)
        os.makedirs(skill_dir, exist_ok=True)

        # Data files the skill ships beside its cases go into the skill
        # directory as well as into each leg: a case names a file relative to
        # its working directory (`train_data: ./egs_small.json`, resolved in
        # the leg), while the classic reader resolves a same-basename `.xyz`
        # beside the `.input` it was handed (which stays in the skill
        # directory). Both lookups are live, so both directories carry them.
        stage_skill_fixtures(skill_name, skill_dir)

        # One working directory per carrier leg: work_dir/<skill>/<mode>/.
        # The three legs of one case run the same input through three entry
        # points, and a case chain is a chain *within* a leg (`00` trains the
        # ensemble folder `01` serves). With all three legs sharing one
        # directory, the second leg's training request meets the folder the
        # first leg left there, and the skill refuses to overwrite a trained
        # surface -- the rule #319 added on purpose. So the collision, not the
        # calculation, decided the round: six `*_train_agent`/`*_train_input`
        # cases of nn_pes_skill failed in the test environment for that reason
        # (round tmp_20261010_133424). The generated configs and the launch
        # scripts stay in the skill directory, shared by the legs; only what a
        # run *writes* is per leg, the same split the examples use
        # (`examples/<case>/<mode>/`).

        shell_commands = _get_shell_commands(config)
        export_vars = _merge_exports(config)

        # === Phase 1: runtime test scripts (priority) ===
        # The test_*.py files of the skill tests directory run ahead
        # of the YAML task tests; cd into the tests directory so
        # package-relative fixtures resolve, and suppress the pytest
        # cache to keep the installed package clean.
        # `runtime_tests` is the switch of this whole phase; a master or
        # skill-level `true` overrides it, because that setting promises every
        # case of the skill runs.
        tasks.extend(_runtime_tasks(skill_name, skill_env, work_dir, config,
                                    force_all, have_pytest))

        # === Phase 2: YAML task tests (three modes) ===
        # The converter parses every test YAML discovered by the
        # handler into a flat .json config (unified replacement of
        # the former per-skill dev_tool).
        # File naming: flat .json configs match the test YAML names
        # (e.g. casscf_energy.yaml -> casscf_energy.json).
        # The certification/converter.py module later converts each
        # .json into the PyRAI2MD classic text input (.input) and
        # the pyrai3md agent config (.config) from one shared
        # sectioned dict.
        test_configs, refused, failed = _generate_case_configs(
            skill_name, skill_dir)
        if refused:
            tasks.append(
                _make_error_task(skill_name, refused) if failed
                else _make_diagnostic_task(skill_name, refused))
            continue

        # Inject merged environment parameters (e.g. molcas_path) into
        # the generated configs so skill_agent mode (pyrai3md_skill,
        # which performs no pmd_env.yaml merge) can find them. Done
        # before convert_to_classic_input so all three modes share the
        # same effective config. All qualification wrappers carry the
        # global environment.shell; the skill-level shell
        # (skills.<name>.shell) reaches the skill config here and
        # surfaces in the execution script the skill backend writes.
        _inject_env_into_configs(test_configs, skill_env)

        # Skill tests are skill-driven by definition: force
        # driver='skill' so the pyrai3md_input/pyrai3md_agent modes
        # execute them through the skill driver (only the main
        # framework examples run driver=main).
        _force_skill_driver(test_configs)

        # One directory per independent case group. Cases naming the same
        # modeldir are a chain and share a directory per carrier leg; every
        # other case gets its own, so groups cannot overwrite each other
        # when a slurm round runs them as separate jobs.
        _, group_of = _case_groups(test_configs)
        # A case's generated configs are written into its own leg by the
        # per-case loop; the shared copies stay in the skill directory and
        # must not be staged into every group.
        exclude_names = set()
        for stem in test_configs:
            for suffix in ('.json', '.config', '.input', '.xyz', '.velo'):
                exclude_names.add(stem + suffix)

        if skill_slurm and _scheduler_is_slurm(config):
            print('  warning: skills.%s.execution_mode is slurm while this '
                  'round runs as grouped slurm jobs; each leg then submits '
                  'a nested job from a compute node. Sites that forbid '
                  'compute-node submission need execution_mode: local for '
                  'this round.' % skill_name, flush=True)

        for profile_name, config_path in test_configs.items():
            task_name = f'{skill_name}_{profile_name}'
            # The run title the case declares inside itself. A framework-driven leg
            # names its run log by that title (`<title>.log`, not `<case stem>.log`),
            # so without this anchor the post-run log check looks for a file the run
            # never wrote -- which is how a skill agent that swallowed its exception
            # and exited 0 stayed a pass after #321's success-word rule was added.
            case_title = _case_run_title(config_path, profile_name)

            # The per-case switch, read only when both the master and
            # `skills.<name>.test` deferred.
            closed = _closed_case_task(
                skill_name, task_name, config_path, profile_name,
                case_flags, force_all)
            if closed is not None:
                tasks.append(closed)
                continue

            gid = group_of[profile_name]
            # The directory keeps only the ordinal (`g07`): the descriptive
            # gid (`g07_casscf_geometry_optimization`) pushed a Molcas
            # calculation directory past the engine's path limit, and the
            # gateway then failed to open the coordinates file it had just
            # been handed (round tmp_20261111_162136). The job group keeps the
            # descriptive name for the report.
            group_dir = os.path.join(skill_dir, gid.split('_', 1)[0])
            try:
                tasks.extend(_plan_case_modes(
                    skill_name, profile_name, config_path, task_name,
                    case_title, skill_dir, group_dir,
                    exclude_names, f'{skill_name}__{gid}', skill_slurm,
                    export_vars, shell_commands, test_timeout))
            except Exception as exc:  # noqa: BLE001 - the case row fails
                # A skill whose query module cannot answer for its own
                # defaults fails the case it was building, and says so: the
                # row is a diagnostic, not a silently missing section, and
                # the other cases of the round still plan.
                tasks.append(_make_error_task(
                    skill_name,
                    '%s: test planning failed: %s' % (profile_name, exc)))

    # Once for the round, not once per skill: these files test the layer the
    # skills share (see _shared_runtime_tasks).
    tasks.extend(_shared_runtime_tasks(
        config, work_dir, have_pytest,
        shared_runtime_open, shared_runtime_closer))

    return tasks


def _host_gate(skill_registry, skill_name, skill_env):
    """Why this skill cannot run on this host, and whether it self-submits.

    Third-party packages the manifest lists under 'requires' are checked
    here: an optional stack that is simply not installed must surface as
    a diagnostic skip naming the package, never as a green run and never
    as a silently absent skill. A skill whose execution_mode is 'slurm'
    needs sbatch now, so the missing binary is a diagnostic skip rather
    than a runtime failure.

    Returns
    -------
    tuple
        ``(reason, skill_slurm)``; ``reason`` is None when the skill may
        run, and ``skill_slurm`` says whether its backend submits itself.
    """
    missing = _missing_requirements(skill_registry, skill_name)
    if missing:
        return ("skill needs %s, which is not importable on this host"
                % ', '.join(missing)), None

    skill_slurm = str(
        skill_env.get('execution_mode', '') or '').lower() == 'slurm'
    if skill_slurm and shutil.which('sbatch') is None:
        return ("skill execution_mode is 'slurm' but sbatch is not "
                "available on this host"), True
    return None, skill_slurm


def _generate_case_configs(skill_name, skill_dir):
    """The flat configs of one skill, or the diagnostic that replaces them.

    A generator failure and an empty generator are both reported as a
    diagnostic skip task: the round must say why no case exists rather
    than show a skill with no rows.

    Returns
    -------
    tuple
        ``(test_configs, reason)``; ``reason`` is None when configs were
        generated, and ``test_configs`` is None otherwise.
    """
    try:
        test_configs = generate_test_configs(skill_name, skill_dir)
    except Exception as exc:  # noqa: BLE001 - surface as a diagnostic error
        return None, f'test config generation failed: {exc}', True
    if not test_configs:
        return None, ('no test cases generated: %s'
                      % _diagnose_missing_tests(skill_name)), False
    return test_configs, None, False


def _closed_case_task(skill_name, task_name, config_path, profile_name,
                      case_flags, force_all):
    """The skipped task of a closed case, or None when the case runs.

    The per-case switch is read only when both the master and
    `skills.<name>.test` deferred: a `true` above promises every case of
    the skill runs, and the map cannot take that back. A closed case
    keeps its generated config on disk (the converter writes the whole
    skill in one pass) and appears in the plan once, as a skipped task,
    instead of three times as three modes that never run.
    """
    if force_all:
        return None
    case_on, case_closer = _skill_case_switch(
        skill_name, case_flags, profile_name)
    if case_on:
        return None
    return {
        'name': task_name,
        'skill': skill_name,
        'mode': 'skill_case',
        'input_file': config_path,
        'skip': True,
        'skip_reason': f'disabled by {case_closer}',
    }


def _plan_case_modes(skill_name, profile_name, config_path, task_name,
                     case_title, skill_dir, group_dir, exclude_names,
                     job_group, skill_slurm, export_vars, shell_commands,
                     test_timeout):
    """Write one case's three carrier legs and return their tasks.

    The three legs run the same effective configuration through three
    entry points (pyrai3md_skill, pyrai3md_agent, pyrai3md <input>).
    Each leg gets its own run directory and its own copy of the config,
    so a leg's log and its products name that leg rather than the last
    writer of a shared path. The directories sit under the case group's
    directory, which is what lets independent cases run as separate
    jobs without meeting each other's files.
    """
    tasks = []
    # === Mode 1: pyrai3md_skill config (local inside the case's job) ===
    # cd into the leg's own directory so relative outputs
    # (e.g. optimize's opt.xyz/opt_geom.xyz/opt_status.txt, a
    # trained ensemble folder) stay inside the run that wrote them,
    # matching the examples isolation policy. The same command string
    # is embedded in the local .sh script.
    skill_run = _stage_leg_directory(
        skill_dir, group_dir, 'skill_agent', exclude_names)
    # The leg runs its OWN copy of the config: a skill agent names its
    # log after the file it is handed (`splitext(config)[0] + '.log'`),
    # so pointing all legs at one shared path made two carrier legs
    # write the same `<stem>.log` beside the configs and overwrite each
    # other -- the last writer's status then stood for all of them. The
    # copy rule in `_stage_leg_directory` already put the file here; running it from
    # here is what makes the log, and the judgement, per-leg. The same
    # copy also gets a leg-private scratch, so no leg starts from the
    # orbitals another leg converged.
    leg_config = _write_leg_config(
        config_path, skill_run, profile_name)
    skill_cmd = f'cd {skill_run} && pyrai3md_skill {leg_config}'
    sh_skill = _write_shell_script(
        skill_dir, f'{task_name}_skill',
        skill_cmd, export_vars, shell_commands,
        timeout=test_timeout,
    )
    task_sa = {
        'name': f'{task_name}_skill',
        'skill': skill_name,
        'mode': 'skill_agent',
        'input_file': config_path,
        'run_dir': skill_run,
        'title': case_title,
        'script': sh_skill,
        'use_slurm': False,
        'skill_slurm': skill_slurm,
        # One job per carrier leg: the legs share no files and no
        # ordering, so only the case chain inside a leg is a necessary
        # group.
        'job_group': '%s__skill_agent' % job_group,
    }
    tasks.append(task_sa)

    # === Mode 2: pyrai3md_agent config ===
    # The agent config is the converter's sectioned dict
    # serialized as JSON ({profile}.config), so pyrai3md_agent
    # consumes the same effective configuration as input mode
    # (skill parameters carried in the skill section).
    agent_config_path = os.path.join(
        skill_dir, f'{profile_name}.config')
    convert_to_pyrai_config(config_path, agent_config_path)
    agent_run = _stage_leg_directory(
        skill_dir, group_dir, 'pyrai3md_agent', exclude_names)
    agent_leg_config = _write_leg_config(
        config_path, agent_run, profile_name)
    agent_leg_path = os.path.join(
        agent_run, f'{profile_name}.config')
    convert_to_pyrai_config(agent_leg_config, agent_leg_path)
    agent_cmd = (f'cd {agent_run} && pyrai3md_agent '
                 f'{os.path.join(agent_run, os.path.basename(agent_leg_path))}')
    sh_agent = _write_shell_script(
        skill_dir, f'{task_name}_agent',
        agent_cmd, export_vars, shell_commands,
        timeout=test_timeout,
    )
    task_pa = {
        'name': f'{task_name}_agent',
        'skill': skill_name,
        'mode': 'pyrai3md_agent',
        'input_file': agent_config_path,
        'run_dir': agent_run,
        'title': case_title,
        'script': sh_agent,
        'use_slurm': False,
        'skill_slurm': skill_slurm,
        'job_group': '%s__pyrai3md_agent' % job_group,
    }
    tasks.append(task_pa)

    # === Mode 3: pyrai3md input ===
    input_name = f'{profile_name}.input'
    input_path = os.path.join(skill_dir, input_name)
    convert_to_classic_input(config_path, input_path)
    input_run = _stage_leg_directory(
        skill_dir, group_dir, 'pyrai3md_input', exclude_names)
    input_leg_config = _write_leg_config(
        config_path, input_run, profile_name)
    input_leg_path = os.path.join(input_run, input_name)
    convert_to_classic_input(input_leg_config, input_leg_path)
    input_cmd = (f'cd {input_run} && pyrai3md '
                 f'{os.path.join(input_run, os.path.basename(input_leg_path))}')
    sh_input = _write_shell_script(
        skill_dir, f'{task_name}_input',
        input_cmd, export_vars, shell_commands,
        timeout=test_timeout,
    )
    task_pi = {
        'name': f'{task_name}_input',
        'skill': skill_name,
        'mode': 'pyrai3md_input',
        'input_file': input_path,
        'run_dir': input_run,
        'title': case_title,
        'script': sh_input,
        'use_slurm': False,
        'skill_slurm': skill_slurm,
        'job_group': '%s__pyrai3md_input' % job_group,
    }
    tasks.append(task_pi)
    return tasks


def _diagnose_missing_tests(skill_name):
    """Diagnose why no test configs were generated for a skill.

    Checks the skill's tests directory for common issues:
    missing package or no YAML files.

    Parameters
    ----------
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').

    Returns
    -------
    str
        Human-readable diagnosis message.
    """
    try:
        skill_pkg = importlib.import_module(
            skill_module_path(skill_name, ''))
    except (ImportError, ModuleNotFoundError):
        return (
            f'skill package {skill_module_path(skill_name, "")} is not '
            'installed; '
            f'rerun pip install to include skill resources'
        )

    skill_dir = os.path.dirname(skill_pkg.__file__)
    tests_dir = os.path.join(skill_dir, 'tests')

    if not os.path.isdir(tests_dir):
        return f'tests/ directory not found in {skill_dir}'

    yaml_files = [
        f for f in os.listdir(tests_dir)
        if f.endswith('.yaml')
    ]
    if not yaml_files:
        return f'no test YAML files found in {tests_dir}'

    return 'test YAML parsing produced no configs'


def _missing_requirements(skill_registry, skill_name):
    """Return the manifest requirements this host cannot import.

    Parameters
    ----------
    skill_registry : dict
        Registry produced by discover_all().
    skill_name : str
        Skill directory name (e.g. 'nn_pes_skill').

    Returns
    -------
    list of str
        Requirement names that cannot be imported. Empty when the skill
        declares none, which is the case for every shipped engine skill.
    """
    for meta in skill_registry.values():
        if meta.get('skill_name') != skill_name:
            continue
        missing = []
        for name in meta.get('requires') or []:
            try:
                if importlib.util.find_spec(name) is None:
                    missing.append(name)
            except (ImportError, ValueError):
                missing.append(name)
        return missing
    return []


def _make_error_task(skill_name, reason):
    """A task row that fails the round instead of skipping it.

    A planner failure or a skill that no longer exists is a defect in the
    plan, not a case this host cannot run. The row carries an error status
    so the summary cannot count the round as a pass.
    """
    return {
        'name': f'{skill_name}_diagnostic',
        'skill': skill_name,
        'mode': 'local',
        'skip': False,
        'status': 'error',
        'details': reason,
        'skip_reason': reason,
    }


def _make_diagnostic_task(skill_name, skip_reason):
    """Create a diagnostic skip task for a skill with test issues.

    Parameters
    ----------
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').
    skip_reason : str
        Human-readable explanation of why tests could not be generated.

    Returns
    -------
    dict
        A skip task descriptor for the test report.
    """
    return {
        'name': f'{skill_name}_diagnostic',
        'skill': skill_name,
        'mode': 'local',
        'skip': True,
        'skip_reason': skip_reason,
    }


def _is_execution_control_key(key):
    """Check whether a skill config key carries the execution policy.

    execution_mode and the skill slurm_* resource keys decide how the
    skill backend submits its own job (local shell vs sbatch -W).
    The qualification config (pmd_env.yaml) is authoritative for these
    keys, so injection overrides existing values instead of filling:
    the test YAMLs hardcode execution_mode=local, which fill-only
    injection could never flip.

    Parameters
    ----------
    key : str
        Skill config key name.

    Returns
    -------
    bool
        True for execution_mode and slurm_* keys.
    """
    return key == 'execution_mode' or key.startswith('slurm_')


def _inject_env_into_configs(test_configs, skill_env):
    """Inject merged environment parameters into generated configs.

    The pyrai3md_skill entry point (skill_agent mode) runs the skill
    as an independent subprocess and performs no pmd_env.yaml merge,
    unlike pyrai3md and pyrai3md_agent. Machine-specific parameters
    such as molcas_path must therefore be present in the config file
    itself. This helper fills them from the merged skills.<name>
    section of pmd_env.yaml (package + user layer) at plan time.
    A skill-level shell key (skills.<name>.shell) rides along the
    same way so the skill run scripts can write the module loads.
    The global environment.shell is NOT injected: it is reserved for
    the main framework scripts.

    Fill semantics follow the three-layer override rules: existing
    non-empty values in the config (from the skill test YAML) are
    never overwritten; a key is filled only when missing or holding
    an empty value (None, '', [], {}). False and 0 are valid values.
    Qualification-only keys ('test', the per-case 'tests' map, and
    'runtime_tests') are never injected: they decide what the planner
    runs and mean nothing to a skill backend.

    Exception: execution control keys (execution_mode and slurm_*)
    use override semantics instead of fill -- pmd_env.yaml is the
    authoritative qualification source for the skill self-submission
    policy, and the test YAMLs hardcode execution_mode=local.

    Parameters
    ----------
    test_configs : dict
        Mapping {profile_name: config_path} returned by generate_all.
    skill_env : dict
        Merged skills.<skill_name> section of pmd_env.yaml.
    """
    env_params = {}
    if isinstance(skill_env, dict):
        env_params = {
            key: value for key, value in skill_env.items()
            if key not in QUALIFICATION_ONLY_KEYS
        }
    if not env_params:
        return

    for config_path in test_configs.values():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (OSError, ValueError):
            continue

        changed = False
        for key, value in env_params.items():
            if value is None or value == '' or value == [] or value == {}:
                continue
            current = config.get(key)
            if _is_execution_control_key(key):
                # Override semantics for the execution policy keys
                # (see docstring).
                if current != value:
                    config[key] = value
                    changed = True
            elif current is None or current == '' or current == [] or current == {}:
                config[key] = value
                changed = True

        if changed:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)


def _force_skill_driver(test_configs):
    """Force driver='skill' into every skill-test config.

    Skill test cases are skill-driven by definition: the framework
    modes (pyrai3md_input / pyrai3md_agent) must execute them
    through the skill driver, so a single call per job with the
    skill owning the whole job. Only the main framework examples
    run driver=main. The override is unconditional -- any driver
    value carried by the test YAML is replaced.

    The injected field reaches the classic input through the
    converter (&CONTROL gains a 'driver skill' line) and the agent
    entry points read it from the flat config.

    Parameters
    ----------
    test_configs : dict
        Mapping {profile_name: config_path} returned by generate_all.
    """
    for config_path in test_configs.values():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (OSError, ValueError):
            continue

        if config.get('driver') != 'skill':
            config['driver'] = 'skill'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
