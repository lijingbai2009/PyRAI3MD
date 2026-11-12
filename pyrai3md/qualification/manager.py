"""
======================================================================
This module parses the pmd_env.yaml configuration and generates
the unified test plan with input files.

It is responsible for:
1. Loading and parsing pmd_env.yaml (YAML format)
2. Planning the main framework tests (verification/ unit tests)
   and the example qualification tasks (examples/*.yaml, three
   modes)
3. Aggregating the skill certification tasks from
   certification.planner.plan_skill_tests and the benchmark tasks
   from benchmark.planner.plan_benchmark_tests
4. Creating a tmp_YYYYMMDD_HHMMSS working directory for all files
5. Applying slurm settings to skill test tasks (input/agent modes only)

The manager does not execute any tests. Skill-side planning lives
in certification/planner.py and benchmark planning in
benchmark/planner.py; both are imported lazily inside generate_plan
so the shared script writers defined here can be consumed by those
modules without an import cycle.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import copy
import json
import os
import re
import shlex
import shutil
import time
import warnings
from functools import lru_cache

import yaml

from ..utilities.config_merge import deep_merge, is_empty_value
from ..skills.query import skill_module_path, skill_stem
from ..kernel.sbatch import render_sbatch_directives
from .certification.converter import (convert as convert_to_classic_input,
                                      yaml_to_config)


# Default path to pmd_env.yaml relative to this file
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), 'pmd_env.yaml')

# User override layer, merged on top of DEFAULT_CONFIG by load_config
USER_CONFIG = os.path.join(
    os.path.expanduser('~'), '.pyrai3md', 'pmd_env.yaml'
)

# Source-tree override layer, resolved from the current working
# directory at load time (the highest-priority layer): when
# qualification runs inside a PyRAI3MD checkout, the workspace
# pmd_env.yaml wins over the installed package copy, so editing the
# workspace config takes effect without a reinstall.
CWD_CONFIG_RELPATH = os.path.join(
    'pyrai3md', 'qualification', 'pmd_env.yaml')

# Example qualification policy: the examples a round *postpones*.
#
# The plan is derived from disk (`pyrai3md/examples/*.yaml`, in the order
# `_order_example_files` gives), so shipping a new YAML adds it to the plan
# with no Python edit. This map is the exception channel: a case whose
# runtime nobody has stabilized yet is named here with the reason, and the
# plan reports it as a skipped task instead of running it or going silent.
# The annotated inventory of the cases and the measurements behind them is
# in dev/QUALIFICATION_DEVELOPMENT.md section 6.6.
POSTPONED_EXAMPLES = {}


# Examples that share one run directory and run in the declared order.
# Every example otherwise gets a directory of its own so a failure keeps
# its own log; a chain is the exception, because its later member reads
# what its earlier member wrote -- the same arrangement a skill's test
# cases use, where case 00 trains and the cases after it serve.
#
# No runtime dependency is propagated: when the first member fails, the
# second fails on its missing model folder, and the skill's own error
# already names the fix ("Train the model with jobtype 'train' first").
# That is a truer report than a skip invented here.
EXAMPLE_CHAINS = {
    'nn': ('nn_train', 'nn_predict'),
}


class _NoDupKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    yaml.safe_load silently keeps the last value of a duplicated
    key, which once collapsed three benchmark.fssh entries (one
    per Tully model, each carrying a leftover 'method: tully1'
    line) into a single method and hid the mistake behind
    misleading duplicate-method skips. A duplicate key in
    pmd_env.yaml is always an editing error, so the load fails
    loudly instead; the raised ConstructorError is a YAMLError
    subclass and follows the existing syntax-error handling paths.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=False)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    'while constructing a mapping', node.start_mark,
                    "found duplicate key '%s'" % key,
                    key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep)


def load_config(config_path=None):
    """Load and parse the pmd_env.yaml configuration file.

    Loads the package qualification/pmd_env.yaml and, when no
    explicit config_path is given, deep-merges the user override
    layer ~/.pyrai3md/pmd_env.yaml and then the source-tree
    override layer ./pyrai3md/qualification/pmd_env.yaml (resolved
    from the current working directory) on top of it, matching the
    runtime Environment semantics: higher-priority values win, and
    empty override values (None, '', [], {}) fall back to the next
    lower-level non-empty value (False and 0 are valid values and
    never fall back). The source-tree layer keeps a qualification
    run launched from a PyRAI3MD checkout in sync with the
    workspace config even when the imported package is a stale
    installation snapshot. Duplicate mapping keys are rejected by
    the loader instead of silently collapsing to the last value.

    Parameters
    ----------
    config_path : str, optional
        Path to the pmd_env.yaml file. Defaults to qualification/pmd_env.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    use_user_override = config_path is None
    if config_path is None:
        config_path = DEFAULT_CONFIG

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=_NoDupKeyLoader) or {}

    if use_user_override:
        user_config = _load_user_config()
        if user_config:
            config = deep_merge(config, user_config)
        cwd_config = _load_cwd_config()
        if cwd_config:
            config = deep_merge(config, cwd_config)

    return config


def _load_user_config():
    """Load the user override layer ~/.pyrai3md/pmd_env.yaml.

    Returns
    -------
    dict or None
        Parsed user configuration, or None when the file is absent
        or unreadable.
    """
    if not os.path.isfile(USER_CONFIG):
        return None
    try:
        with open(USER_CONFIG, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=_NoDupKeyLoader) or {}
    except (OSError, yaml.YAMLError):
        return None


def _load_cwd_config():
    """Load the source-tree override layer from the working directory.

    Returns the parsed ./pyrai3md/qualification/pmd_env.yaml when
    the current working directory is a PyRAI3MD checkout, or None
    when the file is absent (every non-checkout launch falls back
    to the package config + user override chain unchanged). Unlike
    the optional user layer, parse errors are raised: silently
    ignoring a present source-tree config reproduces exactly the
    stale-snapshot divergence this layer exists to prevent.

    Returns
    -------
    dict or None
        Parsed source-tree configuration, or None when absent.
    """
    path = os.path.join(os.getcwd(), CWD_CONFIG_RELPATH)
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=_NoDupKeyLoader) or {}






def create_work_dir(base_dir=None):
    """Create a timestamped working directory for test files.

    Creates a directory named tmp_YYYYMMDD_HHMMSS to store
    all generated input files and test results.

    Parameters
    ----------
    base_dir : str, optional
        Parent directory for the tmp folder. Defaults to project root.

    Returns
    -------
    str
        Absolute path to the created working directory.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join(base_dir, f'tmp_{timestamp}')
    os.makedirs(work_dir, exist_ok=True)

    return work_dir


def generate_plan(config, work_dir=None):
    """Generate a test plan from the parsed configuration.

    The plan contains four categories, executed in this order:
    - main_tests: verification unit tests plus, when `test_integration`
        (or the `_runall` master) opens them, the cross-mode/all-example
        tests from qualification/integration/; always executed locally
    - example_tests: official workflow YAMLs from examples/, qualified
        examples run in two modes (pyrai3md yaml / pyrai3md input);
        the rest are listed as skipped
        reference workflows
    - skill_tests: planned by certification.planner.plan_skill_tests,
        executed per registered skill in three modes:
        - pyrai3md_input: traditional CLI, uses slurm if configured
        - pyrai3md_agent: agent CLI, uses slurm if configured
        - skill_agent: skill-managed via wrapper, no slurm
    - benchmark_tests: planned by benchmark.planner
        .plan_benchmark_tests, application-level algorithm and
        performance benchmarks run after every functional test

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.
    work_dir : str, optional
        Working directory for input files. Created if None.

    Returns
    -------
    dict
        Test plan with structure:
        {
            'main_tests': [...],
            'example_tests': [...],
            'skill_tests': [...],
            'benchmark_tests': [...],
            'slurm': {...},
            'export': {...},
            'work_dir': '...'
        }
    """
    if work_dir is None:
        work_dir = create_work_dir()

    # The verification planner has no manager dependency and is
    # imported at load time; the certification and benchmark planners
    # consume the shared script writers defined in this module, so a
    # top-level import of them here would form a cycle.
    from .verification.planner import plan_main_tests
    from .certification.planner import plan_skill_tests
    from .benchmark.planner import plan_benchmark_tests

    plan = {
        'main_tests': plan_main_tests(config),
        'example_tests': _plan_example_tests(config, work_dir),
        'skill_tests': plan_skill_tests(config, work_dir),
        'benchmark_tests': plan_benchmark_tests(config, work_dir),
        'slurm': config.get('slurm', {}),
        'export': _merge_exports(config),
        'work_dir': work_dir,
    }
    return plan


# The third state of a qualification switch. `true` and `false` state the
# answer for everything below the switch; `condition` means "ask the switch
# one level down", and it is the only setting under which the per-test keys
# are read at all.
SWITCH_CONDITION = 'condition'

# The two-valued spellings a switch answers to, in lower case.
# `_coerce_switch` and `_coerce_flag` both read them; a third list typed
# out somewhere else is how one switch stops meaning the same thing at
# two levels.
TRUE_TOKENS = ('true', 'yes', 'on', '1')
FALSE_TOKENS = ('false', 'no', 'off', '0')

# The keys of a `skills.<name>` section that only the planner reads. They
# decide what runs, so they must not reach the generated skill config: a
# backend with extra keys in its section is a validation question, and one
# that accepts them silently gains a knob nobody documented. The planner
# and the example staging filter this same list.
QUALIFICATION_ONLY_KEYS = ('test', 'tests', 'runtime_tests')

# The example families that carry a switch of their own, as
# {switch key: workflow prefix}. Membership is a name prefix, and
# `_example_state` takes the first prefix that matches, so a new entry is
# appended after the narrower ones and the prefixes stay disjoint over
# the shipped set (measured over the 201 stems in examples/: `md_lammps_` 12,
# `constraint_md_` 33, `optimize_` 25, `path_` 32, `multiscale_` 37,
# `adaptive_` 6, no stem matching two of them;
# verification/test_manager.py pins that no permutation of this table changes
# any stem's decision).
#
# The families, by ruling. `optimize_test` was added to give the optimize
# route one lever - the twenty-five `optimize_*` cases, the seventeen
# `optimize_cell_*` among them - because a round that is meant to check the
# optimizer should not have to queue behind the other examples. The
# name-not-physics half of that
# ruling stands: membership follows the name, not the physics, so
# `constraint_opt_periodic` and `constraint_opt_surface`, whose inputs do say
# `jobtype: optimize`, stay individual and an optimize round does not silently
# certify the constraint subsystem. `energy`, `gradient`, `multicenter_*` and
# the `md_*_fssh` set have no family either. That half is also what keeps the
# two route prefixes from reaching into each other: a path run is asked for as
# `optimize.method`, yet the path cases carry their own prefix, so `path_test`
# opens the path chain without opening the single-structure searches with it -
# and neither family key answers for the other's stems, which
# `integration/test_integration_manager_planning.py::TestPathFamilySwitch`
# pins from both sides.
#
# `path_test` is the fourth, added with the path family itself (`optimize.method`
# mep / irc / neb, one keyword each in this file's terms): the twenty-one
# `path_*` examples get a group switch as well as their own keys, because a round
# that checks the path chain should be one key away from the operator the way an
# optimize round is. Membership is still the *name*: a `path_*` example joins the
# path group by being named that way, and `optimize_*` keeps its own family - the
# two prefixes are disjoint, so neither shadows the other and no row order of
# this table changes any stem's decision.
EXAMPLE_FAMILIES = {
    'md_lammps_test': 'md_lammps_',
    'constraint_md_test': 'constraint_md_',
    'optimize_test': 'optimize_',
    'path_test': 'path_',
    # Multiscale composite cases (ONIOM2/3 and QMMM), all named
    # `multiscale_*`. One key opens the whole family for a round that
    # has openqp/xtb/lammps/molcas available.
    'multiscale_test': 'multiscale_',
    # Adaptive sampling cases (single-method, ONIOM2 and QMMM mechanical
    # embedding, each in a local and a slurm carrier), all named
    # `adaptive_*`. One key opens the whole family for a round that checks
    # the probe -> reference -> train chain, the way `multiscale_test`
    # opens the composite set.
    'adaptive_test': 'adaptive_',
}


def examples_directory():
    """Directory of the example YAMLs the package ships."""
    return os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, 'examples'))


def shipped_examples(directory=None):
    """Case-name stems of the shipped example YAMLs, sorted.

    The plan and the criteria that walk a family both start here, so a
    new YAML under ``pyrai3md/examples/`` joins them with no Python
    edit; family membership is then the name prefix the switch table
    declares (``EXAMPLE_FAMILIES``).
    """
    directory = directory or examples_directory()
    return sorted(os.path.splitext(name)[0]
                  for name in os.listdir(directory)
                  if name.endswith('.yaml'))


def family_members(prefix, directory=None):
    """Shipped example stems that join one family, by name prefix."""
    return [stem for stem in shipped_examples(directory)
            if stem.startswith(prefix)]


def _coerce_switch(value, label, default):
    """Interpret a three-valued qualification switch.

    Parameters
    ----------
    value : any
        Raw switch value from pmd_env.yaml.
    label : str
        Switch name, used in the warning message.
    default : bool or str
        What an unrecognized value falls back to. A typo never silently
        closes a test: the warning is emitted and the caller's default, which
        is always the setting that keeps the most tests running, wins.

    Returns
    -------
    bool or str
        True, False, or :data:`SWITCH_CONDITION`.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value != 0

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_TOKENS:
            return True
        if lowered in FALSE_TOKENS:
            return False
        if lowered == SWITCH_CONDITION:
            return SWITCH_CONDITION

    warnings.warn(
        f"Unrecognized {label} switch {value!r} in pmd_env.yaml; expected "
        f"true, false or '{SWITCH_CONDITION}'. Falling back to {default!r}."
    )
    return default


def _resolve_chain(states):
    """The first decisive state of a switch chain, from the top down.

    Parameters
    ----------
    states : iterable of (bool or str, str)
        ``(state, label)`` pairs, highest switch first.

    Returns
    -------
    tuple
        ``(state, label)`` of the first switch that is not
        :data:`SWITCH_CONDITION`, with the name that gave the answer;
        ``(SWITCH_CONDITION, '')`` when the whole chain defers, which leaves
        the per-test switch to decide.
    """
    for state, label in states:
        if state != SWITCH_CONDITION:
            return state, label
    return SWITCH_CONDITION, ''


def _example_state(master, family_states, workflow):
    """Decide one example from the master and family switches.

    Parameters
    ----------
    master : bool or str
        The resolved `_runall` switch.
    family_states : dict
        Resolved state of each :data:`EXAMPLE_FAMILIES` key.
    workflow : str
        Example workflow name (the YAML file stem).

    Returns
    -------
    tuple
        ``(state, label)``: True or False with the switch that said so, or
        ``(SWITCH_CONDITION, '')`` when the per-example switch decides.
    """
    chain = [(master, '_runall')]
    for name, prefix in EXAMPLE_FAMILIES.items():
        if workflow.startswith(prefix):
            state = family_states[name]
            chain.append((state, ('examples.%s' % name
                                  if state != SWITCH_CONDITION else '')))
            break
    return _resolve_chain(chain)


def _warn_unknown_example_keys(example_flags, workflows):
    """Point at an `examples:` key that names no test and no family switch.

    A family switch is read by name, so `md_lammps_tes: true` would otherwise
    sit in the file doing nothing while every `md_lammps_*` case kept its own
    setting - the shape of mistake that reads as "I turned the family on".
    """
    unknown = sorted(key for key in example_flags
                     if key not in workflows and key not in EXAMPLE_FAMILIES)
    if unknown:
        warnings.warn(
            "Unrecognized key(s) in the pmd_env.yaml examples section: %s. "
            "No example or family switch has that name, so the key changes "
            "nothing; expected an example workflow (%s...) or one of %s."
            % (', '.join(repr(key) for key in unknown),
               sorted(workflows)[0] if workflows else 'none',
               ', '.join(sorted(EXAMPLE_FAMILIES)))
        )


def _coerce_flag(value, label, default=True, noun='test'):
    """Interpret a leaf (two-valued) switch as a boolean.

    A leaf is the end of the chain and has nothing below it to defer to, so
    :data:`SWITCH_CONDITION` is not an available answer there: it is reported
    as the misplaced value it is and the default wins. An unrecognized value
    warns rather than silently closing a test, because a typo in a switch that
    was meant to run something is the expensive direction to get wrong.

    Parameters
    ----------
    value : any
        Raw switch value from pmd_env.yaml.
    label : str
        Switch name as the operator wrote it, used in the message.
    default : bool
        What an unrecognized or misplaced value falls back to.
    noun : str
        Word used for the switched thing in the message ('example', 'case').

    Returns
    -------
    bool
        Whether the switched test runs.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value != 0

    stays = 'enabled' if default else 'disabled'

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_TOKENS:
            return True
        if lowered in FALSE_TOKENS:
            return False
        if lowered == SWITCH_CONDITION:
            warnings.warn(
                f"{label} is the per-{noun} switch at the end of the chain, "
                f"so '{SWITCH_CONDITION}' has no meaning there; write true or "
                f"false. The {noun} stays {stays}."
            )
            return default

    warnings.warn(
        f"Unrecognized {label} switch {value!r} in pmd_env.yaml; expected "
        f"true/false. The {noun} stays {stays}."
    )
    return default


def _coerce_example_flag(value, workflow):
    """Interpret a pmd_env.yaml example switch as a boolean.

    YAML parses unquoted true/false into real booleans, but typos
    like ``fasle`` or quoted values arrive as non-empty strings,
    which would silently count as enabled. Common true/false
    spellings are normalized here; anything unrecognized raises a
    warning and falls back to enabled so a typo never silently
    disables a qualification test.

    Parameters
    ----------
    value : any
        Raw switch value from the 'examples' section of
        pmd_env.yaml.
    workflow : str
        Example workflow name, used in the warning message.

    Returns
    -------
    bool
        Whether the example qualification test is enabled.
    """
    return _coerce_flag(value, f'examples.{workflow}', True, 'example')


def _runall_mode(config):
    """Interpret the top-level `_runall` master switch.

    Three states, by ruling: True forces every configured switch
    (examples.<workflow>, the example family switches,
    skills.<name>.test and its per-case switches,
    benchmark.<task>[].enabled) on; False forces them all off - and it is
    decided above the family switches, so `_runall: false` with a family set to
    `true` still runs nothing of that family;
    :data:`SWITCH_CONDITION` leaves each section's own switches to answer,
    which is what the file defaults to when the key is absent - the meaning
    the bare `false` carried before the three-state switch existed.

    The POSTPONED_EXAMPLES policy is orthogonal to this switch,
    and the verification unit tests carry no switch and keep running; the
    integration tests answer to this master through ``test_integration``.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    bool or str
        True, False, or :data:`SWITCH_CONDITION`.
    """
    return _coerce_switch(config.get('_runall', SWITCH_CONDITION), '_runall',
                          SWITCH_CONDITION)


def _family_states(config):
    """Resolve the example family switches, defaulting to `condition`.

    An absent family key defers, so a file written before the family switches
    existed keeps every per-example switch in charge exactly as it did.
    """
    example_flags = config.get('examples', {}) or {}
    return {
        name: _coerce_switch(example_flags.get(name, SWITCH_CONDITION), name,
                             SWITCH_CONDITION)
        for name in EXAMPLE_FAMILIES
    }


def _inject_example_skill_env(config_path, skills):
    """Share pmd_env.yaml skill settings into an example flat config.

    Examples run through the same yaml -> flat json -> .input
    flow as skill tests, but the example YAML only carries the
    calculation keywords (method/basis/...), never the machine-level
    skill parameters. This fills them from the merged skills.<name>
    section of pmd_env.yaml so the generated artifacts are complete
    (path/nproc/shell/execution_mode/keep_tmp/scratch_path/...) even
    though skills.<name>.test may be false: the test switch only gates
    skill certification, it never hides an already-configured skill
    section from the example tasks.

    A multi-theory example keeps one block per skill under
    'skill_sections' (keyed by stem); the settings are injected into
    each block so the converter's per-skill section builder
    (handler._build_skill_pairs, section_params) reads them. A
    single-skill example has no 'skill_sections', so its primary
    skill's settings are injected flat at the top level (the
    converter's single-skill builder reads the config directly).

    Fill semantics mirror certification.planner._inject_env_into_configs:
    empty values (None, '', [], {}) are skipped; execution_mode and
    slurm_* override existing values, every other key fills only when
    the target is missing/empty. The qualification-only 'test' key is
    never injected.

    Parameters
    ----------
    config_path : str
        Path to the example flat .json config (modified in place).
    skills : dict
        The pmd_env.yaml 'skills' mapping {skill_name: {...}}.
    """
    if not isinstance(skills, dict) or not skills:
        return

    from .certification.planner import _is_execution_control_key

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(config, dict):
        return

    def _env_params(stem):
        skill_env = skills.get('%s_skill' % stem)
        if not isinstance(skill_env, dict):
            return {}
        return {k: v for k, v in skill_env.items()
                if k not in QUALIFICATION_ONLY_KEYS}

    def _apply(target, params):
        changed = False
        for key, value in params.items():
            if is_empty_value(value):
                continue
            current = target.get(key)
            if _is_execution_control_key(key):
                # Override semantics for the execution policy keys.
                if current != value:
                    target[key] = value
                    changed = True
            elif is_empty_value(current):
                target[key] = value
                changed = True
        return changed

    changed = False
    skill_sections = config.get('skill_sections')
    if isinstance(skill_sections, dict) and skill_sections:
        # Multi-theory example: inject each skill into its own block.
        for stem, section in skill_sections.items():
            if not isinstance(section, dict):
                continue
            if _apply(section, _env_params(stem)):
                changed = True
    else:
        # Single-skill example: inject the primary skill flat.
        stem = ''
        theory = config.get('theory')
        if isinstance(theory, (list, tuple)) and theory:
            stem = str(theory[0])
        elif isinstance(theory, str) and theory.strip():
            stem = theory.split()[0]
        if not stem:
            skill_name = config.get('skill_name', '')
            if isinstance(skill_name, str) and skill_name:
                stem = skill_stem(skill_name)
        if stem and _apply(config, _env_params(stem)):
            changed = True

    if changed:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


# Shared topology / force-field fixture suffixes staged from a skill's
# tests/ directory into the example run directory (same contract as
# certification.converter.generate_test_configs).
_TOPOLOGY_SUFFIXES = ('.data', '.in.init', '.in.settings',
                      '.init.xyz')

# Beside an example YAML, a training or test data file is a fixture in the
# same sense a LAMMPS topology is: the input names it by bare name and the
# run resolves it in its own directory. The skill branch of the staging
# below keeps the narrower set, because that directory is scanned for
# project-renamed copies and a generated '<workflow>.json' intermediate
# would be swept into that renaming.
_EXAMPLE_FIXTURE_SUFFIXES = _TOPOLOGY_SUFFIXES + (
    '.json', '.charge', '.xyz', '.freq.molden')


def _stage_fixture(src, dst):
    """Put one read-only fixture where the run can resolve it, without a copy.

    The data files a run reads by bare name (a LAMMPS topology, a training
    set) are inputs: the skill backend reads the launch-directory file and
    writes its own copy into the calculation directory it owns, which is the
    copy the engine actually uses. Copying the bytes into every planning
    directory first is what made a full round stage 3456 files / 133 MB
    (measured in #473). A symlink keeps the source in the package, read-only
    by usage, and an OS/filesystem that refuses links falls back to the copy.
    """
    if os.path.lexists(dst):
        return
    try:
        os.symlink(os.path.realpath(src), dst)
    except OSError:
        shutil.copy(src, dst)


def _stage_example_skill_fixtures(config_path, run_dir, example_path=None):
    """Stage skill topology fixtures into the example run directory.

    A skill backend may resolve launch-directory topology files (the
    LAMMPS backend requires '<project>.data' plus the optional
    '<project>.in.init' / '<project>.in.settings' templates). Skill
    certification stages these fixtures from the skill's tests/
    directory; the example flow shares the same fixtures so example
    YAMLs need no data-file copies of their own. Fixtures are copied
    under their original name and, when the injected '<stem>_project'
    (fallback: title) differs, additionally under the project name
    the backend resolves.

    Two sources, in this order. Files shipped beside the example YAML
    (``example_path``) are staged first, under their own name: they are
    the topologies the skill's tests/ directory does not carry, and an
    example that names such a project in its own skill section has to
    find it. A project-renamed copy is then written only when nothing of
    that name is already in the run directory, because renaming every
    staged fixture would otherwise overwrite the real one with another
    system's topology (h2o content landing as ethanol.data, which the
    backend reads as a 60-atom box for a 9-atom molecule).

    A skill shipping no such fixture turns this into a no-op, and any
    resolution failure is tolerated: the runner then reports the
    missing topology as an ordinary test failure.

    Parameters
    ----------
    config_path : str
        Path to the example flat .json config (after skill env
        injection, so '<stem>_project' is already resolved).
    run_dir : str
        Example run directory shared by the three execution modes.
    example_path : str, optional
        Path of the source example YAML; its directory supplies the
        topology fixtures the skill does not ship.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(config, dict):
        return

    import importlib

    wanted = set()
    if example_path:
        wanted.add(os.path.splitext(os.path.basename(example_path))[0])
    title = str(config.get('title') or '').strip()
    if title:
        wanted.add(title)
    for key, value in config.items():
        if key.endswith('_project') and str(value or '').strip():
            wanted.add(str(value).strip())
    sections = config.get('skill_sections')
    if isinstance(sections, dict):
        for section in sections.values():
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if key.endswith('_project') and str(value or '').strip():
                    wanted.add(str(value).strip())

    text = ''
    if example_path:
        # Beside the example YAML sit the topologies every example shares a
        # directory with, and copying all of them into every run staged a
        # LAMMPS input deck into an xtb or cp2k run that never reads one. Round
        # tmp_20261016_075328 showed what that costs beyond the bytes: the
        # runner's minimum-image check reads `**/*.in.init` under the launch
        # directory to find the cutoff the engine used, so it reported
        # `[PBC] pair cutoff 10.000 A exceeds half of the 10.000 A cell` inside
        # a passing xtb leg - a statement about a force field this run does not
        # have. Stage only what this example can resolve by name: its own
        # stem, its title, and the project each skill section declares.
        example_dir = os.path.dirname(os.path.abspath(example_path))
        try:
            with open(example_path, 'r', encoding='utf-8') as handle:
                text = handle.read()
        except OSError:
            text = ''
        # A pretrained model is a directory, not a file: an adaptive
        # example names it by `modeldir` and the run reads it from its
        # own directory. Copy it whole, under the same basename. The
        # value is read with the parser the run itself uses, so a quoted
        # spelling means what it means there and a commented-out line
        # stays a comment.
        for value in _declared_modeldirs(example_path):
            value = value.strip().rstrip('/')
            if value in ('', '.', './'):
                continue
            source = os.path.join(example_dir, os.path.basename(value))
            if not os.path.isdir(source):
                continue
            target = os.path.join(run_dir, os.path.basename(value))
            if not os.path.exists(target):
                shutil.copytree(source, target)
        for fname in sorted(os.listdir(example_dir)):
            if not fname.endswith(_EXAMPLE_FIXTURE_SUFFIXES):
                continue
            # A fixture is addressed by the part before the first dot:
            # `water25.data`, `md_local_example.init.xyz`, `egs_small.json`.
            base = fname.split('.')[0]
            # Named by the config (its own stem, its title, or a skill
            # section's project) or by the example's own text - a training
            # file is named by `train_data:` rather than by a project, and the
            # engine reads it from the launch directory under that name.
            if base not in wanted and not re.search(
                    r'\b%s\b' % re.escape(base), text):
                continue
            src = os.path.join(example_dir, fname)
            if os.path.isfile(src):
                _stage_fixture(src, os.path.join(run_dir, fname))

    theory = config.get('theory')
    if isinstance(theory, (list, tuple)):
        stems = [str(t) for t in theory if t]
    elif isinstance(theory, str) and theory.strip():
        stems = theory.split()
    else:
        stems = []
    if not stems:
        skill_name = str(config.get('skill_name', ''))
        if skill_name:
            stems = [skill_stem(skill_name)]

    skill_sections = sections if isinstance(sections, dict) else {}

    for stem in stems:
        try:
            query = importlib.import_module(
                skill_module_path(f'{stem}_skill'))
        except ImportError:
            continue
        tests_dir = os.path.join(os.path.dirname(query.__file__), 'tests')
        if not os.path.isdir(tests_dir):
            continue
        fixtures = [
            fname for fname in sorted(os.listdir(tests_dir))
            if fname.endswith(_TOPOLOGY_SUFFIXES)
            and os.path.isfile(os.path.join(tests_dir, fname))
        ]
        if not fixtures:
            continue

        section = skill_sections.get(stem)
        if not isinstance(section, dict):
            section = {}
        project = str(config.get(f'{stem}_project')
                      or section.get(f'{stem}_project')
                      or config.get('title') or '')
        for fname in fixtures:
            # Same name rule as the example's own directory: stage the
            # topology this run can resolve (its project, title, stem, or a
            # bare name its text mentions), not every fixture the skill
            # ships. The backend copies the file it resolves into its own
            # calculation directory, so only that file has to be reachable.
            base = fname.split('.')[0]
            if example_path and base not in wanted and not re.search(
                    r'\b%s\b' % re.escape(base), text):
                continue
            src = os.path.join(tests_dir, fname)
            _stage_fixture(src, os.path.join(run_dir, fname))
            if not project:
                continue
            suffix = next(s for s in _TOPOLOGY_SUFFIXES
                          if fname.endswith(s))
            target = f'{project}{suffix}'
            if target != fname and not os.path.lexists(
                    os.path.join(run_dir, target)):
                _stage_fixture(src, os.path.join(run_dir, target))


def _example_family(workflow):
    """Family name of a chained example, or None when it stands alone.

    Parameters
    ----------
    workflow : str
        Example stem ('nn_train').

    Returns
    -------
    str or None
        The family whose directory the example shares, or None.
    """
    for family, members in EXAMPLE_CHAINS.items():
        if workflow in members:
            return family
    return None


def _order_example_files(filenames):
    """Order example files so a chain runs in its declared sequence.

    A directory listing is alphabetical, and alphabet would put
    `nn_predict` before `nn_train` -- the consumer before the fit that
    writes what it reads. Chain members are therefore laid down in the
    order `EXAMPLE_CHAINS` states them, at the position of their earliest
    member; everything else keeps the alphabetical order it always had.

    Parameters
    ----------
    filenames : iterable of str
        File names discovered in the examples directory.

    Returns
    -------
    list of str
        File names in execution order.
    """
    present = [f for f in sorted(filenames) if f.endswith(('.yaml', '.yml'))]
    stems = [os.path.splitext(f)[0] for f in present]
    by_stem = dict(zip(stems, present))

    ordered = []
    placed = set()
    for stem in stems:
        family = _example_family(stem)
        if family is None:
            ordered.append(by_stem[stem])
            placed.add(stem)
            continue
        for member in EXAMPLE_CHAINS[family]:
            if member in by_stem and member not in placed:
                ordered.append(by_stem[member])
                placed.add(member)
    return ordered


@lru_cache(maxsize=None)
def _declared_modeldirs(example_path):
    """Every `modeldir` an example declares, read through the parser.

    Walks the parsed configuration rather than the raw text: a quoted
    value keeps its spaces, a comment is not a declaration, and a key
    nested in any section is found the same way the run finds it.

    Cached per path: the example staging calls this once per carrier
    (yaml and classic input), and `parse_input` runs the full interface
    pipeline plus skill discovery for one answer that cannot change
    within a round. This call was the largest cost of the example stage
    once the byte copies became links (measured in #473).

    Parameters
    ----------
    example_path : str
        Path to the example YAML.

    Returns
    -------
    tuple of str
        Declared model directory values, in document order where the
        parse preserves it.
    """
    from ..interface.parser import parse_input

    try:
        parsed = parse_input(example_path)
    except (OSError, ValueError, TypeError):
        return ()
    found = []
    stack = [parsed]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            value = node.get('modeldir')
            if isinstance(value, str) and value.strip():
                found.append(value)
            stack.extend(item for item in node.values()
                         if isinstance(item, (dict, list)))
        elif isinstance(node, list):
            stack.extend(item for item in node
                         if isinstance(item, (dict, list)))
    return tuple(found)


def _example_submits_remotely(yaml_path):
    """Whether an example asks the framework to submit its children.

    Read from the example's own input, not from a name convention: the same
    question is answered by `&search remotetask` on one workflow and by
    `&md remotetask` on another, and an example that never says it is a local
    run whether it fails or is skipped.

    Parameters
    ----------
    yaml_path : str
        The example YAML as shipped.

    Returns
    -------
    bool
        True when any input section declares `remotetask` truthy.
    """
    try:
        with open(yaml_path, 'r') as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict):
        return False
    for section in raw.values():
        if isinstance(section, dict) and str(
                section.get('remotetask', '')).strip().lower() in (
                    'true', '1', 'yes'):
            return True
    return False


def _example_skill_registry():
    """Skill metadata of this host, or an empty map when discovery fails.

    The requires gate below is a convenience, not a correctness check: a
    registry that cannot be read must not stop the whole example plan, so
    the failure falls back to "no requirements known".
    """
    try:
        from pyrai3md.skills.query import discover_all
        return discover_all() or {}
    except Exception:
        return {}


def _example_missing_requirements(theory, skill_registry):
    """Optional packages this example's engines need and this host lacks.

    An example whose engine stack is absent is reported as a named skip,
    the way a skill's certification cases are: a red run on a machine that
    never had TensorFlow says nothing about the code, and a silent pass
    would say less.

    Parameters
    ----------
    theory : str
        The example's `control.theory` as one space-separated string.
    skill_registry : dict
        Skill metadata from `discover_all()`, keyed by program label.

    Returns
    -------
    list of str
        Missing requirement names, sorted and deduplicated.
    """
    from pyrai3md.qualification.certification.planner import (
        _missing_requirements,
    )

    missing = []
    for label in str(theory or '').split():
        meta = skill_registry.get(label)
        if not meta:
            continue
        for name in _missing_requirements(skill_registry,
                                         meta.get('skill_name', '')):
            if name not in missing:
                missing.append(name)
    return sorted(missing)


_PLAN_WORKER = {}


def _init_plan_worker(config, export_vars, shell_commands, example_timeout,
                      run_dir):
    """Give a planning worker the round's shared settings once."""
    _PLAN_WORKER.update(config=config, export_vars=export_vars,
                        shell_commands=shell_commands,
                        example_timeout=example_timeout, run_dir=run_dir)


def _plan_example_item(items):
    """Plan one batch of examples that share no writable directory.

    A chain family is one batch, so its members keep using the family
    directory sequentially; a standalone example is a batch of one and
    can run in any order.
    """
    state = _PLAN_WORKER
    tasks = []
    for workflow, yaml_path, task_name, theory, title in items:
        tasks.extend(_plan_example_modes(
            workflow, yaml_path, task_name, theory, title,
            state['run_dir'], state['config'], state['export_vars'],
            state['shell_commands'], state['example_timeout']))
    return tasks


def _plan_example_tests(config, work_dir, examples_dir=None):
    """Plan example qualification tasks from examples/*.yaml.

    The examples/ directory is the single source of truth for
    official workflow YAMLs. Qualification never maintains its own
    input files: it discovers the example YAMLs and renders the classic
    text input from them through the converter, so every mode reads the
    same effective configuration:

        example YAML -> flat JSON config -> classic input + xyz

    Examples are qualified through the two CLI entry points that take a
    user-authored input file:

        pyrai3md <yaml>         (YAML input pipeline)
        pyrai3md <input>        (classic text input pipeline)

    The third entry point, pyrai3md_agent <config> (sectioned dict JSON),
    is deliberately not planned for examples: its input is a machine
    rendered artifact rather than user input, an example is a framework
    computation task rather than an agent-calling-skill scenario, and
    running it costs a third of the suite for no extra coverage of the
    parser layer. The agent loader pipeline stays qualified through the
    skill tests, which run all three modes.

    Current qualification policy: every example the package ships is
    planned (the list is read from `pyrai3md/examples/`), and only a case
    named in POSTPONED_EXAMPLES is reported as postponed instead. The
    qualified dynamics examples (md_*_fssh, md_lammps_*,
    multicenter_md) are disabled by default through their
    'examples' switches in pmd_env.yaml because they require the
    corresponding electronic-structure / MD software on the host.

    Four switch levels decide a qualified example, in this order: the
    three-valued '_runall' master, the three-valued family switch of the
    example's family ('md_lammps_test' for md_lammps_*, 'constraint_md_test'
    for constraint_md_*, 'optimize_test' for optimize_*, 'path_test' for
    path_*), then the
    per-example boolean. A new 'optimize_*.yaml' joins every optimize round
    by its name alone: the family does not bypass the postponement policy,
    which is decided first. A switch set to
    'condition' is the only setting that lets the level below speak, so
    'true' opens a whole family whatever its cases say, 'false' closes one,
    and 'condition' leaves each case in charge of itself. The postponed list
    above outranks all three, and a skipped task always names the switch that
    closed it.

    Example runs are isolated in a work_dir/examples/ subfolder so
    scratch outputs never pollute the repository or the caller cwd.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.
    work_dir : str
        Working directory for generated scripts and outputs.
    examples_dir : str or None
        Directory to discover the example YAMLs in. None (the only value
        production uses) means the examples/ shipped next to this package.

    Returns
    -------
    list of dict
        List of example qualification task descriptors.
    """
    tasks = []

    # In-package examples/ discovered via package-relative path
    # (pyrai3md/qualification -> pyrai3md/examples). The parameter exists
    # for the qualification's own tests, which need a directory they can
    # plant an unqualified YAML in; production never passes it.
    if examples_dir is None:
        examples_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, 'examples'
        ))
    if not os.path.isdir(examples_dir):
        return tasks

    export_vars = _merge_exports(config)
    shell_commands = _get_shell_commands(config)
    run_dir = os.path.join(work_dir, 'examples')
    os.makedirs(run_dir, exist_ok=True)

    # Per-example switches from pmd_env.yaml; a missing section or a
    # missing key keeps the example enabled (backward compatible).
    example_flags = config.get('examples', {}) or {}
    _warn_unknown_example_keys(
        example_flags,
        {os.path.splitext(fname)[0]
         for fname in os.listdir(examples_dir)
         if fname.endswith(('.yaml', '.yml'))})

    # Three levels, decided in this order: the `_runall` master, the example's
    # family switch (`md_lammps_test`, `constraint_md_test`, `optimize_test`),
    # then the per-example boolean. Each of the first two takes three values, where
    # `condition` is the only one that lets the level below speak; the
    # POSTPONED_EXAMPLES policy below outranks all three.
    master = _runall_mode(config)
    families = _family_states(config)

    skill_registry = _example_skill_registry()
    example_timeout = _resolve_timeout(config)

    # The plan is built per example; the text work (YAML parse, carrier
    # rendering, fixture staging) is independent except inside a chain
    # family, whose members share a directory. Slots keep the plan order
    # while the work runs in workers.
    slots = []
    family_slot = {}

    for fname in _order_example_files(os.listdir(examples_dir)):
        workflow = os.path.splitext(fname)[0]
        yaml_path = os.path.join(examples_dir, fname)
        task_name = f'example_{workflow}'
        theory, title = _read_example_meta(yaml_path)

        if workflow in POSTPONED_EXAMPLES:
            slots.append({'task': _example_skip(
                task_name, theory, yaml_path,
                'Qualification postponed: %s' % POSTPONED_EXAMPLES[workflow])})
            continue

        # Which switch closed this case, and therefore what the operator has
        # to change to open it again: the reason names the key rather than
        # saying "disabled" of a plan with three places it could come from.
        # _coerce_switch/_coerce_example_flag normalize strings so a typo like
        # 'fasle' cannot silently keep or drop the example.
        state, closer = _example_state(master, families, workflow)
        if state == SWITCH_CONDITION:
            if not _coerce_example_flag(
                    example_flags.get(workflow, True), workflow):
                state, closer = False, 'examples.%s' % workflow
        if state is False:
            slots.append({'task': _example_skip(
                task_name, theory, yaml_path, 'disabled by %s' % closer)})
            continue

        # A cluster client the host does not have is the same kind of named
        # skip: the example really does submit, and reporting that as a
        # failure would say "the framework is broken" where the sentence
        # should be "this machine has no scheduler".
        if _example_submits_remotely(yaml_path) and shutil.which('sbatch') is None:
            slots.append({'task': _example_skip(
                task_name, theory, yaml_path,
                'example submits its child runtimes to Slurm, '
                'but sbatch is not available on this host')})
            continue

        # An engine stack the host does not have is a named skip, not a
        # failure: the example is planned the same way everywhere, and the
        # reason says what to install.
        missing = _example_missing_requirements(theory, skill_registry)
        if missing:
            slots.append({'task': _example_skip(
                task_name, theory, yaml_path,
                'example needs %s, which is not importable on this host'
                % ', '.join(missing))})
            continue

        entry = (workflow, yaml_path, task_name, theory, title)
        family = _example_family(workflow)
        if family and family in family_slot:
            family_slot[family]['items'].append(entry)
        else:
            slot = {'items': [entry], 'result': None}
            if family:
                family_slot[family] = slot
            slots.append(slot)

    work = [slot for slot in slots if 'items' in slot]
    workers = min(_planning_worker_count(config), max(1, len(work)))
    if workers > 1 and len(work) >= _PLAN_PARALLEL_MIN_ITEMS:
        from multiprocessing import get_context
        initargs = (config, export_vars, shell_commands, example_timeout,
                    run_dir)
        with get_context('spawn').Pool(
                workers, initializer=_init_plan_worker,
                initargs=initargs) as pool:
            results = pool.map(_plan_example_item,
                               [slot['items'] for slot in work])
        for slot, result in zip(work, results):
            slot['result'] = result
    else:
        _init_plan_worker(config, export_vars, shell_commands,
                          example_timeout, run_dir)
        for slot in work:
            slot['result'] = _plan_example_item(slot['items'])

    for slot in slots:
        if 'task' in slot:
            tasks.append(slot['task'])
        else:
            tasks.extend(slot['result'])

    return tasks


def _example_skip(task_name, theory, yaml_path, reason):
    """One example-level task that reports why the example did not run.

    Every skip is the same descriptor with a different sentence, so the
    reader of a suite report sees a task named after the example and a
    reason that names the switch or the missing host component.
    """
    return {
        'name': task_name,
        'skill': 'examples',
        'mode': 'pyrai3md_yaml',
        'theory': theory,
        'input_file': yaml_path,
        'skip': True,
        'skip_reason': reason,
    }


def _plan_example_modes(workflow, yaml_path, task_name, theory, title,
                        run_dir, config, export_vars, shell_commands,
                        example_timeout):
    """Render and plan the runnable modes of one enabled example.

    Two CLI entry points take a user-authored input file, and each gets
    its own directory so a failure keeps its own log: the YAML pipeline
    (pyrai3md <yaml>) and the classic text pipeline (pyrai3md <input>).
    The classic input is rendered from the same flat JSON the YAML mode
    reads, so both modes exercise one effective configuration.
    """
    tasks = []

    # Each mode of each example gets its own directory, and this is
    # the whole point of the change. Twenty-four example tasks used
    # to share one: every backend wrote its engine log and its own
    # timing table into <run_dir>/tmp_LAMMPS, and both modes of an
    # example wrote the same <title>.md.* products, so the last task
    # of the suite overwrote the evidence of the ones that had
    # already failed and the input run overwrote the yaml run of the
    # same example. A directory per (example, mode) makes a failure
    # keep its own log, and it puts each task's calculation directory
    # under its own working directory by the same move.
    modes = (('yaml', 'pyrai3md_yaml', yaml_path),
             ('input', 'pyrai3md_input', None))
    parsed_config = None
    for mode_tag, mode_name, mode_file in modes:
        # A chain shares one directory so its later member can read
        # what its earlier member wrote; every other example keeps a
        # directory per (example, mode).
        family = _example_family(workflow)
        mode_dir = os.path.join(run_dir, family or workflow, mode_tag)
        os.makedirs(mode_dir, exist_ok=True)

        # YAML -> flat json (shared with the skill-test converter);
        # the classic input is derived from it for mode 2. The agent
        # .config is no longer rendered here.
        #
        # Both carriers read the same YAML and produce the same flat
        # config: parse it once and copy the file for the second carrier.
        # The parse runs the runtime input pipeline and was the largest
        # single cost of a full-round plan (60 s of 130 s, measured on
        # #473), and running it twice changed nothing but the path.
        config_path = os.path.join(mode_dir, f'{workflow}.json')
        if parsed_config is None:
            yaml_to_config(yaml_path, config_path)
            parsed_config = config_path
        else:
            shutil.copy(parsed_config, config_path)

        # Share the pmd_env.yaml skill settings into the example
        # config so the generated .input carries each skill
        # section's machine parameters (path/nproc/shell/
        # execution_mode/...), exactly like the skill-test flow.
        _inject_example_skill_env(config_path, config.get('skills', {}))

        # Stage shared skill topology fixtures (e.g. the LAMMPS h2o
        # water box) into the run directory so every mode finds the
        # launch-directory files its backend resolves. The example's
        # own directory goes with it: a project the skill's tests/
        # directory does not ship (ethanol) lives beside the YAML.
        _stage_example_skill_fixtures(config_path, mode_dir, yaml_path)

        if mode_file is None:
            # Mode 2: pyrai3md <input> (classic text + xyz).
            mode_file = os.path.join(mode_dir, f'{workflow}.input')
            convert_to_classic_input(config_path, mode_file)

        script = _write_shell_script(
            mode_dir, f'{task_name}_{mode_tag}',
            f'cd {mode_dir} && pyrai3md {mode_file}', export_vars,
            shell_commands, timeout=example_timeout,
        )
        tasks.append({
            'name': f'{task_name}_{mode_tag}',
            'skill': 'examples',
            'mode': mode_name,
            'theory': theory,
            # One job per example carrier: the yaml and input runs have
            # separate directories, so only a chain's two examples (which
            # share the mode's directory) are a necessary group.
            'job_group': 'example_%s_%s' % (family or workflow, mode_tag),
            'input_file': mode_file,
            # The source YAML the run was planned from. The input
            # mode's 'input_file' is the generated classic input, so
            # a post-run check that reads the declared ensemble from
            # 'input_file' sees no YAML and judges nothing -- which is
            # how all twelve input-mode example tasks came to be
            # reported as passing with no verdict ever run on them.
            'spec_file': yaml_path,
            'run_dir': mode_dir,
            'title': title,
            'script': script,
            'use_slurm': False,
            'timeout': example_timeout,
        })

    return tasks


def _read_example_meta(yaml_path):
    """Read the theory and title from an example workflow YAML.

    The theory comes from the control.theory list (e.g. [pyscf]
    -> 'pyscf', [molcas, openqp] -> 'molcas/openqp') so test output can
    show which electronic-structure method each example uses. The title
    comes from control.title and names every artifact the run writes
    (<title>.log, <title>.md.energies, ...); it deliberately differs
    from the YAML file stem (md_orca_fssh.yaml carries
    title: md_orca_fssh_example), so the runner cannot derive it from
    the input path.

    Parameters
    ----------
    yaml_path : str
        Path to the example YAML file.

    Returns
    -------
    tuple of (str, str)
        '/'-joined theory names and the control.title, each empty
        ('unknown' for theory, '' for title) when unreadable.
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        control = data.get('control') or {}
        theory = control.get('theory', [])
        if isinstance(theory, str):
            theory = [theory]
        title = str(control.get('title') or '')
        theory_string = '/'.join(str(t) for t in theory) if theory else 'unknown'
        return theory_string, title
    except Exception:
        return 'unknown', ''



# The execution script embeds `timeout <N>s` where N is the task's
# subprocess timeout minus this margin, so the script kills the whole
# engine process group just before the runner's subprocess backstop
# fires. This guarantees a timed-out calculation leaves no residual
# processes (which would otherwise slow every subsequent test).
SCRIPT_TIMEOUT_MARGIN = 5


def _resolve_timeout(config):
    """Read the per-test execution timeout (_timeout) from config.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    int
        The _timeout value in seconds (default 300 when unset).
    """
    try:
        return int(config.get('_timeout', 300) or 300)
    except (TypeError, ValueError, AttributeError):
        return 300


# A small plan stays in one process: spawning a pool costs more than the
# text work when only a handful of examples are runnable.
_PLAN_PARALLEL_MIN_ITEMS = 8


def _planning_worker_count(config):
    """Worker processes for the example planning phase.

    `environment.planning_workers` is the site's answer: 0 means auto
    (at most eight, never more than the host has), 1 forces the serial
    path, and a larger value is honoured as written. The plan itself does
    not depend on this number; only how long it takes to build does.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    int
        Worker count, at least 1.
    """
    raw = (config.get('environment', {}) or {}).get('planning_workers', 0)
    try:
        requested = int(raw or 0)
    except (TypeError, ValueError):
        requested = 0
    if requested <= 0:
        requested = min(8, os.cpu_count() or 1)
    return max(1, requested)


def _write_shell_script(output_dir, name, command, export_vars=None,
                        shell_commands=None, timeout=None):
    """Generate a .sh local execution script.

    Parameters
    ----------
    output_dir : str
        Directory to write the script.
    name : str
        Script name (without extension).
    command : str
        Command to execute.
    export_vars : dict, optional
        Environment variables to export.
    shell_commands : list, optional
        Raw shell commands (from environment.shell) copied verbatim
        before the export section.
    timeout : int, optional
        Task subprocess timeout in seconds. When set, the command is
        wrapped in `timeout <timeout - SCRIPT_TIMEOUT_MARGIN>s bash -c`
        so the engine process group is killed just before the runner's
        subprocess backstop, leaving no residual processes. None (the
        default) writes the command verbatim.

    Returns
    -------
    str
        Absolute path to the generated script.
    """
    script_path = os.path.join(output_dir, f'{name}.sh')
    lines = ['#!/bin/bash', '']
    if shell_commands:
        lines.extend(shell_commands)
        lines.append('')
    if export_vars:
        for key, value in export_vars.items():
            lines.append(f'export {key}={value}')
        lines.append('')
    if timeout:
        script_timeout = max(1, int(timeout) - SCRIPT_TIMEOUT_MARGIN)
        command = (f'timeout -k 10s {script_timeout}s '
                   f'bash -c {shlex.quote(command)}')
    lines.append(command)
    lines.append('')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return script_path


def _write_slurm_script(output_dir, name, command, slurm_config,
                        export_vars=None, shell_commands=None):
    """Generate a .sbatch slurm submission script.

    Parameters
    ----------
    output_dir : str
        Directory to write the script.
    name : str
        Script name (without extension).
    command : str
        Command to execute.
    slurm_config : dict
        Slurm parameters from pmd_env.yaml.
    export_vars : dict, optional
        Environment variables to export.
    shell_commands : list, optional
        Raw shell commands (from environment.shell) copied verbatim
        before the export section.

    Returns
    -------
    str
        Absolute path to the generated script.
    """
    script_path = os.path.join(output_dir, f'{name}.sbatch')
    content = build_slurm_script(
        slurm_config, export_vars or {}, command, shell_commands,
    )
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return script_path


def _get_shell_commands(config):
    """Collect raw shell commands from environment.shell.

    The commands (module loads, activation lines, ...) are copied
    verbatim into generated .sh/.sbatch scripts before the export
    section, matching the pmd_env.yaml contract.

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    list
        Non-empty command lines; [] when the section is unset.
    """
    shell = config.get('environment', {}).get('shell')
    fragments = shell if isinstance(shell, list) else [shell]
    commands = []
    for fragment in fragments:
        if fragment is None:
            continue
        # ';' separators support the classic input '&environment shell'
        # single-line form; plain yaml strings pass through unchanged.
        for piece in str(fragment).split(';'):
            piece = piece.strip()
            if piece:
                commands.append(piece)
    return commands


def _merge_exports(config):
    """Merge environment export variables from configuration.

    The export section contains variable names mapped to lists
    of values. This function merges the value lists into
    single string values (comma-separated for multi-value lists).

    Parameters
    ----------
    config : dict
        Parsed pmd_env.yaml configuration.

    Returns
    -------
    dict
        Merged environment variables {name: value_string}.
    """
    env_section = config.get('environment', {})
    exports = env_section.get('export', {})
    merged = {}

    for key, values in exports.items():
        if isinstance(values, list):
            merged[key] = ','.join(str(v) for v in values)
        else:
            merged[key] = str(values)

    return merged


def build_slurm_script(slurm_config, export_vars, command,
                       shell_commands=None):
    """Build a slurm batch script for a test command.

    The `slurm:` section is rendered through the framework's single
    dialect (`kernel.sbatch`): snake_case keys become the canonical
    hyphenated directives, `extra` is copied verbatim, and a key that maps
    to no directive is refused by name. This is the same dialect the
    dispatch job-script builders use; only the wrapped command differs.
    Shell commands from environment.shell are copied verbatim after
    the #SBATCH directives, and environment variables from
    export_vars are exported before the command.

    Parameters
    ----------
    slurm_config : dict
        Slurm parameters from pmd_env.yaml.
    export_vars : dict
        Merged environment variables.
    command : str
        The command to execute under slurm.
    shell_commands : list, optional
        Raw shell commands (from environment.shell) copied verbatim
        before the export section.

    Returns
    -------
    str
        Complete sbatch script content.
    """
    lines = ['#!/bin/bash']
    lines.extend(render_sbatch_directives(slurm_config))

    lines.append('')

    if shell_commands:
        lines.extend(shell_commands)
        lines.append('')

    for key, value in export_vars.items():
        lines.append(f'export {key}={value}')

    if export_vars:
        lines.append('')

    lines.append(command)
    lines.append('')

    return '\n'.join(lines)
