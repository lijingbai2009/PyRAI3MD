"""
======================================================================
This module is the skill-specific half of the certification test
input generators (split out of the former monolithic qualification
converter).

It owns everything that depends on the skill identity or the skill
schema:
1. The execution policy keys forwarded into the classic input skill
   section (_EXECUTION_KEYS, _append_execution_keys)
2. The dynamic schema key sync through each skill's public query
   interface (_skill_defaults, _append_schema_keys)
3. The generic skill section builders (_build_skill_pairs,
   _build_skill_section and _build_method_section)
4. The skill test file discovery: locating each skill's tests/
   directory through the package path, listing the test YAML cases
   (list_skill_test_yamls) and the runtime test scripts
   (list_skill_runtime_tests). The runtime scripts stay in the
   skill packages and are executed in place by pytest.

The planner drives the chain: query discovers the skills, the handler
locates the test files, the converter parses the YAML cases into
configs and renders every output format (the individualized parsing
rules live in converter.parse_test_yaml). Skills no longer ship a
dev_tool: new skills only need the standard tests/ layout.

Reference direction: handler imports the generic rendering
helpers from the sibling module converter.py; the reverse direction
(converter.generate_test_configs -> handler.list_skill_test_yamls)
is only resolved lazily by design.

Aug 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import os

from .converter import (_EXECUTION_KEYS, _append_execution_keys,
                        _collapse_shell_line, _derive_project_name,
                        _render_section)


def _skill_label(skill_name):
    """Program label of a skill package, used as its input section header.

    '&NN' is spelled by the manifest name, not by the folder: the package
    is nn_pes_skill, and a header derived from the folder name would ask
    the parser for a section no input file carries. The skill's own query
    interface answers; a package that cannot be imported keeps the
    historical folder reading.

    Parameters
    ----------
    skill_name : str
        Skill package name (e.g. 'nn_pes_skill').

    Returns
    -------
    str
        Section header stem (e.g. 'nn').
    """
    from ...skills.query import skill_module_path, skill_stem

    fallback = skill_stem(skill_name)
    try:
        module = importlib.import_module(skill_module_path(skill_name))
    except (ImportError, ModuleNotFoundError):
        return fallback
    try:
        return module.get_metadata().get('name') or fallback
    except (AttributeError, KeyError, TypeError):
        # A query module without metadata, or metadata that is not a
        # mapping, falls back to the stem. Any other failure is a defect
        # in that skill's query module and must not be read as "no name".
        return fallback


def _skill_subsections(skill_name):
    """Extra input sections a skill declares through its query interface.

    A skill that reads more than its own '&<stem>' block names those
    blocks in its manifest, and query.get_metadata() hands the list over.
    The framework renders and routes them without knowing what a keyword
    inside one means.

    Parameters
    ----------
    skill_name : str
        Skill identifier (e.g. 'nn_pes_skill').

    Returns
    -------
    list of str
        Declared section names; empty when the skill declares none.
    """
    from ...skills.query import skill_module_path

    try:
        module = importlib.import_module(skill_module_path(skill_name))
    except (ImportError, ModuleNotFoundError):
        return []
    getter = getattr(module, 'get_subsections', None)
    if getter is None:
        return []
    try:
        return list(getter())
    except (TypeError, ValueError):
        # A declaration that is not iterable (or holds a non-integer where
        # a name belongs) is an empty declaration; a failure inside the
        # skill's own code is not swallowed.
        return []


def _skill_subsection_values(config, skill_name, section_params=None):
    """Collect the declared sub-sections one skill carries.

    Parameters
    ----------
    config : dict
        Agent JSON config.
    skill_name : str
        Skill identifier (e.g. 'nn_pes_skill').
    section_params : dict, optional
        This skill's preserved parameter section (multi-theory
        composite); None reads the flat config.

    Returns
    -------
    list of tuple
        (section name, keyword dict) in the order the skill declared.

    Raises
    ------
    ValueError
        For a value the classic text cannot carry: a nested list has no
        unambiguous spelling on one line, and dropping it silently would
        hand the two entry points different configurations.
    """
    params = section_params if section_params is not None else config
    collected = []
    for name in _skill_subsections(skill_name):
        values = params.get(name)
        if not isinstance(values, dict) or not values:
            continue
        for key in sorted(values):
            value = values[key]
            if isinstance(value, (list, tuple)) and any(
                    isinstance(item, (list, tuple)) for item in value):
                raise ValueError(
                    "sub-section &%s keyword %r carries a nested list; the "
                    "classic input text spells one value per line, so write "
                    "it as a string ('1 1;1 10')" % (name, key))
        collected.append((name, dict(values)))
    return collected


def build_subsection_blocks(config, skill_name, section_params=None):
    """Render the declared sub-sections of one skill as input text.

    Each named block becomes its own '&NAME' section, so a skill whose
    parameters live beside its main block round-trips between the yaml,
    the agent config and the classic input text.

    Parameters
    ----------
    config : dict
        Agent JSON config.
    skill_name : str
        Skill identifier (e.g. 'nn_pes_skill').
    section_params : dict, optional
        This skill's preserved parameter section (multi-theory
        composite); None reads the flat config.

    Returns
    -------
    list of str
        Rendered sections, in the order the skill declared them.
    """
    blocks = []
    for name, values in _skill_subsection_values(
            config, skill_name, section_params):
        pairs = [(key, values[key]) for key in sorted(values)]
        blocks.append(_render_section(name, pairs))
    return blocks


def _skill_defaults(skill_name):
    """Load the skill schema default keywords via its query interface.

    Every skill exposes query.get_defaults() returning the full
    Calculation keyword table (schema/reference DEFAULT). The
    converter uses this key set to decide which config keys belong
    to the skill section, so new schema keywords are picked up
    without converter changes.

    Parameters
    ----------
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').

    Returns
    -------
    dict
        Skill default keyword mapping.

    Raises
    ------
    Exception
        Whatever the skill's own query module raises. A broken default
        table must fail the plan row that needs it instead of rendering
        a section with none of that skill's keywords.
    """
    from ...skills.query import skill_module_path

    module = importlib.import_module(skill_module_path(skill_name))
    return module.get_defaults()


def _append_schema_keys(pairs, source, skill_name, handled):
    """Append config keys belonging to the skill schema.

    Skill test YAMLs only use keywords supported by the skill schema,
    and dev_tool copies them flat into the agent config. Every such
    key present in the source is matched against the skill schema
    defaults and written into the skill section, so both outputs
    carry the complete skill configuration (e.g. maxit_md for
    dynamics tests). Keys already emitted by the explicit builder
    logic are skipped via the handled set.

    The ``source`` is the parameter dict of this skill: the flat config
    for a single-skill job, or that skill's preserved section for a
    multi-theory composite (multi-center ONIOM), so the high and low
    level each read their own keywords.

    Flat int lists owned by the skill (e.g. target_states) are kept
    as lists in the pairs; the text renderer joins them into
    space-separated tokens and the schema adapter converts them back
    to lists on the way in. Nested lists and molecule-level lists
    remain intentionally skipped: text input values are stored as raw
    strings downstream and would fail the skill schema list
    validation, and molecule-level lists (ci/spin/ci_roots/
    ci_spin/coupling/state_spin) belong to &MOLECULE where they are
    derived from state_spin.

    Parameters
    ----------
    pairs : list of tuple
        Skill section (key, value) pairs to extend in place.
    source : dict
        This skill's parameter dict (flat config or preserved section).
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').
    handled : set
        Keys already written (or intentionally owned) by the
        explicit builder logic above.
    """
    defaults = _skill_defaults(skill_name)
    if not defaults:
        return

    skip = set(handled) | set(_EXECUTION_KEYS)
    # Molecule-level lists stay owned by &MOLECULE (derived there)
    molecule_lists = {'ci', 'spin', 'ci_roots', 'ci_spin', 'coupling',
                      'state_spin'}
    for key in sorted(defaults):
        if key in skip or key not in source:
            continue
        value = source[key]
        if key == 'shell':
            # Classic input is line-based: collapse list/multi-line
            # shells into a single ';'-separated line
            shell_line = _collapse_shell_line(value)
            if shell_line:
                pairs.append((key, shell_line))
            continue
        if isinstance(value, (list, tuple)):
            # Skill-owned flat int lists pass through as lists (the
            # text renderer joins them into space-separated tokens)
            if (key not in molecule_lists and value
                    and all(isinstance(v, int) and not isinstance(v, bool)
                            for v in value)):
                pairs.append((key, list(value)))
            continue  # Nested/molecule-level lists stay unwritten
        if not isinstance(value, (int, float, str, bool)):
            continue
        if value is None or value == '':
            continue
        pairs.append((key, value))


def _build_skill_pairs(config, skill_name, section_params=None):
    """Assemble the skill section as (header, keyword pairs).

    The pairs are the single source for both the classic input text
    rendering (&STEM) and the pyrai3md config serialization (the
    section is keyed by the full skill name, e.g. 'xtb_skill', so
    multi-backend agent configs stay distinguishable per skill).

    The section is derived entirely from the skill identity carried
    by skill_name and the skill's own query interface -- no skill
    keyword is hardcoded here:

    - '{stem}_path' line from the config (injected by the planner
      from pmd_env.yaml), falling back to the '{STEM}' environment
      variable.
    - '{stem}_project' line from the config (dev_tool injects the
      per-skill project key), falling back to the derived project
      name.
    - method/basis, scratch_path and execution policy keys.
    - Every remaining config key declared in the skill schema
      defaults (query.get_defaults()) through _append_schema_keys.

    ``section_params`` is this skill's own parameter dict for a
    multi-theory composite (multi-center ONIOM): when given, the
    calculation keywords (method/basis/scratch_path and the schema
    keys) are read from it instead of the flat config, so the high and
    low level do not overwrite each other. The per-stem project/path
    keys and the execution policy keys stay read from the shared
    config. When None (single-skill jobs and skill tests) the flat
    config is the source, preserving the historical behavior.

    Parameters
    ----------
    config : dict
        Agent JSON config.
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').
    section_params : dict, optional
        This skill's preserved parameter section; None reads the flat
        config (single-skill default).

    Returns
    -------
    tuple
        (section_name, pairs): full skill name and (key, value) list.
    """
    stem = _skill_label(skill_name)
    params = section_params if section_params is not None else config

    project_key = f'{stem}_project'
    path_key = f'{stem}_path'

    project = (config.get(project_key) or params.get(project_key)
               or _derive_project_name(config))
    path = (config.get(path_key) or params.get(path_key)
            or os.environ.get(stem.upper(), ''))

    pairs = [
        (stem, path),
        (project_key, project),
    ]

    # Calculation keys
    method = params.get('method', '')
    if method:
        pairs.append(('method', method))
    basis = params.get('basis', '')
    if basis:
        pairs.append(('basis', basis))

    scratch_path = params.get('scratch_path', config.get('scratch_path', ''))
    if scratch_path:
        pairs.append(('scratch_path', scratch_path))

    _append_execution_keys(pairs, params, config)

    # The dispatch trio of THIS skill's own block: `multitask` opens the
    # pool, `multipool` sizes it, `remotetask` asks for the slurm arm. They
    # are fields of the skill schema (cp2k declares all three because its
    # 'num' strain route fans twelve perturbed points out) and they are
    # deliberately absent from its reference defaults, because a DEFAULT
    # entry would write them into every generated input - so
    # `_append_schema_keys` below never sees them and, until #395, nothing
    # else did either. Measured on round `tmp_20261019_163711`: the yaml-mode
    # cp2k leg with `multipool: 2` took 589 s against its own 821 s serial
    # reading from round 18, while the `.input` leg of the same file took 815 s
    # against 817 s, because the block it was rendered from carried neither
    # line. Emission is presence-only and only on the sectioned path, so a
    # skill that declares no pool renders exactly as before, and the
    # framework's own trio (which `&MD` flattens to the top level) cannot leak
    # in here - `params` is this block, not the shared config.
    if section_params is not None:
        for key in ('multitask', 'remotetask', 'multipool'):
            value = params.get(key)
            if value is not None:
                pairs.append((key, value))

    # Remaining skill schema keys carried by this skill's params (e.g.
    # nact_elec/xms for molcas, tda/scf_maxit for orca, maxit_md
    # for dynamics tests)
    _append_schema_keys(pairs, params, skill_name, {
        'method', 'basis', project_key, path_key, 'scratch_path',
        'multitask', 'remotetask', 'multipool',
    })

    return skill_name, pairs


def _build_skill_section(config, skill_name, section_params=None):
    """Build the classic input skill section for any skill.

    Parameters
    ----------
    config : dict
        Agent JSON config.
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').
    section_params : dict, optional
        This skill's preserved parameter section (multi-theory
        composite); None reads the flat config.

    Returns
    -------
    str
        Formatted skill section.
    """
    _name, pairs = _build_skill_pairs(config, skill_name, section_params)
    return _render_section(_skill_label(skill_name), pairs)


def _build_method_section(config, skill_name='molcas_skill',
                          section_params=None):
    """Dispatch to the generic skill section builder.

    Parameters
    ----------
    config : dict
        Agent JSON config.
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').
    section_params : dict, optional
        This skill's preserved parameter section (multi-theory
        composite); None reads the flat config.

    Returns
    -------
    str
        Formatted method section.
    """
    return _build_skill_section(config, skill_name, section_params)


def _skill_tests_dir(skill_name):
    """Locate the tests directory of one skill package.

    Parameters
    ----------
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').

    Returns
    -------
    str or None
        Absolute path of the tests directory, or None when the skill
        package or its tests directory is missing.
    """
    from ...skills.query import skill_module_path

    try:
        skill_pkg = importlib.import_module(skill_module_path(skill_name, ''))
    except (ImportError, ModuleNotFoundError):
        return None
    tests_dir = os.path.join(os.path.dirname(skill_pkg.__file__),
                             'tests')
    return tests_dir if os.path.isdir(tests_dir) else None


def list_skill_test_yamls(skill_name):
    """List the test case YAML files of one skill (sorted).

    Parameters
    ----------
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').

    Returns
    -------
    list of str
        Absolute paths of the test YAML files; empty when the skill
        has no tests directory.
    """
    tests_dir = _skill_tests_dir(skill_name)
    if tests_dir is None:
        return []
    return sorted(
        os.path.join(tests_dir, fname)
        for fname in os.listdir(tests_dir)
        if fname.endswith('.yaml')
    )


def list_skill_runtime_tests(skill_name):
    """List the runtime test scripts of one skill (sorted).

    The runtime tests are the test_*.py unittest/pytest files shipped
    in the skill's tests directory; certification plans them ahead of
    the YAML task tests so skill-internal regressions surface before
    the end-to-end modes run.

    Parameters
    ----------
    skill_name : str
        Skill identifier (e.g. 'molcas_skill').

    Returns
    -------
    list of str
        Absolute paths of the test_*.py files; empty when absent.
    """
    tests_dir = _skill_tests_dir(skill_name)
    if tests_dir is None:
        return []
    return sorted(
        os.path.join(tests_dir, fname)
        for fname in os.listdir(tests_dir)
        if fname.startswith('test_') and fname.endswith('.py')
    )


def list_shared_skill_tests():
    """List the test scripts of the skills layer itself (sorted).

    pyrai3md/skills/tests/ is not a skill's directory: it holds the checks
    that span every skill (the agent calculate() forwarding contract, the
    wrapper's config parsing). :func:`list_skill_runtime_tests` walks a skill
    package's own tests directory, so it finds none of these.

    Returns
    -------
    list of str
        Absolute paths of the test_*.py files; empty when absent.
    """
    try:
        skills_pkg = importlib.import_module('pyrai3md.skills')
    except (ImportError, ModuleNotFoundError):
        return []
    tests_dir = os.path.join(os.path.dirname(skills_pkg.__file__), 'tests')
    if not os.path.isdir(tests_dir):
        return []
    return sorted(
        os.path.join(tests_dir, fname)
        for fname in os.listdir(tests_dir)
        if fname.startswith('test_') and fname.endswith('.py')
    )
