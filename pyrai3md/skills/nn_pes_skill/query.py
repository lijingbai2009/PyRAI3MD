"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides self-contained query capabilities for nn_pes_skill.
It reads the local manifest.yaml to expose available profiles,
capabilities, and methods — without depending on any external framework
module.

Nothing here imports TensorFlow: discovery runs through this module, and
a machine without the optional neural network stack must still be able to
see the skill and be told why it cannot run.

"""

import os

import yaml


def _get_manifest_path():
    """Locate the manifest.yaml within this skill package.

    Returns
    -------
    str
        Absolute path to manifest.yaml.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manifest.yaml')


def load_manifest():
    """Load and parse the local manifest.yaml.

    Returns
    -------
    dict
        Parsed manifest dictionary containing name, backend,
        capabilities, and profiles.
    """
    manifest_path = _get_manifest_path()
    with open(manifest_path, 'r') as f:
        return yaml.safe_load(f)


def get_name():
    """Get the skill name from manifest.

    Returns
    -------
    str
        Program label, e.g. 'nn'.
    """
    manifest = load_manifest()
    return manifest.get('name', 'nn')


def get_metadata():
    """Return skill metadata.

    This is the sole interface for the frontend framework to obtain
    skill identification, path, backend class, and config keys. The
    'requires' entry names the optional third-party packages the skill
    needs, so a planner can report a machine without them as skipped
    instead of silently dropping the skill.

    Returns
    -------
    dict
        Metadata with keys: name, units, skill_path, backend_class,
        supports_pbc, resources, requires, subsections, searchable,
        search_targets.
    """
    manifest = load_manifest()
    return {
        'name': manifest.get('name', 'nn'),
        'units': manifest.get('units', {}),
        'skill_path': manifest.get('backend', {}).get('module', 'backend'),
        'backend_class': manifest.get('backend', {}).get('class', 'NNPESBackend'),
        'supports_pbc': bool(manifest.get('supports_pbc', False)),
        # Compute resources the skill consumes, as declared in its
        # manifest (#306): keyword names, so the framework never has to
        # guess where a run said how many threads it wants.
        'resources': dict(manifest.get('resources') or {}),
        'requires': list(manifest.get('requires', [])),
        'subsections': list(manifest.get('subsections', [])),
        'searchable': get_searchable(),
        'search_targets': get_search_targets(),
        # The result channels a machine-learning call must return, and
        # the folder aliases the scan table prints (#3.6).
        'result_contract': dict(manifest.get('result_contract') or {}),
        'scan_role_aliases': dict(manifest.get('scan_role_aliases') or {}),
    }


def model_dir_name(title, key=''):
    """The model directory name one fit writes.

    The framework plans a grid point and an adaptive iteration by this
    name (and the fetch-side audit looks for it), so the spelling lives
    here: renaming a model folder is a change in this skill, not in
    three callers.

    Parameters
    ----------
    title : str
        Run title.
    key : str
        Point key, or the iteration tag ('iter-2'); empty for a plain
        single fit.

    Returns
    -------
    str
        Directory name, e.g. 'NN-water-iter-2' or 'NN-water-lr0.001'.
    """
    return 'NN-%s-%s' % (title, key) if key else 'NN-%s' % title


def scan_requirements(skill_config):
    """The training facts a grid scan needs, resolved by this skill.

    A scan fits one model per point, so the skill answers four things
    about the parent training job: the trainable profile its method or
    pinned profile names, the training data, the training mode, and how
    many ensemble members one fit builds. The framework asks here
    instead of reading these keys itself, which is what keeps the
    keyword names and the trainable-profile table on this side of the
    boundary.

    Parameters
    ----------
    skill_config : dict
        Flat skill configuration of the parent task.

    Returns
    -------
    dict
        {'profile', 'train_data', 'mode', 'members'}.

    Raises
    ------
    ValueError
        When no trainable profile matches (or several do and none was
        pinned), or when the declared train_mode cannot describe a scan.
    """
    config = dict(skill_config or {})
    targets = get_search_targets()
    pinned = str(config.get('profile') or '')
    method = str(config.get('method') or '')
    if pinned in targets:
        profile = pinned
    else:
        candidates = sorted(name for name, entry in targets.items()
                            if entry.get('method') == method)
        if not candidates:
            raise ValueError(
                "skill declares no trainable profile for method '%s'%s; pin "
                "profile in the skill section with one of: %s"
                % (method, '' if method else ' (method is not set)',
                   ', '.join(sorted(targets))))
        if len(candidates) > 1:
            raise ValueError(
                "method '%s' has more than one trainable profile (%s); write "
                "profile in the skill section to say which one the scan fits"
                % (method, ', '.join(candidates)))
        profile = candidates[0]
    mode = str(config.get('train_mode') or 'training')
    if mode != 'training':
        raise ValueError(
            "a grid scan fits every point from scratch, so train_mode %r "
            "cannot describe a scan: each point's model directory is new, "
            "and a continuation could not apply the architecture the grid "
            "varies anyway (the container updates every hyperparameter "
            "category but 'model'). Continue a trained surface with a "
            "single train run, not with a scan" % mode)
    return {
        'profile': profile,
        'train_data': str(config.get('train_data') or ''),
        'mode': mode,
        'members': int(config.get('mult_nn') or 1),
    }


def get_search_targets():
    """Map each trainable profile to what a grid point has to stamp.

    A scan is planned by a framework that does not know this skill's
    folder layout, so the three facts it needs are declared here: the
    model family a profile fits, the role sections its hyperparameters
    are written into, and the channels its fit reports. All three are
    read from the profile declaration and the method vocabulary rather
    than restated, and the manifest's 'runtype: train' is what marks a
    profile trainable at all.

    Returns
    -------
    dict
        Profile name -> {'method', 'sections', 'channels'}.
    """
    from .schema.method import (PROFILE_ENSEMBLES, ROLE_ENERGY, ROLE_NAC,
                                ROLE_SECTIONS, ROLE_SOC)

    # The channel a role's network reports. An energy family answers
    # energy, a coupling family nac, and a spin-orbit folder one scalar
    # per declared pair, which is what its fit error is measured against.
    role_channels = {ROLE_ENERGY: 'energy', ROLE_NAC: 'nac', ROLE_SOC: 'soc'}

    manifest = load_manifest()
    profiles = manifest.get('profiles') or {}
    targets = {}
    for name, info in profiles.items():
        if not (info or {}).get('runtype') == 'train':
            continue
        ensembles = PROFILE_ENSEMBLES.get(name)
        if not ensembles:
            continue
        targets[name] = {
            'method': (info or {}).get('method', ''),
            'sections': [ROLE_SECTIONS[folder] for folder, _kind in ensembles],
            'channels': [role_channels[folder] for folder, _kind in ensembles],
        }
    return targets


def check_searchable(keyword, value):
    """Reject a candidate value this skill cannot train with.

    A grid point is planned before any process starts, so a value the
    container would choke on mid-fit is reported here instead, where the
    message still names the keyword and the run still costs nothing.

    Parameters
    ----------
    keyword : str
        Declared searchable keyword.
    value : object
        One candidate value.

    Returns
    -------
    object
        The value, unchanged, when the keyword admits it.

    Raises
    ------
    ValueError
        For an undeclared keyword, a value of the wrong shape for its
        kind, a candidate outside a declared choice, or a number the
        hyperparameter vocabulary bounds.
    """
    from .schema.hyper import REG_STYLES, check_value

    entries = get_searchable()
    if keyword not in entries:
        raise ValueError(
            "'%s' is not a searchable hyperparameter of this skill; "
            "declared: %s" % (keyword, ', '.join(sorted(entries))))
    entry = entries[keyword]
    kind = entry['kind']
    is_group = isinstance(value, (list, tuple))

    if kind in ('scalar', 'bool', 'reg', 'text'):
        if is_group:
            raise ValueError(
                "'%s' takes one value per grid point (%s is what a member "
                "group of it means); write each alternative as its own "
                "entry of the candidate list" % (keyword, value))
    elif kind in ('list', 'pairs'):
        if not is_group:
            raise ValueError(
                "'%s' is itself a list, so a candidate of it must be a "
                "list; got %r" % (keyword, value))
    elif not is_group:
        raise ValueError("'%s' expects a candidate list for %r"
                         % (keyword, value))

    if entry['values']:
        if str(value) not in [str(item) for item in entry['values']]:
            raise ValueError(
                "'%s' accepts %s, got %r"
                % (keyword, ' / '.join(str(item)
                                       for item in entry['values']), value))
    if kind == 'reg' and str(value).lower() not in REG_STYLES:
        raise ValueError("regularizer %r is not one of %s"
                         % (value, ', '.join(REG_STYLES)))
    if entry['type'] == 'int':
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("'%s' expects an integer candidate, got %r"
                             % (keyword, value))
    elif entry['type'] == 'float':
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("'%s' expects a number, got %r"
                             % (keyword, value))
    elif entry['type'] == 'bool':
        if not isinstance(value, bool):
            raise ValueError("'%s' expects true or false, got %r"
                             % (keyword, value))

    if kind == 'scalar' and not isinstance(value, (list, tuple, str)):
        check_value(keyword, value)
    return value


def read_training_record(modeldir, members=2):
    """Read a trained folder's fit errors straight off the disk.

    No TensorFlow is imported: a scan ranks hundreds of folders by what
    their fits wrote last, and a qualification run on a machine without
    the optional stack still has to be able to say whether a folder is
    complete.

    Parameters
    ----------
    modeldir : str
        Directory holding one subfolder per ensemble.
    members : int, optional
        Ensemble size.

    Returns
    -------
    dict or None
        The folder record from service/train_record.py, or None when the
        directory does not exist.
    """
    from .service.train_record import read_model_dir
    return read_model_dir(modeldir, members=int(members))


def get_searchable():
    """List the hyperparameters a grid scan may vary.

    Each entry carries the declared type plus the keyword's kind from
    the hyperparameter vocabulary. The kind is read, never declared: it
    is what tells a scan whether writing a list means several ensemble
    members or several candidate values, and a manifest copy of it would
    be a second source of truth for one fact.

    Returns
    -------
    dict
        Keyword name -> {'type': str, 'kind': str, 'values': list}.
        'values' is present only for a declared choice, empty otherwise.
    """
    from .schema.hyper import HYPER_KEYWORDS

    manifest = load_manifest()
    declared = dict(manifest.get('searchable') or {})
    entries = {}
    for keyword, info in declared.items():
        info = info or {}
        hyper = HYPER_KEYWORDS.get(keyword)
        entries[keyword] = {
            'type': str(info.get('type', '')),
            'kind': hyper[0] if hyper else '',
            'values': list(info.get('values') or []),
        }
    return entries


def get_roles():
    """Get the method roles this skill can serve as.

    Returns
    -------
    dict
        Role name -> backend class name or False mapping.
    """
    manifest = load_manifest()
    return manifest.get('roles', {})


def get_capabilities():
    """Get all supported capabilities.

    Returns
    -------
    list of str
        List of capability names.
    """
    manifest = load_manifest()
    return manifest.get('capabilities', [])


def get_profiles():
    """Get all available calculation profiles.

    Returns
    -------
    dict
        Profile name -> profile info mapping.
    """
    manifest = load_manifest()
    return manifest.get('profiles', {})


def get_profile(name):
    """Get one profile entry.

    Parameters
    ----------
    name : str
        Profile name (e.g. 'mlp_eg_gradient').

    Returns
    -------
    dict or None
        Profile info, or None when the profile is not declared.
    """
    return get_profiles().get(name)


def list_profiles():
    """List all available profile names.

    Returns
    -------
    list of str
        Sorted list of profile names.
    """
    return sorted(get_profiles().keys())


def get_registry():
    """Get the full registry in a standardized format.

    Returns
    -------
    dict
        Registry dictionary with the program label as key.
    """
    manifest = load_manifest()
    skill_key = manifest.get('name', 'nn')
    backend = manifest.get('backend', {})
    profiles = manifest.get('profiles', {})
    capabilities = manifest.get('capabilities', [])

    registry = {
        skill_key: {
            'description': f"neural network potential energy surface: {skill_key}",
            'module': backend.get('module', 'backend'),
            'backend_class': backend.get('class', 'NNPESBackend'),
            'roles': manifest.get('roles', {}),
            'capabilities': capabilities,
            'tasks': profiles,
            'execution': {
                'type': 'python_module',
                'module': f"{skill_key}.backend",
                'input': 'json',
                'arg': '',
            },
        }
    }

    return registry


def get_subsections():
    """List the extra input sections this skill reads.

    A skill that needs more than its own '&<name>' block declares the
    extra section names here. The framework folds each of them into the
    skill config under the same key and nothing else: which keywords live
    in a section, and what they mean, stays inside the skill.

    Returns
    -------
    list of str
        Section names, e.g. ['eg', 'nac'].
    """
    manifest = load_manifest()
    return list(manifest.get('subsections', []))


def get_defaults():
    """Return skill default configuration values.

    This is the interface for the framework to obtain skill defaults
    without importing internal schema modules. It doubles as the
    keyword vocabulary: a field missing here is dropped by the config
    builder before the backend ever sees it.

    Returns
    -------
    dict
        Default values for all skill config fields.
    """
    from .schema.config import DEFAULT
    return dict(DEFAULT)


def resolve_profile(method, jobtype, driver='main'):
    """Resolve the calculation profile from method, jobtype and driver.

    Framework-facing interface for profile resolution. Delegates to the
    skill's internal resolve_profile logic. The driver keyword
    (framework control.driver) is consulted for a dynamics request only:
    a 'md' jobtype under a driver that would hand the propagation to this
    skill is rejected, because the network never moves the nuclei itself,
    while 'train', 'prediction' and the property jobtypes carry no
    propagation and resolve under any driver.

    Parameters
    ----------
    method : str
        Model family ('mlp_eg', 'mlp_nac2', 'eg_nac', ...).
    jobtype : str
        Framework jobtype (e.g. 'energy', 'energy gradient', 'md',
        'train', 'prediction').
    driver : str, optional
        Dynamics driver; only relevant to a dynamics request.

    Returns
    -------
    str
        Profile name (e.g. 'mlp_eg_gradient', 'eg_nac_nac').
    """
    from .schema.method import resolve_profile as _resolve
    return _resolve(method, jobtype, driver=driver)
