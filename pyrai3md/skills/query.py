"""
======================================================================
This module provides dynamic skill registry querying capabilities.

It is responsible for:
1. Discovering available skills in the skills directory
2. Loading skill registries via manifest.yaml
3. Extracting api information for execution routing
4. Resolving a backend API path string to its class (`resolve_api`)

Used by qualification/manager.py to discover skills and by
the runtime system to route skill execution.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import importlib
import os
import sys
import warnings

import yaml


def _get_skills_dir():
    """Locate the skills directory within the pyrai3md package.

    Returns
    -------
    str
        Absolute path to the pyrai3md/skills/ directory.
    """
    return os.path.dirname(os.path.abspath(__file__))


def list_skills():
    """List all available skill names.

    Scans the skills directory for subdirectories containing
    a manifest.yaml file.

    Returns
    -------
    list of str
        Available skill names.
    """
    skills_dir = _get_skills_dir()
    skill_names = []

    if not os.path.isdir(skills_dir):
        return skill_names

    for name in sorted(os.listdir(skills_dir)):
        skill_path = os.path.join(skills_dir, name)
        manifest_path = os.path.join(skill_path, 'manifest.yaml')
        if os.path.isdir(skill_path) and os.path.isfile(manifest_path):
            skill_names.append(name)

    return skill_names


def _load_manifest(skill_name):
    """Load the manifest.yaml for a skill.

    Parameters
    ----------
    skill_name : str
        Name of the skill.

    Returns
    -------
    dict
        Parsed manifest dictionary.
    """
    skills_dir = _get_skills_dir()
    manifest_path = os.path.join(skills_dir, skill_name, 'manifest.yaml')

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.yaml not found for skill: {skill_name}")

    with open(manifest_path, 'r') as f:
        return yaml.safe_load(f)


def _add_framework_prefix(local_registry, skill_name):
    """Add pyrai3md.skills framework prefix to skill-local registry paths.

    Skills return relative paths (e.g. 'backend', 'agent').
    This function converts them to full framework paths
    (e.g. '<skills package>.molcas_skill.backend').

    Parameters
    ----------
    local_registry : dict
        Skill-local registry with relative paths.
    skill_name : str
        Skill directory name (e.g. 'molcas_skill').

    Returns
    -------
    dict
        Registry with full framework paths.
    """
    framework_registry = {}
    for skill_key, skill_info in local_registry.items():
        # Add framework prefix to module path
        raw_module = skill_info.get('module', 'backend')
        if not raw_module.startswith('pyrai3md.'):
            skill_info['module'] = f'{SKILLS_PACKAGE}.{skill_name}.{raw_module}'

        # Add framework prefix to execution module
        execution = skill_info.get('execution', {})
        if execution:
            # The skill side names the entry by program label ('lammps'),
            # while the package directory is 'lammps_skill'. Rebuild from
            # the directory name instead of prefixing the label a second
            # time (`...lammps_skill.lammps.agent` did not import).
            execution['module'] = skill_module_path(skill_name, 'agent')
            skill_info['execution'] = execution

        framework_registry[skill_key] = skill_info

    return framework_registry


# The package the skill modules live in, taken from this module's own
# import path instead of being spelled at every call site: a build that
# places the skills in another package (a third-party plugin following
# the same layout) changes this one name, not nine f-strings.
SKILLS_PACKAGE = __name__.rpartition('.')[0] or 'pyrai3md.skills'


def skill_module_path(dir_name, tail='query'):
    """Import path of one module inside a skill package.

    ``tail`` is the part after the skill directory ('query', 'backend',
    or '' for the package itself). The prefix is read from
    `SKILLS_PACKAGE`, so a build that places the skills elsewhere
    resolves the same modules without editing any caller.
    """
    tail = tail.strip('.')
    if not tail:
        return '%s.%s' % (SKILLS_PACKAGE, dir_name)
    return '%s.%s.%s' % (SKILLS_PACKAGE, dir_name, tail)


def get_registry(skill_name):
    """Get the full registry for a skill.

    If the skill provides its own query module (query.py with
    get_registry function), delegates to it for self-contained
    registry information. Otherwise, falls back to reading
    manifest.yaml directly.

    Parameters
    ----------
    skill_name : str
        Name of the skill to query.

    Returns
    -------
    dict
        The skill registry dictionary.
    """
    # Prefer the skill's own query module if available
    try:
        skill_query = importlib.import_module(
            f'{SKILLS_PACKAGE}.{skill_name}.query'
        )
        if hasattr(skill_query, 'get_registry'):
            # Get skill-local registry and add framework prefix
            local_registry = skill_query.get_registry()
            return _add_framework_prefix(local_registry, skill_name)
    except (ImportError, AttributeError):
        pass

    # Fallback: read manifest.yaml directly
    manifest = _load_manifest(skill_name)

    # Build registry in a format compatible with existing interface
    skill_key = manifest.get('name', skill_name)
    backend = manifest.get('backend', {})
    profiles = manifest.get('profiles', {})
    capabilities = manifest.get('capabilities', [])

    # Construct full framework path from skill-relative module
    raw_module = backend.get('module', 'backend')
    full_module = f'{SKILLS_PACKAGE}.{skill_name}.{raw_module}'

    registry = {
        skill_key: {
            'description': f"{skill_key} skill",
            'module': full_module,
            'backend_class': backend.get('class', ''),
            'roles': manifest.get('roles', {}),
            'capabilities': capabilities,
            'tasks': profiles,
            'execution': {
                'type': 'python_module',
                'module': f'{SKILLS_PACKAGE}.{skill_name}.agent',
                'input': 'json',
                'arg': '',
            },
        }
    }

    return registry


def get_api(skill_name):
    """Get execution API information from a skill's registry.

    Reserved accessor: no production caller today; kept as the public
    read of the registry's ``execution`` block.

    Queries the skill registry and extracts execution info
    including module path, input format, and other routing data.

    Parameters
    ----------
    skill_name : str
        Name of the skill to query.

    Returns
    -------
    dict
        API information with keys:
        - skill_path: module path for subprocess execution
        - execution: execution configuration dict
        - tasks: available tasks dict
    """
    registry = get_registry(skill_name)
    api_info = {}

    for skill_key, skill_info in registry.items():
        api_info[skill_key] = {
            'skill_path': skill_info.get('module', f'{SKILLS_PACKAGE}.{skill_name}'),
            'execution': skill_info.get('execution', {}),
            'tasks': skill_info.get('tasks', {}),
        }

    return api_info


def skill_stem(skill_name):
    """The program label of a skill package name.

    ``'orca_skill'`` -> ``'orca'``; a name that already is a label comes
    back unchanged. This one spelling is what the qualification layer
    builds ``{stem}_project`` keys and section headers from.
    """
    name = str(skill_name or '')
    return name[:-len('_skill')] if name.endswith('_skill') else name


def method_label(method):
    """The skill label of one theory/reference method spelling.

    ``'qm.pyscf'`` and ``'pyscf'`` both give ``'pyscf'``;
    ``'abinit.orca'`` gives ``'orca'``. The two prefixes are the retired
    spellings of the ``theory`` and ``reference`` queues, accepted so an
    archived input still runs. This is the one place the framework strips
    them: normalize rewrites the queues once, and every later reader sees
    a plain label.
    """
    text = str(method or '').strip()
    for prefix in ('qm.', 'abinit.'):
        if text.lower().startswith(prefix):
            return text[len(prefix):]
    return text


def _skill_entry(label):
    """The discovery entry for a label or a package name.

    Callers name a skill either way ('lammps' in control.theory,
    'lammps_skill' in the task queue), so both spellings resolve here.
    """
    if not str(label):
        return {}
    meta = discover_all() or {}
    entry = meta.get(str(label))
    if entry:
        return entry
    for value in meta.values():
        if value.get('skill_name') == str(label):
            return value
    return {}


def get_roles(label):
    """The role -> backend map a skill declares in its manifest.

    Parameters
    ----------
    label : str
        Skill label as written in control.theory ('molcas') or the
        package name ('molcas_skill').

    Returns
    -------
    dict
        {role: backend class name}; empty when the label names no skill.
    """
    if not str(label):
        return {}
    try:
        meta = _skill_entry(label)
        registry = get_registry(meta.get('_dir_name') or '') or {}
    except (ImportError, AttributeError, KeyError, OSError):
        return {}
    roles = {}
    for entry in registry.values():
        roles.update(entry.get('roles') or {})
    return roles


def manifest_value(label, key, default=None):
    """One manifest value a skill declares, as written.

    The scalar counterpart of `get_declaration`: a skill that states a
    single fact (whether its high-level layer receives an embedding
    field, for one) declares a value, and the framework reads it here
    instead of naming the skill.

    Parameters
    ----------
    label : str
        Skill label ('orca') or package name ('orca_skill').
    key : str
        Manifest key.
    default : any
        Returned when the skill or the key is absent.

    Returns
    -------
    any
        The declared value, or `default`.
    """
    if not str(label):
        return default
    try:
        meta = _skill_entry(label)
        dir_name = meta.get('_dir_name') or ''
        if not dir_name:
            return default
        manifest = _load_manifest(dir_name) or {}
    except (ImportError, AttributeError, KeyError, OSError):
        return default
    value = manifest.get(key, default)
    return default if value is None else value


def get_declaration(label, key):
    """One declaration a skill's manifest makes under `key`.

    The declarative channel for facts the framework must know about a
    skill but must not hardcode (the initial-guess file it can read, for
    one): the manifest states it, this reads it, and adding the same
    block to a second skill's manifest extends the channel with no
    framework change.

    Parameters
    ----------
    label : str
        Skill label ('molcas') or package name ('molcas_skill').
    key : str
        Manifest key; the value must be a mapping.

    Returns
    -------
    dict
        The declaration, or {} when the skill declares none.
    """
    if not str(label):
        return {}
    try:
        meta = _skill_entry(label)
        dir_name = meta.get('_dir_name') or ''
        if not dir_name:
            return {}
        manifest = _load_manifest(dir_name) or {}
    except (ImportError, AttributeError, KeyError, OSError):
        return {}
    value = manifest.get(key)
    return dict(value) if isinstance(value, dict) else {}


def role_of(label):
    """Which side of a composite a skill declares: 'qm', 'mm' or ''.

    A skill that declares an mm binding and no qm/qm2 binding is the
    classical side; one that declares qm or qm2 is the quantum side.
    This is the one reading of `roles` in the framework: the ONIOM2
    classical-outer gate and the QMMM theory binding both use it, so a
    provider handed in the other order still lands on its own side.
    """
    roles = get_roles(label)
    if roles.get('mm') and not (roles.get('qm') or roles.get('qm2')):
        return 'mm'
    if roles.get('qm') or roles.get('qm2'):
        return 'qm'
    return ''


def discover_all():
    """Discover all skills via their query modules.

    This is the sole entry point for pyrai3md to learn about
    all installed skills. Only imports {skill}.query — never
    imports backend/agent modules.

    Returns
    -------
    dict
        {name: {skill_name, name, skill_path, backend_class, ...}}
        e.g. {'pyscf': {'skill_name': 'pyscf_skill', 'name': 'pyscf', ...}}
        The registry key and 'name' field are the program label.
        'skill_name' is the folder name (e.g. 'pyscf_skill').
    """
    registry = {}
    for name in list_skills():
        try:
            skill_query = importlib.import_module(
                f'{SKILLS_PACKAGE}.{name}.query'
            )
            if hasattr(skill_query, 'get_metadata'):
                meta = skill_query.get_metadata()
                label = meta.get('name', '')
                if label:
                    # skill_name = folder name (e.g. 'pyscf_skill')
                    meta['skill_name'] = name
                    # Also store _dir_name for framework path construction
                    meta['_dir_name'] = name
                    registry[label] = meta
        except (ImportError, AttributeError) as error:
            warnings.warn(
                f"skill {name!r} was skipped during discovery: {error}",
                stacklevel=2,
            )

    return registry


def resolve_label(label):
    """Resolve a qm method label to skill info via the registry.

    Parameters
    ----------
    label : str
        QM method label, e.g. 'molcas', 'pyscf'.

    Returns
    -------
    dict or None
        Skill metadata dict if found, None otherwise.
    """
    registry = discover_all()
    return registry.get(label, None)


def resolve_api(api_path):
    """Resolve backend class from import path string.

    Parameters
    ----------
    api_path : str or type
        Full import path (e.g. 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend')
        or backend class object (for backward compatibility in tests).

    Returns
    -------
    type
        Backend class

    Raises
    ------
    ImportError
        If module cannot be imported.
    AttributeError
        If class not found in module.
    TypeError
        If api_path is neither str nor type.
    """
    # Backward compatibility: accept class objects (for tests)
    if isinstance(api_path, type):
        return api_path

    if not isinstance(api_path, str):
        raise TypeError(
            f"Expected str or type for api_path, got {type(api_path).__name__}"
        )

    # Split into module and class name
    try:
        module_name, cls_name = api_path.rsplit('.', 1)
    except ValueError:
        raise ValueError(
            f"Invalid API path format: {api_path}. "
            "Expected format: 'module.path.ClassName'"
        )

    # Import module and get class
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)
