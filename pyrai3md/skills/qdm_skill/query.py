"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides self-contained query capabilities for qdm_skill.
It reads the local manifest.yaml to expose available profiles,
capabilities, and methods — without depending on any external
framework module.

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
        Program label, e.g. 'qdm'.
    """
    manifest = load_manifest()
    return manifest.get('name', 'qdm')


def get_metadata():
    """Return skill metadata.

    This is the sole interface for the frontend framework to obtain
    skill identification, path, backend class, and config keys.

    Returns
    -------
    dict
        Metadata with keys: name, skill_path, backend_class.
    """
    manifest = load_manifest()
    return {
        'name': manifest.get('name', 'qdm'),
        'skill_path': manifest.get('backend', {}).get('module', 'backend'),
        'backend_class': manifest.get('backend', {}).get('class', 'QDMBackend'),
        'supports_pbc': bool(manifest.get('supports_pbc', False)),
        # Compute resources the skill consumes, as declared in its
        # manifest (#306): keyword names, so the framework never has to
        # guess where a run said how many threads it wants.
        'resources': dict(manifest.get('resources') or {}),
    }


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


def get_profile(profile_name):
    """Get information for a specific profile.

    Parameters
    ----------
    profile_name : str
        Name of the profile, e.g. 'tully1_nac'.

    Returns
    -------
    dict or None
        Profile info dict, or None if not found.
    """
    profiles = get_profiles()
    return profiles.get(profile_name, None)


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
        Registry dictionary with skill name as key.
    """
    manifest = load_manifest()
    skill_key = manifest.get('name', 'qdm')
    backend = manifest.get('backend', {})
    profiles = manifest.get('profiles', {})
    capabilities = manifest.get('capabilities', [])

    registry = {
        skill_key: {
            'description': f"QDM skill: {skill_key}",
            'module': backend.get('module', 'backend'),
            'backend_class': backend.get('class', 'QDMBackend'),
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


def get_defaults():
    """Return skill default configuration values.

    This is the interface for the framework to obtain skill defaults
    without importing internal schema modules.

    Returns
    -------
    dict
        Default values for all skill config fields.
    """
    from .schema.config import DEFAULT
    return dict(DEFAULT)


def resolve_profile(method, jobtype, driver='main'):
    """Resolve the calculation profile from method and jobtype.

    Framework-facing interface for profile resolution. Delegates to
    the skill's internal resolve_profile logic. The driver keyword
    (framework control.driver) only affects jobtype 'md':
    'main' selects the '{method}_nac' single-point superset while
    'skill' selects '{method}_reference' (internal quantum dynamics).

    Parameters
    ----------
    method : str
        Analytic model name ('tully1', 'tully2', 'tully3', 'lz1',
        'lz2', 'lz3').
    jobtype : str
        Framework jobtype (e.g. 'energy', 'energy gradient', 'md').
    driver : str, optional
        Dynamics driver ('main' or 'skill'); only affects 'md'.

    Returns
    -------
    str
        Profile name (e.g. 'tully1_energy', 'tully1_nac').
    """
    from .schema.method import resolve_profile as _resolve
    return _resolve(method, jobtype, driver=driver)
