"""
======================================================================
This is a standalone xTB calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides self-contained query capabilities for xtb_skill.
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
        Program label, e.g. 'xtb'.
    """
    manifest = load_manifest()
    return manifest.get('name', 'xtb')


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
        'name': manifest.get('name', 'xtb'),
        'skill_path': manifest.get('backend', {}).get('module', 'backend'),
        'backend_class': manifest.get('backend', {}).get('class', 'XtbBackend'),
        'supports_pbc': bool(manifest.get('supports_pbc', False)),
        # Compute resources the skill consumes, as declared in its
        # manifest (#306): keyword names, so the framework never has to
        # guess where a run said how many threads it wants.
        'resources': dict(manifest.get('resources') or {}),
    }


def get_roles():
    """Get the method roles this skill can serve as.

    Roles define the scope of this skill in multi-scale calculations:
      - qm: Primary quantum mechanics method (GFN2/GFN1/GFN0)
      - qm2: Secondary QM method for embedding (GFN2/GFN1/GFN0)
      - mm: Molecular mechanics method (GFN-FF)

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
        List of capability names, e.g. ['energy', 'gradient'].
    """
    manifest = load_manifest()
    return manifest.get('capabilities', [])


def get_profiles():
    """Get all available calculation profiles.

    Returns
    -------
    dict
        Profile name -> profile info mapping.
        Each profile info contains capability, method, steps, etc.
    """
    manifest = load_manifest()
    return manifest.get('profiles', {})


def get_profile(profile_name):
    """Get information for a specific profile.

    Parameters
    ----------
    profile_name : str
        Name of the profile, e.g. 'gfn2_energy'.

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

    This method provides a standardized registry format expected by
    the global query aggregator, enabling xtb_skill to serve
    as a self-contained registry source.

    Returns
    -------
    dict
        Registry dictionary with skill name as key.
    """
    manifest = load_manifest()
    skill_key = manifest.get('name', 'xtb')
    backend = manifest.get('backend', {})
    profiles = manifest.get('profiles', {})
    capabilities = manifest.get('capabilities', [])

    registry = {
        skill_key: {
            'description': f"xTB skill: {skill_key}",
            'module': backend.get('module', 'backend'),
            'backend_class': backend.get('class', 'XtbBackend'),
            'roles': manifest.get('roles', {}),
            'capabilities': capabilities,
            'tasks': profiles,
            'execution': {
                'type': 'python_module',
                'module': f"{skill_key}.agent",
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
    from .schema.reference import DEFAULT
    return dict(DEFAULT)


def resolve_profile(method, jobtype, driver='main'):
    """Resolve the calculation profile from method and jobtype.

    Framework-facing interface for profile resolution. Delegates to
    the skill's internal resolve_profile logic. Without it the
    framework keeps the schema default profile, which clashes with
    non-default methods. The driver keyword (framework control.driver)
    only matters for an iteration jobtype: a driver other than 'main'
    hands the steps to this skill, which xtb never owns.

    Parameters
    ----------
    method : str
        Computational method ('gfn2', 'gfn1', 'gfn0', or 'gfn_ff').
    jobtype : str
        Framework jobtype (e.g. 'energy', 'energy gradient').
    driver : str, optional
        Execution driver; consulted for an iteration jobtype only.

    Returns
    -------
    str
        Profile name (e.g. 'gfn2_energy', 'gfn_ff_gradient').
    """
    from .schema.method import resolve_profile as _resolve
    return _resolve(method, jobtype, driver=driver)
