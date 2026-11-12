"""
======================================================================
This is a standalone OpenMolcas calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module compiles a workflow graph from a profile name.
It resolves the profile into an ordered list of Molcas modules
and determines special execution requirements (e.g. dual RASSCF for SOC).

"""

from ...schema.method import PROFILE_MODULE_MAP
from ...schema.method import DUAL_RASSCF_PROFILES
from .graph import WorkflowGraph
from ..builder.embedding import (embedding_mode_label,
                                  embedding_requested as _embedding_requested)

_GENERIC_PROFILES = {
    'casscf_properties': 'casscf',
    'caspt2_properties': 'caspt2',
}

_PROPERTY_TOKENS = frozenset({'energy', 'gradient', 'nac', 'soc'})
_ABSORPTION_TOKENS = frozenset({'absorption'})


def _requested_tasks(config):
    """Return the requested property tokens from the jobtype."""
    tokens = set(str(getattr(config.calculation, 'jobtype', '') or '')
                 .lower().split())
    tasks = tokens & (_PROPERTY_TOKENS | _ABSORPTION_TOKENS)
    return frozenset(tasks)


def _is_dual_soc(config, tasks):
    """Whether SOC needs separate singlet/triplet RASSCF jobs."""
    if 'soc' not in tasks:
        return False
    state_spin = list(getattr(config.system, 'state_spin', []) or [])
    return len(set(state_spin)) > 1


def _generic_workflow(profile, config, transition=False):
    method = _GENERIC_PROFILES[profile]
    tasks = set(_requested_tasks(config))
    if transition:
        tasks.add('absorption')
    if not tasks:
        tasks.add('energy')

    dual = _is_dual_soc(config, tasks)
    esfp = _embedding_requested(config)
    embedding = embedding_mode_label(config)
    two_pass = ('nac' in tasks and esfp)

    modules = ['GATEWAY', 'SEWARD', 'RASSCF']
    if method == 'caspt2':
        modules.append('CASPT2')
    if 'gradient' in tasks or 'nac' in tasks:
        modules.append('ALASKA')
    if 'soc' in tasks or 'absorption' in tasks or transition:
        modules.append('RASSI')

    if two_pass:
        pass1 = {
            'espf': esfp,
            'energy': True,
            'gradient': 'gradient' in tasks,
            'nac': False,
            'soc': 'soc' in tasks,
            'absorption': 'absorption' in tasks,
        }
        pass2 = {
            'espf': False,
            'energy': False,
            'gradient': False,
            'nac': True,
            'soc': False,
            'absorption': False,
        }
        pass_plan = [pass1, pass2]
    else:
        pass_plan = [{
            'espf': esfp,
            'energy': True,
            'gradient': 'gradient' in tasks,
            'nac': 'nac' in tasks,
            'soc': 'soc' in tasks,
            'absorption': 'absorption' in tasks,
        }]

    return WorkflowGraph(
        profile=profile,
        modules=modules,
        requires_dual_rasscf=dual,
        requires_rassi_absorption=('absorption' in tasks or transition),
        method=method,
        tasks=frozenset(tasks),
        is_generic=True,
        requires_rassi_soc='soc' in tasks,
        requires_mclr=False,
        requires_esfp=esfp,
        embedding_mode=embedding,
        two_pass=two_pass,
        pass_plan=pass_plan,
    )


def compile_workflow(profile: str,
                     transition: bool = False,
                     config=None) -> WorkflowGraph:
    """Compile a workflow graph from a profile name.

    Parameters
    ----------
    profile : str
        Calculation profile name (e.g. 'casscf_gradient').
    transition : bool
        Whether the jobtype requests transition dipoles on top of
        the profile task. A RASSI step is appended to the module
        chain (upgrading the energy template to the absorption one)
        and runs in absorption mode.
    config : SkillConfig, optional
        Required for generic property profiles; supplies the jobtype,
        state spins and embedding mode.

    Returns
    -------
    WorkflowGraph
        Compiled workflow with module chain and execution flags.

    Raises
    ------
    ValueError
        If the profile is not recognized.
    """
    profile = profile.lower()

    if profile in _GENERIC_PROFILES:
        if config is None:
            raise ValueError(
                "Generic profile '%s' requires the runtime config."
                % profile)
        return _generic_workflow(profile, config, transition=transition)

    if profile not in PROFILE_MODULE_MAP:
        raise ValueError(f"Unknown profile: {profile}")

    modules = list(PROFILE_MODULE_MAP[profile])
    requires_dual_rasscf = profile in DUAL_RASSCF_PROFILES
    requires_rassi_absorption = False

    if transition:
        upper_modules = [m.upper() for m in modules]
        if 'RASSI' in upper_modules:
            # SOC/absorption profiles already end in RASSI; their
            # own RASSI mode (SOC or absorption) covers the request.
            if profile in ('casscf_absorption', 'caspt2_absorption'):
                requires_rassi_absorption = True
        else:
            modules.append('RASSI')
            requires_rassi_absorption = True

    tasks = set(_requested_tasks(config)) if config is not None else set()
    if profile.endswith('_soc'):
        tasks.add('soc')
    if profile.endswith('_nac'):
        tasks.add('nac')
    if profile.endswith('_gradient'):
        tasks.add('gradient')
    if profile.endswith('_energy'):
        tasks.add('energy')
    if profile.endswith('_absorption'):
        tasks.add('absorption')
    esfp = False if config is None else _embedding_requested(config)
    embedding = 'none' if config is None else embedding_mode_label(config)
    method = profile.split('_', 1)[0]
    is_generic = bool(esfp and method in ('casscf', 'caspt2'))
    two_pass = bool(is_generic and 'nac' in tasks and esfp)
    pass_plan = []
    if is_generic:
        if two_pass:
            pass_plan = [
                {'espf': esfp, 'energy': True,
                 'gradient': 'gradient' in tasks, 'nac': False,
                 'soc': 'soc' in tasks,
                 'absorption': 'absorption' in tasks},
                {'espf': False, 'energy': False, 'gradient': False,
                 'nac': True, 'soc': False, 'absorption': False},
            ]
        else:
            pass_plan = [{
                'espf': esfp, 'energy': True,
                'gradient': 'gradient' in tasks,
                'nac': 'nac' in tasks,
                'soc': 'soc' in tasks,
                'absorption': 'absorption' in tasks,
            }]

    return WorkflowGraph(
        profile=profile,
        modules=modules,
        requires_dual_rasscf=requires_dual_rasscf,
        requires_rassi_absorption=requires_rassi_absorption,
        method=method,
        tasks=frozenset(tasks),
        is_generic=is_generic,
        requires_rassi_soc=profile.endswith('_soc'),
        requires_mclr=False,
        requires_esfp=esfp,
        embedding_mode=embedding,
        two_pass=two_pass,
        pass_plan=pass_plan,
    )