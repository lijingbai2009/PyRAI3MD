"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the normalization functions for the LAMMPS skill,
including verification of the system size and ground-state
information derivation.

"""


from .config import SkillConfig, System, Calculation, Properties


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for LAMMPS computation.

    Bridges between the nested SkillConfig and the flat-dict
    verify_* functions: extracts a work dict from submodels,
    runs verification, then writes results back.

    Parameters
    ----------
    config : SkillConfig
        Raw validated config.

    Returns
    -------
    SkillConfig
        Normalized config with derived fields populated.
    """
    # Build a flat work dict from the submodels that verify_* touch
    work = {}
    work.update(config.system.model_dump())
    work.update(config.calculation.model_dump())
    work.update(config.properties.model_dump())

    # Verify the system size (atoms vs coordinates)
    work = verify_system(work)

    # Normalize ground-state information
    work = verify_state_num(work)

    # Write results back to the appropriate submodels
    system_updates = {k: work[k] for k in System.model_fields if k in work}
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    props_updates = {k: work[k] for k in Properties.model_fields if k in work}

    return config.model_copy(update={
        'system': config.system.model_copy(update=system_updates),
        'calculation': config.calculation.model_copy(update=calc_updates),
        'properties': config.properties.model_copy(update=props_updates),
    })


def verify_system(data: dict) -> dict:
    """Verify that the number of atoms matches the coordinates.

    LAMMPS is a classical force field method: no electronic structure
    verification is needed (charges/multiplicity are carried by the
    force field topology file).

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with natom derived.
    """
    atoms = data['atoms']
    coordinates = data['coordinates']

    if len(atoms) != len(coordinates):
        raise ValueError("The number of atoms and coordinates do not match.")
    else:
        data['natom'] = len(atoms)

    return data


def verify_state_num(data: dict) -> dict:
    """Normalize ground-state information for LAMMPS.

    LAMMPS is a ground-state force field method, so only a single
    state is computed regardless of the framework nstate.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with ground-state info normalized.
    """
    natom = data['natom']

    # LAMMPS computes the ground state only
    data['nstate'] = 1
    data['target_states'] = [0]
    data['state_spin'] = [0]
    data['state_label'] = ['s0']
    data['energy'] = [0.0]
    data['gradient'] = [[[0.0, 0.0, 0.0]] * natom]
    data['dipole'] = [[0.0, 0.0, 0.0]]
    data['excitation'] = [0.0]
    data['osc_strength'] = [0.0]
    data['transition_dipole'] = [[0.0, 0.0, 0.0]]
    # No transition channel of its own: the [0, 0] row this used to declare
    # is a state coupled to itself, and a force field or semi-empirical method
    # computes no transition property (#448).

    # LAMMPS provides no nonadiabatic or spin-orbit couplings
    data['nac_pair_map'] = []
    data['soc_pair_map'] = []
    data['nnac'] = 0
    data['nsoc'] = 0

    return data
