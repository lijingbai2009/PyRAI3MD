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

This module provides the normalization functions for the xTB skill,
including verification of electronic structure and ground-state
information derivation.

"""


from .config import SkillConfig, System, Calculation
from .reference import PERIODIC_TABLE


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for xTB computation.

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

    # Verify the electronic structure of the system
    work = verify_electronic_structure(work)

    # Normalize ground-state information
    work = verify_state_num(work)

    # Write results back to the appropriate submodels
    system_updates = {k: work[k] for k in System.model_fields if k in work}
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}

    return config.model_copy(update={
        'system': config.system.model_copy(update=system_updates),
        'calculation': config.calculation.model_copy(update=calc_updates),
    })


def verify_electronic_structure(data: dict) -> dict:
    """Verify that the multiplicity is compatible with the number of electrons.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with electron counts and uhf.
    """
    atoms = data['atoms']
    coordinates = data['coordinates']
    charges = data['charges']
    multiplicity = data['multiplicity']

    if len(atoms) != len(coordinates):
        raise ValueError("The number of atoms and coordinates do not match.")
    else:
        data['natom'] = len(atoms)

    num_electrons = sum(PERIODIC_TABLE.get(sym, 0) for sym in atoms) - charges
    unpaired_electrons = multiplicity - 1

    if unpaired_electrons < 0:
        raise ValueError("Multiplicity must be at least 1.")

    na = (num_electrons + unpaired_electrons) / 2
    nb = (num_electrons - unpaired_electrons) / 2

    if not na.is_integer() or not nb.is_integer():
        raise ValueError("Multiplicity is incompatible with total electron count (parity mismatch).")
    else:
        data['total_electron'] = int(num_electrons)
        data['nalpha'] = int(na)
        data['nbeta'] = int(nb)

    # Number of unpaired electrons for the xTB --uhf flag
    data['uhf'] = int(unpaired_electrons)

    return data


def verify_state_num(data: dict) -> dict:
    """Normalize ground-state information for xTB.

    xTB (GFN-x and GFN-FF) is a ground-state method, so only a
    single state is computed regardless of the framework nstate.

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
    multiplicity = data['multiplicity']

    # xTB computes the ground state only
    data['nstate'] = 1
    data['target_states'] = [0]
    data['state_spin'] = [multiplicity - 1]
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

    # xTB provides no nonadiabatic or spin-orbit couplings
    data['nac_pair_map'] = []
    data['soc_pair_map'] = []
    data['nnac'] = 0
    data['nsoc'] = 0

    return data
