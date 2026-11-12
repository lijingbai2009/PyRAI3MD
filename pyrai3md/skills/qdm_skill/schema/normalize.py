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

This module provides the normalization functions for the QDM skill,
including verification of the model geometry and state bookkeeping.

The QDM models are one-dimensional analytic potentials evaluated on
a single dummy atom; electronic-structure verification (electron
count, multiplicity) therefore does not apply and is skipped.

"""

from .config import SkillConfig, System, Calculation, Properties
from .reference import STATE_LABEL, MODEL_NSTATE


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for QDM computation.

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

    # Verify the model geometry (single dummy atom)
    work = verify_system(work)

    # Verify the state bookkeeping and build state labels / NAC map
    work = verify_state_num(work)

    # Write results back to the corresponding submodels
    system_updates = {k: work[k] for k in System.model_fields if k in work}
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    props_updates = {k: work[k] for k in Properties.model_fields if k in work}

    return config.model_copy(update={
        'system': config.system.model_copy(update=system_updates),
        'calculation': config.calculation.model_copy(update=calc_updates),
        'properties': config.properties.model_copy(update=props_updates),
    })


def verify_system(data: dict) -> dict:
    # Verify that the dummy-atom geometry is well formed. The QDM
    # models use the first atom x coordinate as the model coordinate
    # and never look up element properties, so any symbol (commonly
    # the dummy 'X') is accepted.
    atoms = data['atoms']
    coordinates = data['coordinates']

    if len(atoms) != len(coordinates):
        raise ValueError("The number of atoms and coordinates do not match.")

    if len(atoms) < 1:
        raise ValueError(
            "QDM requires at least one (dummy) atom carrying the "
            "model coordinate.")

    data['natom'] = len(atoms)

    return data


def verify_state_num(data: dict) -> dict:
    # Verify state bookkeeping against the two-state analytic models
    nstate = data['nstate']
    state_spin = data['state_spin']
    multiplicity = data['multiplicity']
    target_states = data['target_states']
    target_nac_pair = data['target_nac_pair']

    # Every QDM analytic model provides exactly two adiabatic states
    if nstate != MODEL_NSTATE:
        raise ValueError(
            f"QDM analytic models provide {MODEL_NSTATE} states, "
            f"but nstate = {nstate} was requested.")

    # check if target_states is set
    if len(target_states) == 0:
        target_states = list(range(nstate))

    target_states = sorted(set(target_states))  # remove duplicates

    # check if target_states are valid
    if max(target_states) + 1 > nstate:
        raise ValueError("target_states must be less than nstate.")

    valid_nac_pair = []
    for pair in target_nac_pair:
        if len(pair) != 2:
            continue

        a, b = pair
        if not (0 <= a < nstate and 0 <= b < nstate):
            raise ValueError("target_nac_pair contains invalid state indices.")
        else:
            valid_nac_pair.append(pair)

    # If state_spin is not provided, assume singlet states (the
    # analytic models carry no spin information)
    if len(state_spin) == 0:
        state_spin = [0] * nstate

    # check if the length of state_spin matches nstate
    if len(state_spin) != nstate:
        raise ValueError("nstate does not match the number of state spins.")

    data['target_states'] = target_states
    data['target_nac_pair'] = valid_nac_pair
    data['state_spin'] = [int(x) for x in state_spin]

    state_label = []
    for spin in data['state_spin']:
        state_label.append(STATE_LABEL.get(spin, 's'))
    data['state_label'] = state_label

    # The two-state models couple only the adjacent pair
    nac_pair_map = [[0, 1]]
    data['nac_pair_map'] = nac_pair_map
    data['soc_pair_map'] = []
    data['nnac'] = len(nac_pair_map)
    data['nsoc'] = 0

    natom = data['natom']
    data['energy'] = [0.0] * nstate
    data['gradient'] = [[[0.0, 0.0, 0.0]] * natom] * nstate
    data['nac'] = [[[0.0, 0.0, 0.0]] * natom] * len(nac_pair_map)

    # multiplicity is carried through untouched (dummy atom system)
    data['multiplicity'] = multiplicity

    return data
