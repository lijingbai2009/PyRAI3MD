"""
======================================================================
This is a standalone BAGEL calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the normalization functions for the BAGEL skill,
including verification of electronic structure and state numbers.
BAGEL does not support SOC, so spin-pair maps are not needed.

"""


from .config import SkillConfig, System, Calculation, Properties
from .reference import PERIODIC_TABLE


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for BAGEL computation.

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

    # Verify the electronic structure of the system
    work = verify_electronic_structure(work)

    # Verify the state numbers and build NAC pair maps
    work = verify_state_num(work)

    # Build active orbital list if not provided
    work = build_active_orbitals(work)

    # Write results back to the appropriate submodels
    system_updates = {k: work[k] for k in System.model_fields if k in work}
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    props_updates = {k: work[k] for k in Properties.model_fields if k in work}

    return config.model_copy(update={
        'system': config.system.model_copy(update=system_updates),
        'calculation': config.calculation.model_copy(update=calc_updates),
        'properties': config.properties.model_copy(update=props_updates),
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
        Updated configuration with electron counts.
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

    return data


def verify_state_num(data: dict) -> dict:
    """Verify and normalize state-related configuration.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with state info normalized.
    """
    nstate = data['nstate']
    state_spin = data['state_spin']
    multiplicity = data['multiplicity']
    target_states = data['target_states']
    target_nac_pair = data['target_nac_pair']
    opt_state_i = data['opt_state_i']
    opt_state_j = data['opt_state_j']
    # An unnamed second state means there is no second surface, so it resolves to
    # the state the run follows. A literal 0 would read as "state 0 was named",
    # and the builder decides the job kind from `opt_state_i != opt_state_j`:
    # an excited-state relaxation would be written as a MECI search over two
    # states, with a second target the input never asked for.
    if opt_state_j is None:
        opt_state_j = data['opt_state_i']
    data['opt_state_j'] = opt_state_j

    if nstate < 1:
        raise ValueError("nstate must be at least 1.")

    if not (0 <= opt_state_i < nstate) or not (0 <= opt_state_j < nstate):
        raise ValueError("opt_state_i/opt_state_j must be within [0, nstate).")

    if len(target_states) == 0:
        target_states = [0]

    target_states = list(set(target_states))

    if max(target_states) + 1 > nstate:
        raise ValueError("target_states must be less than nstate.")

    if len(target_states) > nstate:
        raise ValueError("The number of target states cannot exceed nstate.")

    valid_nac_pair = []
    for pair in target_nac_pair:
        if len(pair) != 2:
            continue
        a, b = pair
        if not (0 <= a < nstate and 0 <= b < nstate):
            raise ValueError("target_nac_pair contains invalid state indices.")
        else:
            valid_nac_pair.append(pair)

    if len(state_spin) == 0:
        state_spin = [multiplicity - 1] * nstate

    if len(state_spin) != nstate:
        raise ValueError("nstate does not match the number of state spins.")

    if state_spin[0] != multiplicity - 1:
        raise ValueError("The first state spin does not match the multiplicity of the system.")

    data['target_states'] = target_states
    data['target_nac_pair'] = valid_nac_pair
    data['state_spin'] = state_spin
    data['energy'] = [0.0] * nstate
    data['dipole'] = [[0.0, 0.0, 0.0]] * nstate
    data['excitation'] = [0.0] * nstate
    # The transition pairs are declared upstream (`schema/normalize.py`) and
    # read through here: rebuilding them per skill is what produced the
    # self-pair row, and this skill's state-to-ground restriction is enforced
    # where the values are read, not by inventing a pair for every state
    # (#448).
    declared_transition = [list(pair) for pair in
                           (data.get('transition_pair') or [])
                           if len(pair) == 2]
    # A framework run declares the pairs (`schema/normalize.py`); a standalone
    # input that declares none gets the state-to-ground convention this
    # skill's reader can actually answer - and it starts at state 1, because
    # the old `range(nstate)` opened with the self pair (0, 0).
    data['transition_pair'] = (declared_transition
                              or [[0, i] for i in range(1, nstate)])
    # One placeholder row per declared transition pair: sizing these by
    # the state ladder promised a row for a pair nobody declared (#448).
    npair = len(data['transition_pair'])
    data['osc_strength'] = [0.0] * npair
    data['transition_dipole'] = [[0.0, 0.0, 0.0]] * npair
    data['gradient'] = [[[0.0, 0.0, 0.0]]] * nstate

    # Build NAC pair map from state_spin (same-spin pairs)
    nac_pair_map = build_nac_pair_map(state_spin)
    data['nac_pair_map'] = nac_pair_map
    data['nnac'] = len(nac_pair_map)
    natom = data['natom']
    data['nac'] = [[[0.0, 0.0, 0.0]] * natom] * len(nac_pair_map)

    return data


def build_nac_pair_map(state_spin):
    """Build the NAC pair map based on the state spins.

    BAGEL only supports NAC between states of the same spin.
    No SOC support, so no SOC pair map.

    Parameters
    ----------
    state_spin : list of int
        Spin quantum number (2S) for each state.

    Returns
    -------
    list of list of int
        NAC pair map.
    """
    nac_pair_map = []
    M = len(state_spin)

    for i in range(M):
        for j in range(i + 1, M):
            if state_spin[i] == state_spin[j]:
                nac_pair_map.append([i, j])

    return nac_pair_map


def build_active_orbitals(data: dict) -> dict:
    """Build the active orbital list if not explicitly provided.

    BAGEL uses nclosed + active orbital indices.
    If active is empty, generate default active orbital indices
    starting from nclosed.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with active orbital list.
    """
    nclosed = data.get('nclosed', 0)
    active = data.get('active', [])
    nact_orb = data.get('nact_orb', 0)

    if not active and nact_orb > 0:
        # Generate default active orbital indices: nclosed, nclosed+1, ...
        data['active'] = list(range(nclosed, nclosed + nact_orb))

    return data
