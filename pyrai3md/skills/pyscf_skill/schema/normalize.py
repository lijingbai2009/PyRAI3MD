"""
======================================================================
This is a standalone quantum chemistry calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 30, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the normalization functions for the PySCF skill,
including verification of electronic structure, state numbers, and state spins.

"""


from .config import SkillConfig, System, Calculation, Properties
from .method import AVAILABLE_METHOD
from .reference import STATE_LABEL
from .reference import PERIODIC_TABLE


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for PySCF computation.

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

    # Verify the state spin and build state labels, NAC and SOC pair maps
    work = verify_state_num(work)
    work = verify_state_spin(work)

    # Write results back to the corresponding submodels
    system_updates = {k: work[k] for k in System.model_fields if k in work}
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    props_updates = {k: work[k] for k in Properties.model_fields if k in work}

    return config.model_copy(update={
        'system': config.system.model_copy(update=system_updates),
        'calculation': config.calculation.model_copy(update=calc_updates),
        'properties': config.properties.model_copy(update=props_updates),
    })


def verify_electronic_structure(data: dict) -> dict:
    # Verify that the multiplicity is compatible with the number of electrons in the system
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

    # Check if multiplicity is compatible with total electrons
    # For a given multiplicity M, we need: (num_electrons - unpaired_electrons) to be even

    if not na.is_integer() or not nb.is_integer():
        raise ValueError("Error: Multiplicity is incompatible with total electron count (parity mismatch).\n")
    else:
        data['total_electron'] = int(num_electrons)
        data['nalpha'] = int(na)
        data['nbeta'] = int(nb)

    return data


def verify_state_num(data: dict) -> dict:
    # Verify that the number of states is at least 1
    nstate = data['nstate']
    state_spin = data['state_spin']
    multiplicity = data['multiplicity']
    target_states = data['target_states']
    target_nac_pair = data['target_nac_pair']
    target_soc_pair = data['target_soc_pair']

    # check if nstate is at least 1
    if nstate < 1:
        raise ValueError("nstate must be at least 1.")

    # check if target_states is set
    if len(target_states) == 0:
        target_states = [0] # if not set, default to ground state only

    target_states = list(set(target_states)) # remove duplicates

    # check if target_states are valid
    if max(target_states) + 1 > nstate:
        raise ValueError("target_states must be less than nstate.")

    # check if the number of target states exceeds nstate
    if len(target_states) > nstate:
        raise ValueError("The number of target states cannot exceed nstate.")

    # check if the method supports excited states: only methods with
    # a tddft post-processing (tddft/tda) can compute excited-state
    # energies and gradients. State numbering is 0-based (0 = S0).
    method = data.get('method', '')
    if max(target_states) > 0:
        method_info = AVAILABLE_METHOD.get(method, {})
        if method_info.get('post_method') != 'tddft':
            raise ValueError(
                "Excited states (target_states > 0) require method "
                "'tda' or 'tddft', got '%s'." % method)

    valid_nac_pair = []
    for pair in target_nac_pair:
        if len(pair) != 2:
            continue

        a, b = pair
        if not (0 <= a < nstate and 0 <= b < nstate):
            raise ValueError("target_nac_pair contains invalid state indices.")
        else:
            valid_nac_pair.append(pair)

    valid_soc_pair = []
    for pair in target_soc_pair:
        if len(pair) != 2:
            continue

        a, b = pair
        if not (0 <= a < nstate and 0 <= b < nstate):
            raise ValueError("target_soc_pair contains invalid state indices.")
        else:
            valid_soc_pair.append(pair)

    # If state_spin is not provided, assume all states have the same spin as the ground state
    if len(state_spin) == 0:
        state_spin = [multiplicity - 1] * nstate

    # check if the length of state_spin matches nstate
    if len(state_spin) != nstate:
        raise ValueError("nstate does not match the number of state spins.")

    # check if the first state spin matches the multiplicity of the system
    if state_spin[0] != multiplicity - 1:
        raise ValueError("The first state spin does not match the multiplicity of the system.")

    data['target_states'] = target_states
    data['target_nac_pair'] = valid_nac_pair
    data['target_soc_pair'] = valid_soc_pair
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
                              or [[0, i] for i in range(1, nstate)]) # pyscf only supports state-to-ground transition, so the transition pair is (0, i) for each state i
    # One placeholder row per declared transition pair: sizing these by
    # the state ladder promised a row for a pair nobody declared (#448).
    npair = len(data['transition_pair'])
    data['osc_strength'] = [0.0] * npair
    data['transition_dipole'] = [[0.0, 0.0, 0.0]] * npair
    data['gradient'] = [[[0.0, 0.0, 0.0]]] * nstate

    return data


def verify_state_spin(data: dict) -> dict:
    natom = data['natom']
    multiplicity = data['multiplicity']
    nstate = data['nstate']
    state_spin = data['state_spin']

    # singlet system
    if multiplicity == 1:
        nsinglet = state_spin.count(0)
        ntriplet = state_spin.count(2)
        if nsinglet + ntriplet != nstate:
            raise ValueError("For a singlet system, state spins must be 0 (singlet) or 2 (triplet).")

    # non-singlet system
    else:
        if any(s != state_spin[0] for s in state_spin):
            raise ValueError("All state spins must be identical for non-singlet systems.")

    # Sorting validates a standalone input only: with pairs declared against
    # these indices, permuting the ladder would re-number the declaration while
    # the framework keeps addressing the rows by it (#448).
    spins = [int(x) for x in state_spin]
    declared_pairs = [list(pair) for pair in
                      (list(data.get('coupling') or [])
                       + list(data.get('nac_pair_map') or [])
                       + list(data.get('soc_pair_map') or []))
                      if len(pair) == 2]
    if declared_pairs and spins != sorted(spins):
        raise ValueError(
            "PySCF orders states by ascending 2S; the input state_spin %r is not,"
            " and this run declares state pairs %r. Sort the state list in the"
            " input instead: re-ordering it here would address the reported rows"
            " by an index other than the declaration's." % (spins, declared_pairs))
    data['state_spin'] = spins if declared_pairs else sorted(spins)
    state_label = []
    for i in range(data['nstate']):
        state_label.append(STATE_LABEL[data['state_spin'][i]])

    data['state_label'] = state_label

    # The pair maps are the framework's declaration, forwarded; nothing is
    # enumerated from the ladder here any more (the cross product used to
    # overwrite what the input named).
    nac_pair = [list(pair) for pair in (data.get('nac_pair_map') or [])
                if len(pair) == 2]
    soc_pair = [list(pair) for pair in (data.get('soc_pair_map') or [])
                if len(pair) == 2]
    if not nac_pair and not soc_pair:
        nac_pair, soc_pair = build_spin_pair_maps(data['state_spin'])

    data['nac_pair_map'] = nac_pair
    data['soc_pair_map'] = soc_pair
    data['nnac'] = len(nac_pair)
    data['nsoc'] = len(soc_pair)
    data['nac'] = [[[0.0, 0.0, 0.0]] * natom ] * len(nac_pair)
    data['soc'] = [[0.0]] * len(soc_pair)

    return data


def build_spin_pair_maps(state_spin):
    # Build the NAC and SOC pair maps based on the state spins
    nac_pair_map = []
    soc_pair_map = []

    M = len(state_spin)

    for i in range(M):
        for j in range(i + 1, M):
            if state_spin[i] == state_spin[j]:
                nac_pair_map.append([i, j])
            else:
                soc_pair_map.append([i, j])

    return nac_pair_map, soc_pair_map
