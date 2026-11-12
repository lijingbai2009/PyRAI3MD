"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the normalization functions for the CP2K skill,
including verification of electronic structure, state numbers, and state spins.

"""


from .config import SkillConfig, System, Calculation, Properties
from .reference import STATE_LABEL
from .reference import PERIODIC_TABLE
from .reference import DEFAULT_BASIS, DEFAULT_AUX_BASIS, DEFAULT_POTENTIAL
from .reference import VALENCE_ELECTRON


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for CP2K computation.

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

    # Fill element-level basis/potential defaults from the atom list
    work = verify_basis(work)

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


def verify_basis(data: dict) -> dict:
    """Fill element-level basis / potential / auxiliary basis names.

    Deduplicates the atom list and fills the per-element dicts with
    the demo-verified defaults (DZVP-MOLOPT-SR-GTH-qN / GTH-PBE /
    admm-dzp-qN); user entries always take priority. The auxiliary
    basis dict is only filled when ADMM is requested (non-empty
    auxiliary_basis on input).

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with element-level basis entries.
    """
    elements = sorted(set(data['atoms']))

    basis = dict(data.get('basis') or {})
    potential = dict(data.get('potential') or {})
    auxiliary_basis = dict(data.get('auxiliary_basis') or {})
    use_admm = len(auxiliary_basis) > 0

    for element in elements:
        qval = VALENCE_ELECTRON.get(element)
        suffix = f'-q{qval}' if qval is not None else ''
        basis.setdefault(element, f'{DEFAULT_BASIS}{suffix}')
        potential.setdefault(element, DEFAULT_POTENTIAL)
        if use_admm:
            auxiliary_basis.setdefault(element, f'{DEFAULT_AUX_BASIS}{suffix}')

    data['basis'] = basis
    data['potential'] = potential
    data['auxiliary_basis'] = auxiliary_basis

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
    target_soc_pair = data['target_soc_pair']

    if nstate < 1:
        raise ValueError("nstate must be at least 1.")

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
        if 0 not in (a, b):
            raise ValueError(
                f"CP2K TDDFPT NAC would only support couplings between S0 and excited states, "
                f"but pair {pair} does not include state 0."
            )
        else:
            valid_nac_pair.append(pair)

    # Default to all S0-SN pairs when not specified
    if len(valid_nac_pair) == 0:
        valid_nac_pair = [[0, i] for i in range(1, nstate)]

    valid_soc_pair = []
    for pair in target_soc_pair:
        if len(pair) != 2:
            continue
        a, b = pair
        if not (0 <= a < nstate and 0 <= b < nstate):
            raise ValueError("target_soc_pair contains invalid state indices.")
        else:
            valid_soc_pair.append(pair)

    if len(state_spin) == 0:
        state_spin = [multiplicity - 1] * nstate

    if len(state_spin) != nstate:
        raise ValueError("nstate does not match the number of state spins.")

    if state_spin[0] != multiplicity - 1:
        raise ValueError("The first state spin does not match the multiplicity of the system.")

    # CP2K TDDFPT requests NSTATES excited roots; cover the
    # framework excited-state count and any explicitly targeted
    # state index. An explicit positive tddft_nstates is kept.
    if multiplicity == 1:
        nsinglet = state_spin.count(0)
        ntriplet = state_spin.count(2)
        data['nsing_root'] = max(nsinglet - 1, 0)
        data['ntrip_root'] = ntriplet
    else:
        data['nsing_root'] = max(nstate - 1, 0)
        data['ntrip_root'] = 0

    if data.get('tddft_nstates', 0) <= 0:
        required = nstate - 1
        if target_states:
            required = max(required, max(target_states))
        # NSTATES is counted per spin manifold: with RKS_TRIPLETS on, CP2K
        # solves that many roots in the singlet block and that many in the
        # triplet block, two tables that each restart their numbering at 1
        # (measured on cp2k_demo/sp/bulk.log, NSTATES 10 against two 1..10
        # tables). An all-singlet request has no triplet roots, so the value is
        # what it always was; the term only bites once a triplet is asked for.
        required = max(required, data.get('ntrip_root', 0))
        data['tddft_nstates'] = max(required, 0)

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
                              or [[0, i] for i in range(1, nstate)])
    # One placeholder row per declared transition pair: sizing these by
    # the state ladder promised a row for a pair nobody declared (#448).
    npair = len(data['transition_pair'])
    data['osc_strength'] = [0.0] * npair
    data['transition_dipole'] = [[0.0, 0.0, 0.0]] * npair
    data['gradient'] = [[[0.0, 0.0, 0.0]]] * nstate

    return data


def verify_state_spin(data: dict) -> dict:
    """Verify state spin consistency and build pair maps.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with spin-derived fields.
    """
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
            "CP2K orders states by ascending 2S; the input state_spin %r is not,"
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

    # Keep same-spin pairs that include the ground state (the only
    # NAC topology the framework expects from this skill).
    nac_pair = [pair for pair in nac_pair if 0 in pair]

    data['nac_pair_map'] = nac_pair
    data['soc_pair_map'] = soc_pair
    data['nnac'] = len(nac_pair)
    data['nsoc'] = len(soc_pair)
    data['nac'] = [[[0.0, 0.0, 0.0]] * natom] * len(nac_pair)
    data['soc'] = [[0.0]] * len(soc_pair)

    return data


def build_spin_pair_maps(state_spin):
    """Build the NAC and SOC pair maps based on the state spins.

    Same-spin state pairs are NAC candidates, different-spin state
    pairs are SOC candidates.

    Parameters
    ----------
    state_spin : list of int
        Spin quantum number (2S) for each state.

    Returns
    -------
    tuple of (list, list)
        NAC pair map and SOC pair map.
    """
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
