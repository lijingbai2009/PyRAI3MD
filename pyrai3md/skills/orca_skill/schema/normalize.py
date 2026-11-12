"""
======================================================================
This is a standalone ORCA calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the normalization functions for the ORCA skill,
including verification of electronic structure, state numbers, and state spins.

"""


from .config import SkillConfig, System, Calculation, Properties
from .reference import STATE_LABEL
from .reference import PERIODIC_TABLE


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for ORCA computation.

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

    ORCA TDDFT only supports nonadiabatic couplings between the
    ground state (S0) and excited states (SN). NAC pairs between
    two excited states are rejected.

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
                f"ORCA TDDFT NAC only supports couplings between S0 and excited states, "
                f"but pair {pair} does not include state 0."
            )
        else:
            valid_nac_pair.append(pair)

    # Nothing is supplied here when the input named no pair: the coupling
    # declaration (`coupling` / the two pair maps) is the authority, and this
    # list only says what the step asks the engine for. The old default filled
    # in every S0-SN pair, asking for couplings nobody declared.

    valid_soc_pair = []
    for pair in target_soc_pair:
        if len(pair) != 2:
            continue
        a, b = pair
        if not (0 <= a < nstate and 0 <= b < nstate):
            raise ValueError("target_soc_pair contains invalid state indices.")
        if 0 in (a, b):
            # The mirror of the NAC rule above: DoSOC numbers excited roots
            # only, so a pair through S0 has no element to read. Refused, not
            # dropped - the declaration asked for a row the engine cannot give.
            raise ValueError(
                "ORCA DoSOC only reports couplings between excited singlet and"
                f" triplet roots, but pair {pair} includes the ground state 0.")
        valid_soc_pair.append(pair)

    if len(state_spin) == 0:
        state_spin = [multiplicity - 1] * nstate

    if len(state_spin) != nstate:
        raise ValueError("nstate does not match the number of state spins.")

    if state_spin[0] != multiplicity - 1:
        raise ValueError("The first state spin does not match the multiplicity of the system.")

    # ORCA computes nroots singlet and nroots triplet roots when
    # 'Triplets true' is active, so request the maximum of excited
    # singlets (ground state excluded) and triplets; pure-singlet
    # runs keep the classic nstate - 1 count.
    if multiplicity == 1:
        nsinglet = state_spin.count(0)
        ntriplet = state_spin.count(2)
        data['nsing_root'] = max(nsinglet - 1, 0)
        data['ntrip_root'] = ntriplet
        data['tddft_nroots'] = max(nsinglet - 1, ntriplet, 0)
    else:
        data['nsing_root'] = max(nstate - 1, 0)
        data['ntrip_root'] = 0
        data['tddft_nroots'] = max(nstate - 1, 0)

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

    # ORCA spin-orbit coupling (DoSOC) is only implemented for a
    # closed-shell singlet (RHF) reference; triplet target states
    # enter through 'Triplets true' in the %tddft block instead.
    if data['profile'] in ('tddft_soc', 'dft_soc') and multiplicity != 1:
        raise ValueError(
            "SOC calculations require a singlet ground state "
            "(multiplicity 1); ORCA DoSOC only supports an RHF "
            f"reference, got multiplicity {multiplicity}."
        )

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

    # The skill's canonical order is ascending 2S, but re-ordering the ladder
    # re-numbers every declared pair, and the framework addresses the returned
    # rows by the declaration's order (`workflow/integrator/fssh.py:543`). A
    # standalone input may still be sorted - that is the skill validating the
    # agent's own input - but with a declaration present the order is checked,
    # never changed (#448).
    spins = [int(x) for x in state_spin]
    declared = [list(pair) for pair in
                (list(data.get('coupling') or [])
                 + list(data.get('nac_pair_map') or [])
                 + list(data.get('soc_pair_map') or []))
                if len(pair) == 2]
    if declared and spins != sorted(spins):
        raise ValueError(
            "ORCA orders its states by ascending 2S; the input state_spin"
            f" {spins} is not, and this run declares state pairs {declared}."
            " Sort the state list in the input: re-ordering it here would"
            " permute the declared pairs and address the reported rows by an"
            " index other than the one the framework used.")
    data['state_spin'] = spins if declared else sorted(spins)
    state_label = []
    for i in range(data['nstate']):
        state_label.append(STATE_LABEL[data['state_spin'][i]])

    data['state_label'] = state_label

    # The declaration is the authority and it arrives from the framework
    # (`schema/adapter.py` forwards `coupling` and both pair maps). This skill
    # classifies what was declared; it no longer enumerates the cross product
    # of its own sorted ladder, which was a second source of truth that
    # overwrote the declaration and re-numbered its indices. Enumerating stays
    # for what it always was: a standalone input that declares nothing, where
    # the skill is validating the agent's own state numbers (#448).
    coupling = [list(pair) for pair in (data.get('coupling') or [])
                if len(pair) == 2]
    nac_pair = [list(pair) for pair in (data.get('nac_pair_map') or [])
                if len(pair) == 2]
    soc_pair = [list(pair) for pair in (data.get('soc_pair_map') or [])
                if len(pair) == 2]
    if coupling or nac_pair or soc_pair:
        declared = coupling or (nac_pair + soc_pair)
        computed_nac, computed_soc = classify_declared_pairs(
            declared, data['state_spin'])
        if (nac_pair and nac_pair != computed_nac) or \
                (soc_pair and soc_pair != computed_soc):
            raise ValueError(
                "molecule.coupling %r and the declared pair maps "
                "nac_pair_map=%r soc_pair_map=%r are one statement and these "
                "two disagree; fix one side." % (declared, nac_pair, soc_pair))
        coupling, nac_pair, soc_pair = declared, computed_nac, computed_soc
        # ORCA's two engine limits - NACME only between S0 and an excited
        # singlet, DoSOC only between excited roots - are refused, not filtered,
        # but they fall on **what this deck is asked to produce**. A pair that
        # is declared and never requested cannot be answered wrongly, because
        # it is not requested at all: `md_orca_fssh` declares three singlets'
        # full pair set and runs kTDC, which estimates the coupling from the
        # energy curvature and sends no `NACME` switch, and refusing on that
        # declaration is the defect #448 shipped with (round
        # tmp_20261102_114641, both carriers). Ask for the same declaration with
        # `nactype: nac` and the refusal below does fire, which is what it is for.
        from .method import PROFILE_STEP_MAP
        steps = PROFILE_STEP_MAP.get(str(data.get('profile') or ''), ())
        if 'nac' in steps:
            for pair in nac_pair:
                if 0 not in pair:
                    raise ValueError(
                        "ORCA TDDFT NAC only supports couplings between S0 and"
                        " excited states, but this run asks for the declared"
                        " pair %r, which does not include state 0." % (pair,))
        if 'soc' in steps:
            for pair in soc_pair:
                if 0 in pair:
                    raise ValueError(
                        "ORCA DoSOC only reports couplings between excited"
                        " singlet and triplet roots, but this run asks for the"
                        " declared pair %r, which includes the ground state 0."
                        % (pair,))
    else:
        # Nothing declared: the standalone case takes the pairs the engine can
        # answer on this ladder - NAC only from S0, SOC only between excited
        # roots - which is a convention for an input that declared nothing, not
        # a rewrite of anybody's declaration.
        nac_pair, soc_pair = build_spin_pair_maps(data['state_spin'])
        nac_pair = [list(pair) for pair in nac_pair if 0 in pair]
        soc_pair = [list(pair) for pair in soc_pair if 0 not in pair]
        coupling = [list(pair) for pair in nac_pair + soc_pair]
    data['coupling'] = coupling
    data['nac_pair_map'] = nac_pair
    data['soc_pair_map'] = soc_pair
    # A step that names no pair of its own asks for the declared ones.
    if not data.get('target_nac_pair'):
        data['target_nac_pair'] = [list(pair) for pair in nac_pair]
    if not data.get('target_soc_pair'):
        data['target_soc_pair'] = [list(pair) for pair in soc_pair]
    data['nnac'] = len(nac_pair)
    data['nsoc'] = len(soc_pair)
    data['nac'] = [[[0.0, 0.0, 0.0]] * natom] * len(nac_pair)
    data['soc'] = [[0.0]] * len(soc_pair)

    return data


def classify_declared_pairs(coupling, state_spin):
    """Split declared state pairs into NAC (same spin) and SOC (other spin).

    The same classification the framework applies when it derives
    `nac_pair_map`/`soc_pair_map` from `molecule.coupling`; it is kept local
    because the skill must not reach into framework internals, and the
    equivalence of the two is pinned by a criterion in
    `qualification/verification/test_state_declaration.py`.

    Parameters
    ----------
    coupling : list of [int, int]
        Declared 0-based state pairs.
    state_spin : list of int
        2S per state.

    Returns
    -------
    tuple of (list, list)
        (nac_pairs, soc_pairs), each in the declaration's order.
    """
    nac_pair_map = []
    soc_pair_map = []
    for pair in coupling:
        first, second = pair
        if state_spin[first] == state_spin[second]:
            nac_pair_map.append(list(pair))
        else:
            soc_pair_map.append(list(pair))
    return nac_pair_map, soc_pair_map


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
