"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the normalization functions for the OpenQP skill,
including verification of electronic structure, state numbers, state
spins, and derivation of SCF/TD calculation types.

"""


from .config import SkillConfig, System, Calculation, Properties
from .reference import STATE_LABEL
from .reference import PERIODIC_TABLE


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for OpenQP computation.

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

    # Resolve the spin-ladder sizes and the spin-orbit run options
    work = derive_soc_settings(work)

    # Derive SCF and TD calculation types from method and multiplicity
    work = derive_scf_type(work)
    work = derive_td_type(work)
    work = derive_functional(work)
    work = derive_nac_type(work)

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

    if len(state_spin) == 0:
        state_spin = [multiplicity - 1] * nstate

    if len(state_spin) != nstate:
        raise ValueError("nstate does not match the number of state spins.")

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

    # The skill's canonical order is ascending 2S, but permuting the ladder
    # re-numbers every declared pair, and the framework indexes the returned
    # rows by the declaration's order (`workflow/integrator/fssh.py`). A
    # standalone input may still be sorted - the skill validating the agent's
    # own state numbers - but with a declaration present the order is checked,
    # never changed (#448).
    spins = [int(x) for x in state_spin]
    # Every pair the input named anywhere: this is only the witness for "is
    # the ladder in the order the pairs were counted in", and a pair that
    # appears on one side only still proves the indices were fixed.
    order_pairs = [list(pair) for pair in
                   (list(data.get('coupling') or [])
                    + list(data.get('nac_pair_map') or [])
                    + list(data.get('soc_pair_map') or []))
                   if len(pair) == 2]
    declared = [list(pair) for pair in
                (data.get('coupling') or []) if len(pair) == 2]
    if not declared:
        declared = ([list(pair) for pair in (data.get('nac_pair_map') or [])
                     if len(pair) == 2]
                    + [list(pair) for pair in (data.get('soc_pair_map') or [])
                       if len(pair) == 2])
    if order_pairs and spins != sorted(spins):
        raise ValueError(
            "OpenQP orders states by ascending 2S; the input state_spin %r is"
            " not, and this run declares state pairs %r. Sort the state list"
            " in the input: re-ordering it here would permute the declared"
            " pairs and address the reported rows by an index other than the"
            " one the declaration used." % (spins, order_pairs))
    data['state_spin'] = spins if order_pairs else sorted(spins)
    state_label = []
    for i in range(data['nstate']):
        state_label.append(STATE_LABEL[data['state_spin'][i]])

    data['state_label'] = state_label

    # Pair maps.  The framework declares the pairs it wants coupled
    # through `coupling` (synced per step), and its integrators index the
    # returned NAC/SOC rows by the maps it derives from that declaration
    # (schema/normalize.py:239-268).  Rebuilding the full cross product of
    # state_spin here would hand those consumers rows in an order they do
    # not expect: for two singlets and two triplets the framework asks for
    # one SOC pair while the cross product produces four.  So classify the
    # declared pairs the same way the framework does, and only fall back to
    # the cross product when nothing was declared (a standalone input that
    # wants every cross-spin pair on the coupling surface).
    if declared:
        nac_pairs, soc_pairs = classify_spin_pairs(declared, data['state_spin'])
        preset_nac = [list(pair) for pair in (data.get('nac_pair_map') or [])
                      if len(pair) == 2]
        preset_soc = [list(pair) for pair in (data.get('soc_pair_map') or [])
                      if len(pair) == 2]
        # `coupling` is the authority and the two maps are its derived side,
        # exactly as `data/center.py::derive_pair_declaration` treats them - so
        # the comparison is coupling against the maps, never a sum of the two.
        # Adding them was my own #448 defect: on a multi-center run the maps
        # arrive as the active center's classification, the union then can
        # never equal either side, and 26 shipped ONIOM2/3 legs were refused
        # for a disagreement that did not exist (round tmp_20261102_114641).
        if (preset_nac and preset_nac != nac_pairs) or \
                (preset_soc and preset_soc != soc_pairs):
            raise ValueError(
                "molecule.coupling %r and the declared pair maps "
                "nac_pair_map=%r soc_pair_map=%r are one statement and these"
                " two disagree; fix one side."
                % (declared, preset_nac, preset_soc))
    else:
        nac_pairs, soc_pairs = build_spin_pair_maps(data['state_spin'])
    data['coupling'] = [list(pair) for pair in declared] or \
        [list(pair) for pair in nac_pairs] + \
        [list(pair) for pair in soc_pairs]

    data['nac_pair_map'] = nac_pairs
    data['soc_pair_map'] = soc_pairs
    data['nnac'] = len(nac_pairs)
    data['nsoc'] = len(soc_pairs)
    data['nac'] = [[[0.0, 0.0, 0.0]] * natom] * len(nac_pairs)
    data['soc'] = [[0.0]] * len(soc_pairs)

    return data


def classify_spin_pairs(pairs, state_spin):
    """Split declared coupling pairs into NAC and SOC pairs by spin.

    Mirrors the framework rule: same spin is a nonadiabatic coupling,
    different spin is a spin-orbit coupling.  A pair naming a state
    outside the ladder is kept as a NAC pair, which is what the
    framework does with it, so both sides agree on the row count.

    Parameters
    ----------
    pairs : list of list of int
        Declared 0-based state pairs.
    state_spin : list of int
        Sorted spin quantum number (2S) per state.

    Returns
    -------
    tuple of (list, list)
        (nac_pairs, soc_pairs), each in declaration order.
    """
    nac_pairs = []
    soc_pairs = []
    for pair in pairs:
        first, second = pair[0], pair[1]
        if (0 <= first < len(state_spin) and 0 <= second < len(state_spin)
                and state_spin[first] != state_spin[second]):
            soc_pairs.append(pair)
        else:
            nac_pairs.append(pair)
    return nac_pairs, soc_pairs


def derive_soc_settings(data: dict) -> dict:
    """Resolve the spin-ladder sizes and the spin-orbit run options.

    `nstate_s` / `nstate_t` size the singlet and triplet ladders of a
    two-manifold run.  Left at 0 they take the number of states of that
    spin the input declares, so the common case needs no second count;
    a value given by the user is kept as-is and checked in validate.
    `soc_2e` arrives as None when the input says nothing, and takes
    OpenQP's own default of 1 (the two-electron mean-field term on).

    Parameters
    ----------
    data : dict
        Configuration dictionary, after the state spin is settled.

    Returns
    -------
    dict
        Updated configuration with nstate_s, nstate_t and soc_2e set.
    """
    state_spin = data.get('state_spin') or []
    for label, spin in (('nstate_s', 0), ('nstate_t', 2)):
        if not data.get(label):
            data[label] = state_spin.count(spin)

    if data.get('soc_2e') is None:
        data['soc_2e'] = 1
    data['dosoc'] = 1 if data.get('dosoc') else 0

    return data


def build_spin_pair_maps(state_spin):
    """Build the NAC and SOC pair maps based on the state spins.

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


def derive_scf_type(data: dict) -> dict:
    """Derive the OpenQP [scf]type from method and multiplicity.

    Rules:
    - mrsf: always rohf with a triplet reference (multiplicity=3 is
      enforced by the input builder).
    - multiplicity == 1: rhf
    - multiplicity > 1: rohf

    An explicitly provided scf_type is kept unless the mrsf method
    forces rohf.

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with derived scf_type.
    """
    method = data['method']
    multiplicity = data['multiplicity']
    scf_type = data.get('scf_type', '').lower()

    if method == 'mrsf':
        data['scf_type'] = 'rohf'
    elif scf_type in ('rhf', 'rohf', 'uhf'):
        data['scf_type'] = scf_type
    elif multiplicity == 1:
        data['scf_type'] = 'rhf'
    else:
        data['scf_type'] = 'rohf'

    return data


def derive_td_type(data: dict) -> dict:
    """Derive the OpenQP [tdhf]type from the method.

    Rules:
    - mrsf: always mrsf
    - tddft: rpa or tda (default tda)
    - hf: no tdhf section, td_type is irrelevant

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with derived td_type.
    """
    method = data['method']
    td_type = data.get('td_type', 'tda').lower()

    if method == 'mrsf':
        data['td_type'] = 'mrsf'
    elif method == 'tddft':
        if td_type not in ('rpa', 'tda'):
            raise ValueError(f"tddft method requires td_type rpa or tda, got '{td_type}'.")
        data['td_type'] = td_type
    else:
        data['td_type'] = ''

    return data


def derive_functional(data: dict) -> dict:
    """Derive the DFT functional from the method.

    Rules:
    - hf: no functional; force an empty string (the schema default
      bhhlyp must not leak into hf calculations)
    - tddft/mrsf: keep the user value (validated elsewhere)

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with derived functional.
    """
    if data['method'] == 'hf':
        data['functional'] = ''

    return data


def derive_nac_type(data: dict) -> dict:
    """Normalize the NAC calculation type for dynamics profiles.

    Rules (*_nonadiabatic_dynamics profiles):
    - 'dcm' -> nacme through MO overlap (runtype=prop, needs the
      previous-step orbital json via [guess]file2).
    - 'ktdc' (and a missing value) -> per-step energies + gradients
      only (runtype=data); the framework kTDC integrator estimates
      the coupling from the energy curvature.
    - 'nac' -> numerical NAC vectors, still on runtype=data, which is
      the path compute_data answers them on ([properties] nac=nac plus
      the [nac] displacement control).  Serving a framework "energy
      gradient nac" request as energies and gradients only would leave
      the integrator without the coupling it asked for.
    Other profiles keep the user value (the *_nac single-point
    profiles drive runtype=nac themselves).

    Parameters
    ----------
    data : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Updated configuration with the normalized nac_type.
    """
    profile = data.get('profile', '')
    if profile.endswith('_nonadiabatic_dynamics'):
        nac_type = str(data.get('nac_type') or 'ktdc').strip().lower()
        if nac_type not in ('dcm', 'nac'):
            # 'ktdc' and anything unrecognised take the plain
            # energy+gradient data path.
            nac_type = 'ktdc'
        data['nac_type'] = nac_type

    return data
