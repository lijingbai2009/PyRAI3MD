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

This module provides the normalization functions for the Molcas skill,
including verification of electronic structure, state numbers, and state spins.

"""

import math


from .config import SkillConfig, System, Properties, Calculation
from .reference import STATE_LABEL
from .reference import PERIODIC_TABLE
from .reference import DEFAULT
from .reference import CASPT2_TYPES


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for Molcas computation.

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
    work = verify_caspt2_settings(work)
    work = verify_embedding_data(work)

    # Verify the state spin and build state labels, NAC and SOC pair maps
    work = verify_state_num(work)
    work = verify_state_spin(work)
    work = verify_caspt2_contracts(work)

    # An unnamed `opt_state_j` means there is no second surface, so it resolves to
    # the state the run already follows. Left at a literal 0 it reads as "the user
    # named state 0", which for an excited-state relaxation (`opt_state_i: 1`)
    # turns a single-state job into one that looks like it asked for a crossing
    # pair - and this skill answers that by emitting a second RASSCF job and
    # reading two energy blocks per iteration.
    if work.get('opt_state_j') is None:
        work['opt_state_j'] = work.get('opt_state_i', 0)

    # Write results back to the corresponding submodels
    system_updates = {k: work[k] for k in System.model_fields if k in work}
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    props_updates = {k: work[k] for k in Properties.model_fields if k in work}

    return config.model_copy(update={
        'system': config.system.model_copy(update=system_updates),
        'calculation': config.calculation.model_copy(update=calc_updates),
        'properties': config.properties.model_copy(update=props_updates),
    })


def verify_caspt2_settings(data: dict) -> dict:
    """Resolve the CASPT2 variant and validate XDW parameters.

    ``caspt2_type`` is authoritative when set; otherwise the legacy
    ``xms`` switch resolves to xms/ss. The legacy switch is kept in
    sync so callers that still read it see a consistent value.
    """
    ctype = str(data.get('caspt2_type', '') or '').strip().lower()
    if ctype:
        if ctype not in CASPT2_TYPES:
            raise ValueError(
                "caspt2_type must be one of %s; got %r"
                % (list(CASPT2_TYPES), ctype))
        data['xms'] = ctype != 'ss'
    else:
        ctype = 'xms' if bool(data.get('xms', True)) else 'ss'
        data['caspt2_type'] = ctype

    if ctype == 'xdw':
        dwtype = int(data.get('dwtype', 3))
        if dwtype not in (1, 2, 3):
            raise ValueError("dwtype must be 1, 2, or 3; got %r" % (dwtype,))
        zeta = float(data.get('dwms_zeta', 1.0))
        if not math.isfinite(zeta) or zeta <= 0.0:
            raise ValueError(
                "dwms_zeta must be a positive finite number; got %r"
                % (zeta,))
        data['dwtype'] = dwtype
        data['dwms_zeta'] = zeta
    return data


def verify_embedding_data(data: dict) -> dict:
    """Normalize embedding/read_charge and validate external potential."""
    embedding = str(data.get('embedding', 'none') or 'none').strip().lower()
    if embedding == 'potential':
        raise ValueError(
            "embedding='potential' is retired: write 'field', the V/E block "
            "per QM atom. A charge file declares its own representation, so "
            "this key is only read when the data arrives as an array.")
    if embedding not in ('none', 'charge', 'field'):
        raise ValueError(
            "embedding must be none, charge, or field; got %r"
            % (data.get('embedding'),))
    read_charge = str(data.get('read_charge', 'none') or 'none').strip()
    if read_charge == 'legacy':
        project = str(data.get('project', 'molcas_calc') or 'molcas_calc')
        read_charge = '%s.charge' % project
    if read_charge == '':
        read_charge = 'none'
    # A file stands alone: its first line declares whether it holds point
    # charges or a field, so `embedding` has nothing to add.

    natom = int(data.get('natom', 0) or 0)
    potential = data.get('external_potential', []) or []
    if potential:
        if embedding not in ('field', 'none'):
            raise ValueError(
                "external_potential is the field representation; drop "
                "embedding='charge' (or name the file in read_charge, whose "
                "declaration decides).")
        if natom and len(potential) != natom:
            raise ValueError(
                "external_potential has %d rows but the system has %d "
                "atoms." % (len(potential), natom))
        for row in potential:
            if len(row) != 4:
                raise ValueError(
                    "each external_potential row must be "
                    "[V, Fx, Fy, Fz]; got %r" % (row,))
            if not all(math.isfinite(float(x)) for x in row):
                raise ValueError(
                    "external_potential contains a non-finite value: %r"
                    % (row,))

    data['embedding'] = embedding
    data['read_charge'] = read_charge
    return data


def verify_caspt2_contracts(data: dict) -> dict:
    """Enforce the bounded CASPT2 gradient/NAC contracts."""
    method = str(data.get('method', '') or '').lower()
    if method != 'caspt2':
        return data

    tokens = set(str(data.get('jobtype', '') or '').lower().split())
    profile = str(data.get('profile', '') or '').lower()
    has_nac = 'nac' in tokens or profile.endswith('_nac')
    has_gradient = 'gradient' in tokens or profile.endswith('_gradient')
    has_soc = ('soc' in tokens or profile.endswith('_soc')
               or any(len(p) == 2 for p in
                      (data.get('target_soc_pair') or []))
               or any(len(p) == 2 for p in
                      (data.get('soc_pair_map') or [])))

    ctype = str(data.get('caspt2_type', 'ss') or 'ss').lower()
    if ctype == 'ss':
        state_spin = data.get('state_spin') or []
        if has_soc:
            counts = {}
            for spin in state_spin:
                counts[spin] = counts.get(spin, 0) + 1
            if any(count > 1 for count in counts.values()):
                raise ValueError(
                    "caspt2_type='ss' with SOC requires one state per "
                    "spin manifold; got state_spin %r." % (state_spin,))
        elif len(state_spin) > 1:
            raise ValueError(
                "caspt2_type='ss' is single-state; use ms/xms/rms/xdw "
                "for multiple states.")
    if ctype == 'ss' and has_nac:
        raise ValueError(
            "caspt2_type='ss' does not support NAC; use ms, xms, rms, "
            "or xdw.")

    if has_gradient:
        target_states = [s for s in (data.get('target_states') or [])
                         if isinstance(s, int)]
        if not target_states:
            raise ValueError(
                "CASPT2 analytic gradient requires at least one target "
                "state.")

    if has_nac:
        pairs = [p for p in (data.get('target_nac_pair') or [])
                 if len(p) == 2]
        if not pairs:
            raise ValueError(
                "CASPT2 NAC requires at least one same-spin pair.")
        state_spin = data.get('state_spin') or []
        for pair in pairs:
            a, b = pair
            if (a < len(state_spin) and b < len(state_spin)
                    and state_spin[a] != state_spin[b]):
                raise ValueError(
                    "CASPT2 NAC requires same-spin pairs; got %r with "
                    "state_spin %r." % (pair, state_spin))

    state_spin = data.get('state_spin') or []
    if state_spin and len(set(state_spin)) > 1 and not has_soc:
        raise ValueError(
            "CASPT2 with mixed spin manifolds requires an explicit SOC "
            "request.")
    return data


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

    # Compute inactive orbital count: (total_electrons - active_electrons) / 2
    nact_elec = data.get('nact_elec', 0)
    data['inactive_orb'] = (int(num_electrons) - nact_elec) // 2

    # If ras2_orb is not explicitly set (0), default to nact_orb
    if data.get('ras2_orb', 0) == 0:
        data['ras2_orb'] = data.get('nact_orb', 2)

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

    # Sort target states so gradient blocks (and their parsed results)
    # follow one canonical order: singlet states first, then triplet.
    target_states = sorted(set(target_states))

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

    # The declaration is the authority: `&MOLECULE coupling` (0-based state
    # pairs) or, on the framework flow, the pair maps the framework already
    # derived from it.  Classifying the declaration is all that happens here -
    # nothing is added to it, and a pair the input did not name is not asked
    # for (#448).  The declared order is kept, because it is the order the
    # deck's rows and the framework's addressing both use.
    coupling = [list(pair) for pair in (data.get('coupling') or [])
                if len(pair) == 2]
    nac_map = [list(pair) for pair in (data.get('nac_pair_map') or [])
               if len(pair) == 2]
    soc_map = [list(pair) for pair in (data.get('soc_pair_map') or [])
               if len(pair) == 2]
    declared = coupling or (nac_map + soc_map)
    if declared:
        for pair in declared:
            for index in pair:
                if not 0 <= index < nstate:
                    raise ValueError(
                        "molecule.coupling: state %d is outside this run's"
                        " table of %d states." % (index, nstate))
        nac_from_coupling, soc_from_coupling = classify_declared_pairs(
            declared, state_spin)
        if (nac_map and nac_map != nac_from_coupling) or \
                (soc_map and soc_map != soc_from_coupling):
            raise ValueError(
                "molecule.coupling %r and the derived pair maps "
                "nac_pair_map=%r soc_pair_map=%r are one statement, and these"
                " two disagree." % (declared, nac_map, soc_map))
        data['coupling'] = declared
        data['nac_pair_map'] = nac_from_coupling
        data['soc_pair_map'] = soc_from_coupling
        data['nnac'] = len(nac_from_coupling)
        data['nsoc'] = len(soc_from_coupling)
        # The step asks for the declared pairs when it named none of its own.
        # `valid_nac_pair` is what the step actually carried: the submodel's
        # unset marker is `[[]]`, which is truthy, so testing
        # `data.get('target_nac_pair')` read "not filled" as "declared" and left
        # the run with an empty request beside a coupling nobody asked about
        # (#448 - measured on the six shipped CASPT2 NAC/prop cases, which
        # declare `coupling` only).
        if not valid_nac_pair:
            data['target_nac_pair'] = nac_from_coupling
        if not valid_soc_pair:
            data['target_soc_pair'] = soc_from_coupling
    elif valid_nac_pair or valid_soc_pair:
        # An explicit `target_nac_pair` / `target_soc_pair` with no coupling
        # list is the skill's own input vocabulary: take it as the
        # declaration.
        data['nac_pair_map'] = valid_nac_pair
        data['soc_pair_map'] = valid_soc_pair
        data['nnac'] = len(valid_nac_pair)
        data['nsoc'] = len(valid_soc_pair)

    data['target_states'] = target_states
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
    # re-numbers every declared pair, and the deck's rows are addressed by that
    # declaration (`service/builder/input_builder.py::nac_deck_order`,
    # `workflow/integrator/fssh.py`). Sorting therefore validates a standalone
    # input and is refused when the run already declared pairs against these
    # indices (#448).
    spins = [int(x) for x in state_spin]
    declared_pairs = [list(pair) for pair in
                      (list(data.get('coupling') or [])
                       + list(data.get('nac_pair_map') or [])
                       + list(data.get('soc_pair_map') or []))
                      if len(pair) == 2]
    if declared_pairs and spins != sorted(spins):
        raise ValueError(
            "Molcas orders states by ascending 2S; the input state_spin %r is"
            " not, and this run declares state pairs %r. Sort the state list in"
            " the input: re-ordering it here would permute the declared pairs"
            " and address the reported rows by an index other than the one the"
            " declaration used." % (spins, declared_pairs))
    data['state_spin'] = spins if declared_pairs else sorted(spins)
    state_label = []
    for i in range(data['nstate']):
        state_label.append(STATE_LABEL[data['state_spin'][i]])

    data['state_label'] = state_label

    # Nothing is enumerated here: with no declaration at all the run simply
    # has no couplings to request (the old cross product of the sorted ladder
    # invented pairs the input never named).
    if not data.get('nac_pair_map') and not data.get('soc_pair_map'):
        data['nac_pair_map'] = []
        data['soc_pair_map'] = []
        data['nnac'] = 0
        data['nsoc'] = 0
    # The requested NAC pairs keep the declaration's order. The deck prints its
    # ALASKA blocks spin-group by spin-group, and the parser maps the rows back
    # through the same function, so the two sides agree without either sorting
    # the declaration (that used to happen here and in the deck, twice).
    data['target_nac_pair'] = [list(pair) for pair in
                               (data.get('target_nac_pair') or [])
                               if len(pair) == 2]
    data['nac'] = [[[0.0, 0.0, 0.0]] * natom] * len(data['nac_pair_map'])
    data['soc'] = [[0.0]] * len(data['soc_pair_map'])

    # Auto-distribute singlet/triplet CI parameters for SOC dual RASSCF
    n_singlet = sum(1 for s in data['state_spin'] if s == 0)
    n_triplet = sum(1 for s in data['state_spin'] if s != 0)
    data['singlet_ci_roots'] = [n_singlet] if n_singlet > 0 else DEFAULT['singlet_ci_roots']
    data['singlet_ci_spin'] = [0] if n_singlet > 0 else []
    data['triplet_ci_roots'] = [n_triplet] if n_triplet > 0 else DEFAULT['triplet_ci_roots']
    data['triplet_ci_spin'] = [2] if n_triplet > 0 else []

    # Derive non-SOC ci_roots/ci_spin by grouping the sorted state_spin
    # by 2S value: the inverse of the framework ci/spin -> state_spin
    # expansion. build_rasscf reads these for normal (non-SOC) RASSCF.
    ci_roots = []
    ci_spin = []
    for two_s in data['state_spin']:
        if ci_spin and two_s == ci_spin[-1]:
            ci_roots[-1] += 1
        else:
            ci_spin.append(two_s)
            ci_roots.append(1)
    data['ci_roots'] = ci_roots
    data['ci_spin'] = ci_spin

    return data


def classify_declared_pairs(coupling, state_spin):
    """Split declared state pairs into NAC (same spin) and SOC (other spin).

    The same classification the framework applies to `molecule.coupling`
    (`data/center.py::classify_pairs`); it is local because the skill must not
    import framework internals, and the equivalence of the two is pinned by a
    criterion in `qualification/verification/test_state_declaration.py`.

    Parameters
    ----------
    coupling : list of [int, int]
        Declared 0-based state pairs, in the declaration's order.
    state_spin : list of int
        2S per state.

    Returns
    -------
    tuple of (list, list)
        (nac_pairs, soc_pairs), each keeping the declaration's order.
    """
    nac_pairs = []
    soc_pairs = []
    for pair in coupling:
        first, second = pair
        if state_spin[first] == state_spin[second]:
            nac_pairs.append(list(pair))
        else:
            soc_pairs.append(list(pair))
    return nac_pairs, soc_pairs


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


def derive_alter_pairs(active, inactive_orb, ras1_orb, ras2):
    """Derive ALTEr orbital interchange pairs from selected active orbitals.

    MOLCAS RASSCF classifies orbitals by count in the orbital-file
    order: the Ras2 (active) window occupies the positions
    inactive_orb + ras1_orb ... inactive_orb + ras1_orb + ras2 - 1
    (0-based). To place user-selected orbitals into that window, the
    ALTEr keyword interchanges orbital pairs before RASSCF reads them.

    Orbitals already shared between the selection and the window are
    locked (no swap); each selected orbital outside the window is
    paired with a window position not covered by the selection. The
    pairs are disjoint transpositions, so their order is irrelevant.

    The molcas skill does not use symmetry: the symmetry species of
    every pair is fixed to 1.

    Parameters
    ----------
    active : list of int
        Selected active orbital indices, 0-based (internal storage).
    inactive_orb : int
        Number of inactive (doubly occupied) orbitals.
    ras1_orb : int
        Number of Ras1 orbitals.
    ras2 : int
        Number of Ras2 (active) orbitals (effective value).

    Returns
    -------
    list of tuple
        Interchange pairs (sym, i, j) in 1-based MOLCAS notation,
        sym always 1. Empty when the selection matches the window.
    """
    window = set(range(inactive_orb + ras1_orb,
                       inactive_orb + ras1_orb + ras2))
    need_in = [orb for orb in active if orb not in window]
    need_out = [pos for pos in sorted(window)
                if pos not in set(active)]
    return [(1, orb + 1, pos + 1)
            for orb, pos in zip(need_in, need_out)]
