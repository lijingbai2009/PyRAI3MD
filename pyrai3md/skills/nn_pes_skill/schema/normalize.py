"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides the normalization functions for the nn_pes skill,
including the geometry check and the state and coupling bookkeeping.

The networks carry no electronic-structure information, so electron
count and multiplicity verification does not apply here.

"""

from .config import SkillConfig, System, Calculation, Properties
from .reference import STATE_LABEL


def _pair_list(raw):
    """Read a declared state-pair list into integer pairs.

    Entries that are not a pair are dropped: a per-center nested form
    carries its pairs one level deeper, and an empty row declares nothing.
    """
    pairs = []
    for entry in (raw or []):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            pairs.append([int(entry[0]), int(entry[1])])
    return pairs


def _requested_pairs(pairs, nstate, keyword):
    """Keep the pairs a caller asked to have reported, in its order.

    A row that is not a pair is skipped rather than guessed at, because a
    half-declared request has no reading that is safe by default.

    Parameters
    ----------
    pairs : list
        Requested state pairs.
    nstate : int
        Number of states, which bounds the indices.
    keyword : str
        Field name, for the message.

    Returns
    -------
    list of list of int
        The valid pairs, in the order requested.

    Raises
    ------
    ValueError
        If a state index falls outside the state ladder.
    """
    valid = []
    for pair in pairs or []:
        if len(pair) != 2:
            continue
        a, b = pair
        if not (0 <= a < nstate and 0 <= b < nstate):
            raise ValueError(
                "%s contains a state index outside the %d state(s) the "
                "config declares." % (keyword, nstate))
        valid.append([int(a), int(b)])
    return valid


def same_spin(pair, spins):
    """True when a declared pair joins two states of one spin.

    Same-spin pairs are the non-adiabatic ones; different-spin pairs are the
    spin-orbit ones. Without spin information every pair is treated as
    non-adiabatic, which is what a singlet-only run means.

    Parameters
    ----------
    pair : list of int
        State indices of one declared pair.
    spins : list of int
        Spin label per state.

    Returns
    -------
    bool
        True when the two states carry the same spin.
    """
    if len(spins) <= max(pair):
        return True
    return spins[pair[0]] == spins[pair[1]]


def coupling_pairs(nstate: int) -> list:
    """Return the adjacent-pair coupling map of a state ladder.

    A fallback for the case where nothing was declared: two neighboring
    states are coupled, which is the convention the framework itself
    starts from. It is not a derivation of the model's shape. The
    channels a coupling network was trained on are the pairs the caller
    listed, and the framework passes them in `coupling`, from which it
    builds `nac_pair_map` and `nnac`. Those declare the dimension; this
    function only labels a run that declared nothing.

    Parameters
    ----------
    nstate : int
        Number of adiabatic states of the trained model.

    Returns
    -------
    list of list of int
        (nnac, 2) pair map, channel order first.
    """
    return [[n, n + 1] for n in range(max(0, nstate - 1))]


def normalize(config: SkillConfig) -> SkillConfig:
    """Normalize the input config for the neural network potential.

    Bridges between the nested SkillConfig and the flat-dict verify_*
    functions: extracts a work dict from submodels, runs verification,
    then writes results back.

    Parameters
    ----------
    config : SkillConfig
        Raw validated config.

    Returns
    -------
    SkillConfig
        Normalized config with derived fields populated.
    """
    work = {}
    work.update(config.system.model_dump())
    work.update(config.calculation.model_dump())
    work.update(config.properties.model_dump())

    work = verify_system(work)
    work = verify_state_num(work)

    system_updates = {k: work[k] for k in System.model_fields if k in work}
    calc_updates = {k: work[k] for k in Calculation.model_fields if k in work}
    props_updates = {k: work[k] for k in Properties.model_fields if k in work}
    # 'nac' names the served couplings here and the coupling role's
    # hyperparameters in Calculation. The placeholder arrays derived above
    # belong to Properties only; a role section is input and passes through
    # untouched.
    calc_updates['eg'] = config.calculation.eg
    calc_updates['nac'] = config.calculation.nac
    calc_updates['soc'] = config.calculation.soc

    return config.model_copy(update={
        'system': config.system.model_copy(update=system_updates),
        'calculation': config.calculation.model_copy(update=calc_updates),
        'properties': config.properties.model_copy(update=props_updates),
    })


def verify_system(data: dict) -> dict:
    # A potential needs a geometry it can be asked about.
    atoms = data['atoms']
    coordinates = data['coordinates']

    if len(atoms) != len(coordinates):
        raise ValueError("The number of atoms and coordinates do not match.")

    if len(atoms) < 2:
        raise ValueError(
            "A neural network potential needs at least two atoms to "
            "build geometric features from.")

    for coordinate in coordinates:
        if len(coordinate) != 3:
            raise ValueError("Coordinates must be (natom, 3).")

    data['natom'] = len(atoms)
    return data


def verify_state_num(data: dict) -> dict:
    # State bookkeeping of the trained ensemble
    nstate = data['nstate']
    state_spin = data['state_spin']
    target_states = data['target_states']
    target_nac_pair = data['target_nac_pair']
    target_soc_pair = data.get('target_soc_pair') or []
    coupling = data.get('coupling') or []

    if nstate < 1:
        raise ValueError("nstate must be at least 1.")

    if len(target_states) == 0:
        target_states = list(range(nstate))
    target_states = sorted(set(target_states))

    if max(target_states) + 1 > nstate:
        raise ValueError("target_states must be less than nstate.")

    if len(state_spin) == 0:
        # The networks carry no spin information
        state_spin = [0] * nstate
    if len(state_spin) != nstate:
        raise ValueError("nstate does not match the number of state spins.")

    # Which state pairs the trained channels stand for is a declaration,
    # not a derivation. The framework builds `nac_pair_map` and
    # `soc_pair_map` (and the two counts) from its coupling section by
    # splitting it on the state spins, and passes them down; when the skill
    # is driven directly the same list arrives as `coupling`, and it is
    # split the same way here. The coupling network is sized by that count,
    # so nothing enumerates a pair set of its own. Without a map the count
    # is still known, and the channels can only be served in the order the
    # model carries them.
    coupling = _pair_list(data.get('coupling'))
    spins = [int(x) for x in state_spin]
    declared_nac = _pair_list(data.get('nac_pair_map'))
    declared_soc = _pair_list(data.get('soc_pair_map'))
    if not declared_nac and not declared_soc and coupling:
        declared_nac = [pair for pair in coupling if same_spin(pair, spins)]
        declared_soc = [pair for pair in coupling if not same_spin(pair, spins)]

    configured = int(data.get('nnac') or 0)
    if declared_nac:
        nac_pairs = declared_nac
        nnac = len(declared_nac)
    elif configured > 0:
        # The count is known but no pair was named for it: the channels
        # can only be served in the order the model carries them.
        nac_pairs = []
        nnac = configured
    else:
        nac_pairs = coupling_pairs(nstate)
        nnac = len(nac_pairs)

    configured_soc = int(data.get('nsoc') or 0)
    if declared_soc:
        soc_pairs = declared_soc
        nsoc = len(declared_soc)
    elif configured_soc > 0:
        soc_pairs = []
        nsoc = configured_soc
    else:
        soc_pairs = []
        nsoc = 0

    valid_nac_pair = _requested_pairs(target_nac_pair, nstate,
                                      'target_nac_pair')
    valid_soc_pair = _requested_pairs(target_soc_pair, nstate,
                                      'target_soc_pair')

    data['target_states'] = sorted(set(target_states))
    data['target_nac_pair'] = valid_nac_pair
    data['target_soc_pair'] = valid_soc_pair
    data['state_spin'] = spins
    data['state_label'] = [STATE_LABEL.get(spin, 's') for spin
                           in data['state_spin']]

    data['nac_pair_map'] = nac_pairs
    data['soc_pair_map'] = soc_pairs
    data['nnac'] = nnac
    data['nsoc'] = nsoc
    data['npair'] = len(valid_nac_pair)
    data['nspair'] = len(valid_soc_pair)

    natom = data['natom']
    data['energy'] = [0.0] * nstate
    data['gradient'] = [[[0.0, 0.0, 0.0] for _ in range(natom)]
                        for _ in range(nstate)]
    data['nac'] = [[[0.0, 0.0, 0.0] for _ in range(natom)]
                   for _ in range(nnac)]
    # One spin-orbit value is one scalar per declared pair, in cm-1: the
    # matrix element depends on geometry but is not itself a derivative.
    data['soc'] = [[0.0] for _ in range(nsoc)]

    return data
