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

This module is the main parsing entry point for ORCA results.
It selects the appropriate parser based on the calculation profile
and zero-fills missing data, keeping the state-resolved shape
contract shared with the other skills.

"""

import os
import numpy as np

from .engrad_parser import parse_engrad
from .log_parser import parse_log, parse_hessian, parse_hess_frequencies, parse_xyz
from .property_parser import parse_property
from ..utils import writes_engrad, writes_nacme

# 1 eV in Hartree (used to convert absorption excitation energies)
EV_TO_HARTREE = 1.0 / 27.211386245988

# Upper bound for the additive correction recovered from the log; a
# larger value means the reference energies were mis-read, and the
# historic positional mapping is safer than a shifted surface.
MAX_CORRECTION = 0.02

# Residual (Eh) identifying which root FINAL SINGLE POINT ENERGY and
# the property state energies refer to.
ROOT_MATCH_TOLERANCE = 1e-5


def _extract_velocities(config, calcdir):
    """Extract the final velocities (reserved fin_velocity channel).

    The orca skill performs no skill-internal MD propagation that
    reports final velocities, so the fin_velocity return channel is
    reserved with an empty value; the uniform channel keeps every
    skill result contract aligned.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.

    Returns
    -------
    list
        Empty list (no velocity output).
    """
    return []


def parse_output(config, calcdir: str, workflow) -> dict:
    """Parse ORCA output files based on the calculation profile.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Parsed results dictionary with numpy arrays:
        energy, gradient, nac, soc, dipole, transition_dipole,
        excitation, osc_strength, and optional
        frequencies/hessian/fin_coord/opt_converge fields.
    """
    project = config.project
    nstate = config.nstate
    natom = config.natom
    target_states = sorted(config.target_states)

    # Transition properties are extracted in transition_pair order
    # (npair, 3), keeping the pair-driven contract shared with the
    # molcas/pyscf skills; a single zero row when no valid pair.
    transition_pairs = [list(pair) for pair in config.transition_pair
                        if len(pair) == 2]
    # No row for a pair nobody declared: the `else 1` used to hand back one
    # zero transition dipole / oscillator strength for a run that asked for no
    # transition property at all (#448).
    npair = len(transition_pairs)

    results = {
        'energy': np.zeros(nstate),
        'gradient': np.zeros((nstate, natom, 3)),
        # The coupling channels start empty: a row exists only for a pair the
        # engine printed (see the NAC/SOC packing below).
        'nac': np.empty((0, natom, 3)),
        'soc': np.empty(0),
        'dipole': np.zeros((nstate, 3)),
        'transition_dipole': np.zeros((npair, 3)),
        'excitation': np.zeros(nstate),
        'osc_strength': np.zeros(npair),
        # Reserved fin_velocity return channel (no velocity output)
        'fin_velocity': _extract_velocities(config, calcdir),
    }

    profile = workflow.profile
    log_path = os.path.join(calcdir, f'{project}.out')

    # Ground-state DFT profiles read the .engrad file for
    # energy/gradient; the log provides dipoles.
    if workflow.method == 'dft' and writes_engrad(workflow, config):
        engrad_path = os.path.join(calcdir, f'{project}.engrad')
        energy, gradient = parse_engrad(engrad_path, natom)
        if energy[0] != 0.0:
            results['energy'][0] = energy[0]
        results['gradient'][0] = gradient

    # Parse the log for all profiles (missing sections stay zero)
    data = parse_log(log_path, config)

    # Energies: S0 from the log reference energy, excited states from
    # TDDFT excitation energies.
    if workflow.method == 'tddft':
        prop_path = os.path.join(calcdir, f'{project}.property.txt')
        prop = parse_property(prop_path)
        _pack_tddft_results(results, data, config, workflow, prop)
    elif data['s0_energy'] is not None and workflow.method == 'dft':
        if not writes_engrad(workflow, config):
            results['energy'][0] = data['s0_energy']

    # Dipole moment (ground-state SP)
    if workflow.has_step('dipole'):
        results['dipole'][0] = data['dipole']

    # Nonadiabatic couplings: fill in target_nac_pair order (S0-SN). A run
    # that printed no coupling - because the engine was never asked, or
    # refused the combination - reports no rows rather than the zeros the
    # array was pre-allocated with: a zero vector reads to the framework as
    # "the two surfaces are uncoupled", which is a physics claim the run did
    # not earn.
    if workflow.has_step('nac'):
        if data['nac_blocks']:
            _pack_nac(results, data, config)
        elif writes_nacme(workflow, config):
            # The deck asked for the coupling and the log carries no vector:
            # the reader's header is wrong or the run failed after TD-DFT.
            # Naming it beats returning nothing for a switch that was written.
            raise RuntimeError(
                'ORCA: the deck carries `NACME true` but %s holds no '
                'derivative-coupling vector; requested pairs %r'
                % (log_path,
                   [pair for pair in config.target_nac_pair if len(pair) == 2]))
        else:
            results['nac'] = np.empty((0, natom, 3))

    # Spin-orbit couplings: fill in target_soc_pair order (cm^-1)
    if workflow.is_soc:
        _pack_soc(results, data, config)

    # Frequencies and Hessian: the frequency list comes from the
    # $vibrational_frequencies block of {project}.hess (the same
    # layout the legacy wigner sampler read); the log's VIBRATIONAL
    # FREQUENCIES section only serves as a fallback. Missing
    # frequencies in both sources is a contract violation.
    if workflow.is_frequency:
        hess_path = os.path.join(calcdir, f'{project}.hess')
        frequencies = parse_hess_frequencies(hess_path)
        if frequencies.size == 0 and data['frequencies']:
            frequencies = np.array(data['frequencies'])
        if frequencies.size == 0:
            raise RuntimeError(
                f'ORCA {workflow.profile} finished but no vibrational '
                f'frequencies were found in {hess_path} or '
                f'{log_path}; check the calculation output.'
            )
        results['frequencies'] = frequencies
        hessian = parse_hessian(hess_path)
        if hessian.size > 0:
            results['hessian'] = hessian
        if data['s0_energy'] is not None:
            results['energy'][0] = data['s0_energy']

    # Geometry optimization: ORCA writes the input geometry into
    # {project}.inp and updates {project}.xyz after every cycle, so
    # the relaxed structure is read from the xyz file; the last
    # CARTESIAN COORDINATES block of the log only serves as a
    # fallback. The result goes through the framework fin_coord
    # channel (natom, 3) in Angstrom, and the convergence flag
    # follows the opt_converge contract. Missing final geometry is a
    # contract violation and fails fast.
    if workflow.is_optimization:
        xyz_path = os.path.join(calcdir, f'{project}.xyz')
        final_coordinates = parse_xyz(xyz_path)
        if not final_coordinates:
            final_coordinates = data['final_coordinates']
        if not final_coordinates:
            raise RuntimeError(
                f'ORCA {workflow.profile} finished but no final '
                f'geometry was found in {xyz_path} or {log_path} '
                f'(calcdir: {calcdir}).'
            )
        fin_coord = np.array(final_coordinates)
        if len(fin_coord) != natom:
            raise ValueError(
                f'ORCA final geometry in {xyz_path} has '
                f'{len(fin_coord)} atoms, but the system has '
                f'{natom}.'
            )
        results['fin_coord'] = fin_coord
        if data['s0_energy'] is not None:
            istate = target_states[0]
            if 0 <= istate < nstate:
                results['energy'][istate] = data['s0_energy']
        results['opt_converge'] = data['opt_converged']

    return results


def _gradient_block_order(config):
    """Build the framework-state order of the printed gradient blocks.

    ORCA selects singlet gradient roots via irootlist (root 0 is the
    ground state) and triplet roots via trootlist; both lists are
    sorted ascending. ORCA prints one block per requested root in
    the order irootlist[1:] + trootlist + [0] when the ground state
    is requested, otherwise irootlist + trootlist. Triplet entries
    carry the state offset (framework state nsing_root + root).

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    list of int
        Framework state index per gradient block, in log order.
    """
    nsing_root = config.nsing_root
    target_states = sorted(config.target_states)
    irootlist = [s for s in target_states if s <= nsing_root]
    trootlist = [s for s in target_states if s > nsing_root]

    if irootlist and irootlist[0] == 0:
        return irootlist[1:] + trootlist + [0]
    return irootlist + trootlist


def _resolve_state_energies(data, prop, state_energies, nstate):
    """Resolve the state energies onto the true ground-state surface.

    ORCA reports the raw SCF energy and the per-root corrections in
    separate places, and ``FINAL SINGLE POINT ENERGY`` is the energy
    of the state whose nuclear gradient was printed last (the ground
    state for a DFT gradient, an excited root otherwise). Three
    sources are tried in order of precision:

    1. ``{project}.property.txt``: ground-state SCF energy plus the
       labelled per-root totals, with the enumerated correction terms
       added back. Trusted only when one of the corrected roots
       reproduces the final single point energy, which proves no
       unenumerated correction term (solvation, relativity) is in
       play.
    2. The log's ``CIS/TD-DFT TOTAL ENERGY`` block: the corrections
       are recovered by subtracting the labelled excitation of the
       matching root from the final single point energy, so the
       returned energies stay consistent with the returned gradient.
    3. Neither source usable: the historic behaviour that treats the
       final single point energy as the ground-state reference.

    Parameters
    ----------
    data : dict
        Raw parsed log data.
    prop : dict
        Raw parsed property data (empty entries when unavailable).
    state_energies : list of float
        Excitation energies (Eh) in framework state order.
    nstate : int
        Number of framework states.

    Returns
    -------
    list
        Total energy per framework state, length nstate; entries no
        source covers stay None.
    """
    final = data['s0_energy']
    excitations = list(state_energies[:max(nstate - 1, 0)])

    def with_ladder(ground):
        """Extend a ground-state energy with the excitation ladder."""
        energies = [None] * nstate
        energies[0] = ground
        for n, exc in enumerate(excitations):
            energies[n + 1] = ground + exc
        return energies

    e_scf = prop['e_scf']
    if e_scf is not None and prop['state_totals']:
        corrections = prop['corrections'] or 0.0
        totals = prop['state_totals']
        consistent = (final is None
                      or min(abs(t + corrections - final)
                             for t in totals.values()) <= ROOT_MATCH_TOLERANCE)
        if consistent and abs(corrections) < MAX_CORRECTION:
            energies = with_ladder(e_scf + corrections)
            for state, total in totals.items():
                if 1 <= state < nstate:
                    energies[state] = total + corrections
            return energies

    if data['escf_energy'] is not None and final is not None:
        reference = final - data['escf_energy']
        dispersion = data['dispersion']
        labelled = data['root_excitations']
        if dispersion is not None and labelled:
            residuals = {root: abs(reference - dispersion - de)
                         for root, de in labelled.items()}
            root = min(residuals, key=residuals.get)
            if residuals[root] <= ROOT_MATCH_TOLERANCE:
                reference -= labelled[root]
        if abs(reference) < MAX_CORRECTION:
            return with_ladder(data['escf_energy'] + reference)

    s0 = final if final is not None else 0.0
    return with_ladder(s0)


def _pack_tddft_results(results: dict, data: dict, config, workflow,
                        prop: dict):
    """Pack TDDFT energies, gradients, and spectral data.

    Parameters
    ----------
    results : dict
        Results dictionary to fill in place.
    data : dict
        Raw parsed log data.
    config : SkillConfig
        Runtime configuration container.
    workflow : WorkflowGraph
        Compiled workflow graph.
    prop : dict
        Raw parsed property data.
    """
    nstate = config.nstate
    natom = config.natom

    # Excited-state excitations; fall back to the absorption spectrum
    # excitation energies (eV -> Eh).
    state_energies = data['state_energies']
    if not state_energies and data['absorption']:
        state_energies = [row['excitation'] * EV_TO_HARTREE
                          for row in data['absorption']]

    for state, energy in enumerate(
            _resolve_state_energies(data, prop, state_energies, nstate)):
        if energy is not None:
            results['energy'][state] = energy

    for k, exc in enumerate(state_energies[:nstate - 1]):
        results['excitation'][k + 1] = exc * 27.211386245988  # au -> eV

    # Transition dipoles and oscillator strengths extracted in
    # transition_pair order (npair, 3). The absorption state number
    # is 1-based in the ORCA log, i.e. the 0-based global excited
    # state index. Only ground-to-state pairs carry computed values;
    # ORCA does not compute excited-state to excited-state
    # transitions, so other pairs stay zero.
    absorption_by_state = {row['state']: row
                           for row in data['absorption']}
    pairs = [pair for pair in config.transition_pair if len(pair) == 2]
    for k, pair in enumerate(pairs):
        if k >= results['transition_dipole'].shape[0]:
            break
        a, b = pair
        if a != 0:
            continue
        row = absorption_by_state.get(b)
        if row is not None:
            results['transition_dipole'][k] = row['transition_dipole']
            results['osc_strength'][k] = row['osc_strength']

    # Excitation energies stay state-resolved (nstate,); fall back to
    # the absorption spectrum rows when no STATE lines were printed.
    for row in data['absorption']:
        s = row['state']
        if 0 <= s < nstate and results['excitation'][s] == 0.0:
            results['excitation'][s] = row['excitation']

    # TDDFT gradients: one CARTESIAN GRADIENT block per requested
    # root, printed in the order derived from irootlist/trootlist
    # (see _gradient_block_order); the block count must match.
    gradients = data['gradients']
    if not writes_engrad(workflow, config) \
            and workflow.has_step('gradient') and not workflow.is_frequency:
        # The run asked for a gradient, and this deck cannot carry one: ORCA
        # 6.1.1 aborts when `EnGrad` and `NACME true` share a deck (measured,
        # round `tmp_20261101_221306`, return code 64), and the coupling is
        # what the run is for. Absence is reported as absence - a zero
        # surface would be read as a measured flat one.
        results['gradient'] = np.empty((0, config.natom, 3))
    if writes_engrad(workflow, config):
        order = _gradient_block_order(config)
        if len(gradients) != len(order):
            raise RuntimeError(
                f'ORCA {workflow.profile} printed {len(gradients)} '
                f'gradient blocks but {len(order)} roots were '
                f'requested (target_states {sorted(config.target_states)}); '
                f'check the calculation output.'
            )
        for state, block in zip(order, gradients):
            results['gradient'][state] = np.array(block)


def _pack_nac(results: dict, data: dict, config):
    """Pack nonadiabatic coupling vectors in target_nac_pair order.

    Parameters
    ----------
    results : dict
        Results dictionary to fill in place.
    data : dict
        Raw parsed log data.
    config : SkillConfig
        Runtime configuration container.
    """
    target_nac_pair = [pair for pair in config.target_nac_pair
                       if len(pair) == 2]
    blocks = data['nac_blocks']
    if not target_nac_pair:
        results['nac'] = np.empty((0,) + results['nac'].shape[1:])
        return

    # Row k is the k-th requested pair. The printed block names no root, so a
    # count that does not match the request cannot be mapped onto the pairs -
    # and a row left as the zero the array was allocated with would read as
    # "these two surfaces are uncoupled".
    if len(blocks) != len(target_nac_pair):
        raise RuntimeError(
            'ORCA: %d non-adiabatic coupling block(s) were printed for %d '
            'requested pair(s) %r; the rows are addressed by the deck root '
            'order, so this log cannot be mapped onto the request.'
            % (len(blocks), len(target_nac_pair), target_nac_pair))
    results['nac'] = np.array(blocks)


def _map_orca_socme_pair(triplet_root: int, singlet_root: int, config):
    """Map one SOCME table row onto framework state indices.

    The table ORCA 6.1.1 prints in a DoSOC run numbers its two columns
    **separately** - the triplet root from 1 and the singlet root from 0,
    where 0 is the ground state (measured on round `tmp_20261101_221306`)::

        T      S
        1      0

    The framework orders its states as S0, the excited singlets, then the
    triplets, so the singlet root *is* the framework index and the T-th
    triplet root sits at `nsinglet + T - 1`.

    This replaces a combined singlet-first numbering that was an assumption
    about a log nobody had read; the assumption is why the reader matched
    nothing and the packer then reported zeros.

    Parameters
    ----------
    triplet_root : int
        1-based triplet root from the table's first column.
    singlet_root : int
        0-based singlet root from the table's second column (0 = S0).
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    tuple of (int, int) or None
        Framework (singlet, triplet) state indices, or None for a root the
        framework state list does not carry.
    """
    nsinglet = int(getattr(config, 'nsing_root', 0)) + 1
    if triplet_root < 1 or singlet_root < 0:
        return None
    return singlet_root, nsinglet + triplet_root - 1


def _pack_soc(results: dict, data: dict, config):
    """Pack spin-orbit couplings in target_soc_pair order.

    ORCA state indices from the SOC section are mapped onto the
    framework state indices first (see _map_orca_socme_pair), because
    the combined singlet+triplet root space does not line up with
    the framework state list when the two counts differ.

    Parameters
    ----------
    results : dict
        Results dictionary to fill in place.
    data : dict
        Raw parsed log data.
    config : SkillConfig
        Runtime configuration container.
    """
    target_soc_pair = [pair for pair in config.target_soc_pair
                       if len(pair) == 2]
    # No fallback onto config.soc_pair_map: that table was the skill's own
    # enumeration of the spin ladder, so reading it here addressed rows by an
    # index space the framework never declared (#448).

    soc_matrix = {}
    for triplet_root, singlet_root, value in data['soc_triples']:
        mapped = _map_orca_socme_pair(triplet_root, singlet_root, config)
        if mapped is None:
            continue
        a, b = mapped
        soc_matrix[(a, b)] = value
        soc_matrix[(b, a)] = value

    if not target_soc_pair:
        # Nothing asked for it: no rows, not a zero-length answer.
        results['soc'] = np.empty(0)
        return

    values = []
    for pair in target_soc_pair:
        a, b = pair
        if (a, b) not in soc_matrix:
            # A pair the log prints no element for is a missing coupling, not
            # a zero one: `soc_matrix.get(..., 0.0)` used to answer "these two
            # states do not couple", which the ISC rate of a surface-hopping
            # run then reads as its own input.
            raise RuntimeError(
                'ORCA: SOC pair %r was requested but this log prints no '
                'spin-orbit matrix element for it (elements read for %s); '
                'both states of a requested pair must be inside the roots '
                'this run solved.'
                % (list(pair), sorted(soc_matrix.keys()) or 'nothing'))
        values.append(soc_matrix[(a, b)])
    results['soc'] = np.array(values, dtype=float)


def format_results(results: dict, config) -> dict:
    """Format parsed results for output.

    Converts numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Raw parsed results.
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    dict
        Formatted results with numpy arrays converted to lists.
    """
    formatted = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            formatted[key] = value.tolist()
        else:
            formatted[key] = value

    return formatted
