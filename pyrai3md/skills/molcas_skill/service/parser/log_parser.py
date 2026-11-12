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

This module parses Molcas log files to extract computational results.
It follows the parsing logic from classic qc_molcas.py _read_data method.

"""

import os
import re
import warnings

import numpy as np

# Unit conversion factors.
# Molcas prints dipoles and multipole matrix elements in atomic
# units; the framework log expects Debye.
AU2DEBYE = 2.541746473
# Molcas thermochemistry prints entropy in cal/(mol*K) although the
# section header claims kcal/(mol*K); the framework log expects
# Hartree/K.
CAL_PER_HARTREE = 627509.474


def _check_log_completion(lines: list) -> str:
    """Verify that a Molcas log terminated normally.

    A successfully finished calculation prints a 'Happy landing'
    banner and every module stops with a benign rc marker
    (/rc=0, _RC_ALL_IS_WELL_, _RC_INVOKED_OTHER_MODULE_ for a module
    that invoked a submodule, or _RC_CONTINUE_LOOP_ for loop-driven
    modules such as SLAPAF); a crashed module leaves a 'Non-zero
    return code' marker or a 'Stop Module ... /rc=N' line with an
    error code such as /rc=-6 or /rc=_RC_INPUT_ERROR_. Logs lacking
    the success marker are treated as incomplete so that zero-filled
    results never mask a failed calculation.

    A geometry optimization that exhausts its step budget stops
    SLAPAF with /rc=_RC_NOT_CONVERGED_. This is a result status, not
    a crash: Molcas still writes the final structure, so the check
    reports it instead of raising, and the caller records the
    unconverged state through the opt_converge contract.

    Parameters
    ----------
    lines : list
        Log file lines.

    Returns
    -------
    str
        'ok' for a normally finished run, 'not_converged' when the
        only anomaly is a SLAPAF _RC_NOT_CONVERGED_ stop.

    Raises
    ------
    RuntimeError
        When the log shows a failure marker or never reached the
        completion banner (and no unconverged stop was recorded).
    """
    # Benign rc markers a normally finished Molcas module may print.
    # Anything else in a 'Stop Module ... /rc=' line counts as a
    # failure (negative codes, _RC_INPUT_ERROR_, ...).
    _BENIGN_RC = (
        '_RC_ALL_IS_WELL_',
        '_RC_INVOKED_OTHER_MODULE_',
        '_RC_CONTINUE_LOOP_',
        '/rc=0',
    )
    not_converged_modules = [
        line.strip() for line in lines
        if 'Stop Module' in line
        and '/rc=' in line
        and '_RC_NOT_CONVERGED_' in line
    ]
    failed_modules = [
        line.strip() for line in lines
        if 'Stop Module' in line
        and '/rc=' in line
        and '_RC_NOT_CONVERGED_' not in line
        and not any(marker in line for marker in _BENIGN_RC)
    ]
    has_failure = any('Non-zero return code' in line for line in lines)
    has_success = any('Happy landing' in line for line in lines)

    # A genuine module crash always wins over the unconverged
    # tolerance: mixed anomalies are reported as failures.
    if has_failure or failed_modules:
        detail = (failed_modules[-1] if failed_modules
                  else 'non-zero return code reported')
        raise RuntimeError(
            f'Molcas calculation did not complete: {detail}'
        )
    if not_converged_modules:
        return 'not_converged'
    if not has_success:
        raise RuntimeError(
            'Molcas log lacks the completion marker (Happy landing); '
            'the calculation appears incomplete'
        )
    return 'ok'


def _workflow_runs_nac(workflow, config) -> bool:
    """Whether the compiled workflow executes a NAC request.

    The generic profiles carry the answer in the pass plan (and the
    task set); legacy profiles predate both and answer through the
    profile name. The jobtype is the last fallback for a direct
    parser call without a compiled workflow. State pairs alone are
    not evidence: the carrier derives a default coupling for every
    multistate case.

    Parameters
    ----------
    workflow : WorkflowGraph or None
        Compiled workflow, when available.
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    bool
        True when the run asks ALASKA for derivative couplings.
    """
    if workflow is not None:
        plans = getattr(workflow, 'pass_plan', None) or []
        if plans:
            return any(plan.get('nac') for plan in plans)
        tasks = getattr(workflow, 'tasks', frozenset()) or frozenset()
        if tasks:
            return 'nac' in tasks
        profile = str(getattr(workflow, 'profile', '') or '').lower()
        if profile.endswith('_nac'):
            return True
    tokens = set(str(getattr(config, 'jobtype', '') or '').lower().split())
    return 'nac' in tokens


def _read_energy_block(log, start, config, slot, block_index, nstate,
                       log_path):
    """Read the state energies of one `Final state energy(ies)` block.

    Scans rows until one stops parsing rather than taking a counted
    slice: a block holds the roots of ONE spin group, and how many that
    is is written by the print, not by `ci_roots`. Measured on the
    deployment's `casscf_soc` log (tmp_20261014_142736,
    tmp_MOLCAS/casscf_soc_test.log:879 and :1465): its first block
    carries two root rows and its second one, while `ci_roots` is
    [2, 1] - so reading every block with the FIRST group's count walks
    off the end of the second block into the orbital section below it,
    and the old `except: pass` swallowed exactly that, leaving the state
    table one surface short while the shape stayed legal.

    Parameters
    ----------
    log : list of str
        Log lines.
    start : int
        Index of the banner line.
    config : SkillConfig
        Runtime configuration (its ci_spin names the spin group).
    slot : int
        Block position in one round.
    block_index : int
        One-based block counter, for the refusal message.
    nstate : int
        Declared state count, for the refusal message.
    log_path : str
        Path, for the refusal message.

    Returns
    -------
    list of float
        The block's state energies.

    Raises
    ------
    RuntimeError
        When the block yields no number: a surface that printed nothing
        cannot be packed as zero.
    """
    if start + 3 < len(log) and '::    RASSCF root number' in log[start + 3]:
        shift_line = 3
        en_col = -1
    elif start + 5 < len(log) and len(log[start + 5].split()) > 1:
        shift_line = 5
        en_col = 1
    else:
        shift_line = 3
        en_col = -1

    energies = []
    index = start + shift_line
    while index < len(log):
        parts = log[index].split()
        if len(parts) < 2:
            break
        try:
            value = float(parts[en_col])
        except (ValueError, IndexError):
            break
        energies.append(value)
        index += 1
    if not energies:
        spin = (config.ci_spin[slot]
                if slot < len(config.ci_spin) else '?')
        raise RuntimeError(
            f'Molcas: RASSCF energy block {block_index + 1} (spin group '
            f'{spin}) yielded no state energy at line {start + 1} of '
            f'{log_path}; the run declares {nstate} state(s), and a '
            f'surface that printed nothing cannot be packed as zero'
        )
    return energies


def parse_log(log_path: str, config, workflow=None) -> dict:
    """Parse a Molcas log file and extract all available results.

    Parameters
    ----------
    log_path : str
        Path to the Molcas .log file.
    config : SkillConfig
        Runtime configuration container with state/coupling information.
    workflow : WorkflowGraph, optional
        Compiled workflow. When it declares ``two_pass``, the log is
        split at the second GATEWAY and pass 1 supplies energy,
        gradient and SOC while pass 2 supplies NAC.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f'Molcas log file not found: {log_path}. The calculation '
            f'produced no output.'
        )

    with open(log_path, 'r') as f:
        log = f.read().splitlines()

    # Reject failed or incomplete runs before extracting results so
    # zero-filled values never mask a calculation failure. An
    # unconverged optimization is a result status, not a failure: the
    # marker travels with the results (see parse_output).
    completion = _check_log_completion(log)

    method = str(getattr(config, 'method', '') or '').lower()
    two_pass = bool(getattr(workflow, 'two_pass', False)) if workflow else False
    split_index = None
    if two_pass:
        gateways = [i for i, line in enumerate(log)
                    if 'Start Module: gateway' in line]
        if len(gateways) < 2:
            raise RuntimeError(
                'Molcas two-pass log has %d GATEWAY sections; expected 2'
                % (len(gateways),))
        split_index = gateways[1]

    nstate = config.nstate
    natom = config.natom
    ci = config.ci_roots
    mult = [m + 1 for m in config.ci_spin]  # convert 2S to multiplicity
    soc_coupling = config.target_soc_pair
    target_states = config.target_states
    tasks = frozenset(getattr(workflow, 'tasks', ()) or ())
    expect_energy = 'energy' in tasks
    expect_gradient = 'gradient' in tasks

    # Fallback singlet/triplet configuration state counts derived
    # from state_spin; the RASSI log usually reports them ('Nr of
    # states' / 'Root nr:'), but the config values keep the SOC
    # matrix reduction correct when those lines are absent.
    state_spin = config.state_spin
    sin_fallback = sum(1 for s in state_spin if s == 0)
    tri_fallback = len(state_spin) - sin_fallback

    casscf = []
    scf_energy = None
    # Blocks per round comes from the same function that built the deck:
    # the reader used to re-derive it from the profile name, so a deck
    # that grouped by spin manifold had its second pass labelled with the
    # first pass's spin (#448).
    if workflow is None:
        # A standalone re-parse has no workflow handed to it. Compiling
        # the same one the builder was given is the single source; the
        # reader gets no vote on the deck's shape.
        from ..workflow.compiler import compile_workflow
        workflow = compile_workflow(config.profile, config=config)
    from ..builder.input_builder import deck_loops, rasscf_pass_count
    blocks_per_round = rasscf_pass_count(config, workflow)
    # Only a deck that reprints its blocks inside one input - SLAPAF's
    # `Do while`, the `FOREACH` dynamics loop - needs the "keep the last
    # round" reset. A property pass is evaluated once per call, and
    # resetting there would discard its first pass.
    blocks_per_iteration = blocks_per_round if deck_loops(workflow) else 0
    slot_count = max(1, blocks_per_round)
    gradient = []
    # Counts the 'Final state energy(ies)' blocks as they arrive, so which spin
    # group a block belongs to - and where a new optimization iteration starts -
    # is known while reading, instead of inferred from how many lines happened to
    # be collected.
    block_index = 0
    # Counts gradient blocks that did not read as a full (natom, 3) table. The
    # packing step is what decides whether one was needed; this is only so the
    # count is not lost between the scan and that decision.
    incomplete_gradients = 0
    nac = []
    soc = []
    soc_mtx = []
    soc_state = 0
    sin_state = sin_fallback
    tri_state = tri_fallback

    # Property containers: CASSCF dipoles are labeled by root number;
    # CASPT2 property blocks carry no label and appear in root order
    # (the k-th unlabeled block belongs to root k).
    casscf_dipoles = {}
    caspt2_dipoles = []
    osc_dict = {}
    mees_mats = {}
    gradient_post_espf = []
    caspt2_energies = []
    caspt2_final = []
    caspt2_variant_seen = False
    rassi_diagonals = []
    nac_total = []
    nac_ci = []
    warnings_out = []

    for i, line in enumerate(log):
        in_pass1 = (split_index is None or i < split_index)
        in_pass2 = (split_index is not None and i >= split_index)
        if in_pass1 and 'Final state energy(ies)' in line:
            slot = block_index % slot_count
            e = _read_energy_block(log, i, config, slot, block_index,
                                   nstate, log_path)
            if blocks_per_iteration and slot == 0 and (casscf or gradient):
                # A new geometry-optimization iteration begins: keep only the
                # last one, whose energies and forces describe the geometry the
                # run ended on. The first iteration's gradient is what the loop
                # held before this reset existed, and a search stepping on it
                # walks a geometry that is no longer the one it was handed.
                casscf = []
                gradient = []
            casscf += e
            block_index += 1

        elif in_pass1 and 'FINAL CASPT2 RESULT' in line:
            for row in log[i + 1:i + 12]:
                if 'Total energy:' in row:
                    try:
                        caspt2_final.append(float(row.split()[-1]))
                    except ValueError:
                        pass
                    break

        elif in_pass1 and _is_caspt2_energy_header(line):
            variant = _caspt2_variant(line)
            if variant != 'generic':
                caspt2_variant_seen = True
            if variant != 'generic' or not caspt2_variant_seen:
                values = _parse_caspt2_root_energies(log, i + 1, 100)
                if values:
                    caspt2_energies.extend(values)

        elif in_pass1 and 'HAMILTONIAN MATRIX FOR THE ORIGINAL STATES' in line:
            values = _parse_rassi_diagonal(log, i + 1, 40)
            if values:
                rassi_diagonals.extend(values)

        elif in_pass1 and 'Molecular gradients' in line:
            g = _parse_gradients(log[i + 8: i + 8 + natom])
            # A short block is skipped rather than fatal: whether one is needed is
            # the packing step's question (it refuses a requested state with no
            # block), and a profile that prints its gradients in another layout
            # must not be broken by a guard meant for the state axis.
            if len(g) == natom:
                if 'after ESPF' in line:
                    gradient_post_espf.append(g)
                else:
                    gradient.append(g)
            else:
                incomplete_gradients += 1

        elif in_pass1 and 'Total SCF energy' in line:
            try:
                scf_energy = float(line.split()[-1])
            except (ValueError, IndexError):
                pass

        elif (in_pass2 if two_pass else in_pass1) and (
                'Total derivative coupling' in line
                or 'CI derivative coupling' in line):
            n = _parse_gradients(log[i + 8: i + 8 + natom])
            if 'Total derivative coupling' in line:
                nac_total.append(n)
            else:
                nac_ci.append(n)

        elif in_pass1 and 'Expectation values of various properties' in line:
            dip = _parse_dipole_block(log[i + 1: i + 80])
            if 'for root number:' in line:
                try:
                    root = int(line.split(':')[-1])
                except ValueError:
                    root = None
                if root is not None and dip is not None:
                    casscf_dipoles[root - 1] = dip
            elif dip is not None:
                caspt2_dipoles.append(dip)

        elif in_pass1 and 'Dipole transition strengths (spin-free states)' in line:
            # Only the dipole table is parsed; the identically
            # formatted velocity table below it is skipped.
            for entry_line in log[i + 1: i + 60]:
                if entry_line.startswith('++') or entry_line.strip() == '--':
                    break
                parts = entry_line.split()
                if len(parts) < 3 or parts[0].startswith('-'):
                    continue
                try:
                    frm = int(parts[0])
                    to = int(parts[1])
                    fval = float(parts[2])
                except ValueError:
                    continue
                osc_dict[(frm - 1, to - 1)] = fval
                osc_dict[(to - 1, frm - 1)] = fval

        elif 'MATRIX ELEMENTS OF 1-ELECTRON OPERATORS' in line:
            if i + 1 < len(log) and 'FOR THE SPIN-FREE EIGENSTATES' in log[i + 1]:
                mees_mats = _parse_spinfree_mltpl(log, i + 2)

        elif in_pass1 and 'Nr of states' in line:
            try:
                soc_state = int(line.split()[-1])
            except (ValueError, IndexError):
                pass

        elif in_pass1 and 'Root nr:' in line:
            try:
                tri_state = int(line.split()[-1])
                if soc_state > tri_state:
                    sin_state = soc_state - tri_state
            except (ValueError, IndexError):
                pass

        elif in_pass1 and 'Spin-orbit section' in line:
            if not soc_state:
                soc_state = sin_state + tri_state
            soc_dim = int(sin_state * mult[0] + tri_state * mult[-1]) if mult else soc_state
            soc_urt = int(soc_dim * (soc_dim + 1) / 2)
            soc_sfs = np.zeros([soc_dim, soc_dim])
            soc_mtx = np.zeros([soc_state, soc_state])

            for so_el in log[i + 11:i + 11 + soc_urt]:
                parts = so_el.split()
                # Columns: I1 S1 MS1 I2 S2 MS2 Real Imag Absolute
                if len(parts) < 9:
                    continue
                try:
                    i1 = int(parts[0]) - 1
                    i2 = int(parts[3]) - 1
                    va = float(parts[8])
                except ValueError:
                    continue
                if 0 <= i1 < soc_dim and 0 <= i2 < soc_dim:
                    soc_sfs[i1, i2] = va
                    soc_sfs[i2, i1] = va

            # Reduce SOC matrix into configuration state
            for s1 in range(sin_state):
                for s2 in range(tri_state):
                    p2 = sin_state + s2
                    first_col = int(sin_state + s2 * mult[-1]) if mult else 0
                    final_col = int(sin_state + (s2 + 1) * mult[-1]) if mult else 0
                    if first_col < soc_dim and final_col <= soc_dim:
                        soc_mtx[s1, p2] = np.sum(soc_sfs[s1, first_col:final_col] ** 2) ** 0.5
                        soc_mtx[p2, s1] = soc_mtx[s1, p2]

    # Extract SOC matrix elements for the requested pairs.  soc_mtx is
    # the configuration-state matrix (singlet states first, triplet
    # states after), so the 0-based global state indices of each
    # pair index it directly.  The triplet sublevel expansion is
    # already reduced into the configuration state (RMS over the
    # sublevel matrix elements).
    #
    # A pair the matrix does not carry is a missing quantity, not a zero
    # one: zero means "these two states do not couple", which is a physics
    # statement this log did not earn, and the ISC rate of an LZSH/SH run
    # reads it as its own input. Refused here, naming the pair.
    if len(soc_coupling) > 0 and len(soc_mtx) > 0:
        for pair in soc_coupling:
            if len(pair) != 2:
                continue
            s1, s2 = pair
            try:
                socme = float(soc_mtx[s1, s2])
            except (IndexError, TypeError):
                raise RuntimeError(
                    'Molcas: SOC pair %r was requested but the spin-orbit '
                    'matrix printed by this run has no such element (matrix '
                    'is %s). Name only pairs inside the states this run '
                    'diagonalises.'
                    % (list(pair), soc_mtx.shape))
            soc.append(socme)

    # Pack results
    energy_source = 'casscf'
    if method == 'caspt2':
        if len(caspt2_energies) < nstate and caspt2_final:
            missing = nstate - len(caspt2_energies)
            caspt2_energies.extend(caspt2_final[-missing:])
            warnings_out.append(
                'CASPT2 single-state block(s) supplied %d energy(ies) '
                'from FINAL CASPT2 RESULT.' % missing)
        if caspt2_energies:
            energy = np.array(caspt2_energies, dtype=float)
            energy_source = 'caspt2_%s' % str(
                getattr(config, 'caspt2_type', '') or 'ss').lower()
            if len(energy) < nstate:
                raise RuntimeError(
                    'Molcas: CASPT2 produced %d state energies but the '
                    'run declares %d; refusing to fall back to CASSCF '
                    'energies.' % (len(energy), nstate))
            if len(energy) > nstate:
                warnings_out.append(
                    'CASPT2 printed %d energies for %d states; using the '
                    'first %d.' % (len(energy), nstate, nstate))
                energy = energy[:nstate]
            if rassi_diagonals:
                diag = np.array(rassi_diagonals, dtype=float)
                if len(diag) >= len(energy):
                    delta = float(np.max(np.abs(diag[:len(energy)] - energy)))
                    if delta > 1.0e-5:
                        warnings_out.append(
                            'CASPT2 block energies differ from the RASSI '
                            'spin-free diagonal by %.2e; using the CASPT2 '
                            'block energies.' % delta)
        elif rassi_diagonals:
            energy = np.array(rassi_diagonals, dtype=float)[:nstate]
            energy_source = 'rassi_spin_free_diagonal'
            warnings_out.append(
                'CASPT2 energy block was not found; using the RASSI '
                'spin-free diagonal as the pre-SOC energy.')
            if len(energy) < nstate:
                raise RuntimeError(
                    'Molcas: RASSI diagonal has %d energies but the run '
                    'declares %d.' % (len(energy), nstate))
        elif casscf:
            raise RuntimeError(
                'Molcas: the log carries neither CASPT2 root energies '
                'nor a RASSI spin-free diagonal; refusing to fall back '
                'to CASSCF energies for a CASPT2 run.')
        elif scf_energy is not None:
            energy = np.array([scf_energy])
            energy_source = 'scf'
        else:
            raise RuntimeError(
                'Molcas: no CASPT2 energy could be parsed.')
    elif casscf:
        energy = np.array(casscf)
        if len(casscf) < nstate:
            # Fewer state energies than the run declares is a missing surface,
            # and a crossing search cannot tell it from a crossing that has
            # closed: the row it wanted is simply not there.
            raise RuntimeError(
                f'Molcas: the log carries {len(casscf)} state energy(ies) '
                f'while the run declares {nstate} states (ci_roots {ci}, '
                f'ci_spin {config.ci_spin}); a surface is missing and which one '
                f'it is cannot be read from the blocks that did arrive'
            )
    elif scf_energy is not None:
        energy = np.array([scf_energy])
        energy_source = 'scf'
    elif expect_energy:
        raise RuntimeError(
            'Molcas: the run asked for an energy but the completed log '
            'carries no CASPT2/RASSCF/SCF energy block.')
    else:
        energy = np.zeros(1)

    # ESPF prints a second gradient table; the post-ESPF one is the
    # one that belongs to the reported energy.
    gradient_blocks = gradient_post_espf if gradient_post_espf else gradient
    gradient = gradient_blocks
    if gradient_post_espf:
        warnings_out.append(
            'Using the post-ESPF molecular gradient; OpenMolcas omits '
            'grid-point derivatives in this table.')

    # The builder requests ALASKA roots in target_states order, and
    # `target_states` is sorted by the normalizer, which is also the order the
    # grouped builder walks the manifolds in (ascending 2S) - so the k-th block
    # is the k-th request entry. A requested state with no block is refused
    # rather than left as a zero row:
    # the shape contract is satisfied either way, but a zero force is a surface
    # nobody computed, and a search stepping on one converges on a flat place it
    # cannot tell from its own convergence.
    if gradient:
        if len(gradient) < len(target_states):
            raise RuntimeError(
                f'Molcas: {len(gradient)} gradient block(s) were read from the '
                f'final iteration while {len(target_states)} state(s) '
                f'{list(target_states)} asked for forces; the state(s) without '
                f'one would be packed as zero forces and read as a surface'
            )
        if len(gradient) > len(target_states):
            warnings.warn(
                f'Molcas: {len(gradient)} gradient blocks were read while '
                f'{len(target_states)} state(s) {list(target_states)} asked for '
                f'forces; the extra {len(gradient) - len(target_states)} block(s) '
                f'are not packed. Diagnostic only - the requested states are all '
                f'answered for.',
                stacklevel=2,
            )
        gradall = np.zeros((nstate, natom, 3))
        for k, s in enumerate(target_states):
            if isinstance(s, int) and 0 <= s < nstate:
                gradall[s] = np.array(gradient[k])
        gradient_arr = gradall
    elif expect_gradient:
        raise RuntimeError(
            'Molcas: the run asked for a gradient but the completed log '
            'carries no molecular-gradient block; a zero force would be a '
            'surface nobody computed.')
    else:
        gradient_arr = np.zeros((1, natom, 3))

    nac_blocks = nac_total if nac_total else nac_ci
    # Whether this run asked for NAC is a workflow fact, not a state fact:
    # a gradient-only job may carry a pair in its config (the input carrier
    # forwards what the run declared), and demanding a derivative-coupling
    # block for it would refuse a run that never asked for one.
    nac_runs = _workflow_runs_nac(workflow, config)
    requested_nac = ([p for p in (getattr(config, 'target_nac_pair', []) or [])
                      if len(p) == 2] if nac_runs else [])
    if requested_nac and not nac_blocks:
        raise RuntimeError(
            'Molcas: NAC pair(s) %r were requested but the log carries no '
            'derivative-coupling block.' % (requested_nac,))
    if requested_nac and nac_blocks:
        # `&ALASKA` prints one coupling table per requested pair and the block
        # itself names no states, so a row's identity comes from where the deck
        # put its pair: group by group (singlet first), each group in the
        # request's order. The rows are mapped back into the requested order
        # here, which is the order the framework addresses them by
        # (`planning/assembly.py::_task_property_request`), and anything that cannot be
        # mapped is refused rather than left in place of another pair.
        from ..builder.input_builder import nac_deck_order
        deck = nac_deck_order(requested_nac,
                              list(getattr(config, 'state_spin', []) or []))
        if (len(nac_blocks) != len(deck)
                or {tuple(pair) for pair in deck}
                != {tuple(pair) for pair in requested_nac}):
            raise RuntimeError(
                'Molcas: %d derivative-coupling block(s) were read for %d '
                'requested pair(s) %r (deck order %r); the rows are mapped '
                'back through the deck, so this log cannot be matched onto the '
                'request.'
                % (len(nac_blocks), len(requested_nac), requested_nac, deck))
        positions = {tuple(pair): index
                     for index, pair in enumerate(deck)}
        nac_blocks = [nac_blocks[positions[tuple(pair)]]
                      for pair in requested_nac]
    # A coupling the engine never printed is a missing quantity, not a zero
    # one. A row of zeros reads as "the two surfaces are uncoupled", which is
    # a physics statement the run did not earn: `meci` then reports a coupling
    # norm of 0.00000000 and drops that direction from the branching plane as
    # if it had been measured. Returning no rows says what happened, and the
    # framework decides whether its optimizer can do without it.
    nac_arr = (np.array(nac_blocks) if nac_blocks
               else np.empty((0, natom, 3)))
    nac_source = ('total_derivative'
                  if nac_total else
                  ('ci_derivative' if nac_ci else ''))
    # Same rule on the SOC channel: no element read is no rows, not a zero.
    soc_arr = np.array(soc) if soc else np.empty(0)

    # State dipole moments in Debye (nstate, 3). CASPT2-level dipoles
    # take precedence over the CASSCF ones when both are printed.
    dipole_arr = np.zeros((nstate, 3))
    if caspt2_dipoles:
        for k, dip in enumerate(caspt2_dipoles[:nstate]):
            dipole_arr[k] = dip
    else:
        for k, dip in casscf_dipoles.items():
            if 0 <= k < nstate:
                dipole_arr[k] = dip

    # Oscillator strengths and transition dipole moments resolved by
    # transition_pair. The MEES matrix indices are the 1-based RASSI
    # spin-free eigenstate numbers, matching the 0-based global state
    # indices of transition_pair directly. Missing entries (e.g. no
    # MEES keyword in older logs) stay zero.
    pairs = [pair for pair in config.transition_pair if len(pair) == 2]
    if pairs:
        osc_arr = np.array([osc_dict.get((a, b), 0.0) for a, b in pairs])
        tmdip_arr = np.zeros((len(pairs), 3))
        for k, (a, b) in enumerate(pairs):
            tmdip_arr[k] = [
                mees_mats.get(comp, {}).get((a + 1, b + 1), 0.0)
                for comp in (1, 2, 3)
            ]
        tmdip_arr *= AU2DEBYE
    else:
        osc_arr = np.empty(0)
        tmdip_arr = np.empty((0, 3))

    # Thermochemistry: fixed 298.15 K block from the MCKINLEY module.
    # Enthalpy/free energy are the thermal corrections to the
    # electronic energy, not the summed total energies.
    zpve, entropy, enthalpy, free_energy = _parse_thermochemistry(log)

    # Harmonic vibrational frequencies in cm^-1. Normal mode vectors
    # are not extracted; Molcas writes them to the freq.molden file.
    frequencies = _parse_frequencies(log)
    freq_arr = np.array(frequencies) if frequencies else np.zeros(1)

    return {
        'energy': energy,
        'gradient': gradient_arr,
        'nac': nac_arr,
        'soc': soc_arr,
        'dipole': dipole_arr,
        'transition_dipole': tmdip_arr,
        'osc_strength': osc_arr,
        'zero_point_energy': np.array([zpve]),
        'enthalpy': np.array([enthalpy]),
        'entropy': np.array([entropy]),
        'free_energy': np.array([free_energy]),
        'frequencies': freq_arr,
        'log_not_converged': completion == 'not_converged',
        'energy_source': energy_source,
        'nac_source': nac_source,
        'two_pass': two_pass,
        'nac_approximation': ('two_pass_no_esfp' if two_pass else ''),
        'nac_csf_included': (False if getattr(config, 'nocsf', True)
                             else True),
        'soc_reduction': ('frobenius_over_spin_components'
                          if len(soc) else ''),
        'esfp_gradient_grid_derivatives': False,
        'warnings': warnings_out,
    }


_CASPT2_HEADERS = (
    'Total XMS-CASPT2 energies',
    'Total MS-CASPT2 energies',
    'Total RMS-CASPT2 energies',
    'Total XDW-CASPT2 energies',
    'Total CASPT2 energies',
)

_ROOT_ENERGY_RE = re.compile(
    r'Root\s+\d+\s+Total energy:\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)')


def _is_caspt2_energy_header(line):
    return any(header in line for header in _CASPT2_HEADERS)


def _caspt2_variant(line):
    for key, name in (
            ('XMS-CASPT2', 'xms'),
            ('MS-CASPT2', 'ms'),
            ('RMS-CASPT2', 'rms'),
            ('XDW-CASPT2', 'xdw')):
        if key in line:
            return name
    return 'generic'


def _parse_caspt2_root_energies(lines, start, limit):
    """Read the Root/Total energy lines following a CASPT2 header."""
    values = []
    for line in lines[start:start + limit]:
        if 'Start Module' in line or 'Happy landing' in line:
            break
        # The generic 'Total CASPT2 energies' header is followed by an
        # explanatory paragraph and then a variant-specific header; stop
        # at that header so its root lines are not counted twice.
        if ('CASPT2 energies' in line
                and _is_caspt2_energy_header(line)):
            break
        match = _ROOT_ENERGY_RE.search(line)
        if match:
            text = match.group(1).replace('D', 'E').replace('d', 'e')
            try:
                values.append(float(text))
            except ValueError:
                continue
    return values


def _parse_rassi_diagonal(lines, start, limit):
    """Read the spin-free RASSI diagonal ('Diagonal, with energies')."""
    for offset, line in enumerate(lines[start:start + limit]):
        if 'Diagonal, with energies' not in line:
            continue
        values = []
        for row in lines[start + offset + 1:start + offset + 10]:
            parts = row.split()
            if not parts:
                break
            parsed = []
            for item in parts:
                text = item.replace('D', 'E').replace('d', 'e')
                try:
                    parsed.append(float(text))
                except ValueError:
                    parsed = []
                    break
            if not parsed:
                break
            values.extend(parsed)
        return values
    return []


def _parse_dipole_block(lines: list):
    """Extract the total dipole from a property expectation block.

    Locates the 1-st cartesian moments section and reads its a.u.
    Total line (the Total electronic / Total nuclear / debye lines
    have a different token count), converted to Debye.

    Parameters
    ----------
    lines : list of str
        Log lines following the 'Expectation values...' header.

    Returns
    -------
    list of float or None
        Dipole [x, y, z] in Debye, or None when the block carries no
        parseable 1-st cartesian moments.
    """
    in_moments = False
    for line in lines:
        stripped = line.strip()
        if '2-nd cartesian moments' in stripped:
            break
        if '1-st cartesian moments' in stripped:
            in_moments = True
            continue
        if in_moments and stripped.startswith('Total'):
            parts = stripped.split()
            if len(parts) == 4:
                try:
                    return [float(x) * AU2DEBYE for x in parts[1:4]]
                except ValueError:
                    continue
    return None


def _parse_spinfree_mltpl(log: list, start: int) -> dict:
    """Parse the MEES spin-free MLTPL 1 component matrices.

    Reads the three dipole-component matrices (PROPERTY: MLTPL 1,
    COMPONENT: 1/2/3) printed by the RASSI MEES keyword.

    Parameters
    ----------
    log : list of str
        Full log lines.
    start : int
        Index of the first line after the block header.

    Returns
    -------
    dict
        Mapping component (1/2/3) -> {(row_state, col_state): value}
        with 1-based state indices and values in a.u.
    """
    mats = {}
    i = start
    n = len(log)
    while i < n:
        line = log[i]
        if line.startswith('++'):
            break
        parts = line.split()
        if parts and parts[0] == 'PROPERTY:' and len(parts) >= 5:
            try:
                level = int(parts[2])
                comp = int(parts[4])
            except ValueError:
                i += 1
                continue
            if parts[1] == 'MLTPL' and level == 1 and comp in (1, 2, 3):
                entries, i = _parse_state_matrix(log, i + 1)
                mats[comp] = entries
                continue
            if len(mats) == 3:
                break
        i += 1
    return mats


def _parse_state_matrix(log: list, start: int):
    """Parse one square state-property matrix block.

    The STATE header row lists the column state numbers; each data
    row starts with its row state number. When more than four states
    are present, Molcas prints consecutive column blocks that share
    the same row states, so a new STATE header extends the columns.

    Parameters
    ----------
    log : list of str
        Full log lines.
    start : int
        Index of the first line after the PROPERTY header.

    Returns
    -------
    tuple
        ({(row, col): value}, index of the first unparsed line).
    """
    entries = {}
    columns = []
    i = start
    n = len(log)
    while i < n:
        parts = log[i].split()
        if not parts:
            i += 1
            continue
        if parts[0] == 'STATE':
            try:
                columns = [int(x) for x in parts[1:]]
            except ValueError:
                columns = []
            i += 1
            continue
        if parts[0] in ('ORIGIN:', 'PROPERTY:'):
            if columns and entries:
                return entries, i
            i += 1
            continue
        try:
            row = int(parts[0])
        except ValueError:
            if entries:
                return entries, i
            i += 1
            continue
        for col, val in zip(columns, parts[1:]):
            try:
                entries[(row, col)] = float(val)
            except ValueError:
                pass
        i += 1
    return entries, i


def _parse_thermochemistry(log: list):
    """Extract thermochemistry from the MCKINLEY output.

    Fixed to the 298.15 K temperature block. Enthalpy and free
    energy are taken from the 'Thermal contributions to' section,
    i.e. they are corrections to the electronic energy, not the
    summed total energies.

    Parameters
    ----------
    log : list of str
        Full log lines.

    Returns
    -------
    tuple
        (zpve, entropy, enthalpy, free_energy); zpve/enthalpy/
        free_energy in Hartree, entropy in Hartree/K. Fields stay
        zero when the log has no thermochemistry block.
    """
    zpve = 0.0
    entropy = 0.0
    enthalpy = 0.0
    free_energy = 0.0
    for i, line in enumerate(log):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'ZPVE' and len(parts) == 5:
            try:
                float(parts[1])
                zpve = float(parts[3])
            except ValueError:
                pass
        elif 'Temperature =' in line and 'kelvin' in line:
            try:
                temp = float(parts[2])
            except (ValueError, IndexError):
                continue
            if abs(temp - 298.15) > 1e-2:
                continue
            # Block ends at the closing ---- separator; trimming the
            # window there prevents leaking into the next temperature
            # block. Start after the separator line directly below
            # the Temperature header.
            block = []
            found_sep = False
            for block_line in log[i + 1: i + 60]:
                if set(block_line.strip()) == {'-'}:
                    if found_sep:
                        break
                    found_sep = True
                    continue
                block.append(block_line)
            entropy, enthalpy, free_energy = _parse_thermo_block(block)
    return zpve, entropy, enthalpy, free_energy


def _parse_thermo_block(lines: list):
    """Parse one MCKINLEY temperature block.

    Parameters
    ----------
    lines : list of str
        Log lines inside the selected temperature block.

    Returns
    -------
    tuple
        (entropy in Hartree/K, enthalpy in Hartree, free energy in
        Hartree), each the thermal correction value.
    """
    entropy = 0.0
    enthalpy = 0.0
    free_energy = 0.0
    after_entropy_header = False
    after_correction_header = False
    for line in lines:
        stripped = line.strip()
        parts = stripped.split()
        if not parts:
            continue
        if 'Molecular Partition Function and Molar Entropy' in stripped:
            after_entropy_header = True
            continue
        if after_entropy_header and parts[0] == 'TOTAL':
            try:
                entropy = float(parts[-1]) / CAL_PER_HARTREE
            except ValueError:
                pass
            after_entropy_header = False
            continue
        if stripped == 'Thermal contributions to':
            after_correction_header = True
            continue
        if after_correction_header:
            if 'Sum of energy' in stripped:
                after_correction_header = False
            elif parts[0] == 'ENTHALPY' and len(parts) == 5:
                try:
                    enthalpy = float(parts[3])
                except ValueError:
                    pass
            elif parts[0] == 'GIBBS' and len(parts) == 7:
                try:
                    free_energy = float(parts[5])
                except ValueError:
                    pass
    return entropy, enthalpy, free_energy


def _parse_frequencies(log: list) -> list:
    """Extract harmonic vibrational frequencies.

    Reads the 'Harmonic frequencies in cm-1' section printed after
    the numerical differentiation. Translational and rotational
    degrees of freedom are already removed by Molcas, so only the
    3N-6 (or 3N-5) genuine vibrations are listed. Imaginary
    frequencies are printed with a leading 'i' and are returned as
    negative values.

    Parameters
    ----------
    log : list of str
        Full log lines.

    Returns
    -------
    list of float
        Frequencies in cm^-1 (negative for imaginary modes); empty
        when the log carries no frequency section.
    """
    freqs = []
    for i, line in enumerate(log):
        if 'Harmonic frequencies in cm-1' not in line:
            continue
        for block_line in log[i + 1: i + 500]:
            stripped = block_line.strip()
            # Section ends before the rotational constants summary;
            # the guards also stop on banners and module markers.
            if ('Rotational Constants' in stripped
                    or 'Isotope shifted' in stripped
                    or stripped.startswith(('++', '***'))):
                break
            parts = stripped.split()
            if parts and parts[0] == 'Frequency:':
                for tok in parts[1:]:
                    try:
                        if tok.startswith('i'):
                            freqs.append(-float(tok[1:]))
                        else:
                            freqs.append(float(tok))
                    except ValueError:
                        pass
        break
    return freqs


def _parse_gradients(lines: list) -> list:
    """Parse gradient lines from Molcas log.

    Parameters
    ----------
    lines : list of str
        Gradient lines from log file.

    Returns
    -------
    list of list of float
        Parsed gradient values.
    """
    gradient = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            try:
                gx = float(parts[-3])
                gy = float(parts[-2])
                gz = float(parts[-1])
                gradient.append([gx, gy, gz])
            except ValueError:
                pass
    return gradient
