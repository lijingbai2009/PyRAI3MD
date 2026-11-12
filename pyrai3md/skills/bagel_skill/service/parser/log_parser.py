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

This module parses BAGEL log files and output data files to extract
computational results, aligned with real cluster output:

BAGEL outputs:
- Coordinates in the .log file (Angstrom; the molecule block sets
  "angstrom": "true")
- ENERGY.out: state energies
- FORCE_*.out: gradients per state (leading row-index column)
- NACME_*_*.out: derivative coupling vectors per pair (leading
  row-index column); BAGEL already divides by the energy gap, so the
  parser multiplies by (E_b - E_a) to match the CI-derivative
  convention used by the molcas skill
- Hessian matrix and vibrational frequencies are printed in the log
  ("++ Hessian ++" section and "Freq (cm-1)" lines); BAGEL does not
  write HESSIAN.out/FREQUENCY.out files
- Dipole/transition-dipole moments and oscillator strengths are
  printed in the log ("* CASSCF dipole moments" / "* CASPT2 dipole
  moments" sections); BAGEL does not export dipole files

"""

import os
import re
import warnings

import numpy as np


# Energy conversion factor (1 Hartree in eV)
HARTREE_TO_EV = 27.211386245988


def _extract_velocities(config, workdir):
    """Extract the final velocities (reserved fin_velocity channel).

    The bagel skill performs no skill-internal MD propagation that
    reports final velocities, so the fin_velocity return channel is
    reserved with an empty value; the uniform channel keeps every
    skill result contract aligned.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workdir : str
        Calculation directory.

    Returns
    -------
    list
        Empty list (no velocity output).
    """
    return []


def _check_log_completion(log_path: str) -> None:
    """Verify that a BAGEL log terminated normally.

    A crashed BAGEL run prints an 'ERROR: EXCEPTION RAISED: ...'
    line and stops before writing the output data files. Logs with
    such a marker - or missing logs - must never be parsed into
    zero-filled results, so the check raises before any result
    extraction happens (same policy as the molcas completion
    check).

    Parameters
    ----------
    log_path : str
        Path to the BAGEL .log file.

    Raises
    ------
    FileNotFoundError
        When the log file is missing (the calculation produced no
        output).
    RuntimeError
        When the log contains an ERROR marker.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f'BAGEL log file not found: {log_path}. The calculation '
            f'produced no output.'
        )

    with open(log_path, 'r') as f:
        lines = f.read().splitlines()

    error_lines = [line.strip() for line in lines
                   if 'ERROR' in line]
    if error_lines:
        raise RuntimeError(
            f'BAGEL calculation did not complete: {error_lines[-1]}'
        )


def parse_log(log_path: str, config) -> dict:
    """Parse BAGEL output files and extract all available results.

    Parameters
    ----------
    log_path : str
        Path to the BAGEL .log file.
    config : SkillConfig
        Runtime configuration container with state/coupling information.

    Returns
    -------
    dict
        Dictionary with keys: energy, gradient, nac, coordinates.

    Raises
    ------
    FileNotFoundError
        When the log file is missing (the calculation produced no
        output).
    RuntimeError
        When the log contains an ERROR marker.
    """
    nac_coupling = config.target_nac_pair
    nac_coupling = [p for p in nac_coupling if len(p) == 2]
    profile = config.profile

    # Use the directory containing the log file as the working directory
    workdir = os.path.dirname(os.path.abspath(log_path))

    # Reject failed or missing runs before extracting results so
    # zero-filled values never mask a calculation failure.
    _check_log_completion(log_path)

    with open(log_path, 'r') as f:
        log = f.read().splitlines()

    natom = config.natom or 1

    # BAGEL prints the background point charges as extra `Q` atoms appended
    # to the geometry, so every geometry block in the log is
    # natom + n_q rows long. Sizing the block by natom alone would close it
    # early and prepend the leftover Q rows to the next block, shifting every
    # block after the first; the returned rows keep only the real atoms,
    # which lead. The per-atom file readers (`FORCE_*.out`, `NACME_*.out`,
    # `opt.molden`, the Hessian's top-left block) already take the leading
    # natom rows or entries for the same reason.
    # The emptiness test cannot ask for the truth value of the rows:
    # ONIOM electrostatic embedding hands over an (n, 4) array, and
    # `charges or []` on an array raises instead of counting it.
    charges = getattr(config, 'point_charges', None)
    n_charges = 0 if charges is None else len(charges)
    block_atoms = natom + n_charges

    # Parse coordinates from the log (Angstrom, see module notes)
    coord = [block[:natom]
             for block in _parse_coordinates_from_log(log, block_atoms)]

    # Parse energies from the log (primary source); ENERGY.out is a
    # shared file that can be overwritten by subsequent tasks and may
    # not exist for non-force profiles, so it only serves as a fallback.
    energy = _parse_energy(config, workdir, log, config.method)

    # Parse gradients from FORCE_*.out
    gradient = _parse_gradients(config, workdir)

    # Parse NAC from NACME_*_*.out and rescale to the CI-derivative
    # convention shared with the molcas skill
    nac = _parse_nac(config, nac_coupling, workdir, energy, log)

    # Parse Hessian and frequencies from the log (BAGEL does not
    # write HESSIAN.out/FREQUENCY.out)
    hessian = _parse_hessian_from_log(log, natom)
    frequencies = _parse_frequencies_from_log(log)

    results = {
        'energy': energy,
        'gradient': gradient,
        'nac': nac,
        'coordinates': coord,
        'hessian': hessian,
        'frequencies': frequencies,
        # Reserved fin_velocity return channel (no velocity output)
        'fin_velocity': _extract_velocities(config, workdir),
    }

    # Geometry optimization iterates inside BAGEL: report the final
    # structure through the framework fin_coord channel and the
    # convergence status through opt_converge (same contract as the
    # molcas skill). BAGEL does not print a dedicated unconverged
    # marker, so a run that passes the completion check counts as
    # converged (pending cluster validation).
    if 'geometry_optimization' in profile:
        # Converged runs print the final structure in a '*** Geometry
        # ***' block at the end of the log. A log that carries no
        # complete geometry block falls back to opt.molden, where
        # BAGEL keeps the optimization geometry (runs interrupted by
        # an ERROR are rejected above, so their opt.molden content is
        # never parsed here).
        fin_coord = _parse_last_coordinates_from_log(log, block_atoms)
        fin_coord = list(fin_coord or [])[:natom]
        if not fin_coord:
            fin_coord = _parse_coordinates_from_molden(
                os.path.join(workdir, 'opt.molden'), natom)
        if fin_coord:
            results['fin_coord'] = np.array(fin_coord)
        results['opt_converge'] = True

    # Dipole profiles print state dipoles, transition dipoles and
    # oscillator strengths in the log dipole-moments section.
    if 'dipole' in profile or 'transition_dipole' in profile:
        dipoles, trdipoles, osc_map = _parse_dipole_sections(log)
        results['dipole'] = _select_state_dipoles(
            config, dipoles)
        transition_dipole = _select_transition_dipoles(
            config, trdipoles)
        results['transition_dipole'] = transition_dipole
        results['excitation'] = _excitation_energies(energy)
        results['osc_strength'] = _select_osc_strengths(
            config, osc_map)

    return results


def _read_atom_lines(lines: list, natom: int) -> list:
    """Collect complete coordinate blocks from log lines.

    BAGEL prints the geometry as JSON-like lines:
      { "atom" : "C", "xyz" : [ x, y, z ] },
    Coordinates are in Angstrom because the molecule block sets
    "angstrom": "true".

    Parameters
    ----------
    lines : list of str
        Log lines.
    natom : int
        Number of atoms.

    Returns
    -------
    list
        Complete coordinate blocks (each a list of natom [x, y, z]).
    """
    blocks = []
    current = []
    for line in lines:
        if '"atom"' in line:
            parts = line.replace(',', ' ').replace('"', ' ').split()
            if len(parts) >= 10:
                try:
                    current.append([float(parts[7]), float(parts[8]),
                                    float(parts[9])])
                    if len(current) == natom:
                        blocks.append(current)
                        current = []
                except (ValueError, IndexError):
                    pass

    return blocks


def _parse_coordinates_from_log(log: list, natom: int) -> list:
    """Parse the first coordinate block from BAGEL log lines.

    Parameters
    ----------
    log : list of str
        Log lines.
    natom : int
        Number of atoms.

    Returns
    -------
    list
        Parsed coordinates in Angstrom.
    """
    blocks = _read_atom_lines(log, natom)
    return blocks[0] if blocks else []


def _parse_last_coordinates_from_log(log: list, natom: int) -> list:
    """Parse the last coordinate block from BAGEL log lines.

    A geometry optimization prints the geometry at every step, so
    the log carries several '"atom"' blocks; the final structure is
    the last complete block.

    Parameters
    ----------
    log : list of str
        Log lines.
    natom : int
        Number of atoms.

    Returns
    -------
    list
        Final coordinates in Angstrom, or an empty list when no
        complete block is found.
    """
    blocks = _read_atom_lines(log, natom)
    return blocks[-1] if blocks else []


def _parse_coordinates_from_molden(path: str, natom: int) -> list:
    """Parse coordinates from the opt.molden fallback file.

    BAGEL writes the optimization geometry to opt.molden in Molden
    format; the '[Atoms] Angs' section already carries Angstrom
    coordinates:
      [Atoms] Angs
     c       1                    6   x   y   z
    Only the first atom block is read (the final structure).

    Parameters
    ----------
    path : str
        Path to opt.molden.
    natom : int
        Number of atoms expected.

    Returns
    -------
    list
        Coordinates in Angstrom (natom x 3), or an empty list when
        the file is missing or malformed.
    """
    if not os.path.exists(path):
        return []

    coords = []
    in_atoms = False
    try:
        with open(path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('[Atoms]'):
                    in_atoms = True
                    continue
                if not in_atoms:
                    continue
                if stripped.startswith('['):
                    # Next Molden section ([GTO]) ends the atom block
                    break
                parts = stripped.split()
                if len(parts) >= 6:
                    try:
                        coords.append([float(parts[3]),
                                       float(parts[4]),
                                       float(parts[5])])
                    except ValueError:
                        continue
    except OSError:
        return []

    return coords[:natom] if len(coords) >= natom else []


def _parse_energy(config, workdir: str, log=None, method=None) -> np.ndarray:
    """Parse energies from the BAGEL log or ENERGY.out file.

    BAGEL prints state energies in the log depending on the method:
    - CASSCF: ``=== FCI iteration ===`` section (iter root * energy)
    - CASPT2: ``* CASPT2 energy : state N`` lines
    - XMS-CASPT2: ``* MS-CASPT2 energy : state N`` lines

    ENERGY.out is a shared file that can be overwritten by subsequent
    tasks and may not exist for non-force profiles. It only serves as
    a fallback when log parsing fails.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workdir : str
        Working directory containing output files.
    log : list of str or None
        Log lines (primary source).
    method : str or None
        Calculation method (casscf, caspt2, xmscaspt2).

    Returns
    -------
    np.ndarray
        Array of state energies.
    """
    nstate = config.nstate

    if log is not None:
        energies = _parse_energy_from_log(log, method, nstate)
        if len(energies) == nstate and not np.all(np.array(energies) == 0):
            return np.array(energies)

    # Fallback: ENERGY.out. This file is not this run's document: BAGEL writes it
    # under a fixed name into the working directory, and a run whose own log
    # printed no energy used to land on whatever a sibling job had left there.
    # The measured case is the finite-difference Hessian job before its chain was
    # given a solve (`xmscaspt2_frequency_test.log` of round tmp_20261014_185533
    # has only `Hessian evaluation ( k / 18)` rows): in round tmp_20261015_100320
    # the file's mtime (11:15:50) was five minutes after that job ran (11:10), and
    # its three rows are the energies seven *sibling* jobs print for the same
    # input - a shared geometry, which is why the number looked right and the case
    # passed. Two things have since made the fallback unnecessary rather than
    # merely noisy: the calculation directory is named per project (#354), and the
    # run's own log now answers for it - the frequency chain solves its reference
    # state (#355), and clearing the fixed-name files before the launch line means
    # a borrowed number is a failure instead of a pass. What sits in the file still
    # belongs to whichever job last wrote there, so it is read with a warning, not
    # silently.
    energy_file = os.path.join(workdir, 'ENERGY.out')
    if os.path.exists(energy_file):
        try:
            energy = np.loadtxt(energy_file).reshape(-1)
        except (ValueError, OSError):
            energy = np.array([])
        if len(energy) >= nstate:
            warnings.warn(
                f'BAGEL: the {nstate} state energy/energies this run declares were '
                f'taken from {energy_file}, not from the log, because the log of '
                f'method {method!r} printed none. That file carries a fixed name, '
                f'so its numbers are whichever job last wrote it, not necessarily '
                f'this one; treat them as this run\'s only if the job ran alone in '
                f'{workdir}',
                stacklevel=2,
            )
            return energy[:nstate]

    # Neither source answered. Returning zeros here satisfied the (nstate,)
    # shape while describing no surface at all, and a caller that read a state
    # pair out of it saw two flat surfaces - which is how an unconverged or
    # differently-printed run went through as a converged one.
    raise RuntimeError(
        f'BAGEL: no state energies could be read for the {nstate} state(s) this '
        f'run declares (method {method!r}); neither the log section for that '
        f'method nor {os.path.join(workdir, "ENERGY.out")} carried that many '
        f'values, and a zero-filled array would satisfy the shape contract '
        f'without describing any surface'
    )


def _parse_energy_from_log(log: list, method: str, nstate: int) -> list:
    """Parse state energies from BAGEL log lines.

    Parameters
    ----------
    log : list of str
        Log lines.
    method : str or None
        Calculation method (casscf, caspt2, xmscaspt2).
    nstate : int
        Number of states to extract.

    Returns
    -------
    list
        State energies, truncated to nstate; empty list when parsing
        fails.
    """
    if method in ('xmscaspt2',):
        energies = _parse_xms_caspt2_energies(log)
        if not energies and nstate == 1:
            # A one-state run has no multistate Hamiltonian to diagonalise, and
            # SMITH says so by running without printing any MS row: measured on
            # the frequency job of round tmp_20261015_211219, whose own log
            # carries `* METHOD: SMITH 0.68` and no `* MS-CASPT2 energy` line,
            # while the CASPT2 solve that precedes it printed
            # `* CASPT2 energy : state  0      -77.0713120427`. The eigenvalue of
            # a 1x1 matrix is its diagonal, so that line is this run's surface.
            # Accepted only when the log carries exactly one of them: more than
            # one means the CI solved more roots than this run declares, and
            # taking the first of those would hand back a root nobody asked for.
            single = _parse_caspt2_energies(log)
            energies = single if len(single) == 1 else []
    elif method in ('caspt2',):
        energies = _parse_caspt2_energies(log)
    else:
        # CASSCF (default): FCI iteration section
        energies = _parse_fci_energies(log)

    return energies[:nstate] if len(energies) >= nstate else []


def _parse_fci_energies(log: list) -> list:
    """Parse CASSCF state energies from the FCI iteration section.

    Row shape, from the test round tmp_20261014_185533 (both shapes occur
    there)::

        === FCI iteration ===

            0  0 *     -77.92051100     2.78e-16      0.00
            0  1 *     -77.80674018     5.55e-17      0.00
            0  2 *     -77.80062527     5.55e-17      0.00

            0  0       -77.74685469     4.96e-10      0.00
            1  0       -77.94047011     1.78e-02      0.00
          - linear dependency detected:    1 /    3    min eigenvalue: ...
          * Using canonical orthogonalization due to linear dependency

            2  0       -77.94689388     3.14e-08      0.00
            3  0 *     -77.94689388     7.85e-17      0.00

    The second column is the state index, not the first. Measured on every log in
    that round's BAGEL directory: the block that prints `0 0 / 0 1 / 0 2` is
    followed by `* ci vector, state 0`, `state 1`, `state 2`, and the block that
    prints `0 0 / 1 0 / 2 0 / 3 0` is followed by one CI vector, `state 0`. So the
    first column counts the CI solves of the block (a single root reprinted once
    per solve) while the second is the root, and a block may hold several rows for
    one state. Reading the first column as the state did both of the damage this
    round showed: it refused the three-state cases outright ("root indices
    [0, 0, 0]"), and for a one-state case it left 0..3 in order, so the guard
    passed and `energies[:nstate]` took row 0 - the first, unconverged solve at
    -77.74685469 rather than the converged -77.94689388, 0.20 Hartree away, with
    the case reporting PASS.

    Within a state the last row is the converged one: in every block of every log
    above the `*` (the engine's marker for the selected root) sits on the last row
    of its state, never earlier. Rows are recognized by shape and everything else
    is skipped, because BAGEL's own diagnostics interrupt the block and several of
    them start with `*`; the block ends at the next `=== ... ===` header or at EOF.

    A block whose state numbers are not `0, 1, 2, ...` is refused rather than
    packed by row position: rows in another order would land on the wrong
    framework state while the row count still agreed, which is the shape of a wrong
    answer no downstream check can see.

    For geometry optimizations the log contains multiple FCI blocks; only the last
    complete block is returned.

    Parameters
    ----------
    log : list of str
        Log lines.

    Returns
    -------
    list
        State energies from the last FCI iteration block.
    """
    blocks = []
    current = []
    in_fci = False

    for line in log:
        if '=== FCI iteration ===' in line:
            in_fci = True
            if current:
                blocks.append(current)
            current = []
            continue
        if not in_fci:
            continue
        # Any other section header ends the block. Nothing else does: BAGEL's own
        # diagnostics inside the block start with '*' and with '-'.
        section = re.match(r'^\s*===\s*(\S.*?)\s*===\s*$', line)
        if section is not None:
            if current:
                blocks.append(current)
            in_fci = False
            current = []
            continue
        # [ci-solve counter] state [selection star] energy residual overlap
        match = re.match(
            r'^\s*\d+\s+(\d+)\s+\*?\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)', line)
        if match:
            current.append((int(match.group(1)), float(match.group(2))))

    if current:
        blocks.append(current)

    if not blocks:
        return []

    # One row per state, keeping the last print of each.
    by_state = {}
    for state, energy in blocks[-1]:
        by_state[state] = energy
    states = list(by_state)
    if states != list(range(len(states))):
        raise RuntimeError(
            f'BAGEL: the last FCI iteration block prints state numbers {states} '
            f'rather than 0..{len(states) - 1} in order, so which state each '
            f'energy belongs to is not readable from the block and the energies '
            f'cannot be packed by row position'
        )
    return [by_state[state] for state in states]


def _parse_caspt2_energies(log: list) -> list:
    """Parse CASPT2 state energies from ``* CASPT2 energy`` lines.

    Parameters
    ----------
    log : list of str
        Log lines.

    Returns
    -------
    list
        CASPT2 state energies in state order.
    """
    energies = []
    for line in log:
        if '* CASPT2 energy :' in line and 'state' in line:
            try:
                energies.append(float(line.split()[-1]))
            except (ValueError, IndexError):
                pass
    return energies


def _parse_xms_caspt2_energies(log: list) -> list:
    """Parse XMS-CASPT2 state energies from ``* MS-CASPT2 energy`` lines.

    Parameters
    ----------
    log : list of str
        Log lines.

    Returns
    -------
    list
        XMS-CASPT2 state energies in state order.
    """
    energies = []
    for line in log:
        if '* MS-CASPT2 energy :' in line and 'state' in line:
            try:
                energies.append(float(line.split()[-1]))
            except (ValueError, IndexError):
                pass
    return energies


def _read_indexed_block(path: str, natom: int) -> list:
    """Read a per-atom vector block with a leading row-index column.

    FORCE_*.out and NACME_*_*.out share the format: a header line
    with the column indices followed by natom rows whose first token
    is the row index:
                             0                   1                   2
         0        -0.0000128108       -0.0000109557   ...
    The parser keeps columns 1:4 (x, y, z), matching the legacy
    string2float behaviour.

    Parameters
    ----------
    path : str
        Path to the output file.
    natom : int
        Number of atoms (data rows to read).

    Returns
    -------
    list
        (natom, 3) vectors, or zeros when the file is missing or
        malformed.
    """
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                lines = f.read().splitlines()
            rows = []
            for line in lines[1:natom + 1]:
                parts = line.split()
                if len(parts) >= 4:
                    rows.append([float(parts[1]), float(parts[2]),
                                 float(parts[3])])
            if len(rows) == natom:
                return rows
        except (ValueError, OSError):
            pass

    return [[0.0, 0.0, 0.0]] * natom


def _parse_gradients(config, workdir: str) -> np.ndarray:
    """Parse gradients from FORCE_*.out files.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    workdir : str
        Working directory containing output files.

    Returns
    -------
    np.ndarray
        Gradient array of shape (nstate, natom, 3).
    """
    nstate = config.nstate
    natom = config.natom or 1

    gradient = []
    for i in range(nstate):
        force_file = os.path.join(workdir, f'FORCE_{i}.out')
        gradient.append(_read_indexed_block(force_file, natom))

    return np.array(gradient)


def _parse_energy_gaps_from_log(log: list) -> dict:
    """Extract NACME energy gaps printed in the log.

    Each NACME evaluation prints:
      * NACME Target states: a - b
      * Energy gap is:       g eV
    where g = (E_a - E_b) in eV. Stored as (E_b - E_a) in Hartree
    per unordered pair so the nac rescaling can fall back to it when
    ENERGY.out is unavailable.

    Parameters
    ----------
    log : list of str
        Log lines.

    Returns
    -------
    dict
        frozenset(a, b) -> (E_b - E_a) in Hartree.
    """
    gaps = {}
    pair = None
    for line in log:
        match = re.search(r'NACME Target states:\s*(\d+)\s*-\s*(\d+)',
                          line)
        if match:
            pair = frozenset((int(match.group(1)), int(match.group(2))))
            continue
        match = re.search(r'Energy gap is:\s*(-?[\d.]+)', line)
        if match and pair is not None:
            gaps[pair] = -float(match.group(1)) / HARTREE_TO_EV
            pair = None

    return gaps


def _parse_nac(config, nac_coupling: list, workdir: str,
               energy: np.ndarray, log: list) -> np.ndarray:
    """Parse NAC from NACME_*_*.out files.

    BAGEL exports the derivative coupling vector d_ij = H'_ij /
    (E_j - E_i), i.e. already divided by the energy gap, while the
    molcas skill reports the raw CI derivative coupling H'_ij. The
    vectors are multiplied by (E_b - E_a) so every skill returns the
    same CI-derivative convention. The gap printed in the log belongs
    to the same NACME evaluation and takes precedence; the ENERGY.out
    states only serve as a fallback.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    nac_coupling : list of list of int
        NAC coupling pairs.
    workdir : str
        Working directory containing output files.
    energy : np.ndarray
        State energies in Hartree (used for the gap factor).
    log : list of str
        Log lines (fallback source for the energy gaps).

    Returns
    -------
    np.ndarray
        NAC array of shape (nnac, natom, 3).
    """
    natom = config.natom or 1
    gaps = _parse_energy_gaps_from_log(log)

    nac = []
    for pair in nac_coupling:
        pa, pb = pair
        nac_file = os.path.join(workdir, f'NACME_{pa}_{pb}.out')
        vector = np.array(_read_indexed_block(nac_file, natom))

        gap = gaps.get(frozenset((pa, pb)), 0.0)
        if gap == 0.0:
            gap = float(np.asarray(energy, dtype=float)[pb]
                        - np.asarray(energy, dtype=float)[pa])
        vector = vector * gap
        nac.append(vector.tolist())

    if nac:
        return np.array(nac)
    else:
        return np.zeros((1, natom, 3))


def _parse_hessian_from_log(log: list, natom: int) -> np.ndarray:
    """Parse the Hessian matrix from the log.

    BAGEL prints the mass-unweighted Hessian in the '++ Hessian ++'
    section as 6-column blocks; each data row starts with the row
    index. The column-header lines (six increasing integers) delimit
    the blocks.

    Parameters
    ----------
    log : list of str
        Log lines.
    natom : int
        Number of atoms.

    Returns
    -------
    np.ndarray
        Hessian matrix of shape (3*natom, 3*natom).
    """
    ndim = 3 * natom

    in_section = False
    col_offset = None
    hessian = np.zeros((ndim, ndim))
    filled = 0
    for line in log:
        if '++ Hessian ++' in line:
            in_section = True
            col_offset = None
            continue
        if not in_section:
            continue
        if line.strip().startswith('++'):
            break

        parts = line.split()
        if len(parts) == 6:
            try:
                indices = [int(p) for p in parts]
            except ValueError:
                continue
            if indices == list(range(indices[0], indices[0] + 6)):
                # Column header line: sets the block column offset
                col_offset = indices[0]
                continue
        if len(parts) == 7 and col_offset is not None:
            try:
                row = int(parts[0])
                values = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            for k, value in enumerate(values):
                col = col_offset + k
                if row < ndim and col < ndim:
                    hessian[row, col] = value
                    filled += 1

    if filled >= ndim * ndim:
        return hessian

    return np.zeros((ndim, ndim))


def _parse_frequencies_from_log(log: list) -> np.ndarray:
    """Parse vibrational frequencies from the log.

    The 'Vibrational frequencies, IR intensities' section prints
    the modes in blocks of six:
      Freq (cm-1)   -718.14   0.00   ...
    All blocks are concatenated into a single array.

    Parameters
    ----------
    log : list of str
        Log lines.

    Returns
    -------
    np.ndarray
        Vibrational frequencies in cm^-1 (empty when absent).
    """
    frequencies = []
    for line in log:
        if 'Freq (cm-1)' in line:
            values = line.split('Freq (cm-1)')[-1].split()
            for value in values:
                try:
                    frequencies.append(float(value))
                except ValueError:
                    pass

    return np.array(frequencies)


def _parse_dipole_sections(log: list) -> tuple:
    """Parse dipole-moments sections from the log.

    BAGEL prints one section per method level:
      * CASSCF dipole moments
        * State             0 :   ( x, y, z) a.u.
        ...
        * Transition    1 - 0 :   ( x, y, z) a.u.
        * Oscillator strength :   f a.u.
    The CASPT2-level section takes precedence when both appear.

    Parameters
    ----------
    log : list of str
        Log lines.

    Returns
    -------
    tuple
        (state_dipoles, transition_dipoles, osc_map) for the highest
        level found: state dipoles as {state: [x, y, z]},
        transition dipoles as {frozenset(a, b): [x, y, z]} and
        oscillator strengths as {frozenset(a, b): f}.
    """
    sections = {}
    current = None
    last_transition = None
    for line in log:
        if 'dipole moments' in line and '*' in line:
            level = 'caspt2' if 'CASPT2' in line else 'casscf'
            current = level
            sections.setdefault(
                level, ({}, {}, {}))
            last_transition = None
            continue

        if current is None:
            continue

        state_dipoles, trdipoles, osc_map = sections[current]

        match = re.search(
            r'\*\s*State\s+(\d+)\s*:\s*\(([^)]*)\)', line)
        if match:
            state = int(match.group(1))
            state_dipoles[state] = _split_vector(match.group(2))
            continue

        match = re.search(
            r'\*\s*Transition\s+(\d+)\s*-\s*(\d+)\s*:\s*\(([^)]*)\)',
            line)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            key = frozenset((a, b))
            trdipoles[key] = _split_vector(match.group(3))
            last_transition = key
            continue

        if 'Oscillator strength :' in line and last_transition is not None:
            parts = line.split('Oscillator strength :')[-1].split()
            if parts:
                try:
                    osc_map[last_transition] = float(parts[0])
                except ValueError:
                    pass
            last_transition = None

    if 'caspt2' in sections:
        return sections['caspt2']
    if 'casscf' in sections:
        return sections['casscf']
    return {}, {}, {}


def _split_vector(text: str) -> list:
    """Split a '(x, y, z)' payload into three floats.

    Parameters
    ----------
    text : str
        Comma-separated float payload.

    Returns
    -------
    list
        [x, y, z] values (zeros when malformed).
    """
    parts = text.replace(',', ' ').split()
    if len(parts) >= 3:
        try:
            return [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            pass
    return [0.0, 0.0, 0.0]


def _select_state_dipoles(config, state_dipoles: dict) -> np.ndarray:
    """Assemble the state dipole array in state order.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    state_dipoles : dict
        {state: [x, y, z]} parsed from the log.

    Returns
    -------
    np.ndarray
        Dipole array of shape (nstate, 3) in atomic units.
    """
    nstate = config.nstate

    dipoles = []
    for i in range(nstate):
        dipoles.append(state_dipoles.get(i, [0.0, 0.0, 0.0]))

    return np.array(dipoles)


def _select_transition_dipoles(config, trdipoles: dict) -> np.ndarray:
    """Assemble transition dipoles in transition_pair order.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    trdipoles : dict
        {frozenset(a, b): [x, y, z]} parsed from the log.

    Returns
    -------
    np.ndarray
        Transition dipole array of shape (npair, 3) in atomic units.
    """
    pairs = [pair for pair in config.transition_pair
             if len(pair) == 2]

    if not pairs:
        return np.empty((0, 3))

    transition_dipole = []
    for a, b in pairs:
        value = trdipoles.get(frozenset((a, b)))
        if value is None:
            # The pair was declared and the log printed no transition for it.
            # A zero row here says "no transition dipole", which is a physics
            # statement about a pair this run never reported (#448).
            raise RuntimeError(
                'BAGEL: transition pair (%d, %d) is declared but the dipole '
                'section prints no transition for it.' % (a, b))
        transition_dipole.append(value)
    return np.array(transition_dipole)


def _select_osc_strengths(config, osc_map: dict) -> np.ndarray:
    """Assemble oscillator strengths in transition_pair order.

    BAGEL prints oscillator strengths for every transition in the
    dipole section, so no derivation is needed.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    osc_map : dict
        {frozenset(a, b): f} parsed from the log.

    Returns
    -------
    np.ndarray
        Oscillator strengths of shape (npair,).
    """
    pairs = [pair for pair in config.transition_pair
             if len(pair) == 2]

    osc_strength = []
    for a, b in pairs:
        osc_strength.append(osc_map.get(frozenset((a, b)), 0.0))

    if osc_strength:
        return np.array(osc_strength)
    # Same rule as the transition dipoles: no declared pair, no row (#448).
    return np.empty(0)


def _excitation_energies(energy: np.ndarray) -> np.ndarray:
    """Derive excitation energies relative to the ground state.

    Parameters
    ----------
    energy : np.ndarray
        State energies in Hartree.

    Returns
    -------
    np.ndarray
        Excitation energies of shape (nstate,) with the ground
        state at zero.
    """
    energy = np.asarray(energy, dtype=float)
    return energy - energy[0]
