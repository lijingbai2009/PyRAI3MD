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

This module parses the CP2K log file and the auxiliary output files
(force xyz, tdfrc, trajectory xyz). Missing sections are returned as
empty/zero structures; the caller (output_parser) applies
zero-filling according to the profile.

"""

import os
import re
import numpy as np


def parse_log(log_path: str, config) -> dict:
    """Parse a CP2K log file and extract the available results.

    Parameters
    ----------
    log_path : str
        Path to the CP2K .log file.
    config : SkillConfig
        Runtime configuration container with state information.

    Returns
    -------
    dict
        Raw parsed data with keys: s0_energy, scf_converged,
        tddft_states, energies, opt_converged.
    """
    data = {
        's0_energy': None,       # last 'Total energy:' value in Eh
        'scf_converged': False,
        'tddft_states': [],      # excitation table rows (see below)
        'energies': [],          # all 'Total energy:' values in Eh
        'opt_converged': False,
    }
    # A log without a manifold banner (a run that asked for no triplet block)
    # carries singlet rows only, which is what the reader must assume.
    manifold = 1

    if not os.path.exists(log_path):
        # The log is the one input whose absence used to be absorbed: an empty
        # `data` came back, and the refusal the caller then hit named a state row
        # that "nobody answered" three layers away from the job that never wrote a
        # file. A missing log is not an unreadable one, and with two runs per
        # evaluation it must say which run is missing.
        raise FileNotFoundError(
            f'CP2K output file not found: {log_path}. The run was asked to write '
            f'this file and it is not there, so nothing about its states is '
            f'known; check the calculation directory and the run script beside it.'
        )

    with open(log_path, 'r') as f:
        log = f.read().splitlines()

    for line in log:
        # The excitation tables are preceded by a banner naming the spin
        # manifold they belong to (' R-TDDFPT states of multiplicity 1', and
        # '... multiplicity 3' when RKS_TRIPLETS is on). Each manifold prints
        # its own table with the state numbers restarting at 1, so the row
        # lines carry no spin column and two rows can share one number while
        # describing different surfaces. The banner is the only place the spin
        # is written, so it is recorded here and carried onto every row.
        banner = _parse_manifold_banner(line)
        if banner is not None:
            manifold = banner
            continue

        # Ground-state / active-state total energy; the last
        # occurrence is the final value of the run.
        if 'Total energy:' in line:
            try:
                energy = float(line.split()[-1])
                data['energies'].append(energy)
                data['s0_energy'] = energy
            except (ValueError, IndexError):
                pass

        elif 'SCF run converged' in line:
            data['scf_converged'] = True

        elif 'GEOMETRY OPTIMIZATION COMPLETED' in line:
            data['opt_converged'] = True

        # TDDFPT excitation table rows:
        # 'TDDFPT|   1   5.46054  x  y  z  osc'
        elif line.lstrip().startswith('TDDFPT|'):
            row = _parse_tddft_row(line)
            if row is not None:
                row['multiplicity'] = manifold
                data['tddft_states'].append(row)

    return data


MANIFOLD_BANNER = re.compile(
    r'\bTDDFPT states of multiplicity\s+(\d+)\s*$')


def _parse_manifold_banner(line: str):
    """The spin manifold a following excitation table belongs to, or None.

    CP2K writes one banner per manifold, `R-TDDFPT states of multiplicity 1`
    for singlets and `... multiplicity 3` for triplets, immediately above that
    manifold's table. The number is CP2K's own multiplicity (2S+1) and is kept
    as printed here; the conversion to the framework's 2S spelling of
    `state_spin` belongs to `state_map`, which is the one place the two
    conventions are allowed to meet.

    Returns
    -------
    int or None
        2S+1 of the manifold (1 for a singlet table, 3 for a triplet table),
        or None when the line is not a banner.
    """
    match = MANIFOLD_BANNER.search(line)
    if match is None:
        return None
    return int(match.group(1))


def _parse_tddft_row(line: str):
    """Parse one TDDFPT excitation table row.

    Row layout: state number, excitation energy (eV), transition
    dipole components x/y/z (a.u.), oscillator strength (a.u.).

    Parameters
    ----------
    line : str
        Log line starting with 'TDDFPT|'.

    Returns
    -------
    dict or None
        Row dict with keys state, excitation, transition_dipole,
        osc_strength; None when the line is not a state row.
    """
    parts = line.split('|', 1)[-1].split()
    if len(parts) < 6:
        return None
    try:
        return {
            'state': int(parts[0]),
            'excitation': float(parts[1]),
            'transition_dipole': [float(parts[2]), float(parts[3]),
                                  float(parts[4])],
            'osc_strength': float(parts[5]),
        }
    except (ValueError, IndexError):
        return None


def parse_force_xyz(force_path: str, natom: int) -> np.ndarray:
    """Parse the CP2K force file ({project}-force-1_0.xyz).

    File layout (demo verified):
        ATOMIC FORCES in [a.u.]
        # Atom   Kind   Element   X   Y   Z
        1   1   P   fx  fy  fz

    Parameters
    ----------
    force_path : str
        Path to the force file.
    natom : int
        Expected number of atoms.

    Returns
    -------
    np.ndarray
        Forces (natom, 3) in hartree/bohr; empty (0, 3) when the
        file is missing or malformed.
    """
    if not os.path.exists(force_path):
        return np.zeros((0, 3))

    with open(force_path, 'r') as f:
        lines = f.read().splitlines()

    forces = []
    for line in lines:
        parts = line.split()
        # Data rows start with the 1-based atom index and carry
        # index, kind, element, fx, fy, fz.
        if len(parts) != 6:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        try:
            forces.append([float(parts[3]), float(parts[4]),
                           float(parts[5])])
        except ValueError:
            continue
        if index >= natom:
            break

    if len(forces) != natom:
        return np.zeros((0, 3))
    return np.array(forces)


def parse_tdfrc(tdfrc_path: str, natom: int) -> list:
    """Parse the TDDFPT force file ({project}-td_forces-1_0.tdfrc).

    File layout (demo verified): for every requested excited state,
        # STATE NR.   n    <excitation eV> eV   Oscillator strength: <f>
        <natom>
        <blank line>
        natom lines of fx fy fz (a.u.)
    The blocks appear in the &PRINT &FORCES LIST order.

    Parameters
    ----------
    tdfrc_path : str
        Path to the tdfrc file.
    natom : int
        Expected number of atoms.

    Returns
    -------
    list of dict
        One entry per state block with keys state (1-based CP2K
        excited-state number), excitation (eV), osc_strength,
        forces (natom, 3) in hartree/bohr.
    """
    if not os.path.exists(tdfrc_path):
        return []

    with open(tdfrc_path, 'r') as f:
        lines = f.read().splitlines()

    # The exponent sign is part of the number: CP2K writes a large value as
    # `1.00000e+01`, and a class that allows only `-` stops the match at the `+`
    # and leaves a string no float() accepts.
    header = re.compile(
        r'#\s*STATE\s+NR\.\s+(\d+)\s+([-+\d.Ee]+)\s*eV\s+'
        r'Oscillator strength:\s+([-+\d.Ee]+)')

    blocks = []
    n = 0
    while n < len(lines):
        match = header.search(lines[n])
        if not match:
            n += 1
            continue

        state = int(match.group(1))
        excitation = float(match.group(2))
        osc_strength = float(match.group(3))

        # Skip the atom-count and blank lines, then read natom rows
        forces = []
        m = n + 1
        while m < len(lines) and len(forces) < natom:
            parts = lines[m].split()
            m += 1
            if len(parts) != 3:
                continue
            try:
                forces.append([float(parts[0]), float(parts[1]),
                               float(parts[2])])
            except ValueError:
                continue

        if len(forces) == natom:
            blocks.append({
                'state': state,
                'excitation': excitation,
                'osc_strength': osc_strength,
                'forces': np.array(forces),
            })
        n = m

    return blocks


def parse_stress_tensor(log_path: str):
    """Parse the CP2K stress tensor block from a stress file or log.

    Reads the nine x/y/z row values of the whole-system stress tensor.
    The reader is format tolerant so it works both on the dedicated
    ``{project}_stress_tensor.dat`` print file (the builder's FILENAME
    target) and, as a fallback, on the main ``.log``, whose block carries
    a ``STRESS|`` module prefix. Both layouts are handled::

        STRESS| Analytical stress tensor [bar]      (main log)
        STRESS|                        x       y       z
        STRESS|      x        8.10E+04   4.02E-01   1.81E-01
        ...

         STRESS TENSOR [GPa]                        (dedicated file)
                    X               Y               Z
         X     1.23456789     0.00000000     0.00000000
         ...

    Rules (dev/skills/STRESS_DEVELOPMENT.md section 2.2):
    - the header match is case-insensitive on 'stress tensor' with an
      optional '[unit]' label; headers containing 'eigen' belong to the
      eigenvector block and are skipped so its x/y/z rows (eigenvector
      components) are never mistaken for the tensor;
    - a leading CP2K module prefix token (e.g. 'STRESS|') is stripped
      before a row is read, and the row label is matched case
      insensitively;
    - the '1/3 Trace', 'Determinant' and 'Eigenvalue' lines end the
      block; when several tensor blocks exist the LAST valid one (the
      final evaluation) is returned.

    The value is the raw CP2K tensor (compressive-positive) in the
    printed unit; the caller applies the sign flip and unit conversion.

    Parameters
    ----------
    log_path : str
        Path to the dedicated stress file (or the CP2K .log fallback).

    Returns
    -------
    tuple of (np.ndarray or None, str or None)
        The raw 3x3 stress tensor and its unit label (e.g. 'bar');
        (None, None) when no valid block is found.
    """
    if not os.path.exists(log_path):
        return None, None

    with open(log_path, 'r') as f:
        lines = f.read().splitlines()

    header = re.compile(r'stress\s+tensor(?:\s*\[([^\]]+)\])?',
                        re.IGNORECASE)

    def _strip_prefix(parts):
        # Drop a leading CP2K module prefix token such as 'STRESS|'.
        while parts and parts[0].endswith('|'):
            parts = parts[1:]
        return parts

    def _row_values(raw):
        parts = _strip_prefix(raw.split())
        if len(parts) != 4:
            return None
        if parts[0].lower() not in ('x', 'y', 'z'):
            return None
        try:
            return [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            return None

    matrix = None
    unit = None
    n = 0
    while n < len(lines):
        line = lines[n]
        match = header.search(line)
        if not match or 'eigen' in line.lower():
            n += 1
            continue
        unit_label = (match.group(1) or '').strip()

        rows = []
        k = n + 1
        while k < len(lines) and len(rows) < 3:
            raw = lines[k].strip()
            k += 1
            if not raw:
                continue
            # Column header row 'x y z' (label-only, no values).
            colhead = _strip_prefix(raw.split())
            if ([p.lower() for p in colhead] == ['x', 'y', 'z']):
                continue
            values = _row_values(raw)
            if values is None:
                # 1/3 Trace / Determinant / Eigenvalue / next section.
                break
            rows.append(values)

        if len(rows) == 3:
            matrix = np.array(rows)
            unit = unit_label
        n = k

    return matrix, unit


def parse_xyz_cell(xyz_path: str) -> list:
    """Read the simulation cell of the last frame of an XYZ trajectory.

    CP2K states the cell in the frame's comment line --
    ``Lattice=`` followed by the nine components of the three cell
    vectors, in Angstrom, quoted or braced depending on the version --
    when the run is periodic. Taking it back matters for the same reason
    ``fin_coord`` does: whatever the engine moved is what the next
    evaluation must be told about, and a fixed-cell run simply reports
    the cell it was given, which the framework notices and leaves alone.

    The comment line is read from the same frame :func:`parse_xyz` takes
    its coordinates from, so the two cannot describe different steps.

    Parameters
    ----------
    xyz_path : str
        Path to the .xyz file.

    Returns
    -------
    list of list of float
        The (3, 3) lattice of the last frame, rows being the cell
        vectors in Angstrom; empty when the file is missing or no frame
        carried a cell at all.
    """
    if not os.path.exists(xyz_path):
        return []

    with open(xyz_path, 'r') as handle:
        lines = handle.read().splitlines()

    lattice = []
    n = 0
    while n < len(lines):
        try:
            natom = int(lines[n].split()[0])
        except (ValueError, IndexError):
            n += 1
            continue
        comment = lines[n + 1] if n + 1 < len(lines) else ''
        found = _lattice_from_comment(comment)
        if found:
            lattice = found
        n += 2 + natom
    return lattice


def _lattice_from_comment(comment: str) -> list:
    """Pull the nine lattice components out of an XYZ comment line.

    Only the numbers after ``Lattice=`` are taken, and any leading word
    inside the value (some builds print ``"Cell"`` there) is dropped, so a
    line that mentions a cell in another context cannot supply them.
    """
    marker = comment.find('Lattice=')
    if marker < 0:
        return []
    numbers = []
    for token in comment[marker + len('Lattice='):].replace('"', ' ') \
            .replace('{', ' ').replace('}', ' ').split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    if len(numbers) < 9:
        return []
    lattice = [[numbers[0], numbers[1], numbers[2]],
               [numbers[3], numbers[4], numbers[5]],
               [numbers[6], numbers[7], numbers[8]]]
    # An all-zero lattice is how a molecular structure is written with the
    # cell slot left in place (ASE stamps exactly that, with pbc="F F F").
    # Reporting it would hand the framework a cell of zero volume and a
    # contract violation it was right to reject, so a run that never had a
    # cell reports none.
    if not any(abs(value) > 0.0 for row in lattice for value in row):
        return []
    return lattice


def parse_xyz(xyz_path: str) -> list:
    """Parse a standard XYZ trajectory file and return the last frame.

    GEO_OPT writes the relaxed structure and MD appends one frame
    per step into {project}-pos-1.xyz; the final geometry is always
    the last frame.

    Parameters
    ----------
    xyz_path : str
        Path to the .xyz file.

    Returns
    -------
    list of list of float
        Coordinates in Angstrom, one row per atom of the last
        frame; empty when the file is missing or malformed.
    """
    if not os.path.exists(xyz_path):
        return []

    with open(xyz_path, 'r') as f:
        lines = f.read().splitlines()

    coord = []
    n = 0
    while n < len(lines):
        try:
            natom = int(lines[n].split()[0])
        except (ValueError, IndexError):
            n += 1
            continue

        frame = []
        for line in lines[n + 2: n + 2 + natom]:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                frame.append([float(parts[1]), float(parts[2]),
                              float(parts[3])])
            except ValueError:
                continue

        if len(frame) == natom:
            coord = frame
        n += 2 + natom

    return coord
