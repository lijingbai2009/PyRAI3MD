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

This module parses the ORCA .out log file and extracts energies,
gradients, dipole moments, transition dipoles, spin-orbit couplings,
nonadiabatic coupling vectors, frequencies, and optimized geometries.

Missing sections are returned as empty/zero structures; the caller
(output_parser) applies zero-filling according to the profile.

"""

import os
import re
import numpy as np


def parse_log(log_path: str, config) -> dict:
    """Parse an ORCA .out log file and extract all available results.

    Parameters
    ----------
    log_path : str
        Path to the ORCA .out file.
    config : SkillConfig
        Runtime configuration container with state information.

    Returns
    -------
    dict
        Raw parsed data with keys: s0_energy, state_energies,
        gradients, dipole, absorption, soc_triples, nac_blocks,
        frequencies, final_coordinates, opt_converged, escf_energy,
        root_excitations, dispersion.
    """
    natom = config.natom or 1

    data = {
        's0_energy': None,            # gradient-state energy in Eh
        'state_energies': [],         # excitation energies in au
        'gradients': [],              # list of (natom, 3) blocks
        'dipole': [0.0, 0.0, 0.0],    # total dipole in e*bohr
        'absorption': [],             # transition dipole spectrum rows
        'soc_triples': [],            # (i, j, value) 0-based, cm^-1
        'nac_blocks': [],             # (natom, 3) per requested root, deck order
        'frequencies': [],            # cm^-1
        'final_coordinates': [],      # last geometry in Angstrom
        'opt_converged': False,
        'escf_energy': None,          # raw SCF energy, no corrections
        'root_excitations': {},       # 1-based ORCA root -> excitation
        'dispersion': None,           # additive dispersion term in Eh
    }

    if not os.path.exists(log_path):
        return data

    with open(log_path, 'r') as f:
        log = f.read().splitlines()

    final_energy = None
    escf_energy = None
    dispersion = None
    root_excitations = {}

    for n, line in enumerate(log):
        # Energy the printed nuclear gradient belongs to: the ground
        # state for a DFT gradient, but the excited root whose
        # gradient was last computed for an EnGrad + irootlist run.
        if 'FINAL SINGLE POINT ENERGY' in line:
            try:
                final_energy = float(line.split()[-1])
            except (ValueError, IndexError):
                pass

        # Raw SCF energy printed by the CIS/TD-DFT total energy block
        elif 'E(SCF)' in line:
            try:
                escf_energy = float(line.split()[-2])
            except (ValueError, IndexError):
                pass

        # Per-root excitation of that same block; carries an explicit
        # root label and more digits than the STATE summary lines.
        elif line.strip().startswith('DE('):
            parsed = _parse_de_line(line)
            if parsed is not None:
                root_excitations[parsed[0]] = parsed[1]

        # TDDFT state excitation energies (au) from gradient runs
        elif 'STATE' in line and '<S' in line:
            parsed = _parse_state_line(line)
            if parsed is not None:
                root_excitations.setdefault(parsed[0], parsed[1])

        # Damped dispersion folded into the final single point energy
        elif 'Dispersion correction' in line:
            try:
                dispersion = float(line.split()[-1])
            except (ValueError, IndexError):
                pass

        # Cartesian gradient blocks (Eh/bohr)
        elif 'CARTESIAN GRADIENT' in line:
            g = _parse_atom_block(log[n + 3: n + 3 + natom])
            if len(g) == natom:
                data['gradients'].append(g)

        # Total dipole moment (e*bohr)
        elif 'Total Dipole Moment' in line:
            try:
                data['dipole'] = [float(x) for x in line.split()[-3:]]
            except (ValueError, IndexError):
                pass

        # Vibrational frequencies (cm^-1)
        elif 'VIBRATIONAL FREQUENCIES' in line:
            data['frequencies'] = _parse_frequencies(log[n + 1:])

        # Optimized geometry: keep the last block
        elif 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            coord = _parse_coord_block(log[n + 3: n + 3 + natom])
            if len(coord) == natom:
                data['final_coordinates'] = coord

        elif 'OPTIMIZATION RUN DONE' in line:
            data['opt_converged'] = True

    # Transition dipole spectrum (also provides excitation/oscillator)
    data['absorption'] = _parse_absorption(log)

    # Spin-orbit coupling matrix elements from DoSOC runs (cm^-1)
    data['soc_triples'] = _parse_soc(log)

    # Nonadiabatic coupling vectors from NACME runs (S0-SN)
    data['nac_blocks'] = _parse_nac(log, natom)

    # Excitation energies ordered by the ORCA root label (1-based);
    # kept positionally compatible with the historic STATE-line order.
    data['root_excitations'] = root_excitations
    data['state_energies'] = [root_excitations[root]
                              for root in sorted(root_excitations)]

    data['escf_energy'] = escf_energy
    data['dispersion'] = dispersion
    data['s0_energy'] = final_energy

    return data


def _parse_state_line(line: str):
    """Parse a 'STATE  1:  E=   0.281673 au ... <S**2>' summary line.

    Parameters
    ----------
    line : str
        Log line of the TD-DFT excited states table.

    Returns
    -------
    tuple of (int, float) or None
        (1-based ORCA root, excitation energy in Eh).
    """
    parts = line.split()
    try:
        root = int(parts[1].rstrip(':'))
        return root, float(parts[3])
    except (ValueError, IndexError):
        return None


def _parse_de_line(line: str):
    """Parse a 'DE(CIS) = 0.281673020 Eh (Root  1)' total energy line.

    Parameters
    ----------
    line : str
        Log line of the CIS/TD-DFT total energy block.

    Returns
    -------
    tuple of (int, float) or None
        (1-based ORCA root, excitation energy in Eh).
    """
    parts = line.replace('=', ' ').split()
    try:
        energy = float(parts[1])
    except (ValueError, IndexError):
        return None
    try:
        root = int(parts[-1].rstrip(')'))
    except (ValueError, IndexError):
        return None
    return root, energy


def _parse_atom_block(lines: list) -> np.ndarray:
    """Parse an atom-resolved vector block (gradient/NAC style).

    Lines look like '   1   C   :    0.001    0.002   -0.003';
    the last three floats form the vector component.

    Parameters
    ----------
    lines : list of str
        Block lines, one per atom.

    Returns
    -------
    np.ndarray
        Parsed (n, 3) array (n = number of successfully parsed lines).
    """
    vectors = []
    for line in lines:
        parts = line.replace(':', ' ').split()
        if len(parts) < 3:
            continue
        try:
            vectors.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
        except ValueError:
            continue

    if not vectors:
        return np.zeros((0, 3))
    return np.array(vectors)


def _parse_coord_block(lines: list) -> list:
    """Parse a coordinate block in Angstrom.

    Lines look like '  C     0.000000   1.234567   0.000000'.

    Parameters
    ----------
    lines : list of str
        Block lines, one per atom.

    Returns
    -------
    list of list of float
        Parsed coordinates.
    """
    coord = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coord.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    return coord


def _parse_frequencies(lines: list) -> list:
    """Parse the VIBRATIONAL FREQUENCIES section.

    Lines look like '   6:      1012.34 cm**-1'.

    Parameters
    ----------
    lines : list of str
        Log lines following the section header.

    Returns
    -------
    list of float
        Frequencies in cm^-1.
    """
    frequencies = []
    started = False
    pattern = re.compile(r'^\s*\d+\s*:\s*([-\d.]+)\s*cm')
    for line in lines:
        match = pattern.match(line)
        if match:
            try:
                frequencies.append(float(match.group(1)))
                started = True
            except ValueError:
                pass
        elif started and line.strip() == '':
            # Stop at the blank line after the last frequency entry
            break
    return frequencies


def _parse_absorption(log: list) -> list:
    """Parse the absorption spectrum via transition dipole moments.

    Section rows look like:
      1( 1A  )  0.000  0.000  0.000  0.000  0.000  2.96261  0.000000000
    where the floats after the symmetry label are: X, Y, Z dipole
    components, |D|^2, |D|, excitation energy (eV), oscillator strength.

    Parameters
    ----------
    log : list of str
        Full log lines.

    Returns
    -------
    list of dict
        One entry per state with keys: state, transition_dipole,
        excitation (eV), osc_strength.
    """
    rows = []
    in_section = False

    for line in log:
        if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
            in_section = True
            continue

        if not in_section:
            continue

        # Section ends at the next section header
        if line.strip().startswith('ABSORPTION SPECTRUM') and 'TRANSITION ELECTRIC' not in line:
            break

        match = re.match(r'^\s*(\d+)\s*(\(.*?\))?\s*([-+\d.Ee\s]+)$', line)
        if not match:
            continue

        try:
            state = int(match.group(1))
            values = [float(x) for x in match.group(3).split()]
        except (ValueError, TypeError):
            continue

        if len(values) >= 7:
            rows.append({
                'state': state,
                'transition_dipole': values[0:3],
                'excitation': values[5],
                'osc_strength': values[6],
            })
        elif len(values) >= 3:
            # Fallback: only dipole components available
            rows.append({
                'state': state,
                'transition_dipole': values[0:3],
                'excitation': 0.0,
                'osc_strength': 0.0,
            })

    return rows


_SOCME_ROW = re.compile(
    r'^\s*(\d+)\s+(\d+)\s+'
    r'\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)\s+'
    r'\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)\s+'
    r'\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)\s*$')


def _parse_soc(log: list) -> list:
    """Parse the `CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS` table.

    One row per (triplet root, singlet root) with the spin-orbit matrix
    element in the Z, X and Y components, each printed as `(Re, Im)`; the
    value kept is the norm over the three components, in cm^-1.

    The layout is taken from the deployment (round `tmp_20261101_221306`,
    ORCA 6.1.1)::

        CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS
             Root                          <T|HSO|S>  (Re, Im) cm-1
           T      S             Z                     X                     Y
           1      0    (   0.00 ,   -1.88 )    (   0.00 ,    0.00 ) ...

    This reader used to look for a header ORCA never prints
    (`SPIN-ORBIT COUPLING (cm**-1)`) and a three-column row, so it matched
    nothing on a real log - and the packer turned "matched nothing" into
    zeros. The strict packer (#447) is what surfaced it.

    Returns
    -------
    list of tuple
        (triplet_root, singlet_root, magnitude) with **ORCA's own** 1-based
        triplet root and 0-based singlet root; mapping onto framework state
        indices is the packer's job, because it knows the spin ladders.
    """
    triples = []
    in_section = False
    for line in log:
        if 'CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS' in line:
            in_section = True
            continue
        if not in_section:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith('---') \
                or stripped.startswith('Root') or stripped[0].isalpha():
            continue
        match = _SOCME_ROW.match(line)
        if match is not None:
            numbers = [float(g) for g in match.groups()[2:]]
            magnitude = (sum(value ** 2
                             for value in numbers) ** 0.5)
            triples.append((int(match.group(1)), int(match.group(2)),
                            magnitude))
            continue
        # Anything else after the table started is the next section.
        if triples:
            break
    return triples


def _parse_nac(log: list, natom: int) -> list:
    """Parse the `CARTESIAN NON-ADIABATIC COUPLINGS` blocks of a NACME run.

    One block per requested root, in the deck's root order, in the shape the
    deployment prints (round `tmp_20261101_224544`, ORCA 6.1.1)::

        ---------------------------------
        CARTESIAN NON-ADIABATIC COUPLINGS
                  <GS|d/dx|ES>
        ---------------------------------
           1   C   :   -0.000000029  0.000000071  -0.036012173
           ...
        Norm of the NACs                   ...    0.3279089623

    The block names no root, so its position is its identity - the packer
    zips it with the requested pairs and refuses a count that does not match.

    The previous reader looked for `Coupling vector between states i and j`,
    a string ORCA does not print, so it matched nothing on a real log and the
    packer turned that into zero couplings.

    Returns
    -------
    list of ndarray
        (natom, 3) blocks in print order.
    """
    blocks = []
    index = 0
    while index < len(log):
        if 'CARTESIAN NON-ADIABATIC COUPLINGS' not in log[index]:
            index += 1
            continue
        vec = _parse_atom_block(log[index + 1: index + 2 + natom * 2])
        if len(vec) >= natom:
            blocks.append(vec[:natom])
        index += 1
    return blocks

def parse_xyz(xyz_path: str) -> list:
    """Parse a standard XYZ file for the final geometry.

    ORCA updates the ``{project}.xyz`` file after every optimization
    cycle, so the relaxed structure lives there (the input geometry
    only exists inside ``{project}.inp``).

    Parameters
    ----------
    xyz_path : str
        Path to the .xyz file.

    Returns
    -------
    list of list of float
        Coordinates in Angstrom, one row per atom; empty when the
        file is missing or malformed.
    """
    if not os.path.exists(xyz_path):
        return []

    with open(xyz_path, 'r') as f:
        lines = f.read().splitlines()

    if len(lines) < 3:
        return []

    try:
        natom = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return []

    coord = []
    for line in lines[2:2 + natom]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            coord.append(
                [float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue

    return coord if len(coord) == natom else []


def parse_hessian(hess_path: str):
    """Parse the ORCA .hess file for the Hessian matrix.

    The $hessian block starts with the dimension (3*natom), followed
    by column-blocked matrix rows prefixed with the row index.

    Parameters
    ----------
    hess_path : str
        Path to the .hess file.

    Returns
    -------
    np.ndarray
        Hessian matrix (3*natom, 3*natom), empty if unavailable.
    """
    if not os.path.exists(hess_path):
        return np.zeros(0)

    with open(hess_path, 'r') as f:
        lines = f.read().splitlines()

    dim = 0
    in_hessian = False
    rows = []

    for line in lines:
        stripped = line.strip()
        if stripped == '$hessian':
            in_hessian = True
            rows = []
            continue
        if not in_hessian:
            continue
        if stripped.startswith('$') and stripped != '$hessian':
            break
        if not stripped:
            continue

        parts = stripped.split()
        if dim == 0:
            try:
                dim = int(parts[0])
                rows = [[] for _ in range(dim)]
            except (ValueError, IndexError):
                break
            continue

        try:
            idx = int(parts[0])
            values = [float(x) for x in parts[1:]]
        except (ValueError, IndexError):
            continue  # column index header line

        if 0 <= idx < dim:
            rows[idx].extend(values)

    if dim == 0:
        return np.zeros(0)

    try:
        return np.array(rows)
    except ValueError:
        return np.zeros(0)


def parse_hess_frequencies(hess_path: str):
    """Parse the vibrational frequencies from the ORCA .hess file.

    The $vibrational_frequencies block starts with the number of
    modes (3*natom), followed by '<index> <frequency>' lines in
    cm^-1 (same layout the legacy wigner sampler read for ORCA).

    Parameters
    ----------
    hess_path : str
        Path to the .hess file.

    Returns
    -------
    np.ndarray
        Frequencies (3*natom,) in cm^-1, empty if unavailable.
    """
    if not os.path.exists(hess_path):
        return np.zeros(0)

    with open(hess_path, 'r') as f:
        lines = f.read().splitlines()

    nmode = 0
    for n, line in enumerate(lines):
        if '$vibrational_frequencies' in line:
            try:
                nmode = int(lines[n + 1].split()[0])
            except (ValueError, IndexError):
                return np.zeros(0)
            freqs = []
            for row in lines[n + 2: n + 2 + nmode]:
                parts = row.split()
                if len(parts) < 2:
                    continue
                try:
                    freqs.append(float(parts[1]))
                except ValueError:
                    continue
            return np.array(freqs) if len(freqs) == nmode else np.zeros(0)

    return np.zeros(0)
