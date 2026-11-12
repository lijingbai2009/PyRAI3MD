"""
======================================================================
This is a standalone xTB calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module parses xTB output files (engrad and charge files)
into standardized calculation results.

"""

import os
import warnings

import numpy as np

# Angstrom -> Bohr, matching the builder's coord/lattice writer so the
# gradlatt cell (Bohr) can be checked against the requested lattice.
ANG_TO_BOHR = 1.0 / 0.529177249


def _read_gradlatt_3x3(rows):
    """Parse three gradlatt rows into a 3x3 float array.

    xTB writes Turbomole-style fixed records that may use a 'D'/'d'
    exponent; each of the three rows must carry exactly three finite
    values (dev/skills/STRESS_DEVELOPMENT.md section 3.2).

    Parameters
    ----------
    rows : list of str
        Three text lines.

    Returns
    -------
    np.ndarray or None
        The (3, 3) block, or None when malformed.
    """
    try:
        array = np.array([
            [float(x.replace('D', 'E').replace('d', 'e'))
             for x in line.split()]
            for line in rows
        ])
    except (ValueError, IndexError):
        return None
    if array.shape != (3, 3) or not np.isfinite(array).all():
        return None
    return array


def _expected_cell_volume(config):
    """Requested cell volume in Bohr^3 from config.lattice_vector.

    Returns None when no valid periodic cell is declared (an all-zero
    or absent lattice), so the caller skips the staleness guard for
    non-periodic runs.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration carrying lattice_vector (3x3, Angstrom).

    Returns
    -------
    float or None
        The cell volume in Bohr^3, or None when unavailable.
    """
    lattice = getattr(config, 'lattice_vector', None)
    if lattice is None:
        return None
    array = np.asarray(lattice, dtype=float)
    if array.size != 9:
        return None
    array = array.reshape(3, 3)
    volume_ang = abs(np.linalg.det(array))
    if volume_ang <= 0.0:
        return None
    return volume_ang * ANG_TO_BOHR ** 3


def _parse_gradlatt(calcdir, project, expected_volume=None):
    """Extract the tensile-positive stress tensor from the gradlatt file.

    Periodic GFN-FF gradient runs write a Turbomole 'gradlatt' record:
    the first three rows are the lattice vectors H (Bohr) and the next
    three are the lattice gradient G (Hartree/Bohr). The stress is
    ``G.T @ H / abs(det(H))`` in Hartree/Bohr^3, tensile-positive
    (dev/skills/STRESS_DEVELOPMENT.md section 3.2). The last record of the
    file is used. The file name follows the run namespace, so both
    '{project}.gradlatt' and the un-prefixed 'gradlatt' are probed.

    Parameters
    ----------
    calcdir : str
        Calculation directory.
    project : str
        xTB project (namespace) name.
    expected_volume : float or None
        Requested cell volume in Bohr^3. When given, the gradlatt cell
        volume must match it within a tight relative tolerance; a gross
        mismatch means the file is a stale leftover of a previous run
        with a different cell, so the stress is treated as unavailable
        (dev/skills/STRESS_DEVELOPMENT.md section 3.2). The volume is used
        because it is invariant to the lattice row/column convention.

    Returns
    -------
    np.ndarray or None
        The (3, 3) stress tensor in Hartree/Bohr^3, or None when the
        gradlatt file is missing, malformed, cell-inconsistent, or
        writes an all-zero lattice gradient (non-blocking).
    """
    candidates = [os.path.join(calcdir, f'{project}.gradlatt'),
                  os.path.join(calcdir, 'gradlatt')]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None

    with open(path, 'r') as f:
        lines = f.read().splitlines()

    records = [j for j, line in enumerate(lines) if 'cycle' in line]
    if not records:
        return None
    i = records[-1]
    if i + 7 > len(lines):
        return None

    h_matrix = _read_gradlatt_3x3(lines[i + 1:i + 4])
    g_matrix = _read_gradlatt_3x3(lines[i + 4:i + 7])
    if h_matrix is None or g_matrix is None:
        return None

    volume = abs(np.linalg.det(h_matrix))
    if volume <= 0.0:
        return None
    if not np.any(g_matrix):
        # An exactly-zero lattice gradient is the engine writing the record
        # without computing it, not a box at equilibrium: measured on the
        # deployment, xtb 6.7.1 writes nine `0.0000000000000E+00` rows for a
        # periodic GFN1-xTB gradient run (`|dE/dlatt| =  0.000000`), while the
        # same coord file under GFN-FF writes `|dE/dlatt| =  0.008680` and a
        # nonzero block. Returning it as a tensor would tell a variable-cell
        # step that this box carries no stress, which is the one reading the
        # record does not support. A physical near-zero is never nine exact
        # zeros, so this refuses the sentinel rather than the physics.
        warnings.warn(
            f'xTB: {path} writes the lattice gradient as exact zeros, so this '
            f'method does not report a cell stress in this build; the stress '
            f'tensor is left unreported rather than read as zero',
            stacklevel=2)
        return None
    if expected_volume and expected_volume > 0.0:
        if abs(volume - expected_volume) / expected_volume > 1.0e-3:
            # Stale gradlatt from a different cell: do not trust it.
            return None
    return g_matrix.T @ h_matrix / volume


def _extract_velocities(config, calcdir):
    """Extract the final velocities (reserved fin_velocity channel).

    The xtb skill performs no skill-internal MD propagation that
    reports final velocities, so the fin_velocity return channel is
    reserved with an empty value; the uniform channel keeps every
    skill result contract aligned.

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.

    Returns
    -------
    list
        Empty list (no velocity output).
    """
    return []


def parse_output(config, calcdir: str, workflow) -> dict:
    """Parse xTB output files based on the calculation workflow.

    Reads the total energy and gradient from the engrad file and
    the atomic charges from the charge file written by xTB
    ('charges' for GFN-x methods, 'gfnff_charges' for GFN-FF).

    Parameters
    ----------
    config : SkillConfig
        Configuration container.
    calcdir : str
        Calculation directory.
    workflow : WorkflowGraph
        Compiled workflow graph.

    Returns
    -------
    dict
        Parsed results dictionary with 'energy', 'gradient', and
        'computed_charges' ((1, natom) atom charges computed by xTB;
        the key is omitted when xTB wrote no charge file, matching
        the empty-default "not computed" schema contract).

    Raises
    ------
    RuntimeError
        If the engrad file is missing or contains no data.
    """
    project = config.xtb_project
    natom = config.natom
    method = config.method

    engrad_path = os.path.join(calcdir, f'{project}.engrad')
    if not os.path.exists(engrad_path):
        raise RuntimeError(
            f"xTB engrad file not found: {engrad_path}. "
            f"Check {os.path.join(calcdir, f'{project}.out')} for details."
        )

    with open(engrad_path, 'r') as f:
        log = f.read().splitlines()

    # Pack ground-state energy and gradient
    energy = []
    gradient_values = []
    for n, line in enumerate(log):
        if 'The current total energy in Eh' in line:
            energy = [float(log[n + 2])]

        if 'The current gradient in Eh/bohr' in line:
            for value_line in log[n + 2: n + 2 + natom * 3]:
                value = value_line.strip()
                if value:
                    gradient_values.append(float(value))

    if len(energy) == 0:
        raise RuntimeError(
            f"xTB engrad file contains no energy: {engrad_path}"
        )

    gradient = np.array(gradient_values).reshape((1, natom, 3))

    results = {
        'energy': np.array(energy),
        'gradient': gradient,
        # Whole-system stress tensor (1, 3, 3); zero-filled and only
        # populated from the periodic gradlatt when config.stress is set
        # (non-blocking). xTB is a ground-state method: S0 only.
        'stress_tensor': np.zeros((1, 3, 3)),
        # Reserved fin_velocity return channel (no velocity output)
        'fin_velocity': _extract_velocities(config, calcdir),
    }

    # Non-blocking stress read: a periodic GFN-FF run writes the
    # gradlatt lattice gradient; a missing/malformed file keeps the
    # zero-filled (1, 3, 3) contract intact. The requested cell volume
    # (lattice_vector in Angstrom -> Bohr^3) guards against a stale
    # gradlatt left by a previous run with a different cell.
    if getattr(config, 'stress', False):
        expected_volume = _expected_cell_volume(config)
        stress = _parse_gradlatt(calcdir, project,
                                 expected_volume=expected_volume)
        if stress is not None:
            results['stress_tensor'][0] = stress

    # Pack atomic charges written by xTB through the framework
    # computed_charges channel, state-resolved as (1, natom) for the
    # single ground state. Missing charge file means not computed.
    if method == 'gfn_ff':
        charge_file = os.path.join(calcdir, 'gfnff_charges')
    else:
        charge_file = os.path.join(calcdir, 'charges')

    if os.path.exists(charge_file):
        # The calcdir is reused for the backend's lifetime, so a charge file
        # can survive a run that did not rewrite it. Every run rewrites the
        # geometry before it starts, so a charge file older than that
        # geometry belongs to a previous step and is ignored rather than
        # returned as this step's charges (the same rule `gradlatt` follows).
        anchor = os.path.join(calcdir, f'{project}.xyz')
        stale = (os.path.exists(anchor)
                 and os.path.getmtime(charge_file)
                 < os.path.getmtime(anchor))
        if not stale:
            atom_charges = np.loadtxt(charge_file)
            results['computed_charges'] = \
                np.array(atom_charges).reshape((1, natom))

    return results


def format_results(results: dict, config) -> dict:
    """Format parsed results for output.

    Converts numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Raw parsed results.
    config : SkillConfig
        Configuration container.

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
