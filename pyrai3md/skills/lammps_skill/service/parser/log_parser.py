"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module parses the LAMMPS log file thermo output to extract
the potential energy.

Two thermo output formats are supported:
  - table format ('thermo_style custom ...'): a header line starting
    with 'Step' lists the column names (e.g. Step PotEng TotEng)
    followed by numeric data rows
  - multi format ('thermo_style multi', used by the internal-
    iteration profiles): entries like 'PotEng   =        -0.3758'

"""

import os

from ..utils import log_tail


def _parse_table_format(log_lines: list):
    """Parse the thermo table format for the potential energy.

    The last thermo header carrying an energy column (PotEng
    preferred, TotEng fallback) selects the table; the last data row
    of that table holds the energy.

    Parameters
    ----------
    log_lines : list of str
        Lines of the LAMMPS log file.

    Returns
    -------
    tuple (float or None, int)
        The parsed energy and the index of its data row; (None, -1)
        when no table with an energy column is found.
    """
    header_index = -1
    energy_column = -1

    # Locate the last thermo header carrying an energy column
    for n, line in enumerate(log_lines):
        tokens = line.split()
        if not tokens or tokens[0] != 'Step':
            continue
        lower_tokens = [token.lower() for token in tokens]
        if 'poteng' in lower_tokens:
            energy_column = lower_tokens.index('poteng')
            header_index = n
        elif 'toteng' in lower_tokens:
            energy_column = lower_tokens.index('toteng')
            header_index = n

    if header_index < 0:
        return None, -1

    # Read data rows after the header until a non-numeric line
    energy = None
    last_index = -1
    for n, line in enumerate(log_lines[header_index + 1:],
                              start=header_index + 1):
        tokens = line.split()
        if len(tokens) <= energy_column:
            break
        try:
            energy = float(tokens[energy_column])
            last_index = n
        except ValueError:
            break

    return energy, last_index


def _parse_multi_format(log_lines: list):
    """Parse the thermo_style multi format for the potential energy.

    Multi-style thermo output lists one entry per line, e.g.
    'PotEng   =        -0.3758'. The last occurrence holds the
    most recent energy.

    Parameters
    ----------
    log_lines : list of str
        Lines of the LAMMPS log file.

    Returns
    -------
    tuple (float or None, int)
        The parsed energy and the line index of its occurrence;
        (None, -1) when no multi-style PotEng entry is found.
    """
    energy = None
    last_index = -1
    for n, line in enumerate(log_lines):
        tokens = line.split()
        if len(tokens) < 3 or tokens[0].lower() != 'poteng':
            continue
        if tokens[1] != '=':
            continue
        try:
            energy = float(tokens[2])
            last_index = n
        except ValueError:
            continue

    return energy, last_index


def parse_log(log_lines: list) -> float:
    """Parse the LAMMPS log thermo output for the potential energy.

    Both the table format ('thermo_style custom', used by the
    single-point section) and the multi format ('thermo_style multi',
    used by the internal-iteration profiles) are supported. When both
    formats appear, the most recent occurrence wins.

    Note: the internal-iteration profiles run with
    'thermo_modify norm yes', so the returned value is the per-atom
    average; the caller multiplies it by natom to recover the total
    energy when needed.

    Parameters
    ----------
    log_lines : list of str
        Lines of the LAMMPS log file.

    Returns
    -------
    float
        Potential energy in the native LAMMPS units style.

    Raises
    ------
    RuntimeError
        If no thermo output with an energy value is found.
    """
    table_energy, table_index = _parse_table_format(log_lines)
    multi_energy, multi_index = _parse_multi_format(log_lines)

    if table_energy is None and multi_energy is None:
        # Surface the LAMMPS failure directly: the log tail carries the
        # actual ERROR line, so a remote run is diagnosable without
        # fetching the calculation directory.
        raise RuntimeError(
            "LAMMPS log contains no thermo output with an energy "
            "value (table 'Step ... PotEng' or multi-style "
            "'PotEng = ...'); the calculation likely aborted. "
            "LAMMPS log tail:\n" + log_tail(log_lines)
        )

    if table_energy is not None and multi_energy is not None:
        return table_energy if table_index >= multi_index else multi_energy

    return table_energy if table_energy is not None else multi_energy


def parse_stress_file(stress_path: str):
    """Parse the six virial pressure components from a stress file.

    The single-point builder writes the configurational pressure to a
    dedicated file with a LAMMPS ``print ... file`` command::

        STRESS <xx> <yy> <zz> <xy> <xz> <yz>

    (component order xx yy zz xy xz yz, dev/skills/STRESS_DEVELOPMENT.md
    section 4.1). ``print ... file`` overwrites, so the file holds this
    step's line alone; a missing file or a malformed line returns None and
    the caller turns that into a failure, since the thermo table in the log
    quantizes these numbers to eight digits and is no longer read.

    Parameters
    ----------
    stress_path : str
        Path to the dedicated stress file ({title}.stress).

    Returns
    -------
    list of float or None
        The six pressure components [xx, yy, zz, xy, xz, yz] in the
        native LAMMPS pressure unit, or None when unavailable.
    """
    if not os.path.exists(stress_path):
        return None

    with open(stress_path, 'r') as f:
        lines = f.read().splitlines()

    values = None
    for line in lines:
        tokens = line.split()
        if len(tokens) >= 7 and tokens[0].upper() == 'STRESS':
            try:
                values = [float(token) for token in tokens[1:7]]
            except ValueError:
                continue
    return values


def parse_cell_file(path):
    """The six box numbers an engine-side box relaxation ends with.

    `fix box/relax` moves the box inside `minimize`, so the answer is a cell the
    framework never declared and the coordinates it returned belong to it - the
    same fact `fin_cell` carries for a barostat run, read here from the line the
    input writer prints after the relaxation. A missing or malformed file is
    None, and the caller refuses the box rather than keeping the one it asked
    for: a box that travelled nowhere looks exactly like one that did not move.

    Parameters
    ----------
    path : str
        Path to the ``{project}_{profile}.cell`` file.

    Returns
    -------
    list of float or None
        ``[lx, ly, lz, xy, xz, yz]`` in the run's length unit, or None.
    """
    if not os.path.exists(path):
        return None
    with open(path, 'r') as handle:
        for line in handle.read().splitlines():
            tokens = line.split()
            if len(tokens) == 7 and tokens[0] == 'CELL':
                try:
                    return [float(value) for value in tokens[1:]]
                except ValueError:
                    return None
    return None
