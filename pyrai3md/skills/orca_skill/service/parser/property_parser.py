"""
======================================================================
This is a standalone ORCA calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module parses the ORCA .property.txt companion file written by
gradient and property calculations, which carries the ground-state
reference energy, the per-root state energies and the additive
correction terms at full double precision.

Every energy reported here is dispersion-free; the caller adds the
correction terms to obtain the energies consistent with the ORCA
nuclear gradient. Missing blocks stay None/empty so the caller can
fall back to the .out log.

"""

import os

# Blocks whose scalar field holds an energy term ORCA folds into
# FINAL SINGLE POINT ENERGY on top of the raw SCF/state energies.
_CORRECTION_FIELDS = (
    ('VdW_Correction', 'vdW'),
)


def parse_property(property_path: str) -> dict:
    """Parse an ORCA .property.txt file for state-resolved energies.

    Parameters
    ----------
    property_path : str
        Path to the .property.txt file.

    Returns
    -------
    dict
        Raw property data with keys: e_scf (ground-state energy),
        state_totals (framework state index -> dispersion-free total
        energy), corrections (sum of additive correction terms),
        final_energy (ORCA final single point energy). Empty/None
        entries when the file or a block is missing.
    """
    data = {
        'e_scf': None,
        'state_totals': {},
        'corrections': None,
        'final_energy': None,
    }

    if not os.path.exists(property_path):
        return data

    with open(property_path, 'r') as f:
        lines = f.read().splitlines()

    blocks = _split_blocks(lines)

    cis = blocks.get('CIS_Energies', {})
    if 'e0' in cis:
        data['e_scf'] = cis['e0']
    data['state_totals'] = _cis_state_totals(cis)

    corrections = 0.0
    found = False
    for block_name, field in _CORRECTION_FIELDS:
        value = blocks.get(block_name, {}).get(field)
        if value is not None:
            corrections += value
            found = True
    if found:
        data['corrections'] = corrections

    data['final_energy'] = blocks.get('Single_Point_Data', {}).get('FinalEnergy')

    return data


def _split_blocks(lines: list) -> dict:
    """Split the property file into {block name: {field: value}}.

    Scalar fields carry their value on the header line; one-column
    arrays carry a column-index line followed by 'row value' lines.
    Unparsable fields are skipped rather than raising.

    Parameters
    ----------
    lines : list of str
        Raw property file lines.

    Returns
    -------
    dict
        Nested mapping of block name and field name to the parsed
        value (float for scalars, list of float for arrays).
    """
    blocks = {}
    block = {}
    name = None
    pending = None
    skip_row = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith('$End'):
            pending = None
            if name:
                blocks[name] = block
            block = {}
            name = None
            continue

        if stripped.startswith('$'):
            pending = None
            if name:
                blocks[name] = block
            tokens = stripped[1:].split()
            name = tokens[0] if tokens else ''
            block = {}
            continue

        if name is None:
            continue

        if not stripped.startswith('&'):
            # Array body: the first line indexes the columns and the
            # following 'row-index value...' lines carry the entries.
            if pending is None or skip_row:
                skip_row = False
                continue
            field, values = pending
            if _read_array_row(stripped, values):
                block[field] = values
            else:
                pending = None
            continue

        field = stripped[1:].split()[0].split('[')[0]
        skip_row = True
        if '[&Type' in stripped and 'ArrayOf' in stripped.split('[&Type')[1]:
            block[field] = []
            pending = (field, block[field])
        else:
            pending = None
            value = _scalar_value(stripped)
            if value is not None:
                block[field] = value

    if name:
        blocks[name] = block

    return blocks


def _read_array_row(line: str, values: list) -> bool:
    """Append one 'row-index value...' array line to the values list.

    Parameters
    ----------
    line : str
        Candidate array row.
    values : list
        Accumulated array values, appended in file order.

    Returns
    -------
    bool
        False when the line is not an array row (the field ended).
    """
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
    except ValueError:
        return False
    try:
        values.extend(float(x) for x in parts[1:])
    except ValueError:
        return False
    return True


def _scalar_value(line: str):
    """Return the first float following the [&Type ...] descriptor.

    Parameters
    ----------
    line : str
        A property file field line.

    Returns
    -------
    float or None
        Parsed value, or None when the line carries no number.
    """
    tail = line.split(']', 1)[-1] if '[&Type' in line else line
    for token in tail.split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def _cis_state_totals(cis: dict) -> dict:
    """Map CIS root energies onto framework state indices.

    ORCA labels the roots 0-based within the CIS block, and framework
    state 0 is the ground state, so root r becomes state r + 1. The
    &Root labels drive the mapping; the positional order is only a
    fallback when the labels are absent.

    Parameters
    ----------
    cis : dict
        Parsed $CIS_Energies block.

    Returns
    -------
    dict
        Framework state index -> dispersion-free total energy.
    """
    totals = cis.get('totalEnergy') or []
    if not totals:
        return {}

    roots = cis.get('Root') or []
    state_totals = {}
    for n, value in enumerate(totals):
        root = roots[n] if n < len(roots) else n
        state_totals[int(root) + 1] = value
    return state_totals
