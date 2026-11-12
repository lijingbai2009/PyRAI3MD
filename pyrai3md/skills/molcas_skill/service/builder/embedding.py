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

This module resolves ESPF embedding data: point charges or an external
potential/field. It builds the &ESPF block and never emits XField.
"""

import os

BOHR_TO_ANGSTROM = 0.529177210903


def _readable(path, calcdir=None):
    """Resolve a read_charge filename from calcdir or the cwd."""
    if os.path.isabs(path):
        return path
    candidates = []
    if calcdir:
        candidates.append(os.path.join(calcdir, path))
    candidates.append(path)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _data_lines(path):
    """Yield non-blank, non-comment lines from a charge/potential file."""
    with open(path, 'r') as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            yield line


def load_charge_file(path):
    """Load a charge file in either accepted layout.

    Self-describing layout: first line ``charge angstrom`` or
    ``charge bohr``, then ``x y z q`` rows.
    Legacy ``.charge`` layout: a charge count line, a comment line, then
    one ``q x y z`` row per charge with Angstrom coordinates. That is the
    file the framework writes and reads for its own charge store, so a
    name ending in ``.charge`` (including the ``read_charge: true`` alias,
    which expands to ``<project>.charge``) has to be readable here. The
    skill never imports the framework, so the two readers are held
    together by this layout, not by shared code.

    Both branches return ``[q, x, y, z]`` rows in Angstrom.

    Raises
    ------
    ValueError
        When the layout cannot be told, the legacy count does not match
        the rows it declares, or a row is short or non-numeric.
    """
    import math

    raw = open(path, 'r').read().splitlines()
    first = next((line for line in raw if line.strip()), None)
    if first is None:
        raise ValueError("charge file '%s' is empty" % path)
    tokens = first.split()
    if len(tokens) == 1 and tokens[0].lstrip('-+').isdigit():
        return _load_legacy_charge_rows(path, raw, tokens[0])
    lines = list(_data_lines(path))
    header = lines[0].lower().split()
    if len(header) < 2 or header[0] != 'charge':
        raise ValueError(
            "charge file '%s' must start with 'charge angstrom', "
            "'charge bohr', or a legacy charge count" % path)
    unit = header[1]
    if unit not in ('angstrom', 'bohr'):
        raise ValueError(
            "charge file '%s' has unsupported length unit %r"
            % (path, unit))
    factor = 1.0 if unit == 'angstrom' else BOHR_TO_ANGSTROM
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                "charge file '%s' row must be 'x y z q': %r"
                % (path, line))
        x, y, z, q = (float(item) for item in parts)
        rows.append([q, x * factor, y * factor, z * factor])
    if not rows:
        raise ValueError("charge file '%s' has no charge rows" % path)
    return rows


def _load_legacy_charge_rows(path, raw, count_token):
    """Read the framework's legacy ``.charge`` layout.

    Positional on purpose: line 0 is the count, line 1 the comment, lines
    2 onward the charges. A comment line here is part of the format, not
    something to skip, because the framework's own reader takes those
    exact offsets.
    """
    import math

    ncharge = int(count_token)
    rows = raw[2:2 + ncharge]
    if len(rows) != ncharge:
        raise ValueError(
            "charge file '%s' declares %d charges but holds %d rows"
            % (path, ncharge, len(rows)))
    out = []
    for line in rows:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                "charge file '%s' row must be 'q x y z': %r" % (path, line))
        try:
            row = [float(item) for item in parts[:4]]
        except ValueError:
            raise ValueError(
                "charge file '%s' has a non-numeric row: %r" % (path, line))
        if not all(math.isfinite(value) for value in row):
            raise ValueError(
                "charge file '%s' has a non-finite row: %r" % (path, line))
        out.append(row)
    if not out:
        raise ValueError("charge file '%s' has no charge rows" % path)
    return out



def load_embedding_file(path, natom=0):
    """Read a ``{title}.charge`` embedding document.

    The first line declares what the file holds, so the representation is a
    property of the data rather than of whoever reads it:

        point angstrom external   # rows: q x y z, arbitrary external points
        point angstrom atom       # rows: q x y z, one per mid-shell atom
        field au                  # rows: V Fx Fy Fz, one per QM atom

    Two older spellings are still read: ``charge angstrom|bohr`` followed by
    ``x y z q`` rows (this module's first layout) and a bare count line
    followed by ``q x y z`` rows (the framework charge store's layout).

    An ``atom`` role is refused here. Only the framework's charge store
    consumes an atom-indexed file - it reads the q column and takes the
    positions from the live geometry, so a frozen copy of the coordinates in
    the file is never the one used. A skill run that reads the file itself
    would use those coordinates, and the two would drift apart as soon as the
    geometry moved.

    Returns ``(mode, rows)`` with mode ``'charge'`` or ``'field'``; charge
    rows are ``[q, x, y, z]`` in Angstrom, field rows ``[V, Fx, Fy, Fz]`` in
    atomic units.
    """
    raw = open(path, 'r').read().splitlines()
    lines = [line.strip() for line in raw
             if line.strip() and not line.strip().startswith(('#', '!'))]
    if not lines:
        raise ValueError("charge file '%s' is empty" % path)
    tokens = lines[0].split()
    head = tokens[0].lower()

    if head == 'point':
        if len(tokens) != 3:
            raise ValueError(
                "charge file '%s': a point declaration is "
                "'point <angstrom|bohr> <atom|external>'; got %r"
                % (path, lines[0]))
        unit = tokens[1].lower()
        role = tokens[2].lower()
        if unit not in ('angstrom', 'bohr'):
            raise ValueError(
                "charge file '%s' has unsupported length unit %r"
                % (path, tokens[1]))
        if role not in ('atom', 'external'):
            raise ValueError(
                "charge file '%s' row role must be atom or external; got %r"
                % (path, tokens[2]))
        if role == 'atom':
            raise ValueError(
                "charge file '%s' is atom-indexed: the framework's charge "
                "store consumes it (q column only, positions from the live "
                "geometry). A skill run needs an 'external' file, or the "
                "per-step point_charges array." % path)
        factor = 1.0 if unit == 'angstrom' else BOHR_TO_ANGSTROM
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    "charge file '%s' row must be 'q x y z': %r"
                    % (path, line))
            q, x, y, z = (float(item) for item in parts)
            rows.append([q, x * factor, y * factor, z * factor])
        if not rows:
            raise ValueError("charge file '%s' has no charge rows" % path)
        return 'charge', rows

    if head == 'field':
        if len(tokens) != 2 or tokens[1].lower() not in ('au', 'a.u.'):
            raise ValueError(
                "charge file '%s' must declare a field as 'field au'; got %r"
                % (path, lines[0]))
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    "charge file '%s' row must be 'V Fx Fy Fz': %r"
                    % (path, line))
            rows.append([float(item) for item in parts])
        if not rows:
            raise ValueError("charge file '%s' has no field rows" % path)
        if natom and len(rows) != natom:
            raise ValueError(
                "charge file '%s' has %d field rows but the system has %d "
                "atoms" % (path, len(rows), natom))
        return 'field', rows

    return _legacy_charge_mode(path)


def _legacy_charge_mode(path):
    """Read the two pre-declaration layouts, both point charges."""
    first = None
    for line in _data_lines(path):
        first = line
        break
    tokens = first.split() if first else []
    if len(tokens) == 1 and tokens[0].lstrip('-+').isdigit():
        return 'charge', load_charge_file(path)
    lines = list(_data_lines(path))
    header = lines[0].lower().split()
    if len(header) >= 2 and header[0] == 'potential':
        if header[1] not in ('au', 'a.u.'):
            raise ValueError(
                "charge file '%s' must declare a potential as "
                "'potential au'; got %r" % (path, lines[0]))
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    "charge file '%s' row must be 'V Fx Fy Fz': %r"
                    % (path, line))
            rows.append([float(item) for item in parts])
        if not rows:
            raise ValueError("charge file '%s' has no field rows" % path)
        return 'field', rows
    if len(header) >= 2 and header[0] == 'charge':
        unit = header[1]
        if unit not in ('angstrom', 'bohr'):
            raise ValueError(
                "charge file '%s' has unsupported length unit %r"
                % (path, unit))
        factor = 1.0 if unit == 'angstrom' else BOHR_TO_ANGSTROM
        rows = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    "charge file '%s' row must be 'x y z q': %r"
                    % (path, line))
            x, y, z, q = (float(item) for item in parts)
            rows.append([q, x * factor, y * factor, z * factor])
        if not rows:
            raise ValueError("charge file '%s' has no charge rows" % path)
        return 'charge', rows
    raise ValueError(
        "charge file '%s' must declare 'point <unit> <role>' or 'field au' "
        "(or use one of the two older layouts); got %r" % (path, first))


def embedding_requested(config):
    """Whether this run will carry an ESPF block at all.

    The workflow compiler has to know before any file is opened, and the answer
    is not the `embedding` keyword alone: a named file embeds on its own - its
    first line says whether it holds charges or a field - and a single-theory
    run is embedded exactly when that file is named. Asking here, once, keeps
    the compiler, the resolver and the validation from each spelling the rule
    differently.
    """
    read_charge = str(getattr(config.system, 'read_charge', 'none')
                      or 'none').strip()
    embedding = str(getattr(config.system, 'embedding', 'none')
                    or 'none').strip().lower()
    return read_charge != 'none' or embedding in ('charge', 'field')


def embedding_mode_label(config):
    """What this run will emit, as far as the config alone can say.

    A named file declares its representation inside the file, which the
    compiler does not open, so the label is 'file' there; the array path names
    it in `embedding` and the resolver follows that.
    """
    read_charge = str(getattr(config.system, 'read_charge', 'none')
                      or 'none').strip()
    if read_charge != 'none':
        return 'file'
    return str(getattr(config.system, 'embedding', 'none')
               or 'none').strip().lower()


def resolve_embedding_field(config, calcdir=None):
    """Resolve the ESPF field for this calculation.

    A named file decides both the numbers and the representation: its first
    line says ``point`` or ``field``. The config's ``embedding`` is consulted
    only when there is no file, which is the array path - the caller hands
    point charges or a V/E block at each step. In a single-theory run the
    framework ignores ``embedding`` entirely and the file *is* the switch, so
    a file is honoured even when ``embedding`` says none.
    """
    embedding = str(getattr(config.system, 'embedding', 'none')
                    or 'none').lower()
    read_charge = str(getattr(config.system, 'read_charge', 'none')
                      or 'none')
    natom = len(getattr(config.system, 'atoms', []) or [])

    if read_charge != 'none':
        path = _readable(read_charge, calcdir)
        if path is None:
            raise ValueError(
                "read_charge file '%s' was not found" % read_charge)
        mode, rows = load_embedding_file(path, natom)
        return {'mode': mode, 'rows': rows}

    if embedding == 'none':
        return None
    if embedding == 'potential':
        raise ValueError(
            "embedding='potential' is retired: write 'field' (a V/E block "
            "per QM atom). A charge file declares the representation itself.")
    if embedding not in ('charge', 'field'):
        raise ValueError(
            "Unsupported embedding %r; expected none, charge, or field."
            % (embedding,))

    if embedding == 'charge':
        # The emptiness test cannot ask for the truth value of the rows:
        # ONIOM electrostatic embedding hands over an (n, 4) array, and
        # `charges or []` on an array raises instead of counting it.
        charges = getattr(config.system, 'point_charges', None)
        rows = [] if charges is None else [list(row) for row in charges]
        if not rows:
            raise ValueError(
                "embedding='charge' requires point charges: name a charge "
                "file in read_charge or provide system.point_charges each "
                "step.")
        for row in rows:
            if len(row) != 4:
                raise ValueError(
                    "point charge rows must be [q, x, y, z]; got %r" % (row,))
        return {'mode': 'charge', 'rows': rows}

    rows = [list(row) for row in
            (getattr(config.system, 'external_potential', []) or [])]
    if len(rows) != natom:
        raise ValueError(
            "embedding='field' requires external_potential with %d rows; "
            "got %d" % (natom, len(rows)))
    for row in rows:
        if len(row) != 4:
            raise ValueError(
                "field rows must be [V, Fx, Fy, Fz]; got %r" % (row,))
    return {'mode': 'field', 'rows': rows}


def build_esfp_block(field):
    """Build the &ESPF input text for a resolved field.

    Both representations end with the pinned PNT grid step: OpenMolcas
    stores DELTAR with 9 decimals and a later module rebuilds the grid
    from it, so a value with more decimals changes the bucket count
    between the two calls and MkGrid aborts.
    """
    if field is None:
        return ''
    from ..templates.molcas_templates import ESPF_PNT_GRID
    if field['mode'] == 'charge':
        lines = []
        for q, x, y, z in field['rows']:
            lines.append(
                " %16.10f %16.10f %16.10f %10.6f 0.0 0.0 0.0"
                % (x, y, z, q))
        from ..templates.molcas_templates import ESPF_CHARGE_TEMPLATE
        return ESPF_CHARGE_TEMPLATE.format(
            ncharges=len(lines), charge_block='\n'.join(lines),
            grid_block=ESPF_PNT_GRID)
    if field['mode'] == 'field':
        lines = []
        for i, (v, fx, fy, fz) in enumerate(field['rows'], start=1):
            lines.append(
                " %4d %16.10f %16.10f %16.10f %16.10f "
                "0.0 0.0 0.0 0.0 0.0 0.0"
                % (i, v, fx, fy, fz))
        from ..templates.molcas_templates import ESPF_POTENTIAL_TEMPLATE
        return ESPF_POTENTIAL_TEMPLATE.format(
            potential_block='\n'.join(lines), grid_block=ESPF_PNT_GRID)
    raise ValueError("Unknown embedding mode %r" % (field.get('mode'),))
