"""
======================================================================
This module is the header and row format of `opt_status.txt`, the
per-step convergence record an optimization leaves in its run directory.

The artifact name does not change. This module owns the columns, their
order and their widths: the fixed atomic block, the optional cell block,
the optimizer-specific method columns, and the composite term columns.
`print_results.py` owns the file's lifecycle (the immutable record, the
reader, the writer classification and the refusal to truncate another
writer's file) and imports the column names from here. Keeping the write
format import-free is what lets both sides share it without a cycle.

Pure formatting only: no file is opened and no run state is stored.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

# The column names, in the order they are written. The step column is an
# integer and the rest are floats in the framework's own units (Hartree for
# energy, Angstrom for displacements, Hartree/Bohr for gradients).
STATUS_ATOM_COLUMNS = ('step', 'energy', 'shift', 'max_d', 'rms_d',
                       'max_g', 'rms_g')
STATUS_CELL_COLUMNS = ('cell_grms', 'cell_gmax', 'cell_drms', 'cell_dmax',
                       'volume')

# The one column width the header and the rows share. A table whose header
# and body are formatted apart is how a reader starts counting columns wrong.
STATUS_STEP_WIDTH = 6
STATUS_FIELD_WIDTH = 22


def _column_token(name):
    """Validate one whitespace-free column name."""
    token = str(name)
    if not token or token.split() != [token]:
        raise ValueError(
            'opt_status.txt columns are whitespace-separated tokens, so a '
            'column name must be one non-empty word; got %r' % (name,))
    return token


def status_header(cell=False, terms=(), method=()):
    """The column header line an optimization run writes.

    Parameters
    ----------
    cell : bool, optional
        Append the cell columns, for a run whose lattice is a variable.
    terms : sequence of str, optional
        One column per energy term of a composite assembly, appended after the
        fixed ones. The names are not this module's invention: they come from
        `workflow/multiscale/composite.py::term_label`, through the writer that
        reads them off the assembled result (`workflow/optimize.py`). Each has to
        be one whitespace-free token, because the record is split on whitespace
        and a label with a space in it would turn one column into two and shift
        everything written after it.
    method : sequence of str, optional
        One column per optimizer-specific reading (TS negative modes, MECI
        coupling norm, ...), appended after the cell columns and before the
        composite terms. Keeping them after the fixed prefix is what lets the
        reader in `print_results.py` and every 7/12-column reader keep treating
        the table as the same writer's record.

    Returns
    -------
    str
        The header line, without a trailing newline.
    """
    columns = list(STATUS_ATOM_COLUMNS)
    if cell:
        columns += list(STATUS_CELL_COLUMNS)
    method_names = []
    for name in method:
        token = _column_token(name)
        if token in columns or token in method_names:
            raise ValueError(
                'opt_status.txt reads its rows into a mapping keyed by column '
                'name, so a repeated column (%r) would silently drop one '
                'reading' % token)
        method_names.append(token)
    columns += method_names
    names = []
    for name in terms:
        token = _column_token(name)
        if token in columns or token in names:
            raise ValueError(
                'opt_status.txt reads its rows into a mapping keyed by column '
                'name, so a repeated column (%r) would silently drop one term'
                % token)
        names.append(token)
    columns += names
    line = '{:>{w}s}'.format(columns[0], w=STATUS_STEP_WIDTH)
    for name in columns[1:]:
        line += '  {:>{w}s}'.format(name, w=STATUS_FIELD_WIDTH)
    return line


def status_line(step, values, extra=0, method=0):
    """One row of the record, formatted with the widths of `status_header`.

    Parameters
    ----------
    step : int
        The 1-based step number.
    values : sequence of float
        The row after the step column, in `status_header`'s order. Its
        length decides whether this is a fixed-cell or a variable-cell row,
        so a writer cannot append cell readings to a seven-column header.
    extra : int, optional
        The number of composite term columns `status_header` was handed for this
        run.
    method : int, optional
        The number of optimizer-specific columns `status_header` was handed for
        this run. One caller supplies the header and the row, which is what
        keeps a row from being written against a header of a different width.

    Returns
    -------
    str
        The row, without a trailing newline.
    """
    expected = (len(STATUS_ATOM_COLUMNS) - 1
                + int(method) + int(extra))
    if len(values) != expected and len(values) != expected + len(
            STATUS_CELL_COLUMNS):
        raise ValueError(
            'opt_status.txt takes %d or %d values after the step column '
            '(%d method column(s), %d composite term column(s) declared), '
            'got %d (%r)'
            % (expected, expected + len(STATUS_CELL_COLUMNS), int(method),
               int(extra), len(values), values))
    line = '{:{w}d}'.format(step, w=STATUS_STEP_WIDTH)
    for value in values:
        line += '  {:>{w}.{p}e}'.format(float(value), w=STATUS_FIELD_WIDTH,
                                       p=15)
    return line


__all__ = [
    'STATUS_ATOM_COLUMNS',
    'STATUS_CELL_COLUMNS',
    'STATUS_STEP_WIDTH',
    'STATUS_FIELD_WIDTH',
    'status_header',
    'status_line',
]