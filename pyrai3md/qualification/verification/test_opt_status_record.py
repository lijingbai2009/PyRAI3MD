"""
======================================================================
Test suite for the `opt_status.txt` format and record: the header/row
shape in `pyrai3md/utilities/opt_formatter.py` and the reader, record
type and write guard in `pyrai3md/utilities/print_results.py`.

`opt_status.txt` has two writers and one plain name. The framework writes a
seven-column table for a fixed-cell run and a twelve-column one when
`optimize_cell` makes the lattice a variable; an engine writes its own
convergence record under the same name inside a calculation directory it
keeps (openqp does, an eleven-token title-case header over seven-token rows
and a dashes separator). Before this module the header was formatted in one
place and the rows in another, and the only reader in the qualification
layer dropped any row whose width differed from its header without saying
so. Both halves of that are pinned here: the header and a row of the same
table are rendered from one set of constants and reproduce the bytes a real
round wrote, and a line that does not match its header is reported rather
than lost.

The fixtures are copied verbatim from measured rounds: the framework rows
from `tmp_20261016_165643` (the eighth round's LAMMPS variable-cell leg and
its `optimize_minima` fixed-cell leg) and the engine table from the sixth
round's openqp HF relaxation. No calculation is executed here.

Oct 16 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest

from pyrai3md.utilities.opt_formatter import (
    STATUS_ATOM_COLUMNS,
    STATUS_CELL_COLUMNS,
    status_header,
    status_line,
)
from pyrai3md.utilities.print_results import (
    WRITER_ATOM,
    WRITER_CELL,
    WRITER_FOREIGN,
    assert_writable,
    classify_header,
    format_malformed,
    read_status,
)

# Verbatim from tmp_20261016_165643/examples/optimize_cell_lammps/yaml.
CELL_HEADER = (
    '  step                  energy                   shift'
    '                   max_d                   rms_d                   max_g'
    '                   rms_g               cell_grms               cell_gmax'
    '               cell_drms               cell_dmax                  volume')
CELL_ROW = (
    '     1  -8.265022160551094e+00   0.000000000000000e+00'
    '   1.501118817262213e-02   3.835617010285512e-03'
    '   5.003729390874045e-02   1.278539003428504e-02'
    '   1.248755444438083e-01   1.904484168349469e-01'
    '   3.746266333314250e-02   5.713452505048406e-02'
    '   1.572157528502303e+04')
# The eleven values after the step column of that row, as the writer has them.
CELL_VALUES = (-8.265022160551094, 0.0, 1.501118817262213e-02,
               3.835617010285512e-03, 5.003729390874045e-02,
               1.278539003428504e-02, 1.248755444438083e-01,
               1.904484168349469e-01, 3.746266333314250e-02,
               5.713452505048406e-02, 1.572157528502303e+04)

# Verbatim from tmp_20261016_165643/examples/optimize_minima/yaml.
ATOM_HEADER = (
    '  step                  energy                   shift'
    '                   max_d                   rms_d                   max_g'
    '                   rms_g')
ATOM_ROW = (
    '     1  -7.701349942083050e+01   0.000000000000000e+00'
    '   3.472983293669624e-02   1.107900994157882e-02'
    '   1.157661097889875e-01   3.693003313859609e-02')
ATOM_VALUES = (-7.701349942083050e+01, 0.0, 3.472983293669624e-02,
               1.107900994157882e-02, 1.157661097889875e-01,
               3.693003313859609e-02)

# Verbatim from the sixth round's openqp HF geometry relaxation, which is an
# engine writing under the same plain file name.
ENGINE_HEADER = (' Step           Energy            Shift      RMSD Step'
                 '       Max Step      RMSD Grad       Max Grad')
ENGINE_SEPARATOR = '-' * 118
ENGINE_ROW = ('    1     -76.85494948     -76.85494948       0.000000'
              '       0.000000       0.042160       0.076263')


def _engine_record(rows=2):
    return '\n'.join([ENGINE_HEADER, ENGINE_SEPARATOR]
                     + [ENGINE_ROW] * rows) + '\n'


class RecordCase(unittest.TestCase):
    """Shared fixture: one scratch directory for records."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='opt_status_')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, text, name='opt_status.txt'):
        path = os.path.join(self.root, name)
        with open(path, 'w') as handler:
            handler.write(text)
        return path

    def _read(self, text, name='opt_status.txt'):
        return read_status(self._write(text, name))


class TestOneSourceForHeaderAndRows(RecordCase):
    """The header and a row of the same table come from one set of widths."""

    def test_the_fixed_cell_header_is_the_measured_bytes(self):
        self.assertEqual(status_header(), ATOM_HEADER)

    def test_the_variable_cell_header_is_the_measured_bytes(self):
        self.assertEqual(status_header(cell=True), CELL_HEADER)

    def test_a_measured_cell_row_is_reproduced_from_its_values(self):
        self.assertEqual(status_line(1, CELL_VALUES), CELL_ROW)

    def test_a_measured_fixed_cell_row_is_reproduced_from_its_values(self):
        self.assertEqual(status_line(1, ATOM_VALUES), ATOM_ROW)

    def test_a_row_neither_wide_nor_narrow_is_refused(self):
        # Eight values belongs to no table this module writes: it is the shape
        # a caller gets by appending one cell reading to the atomic six, and a
        # row that wide would be read against a header of a different length.
        for width in (0, 7, 11, 12):
            if width in (6, 11):
                continue
            with self.assertRaises(ValueError):
                status_line(1, tuple([0.0] * width))

    def test_write_then_read_returns_the_values_the_writer_handed(self):
        text = '%s\n%s\n' % (status_header(cell=True),
                             status_line(1, CELL_VALUES))
        record = self._read(text)
        self.assertEqual(record.writer, WRITER_CELL)
        self.assertEqual(record.columns,
                         list(STATUS_ATOM_COLUMNS) + list(STATUS_CELL_COLUMNS))
        row = record.rows[0]
        self.assertEqual(row['step'], 1)
        self.assertAlmostEqual(row['volume'], CELL_VALUES[-1])
        self.assertAlmostEqual(row['cell_gmax'], CELL_VALUES[7])
        self.assertEqual(record.malformed, [])
        self.assertEqual(record.separators, 0)


class TestWhichTableAFileIs(RecordCase):
    """An engine's record is named as one, and its rows are not taken as steps."""

    def test_an_engine_record_is_classified_by_its_header(self):
        record = self._read(_engine_record())
        self.assertEqual(record.writer, WRITER_FOREIGN)
        self.assertEqual(record.separators, 1)
        # Its rows are seven tokens against an eleven-token header: the reader
        # reports them instead of parsing them as a framework relaxation.
        self.assertEqual(record.rows, [])
        self.assertEqual(len(record.malformed), 2)

    def test_a_framework_record_is_not_mistaken_for_an_engine_one(self):
        self.assertEqual(self._read('%s\n%s\n' % (ATOM_HEADER,
                                                  ATOM_ROW)).writer,
                         WRITER_ATOM)
        self.assertEqual(self._read('%s\n%s\n' % (CELL_HEADER,
                                                  CELL_ROW)).writer,
                         WRITER_CELL)

    def test_a_header_with_no_steps_still_names_its_writer(self):
        # A run that opened its record and died before step one: zero steps,
        # not a file the framework never wrote.
        record = self._read(CELL_HEADER + '\n')
        self.assertEqual(record.writer, WRITER_CELL)
        self.assertEqual(record.rows, [])

    def test_a_truncated_row_is_counted_and_named(self):
        # The shape a killed run leaves: a final line cut mid-row. Dropping it
        # in silence would report a shorter relaxation than the run recorded.
        record = self._read('%s\n%s\n     2  -8.5e+00\n' % (CELL_HEADER,
                                                            CELL_ROW))
        self.assertEqual(len(record.rows), 1)
        self.assertEqual(record.malformed, [{'line': 3, 'tokens': 2}])
        words = format_malformed(record)
        self.assertIn('1 line(s)', words)
        self.assertIn('12-column header', words)
        self.assertIn('widths seen: 2', words)

    def test_a_matching_record_has_nothing_to_report(self):
        record = self._read('%s\n%s\n' % (CELL_HEADER, CELL_ROW))
        self.assertEqual(format_malformed(record), '')


class TestTheMethodColumnsArePartOfTheTable(RecordCase):
    """Optimizer-specific columns sit after the cell block, before the terms."""

    METHOD = ('cross_gap', 'coupling_norm')

    def test_method_columns_follow_the_cell_block_and_keep_the_prefix(self):
        columns = status_header(cell=True,
                                method=self.METHOD).split()
        self.assertEqual(columns[:len(STATUS_ATOM_COLUMNS)],
                         list(STATUS_ATOM_COLUMNS))
        start = len(STATUS_ATOM_COLUMNS)
        self.assertEqual(columns[start:start + len(STATUS_CELL_COLUMNS)],
                         list(STATUS_CELL_COLUMNS))
        start += len(STATUS_CELL_COLUMNS)
        self.assertEqual(columns[start:start + len(self.METHOD)],
                         list(self.METHOD))
        self.assertEqual(classify_header(columns), WRITER_CELL)
        self.assertEqual(
            classify_header(status_header(method=('x',)).split()),
            WRITER_ATOM)

    def test_a_row_carries_the_declared_method_columns(self):
        columns = status_header(method=self.METHOD).split()
        values = tuple([0.0] * 6) + (0.01, float('nan'))
        record = self._read('%s\n%s\n' % (
            ' '.join(columns),
            status_line(1, values, method=len(self.METHOD))))
        self.assertEqual(record.malformed, [])
        self.assertEqual(record.rows[0]['cross_gap'], 0.01)

    def test_a_row_missing_the_method_count_is_refused(self):
        # The negative control: the same six values are a valid fixed-cell
        # row with no method columns and an invalid row with one.
        status_line(1, tuple([0.0] * 6), method=0)
        with self.assertRaises(ValueError):
            status_line(1, tuple([0.0] * 6), method=1)

    def test_a_method_name_colliding_with_a_fixed_column_is_refused(self):
        with self.assertRaises(ValueError):
            status_header(method=('energy',))
        with self.assertRaises(ValueError):
            status_header(cell=True, method=('volume',))
        with self.assertRaises(ValueError):
            status_header(terms=('cross_gap',),
                          method=('cross_gap',))


class TestOneDirectoryOneOwner(RecordCase):
    """The writer refuses to truncate a record it does not own."""

    def test_writing_over_an_engine_record_is_refused_by_name(self):
        path = self._write(_engine_record())
        with self.assertRaises(ValueError) as caught:
            assert_writable(path)
        message = str(caught.exception)
        self.assertIn('openqp', message)
        self.assertIn('Step', message)
        self.assertTrue(os.path.exists(path),
                        'refusing to write must leave the engine record intact')

    def test_writing_over_the_frameworks_own_record_is_allowed(self):
        # A rerun truncates its own table: that is how a step record restarts.
        self._write('%s\n%s\n' % (CELL_HEADER, CELL_ROW))
        assert_writable(os.path.join(self.root, 'opt_status.txt'))

    def test_an_absent_record_is_writable(self):
        assert_writable(os.path.join(self.root, 'nobody_wrote_this.txt'))


if __name__ == '__main__':
    unittest.main()


class TestTheCompositeColumnsArePartOfTheTable(RecordCase):
    """A header that carries one column per energy term is still this table.

    The fixed columns are seven, or twelve with the cell block, and a composite
    run appends one more per term after them (`workflow/optimize.py` reads the
    names off the assembled result). The classification has to follow the prefix
    rather than the whole line, because the term names belong to the assembly and
    this module cannot list them - and a record this reader refused would take
    the variable-cell verdict with it, since that verdict reads its columns by
    name out of this same parse.
    """

    TERMS = ('high[qm]', 'low[qm]', 'low[mm]')

    def test_a_cell_header_plus_terms_round_trips_as_the_cell_table(self):
        columns = status_header(cell=True, terms=self.TERMS).split()
        self.assertEqual(self._read('%s\n%s\n' % (
            ' '.join(columns),
            status_line(1, tuple([0.0] * 11) + (-78.0, 77.0, -100.0),
                        extra=3))).writer, WRITER_CELL)
        record = self._read('%s\n%s\n' % (
            ' '.join(columns),
            status_line(1, tuple([0.0] * 11) + (-78.0, 77.0, -100.0),
                        extra=3)))
        self.assertEqual(record.malformed, [])
        self.assertEqual(record.rows[0]['low[mm]'], -100.0)
        self.assertEqual(record.rows[0]['volume'], 0.0)

    def test_the_terms_come_last_and_the_fixed_columns_keep_their_order(self):
        columns = status_header(cell=True, terms=self.TERMS).split()
        width = len(STATUS_CELL_COLUMNS)
        self.assertEqual(columns[:len(STATUS_ATOM_COLUMNS)],
                         list(STATUS_ATOM_COLUMNS))
        self.assertEqual(columns[len(STATUS_ATOM_COLUMNS):
                                 len(STATUS_ATOM_COLUMNS) + width],
                         list(STATUS_CELL_COLUMNS))
        self.assertEqual(columns[-len(self.TERMS):], list(self.TERMS))

    def test_a_label_with_a_space_in_it_is_refused(self):
        # One label, two tokens: everything written after it would be read
        # against the wrong column name, which is the class of defect this
        # module exists to make impossible.
        with self.assertRaises(ValueError):
            status_header(terms=['high level'])

    def test_a_repeated_label_is_refused(self):
        # Rows are read into a mapping keyed by column name, so a second column
        # with the first one's name is a term that would vanish on the way back.
        with self.assertRaises(ValueError):
            status_header(terms=['high[qm]', 'high[qm]'])

    def test_a_row_is_still_refused_when_it_is_the_wrong_width_for_its_header(self):
        for width in (6, 10, 13):
            with self.assertRaises(ValueError):
                status_line(1, tuple([0.0] * width), extra=3)
        # And the two widths that ARE right for a three-term header:
        status_line(1, tuple([0.0] * 9), extra=3)
        status_line(1, tuple([0.0] * 14), extra=3)
