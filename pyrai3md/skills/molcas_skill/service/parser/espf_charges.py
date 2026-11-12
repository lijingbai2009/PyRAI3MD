"""
======================================================================
ESPF 0-order multipole charges.

The ESPF module fits atomic charges to build the embedding potential and
prints them in its `++ ESPF analysis` block; this reader turns those into
the framework's ``computed_charges`` channel (the post-ESPF 0-order
charges, in highlevel order), so QMMM can use them as the classical seed
and the run log can print the values actually used. A sidecar named
``ESPF.DATA`` is preferred when the run produced one, because it can carry
more digits than the log's four decimals.

The accepted sidecar layout is deliberately narrow:

    <optional header lines with no numbers>
    <charge>                 one row per atom, OR
    <index> <charge>         one-based index then charge

Blank lines and lines starting with '#' or '!' are ignored. Any other
shape, or a row count that does not equal the atom count, makes the
reader return None: emitting a guessed charge would be worse than
emitting none (dev/methods/QMMM_DEVELOPMENT.md and the log contract).

The log layout was confirmed on the deployed build (round
`tmp_20261031_183037`): three-atom and six-atom QM regions both print
`Charge on <element><one-based index> = <value>` in atom order, and a
two-pass job prints the block twice with the same numbers. Until that
reading, this file only knew the sidecar, which no deployed run writes -
so every QMMM case silently kept whatever `molecule.q_qm` it pinned.

Oct 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import re

ESPF_CHARGE_FILE = 'ESPF.DATA'

# One line of the ESPF analysis block, as the deployed build writes it:
#
#       Charge on O1          =    -0.7164
#
# The label is element plus the one-based atomic index, so the block order
# is the atom order the framework's `highlevel` slice expects.
_CHARGE_LINE = re.compile(r'^\s*Charge on \S+\s*=\s*(-?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$')


def _from_sidecar(calcdir, natom):
    """Charges from ``ESPF.DATA``, the narrow one-number-per-row layout."""
    path = os.path.join(calcdir, ESPF_CHARGE_FILE)
    if not os.path.isfile(path):
        return None
    values = []
    try:
        with open(path) as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line[0] in '#!':
                    continue
                fields = line.replace(',', ' ').split()
                try:
                    numbers = [float(field) for field in fields]
                except ValueError:
                    # Header / free-text line: ignore it.
                    continue
                if len(numbers) == 1:
                    values.append(numbers[0])
                elif len(numbers) == 2 and float(numbers[0]).is_integer():
                    values.append(numbers[1])
                else:
                    return None
    except OSError:
        return None
    if len(values) != natom:
        return None
    return values


def _from_log(calcdir, natom):
    """Every state's charges the log prints, in root order.

    This is what the deployed OpenMolcas actually produces: the fitted
    0-order multipoles are printed, not written to a sidecar, so a reader
    that asked only for ``ESPF.DATA`` returned None for every real run and
    the QMMM path fell back to whatever charge the input happened to pin.

    A run prints several of these blocks, and the count is not a constant:
    measured on round ``tmp_20261031_195117``, a two-root XMS-CASPT2 job
    printed twelve, a single-root one printed one, and a three-pass
    RASSCF job printed three. Nothing inside a block names its state, so
    position in the file says nothing about which state it is - reading the
    last block handed the classical stage a CASPT2-response vector for a
    root that was not the one being computed, whose charges summed to
    0.0276 instead of the active state's 0.0012.

    The state is therefore recovered from the module that is running: the
    `--- Start Module: <name> ---` banner is tracked, and within a RASSCF
    section the k-th block is the k-th root - the same order `&RASSCF`
    reports them in, and the only section whose blocks are per-state rather
    than per-pass. Sections are read in file order, so a run that declares
    two singlets and one triplet yields exactly three rows in that order,
    which is the global 0-based state numbering the framework uses. Blocks
    printed by `&CASPT2` are response/multi-state quantities without a state
    label to bind to, so they are not used.

    The reader returns every row it can bind, in that order, and chooses
    nothing: which state embeds is the framework's question, and the rows it
    got are what it has to answer from. The count is not capped by the
    declared `nstate` either - the run's roots come from the input template
    this skill was handed, which can name more of them than the framework's
    state field says, and dropping the extra rows would turn a row the
    framework could address into a missing one.
    """
    rows = []
    try:
        names = sorted(n for n in os.listdir(calcdir)
                       if n.lower().endswith('.log'))
    except OSError:
        return None
    for name in names:
        try:
            with open(os.path.join(calcdir, name)) as handle:
                module = ''
                run = []
                for raw in handle:
                    head = raw.strip()
                    banner = head.startswith('--- Start Module:')
                    match = None if banner else _CHARGE_LINE.match(raw)
                    if match is not None:
                        run.append(float(match.group(1)))
                        continue
                    # A block is closed by any non-blank line that is not one
                    # of its own rows - including a module banner, since the
                    # last block of a section ends when the next section
                    # starts. Blank lines stay inside the block.
                    if head or banner:
                        if len(run) == natom and module == 'rasscf':
                            rows.append(run)
                        run = []
                    if banner:
                        fields = head.split()
                        module = fields[3].lower() if len(fields) > 3 else ''
        except (OSError, ValueError):
            continue
        if len(run) == natom and module == 'rasscf':
            rows.append(run)
    return rows or None


def parse_espf_charges(calcdir, natom):
    """Read the ESPF 0-order charges, one row per state.

    Parameters
    ----------
    calcdir : str
        Molcas calculation directory.
    natom : int
        Number of QM atoms each row must describe.

    Returns
    -------
    list of list of float or None
        ``(nstate_read, natom)`` charges in root order, or None when no
        source in this directory describes exactly `natom` charges. The log
        prints four decimals, so a reader that needs more asks Molcas for
        the sidecar rather than guessing at the digits.
    """
    if natom <= 0:
        return None
    sidecar = _from_sidecar(calcdir, natom)
    if sidecar:
        # The sidecar carries no state either; it only exists for runs that
        # ask for one, which is the single-state case by construction.
        return [sidecar]
    return _from_log(calcdir, natom)
