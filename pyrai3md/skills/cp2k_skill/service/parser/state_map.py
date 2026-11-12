"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 14 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module is the one place where the framework's state rows and CP2K's
excited-state numbers are allowed to meet, so the conventions are changed in
one location instead of in every caller.

CP2K prints one excitation table per spin manifold when `RKS_TRIPLETS` is on
and restarts the state numbering inside each table: measured on
`cp2k_demo/sp/bulk.log`, whose tables carry `multiplicity 1` then
`multiplicity 3`, both numbered 1..10. The framework keeps a single `(nstate,)`
axis ordered by the sorted `state_spin` - row 0 the ground state, then the
excited singlets, then the triplets - so a row is addressed by the PAIR
(manifold, number inside it). A number alone is ambiguous between two surfaces
that share it, which is the whole reason the reader used to hand the triplet
table to the singlet rows and never notice: the counts agreed.

Two conventions are kept apart rather than chosen:

- the PRINTED tables restart per manifold. That is measured, not assumed, and
  `resolve_log_row` reads only this.
- the `&PRINT &FORCES LIST` the skill WRITES is settled by an engine abort, not by
  a local artifact: the continuing order was tried first (`LIST 1 3`, for the first
  singlet root and the first triplet root) and CP2K stopped mid-print with
  `State List contains invalid state` / `TDDFPT Print Forces: Invalid State`
  (qs_tddfpt2_fprint.F:395-396, round tmp_20261014_142736). The accepted numbers
  therefore restart per manifold, like the tables, and a request spanning two
  manifolds has no number the keyword can use - `require_addressable` refuses it at
  the input rather than sending an input known to abort. Whether a triplet root can
  be given forces at all in this build is still open.

`attribute_blocks` keeps both readings when interpreting the headers the engine does
print, and holds to the one the printed energies agree with: an energy match is what
tells the first singlet root from the first triplet root when both are numbered 1.

No I/O and no CP2K dependency live here, so every rule below is unit-testable
on its own (`tests/test_state_map.py`).

"""

# The two readings of a printed force-block number. MANIFOLD restarts inside
# each spin table (what the excitation tables do, measured); GLOBAL continues
# the singlet count into the triplet table.
MANIFOLD = 'manifold'
GLOBAL = 'global'

# The order written into `&PRINT &FORCES LIST`. The continuing count was tried on
# the deployment first and is falsified: with `NSTATES 2` and `RKS_TRIPLETS T`,
# `LIST 1 3` made CP2K abort with the warning `State List contains invalid state`
# and the message `TDDFPT Print Forces: Invalid State` (round
# tmp_20261014_142736, cp2k_skill/tmp/tmp_CP2K/tddft_st_forces_test.log:674), so
# the accepted numbers run 1..NSTATES and restart per manifold, which is the same
# counting the printed tables use. What remains unknown is whether a triplet root
# can be asked for by force at all in this build, which is why
# `require_addressable` refuses a two-manifold request rather than guess.
LIST_ORDER = MANIFOLD

# Energy agreement between an excitation table row and the matching force-block
# header, in eV. Both print the same value to five decimals
# (cp2k_demo/td_grad/equil-td_forces-1_0.tdfrc:1 against azo-1.log:356), so the
# tolerance only has to survive that rounding, not any physics.
ENERGY_TOLERANCE = 1.0e-3


def manifold_of_spin(two_s: int) -> int:
    """CP2K's multiplicity (2S+1) for the framework's 2S spelling."""
    return int(two_s) + 1


def spin_of_manifold(multiplicity: int) -> int:
    """The framework's 2S for CP2K's multiplicity (2S+1)."""
    return int(multiplicity) - 1


def excited_rows(config) -> dict:
    """{2S: [framework rows ascending]} holding excited states of that spin.

    Row 0 is the ground state whatever its spin and never appears here, because
    CP2K gives it no excited-state number: its energy is the `Total energy:`
    line and its gradient the `force` file, not the TDDFPT table.
    """
    spins = [int(value) for value in (getattr(config, 'state_spin', None)
                                      or [])]
    groups = {}
    for row, two_s in enumerate(spins):
        if row == 0:
            continue
        groups.setdefault(two_s, []).append(row)
    return groups


def job_manifold(config):
    """The one 2S this run solves, or None when the counts do not say.

    This reads `nsing_root`/`ntrip_root`, the two counts the skill already
    carries and that `service/jobs.py` zeroes for the manifold a job does NOT
    solve. It is what keeps the split inside existing fields instead of a new
    keyword: a run with exactly one non-zero count addresses one manifold, so the
    numbers written into `&PRINT &FORCES LIST` and the rows the parse is held to
    answer both follow it. None means "the counts name nothing" - no excited
    state at all, or a request that was never split - and then every declared
    manifold is in scope, which is the behaviour every shipped case has.
    """
    counts = [(0, int(getattr(config, 'nsing_root', 0) or 0)),
              (2, int(getattr(config, 'ntrip_root', 0) or 0))]
    solved = [two_s for two_s, count in counts if count > 0]
    return solved[0] if len(solved) == 1 else None


def spin_of_row(config, row: int) -> int:
    """The 2S of one framework row, from the normalized `state_spin`."""
    spins = [int(value) for value in (getattr(config, 'state_spin', None)
                                      or [])]
    if not 0 <= int(row) < len(spins):
        raise ValueError(
            'state row %r is outside the %d rows this run declares'
            % (row, len(spins)))
    return spins[int(row)]


def requested_rows(config) -> list:
    """The excited rows this run asked to have a gradient printed for.

    This is the set the `&PRINT &FORCES LIST` is built from and the set the
    force blocks are answerable for, so the two cannot drift: a block for a root
    nobody asked for is refused, and a root that was asked for and never answered
    is refused, rather than left as a zero row with the right shape. It is read
    from `target_states` rather than from every declared state because that is
    the keyword an input already uses to say which surfaces it wants forces on -
    a run that declares three states and follows one keeps printing one gradient.

    A run that solves one manifold only answers for that manifold's rows, so the
    set is restricted to `job_manifold` when the counts name one; that is what
    lets the same rule hold per job after `service/jobs.py` splits a two-manifold
    request, without the split introducing a second bookkeeping list.
    """
    spins = [int(value) for value in (getattr(config, 'state_spin', None)
                                      or [])]
    groups = excited_rows(config)
    manifold = job_manifold(config)
    rows = set()
    for value in (getattr(config, 'target_states', None) or []):
        row = int(value)
        if 0 < row < len(spins) and row in groups.get(spins[row], []):
            if manifold is not None and spins[row] != manifold:
                continue
            rows.add(row)
    return sorted(rows)


def require_addressable(config) -> None:
    """Refuse a force request one run's LIST cannot express.

    One manifold's roots are addressable by number; two are not, because the
    numbering restarts and `&FORCES` takes no spin argument - and the continuing
    spelling of that number was measured and rejected by the engine
    (`LIST_ORDER`'s note). `service/jobs.py` answers that by giving each manifold
    its own run, so after the split this is an invariant: a config that still
    reaches this point with two manifolds in one LIST was not planned through
    `plan_jobs`, and it is refused here rather than sent to an engine that aborts
    mid-print.
    """
    groups = excited_rows(config)
    wanted = set(requested_rows(config))
    spins = sorted(two_s for two_s, rows in groups.items()
                   if wanted.intersection(rows))
    if len(spins) > 1:
        raise ValueError(
            'cp2k: &PRINT &FORCES LIST was asked for excited states of %d spin '
            'manifolds (2S %r) in ONE run, but LIST numbers roots inside one '
            'manifold and takes no spin argument, so a second manifold has no '
            'number the keyword can use. Inputs reach this point through '
            'service/jobs.py, which gives every manifold its own run; build the '
            'jobs, do not hand a split request to the builder'
            % (len(spins), spins))


def _shift_before(config, two_s: int, order: str) -> int:
    """How many printed numbers the manifolds before this one occupy."""
    if order != GLOBAL:
        return 0
    nstates = int(getattr(config, 'tddft_nstates', 0) or 0)
    return sum(nstates for other in sorted(excited_rows(config))
               if other < two_s)


def state_number(config, row: int, order: str = LIST_ORDER) -> int:
    """The number that addresses one framework row under a given order."""
    two_s = spin_of_row(config, row)
    rows = excited_rows(config).get(two_s, [])
    if row not in rows:
        raise ValueError('state row %r is the ground state and has no '
                         'excited-state number' % row)
    if order not in (MANIFOLD, GLOBAL):
        raise ValueError('unknown state numbering %r' % order)
    return (rows.index(row) + 1
            + _shift_before(config, two_s, order))


def list_numbers(config) -> list:
    """The integers to write into `&PRINT &FORCES LIST`, one per requested row."""
    return [state_number(config, row, LIST_ORDER)
            for row in requested_rows(config)]


def resolve_log_row(config, multiplicity: int, state: int):
    """The framework row an excitation-table row names, or None.

    The tables restart inside each manifold (measured), so this is the one
    resolution that needs no hypothesis. None means "a root CP2K computed that
    this run did not ask for", which is normal: `NSTATES` must cover the largest
    manifold, so a run wanting one triplet and three singlets is handed ten of
    each.
    """
    rows = excited_rows(config).get(spin_of_manifold(multiplicity))
    if not rows:
        return None
    if not 1 <= int(state) <= len(rows):
        return None
    return rows[int(state) - 1]


def attribute_blocks(config, rows: list, blocks: list,
                     tolerance: float = ENERGY_TOLERANCE):
    """Map each force block onto a framework row, and report which order did it.

    A tdfrc header prints a state number and an excitation energy but no spin
    (cp2k_demo/td_grad/equil-td_forces-1_0.tdfrc:1), so the number alone cannot
    tell a singlet root from a triplet root - both print 1. The energy is the
    other half of the key, and the excitation table read from the log carries
    the spin, so a block is attributed to the log row that agrees with it on
    BOTH. Whether the header's number continues across manifolds or restarts is
    read off that agreement instead of assumed: each order is tried, and the one
    under which every block finds exactly one agreeing row is the answer.

    Outcomes, distinguished on purpose:

    - one order explains every block: returned with `assignment` and the order;
    - both orders explain every block: the request cannot tell them apart (only
      a single manifold was asked for, where the two coincide), and the written
      `LIST_ORDER` is kept;
    - neither does: refused by name, with the first block that failed, because
      that is what a numbering the engine does not use looks like.

    Rows whose two candidate surfaces are degenerate cannot be separated by
    energy at all; those fall to the request order and come back in `by_order`
    so the caller reports the tie-break instead of hiding it.

    Parameters
    ----------
    config : SkillConfig
        Normalized configuration carrying `state_spin` and `tddft_nstates`.
    rows : list of dict
        Excitation table rows, each with `multiplicity`, `state`, `excitation`.
    blocks : list of dict
        Force blocks, each with `state`, `excitation` and `forces`.
    tolerance : float
        Energy agreement in eV.

    Returns
    -------
    tuple
        (assignment {row: block index}, by_order [rows settled only by order],
        order the header numbers turned out to use).
    """
    requested = requested_rows(config)
    winners = []
    failures = {}
    for order in (MANIFOLD, GLOBAL):
        outcome = _assign_under(config, rows, blocks, requested, order,
                                tolerance)
        if outcome is not None:
            winners.append((order, outcome))
        else:
            failures[order] = _first_unmatched(config, rows, blocks, requested,
                                               order, tolerance)

    if not winners:
        # A refusal about what the engine printed, so it is the same kind of
        # failure the rest of the parser raises: the run is over and its output
        # does not answer for the surfaces the input asked to follow.
        raise RuntimeError(
            'CP2K printed %d excited-state force blocks for the %d roots this '
            'run requested, and neither reading of the header numbering '
            'explains them. Restart-per-manifold: %s. Continue-across-manifolds: '
            '%s.' % (len(blocks), len(requested),
                     failures.get(MANIFOLD) or 'unknown',
                     failures.get(GLOBAL) or 'unknown'))

    if len(winners) == 1:
        order, (assignment, by_order) = winners[0]
        return assignment, by_order, order

    # Both orders fit: a single-manifold request, where they are the same list.
    chosen = dict(winners).get(LIST_ORDER, winners[0][1])
    return chosen[0], chosen[1], LIST_ORDER


def _assign_under(config, rows, blocks, requested, order, tolerance):
    """The assignment under one numbering, or None if it does not explain all."""
    assignment = {}
    by_order = []
    for index, block in enumerate(blocks):
        match = _match_row(config, rows, block, requested, order, tolerance,
                           assignment)
        if match is None:
            return None
        row, settled_by_order = match
        if row in assignment:
            # Two blocks naming one row: one of them is a surface no requested
            # row describes, and letting the later one win would drop a surface
            # while every count still added up.
            return None
        assignment[row] = index
        if settled_by_order:
            by_order.append(row)
    missing = [row for row in requested if row not in assignment]
    if missing:
        return None
    return assignment, by_order


def _candidates(config, rows, block, requested, order, tolerance):
    """Log rows a block could belong to: same number under this order, same energy."""
    energy = float(block.get('excitation', 0.0))
    found = []
    for row in requested:
        two_s = spin_of_row(config, row)
        number = state_number(config, row, order)
        if int(block.get('state', -1)) != int(number):
            continue
        for entry in rows:
            if int(entry['multiplicity']) != manifold_of_spin(two_s):
                continue
            if int(entry['state']) != state_number(config, row, MANIFOLD):
                continue
            if abs(float(entry['excitation']) - energy) > tolerance:
                continue
            found.append(row)
            break
    return sorted(set(found))


def _match_row(config, rows, block, requested, order, tolerance, assignment):
    """One block's row, falling back to request order when the pair is degenerate.

    Returns (row, settled_by_order) or None when the block answers for no
    requested row.
    """
    candidates = _candidates(config, rows, block, requested, order, tolerance)
    if len(candidates) == 1:
        return candidates[0], False
    if not candidates:
        # Two surfaces degenerate at this geometry print the same energy, so the
        # energies cannot separate them; the order the roots were requested in
        # still can, and the caller is told that is what happened.
        free = [row for row in _candidates_ignore_energy(config, rows, block,
                                                        requested, order)
                if row not in assignment]
        if not free:
            return None
        return free[0], True
    free = [row for row in candidates if row not in assignment]
    if free:
        # Two surfaces degenerate at this geometry agree on both the number and
        # the energy, so nothing in the engine's output separates them. The
        # request order is what settles it, and the row is returned flagged so
        # the caller reports the tie-break instead of letting it look labelled.
        return free[0], len(free) > 1
    return None


def _candidates_ignore_energy(config, rows, block, requested, order):
    """Rows whose printed number matches the block's, energy aside.

    Used only when the energies are degenerate, where they carry no
    information; the printed number is then the only handle left.
    """
    found = []
    for row in requested:
        two_s = spin_of_row(config, row)
        if int(block.get('state', -1)) != int(state_number(config, row, order)):
            continue
        number = state_number(config, row, MANIFOLD)
        manifold = manifold_of_spin(two_s)
        if any(int(entry['multiplicity']) == manifold
               and int(entry['state']) == number for entry in rows):
            found.append(row)
    return sorted(set(found))


def _first_unmatched(config, rows, blocks, requested, order, tolerance):
    """Describe the first block an order could not place, for the error text."""
    for index, block in enumerate(blocks):
        if not _candidates(config, rows, block, requested, order, tolerance):
            return ('block %d, state %r at %.5f eV, which matches no requested '
                    'root under the %r reading (state_spin %r, NSTATES %d)'
                    % (index + 1, block.get('state'),
                       float(block.get('excitation', 0.0)), order,
                       list(getattr(config, 'state_spin', None) or []),
                       int(getattr(config, 'tddft_nstates', 0) or 0)))
    return ('all %d blocks placed by the %r reading but some requested root was '
            'never answered' % (len(blocks), order))
