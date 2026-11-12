"""
======================================================================
This module is the single owner of the run-health checks that watch a
trajectory without changing it: who decides that a violated check stops
the run, where the finding is written, and how often it is repeated.

It exists because the variable-cell gate built in #250/#251 mixed two
different things in one raise. A cell whose determinant is zero or whose
extended coordinates went non-finite is not "a run that drifted too far";
it is a step that cannot be computed at all, and no setting can make
continuing it meaningful. A box that has left its starting size by 50% is
a different case: the trajectory is still well defined, and whether it
should be stopped is a QUESTION OF TESTING, not of arithmetic. The user
ruled that gating, monitoring and early interruption belong to the test
side of the package, so the second family now answers to `control.debug`
(default false: report the finding and keep integrating) while the first
keeps stopping the run for everyone -- but with this module's named
message instead of a pydantic traceback, which is what the 07:13 round
actually surfaced ("Value error, thermostat_dof contains a non-finite
value", naming the victim and not the cause).

Two rules the module keeps:

- It never changes the dynamics. Every finding is recorded on
  RuntimeState.diagnostics and written to the run log; the caller decides
  what to do with the verdict, because only the caller knows whether the
  state is still consistent to commit.
- Repetition is bounded by what the run already asked to see. A check
  speaks on the steps `md.direct`/`md.buffer` records a trajectory frame,
  because `workflow.records.output.record_due` is the one predicate that
  decides
  when this run wants anything said about a step, and `md.ensemble`
  carries the numbers a reader would want next to the sentence. The
  Nose-Hoover chain of the 07:13 run reached `inf` within ONE step of the
  last readable frame, so a monitor that emitted per step would have
  printed two thousand lines to say one thing while the trajectory itself
  kept twenty-one.

A second family watches a step that has already been recorded, rather than
the numbers a step is about to write: `check_recorded_step` asks a frame
whether its own kinetic energy and its own crossing counts still say what the
loop put into them, whether the region a center is cut out of those counts is
still one piece, and `workflow/integrator/operations.py::verify_recorded`
asks whether its geometry still satisfies the constraints it was projected
onto. That is the residue of the `verification` operation the integrator used
to reserve a slot for, and it lives here because a violated check has to reach
the same two outlets everything else in this module reaches.

Leaf module: imports numpy, the unit table, the periodic geometry helpers the
region watch measures with, and the output cadence predicate; it reads the
state through getattr, never through an integrator import, so the integrator
can import it without a cycle.

Sep 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import math

import numpy as np

from ...utilities.bonds import bond_lib
from ...utilities.bonds import BOND_SCALE
from ...utilities.lattice import (cart_to_frac, frac_to_cart, half_box_bound,
                                  min_image_displacement)
from ...utilities.units import AMU_TO_AU
from .output import record_due

# How much one step is allowed to do to a periodic box. The bounds are
# not the divergence threshold -- a run that drifts 20% over 2000 steps is
# a stability question for the qualification judgement, not for the
# integrator -- they are the point past which a SINGLE commit has done
# something a cell cannot survive. Measured against them: a healthy
# anisotropic NPT run turns its frame by 5.7e-5 rad per step and changes
# an axis by well under one percent per step, so 0.05 rad and a factor of
# 1.6 sit three orders of magnitude clear of anything real, and a run that
# trips them was already lost. These two are `fatal`: they are not a
# judgement about the ensemble, they are a commit no lattice can hold.
MAX_STEP_FRAME_ROTATION_RAD = 0.05
MAX_STEP_AXIS_RATIO = 1.6
# How far the cell may drift from the box the run STARTED with. The
# per-step bounds above cannot catch a gradual runaway: the anisotropic
# examples of tmp_20260928_223009 crept for hundreds of steps at ratios
# well inside any single-step bound, and the failure that finally stopped
# them came from the engine -- 'Triclinic box skew is too large
# (src/domain.cpp:222)' after the framework had sent it a box 20238 A
# along x. Read from barostat_strain (epsilon), not from the output
# history: the first cut of this bound compared against the oldest frame
# still in history, which the 07:13 run proved useless -- history_limit
# trims, so the reference sits a few steps back and a box creeping
# 5e57-fold never looked like 2x to it.
# The value is the gap in the 09:19 round's measurements. Over 2000 steps
# the healthiest runs that FINISH (the three berendsen examples) reach
# 1.03x on a cell axis and 1.19x in volume; every parrinello-rahman and
# mttk example is on its way to a non-finite cell and has passed 1.7x on
# an axis (nh_pr 0.568x, nh_mttk 0.488x, csvr_pr 5.66x) by the frame
# before it dies. ln(1.5) fires on all three at steps 374-855, hours of
# wall time before the chain overflows. Unlike the two bounds above this
# one is a judgement, so it interrupts only under `control.debug`.
MAX_CUMULATIVE_LOG_STRAIN = 0.4054651081081644
# The first line a non-debug run says out loud. The finished berendsen
# examples reach 0.174 in log volume and 0.03 on an axis, so 0.22 sits
# above every measured healthy value with a 1.3x margin and below the
# interrupt bound: a run that never crosses it is not worth a log line.
WARN_CUMULATIVE_LOG_STRAIN = 0.22
# Check identifiers. Stable strings: they are the keys of
# RuntimeState.diagnostics and the prefix of every emitted line, so a
# reader (or a later harvest into the qualification record) selects on
# them rather than on message text.
CELL_IMPOSSIBLE = 'cell_impossible'
CELL_GATE = 'cell_gate'
CELL_NOTE = 'cell_note'
ENSEMBLE_DOF_IMPOSSIBLE = 'ensemble_dof_impossible'
KINETIC_IDENTITY = 'kinetic_identity'
IMAGE_INTEGRITY = 'image_integrity'
CONSTRAINT_GEOMETRY = 'constraint_geometry'
REGION_RECONSTRUCTION = 'region_reconstruction'

# The recorded frame's kinetic energy is recomputed from the frame's own
# velocities, so the two numbers being compared are of the same magnitude and
# differ only by what the loop wrote between them. The bound below is four
# orders above double-precision roundoff on the energies a trajectory carries
# (Ekin ~ 1e-2 Hartree), which is what made it a check rather than a second
# print: the defect class it guards is a step that sampled the kinetic energy
# twice, before one action that rewrites the velocities and after another, and
# those two readings disagreed by 1e-3 and up (#310).
KINETIC_IDENTITY_RELATIVE = 1e-12


class CellDivergenceError(RuntimeError):
    """One cell commit deformed the box past what a lattice can carry.

    Raised by the variable-cell commit (_scale_cell) instead of letting
    the deformation continue into a NaN lattice several hundred steps
    later. It names the step, what the commit did to the box, and the
    barostat numbers of the step that did it, because the qualification
    run's anisotropic failures surfaced as a bare validator traceback on
    lattice_vector -- the victim -- and said nothing about the cell
    momentum that produced it.
    """


class EnsembleDivergenceError(CellDivergenceError):
    """An extended coordinate went non-finite, and for what it was worth.

    Subclassed from CellDivergenceError so every caller and test that has
    learned to catch the cell failure catches this too. It exists because
    the pydantic validator that actually stopped those runs reports the
    FIELD ('thermostat_dof contains a non-finite value') and knows nothing
    about the step, the temperature or the drive that produced it.
    """


class RecordedStepError(RuntimeError):
    """A frame the run recorded does not say what the loop believes it says.

    Raised only on the debug path, like every other judgement in this module:
    the finding itself goes to the runtime record and the run log whatever the
    mode says. It is a separate class from the cell failures because a bad
    frame is not an impossible step - the trajectory is well defined and the
    question is whether the run's own bookkeeping is consistent.
    """


def is_debug(state):
    """Whether this run was asked to interrupt on a threshold violation.

    Read through getattr so a bare state built by a unit test -- which has
    no control section at all -- means "not debugging" rather than an
    AttributeError on the way to the integrator.
    """
    control = getattr(getattr(state, 'pmd', None), 'control', None)
    return bool(getattr(control, 'debug', False))


def record(state, check_id, worst=None, aborted=False):
    """Note one finding and return its running entry.

    The entry is the audit trail of the monitor itself: how many times the
    check spoke, when it first and last fired, the worst value seen, and
    whether it ended the run. It rides on RuntimeState.diagnostics, which
    is serialized and copied like the per-step ensemble diagnostics, so a
    dispatched child reports what it saw instead of losing it.
    """
    runtime = getattr(state, 'runtime', None)
    entries = getattr(runtime, 'diagnostics', None) if runtime else None
    if not isinstance(entries, dict):
        return None
    step = int(getattr(runtime, 'current_step', 0) or 0)
    entry = entries.setdefault(check_id, {'count': 0, 'first_step': step,
                                          'last_step': step, 'worst': None,
                                          'aborted': False, 'said': False})
    if entry.get('last_step') != step:
        # A new step lets a repeating check speak again, on cadence; the
        # same step never says the same finding twice.
        entry['said'] = False
    entry['count'] += 1
    entry['last_step'] = step
    if worst is not None and (entry['worst'] is None
                              or abs(float(worst)) > abs(entry['worst'])):
        entry['worst'] = float(worst)
    if aborted:
        entry['aborted'] = True
    return entry


def _says_aloud(state, entry):
    """Whether this occurrence is a step the run records anyway.

    Cadence, not a second threshold: `md.direct` and `md.buffer` already
    decide which steps exist in the trajectory record, and a monitor that
    spoke off that schedule would produce a log whose line count has no
    relationship to the data it is commenting on.
    """
    if entry is None:
        return True
    dynamics = getattr(getattr(state, 'pmd', None), 'dynamics', None)
    step = int(getattr(getattr(state, 'runtime', None), 'current_step', 0)
               or 0)
    total = int(getattr(getattr(state, 'runtime', None), 'total_steps', 0)
                or 0)
    return record_due(step, int(getattr(dynamics, 'direct', 1) or 1),
                      int(getattr(dynamics, 'buffer', 1) or 1),
                      final=bool(total) and step >= total)


def _write(state, line):
    """Write one line to the run log, if this state has one."""
    logger = getattr(state, 'logger', None)
    write = getattr(logger, 'write', None)
    if callable(write):
        write(line)


def report(state, check_id, message, fatal=False, worst=None):
    """Record a violation and say whether the caller must stop.

    Parameters
    ----------
    state : PyRAIState
        State carrying pmd.control, runtime and (optionally) logger.
    check_id : str
        One of the module constants; keys the record and prefixes the line.
    message : str
        The full finding, with the numbers that justify it. Written to the
        run log and raised verbatim when the run stops, so the two paths
        cannot disagree about what was measured.
    fatal : bool
        True for a step that cannot be computed (a collapsed cell, a
        non-finite coordinate). Those stop the run whatever `debug` says;
        a threshold judgement stops it only when the run is being tested.
    worst : float, optional
        Magnitude of the violation, kept to report the worst of a
        repeating check.

    Returns
    -------
    bool
        True when the caller must abort. The caller raises: it is the only
        place that knows whether the state is still consistent to commit.
    """
    stop = bool(fatal) or is_debug(state)
    entry = record(state, check_id, worst=worst, aborted=stop)
    if stop:
        if entry is not None:
            entry['message'] = message
        return True
    if entry is None:
        # A state with no runtime to record on (a bare unit-test double, a
        # worker that never got one) still gets the sentence: silence would
        # read as "nothing was watched".
        _write(state, '[DIAG %s] %s' % (check_id, message))
        return False
    if _says_aloud(state, entry):
        _write(state, '[DIAG %s] %s' % (check_id, _with_since(entry, message)))
        entry['said'] = True
    return False


def _with_since(entry, message):
    """Add how long the check has been violated, when that is not obvious.

    Cadence means a violation can begin between two steps that are said
    aloud, so the sentence carries its own start step rather than making the
    reader reconstruct the interval from md.ensemble.
    """
    step = entry.get('last_step')
    first = entry.get('first_step')
    if first is None or step == first or entry.get('said'):
        return message
    return '%s (since step %d)' % (message, first)


def note(state, check_id, message, worst=None):
    """Say one thing once, and count it.

    The early bound is a remark, not a finding: a reader wants to know
    that the box left the thermal band at all, and does not want the same
    sentence 1700 times. The count still goes up every step, so the
    closing summary can state how long it lasted.
    """
    entry = record(state, check_id, worst=worst)
    if entry is not None and not entry.get('said') and _says_aloud(state,
                                                                    entry):
        _write(state, '[DIAG %s] %s' % (check_id, message))
        entry['said'] = True


def summary(state):
    """Close the run with one line per check that fired.

    Called after the trajectory loop: a warn-only run has to leave the
    same evidence a failing one would, otherwise "nothing was reported"
    and "nothing was watched" look identical in the log.
    """
    entries = getattr(getattr(state, 'runtime', None), 'diagnostics', None)
    if not entries:
        return
    for check_id in sorted(entries):
        entry = entries[check_id]
        worst = entry.get('worst')
        message = entry.get('message') if not entry.get('said') else None
        _write(state, '[DIAG %s] %d finding(s), first step %d, last step %d%s'
               % (check_id, entry['count'], entry['first_step'],
                  entry['last_step'],
                  '' if worst is None else ', worst %.4f' % worst))
        if message:
            # Nothing was on a record step while this was violated; say it
            # once now instead of leaving the run silent about its own end.
            _write(state, '[DIAG %s] %s' % (check_id, message))


def frame_rotation_angle(rotation):
    """Angle of an orthogonal matrix, arccos((tr - 1) / 2)."""
    trace = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(trace))


# The state fields that carry the extended coordinates of a variable-cell
# or thermostatted run. They are named here because they are what a
# diverging ensemble writes first: in tmp_20260929_071340 the run died
# with 'thermostat_dof contains a non-finite value', which is the schema
# validator noticing two steps after the physics stopped making sense.
EXTENDED_FIELDS = ('barostat_strain', 'barostat_momentum',
                   'barostat_velocity', 'thermostat_dof',
                   'barostat_thermostat_dof', 'lattice_vector', 'cell',
                   'volume')


def _finite_or_name(value):
    """Return (bad, worst magnitude) for one candidate field value."""
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None, None
    bad = ~np.isfinite(array)
    if not bad.any():
        return None, None
    finite = array[np.isfinite(array)]
    worst = float(np.max(np.abs(finite))) if finite.size else None
    return bad, worst


def check_finite(state, updates):
    """Guard a state write whose numbers may have stopped being numbers.

    Called by the integrator before committing extended coordinates, a
    lattice or a volume. A non-finite value is arithmetic, not a judgement,
    so it stops the run whatever `control.debug` says -- there is no
    trajectory to continue -- but the caller gets to say WHICH number, on
    which step, with which drive, which is more than the schema validator
    that would otherwise catch it.

    Parameters
    ----------
    state : PyRAIState
        State about to be written.
    updates : dict
        The pending `state.update_system` payload.

    Raises
    ------
    EnsembleDivergenceError
        When one of the extended fields went non-finite.
    """
    if not updates:
        return
    offenders = []
    for key in sorted(updates):
        if key not in EXTENDED_FIELDS:
            continue
        bad, worst = _finite_or_name(updates[key])
        if bad is None:
            continue
        offenders.append('%s=%s' % (key, _short(updates[key])))
        record(state, ENSEMBLE_DOF_IMPOSSIBLE, worst=worst, aborted=True)
    if not offenders:
        return
    runtime = getattr(state, 'runtime', None)
    diag = getattr(runtime, 'ensemble', None)
    diag = diag if isinstance(diag, dict) else {}

    def read(key, unit=''):
        value = diag.get(key)
        if value is None:
            return 'n/a'
        try:
            return '%.4f%s' % (float(value), unit)
        except (TypeError, ValueError):
            return 'n/a'

    raise EnsembleDivergenceError(
        'ensemble diverged at step %d: %s; this step read T_inst %s K, '
        'velocity scale %s, |strain| %s (a non-finite extended coordinate '
        'ends the run in every mode -- see md.ensemble for the series)'
        % (int(getattr(runtime, 'current_step', -1) or -1),
           '; '.join(offenders), read('T_inst'),
           read('scale_post'), read('strain_norm')))


def _short(value, limit=48):
    """Compact rendering of an offending array, capped to keep the line usable."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    head = ', '.join('%.6g' % item for item in array[:4])
    if array.size > 4:
        head += ', ... (%d values)' % array.size
    return '(%s)' % head[:limit]


# ------------------------------------------------------- recorded-step watch

def _frame_system(frame):
    """The system section of a recorded frame, as a mapping."""
    system = frame.get('system')
    return system if isinstance(system, dict) else {}


def check_recorded_step(state, frame):
    """Watch the step the run just recorded, and change nothing about it.

    A recorded frame is the object every consumer reads: the trajectory files,
    the run log, the next step's velocity kick, a restart. Four of the
    questions `verification` was meant to ask of it are answerable from the
    frame itself, and they are answered here, on the steps the run already
    asked to see:

    - does the kinetic energy the frame carries agree with the velocities the
      frame carries (`KINETIC_IDENTITY`);
    - do the periodic crossing counts the frame carries still look like
      crossing counts (`IMAGE_INTEGRITY`);
    - does the geometry a center is cut out of those counts still sit within
      bond reach of itself (`REGION_RECONSTRUCTION`);
    - does the geometry the frame carries still satisfy the constraints it was
      projected onto (`CONSTRAINT_GEOMETRY`, measured by
      `workflow/integrator/operations.py::verify_recorded`, which is where the
      plan lives).

    None of them is fatal: a frame that fails one is still a trajectory, and
    whether finding out should interrupt the run is a question of testing, so
    the verdict goes through `report()` and answers to `control.debug`.

    The caller decides when to ask. `AIMD.checkpoint` gates this on
    `output.record_due`, the one predicate that says which steps this run
    wants anything said about, so a check never costs time on a step the
    trajectory itself skips.
    """
    if not isinstance(frame, dict):
        return
    _check_kinetic_identity(state, frame)
    _check_image_integrity(state, frame)
    _check_region_reconstruction(state, frame)


def _check_kinetic_identity(state, frame):
    """Compare the recorded Ekin with the one the frame's velocities give."""
    nuclear = frame.get('nuclear_sample')
    if not isinstance(nuclear, dict):
        return
    recorded = nuclear.get('K_pre')
    velocities = nuclear.get('V_pre')
    masses = _frame_system(frame).get('mass')
    if recorded is None or velocities is None or not masses:
        return
    mass_au = np.asarray(masses, dtype=np.float64).reshape(-1) * AMU_TO_AU
    array = np.asarray(velocities, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != mass_au.shape[0]:
        return
    recomputed = 0.5 * float(np.sum(mass_au[:, None] * array ** 2))
    recorded = float(recorded)
    deviation = abs(recomputed - recorded)
    scale = max(abs(recomputed), abs(recorded))
    if deviation <= KINETIC_IDENTITY_RELATIVE * scale:
        return
    message = (
        'step %s recorded Ekin %.16g but its own velocities give %.16g '
        '(difference %.3g): a step that samples the kinetic energy twice, on '
        'either side of an action that rewrites the velocities, is describing '
        'two different states with one number'
        % (frame.get('step'), recorded, recomputed, deviation))
    if report(state, KINETIC_IDENTITY, message, worst=deviation):
        raise RecordedStepError(message)


def _check_image_integrity(state, frame):
    """Require the periodic crossing counts to still be crossing counts.

    The image is what lets a restart, a cross-frame difference and a periodic
    engine agree on which copy of an atom the run is looking at, so a frame
    whose counts went fractional, non-finite or short is not a cosmetic
    problem: the continuous position rebuilt from it is a different path.
    """
    system = _frame_system(frame)
    pbc = [int(flag) for flag in (system.get('pbc') or [])]
    if not any(pbc):
        return
    image = system.get('image')
    if image is None or len(image) == 0:
        # A configuration whose perception refused leaves no counts behind
        # (utilities/bonds returns empty with a warning); there is
        # nothing here to rejudge.
        return
    natom = len(system.get('coordinates') or [])
    array = np.asarray(image, dtype=np.float64)

    def verdict(message, worst):
        """Record the finding; a judgement run that is being tested stops."""
        if report(state, IMAGE_INTEGRITY, message, worst=worst):
            raise RecordedStepError(message)

    if array.shape != (natom, 3):
        verdict('step %s carries an image of shape %s for %d atoms'
                % (frame.get('step'), array.shape, natom), np.nan)
        return
    if not np.all(np.isfinite(array)):
        verdict('step %s carries a non-finite crossing count: %s'
                % (frame.get('step'), _short(array)), np.nan)
        return
    fractional = float(np.max(np.abs(array - np.rint(array)))) if array.size \
        else 0.0
    if fractional > 0.0:
        verdict('step %s carries a crossing count %g away from an integer '
                '(%s); counts accrue in whole cells, so this is not a rounding '
                'of a real value' % (frame.get('step'), fractional,
                                     _short(array)),
                fractional)


def _center_bond_pairs(centers, natom):
    """Every pair of atoms a center owns and a region is cut from.

    Rows are ``(center_index, i, j)`` with 0-based ordinals, in declared
    center order. A center of one high-level atom owns no pair; an index
    outside the system is left to `validate_region`, which already refuses
    it before the run starts.
    """
    pairs = []
    for center_index, center in enumerate(centers):
        region = [int(index) for index in (center.get('high_indices') or [])
                  if 0 <= int(index) < natom]
        for first in range(len(region) - 1):
            for second in range(first + 1, len(region)):
                pairs.append((center_index, region[first],
                              region[second]))
    return pairs


def _check_region_reconstruction(state, frame):
    """Require the bonds a center perceives to survive the cut out of the box.

    A periodic run does not hand a center the wrapped picture: it rebuilds the
    continuous view from this frame's coordinates and crossing counts and cuts
    the region out of that (`workflow/multiscale/regions.py::materialize_region`
    through `utilities/lattice.py::continuous_view`). Counts kept in fractional
    units are exact for the cell they were accrued under, which is the limitation
    dev/methods/PBC_DEVELOPMENT.md section 4.1 states outright: a variable-cell
    run moves the ground under them, and what that produces is not an exception
    but a center arriving at its backend as two fragments a box length apart.

    So the frame is asked the one question that decides whether its region is
    usable, and asked of the center's own atoms only. A global re-perception
    would cost the record cadence an O(N^2) scan, which is the D-17 mistake this
    module has already paid for once; the pairs that can be wrong are exactly
    the ones inside a region, and their count grows with the centers, not with
    the box. A pair must rebuild to less than half the shortest cell extent, the
    criterion `derive_initial_image` applies to the same data before it will hand
    out counts at all, and a pair that rebuilds longer than that is a finding only
    if the center would perceive it as a bond: its nearest-image distance under
    the framework's own reach (`bond_lib(...) * BOND_SCALE`,
    `utilities/bonds.py::perceive_bonds`). The two questions are asked in
    that order, cheap one first, because naming the nearest image searches the
    candidate cells and a healthy box answers "no pair rebuilds long" without
    reaching it. Measured cost is in dev/methods/PBC_DEVELOPMENT.md.

    Both readings come from the same helpers the cut uses (`cart_to_frac`,
    `frac_to_cart`), so passing here means the region really was connected, not
    that a second, slightly different arithmetic agreed. A center whose atoms all
    carry the same count is not exempt, and neither is a frame that carries no
    counts at all: that is the case where a bond crossing a face never got its
    count, whether because the cell moved under counts accrued for the old one or
    because the seed could not perceive the bond, and the tear is the same.
    """
    system = _frame_system(frame)
    pbc = [int(flag) for flag in (system.get('pbc') or [])]
    if not any(pbc):
        return
    centers = frame.get('centers') or []
    if len(centers) < 2:
        # A single-center run cuts no region: the whole system goes to the
        # backend as it stands, torn or not, and tearing is nothing the
        # reconstruction could have introduced.
        return
    coordinates = np.asarray(system.get('coordinates') or [],
                             dtype=np.float64)
    lattice = np.asarray(system.get('lattice_vector') or [], dtype=np.float64)
    atoms = list(system.get('atoms') or [])
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3
            or not coordinates.size or lattice.shape != (3, 3)
            or len(atoms) != coordinates.shape[0]):
        return
    image = system.get('image')
    if image is None or len(image) == 0:
        # No counts recorded is also the answer for an input whose perception
        # refused (utilities/bonds hands back nothing with a warning),
        # and for a structure in which nothing ever crossed. The cut takes the
        # identity path in all three, and a bond that straddles a face is
        # exactly as torn there, so the same question is asked with zero
        # counts rather than no question at all.
        counts = np.zeros(coordinates.shape, dtype=np.int64)
    else:
        counts = np.asarray(image, dtype=np.float64)
        if counts.shape != coordinates.shape \
                or not np.all(np.isfinite(counts)):
            # Whole, integer and the right shape is IMAGE_INTEGRITY's
            # finding; two checks naming one broken array would be noise.
            return
        counts = counts.astype(np.int64)
    bound = half_box_bound(lattice)
    if bound <= 0.0:
        return
    pairs = _center_bond_pairs(centers, coordinates.shape[0])
    if not pairs:
        return

    left = np.array([row[1] for row in pairs])
    right = np.array([row[2] for row in pairs])
    try:
        fractional = cart_to_frac(coordinates, lattice)
    except ValueError:
        # A singular cell already stops the run in the cell commit.
        return
    rebuilt = frac_to_cart(fractional[right] - fractional[left]
                           + counts[right] - counts[left], lattice)
    lengths = np.linalg.norm(rebuilt, axis=1)
    long_rows = np.nonzero(lengths >= bound)[0]
    if not long_rows.size:
        return

    # Only the pairs that rebuild long are then asked whether they are bonds at
    # all, and that is the expensive question: naming the nearest image searches
    # the candidate cells (utilities/lattice.py::min_image_displacement). On a
    # box whose centers are whole molecules no pair reaches this line, so the
    # search is paid for only by a frame that is already bad.
    try:
        nearest = min_image_displacement(coordinates[right[long_rows]]
                                         - coordinates[left[long_rows]],
                                         lattice)
    except ValueError:
        # A cell so far from size-reduced that its nearest image cannot be
        # named is the cell gate's finding, not this one.
        return
    min_lengths = np.linalg.norm(nearest, axis=1)
    reach = np.array([bond_lib(atoms[pairs[row][1]], atoms[pairs[row][2]])
                      * BOND_SCALE for row in long_rows])
    bonded = np.nonzero(min_lengths < reach)[0]
    if not bonded.size:
        return

    first_row = int(bonded[0])
    row = int(long_rows[first_row])
    center_index, first, second = pairs[row]
    length = float(lengths[row])
    message = (
        'step %s cuts center %d (c=%d) out of a torn cluster: atoms %d and %d '
        'read %.3f A apart across the cell, within the perception reach of '
        '%.3f A, but the crossing counts this frame carries rebuild that pair '
        '%.3f A apart, at or beyond half of the shortest cell extent (%.3f A); '
        '%d bonded pair(s) of the centers rebuild that way this step. What the '
        'center is handed is fragments, not one molecule'
        % (frame.get('step'), center_index + 1, center_index, first + 1,
           second + 1, float(min_lengths[first_row]),
           float(reach[first_row]), length, bound, int(bonded.size)))
    if report(state, REGION_RECONSTRUCTION, message, worst=length):
        raise RecordedStepError(message)
