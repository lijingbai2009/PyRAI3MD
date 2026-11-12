"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 27 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

The resident LAMMPS service: the only module of this skill that imports
LAMMPS or MPI.

Run as ``python -m pyrai3md.skills.lammps_skill.service.persistent.server``
by service/persistent/launcher.py, or by an operator with the same
arguments. It holds one LAMMPS instance whose topology and force field
are loaded once, applies the geometry of one calculation step, and
answers with the numbers that step produces.

Two rules shape its structure.

Every rank runs the same sequence. Only rank 0 touches the socket; it
reads one operation and broadcasts it, and it broadcasts exactly once per
loop iteration, so the ranks cannot drift apart in their collectives. A
request rejected before any LAMMPS call is answered without a broadcast:
no collective had started, so nothing needs to be unwound.

A LAMMPS error is not recoverable from Python either. LAMMPS reports
input errors through MPI_Abort, which takes the process group down, so
the code does not pretend a try/except around a run could keep a session
alive: after the first LAMMPS call, failure means the process ends and
the owner watchdog cleans up.

The instance is closed, never finalized. ``finalize()`` shuts down MPI
itself, which would kill a sibling instance and leave the other ranks
waiting in a collective that can no longer complete.
"""

import argparse
import ctypes
import os
import select
import socket
import sys
import time

try:
    from .protocol import (
        IDLE_TIMEOUT_S,
        LammpsProtocolError,
        LammpsServiceDead,
        LammpsServiceTimeout,
        OP_CLOSE,
        OP_EVALUATE,
        OP_HELLO,
        OP_INIT,
        OWNER_POLL_INTERVAL_S,
        PROTOCOL_VERSION,
        TILT_EPSILON,
        build_error,
        build_result,
        check_fields,
        encode_frame,
        read_frame,
    )
except ImportError:  # pragma: no cover - taken when run as a script
    # The service is launched as a file rather than through ``python -m``,
    # so the service interpreter needs no PyRAI3MD installation and no
    # pydantic: protocol.py beside this file is the whole contract that
    # crosses the process boundary.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from protocol import (
        IDLE_TIMEOUT_S,
        LammpsProtocolError,
        LammpsServiceDead,
        LammpsServiceTimeout,
        OP_CLOSE,
        OP_EVALUATE,
        OP_HELLO,
        OP_INIT,
        OWNER_POLL_INTERVAL_S,
        PROTOCOL_VERSION,
        TILT_EPSILON,
        build_error,
        build_result,
        check_fields,
        encode_frame,
        read_frame,
    )

try:
    from .profile import PROFILE_EVERY, perf_row
except ImportError:  # pragma: no cover - taken when run as a script
    from profile import PROFILE_EVERY, perf_row

# Read-back tolerance, in the native length unit of the input.
BOX_TOLERANCE = 1e-8
# How long rank 0 waits for a request before checking on its owner.
READ_WAIT_S = 2.0

# LAMMPS interface names that changed shape between releases. Each entry
# is tried in order and the first working form is remembered, so an API
# drift is one adapter table to fix rather than a hunt through the loop.
# (cstyle, ctype) spellings for a global scalar and a global vector. The
# SWIG wrapper documents the names; the pybind11 one passes the C integers
# straight through (LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR/VECTOR), and neither
# says which one a build accepts, so both are tried.
_COMPUTE_SCALAR_ALIASES = (('global', 'scalar'), ('global', 0), (0, 0),
                           (0, 'scalar'))
_COMPUTE_VECTOR_ALIASES = (('global', 'vector'), ('global', 1), (0, 1),
                           (0, 'vector'))

# Operations the service accepts from a client.
_REQUEST_OPS = (OP_HELLO, OP_INIT, OP_EVALUATE, OP_CLOSE)

# Box numbers in the order both sides agree on, in the frame a data
# file states them (the bounding planes and the tilt factors).
_BOX_ORDER = ('xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'xy', 'xz', 'yz')

# The session's phase table, written where the run's other artifacts are.
_PERF_FILE = 'lammps_resident.perf'

# That table's refresh cadence is the backend table's (PROFILE_EVERY,
# imported above): one render is a handful of formatted lines, so at this
# cadence the writes are invisible against the per-step cost they measure,
# while a short session (a skill test of a few dozen steps) still leaves
# several snapshots behind.
# The tilt subset of that order, which change_box spells separately.
_TILT_ORDER = ('xy', 'xz', 'yz')


def _import_native():
    """Import LAMMPS inside this process and return the module.

    Imported only here, and only after the working directory is set, so
    the framework interpreter never links liblammps.
    """
    import lammps
    return lammps


def _import_mpi():
    """Import mpi4py, or None for a single-rank service."""
    from mpi4py import MPI
    return MPI


def _has_mpi_support(lammps) -> bool:
    """Read the module's MPI capability wherever its packaging keeps it.

    A LAMMPS build exposes 'has_mpi_support' on the module, the class or
    nowhere at all, so a module-level read alone calls an MPI-linked
    build a serial one. The class answers when the module does not.
    """
    for holder in (lammps, getattr(lammps, 'lammps', None)):
        if holder is not None and hasattr(holder, 'has_mpi_support'):
            return bool(getattr(holder, 'has_mpi_support'))
    return False


class _ApiAdapter:
    """Remember which call shape works on this LAMMPS build.

    The extraction calls are the part of the interface that has changed
    across releases, so they are resolved once by trial and recorded.
    Nothing is assumed silently: an extraction that cannot be resolved
    raises with the candidate spellings it tried.
    """

    def __init__(self, lmp):
        self._lmp = lmp
        self._compute_form = {}
        self._subset_form = None
        self._scatter_form = None

    def gather_subset(self, name, dtype, count, ids):
        """Gather one per-atom property for the requested atom ids.

        The two Python interfaces disagree twice over. The pybind11 one
        mirrors the C signature, so it takes ``ndata`` (the number of ids)
        and wants a ctypes array of ints; the older one takes four
        arguments and a plain list. The first call finds out which shape
        this build accepts and records it, because guessing wrong is a
        TypeError inside a load rather than a wrong number, and a wrong
        number is the only thing worse than that.

        Parameters
        ----------
        name : str
            Atom property ('id', 'f', ...).
        dtype : int
            0 for integer data, 1 for floating-point data.
        count : int
            Values per atom.
        ids : sequence of int
            Atom ids, in the order the answer comes back.

        Returns
        -------
        list of float
        """
        lmp = self._lmp
        if not hasattr(lmp, 'gather_atoms_subset'):
            raise RuntimeError(
                'this LAMMPS Python interface has no gather_atoms_subset; '
                'forces could not be gathered in atom-id order without '
                'risking ghost copies')
        tags = [int(tag) for tag in ids]
        self._gather_state()
        candidates = []
        if self._subset_form is not None:
            candidates.append(self._subset_form)
        candidates += [_gather_ndata_ctypes, _gather_ndata_list,
                       _gather_flat_list, _gather_flat_ctypes]
        failure = None
        for call in candidates:
            try:
                flat = call(lmp, name, dtype, count, tags)
            except (TypeError, ValueError, ctypes.ArgumentError) as exc:
                failure = exc
                continue
            self._subset_form = call
            return _flatten(flat)
        raise RuntimeError(
            f'gather_atoms_subset accepted none of the known call shapes '
            f'for {name!r}: {failure}')

    def probe(self):
        """Report which interface names exist.

        Returns
        -------
        dict
            Name to boolean, sent to the client in the HELLO reply so an
            unusable build is visible before anything is loaded.
        """
        names = ('file', 'command', 'commands_list', 'scatter_atoms',
                 'gather_atoms', 'gather_atoms_subset', 'extract_compute',
                 'extract_modify', 'extract_box', 'get_natoms',
                 'gather_state', 'close')
        return {name: hasattr(self._lmp, name) for name in names}

    def commands(self, lines):
        """Issue one command sequence in the order given.

        Every step of a resident run is evaluated by the same sequence the
        client sent at INIT, so the loaded script never has to run on its
        own and a step cannot differ from the subprocess path in what
        LAMMPS is asked to do. An empty sequence is a bug, not a no-op.
        """
        lines = [str(line) for line in lines if str(line).strip()]
        if not lines:
            raise RuntimeError(
                'the resident evaluation sequence is empty; the step would '
                'answer with the numbers of whatever ran last')
        if hasattr(self._lmp, 'commands_list'):
            self._lmp.commands_list(lines)
            return
        for line in lines:
            self._lmp.command(line)

    def energy(self):
        """Total potential energy of the last run, in native units."""
        return _compute_numbers(self._extract_compute('thermo_pe'))[0]

    def pressure6(self):
        """Six configurational pressure components, native units."""
        # A pressure compute is a global *vector*, and on the pybind11
        # interface a vector comes back as a pointer, which carries no
        # length of its own: the six components are the contract of this
        # method, so the count is stated rather than discovered.
        flat = _compute_numbers(
            self._extract_compute('Pconf', vector=True), count=6)
        if len(flat) != 6:
            raise RuntimeError(
                f'Pconf returned {len(flat)} values, expected the six global '
                f'components xx yy zz xy xz yz')
        return flat

    def forces(self, order):
        """(natom, 3) forces in the requested atom-id order, no ghosts."""
        # gather_atoms (without a subset) returns nlocal + nghost joined
        # across ranks, whose order cannot be undone. gather_atoms_subset
        # resolves each id through atom->map(), which never matches a
        # ghost (LAMMPS stores ghost tags negated), so exactly one owned
        # copy of each id comes back, in the order asked for.
        flat = self.gather_subset('f', 1, 3, order)
        if len(flat) != 3 * len(order):
            raise RuntimeError(
                f'force gather returned {len(flat)} numbers for '
                f'{len(order)} atoms')
        return [[float(flat[3 * n + 0]), float(flat[3 * n + 1]),
                 float(flat[3 * n + 2])] for n in range(len(order))]

    def per_atom_compute(self, cid, ids):
        """{atom id: value} of a per-atom compute, gathered across ranks.

        The LAMMPS Python interface exposes a per-atom compute as a
        local-length array (style 1, type 1), so the atom ids of the same
        rank are read beside it and the pairs are joined. Under MPI the
        ranks each contribute their owned atoms, through mpi4py when it
        is importable; a missing id is an error rather than a silently
        short dictionary.

        Parameters
        ----------
        cid : str
            Compute id, which must produce a per-atom vector.
        ids : set of int
            Atom ids to keep.

        Returns
        -------
        dict
            ``{atom id: float value}`` for every requested id.
        """
        wanted = {int(value) for value in ids}
        try:
            import numpy as np
            local_ids = np.asarray(
                self._lmp.numpy.extract_atom('id')).ravel()
            values = np.asarray(
                self._lmp.numpy.extract_compute(cid, 1, 1)).ravel()
        except Exception as exc:                    # noqa: BLE001
            raise RuntimeError(
                f'cannot read the per-atom compute {cid!r} from this '
                f'LAMMPS interface: {type(exc).__name__}: {exc}')
        local = {int(atom_id): float(value)
                 for atom_id, value in zip(local_ids, values)
                 if int(atom_id) in wanted}
        try:
            from mpi4py import MPI
            parts = MPI.COMM_WORLD.allgather(local)
        except ImportError:
            parts = [local]
        merged = {}
        for part in parts:
            merged.update(part)
        missing = wanted - set(merged)
        if missing:
            raise RuntimeError(
                f'the per-atom compute {cid!r} returned no value for atom '
                f'ids {sorted(missing)}; a resident MPI run needs mpi4py '
                f'for this channel')
        return merged

    def scatter_positions(self, coordinates):
        """Write the full position array on every rank.

        The data argument follows the same interface split as the gathers:
        the pybind11 one hands the array straight to the C function and so
        needs a ctypes array, the older one converts a Python list. The
        accepted form is learned once.
        """
        flat = [float(value) for row in coordinates for value in row]
        lmp = self._lmp
        candidates = []
        if self._scatter_form is not None:
            candidates.append(self._scatter_form)
        candidates += [_scatter_list, _scatter_ctypes]
        failure = None
        for call in candidates:
            try:
                call(lmp, flat)
            except (TypeError, ValueError, ctypes.ArgumentError) as exc:
                failure = exc
                continue
            self._scatter_form = call
            return
        raise RuntimeError(
            f'scatter_atoms accepted neither a list nor a ctypes array of '
            f'{len(flat)} positions: {failure}')

    def box(self):
        """The domain's box numbers, or None if it cannot report them."""
        lmp = self._lmp
        if hasattr(lmp, 'extract_box'):
            numbers = _normalize_extract_box(lmp.extract_box())
            if numbers is not None:
                return numbers
        if hasattr(lmp, 'extract_modify'):
            try:
                lo = _pointer_list(lmp.extract_modify('boxlo'))
                hi = _pointer_list(lmp.extract_modify('boxhi'))
                tilt = [_pointer_list(lmp.extract_modify(name))[0]
                        for name in ('xy', 'xz', 'yz')]
                return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]] + tilt
            except (TypeError, IndexError, ValueError):
                return None
        return None

    def triclinic(self):
        """Whether the domain currently carries tilt factors, or None."""
        lmp = self._lmp
        if hasattr(lmp, 'extract_setting'):
            try:
                return bool(lmp.extract_setting('triclinic'))
            except TypeError:                       # pragma: no cover
                return None
        return None

    def _gather_state(self) -> None:
        """Materialize per-atom vectors, where the interface needs it.

        The SWIG interface exposes a ``gather_state`` method; the pybind11
        one does not, because its gathers read the live atom arrays. There
        is deliberately no ``command('gather_state')`` fallback: that is not
        a LAMMPS input command, and an unknown command makes LAMMPS abort
        the run rather than tell us the method was missing.
        """
        lmp = self._lmp
        if hasattr(lmp, 'gather_state'):
            lmp.gather_state()

    def _extract_compute(self, cid: str, vector: bool = False):
        """Fetch a global compute by whichever call shape works.

        The two interfaces spell the style and type constants differently
        (the strings the SWIG wrapper documented, the integers the C
        interface takes) and answer a vector with a pointer that carries no
        length, so a scalar and a vector request are resolved and recorded
        separately.
        """
        lmp = self._lmp
        key = 'vector' if vector else 'scalar'
        forms = []
        if self._compute_form.get(key):
            forms.append(self._compute_form[key])
        forms += [form for form in (_COMPUTE_VECTOR_ALIASES if vector
                                    else _COMPUTE_SCALAR_ALIASES)
                  if form != self._compute_form.get(key)]
        tried = []
        for style, kind in forms:
            try:
                value = lmp.extract_compute(cid, style, kind)
            except Exception as exc:                # noqa: BLE001
                tried.append(f'{style}/{kind}: {type(exc).__name__}')
                continue
            if value is None:
                tried.append(f'{style}/{kind}: returned None')
                continue
            self._compute_form[key] = (style, kind)
            return value
        raise RuntimeError(
            f'cannot read the global {"vector" if vector else "scalar"} '
            f'compute {cid!r} from this LAMMPS interface '
            f'(tried {"; ".join(tried)})')


def _session_cwd(cwd) -> str:
    """The request cwd the service may move into, or '' for none.

    Only an absolute path counts. ``main()`` has already chdir'd into the
    session directory, which the launcher resolved absolutely, and the
    request names the same calculation directory: applying a relative base
    again would resolve it against the session directory itself, one level
    too deep, so the deck the client wrote would not be found. The client's
    ``calcdir`` can be relative when ``environment.scratch_path`` is
    written that way.
    """
    if cwd and os.path.isabs(str(cwd)):
        return str(cwd)
    return ''


class ResidentEngine:
    """One loaded LAMMPS instance and the operations it can serve.

    Parameters
    ----------
    lmp : lammps.lammps
        Live instance, created with the service communicator.
    adapter : _ApiAdapter
        Resolved interface shapes.
    state : dict
        Service facts reported through HELLO.
    """

    def __init__(self, lmp, adapter, state=None):
        self.lmp = lmp
        self.adapter = adapter
        self.state = dict(state or {})
        # The flags the instance holds now. None until the first
        # evaluation restates them, because what read_data derived from
        # box containment is not something to trust blindly.
        self.state.setdefault('applied_image', None)
        # The cell the domain holds now, in the same not-trusted-until-
        # stated form: None until the first evaluation applies one, so
        # the step that follows INIT always gets its read-back check.
        self.state.setdefault('applied_box', None)
        # Wall time per engine phase, reported when the session ends.
        self.perf = {}
        self.perf_steps = 0
        self.generation = 0

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def init(self, request: dict) -> dict:
        """Load the static package and evaluate it once.

        The script is the head of the very input file the subprocess path
        writes, so what gets loaded is the same text: same includes, same
        read_data, same kspace re-issue, same shake decision. It stops
        short of running: the evaluation belongs to ``step_tail``, which
        is also what every later step runs, so the smoke number and the
        first step's number come from one command sequence.
        """
        cwd = _session_cwd(request.get('cwd'))
        if cwd and os.path.isdir(cwd):
            os.chdir(cwd)
        # Remembered for the phase table, which belongs beside the run's
        # other artifacts rather than only in the launcher's log.
        self.state['cwd'] = os.getcwd()
        script = request['script']
        if not script.endswith('.in'):
            script = f'{script}.in'
        self.lmp.file(script)

        self.state['order'] = [int(tag) for tag in request['order']]
        self.state['natom'] = int(request['natom'])
        self.state['capabilities'] = dict(request['capabilities'])
        self.state['step_tail'] = [str(line) for line in request['step_tail']]
        self.state['periodic'] = any(int(flag) for flag in request['pbc'])
        self.state['kspace_line'] = request.get('kspace') or ''
        # The computes a QM/MM step reads, or None for an ordinary run.
        self.state['qmmm'] = request.get('qmmm') or None

        natoms = int(self.lmp.get_natoms())
        if natoms != self.state['natom']:
            raise RuntimeError(
                f'instance loaded {natoms} atoms, the model declares '
                f'{self.state["natom"]}')
        # 'id' is integer data, so dtype 0: the pybind11 interface picks the
        # element type of the array it allocates from that argument, and a
        # 1 there asks the C function to write ints into doubles.
        tags = [int(value) for value in
                self.adapter.gather_subset('id', 0, 1, self.state['order'])]
        if sorted(tags) != sorted(self.state['order']):
            raise RuntimeError(
                f'instance atom ids do not match the model mapping '
                f'({len(set(tags))} distinct ids for '
                f'{len(self.state["order"])} expected); the resident path '
                f'needs a one-to-one id mapping')

        # The domain's tilt capability is left exactly as the loaded script
        # made it, and remembered: conversion is not free of consequence.
        # Measured in the test environment, an untilted 11x10x10 cell computes
        # -19.811016 kcal/mol, while the same cell on a domain converted
        # for nothing computes -19.760369, which is the whole of the
        # variable-cell parity gap this path exists to close. A cell that
        # carries tilt converts the domain when it arrives (see
        # _apply_box), which is what the cold path does by writing tilt
        # factors into its data file.
        triclinic = self.adapter.triclinic()
        self.state['triclinic'] = bool(triclinic) if triclinic is not None \
            else None

        self.adapter.commands(self.state['step_tail'])
        self.state['tail_line'] = self.state['step_tail'][0]
        energy = self.adapter.energy()
        if energy != energy:                        # NaN guard
            raise RuntimeError('smoke evaluation returned a NaN energy')
        box = self.adapter.box()
        if self.state['periodic'] and box is None:
            raise RuntimeError(
                'this LAMMPS interface reports no box, so a variable-cell '
                'resident run cannot be verified; run the bringup tool, or '
                'use the subprocess path')
        return {
            'natom': natoms,
            'triclinic': bool(triclinic) if triclinic is not None else None,
            'box': [float(value) for value in box] if box else None,
            'smoke_energy_raw': float(energy),
        }

    def evaluate(self, request: dict) -> dict:
        """Apply one step's geometry and return the raw numbers.

        The order is fixed: box, then positions, then the image flags that
        belong to those positions, then the evaluation sequence the client
        installed at INIT. Coordinates arrive wrapped into the main image
        exactly as the framework stores them, and the crossing counts that
        say which image each atom sits in arrive beside them, because the
        live instance still holds the flags of the previous step and a new
        wrapped coordinate written onto a stale flag relocates the atom by
        a whole box. Under ``debug`` that sequence also carries the write
        commands, so a resident step leaves the same files behind as the
        subprocess step it stands in for.
        """
        box = request.get('box')
        started = time.perf_counter()
        if box is not None:
            self._timed('box', self._apply_box, box)
        self._timed('scatter', self.adapter.scatter_positions,
                    request['coordinates'])
        self._timed('image', self._apply_image, request.get('image'))
        qmmm = self.state.get('qmmm')
        profile = str((qmmm or {}).get('profile', '')).lower()

        if profile == 'qmmm_field':
            reply = self._evaluate_field(request, qmmm)
            self.perf_steps += 1
            self.perf['evaluate'] = self.perf.get('evaluate', 0.0) \
                + (time.perf_counter() - started)
            if not self.perf_steps % PROFILE_EVERY:
                self._write_perf(self._perf_lines())
            return reply

        if profile == 'qmmm_energy':
            q_qm = request.get('q_qm')
            qm_ids = [int(value) for value in qmmm['qm_ids']]
            if q_qm is None or len(q_qm) != len(qm_ids):
                raise RuntimeError(
                    f'qmmm_energy step carries {0 if q_qm is None else len(q_qm)} '
                    f'q_qm values for {len(qm_ids)} QM atoms')
            self._timed('step_tail', self.adapter.commands, [
                f'set atom {atom_id} charge {float(charge):.16e}'
                for atom_id, charge in zip(qm_ids, q_qm)])
            self._timed('step_tail', self.adapter.commands,
                        self.state['step_tail'])
        else:
            self._timed('step_tail', self.adapter.commands,
                        self.state['step_tail'])

        capabilities = self.state['capabilities']
        reply = {
            'step': int(request['step']),
            'natom': self.state['natom'],
            'order': 'atom_id',
            'units': 'lammps-native',
            'energy_raw': self._timed('extract', self.adapter.energy),
        }
        if capabilities.get('gradient'):
            reply['forces_raw'] = self._timed(
                'extract', self.adapter.forces, self.state['order'])
        if capabilities.get('stress'):
            reply['pressure6_raw'] = self._timed(
                'extract', self.adapter.pressure6)
        self.perf_steps += 1
        self.perf['evaluate'] = self.perf.get('evaluate', 0.0) \
            + (time.perf_counter() - started)
        if not self.perf_steps % PROFILE_EVERY:
            # Refreshed during the run, not only at its end: nothing in
            # the framework closes a resident session today (the service
            # is reaped by the owner-pid watchdog when the owning process
            # exits), so a table written only on CLOSE is a table a normal
            # run never produces. One measured example proved it: the log
            # finished with no [perf] line anywhere.
            self._write_perf(self._perf_lines())
        return reply

    def _qm_forces(self, qm_ids):
        """Forces of the requested atom ids, in the requested order."""
        order = [int(value) for value in self.state['order']]
        forces = self.adapter.forces(order)
        by_id = {atom_id: [float(value) for value in row]
                 for atom_id, row in zip(order, forces)}
        return [by_id[int(atom_id)] for atom_id in qm_ids]

    def _evaluate_field(self, request, qmmm):
        """Run the A/B Coulomb probe on the loaded instance.

        A sets the probe group to +1 and keeps the environment charges; B
        sets the environment to zero. Both runs use the same loaded
        coefficients, cutoffs, special_bonds and kspace, so the per-atom
        difference is the environment contribution at the probed atoms. The
        environment is restored before the reply so the next step starts
        from a known charge set.

        Under several centers the environment is not only the MM subsystem:
        the centers this probe does not measure sit in it at their current
        0-order values, so those values are written first (one ``set atom``
        per QM atom, the same channel the energy step already uses) and the
        rows are extracted for the probed center alone. With one center no
        charges arrive, the probe is the whole QM set, and the command
        sequence is exactly what it always was.
        """
        qm_ids = [int(value) for value in qmmm['qm_ids']]
        probe_ids = [int(value) for value in
                     (qmmm.get('probe_ids') or qm_ids)]
        probe_set = set(probe_ids)
        env_q = request.get('q_qm')
        commands = []
        if env_q:
            by_id = dict(zip(qm_ids, [float(value) for value in env_q]))
            commands = [f'set atom {atom_id} charge {by_id[atom_id]:.16e}'
                        for atom_id in qm_ids]
            self._timed('step_tail', self.adapter.commands, commands)
        step_tail = self.state['step_tail']
        gg = qmmm['gg']
        self._timed('step_tail', self.adapter.commands,
                    ['set group QMMM_QM charge 1.0'])
        self._timed('step_tail', self.adapter.commands, step_tail)
        pe_a = self._timed('extract_qmmm', self.adapter.per_atom_compute,
                           qmmm['peq'], probe_set)
        forces_a = self._timed('extract_qmmm', self._qm_forces, probe_ids)
        u_gg = self._timed(
            'extract_qmmm',
            lambda: _compute_numbers(self.adapter._extract_compute(gg))[0])
        f_gg = self._timed(
            'extract_qmmm',
            lambda: _compute_numbers(
                self.adapter._extract_compute(gg, vector=True), count=3))
        energy_a = self._timed('extract', self.adapter.energy)
        self._timed('step_tail', self.adapter.commands,
                    ['set group QMMM_MM charge 0.0'])
        self._timed('step_tail', self.adapter.commands, step_tail)
        pe_b = self._timed('extract_qmmm', self.adapter.per_atom_compute,
                           qmmm['peq'], probe_set)
        forces_b = self._timed('extract_qmmm', self._qm_forces, probe_ids)
        for group in qmmm.get('restore_groups') or []:
            self._timed(
                'step_tail', self.adapter.commands,
                [f"set group {group['name']} charge "
                 f"{float(group['charge']):.16e}"])
        return {
            'step': int(request['step']),
            'natom': self.state['natom'],
            'order': 'atom_id',
            'units': 'lammps-native',
            'energy_raw': float(energy_a),
            'qmmm_raw': {
                'pe_a': {int(key): float(value)
                         for key, value in pe_a.items()},
                'pe_b': {int(key): float(value)
                         for key, value in pe_b.items()},
                'forces_a': forces_a,
                'forces_b': forces_b,
                'u_gg': float(u_gg),
                'f_gg': [float(value) for value in f_gg],
            },
        }

    def _timed(self, phase, call, *args):
        """Run one engine phase, accumulating the wall time it took.

        Always on: two clock reads per phase are nanoseconds against the
        milliseconds they measure, and a phase that has to be switched on
        to be measured is a phase nobody measures before it goes wrong.
        """
        started = time.perf_counter()
        try:
            return call(*args)
        finally:
            self.perf[phase] = self.perf.get(phase, 0.0) \
                + (time.perf_counter() - started)

    def report_perf(self, stream=None) -> None:
        """Print one line per phase, in milliseconds per evaluated step.

        The subprocess path spent about 0.95 s per step restarting LAMMPS,
        and this service spends milliseconds on a step whose own force
        evaluation is a few microseconds, so the question is where those
        milliseconds go. The answer goes to two places: the service's
        stdout, which the launcher redirects into its log, and a file in
        the session directory, because that directory is what a remote run
        hands back and a log left on the cluster answers nothing about the
        run that is being read. Nothing is judged here.
        """
        lines = self._perf_lines()
        if not lines:
            return
        out = stream if stream is not None else sys.stdout
        for line in lines:
            print(line, file=out, flush=True)
        self._write_perf(lines)

    def _perf_lines(self) -> list:
        """Render the phase table, or nothing before the first step."""
        steps = self.perf_steps or 0
        if not steps:
            return []
        lines = []
        for phase in ('evaluate', 'box', 'scatter', 'image', 'step_tail',
                      'extract'):
            seconds = self.perf.get(phase)
            if seconds is None:
                continue
            lines.append(perf_row(phase, steps, seconds))
        skipped = self.perf.get('box_skipped')
        if skipped:
            # How much of the box phase the fixed-cell steps gave back.
            lines.append('[perf] box_skipped=%d of %d steps'
                         % (skipped, steps))
        tail = self.state.get('tail_line')
        if tail:
            # Which command the run evaluated with, printed into the run's own
            # artifacts. An A/B of two builds is otherwise unreadable: one
            # version of this harness flipped a constant that never took
            # effect, both sides produced the same digest, and only this row
            # showed that the two sides had run the same code.
            lines.append("[perf] per_step_line='%s'" % tail)
        return lines

    def _write_perf(self, lines) -> None:
        """Keep the phase table beside the run's own artifacts.

        Best-effort by design: a service that cannot write its timings must
        still answer the calculation that was asked of it, so every failure
        here is swallowed rather than raised into the client's result.
        """
        directory = self.state.get('cwd')
        if not directory:
            return
        try:
            with open(os.path.join(directory, _PERF_FILE), 'w',
                      encoding='utf-8') as handle:
                handle.write('\n'.join(lines) + '\n')
        except OSError:                           # noqa: PERF203
            pass

    def close(self) -> None:
        """Release the instance without terminating MPI.

        Closing is the last act of a session whose answer is already
        written, so a failure here cannot be raised into a result; it is
        printed to the service log instead of vanishing.
        """
        self.report_perf()
        try:
            self.lmp.close()
        except Exception as exc:                    # noqa: BLE001
            print(f'resident close failed: {type(exc).__name__}: {exc}',
                  file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Box handling
    # ------------------------------------------------------------------

    def _apply_box(self, box: dict) -> None:
        """Move the live domain to the box the client computed.

        A cell that carries tilt first makes the domain tilt-capable,
        which is the same thing the cold path does by writing tilt factors
        into its data file; a cell without tilt leaves the domain as the
        script loaded it. The split is not cosmetic: the conversion moves
        the reciprocal sum, so an untilted cell computed on a domain that
        was converted for nothing differs from the subprocess path by
        more than the parity this path is judged on.

        The read-back check that follows is what turns a deployment whose
        LAMMPS spells this differently into a loud failure at the first
        step instead of a run of numbers computed on the wrong cell.

        A step that repeats the cell already applied is skipped. This is
        not a micro-optimisation of one command: ``change_box`` makes
        LAMMPS rebuild the reciprocal representation, so a fixed-cell run
        pays a PPPM initialisation per step for a box that never changed.
        One measured NVE example sent the same cell 10000 times and drew
        10001 ``PPPM initialization`` blocks with it. The signature is
        recorded only after a verified application, so a cell that was
        refused never becomes the one later steps are compared against,
        and the first step still applies and verifies because nothing is
        recorded yet - which is also what keeps the tilt conversion on the
        path that needs it.
        """
        signature = tuple(float(box[key]) for key in _BOX_ORDER)
        if signature == self.state.get('applied_box'):
            self.perf['box_skipped'] = self.perf.get('box_skipped', 0) + 1
            return
        tilted = any(abs(float(box[key])) > TILT_EPSILON
                     for key in _TILT_ORDER)
        if tilted and not self.state.get('triclinic'):
            self._make_triclinic()
        lines = _change_box_lines(box, triclinic=bool(
            self.state.get('triclinic')))
        for line in lines:
            self.lmp.command(line)
        self._verify_box(box)
        self.state['applied_box'] = signature

    def _apply_image(self, image):
        """Move the instance's image flags onto this step's crossings.

        The framework stores wrapped (main image) coordinates, so the
        counts are the only thing that says which copy of the cell an atom
        means. Writing them together with the positions is what keeps the
        live instance's unwrapped coordinates continuous; leaving them from
        the previous step is what makes LAMMPS report inconsistent image
        flags and evaluate a bond across the whole box.

        A non-periodic run owns no flags and is left alone. A step that
        reports no rows at all (a client from before the counts existed)
        is read as "nothing has crossed", which is what such a run meant.
        """
        if not self.state.get('periodic'):
            return
        natom = int(self.state.get('natom') or 0)
        rows = [[int(value) for value in row] for row in (image or [])]
        if not rows:
            rows = [[0, 0, 0]] * natom
        if len(rows) != natom:
            raise RuntimeError(
                f'the step shipped image counts for {len(rows)} atoms; the '
                f'loaded instance holds {natom}')
        commands = _image_commands(rows, self.state.get('applied_image'))
        if commands:
            self.adapter.commands(commands)
        self.state['applied_image'] = rows

    def _make_triclinic(self) -> None:
        """Give the domain tilt factors, and bring kspace back with it.

        The conversion invalidates ``kspace_style`` (LAMMPS drops it rather
        than mis-frame a reciprocal sum on a box whose shape it no longer
        knows), so the line the client installed at INIT is re-issued, as
        the cold path does after reading a tilted data file.
        """
        self.lmp.command('change_box all triclinic')
        # One command per line: the field probe ships its Ewald pin on the
        # second line, and LAMMPS's Input::one does not split newlines.
        for command in str(self.state.get('kspace_line') or '').splitlines():
            if command.strip():
                self.lmp.command(command.strip())
        self.state['triclinic'] = True

    def _verify_box(self, box: dict) -> None:
        """Refuse to keep running on a box the domain did not accept."""
        read_back = self.adapter.box()
        if read_back is None:
            raise RuntimeError(
                'box read-back is unavailable on this LAMMPS interface; a '
                'variable-cell resident run cannot be verified')
        if len(read_back) < len(_BOX_ORDER):
            raise RuntimeError(
                f'box read-back returned {len(read_back)} numbers, expected '
                f'{len(_BOX_ORDER)}')
        mismatched = [
            f'{key}: asked {float(box[key])!r}, domain holds '
            f'{float(read_back[index])!r}'
            for index, key in enumerate(_BOX_ORDER)
            if abs(float(box[key]) - float(read_back[index]))
            > BOX_TOLERANCE]
        if mismatched:
            # Both whole vectors and the untouched answer they came from: a
            # tilt convention that disagrees shows up in several entries at
            # once, and a normalizer that misread the interface looks just
            # like a domain that refused the cell. Guessing from one number
            # is how a box bug gets a second bug piled on top of it.
            asked = [float(box[key]) for key in _BOX_ORDER]
            try:
                raw = repr(self.lmp.extract_box())
            except Exception:                       # noqa: BLE001
                raw = 'unavailable'
            raise RuntimeError(
                'box read-back mismatch (' + '; '.join(mismatched) + '); '
                f'asked {asked!r}, normalized {read_back[:9]!r}, '
                f'extract_box() {raw}'
            )


def _flatten(value):
    """Flatten nested sequences of numbers into one list.

    A ctypes array is a sequence too, and it is what the gathers answer
    with; leaving it out of the test turns the whole array into one
    element, and int() of it then reads as a run of bytes.
    """
    if value is None:
        return []
    if isinstance(value, ctypes.Array) or isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _pointer_list(pointer):
    """Read the floats a ctypes pointer from extract_modify addresses."""
    if pointer is None:
        raise TypeError('extract_modify returned no pointer')
    try:
        return [float(pointer[0]), float(pointer[1]), float(pointer[2])]
    except TypeError:
        return [float(value) for value in pointer]


def _compute_numbers(value, count: int = None):
    """Read a compute result as numbers, in whichever shape it arrived.

    A global scalar comes back as a float on one interface and as a
    ``c_double`` on the other, and a global vector as a pointer that
    carries no length, which is why *count* is stated by the caller
    instead of being discovered from the object.
    """
    if value is None:
        raise RuntimeError('the compute returned no data on this build')
    if isinstance(value, ctypes.Array):
        values = list(value)
    elif isinstance(value, (list, tuple)):
        values = _flatten(value)
    elif hasattr(value, 'value') and isinstance(
            getattr(value, 'value'), (int, float)):
        # A ctypes scalar (c_double, c_int), which has no __float__ of its
        # own: the number is behind .value.
        values = [value.value]
    else:
        try:
            # A ctypes pointer: subscriptable, but with no known size.
            values = [value[index] for index in
                      range(count if count is not None else 1)]
        except TypeError:
            values = [value]
    numbers = [float(item) for item in values]
    if count is not None and len(numbers) != count:
        raise RuntimeError(
            f'compute returned {len(numbers)} values, expected {count}')
    return numbers


def _int_array(values):
    """A ctypes int array, the shape the pybind11 gathers pass to the C API."""
    return (ctypes.c_int * len(values))(*[int(value) for value in values])


def _double_array(values):
    """A ctypes double array, for scattering positions on that interface."""
    return (ctypes.c_double * len(values))(*[float(v) for v in values])


def _gather_ndata_ctypes(lmp, name, dtype, count, ids):
    return lmp.gather_atoms_subset(name, dtype, count, len(ids),
                                   _int_array(ids))


def _gather_ndata_list(lmp, name, dtype, count, ids):
    return lmp.gather_atoms_subset(name, dtype, count, len(ids), ids)


def _gather_flat_list(lmp, name, dtype, count, ids):
    return lmp.gather_atoms_subset(name, dtype, count, ids)


def _gather_flat_ctypes(lmp, name, dtype, count, ids):
    return lmp.gather_atoms_subset(name, dtype, count, _int_array(ids))


def _scatter_list(lmp, flat):
    lmp.scatter_atoms('x', 1, 3, flat)


def _scatter_ctypes(lmp, flat):
    lmp.scatter_atoms('x', 1, 3, _double_array(flat))


def _change_box_lines(box: dict, triclinic: bool = True) -> list:
    """Spell one box change the way LAMMPS accepts it.

    ``change_box`` gives every dimension its own keyword, so the nine true
    numbers travel as two commands, the bounds and the tilt factors. The
    single-clause form a reader might expect from the data file,
    ``change_box all final`` followed by nine numbers, is an illegal
    command (measured on 23Jun2022u1, where it aborts the process group at
    the first variable-cell step).

    No ``remap`` clause is sent either, and that is a choice, not an
    oversight: remap wraps the atoms into the new box and counts each
    crossing into their image flags, while this service states the flags
    itself beside the coordinates of the same step (_apply_image). Doing
    both would count every crossing twice.

    The tilt line is sent only when the domain can hold tilt factors:
    LAMMPS rejects a tilt change on a domain it keeps as orthogonal, and
    the cold path never states tilt factors for such a cell either.
    """
    def number(key):
        return f'{float(box[key]):.10f}'

    bounds = ' '.join(f'{axis} final {number(axis + "lo")} '
                      f'{number(axis + "hi")}'
                      for axis in ('x', 'y', 'z'))
    lines = [f'change_box all {bounds}']
    if triclinic:
        tilts = ' '.join(f'{key} final {number(key)}' for key in _TILT_ORDER)
        lines.append(f'change_box all {tilts}')
    return lines


def _image_commands(image: list, applied) -> list:
    """Spell the commands that move the instance's flags onto ``image``.

    ``applied`` is what the service believes the engine holds right now,
    or None when it has never been told. The first evaluation therefore
    restates every atom, per distinct target, instead of trusting what
    read_data derived from box containment: the two agree while every atom
    is inside the cell, and only the restated form is still right for an
    atom that crossed a face and came back. After that it is a diff, so a
    step where nothing crossed sends no command at all.

    Atoms bound for the same triplet are emitted together, as far as the
    selection grammar allows: ``set`` takes one value per command, not a
    value per atom.
    """
    targets = {}
    for index, row in enumerate(image):
        if applied is not None and list(applied[index]) == list(row):
            continue
        targets.setdefault(tuple(row), []).append(index + 1)
    return ['set atom %s image %d %d %d' % (spec, nx, ny, nz)
            for (nx, ny, nz), ids in sorted(targets.items())
            for spec in _id_ranges(ids)]


def _id_ranges(ids: list) -> list:
    """Spell an atom selection that ``set atom`` actually accepts.

    ``set`` parses its selection as one token through ``utils::bounds``, so
    it takes a single ID or a ``lo*hi`` range and nothing else; a
    space-separated list of IDs is read as a second keyword and aborts the
    job with "Illegal set command". A target group is therefore emitted as
    its maximal contiguous runs, which collapses the common case (every
    atom heading for the same triplet) into one command.
    """
    specs = []
    ordered = sorted(ids)
    start = previous = ordered[0]
    for current in ordered[1:]:
        if current == previous + 1:
            previous = current
            continue
        specs.append(_range_spec(start, previous))
        start = previous = current
    specs.append(_range_spec(start, previous))
    return specs


def _range_spec(start: int, end: int) -> str:
    return str(start) if start == end else '%d*%d' % (start, end)


def _normalize_extract_box(raw):
    """Turn either interface's extract_box() answer into nine true numbers.

    The order is the one the whole resident path uses: xlo xhi ylo yhi zlo
    zhi xy xz yz. The pybind11 interface returns the C function's seven
    parts, ``boxlo(3), boxhi(3), xy, yz, xz, periodicity(3), box_change``,
    where the tilt factors sit at the end of a flattened view and mean
    something else entirely, so it is matched by shape before any
    flattening happens. The older interface hands back an already flat
    sequence of bounds, origin and tilts.
    """
    if isinstance(raw, (list, tuple)) and len(raw) == 7:
        boxlo, boxhi, xy, yz, xz = raw[0], raw[1], raw[2], raw[3], raw[4]
        lo = [float(value) for value in _flatten(boxlo)]
        hi = [float(value) for value in _flatten(boxhi)]
        if len(lo) != 3 or len(hi) != 3:
            return None
        return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
                float(xy), float(xz), float(yz)]
    values = [float(value) for value in _flatten(raw)]
    if len(values) >= 12:
        return values[:6] + values[-3:]
    if len(values) >= 9:
        return values[:9]
    return None


# ----------------------------------------------------------------------
# Loop and process plumbing
# ----------------------------------------------------------------------

def _bcast(comm, value):
    """Broadcast *value* from rank 0, or pass it through when serial."""
    if comm is None:
        return value
    return comm.bcast(value, root=0)


def _owner_alive(pid: int) -> bool:
    """Whether process *pid* still exists."""
    if not pid or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _stop_reason(args):
    """Why the service should stop waiting for its client, or None.

    The one owner/idle rule both waits use: the accept loop before a
    client arrives, and the frame loop between requests. A client that
    went away (or was killed) is the only reason a service stops without
    being told, and the idle deadline bounds a silent one.
    """
    if not _owner_alive(args.owner_pid):
        return 'owner_exit'
    if time.monotonic() > args.idle_deadline:
        return 'idle_exit'
    return None


def _close_op(reason: str, detail: str = '') -> dict:
    """Build the internal CLOSE operation."""
    op = {'op': OP_CLOSE, 'id': 0, 'reason': reason, 'v': PROTOCOL_VERSION}
    if detail:
        op['detail'] = detail
    return op


def _read_operation(conn, args):
    """Read and validate the next request, or produce a control operation.

    Rejections happen here, before anything is broadcast: a request that
    never reached LAMMPS needs no unwinding, and answering it locally
    keeps rank 0's broadcast count at one per served operation.

    Returns
    -------
    tuple of (dict, dict or None)
        The operation to broadcast and, when set, the ERROR reply for the
        rejected request (nothing is broadcast for it).
    """
    deadline = time.monotonic() + max(READ_WAIT_S, OWNER_POLL_INTERVAL_S)
    while True:
        try:
            message = read_frame(conn, deadline=deadline)
        except LammpsServiceTimeout as exc:
            # Nothing arrived: the owner and the idle deadline decide
            # whether that is a reason to stop.
            reason = _stop_reason(args)
            if reason is not None:
                return _close_op(reason, f'{type(exc).__name__}'), None
            deadline = time.monotonic() + READ_WAIT_S
            continue
        except LammpsServiceDead as exc:
            return _close_op('client_closed', f'{type(exc).__name__}'), None
        except LammpsProtocolError as exc:
            # A frame that cannot be decoded is the client's defect. The
            # service answers it (session not usable) and stops this
            # connection: waiting on a stream whose framing is broken only
            # turns the defect into an idle timeout three seconds later.
            reply = build_error(for_op='unknown', request_id=0,
                                generation=0, kind='protocol',
                                stage='decode', session_usable=False,
                                message=str(exc))
            return _close_op('protocol', str(exc)), reply

        try:
            check_fields(message)
        except LammpsProtocolError as exc:
            reply = build_error(for_op=(message or {}).get('op', 'unknown')
                                if isinstance(message, dict) else 'unknown',
                                request_id=(message or {}).get('id', 0)
                                if isinstance(message, dict) else 0,
                                generation=0, kind='protocol',
                                stage='validate', session_usable=True,
                                message=str(exc))
            return None, reply
        except Exception as exc:                    # noqa: BLE001
            # Anything other than the protocol refusal is a defect in the
            # validator itself. It is reported as internal and the
            # connection stops: a validator that raises on a well-formed
            # message cannot be trusted with the next one.
            reply = build_error(for_op=(message or {}).get('op', 'unknown')
                                if isinstance(message, dict) else 'unknown',
                                request_id=(message or {}).get('id', 0)
                                if isinstance(message, dict) else 0,
                                generation=0, kind='internal',
                                stage='validate', session_usable=False,
                                message='validator failed: %s' % exc)
            return _close_op('internal', str(exc)), reply
        if message['op'] not in _REQUEST_OPS:
            reply = build_error(for_op=message['op'], request_id=message['id'],
                                generation=0, kind='protocol',
                                stage='validate', session_usable=True,
                                message='operation not accepted')
            return None, reply
        return message, None


def _dispatch(conn, engine, rank: int, op: dict) -> bool:
    """Perform one operation on this rank; return False to stop the service.

    The LAMMPS work runs on every rank in the same order; only the reply
    belongs to rank 0, which is also the only rank holding a socket.
    """
    kind = op['op']
    engine.generation += 1
    generation = engine.generation

    if kind == OP_CLOSE:
        engine.close()
        if rank == 0:
            _send(conn, build_result(for_op=OP_CLOSE,
                                     request_id=op.get('id', 0),
                                     generation=generation, closed=True))
        return False

    try:
        if kind == OP_HELLO:
            payload = build_result(for_op=OP_HELLO, request_id=op['id'],
                                   generation=generation,
                                   info=_service_info(engine))
        elif kind == OP_INIT:
            payload = build_result(for_op=OP_INIT, request_id=op['id'],
                                   generation=generation,
                                   fingerprint=op['fingerprint'],
                                   **engine.init(op))
        else:
            payload = build_result(for_op=OP_EVALUATE, request_id=op['id'],
                                   generation=generation,
                                   session=op['session'],
                                   fingerprint=op['fingerprint'],
                                   **engine.evaluate(op))
    except Exception as exc:                        # noqa: BLE001
        # Reached only where a rank can no longer be sure the others are
        # at the same point. Report if the socket still works, then let the
        # process group end: recovering would need a checkpointed geometry
        # and a fresh communicator, which is a different design.
        if rank == 0:
            # The phase is the operation, not a word in the message: a
            # failure while evaluating a step is a run failure, and a
            # failure while loading the topology is a load failure.
            stage = 'run' if kind == OP_EVALUATE else 'load'
            _send(conn, build_error(
                for_op=kind, request_id=op.get('id', 0),
                generation=generation,
                kind='eval_failed' if kind == OP_EVALUATE else 'init_failed',
                stage=stage, session_usable=False,
                message=f'{type(exc).__name__}: {exc}'))
        raise
    if rank == 0:
        _send(conn, payload)
    return True


def _service_info(engine) -> dict:
    """Describe this service, so the client checks before loading."""
    return {
        'protocol': PROTOCOL_VERSION,
        'lammps_version': engine.state.get('lammps_version', ''),
        'has_mpi_support': engine.state.get('has_mpi_support', False),
        'mpi4py': engine.state.get('mpi4py', False),
        'world_size': engine.state.get('world_size', 1),
        'nproc_expected': engine.state.get('nproc_expected', 1),
        'python': {'executable': sys.executable,
                   'version': '%d.%d.%d' % sys.version_info[:3]},
        'api_probe': engine.state.get('api_probe', {}),
    }


def _send(conn, message: dict) -> None:
    """Write one frame, tolerating a client that already left."""
    if conn is None:
        return
    try:
        conn.sendall(encode_frame(message))
    except OSError:
        pass


def _serve_rank0(engine, comm, args) -> int:
    """Accept one client, serve it until it stops, then leave.

    Rank 0 broadcasts exactly once per served operation, so a request it
    rejects before touching LAMMPS is answered without a broadcast: no
    collective had started, and the other ranks just wait for the next one.
    """
    try:
        listener = _bind(args.endpoint)
    except OSError as exc:
        print(f'cannot bind {args.endpoint}: {exc}', file=sys.stderr,
              flush=True)
        return 5
    print(f'{time.strftime("%H:%M:%S")} lammps service listening on '
          f'{args.endpoint} (world {engine.state.get("world_size")})',
          flush=True)

    # One service, one session: after CLOSE the process ends rather than
    # waiting for a second client, because a second backend would have no
    # way to tell what the first one loaded.
    status = 0
    try:
        conn = _accept_one(listener, args)
        if conn is not None:
            try:
                while True:
                    op, rejection = _read_operation(conn, args)
                    if rejection is not None:
                        _send(conn, rejection)
                        if op.get('op') == OP_CLOSE:
                            break
                        continue
                    if not _broadcast_and_dispatch(conn, engine, comm, op):
                        break
            except Exception as exc:                # noqa: BLE001
                print(f'service failed: {type(exc).__name__}: {exc}',
                      file=sys.stderr, flush=True)
                status = 6
            finally:
                conn.close()
    finally:
        # None is the workers' cue to leave their broadcast loop.
        _bcast(comm, None)
        _unlink(args.endpoint)
    return status


def _broadcast_and_dispatch(conn, engine, comm, op) -> bool:
    """Share one request with the other ranks, then perform it here."""
    return _dispatch(conn, engine, 0, _bcast(comm, op))


def _serve_worker(engine, comm, args) -> int:
    """Perform the operations rank 0 broadcasts.

    A worker never touches the socket, so it cannot act on a request rank 0
    turned down; None marks the service shutting down.
    """
    while True:
        op = comm.bcast(None, root=0)
        if op is None:
            return 0
        try:
            if not _dispatch(None, engine, 1, op):
                return 0
        except Exception:                           # noqa: BLE001
            # rank 0 already reported; this rank may sit behind in a
            # collective, so the group ends instead of resuming.
            return 6


def serve(args) -> int:
    """Create the LAMMPS instance and run this rank's loop.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    int
        Process exit status.
    """
    rank, comm, world, mpi4py = 0, None, 1, False
    if args.nproc > 1:
        MPI = _import_mpi()
        mpi4py = True
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world = comm.Get_size()
        if world != args.nproc:
            if rank == 0:
                # Each rank keeping its own singleton world is what a
                # launcher from a different MPI family does to a library
                # it did not start, so name the two sides here. PMI_SIZE
                # is set by MPICH/Intel MPI's hydra and ignored by Open
                # MPI; reading it beats squinting at a launcher banner.
                vendor = '/'.join(str(part) for part in MPI.get_vendor())
                foreign = [name for name in ('PMI_SIZE', 'MPICH_VERSION',
                                             'I_MPI_HYDRA_UUID')
                           if os.environ.get(name)]
                print(f'the MPI world has {world} ranks, the launcher asked '
                      f'for {args.nproc} (mpi4py vendor {vendor}; variables '
                      f'from another MPI: {", ".join(foreign) or "none"})',
                      file=sys.stderr, flush=True)
            return 4

    lammps = _import_native()
    instance = lammps.lammps(cmdargs=['-echo', 'none', '-log', 'none'],
                             **({} if comm is None else {'comm': comm}))
    adapter = _ApiAdapter(instance)
    engine = ResidentEngine(instance, adapter, state={
        # The SWIG interface publishes LAMMPS_VERSION; the pybind11 package
        # only publishes the numeric __version__ (e.g. '20220623'). Report
        # whichever the installed one offers, because this string is what a
        # reader of the log anchors a version question on.
        'lammps_version': (str(getattr(lammps, 'LAMMPS_VERSION', '')
                               or getattr(lammps, '__version__', ''))),
        'has_mpi_support': _has_mpi_support(lammps),
        'mpi4py': mpi4py,
        'world_size': world,
        'nproc_expected': args.nproc,
        'api_probe': adapter.probe(),
    })

    if rank == 0:
        return _serve_rank0(engine, comm, args)
    return _serve_worker(engine, comm, args)


def _accept_one(listener, args):
    """Wait for a client, giving up when the owner is gone or idle expired.

    select() rather than a blocking accept(): a service nobody connected
    to must still stop on its own once its owner disappears.
    """
    while True:
        if _stop_reason(args) is not None:
            return None
        ready, _, _ = select.select([listener], [], [], READ_WAIT_S)
        if ready:
            conn, _ = listener.accept()
            conn.settimeout(READ_WAIT_S)
            return conn


def _bind(endpoint: str):
    """Create and listen on the AF_UNIX endpoint.

    bind() failing with EADDRINUSE is the claim being taken, so two
    services can never share one path. The endpoint sits in the
    calculation directory, which a shared submission directory may make
    group-writable, so it is narrowed before it starts listening: on
    AF_UNIX a connect() needs write permission on the path, which the
    client has as its owner and nobody else keeps.
    """
    parent = os.path.dirname(endpoint)
    if parent:
        os.makedirs(parent, exist_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(endpoint)
        os.chmod(endpoint, 0o600)
    except OSError:
        listener.close()
        raise
    listener.listen(1)
    return listener


def _unlink(path: str) -> None:
    """Remove the endpoint inode, ignoring a missing file."""
    try:
        os.unlink(path)
    except OSError:
        pass


def parse_args(argv=None):
    """Parse the service command line."""
    parser = argparse.ArgumentParser(
        prog='lammps persistent server',
        description='Resident LAMMPS service of the lammps skill.')
    parser.add_argument('--endpoint', required=True,
                        help='AF_UNIX socket path to serve on')
    parser.add_argument('--session', default=None,
                        help='session directory to work in')
    parser.add_argument('--owner-pid', type=int, default=0,
                        help='process the watchdog follows')
    parser.add_argument('--nproc', type=int, default=1,
                        help='expected MPI rank count')
    parser.add_argument('--startup-timeout', type=float, default=120.0,
                        help=argparse.SUPPRESS)
    parser.add_argument('--idle-timeout', type=float, default=IDLE_TIMEOUT_S,
                        help='quit after this many seconds with no client')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point of the service process.

    The working directory is set before any native import: the script the
    client wrote loads its inputs by relative name, exactly as the
    subprocess path does after run_local chdirs into its calcdir.
    """
    args = parse_args(argv)
    if args.session:
        os.makedirs(args.session, exist_ok=True)
        os.chdir(args.session)
    args.idle_deadline = time.monotonic() + max(1.0, args.idle_timeout)
    try:
        return serve(args)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
