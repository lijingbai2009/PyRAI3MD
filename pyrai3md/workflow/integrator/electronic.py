"""
======================================================================
Multi-center electronic dispatch layer for surface hopping (docs
MULTICENTER_DEVELOPMENT.md, refactor R5).

This module inserts a thin dispatch layer between the surface_hopping
factory and the concrete algorithms (FSSH / GSH / LZSH). The algorithm
cores stay byte-for-byte unchanged: they keep reading one
PyRAIState-shaped object and keep writing back through update_system /
update_dynamics. The dispatch layer hands each center its own
CenterView -- a proxy that translates the center's per-center
electronic data (state.electronic[c], the R4 storage) into the
single-center interface the algorithm already expects, and captures
the algorithm's write-back so it lands on that center only.

Design rules (locked with the user, R5):
- The algorithm cores are NOT modified; only the way they obtain their
  input data changes (center id -> that center's own arrays).
- C=1 runs a TRANSPARENT binding: the view IS the real state, so the
  legacy single-center path stays bit-for-bit identical.
- Per-center data is served in a global-shaped array with a zero mask
  outside the center region: energy / SOC stay in the center's LOCAL
  state framing; gradient / NAC are scattered to the global atom axis
  (regions.project_*); coordinates follow the SAME zero-mask rule so a
  coordinate-driven kernel (GSH Zhu-Nakamura) reads the center's atoms
  only. The mask region is the center's high_indices, exactly the
  region the electronic provider was projected onto.
- Each center's velocity write-back is restricted to its own atoms, so
  one center's hop never changes another center's velocity (centers
  are disjoint).
- The RNG is not unified: determinism comes from the planning seed and
  the fixed center order, not from a shared random stream.

Sep 08 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...data.runtime.result import Result
from ..multiscale.regions import project_gradient, project_nac
from .fssh import FsshIntegrator
from .gsh import GshIntegrator
from .lzsh import LzshIntegrator


_ALGORITHMS = {
    'fssh': FsshIntegrator,
    'gsh': GshIntegrator,
    'lzsh': LzshIntegrator,
}


def _algorithm_class(method):
    """Algorithm integrator class of one surface hopping method.

    Parameters
    ----------
    method : str
        'fssh', 'gsh' or 'lzsh' (case-insensitive).

    Returns
    -------
    type
        The integrator class bound to the center view.

    Raises
    ------
    ValueError
        If the method has no dispatchable algorithm.
    """
    cls = _ALGORITHMS.get(str(method).lower())
    if cls is None:
        raise ValueError(
            "Unknown surface hopping method for electronic dispatch: "
            "%s" % (method,))
    return cls


# ----------------------------------------------------------------------
# Per-center data shaping (zero-mask + local/global framing)
# ----------------------------------------------------------------------

def _mask_atoms(field, high_indices, natom):
    """Zero-mask a global per-atom (natom, 3) field to one center.

    Atoms outside the center region are set to zero; an empty region
    keeps the whole system. Applied to the coordinates so a
    coordinate-driven kernel (GSH) reads the center's atoms only: the
    kernel uses coordinates strictly as differences (r - rpp, rp - r),
    so a consistent zero mask makes every out-of-region displacement
    vanish and the coupling localizes to the center, exactly like the
    zero-masked gradient / NAC.

    Parameters
    ----------
    field : array-like
        Global per-atom data, reshaped to (natom, 3).
    high_indices : list of int
        The center's 0-based atom indices (its projection region).
    natom : int
        Number of global atoms.

    Returns
    -------
    np.ndarray
        The masked (natom, 3) array (the input when the region is
        empty or the field is blank).
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = arr.reshape(-1, 3)
    if not high_indices:
        return arr
    masked = np.zeros((natom, 3), dtype=np.float64)
    idx = np.asarray(high_indices, dtype=int)
    masked[idx] = arr[idx]
    return masked


def _project_carrier(carrier, high_indices, natom):
    """Build the results dict served to one center's algorithm.

    energy / SOC are the center's LOCAL arrays, shape (S_c,) and
    (P_soc,); gradient / NAC are scattered to the global atom axis
    with a zero mask outside the center (regions.project_*), so the
    coupling the algorithm reads is nonzero only on the center's
    atoms. Keys the provider did not return stay absent (the cores
    read them through results.get and degrade gracefully).

    Parameters
    ----------
    carrier : dict or None
        One center's electronic arrays in LOCAL framing.
    high_indices : list of int
        The center's 0-based atom indices (its projection region).
    natom : int
        Number of global atoms.

    Returns
    -------
    dict
        {'energy', 'soc', 'gradient', 'nac'} subset ready for Result.
    """
    carrier = carrier or {}
    out = {}
    energy = carrier.get('energy')
    if energy is not None:
        out['energy'] = np.asarray(energy, dtype=np.float64)
    soc = carrier.get('soc')
    if soc is not None:
        out['soc'] = np.asarray(soc, dtype=np.float64)
    gradient = carrier.get('gradient')
    if gradient is not None:
        out['gradient'] = project_gradient(
            gradient, high_indices, natom)
    nac = carrier.get('nac')
    # An empty array is "no coupling rows", not a malformed (A_c, 3) one.
    # `nactype: ktdc` builds its couplings from the curvature and a
    # provider that reports none leaves `[]` on the carrier; projecting
    # that raised `Local NAC must have ... got (0,)` and killed
    # `multicenter_md` at its first hop (#466). A real shape error still
    # reaches project_nac and is refused there.
    if nac is not None and np.asarray(nac).size:
        out['nac'] = project_nac(nac, high_indices, natom)
    return out


# ----------------------------------------------------------------------
# CenterView sub-proxies (pmd.system / pmd.control / pmd)
# ----------------------------------------------------------------------

class _ParentView:
    """A view that reads through the CenterView it belongs to."""

    def __init__(self, parent):
        self._p = parent


class _SystemView(_ParentView):
    """pmd.system proxy exposing one center's electronic framing.

    The coupling pair maps and the spin list come from the center's
    ElectronicSpec (LOCAL state indices); the density matrix is the
    center's persisted coherence; the coordinates are zero-masked to
    the center region; the velocities stay the global matrix (the
    zero-masked NAC localizes any adjustment). Any other field falls
    through to the real system (no per-center framing).
    """

    @property
    def natom(self):
        return self._p._natom

    @property
    def nac_pair_map(self):
        return self._p._spec.electronic_spec.nac_pair_map

    @property
    def soc_pair_map(self):
        return self._p._spec.electronic_spec.soc_pair_map

    @property
    def state_spin(self):
        return self._p._spec.electronic_spec.state_spin

    @property
    def state_density(self):
        return self._p._electronic().get('density')

    @property
    def coordinates(self):
        return self._p.coordinates

    @property
    def velocities(self):
        return self._p.velocities

    @property
    def current_state(self):
        return self._p.current_state

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._p._state.pmd.system, name)


class _ControlView(_ParentView):
    """pmd.control proxy giving each center a unique dump title.

    Only the title is center-specific ('{title}_c{c}') so a verbose
    FSSH dump never collides across centers; every other control
    field (verbose included) passes through unchanged.
    """

    @property
    def verbose(self):
        return getattr(self._p._real_control, 'verbose', 2)

    @property
    def title(self):
        base = str(getattr(self._p._real_control, 'title',
                           'pyrai3md'))
        return '%s_c%d' % (base, self._p._c)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._p._real_control, name)


class _PmdView(_ParentView):
    """pmd proxy: per-center system, shared dynamics, no global result.

    dynamics is the whole-system configuration (substep, adjust,
    reflect, maxh, gap, deco, ...), shared by every center and passed
    through unchanged; results is empty because the per-center result
    is served through CenterView.result, never the global pmd.results.
    """

    @property
    def system(self):
        return self._p._system_view

    @property
    def dynamics(self):
        return self._p._state.pmd.dynamics

    @property
    def control(self):
        return self._p._control_view

    @property
    def results(self):
        return {}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._p._state.pmd, name)


# ----------------------------------------------------------------------
# CenterView: the single-center proxy handed to one algorithm instance
# ----------------------------------------------------------------------

class CenterView:
    """Single-center proxy of a PyRAIState for one algorithm instance.

    Reads translate the center's per-center electronic data into the
    single-center interface the unchanged FSSH/GSH/LZSH core expects
    (nstate -> S_c, current_state -> the center's active local state,
    result -> the center's projected arrays, history -> the center's
    reshaped snapshots). Writes are captured here, never applied to
    the parent state directly, so the owning ElectronicBinding lands
    them on this center only.

    Parameters
    ----------
    state : PyRAIState
        The global state (mutated in place across the trajectory).
    center_index : int
        The center this view proxies (contiguous, 0-based).
    """

    def __init__(self, state, center_index):
        self._state = state
        self._c = int(center_index)
        self._spec = state.center(self._c)
        self._high = list(self._spec.high_indices)
        self._natom = int(state.pmd.system.natom)
        self._real_control = getattr(state.pmd, 'control', None)
        self._system_view = _SystemView(self)
        self._control_view = (_ControlView(self)
                              if self._real_control is not None
                              else None)
        self._pmd_view = _PmdView(self)
        self._pending_system = {}
        self._pending_dynamics = {}

    # ---- lifecycle ----

    def refresh(self, state=None):
        """Re-point at the (mutated) state and clear captured writes.

        Called by the binding before each algorithm step so the view
        reads the current step and starts with an empty write-back
        buffer.
        """
        if state is not None:
            self._state = state
        self._pending_system = {}
        self._pending_dynamics = {}
        return self

    # ---- PyRAIState-shaped read interface ----

    @property
    def pmd(self):
        return self._pmd_view

    @property
    def runtime(self):
        return self._state.runtime

    @property
    def nstate(self):
        """The center's local state count S_c (never the global sum)."""
        return int(self._spec.nstate)

    @property
    def current_state(self):
        """The center's active LOCAL state (get_active_configuration)."""
        return int(self._state.get_active_configuration()[self._c])

    @property
    def timestep(self):
        return self._state.timestep

    @property
    def velocities(self):
        """Global velocity matrix (the zero-masked NAC localizes it)."""
        return self._state.velocities

    @property
    def mass_au(self):
        return self._state.mass_au

    @property
    def coordinates(self):
        """Global coordinates zero-masked to the center region."""
        return _mask_atoms(self._state.coordinates,
                           self._high, self._natom)

    @property
    def history_size(self):
        return self._state.history_size

    @property
    def result(self):
        """The center's projected electronic result (or None)."""
        out = _project_carrier(self._electronic(), self._high,
                               self._natom)
        return Result(data=out) if out else None

    def get_history_step(self, index=-1, copy=True):
        """Per-center reshape of one structured HistoryFrame.

        Returns the {'system', 'results'} snapshot the cores already
        read, rebuilt from the frame's structured fields: the center's
        electronic_outcome (projected) and the frame's nuclear sample
        (coordinates zero-masked to the center, velocities global).
        ``copy`` is accepted for interface parity; the reshape already
        builds a fresh dict.
        """
        frame = self._state.get_history_step(index, copy=False)
        return self._center_snapshot(frame)

    # ---- captured write interface (applied by the binding) ----

    def update_system(self, updates):
        self._pending_system.update(updates or {})

    def update_dynamics(self, updates):
        self._pending_dynamics.update(updates or {})

    # ---- internal helpers ----

    def _electronic(self):
        """This center's mutable electronic carrier."""
        return self._state.center_electronic(self._c)

    def _center_snapshot(self, frame):
        """Reshape one HistoryFrame into this center's snapshot."""
        if not frame:
            return None
        outcome = frame.get('electronic_outcome') or {}
        carrier = outcome.get(self._c)
        if carrier is None:
            carrier = outcome.get(str(self._c)) or {}
        results = _project_carrier(carrier, self._high, self._natom)
        # The legacy snapshot serves system['coordinates'] /
        # ['velocities'] as plain lists (pmd.model_dump()), and the
        # unchanged cores branch on them with ``if prev_velo:`` -- a
        # numpy array would raise the ambiguous-truth-value error. Keep
        # the list type so the reshaped frame is interface-identical.
        nuclear = frame.get('nuclear_sample') or {}
        system = {}
        coords = nuclear.get('R')
        if coords is not None:
            system['coordinates'] = _mask_atoms(
                coords, self._high, self._natom).tolist()
        velo = nuclear.get('V_pre')
        if velo is not None:
            system['velocities'] = np.asarray(
                velo, dtype=np.float64).tolist()
        return {'results': results, 'system': system}


# ----------------------------------------------------------------------
# ElectronicBinding: one center's algorithm + its write-back landing
# ----------------------------------------------------------------------

class ElectronicBinding:
    """One center's algorithm instance bound to its data view.

    A transparent binding (C=1) binds the algorithm directly to the
    real state, reproducing the legacy single-center path bit-for-bit.
    Otherwise the algorithm runs on a CenterView and the captured
    write-back is applied to the parent state on this center's own
    data only (density, active local state, region-restricted
    velocity).

    Parameters
    ----------
    method : str
        Algorithm selector ('fssh' / 'gsh' / 'lzsh').
    state : PyRAIState
        The global state.
    center_index : int
        The center this binding drives.
    transparent : bool
        True binds the algorithm to the real state (C=1).
    """

    def __init__(self, method, state, center_index, transparent):
        self.method = str(method).lower()
        self.center_index = int(center_index)
        self.transparent = bool(transparent)
        cls = _algorithm_class(self.method)
        if self.transparent:
            self.view = state
            self.algorithm = cls(state)
        else:
            self.view = CenterView(state, self.center_index)
            self.algorithm = cls(self.view)

    def run(self, state):
        """Run this center's algorithm on the current state.

        Returns
        -------
        PyRAIState
            The same state object (the cores mutate and return it).
        """
        if self.transparent:
            return self.algorithm(state)
        self.view.refresh(state)
        self.algorithm(self.view)
        self._apply(state)
        return state

    def _apply(self, state):
        """Land the captured write-back on this center only."""
        pend_sys = self.view._pending_system
        pend_dyn = self.view._pending_dynamics

        # Persisted coherence: the center's density matrix (the key is
        # part of the documented carrier layout; population is the
        # real diagonal of the density and is derived, not stored).
        density = pend_sys.get('state_density')
        if density is not None:
            state.set_center_electronic(
                self.center_index, density=density)

        # Post-hop active LOCAL state of this center.
        new_state = pend_sys.get('current_state')
        if new_state is not None:
            self._apply_active(state, int(new_state))

        # Adjusted velocity, restricted to this center's atoms.
        velocities = pend_sys.get('velocities')
        if velocities is not None:
            self._apply_velocity(state, velocities)

        # state_population is captured but not stored: it mirrors the
        # density diagonal and the per-center output writer derives it
        # (avoids extending the serialized carrier layout in R5).
        _ = pend_dyn

    def _apply_active(self, state, local_state):
        """Write this center's post-hop active local state."""
        surface = state.active_surface
        if not surface or surface.get('active_configuration') is None:
            return
        config = list(surface['active_configuration'])
        if 0 <= self.center_index < len(config):
            config[self.center_index] = int(local_state)
            surface['active_configuration'] = tuple(config)

    def _apply_velocity(self, state, velocities):
        """Write the adjusted velocity, restricted to this center.

        The core returns a global-shaped velocity; only this center's
        atoms take the adjustment, so an isotropic rescaling (LZSH /
        GSH intersystem crossing) can never touch another center's
        velocity. With a zero-masked NAC the NAC-direction adjustment
        is already local; this mask makes the guarantee unconditional.
        """
        new_v = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
        high = self.view._high
        cur = np.asarray(state.velocities, dtype=np.float64)
        cur = cur.reshape(-1, 3) if cur.size else cur
        if not high or new_v.shape != cur.shape:
            state.update_system({'velocities': new_v.tolist()})
            return
        out = cur.copy()
        idx = np.asarray(high, dtype=int)
        out[idx] = new_v[idx]
        state.update_system({'velocities': out.tolist()})


# ----------------------------------------------------------------------
# ElectronicStepper: the multi-center surface hopping dispatcher
# ----------------------------------------------------------------------

class ElectronicStepper:
    """Multi-center surface hopping dispatcher (state -> state).

    Initialized once per trajectory by the surface_hopping factory. It
    builds one ElectronicBinding per center (a single transparent
    binding for C=1) and, on every timestep, runs the centers in their
    fixed declared order. Because each center's write-back lands on the
    state before the next center refreshes its view, the centers form a
    sequential event ledger: a later center reads the earlier centers'
    post-event velocities and active states (docs R6, r6d). Each
    center's velocity adjustment is masked to its own atoms, so the
    adjustments stay independent across the disjoint centers; that an
    isotropic hop rescales by the global kinetic energy is a documented
    property of running the unchanged core on the global velocity, and
    a center-local kinetic pool is a separate physical model deferred
    to an independent algorithm change (docs 10.3). The algorithm cores
    are unchanged; the dispatch only decides which center's data each
    one reads and where its write-back lands. The returned object is
    the same callable the workflow already chains as
    ``state = surfacehop(state)``.

    Parameters
    ----------
    method : str
        Algorithm selector ('fssh' / 'gsh' / 'lzsh').
    state : PyRAIState
        State carrying the trajectory configuration and (for C>1) the
        bound partition plan; consumed at initialization only.
    """

    def __init__(self, method, state):
        self.method = str(method).lower()
        n_center = self._resolve_n_center(state)
        self._n_center = n_center
        self._transparent = n_center <= 1
        if self._transparent:
            self._bindings = [ElectronicBinding(
                self.method, state, 0, transparent=True)]
        else:
            self._bindings = [ElectronicBinding(
                self.method, state, c, transparent=False)
                for c in range(n_center)]

    @staticmethod
    def _resolve_n_center(state):
        """Center count C of the state (1 when no plan is bound)."""
        n_center = getattr(state, 'n_center', 1)
        try:
            n_center = int(n_center)
        except (TypeError, ValueError):
            return 1
        return max(n_center, 1)

    @property
    def n_center(self):
        """Number of centers this stepper dispatches."""
        return self._n_center

    @property
    def transparent(self):
        """True when the single-center legacy path is in force."""
        return self._transparent

    def binding(self, c):
        """The ElectronicBinding of center c."""
        return self._bindings[int(c)]

    def __call__(self, state):
        """Run every center's algorithm once, in declared order."""
        for binding in self._bindings:
            binding.run(state)
        return state
