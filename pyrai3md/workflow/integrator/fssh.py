"""
======================================================================
This module provides the fewest switches surface hopping (FSSH)
main flow of the NAMD workflow (dev/methods/FSSH_DEVELOPMENT.md).

The algorithm keeps the strict two-level structure mandated by the
design document:

    FSSH -> function -> numpy / numba

Every mathematical function has one NumPy reference implementation
(fssh_numpy.py) and one optional Numba implementation
(fssh_numba.py); the implementations are selected once at
initialization from the &md fssh_* keywords (design document
section 5), so the high-frequency substep loop never branches on
the backend. No backend/registry/dispatcher abstraction exists by
design.

The main flow is the FsshIntegrator class: it is initialized once
per trajectory by the surface_hopping factory (surface_hopping.py)
and called once per timestep. Initialization parses the static
trajectory configuration (dynamics keywords, coupling pair maps,
kernel selection) exactly once; the integrator never caches
trajectory data. Every step reads the current data through the
PyRAIState official interfaces (system.state_density, state.result,
the read-only history view) and writes back through
state.update_system() and state.update_dynamics(), so the data
ownership stays with the main framework.

The integrator itself owns:
    state management        density matrix persistence (system
                            state_density, [re, im] pairs) and the
                            state_population bookkeeping (the real
                            diagonal of A)
    hopping decision        per-substep fewest-switches sampling
                            with an independent random number
    velocity adjustment     legacy adjust/reflect semantics along
                            the NAC vector, isotropic rescaling when
                            no NAC vector is available
    numerical protection    pair-level gap validity (identity-order
                            rule for NAC pairs, degeneracy-only
                            rule for SOC/ISC pairs), population
                            diagnostics (Phase 6 of the design)
    debug dump              control verbose >= 4 persists the
                            per-step intermediate data (A, D, B
                            matrices and the per-substep g
                            probabilities) into {title}.fssh next
                            to the trajectory output files

Aug 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import time
import warnings

import numpy as np

from ...utilities.units import CM_TO_HARTREE, EV_TO_HARTREE
from .fssh_numpy import (
    dPdt_numpy, rk4_numpy, matB_numpy, ktdc_numpy, decoherence_numpy,
    GAP_CUTOFF, state_gap,
)
from .fssh_numba import (
    NUMBA_AVAILABLE,
    dPdt_numba, rk4_numba, matB_numba, ktdc_numba, decoherence_numba,
)
from .fssh_profile import FsshProfiler, warm_up
from .tsh_helper import (reflect_velo, adjust_velo, adjust_velo_isotropic,
                         energy_of as _energy_of)

# Steps required before hopping starts (current step needs E(t),
# E(t-dt) and E(t-2dt); matches the legacy iter < 4 warm-up with the
# history updated before surface hopping is called).
WARMUP_STEPS = 4

# Header of the {title}.fssh debug dump (control verbose >= 4).
FSSH_DUMP_HEADER = (
    '# PyRAI3MD FSSH debug dump format 1\n'
    '# One block per propagated md step: the step index (propagated\n'
    '# steps only, warm-up steps excluded), time = index * dt, the\n'
    '# active state before -> after the step, the final density\n'
    '# matrix A (columns: re im interleaved per row), the coupling\n'
    '# matrix D (same layout), the accumulated hopping matrix B and\n'
    '# the per-substep hopping probability vectors g (one row per\n'
    '# electronic substep).\n'
)


def _select_kernels(dynamics):
    """Bind the numpy/numba implementation of every FSSH kernel.

    Called once per integrator instance (design document section 5:
    the functions are bound at initialization, the substep loop never
    re-selects); the &md fssh_* keywords choose the implementation
    per function ('numpy' is the reference default). Requesting
    numba without the optional dependency warns once and falls back
    to the NumPy reference.

    Parameters
    ----------
    dynamics : Dynamics
        The pmd.dynamics submodel carrying the fssh_* keywords.

    Returns
    -------
    (dict, dict)
        ({label: callable}, {label: 'numpy' or 'numba'}).
    """
    choices = {
        'dPdt': str(getattr(dynamics, 'fssh_dpdt', 'numpy')).lower(),
        'rk4': str(getattr(dynamics, 'fssh_rk4', 'numpy')).lower(),
        'matB': str(getattr(dynamics, 'fssh_matb', 'numpy')).lower(),
        'ktdc': str(getattr(dynamics, 'fssh_ktdc', 'numpy')).lower(),
        'decoherence': str(getattr(dynamics, 'fssh_decoherence',
                                   'numpy')).lower(),
    }
    numba_kernels = {'dPdt': dPdt_numba, 'rk4': rk4_numba,
                     'matB': matB_numba, 'ktdc': ktdc_numba,
                     'decoherence': decoherence_numba}
    numpy_kernels = {'dPdt': dPdt_numpy, 'rk4': rk4_numpy,
                     'matB': matB_numpy, 'ktdc': ktdc_numpy,
                     'decoherence': decoherence_numpy}

    kernels = {}
    impls = {}
    for label, choice in choices.items():
        if choice == 'numba' and NUMBA_AVAILABLE:
            kernels[label] = numba_kernels[label]
            impls[label] = 'numba'
        else:
            if choice == 'numba':
                warnings.warn(
                    "FSSH kernel '%s' requested numba but Numba is "
                    "not installed; falling back to the numpy "
                    "reference." % label)
            elif choice != 'numpy':
                warnings.warn(
                    "FSSH kernel '%s' got unknown implementation "
                    "'%s'; using the numpy reference."
                    % (label, choice))
            kernels[label] = numpy_kernels[label]
            impls[label] = 'numpy'
    return kernels, impls


# ---- Density matrix serialization (system.state_density) ----

def _matrix_text(matrix):
    """Render a real matrix as fixed-width exponential rows.

    Helper of the {title}.fssh debug dump (control verbose >= 4);
    never called when the dump is disabled.
    """
    rows = []
    for row in np.asarray(matrix, dtype=np.float64):
        rows.append('  '.join('%20.12e' % float(v) for v in row))
    return '\n'.join(rows)


def _complex_matrix_text(matrix):
    """Render a complex matrix with interleaved re/im columns.

    Each output row carries re im pairs of the matrix row, so the
    dump stays column-aligned with the real matrices.
    """
    rows = []
    for row in np.asarray(matrix, dtype=np.complex128):
        cells = []
        for value in row:
            cells.append('%20.12e%20.12e' % (value.real, value.imag))
        rows.append('  '.join(cells))
    return '\n'.join(rows)


def _density_to_list(A):
    """Convert a complex density matrix to [re, im] nested lists.

    Vectorized through a single numpy stack + tolist (the hot-loop
    write-back path on every timestep).
    """
    return np.stack([np.real(A), np.imag(A)], axis=-1).tolist()


def _density_from_list(data):
    """Convert [re, im] nested lists to a complex density matrix.

    Returns
    -------
    np.ndarray or None
        Complex matrix, or None when the stored data is empty or
        malformed.
    """
    try:
        array = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 3 or array.shape[2] != 2 or array.shape[0] == 0:
        return None
    if array.shape[0] != array.shape[1]:
        return None
    return array[..., 0] + 1j * array[..., 1]


def _initial_density(nstate, current_state, population):
    """Seed the density matrix from the state_population keyword.

    The state_population entries are the initial real diagonal of A
    (user-defined convention); a population list that does not match
    nstate falls back to unit occupation of the active state.

    Parameters
    ----------
    nstate : int
        Number of electronic states.
    current_state : int
        Active state index (0-based).
    population : list of float
        The &md state_population keyword value.

    Returns
    -------
    np.ndarray
        Initial density matrix (nstate, nstate), complex.
    """
    A = np.zeros((nstate, nstate), dtype=np.complex128)
    population = np.asarray(population, dtype=np.float64)
    if population.size == nstate:
        for k in range(nstate):
            A[k, k] = population[k]
    elif 0 <= current_state < nstate:
        A[current_state, current_state] = 1.0
    return A


# ---- Velocity adjustment (shared layer, integrator/tsh_helper.py) ----


def _lookup_nac(nac_index, nac_values, pair):
    """NAC vector of a coupling pair, or None (precomputed index).

    Parameters
    ----------
    nac_index : dict
        {(s1, s2): row index} precomputed once at initialization.
    nac_values : array-like or None
        NAC vectors aligned with the NAC pair list.
    pair : tuple
        Sorted 0-based (s1, s2) pair of the hop.

    Returns
    -------
    np.ndarray or None
        NAC vector, or None when unavailable (kTDC runs, uncovered
        pairs or missing data).
    """
    if nac_values is None:
        return None
    if pair not in nac_index:
        return None
    nac_array = np.asarray(nac_values)
    idx = nac_index[pair]
    if idx >= nac_array.shape[0]:
        return None
    return np.asarray(nac_array[idx], dtype=np.float64)


def _adjust_for_hop(energy, from_state, to_state, velocities, mass_au,
                    nac, adjust, reflect):
    """Velocity adjustment of one hop candidate.

    Uses the NAC-direction adjustment when the pair's NAC vector is
    available (legacy adjust/reflect semantics) and the isotropic
    rescaling policy when it is not (design document section 18).

    Returns
    -------
    (np.ndarray, int)
        Adjusted velocity and the frustrated flag.
    """
    if nac is not None and nac.shape == velocities.shape:
        return adjust_velo(energy[from_state], energy[to_state],
                           velocities, mass_au, nac, adjust, reflect)
    return adjust_velo_isotropic(energy[from_state], energy[to_state],
                                 velocities, mass_au)


# ---- Matrix assembly from results and history ----

def _coupling_pairs(pair_map):
    """Filter a pair map down to valid 0-based [i, j] entries."""
    pairs = []
    for pair in pair_map or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            pairs.append((int(pair[0]), int(pair[1])))
    return pairs


def _build_hamiltonian(energy, soc_values, soc_pairs, nstate):
    """Assemble the model Hamiltonian (diagonal E + SOC off-diagonal).

    Parameters
    ----------
    energy : np.ndarray
        State energies (nstate,) in Hartree.
    soc_values : np.ndarray or None
        Spin-orbit couplings aligned with soc_pairs (cm-1).
    soc_pairs : list of tuple
        0-based SOC pair list.
    nstate : int
        Number of electronic states.

    Returns
    -------
    np.ndarray
        Hamiltonian (nstate, nstate), complex.
    """
    H = np.diag(np.asarray(energy, dtype=np.float64)).astype(
        np.complex128)
    if soc_values is not None:
        soc = np.asarray(soc_values, dtype=np.float64).reshape(-1)
        for index, (s1, s2) in enumerate(soc_pairs):
            if index >= soc.size or not (0 <= s1 < nstate
                                         and 0 <= s2 < nstate):
                continue
            socme = soc[index] * CM_TO_HARTREE
            H[s1, s2] = socme
            H[s2, s1] = socme
    return H


def _pair_gap_valid(energy, pair):
    """NAC-pair gap validity in PES identity order (Phase 6).

    gap = E[j] - E[i] with (i, j) in identity order. A degenerate
    gap (|gap| <= cutoff) or an inverted gap (gap < 0, NN state
    identity differs from the instantaneous energy order) invalidates
    the pair for the current step only; energies are never resorted.
    The identity-order rule protects the NAC matrix assembly, where
    the coupling sign follows the (E_j - E_i) convention.

    Returns
    -------
    bool
        True when the pair may participate in the current step.
    """
    s1, s2 = pair
    gap = energy[s2] - energy[s1]
    if gap < 0 or abs(gap) <= GAP_CUTOFF:
        return False
    return True


def _pair_gap_valid_soc(energy, pair):
    """SOC-pair gap validity for intersystem crossing hops.

    SOC pairs connect states of different spin whose PES identities
    are independent (a T1-below-S1 crossing region is a normal ISC
    scenario, not an identity inversion), so only a degenerate gap
    (|gap| <= cutoff) invalidates the pair; the energy-order
    inversion check of the NAC pairs does not apply.

    Returns
    -------
    bool
        True when the pair may participate in the current step.
    """
    s1, s2 = pair
    gap = energy[s2] - energy[s1]
    if abs(gap) <= GAP_CUTOFF:
        return False
    return True


def _pair_validity(energy, pairs, nstate, kind='nac'):
    """Pair-level gap validity map of one step.

    Parameters
    ----------
    kind : str
        'nac' applies the identity-order rule (degenerate or
        inverted gaps invalidate); 'soc' applies the ISC rule
        (degenerate gaps only).
    """
    check = _pair_gap_valid if kind == 'nac' else _pair_gap_valid_soc
    valid = {}
    for pair in pairs:
        if 0 <= pair[0] < nstate and 0 <= pair[1] < nstate:
            valid[pair] = check(energy, pair)
    return valid


def _build_nac_matrix(E, Ep, Epp, V, nac_values, nac_index, valid,
                      nstate, dt, gap_au, nactype, ktdc_kernel):
    """Assemble the nonadiabatic coupling matrix D of the step.

    Only pair-valid entries are filled (design document sections 10
    and 11): invalid pairs keep D at zero, so neither the electronic
    propagation nor the hopping sampling can involve them this step.

    In 'nac' mode the stored NAC vector is the CI-derivative
    quantity <i| dH/dR |j> = d_ij * (E_j - E_i) shared by every
    skill (molcas/bagel/qdm convention); the projection onto the
    nuclear velocity divides by the literature gap (E_j - E_i) to
    recover the standard derivative coupling d_ij.

    Parameters
    ----------
    E, Ep, Epp : np.ndarray
        State energies at t, t-dt and t-2dt (Hartree). Epp is the
        oldest point of the kTDC curvature window: for the previous-
        step reconstruction the caller passes the energies at t-dt,
        t-2dt and t-3dt in these slots.
    V : np.ndarray or None
        Current nuclear velocity (natom, 3), Bohr/au ('nac' mode).
    nac_values : np.ndarray or None
        NAC vectors <i| dH/dR |j> aligned with the NAC pair list
        ('nac' mode).
    nac_index : dict
        {(s1, s2): row index} precomputed once at initialization.
    valid : dict
        {(s1, s2): bool} pair validity of the current step.
    nstate : int
        Number of electronic states.
    dt : float
        Full timestep (au).
    gap_au : float
        Baeck-An gap threshold in Hartree.
    nactype : str
        'nac' (V . d / gap projection) or 'ktdc' (curvature-driven).
    ktdc_kernel : callable
        Selected kTDC kernel implementation.

    Returns
    -------
    np.ndarray
        Coupling matrix D (nstate, nstate), complex.
    """
    D = np.zeros((nstate, nstate), dtype=np.complex128)
    for (s1, s2) in nac_index:
        if not (0 <= s1 < nstate and 0 <= s2 < nstate):
            continue
        if not valid.get((s1, s2), False):
            continue
        if nactype == 'nac':
            if nac_values is None or V is None:
                continue
            idx = nac_index[(s1, s2)]
            nac = np.asarray(nac_values)
            if idx >= nac.shape[0]:
                continue
            nacme = (np.sum(V * nac[idx])
                     / state_gap(E, s1, s2))
        else:
            nacme = ktdc_kernel(s1, s2, E, Ep, Epp, dt, gap_au)
        D[s1, s2] = nacme
        D[s2, s1] = -D[s1, s2]
    return D


def _history_results(snapshot):
    """Extract the results dict of one history snapshot."""
    if not snapshot:
        return {}
    return snapshot.get('results') or {}



# ---- Diagnostics ----

def _diagnose_population(A, state):
    """Population numerical-error diagnostics (Phase 6, sections 15/16).

    Detects out-of-range populations and a vanishing active-state
    population; reports only, never clips or resets (the values keep
    propagating unchanged).
    """
    population = np.diag(np.real(A))
    for k, value in enumerate(population):
        if value < 0 or value > 1:
            warnings.warn(
                "FSSH population of state %d is %.6e (outside "
                "[0, 1]); kept as-is for propagation." % (k, value))
    if abs(population[state]) < 1e-8:
        warnings.warn(
            "FSSH active-state population of state %d is nearly "
            "zero (%.3e); diagnostic only, no reset applied."
            % (state, population[state]))


# ---- Main entry ----

class FsshIntegrator:
    """Fewest switches surface hopping integrator (one trajectory).

    Initialized once per trajectory by the surface_hopping factory:
    parses the static &md configuration (dynamics keywords, coupling
    pair maps, kernel implementation selection) exactly once and
    pre-computes the per-trajectory lookup structures. The instance
    never caches trajectory data; every call reads the current step
    through the PyRAIState official interfaces (system.state_density,
    state.result, the read-only history view) and writes back through
    state.update_system() / state.update_dynamics().

    Parameters
    ----------
    state : PyRAIState
        State carrying the trajectory configuration (pmd.dynamics
        and the system coupling pair maps); unchanged by the
        initialization.

    Raises
    ------
    ValueError
        If the state lacks a pmd.dynamics configuration.
    """

    def __init__(self, state):
        dynamics = getattr(getattr(state, 'pmd', None), 'dynamics',
                           None)
        if state is None or state.pmd is None or dynamics is None:
            raise ValueError(
                "FsshIntegrator requires a state with pmd.dynamics")

        system = state.pmd.system

        # ---- Static trajectory configuration (parsed once) ----
        self.nstate = int(state.nstate)
        self.nac_pairs = _coupling_pairs(system.nac_pair_map)
        self.soc_pairs = _coupling_pairs(system.soc_pair_map)
        self.nac_index = {pair: idx
                          for idx, pair in enumerate(self.nac_pairs)}
        # Unified hopping pair set: NAC pairs plus the SOC pairs not
        # already covered. Both pair types enter the same cumulative
        # energy-ordered sampling (no per-multiplicity branching).
        self.hop_pairs = list(self.nac_pairs)
        self.pair_kind = {pair: 'nac' for pair in self.nac_pairs}
        for pair in self.soc_pairs:
            if pair not in self.pair_kind:
                self.hop_pairs.append(pair)
                self.pair_kind[pair] = 'soc'

        self.substep = max(int(dynamics.substep), 1)
        self.timestep = float(state.timestep)
        self.delt = self.timestep / self.substep
        self.nactype = str(dynamics.nactype).lower()
        if self.nactype not in ('nac', 'ktdc'):
            raise ValueError(
                "Invalid nactype %r: expected one of ('nac', 'ktdc')"
                % dynamics.nactype)
        self.gap_au = float(dynamics.gap) * EV_TO_HARTREE
        self.use_rk4 = str(dynamics.rk4) != '0'
        # Electronic substep integration mode only (0: reset the B
        # matrix every substep, non-zero: accumulate B across
        # substeps). Debug force-hopping is isolated in fssh_force_hop.
        self.integrate = int(dynamics.integrate)
        self.force_hop = int(dynamics.fssh_force_hop) != 0
        self.adjust = int(dynamics.adjust)
        self.reflect = int(dynamics.reflect)
        self.maxh = int(dynamics.maxh)
        deco_text = str(dynamics.deco).upper()
        self.use_deco = deco_text != 'OFF'
        self.deco = float(deco_text) if self.use_deco else 0.0
        self.population_seed = list(dynamics.state_population)

        # ---- Debug dump (control verbose >= 4) ----
        # The per-step intermediate data (A, D, B, g) is persisted
        # into {title}.fssh next to the trajectory output files;
        # dispatched trajectories run in their own working
        # directories, so every trajectory owns its dump file.
        control = getattr(state.pmd, 'control', None)
        verbose = int(getattr(control, 'verbose', 2)) \
            if control is not None else 2
        title = str(getattr(control, 'title', 'pyrai3md')) \
            if control is not None else 'pyrai3md'
        self.dump_path = None
        self.dump_steps = 0
        if verbose >= 4:
            self.dump_path = os.path.join(
                os.getcwd(), '%s.fssh' % title)
            if not os.path.exists(self.dump_path):
                with open(self.dump_path, 'w',
                          encoding='utf-8') as dump:
                    dump.write(FSSH_DUMP_HEADER)

        # ---- One-time kernel selection (design document section 5)
        self.kernels, self.kernel_impls = _select_kernels(dynamics)
        self.profile_enabled = bool(getattr(dynamics, 'fssh_profile',
                                            0))
        self.profiler = FsshProfiler()
        self.profiler.set_implementations({
            'dPdt': self.kernel_impls['dPdt'],
            'RK4': self.kernel_impls['rk4'],
            'matB': self.kernel_impls['matB'],
            'kTDC': self.kernel_impls['ktdc'],
            'decoherence': self.kernel_impls['decoherence'],
        })
        warm_up({label: (kernel, self.kernel_impls[label] == 'numba')
                 for label, kernel in self.kernels.items()})

        # Profiling wraps are applied once here (not per call), so
        # the hot loop carries no per-call wrapping overhead.
        self.kernel_dPdt = self.kernels['dPdt']
        self.kernel_rk4 = self.kernels['rk4']
        self.kernel_matB = self.kernels['matB']
        self.kernel_ktdc = self.kernels['ktdc']
        self.kernel_deco = self.kernels['decoherence']
        if self.profile_enabled:
            self.kernel_dPdt = self.profiler.wrap(
                'dPdt', self.kernel_dPdt)
            self.kernel_rk4 = self.profiler.wrap(
                'RK4', self.kernel_rk4)
            self.kernel_matB = self.profiler.wrap(
                'matB', self.kernel_matB)
            self.kernel_ktdc = self.profiler.wrap(
                'kTDC', self.kernel_ktdc)
            self.kernel_deco = self.profiler.wrap(
                'decoherence', self.kernel_deco)
        self.profile_calls = 0

    def __call__(self, state):
        """Run one FSSH timestep (state -> state).

        Propagates the electronic density matrix across the
        electronic substeps of the current timestep, evaluates the
        fewest-switches hopping probabilities (an independent random
        number per substep), performs the hop through the official
        state update path and persists the updated density matrix
        and populations.

        The first WARMUP_STEPS steps only seed the density matrix
        from state_population (insufficient history), mirroring the
        legacy iter < 4 warm-up.

        Parameters
        ----------
        state : PyRAIState
            State with the recorded timestep (history updated) and
            the current electronic properties.

        Returns
        -------
        PyRAIState
            Same state object; a hop updates system.current_state
            and system.velocities inside the algorithm.
        """
        if state is None or state.pmd is None:
            return state

        nstate = self.nstate
        nac_pairs = self.nac_pairs
        soc_pairs = self.soc_pairs
        nac_index = self.nac_index
        hop_pairs = self.hop_pairs
        substep = self.substep
        delt = self.delt

        # ---- Read the step data through the official interfaces ----
        current_state = int(state.current_state)
        if nstate < 1 or not 0 <= current_state < nstate:
            return state

        system = state.pmd.system

        result = state.result
        results = result if result is not None else {}
        energy = _energy_of(results, nstate) if result is not None \
            else None
        if energy is None:
            return state

        # Load (or seed) the persisted density matrix.
        A = _density_from_list(system.state_density)
        if A is None or A.shape != (nstate, nstate):
            A = _initial_density(nstate, current_state,
                                 self.population_seed)

        # Warm-up: insufficient history for the kTDC curvature and
        # the previous-step H/D reconstruction; seed A and skip
        # hopping.
        if state.history_size < WARMUP_STEPS:
            state.update_system({'state_density': _density_to_list(A)})
            state.update_dynamics(
                {'state_population': [float(p)
                                      for p in np.diag(np.real(A))]})
            return state

        # ---- Assemble the current-step matrices ----
        soc_values = results.get('soc') if result is not None else None
        Ht = _build_hamiltonian(energy, soc_values, soc_pairs, nstate)

        # Pair-level gap validity of the current step. valid_nac
        # (identity-order rule) gates the NAC matrix assembly; the
        # sampling gate uses the pair-type rules so SOC pairs may
        # hop across spin manifolds (ISC) without the identity-order
        # inversion exclusion.
        valid_nac = _pair_validity(energy, nac_pairs, nstate,
                                   kind='nac')
        valid_hop = {}
        for pair in hop_pairs:
            kind = self.pair_kind.get(pair, 'nac')
            if 0 <= pair[0] < nstate and 0 <= pair[1] < nstate:
                check = _pair_gap_valid if kind == 'nac' \
                    else _pair_gap_valid_soc
                valid_hop[pair] = check(energy, pair)

        velocities = np.asarray(state.velocities, dtype=np.float64)
        V = velocities if velocities.size > 0 else None
        nac_values = (results.get('unbiased_nac')
                      if result is not None else None)
        if nac_values is None:
            nac_values = results.get('nac') if result is not None else None

        # Read-only history views (no per-step snapshot deep copy).
        prev_snap = state.get_history_step(-1, copy=False)
        previous = _history_results(prev_snap)
        before = _history_results(state.get_history_step(
            -2, copy=False))
        Ep = _energy_of(previous, nstate)
        Epp = _energy_of(before, nstate)
        # E(t-3dt) feeds the kTDC curvature window of the
        # previous-step D reconstruction (legacy keeps the computed
        # last_d; here the matrix is rebuilt from the history).
        earlier = _history_results(state.get_history_step(
            -3, copy=False))
        Eppp = _energy_of(earlier, nstate)
        if Ep is None or Epp is None:
            # Recorded steps lack energies: keep the propagation
            # well-defined with zero curvature data.
            Ep = energy.copy()
            Epp = energy.copy()
        if Eppp is None:
            Eppp = Epp.copy()

        Dt = _build_nac_matrix(energy, Ep, Epp, V, nac_values,
                               nac_index, valid_nac, nstate,
                               self.timestep, self.gap_au,
                               self.nactype, self.kernel_ktdc)

        # ---- Reconstruct the previous-step H/D from the history ----
        Ep_h = _build_hamiltonian(
            Ep, previous.get('soc'), soc_pairs, nstate)
        # The previous velocity lives in the system snapshot of the
        # previous step (the history dumps the full pmd state).
        prev_system = (prev_snap or {}).get('system') or {}
        prev_velo = prev_system.get('velocities')
        Vp = (np.asarray(prev_velo, dtype=np.float64).reshape(-1, 3)
              if prev_velo else None)
        valid_prev = _pair_validity(Ep, nac_pairs, nstate, kind='nac')
        Dp = _build_nac_matrix(Ep, Epp, Eppp, Vp, previous.get('nac'),
                               nac_index, valid_prev, nstate,
                               self.timestep, self.gap_au,
                               self.nactype, self.kernel_ktdc)

        dHdt = (Ht - Ep_h) / substep
        dDdt = (Dt - Dp) / substep

        # ---- Electronic substep propagation and hopping ----
        H = Ep_h.copy()
        D = Dp.copy()
        B = np.zeros((nstate, nstate))
        stateindex = np.argsort(energy)

        mass_au = state.mass_au if velocities.size > 0 else None
        ekin = float(state.runtime.kinetic_energy) \
            if state.runtime is not None else 0.0
        if ekin <= 0:
            if mass_au is not None:
                ekin = float(np.sum(0.5 * mass_au[:, None]
                                    * velocities ** 2))
        # OpenMolcas tully.F90 clamps a non-positive kinetic energy
        # to 1e-5 au before the tau = |1/dE|(1 + deco/ekin) ratio;
        # keep the same floor so low-kinetic-energy regions do not
        # over-damp the coherences.
        ekin = max(ekin, 1e-5)

        # One draw per step consumed in order: the array sample uses
        # the same random stream as the per-substep scalar draws.
        z_steps = np.random.uniform(0, 1, substep)

        profile_hopping = self.profile_enabled
        dump_enabled = self.dump_path is not None
        g_history = []
        active_state = current_state
        for i in range(substep):
            if self.integrate == 0:
                B = np.zeros((nstate, nstate))
            g = np.zeros(nstate)

            H = H + dHdt
            D = D + dDdt

            if self.use_rk4:
                dAdt = self.kernel_rk4(A, H, D, dHdt, dDdt, delt)
            else:
                dAdt = self.kernel_dPdt(A, H, D)
            A = A + dAdt * delt

            hop_start = time.perf_counter() if profile_hopping else 0.0
            dB = self.kernel_matB(A, H, D)
            B = B + dB
            for j in range(nstate):
                if j == active_state:
                    continue
                pair = (min(j, active_state), max(j, active_state))
                if not valid_hop.get(pair, False):
                    continue
                # Legacy semantics: g vanishes for a non-positive
                # active population (the max(0, ...) kills the
                # negative ratio); only an exactly zero population is
                # regularized against division by zero.
                Aii = np.real(A[active_state, active_state])
                if Aii == 0:
                    Aii = GAP_CUTOFF
                g[j] = max(0.0, B[j, active_state] * delt / Aii)

            if dump_enabled:
                g_history.append(g.copy())

            z = z_steps[i]
            if self.force_hop:
                z = 0
            gsum = 0.0
            event = 0
            new_state = active_state
            for j in range(nstate):
                gsum += g[stateindex[j]]
                nhop = abs(stateindex[j] - active_state)
                if gsum > z and 0 < nhop <= self.maxh:
                    new_state = int(stateindex[j])
                    event = 1
                    break
            if profile_hopping:
                self.profiler.record(
                    'hopping', time.perf_counter() - hop_start)

            if event:
                # In-substep hop trial (legacy semantics): the active
                # state only advances when the hop has enough kinetic
                # energy; the velocity itself is adjusted once from
                # the original V after the loop (trial result
                # discarded).
                if mass_au is not None:
                    trial_pair = (min(active_state, new_state),
                                  max(active_state, new_state))
                    nacv = _lookup_nac(nac_index, nac_values,
                                       trial_pair)
                    _, frustrated = _adjust_for_hop(
                        energy, active_state, new_state, velocities,
                        mass_au, nacv, self.adjust, self.reflect)
                else:
                    frustrated = 0
                if frustrated == 0:
                    active_state = new_state

            if self.use_deco:
                A = self.kernel_deco(A, H, delt, active_state,
                                     self.deco, ekin)

        # ---- Population diagnostics (detect only, never clip) ----
        _diagnose_population(A, active_state)

        # ---- Debug dump of the step's intermediate data ----
        if dump_enabled:
            self._dump_step(current_state, active_state, A, D, B,
                            g_history)

        # ---- Final hopping decision and velocity adjustment ----
        adjust_start = (time.perf_counter() if self.profile_enabled
                        else 0.0)
        if active_state != current_state and mass_au is not None:
            hop_pair = (min(current_state, active_state),
                        max(current_state, active_state))
            nacv = _lookup_nac(nac_index, nac_values, hop_pair)
            new_velo, frustrated = _adjust_for_hop(
                energy, current_state, active_state, velocities,
                mass_au, nacv, self.adjust, self.reflect)
            if frustrated == 0:
                state.update_system({
                    'current_state': active_state,
                    'velocities': new_velo.tolist(),
                    'state_density': _density_to_list(A),
                })
            else:
                # Frustrated hopping: keep the state and the velocity.
                state.update_system(
                    {'state_density': _density_to_list(A)})
        else:
            state.update_system({'state_density': _density_to_list(A)})
        if self.profile_enabled:
            self.profiler.record('velocity adjustment',
                                 time.perf_counter() - adjust_start)

        # state_population mirrors the real diagonal of A.
        state.update_dynamics(
            {'state_population': [float(p)
                                  for p in np.diag(np.real(A))]})

        if self.profile_enabled:
            self.profile_calls += 1
            if self.profile_calls % 100 == 0:
                print(self.profiler.summary())

        return state

    def _dump_step(self, from_state, to_state, A, D, B, g_history):
        """Append one step's intermediate data to {title}.fssh.

        The dump carries the final density matrix A of the step,
        the final interpolated coupling matrix D, the accumulated
        hopping matrix B and one g probability row per electronic
        substep, enough to replay every hopping decision offline.

        Parameters
        ----------
        from_state : int
            Active state at the start of the step (0-based).
        to_state : int
            Active state at the end of the step (0-based).
        A : np.ndarray
            Final density matrix (nstate, nstate), complex.
        D : np.ndarray
            Final coupling matrix (nstate, nstate), complex.
        B : np.ndarray
            Accumulated hopping matrix (nstate, nstate), real.
        g_history : list of np.ndarray
            Per-substep hopping probability vectors.
        """
        self.dump_steps += 1
        text = '\n&fssh step %8d  time %16.6f  state %d -> %d\n' % (
            self.dump_steps, self.dump_steps * self.timestep,
            from_state + 1, to_state + 1)
        text += 'A density matrix (re im per row):\n%s\n' % (
            _complex_matrix_text(A))
        text += 'D coupling matrix (re im per row):\n%s\n' % (
            _complex_matrix_text(D))
        text += 'B hopping matrix:\n%s\n' % _matrix_text(B)
        text += 'g probabilities per substep:\n%s\n' % (
            _matrix_text(np.asarray(g_history)))
        with open(self.dump_path, 'a', encoding='utf-8') as dump:
            dump.write(text)


def run_fssh(state):
    """Run one FSSH timestep through a disposable integrator.

    Thin compatibility entry for standalone calls: builds a temporary
    FsshIntegrator from the state configuration and executes one
    step. Trajectories use the surface_hopping factory instead,
    which initializes the integrator once and reuses it.

    Parameters
    ----------
    state : PyRAIState
        State with the recorded timestep and the current electronic
        properties.

    Returns
    -------
    PyRAIState
        The updated state (unchanged when the configuration is
        missing).
    """
    if state is None or state.pmd is None:
        return state
    if getattr(state.pmd, 'dynamics', None) is None:
        return state
    return FsshIntegrator(state)(state)
