"""
======================================================================
This module tests the FSSH electronic propagator of the NAMD
workflow (dev/methods/FSSH_DEVELOPMENT.md):

Tests cover:
- kernel regression against the legacy scalar-loop formulas
  (dPdt, rk4, matB, ktdc, decoherence)
- numba vs numpy kernel numerical regression (auto-skipped without
  the optional numba dependency)
- density matrix [re, im] serialization round trip and the
  system.state_density schema field
- warm-up steps (insufficient history) seed A without hopping
- hop scenario through the official update path
  (current_state, velocities, state_density, state_population)
- Phase 6 numerical protection: degenerate/inverted gap pair
  exclusion, NAC-missing isotropic rescaling, frustrated hopping
- intersystem crossing: SOC pairs join the unified energy-ordered
  cumulative sampling (T1-below-S1 crossings stay valid, ISC hops
  rescale isotropically, no-soc control, no multiplicity branching)
- surface_hopping factory dispatch into the FSSH integrator
- fssh_* keyword defaults and the function-level profiler
- one-time initialization: multi-trajectory kernel isolation and
  instance-vs-disposable step equivalence
- the verbose >= 4 debug dump: {title}.fssh carries the per-step
  A/D/B matrices and the substep g vectors, and stays absent below
  verbose 4

Aug 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np

from pyrai3md.data.state import PyRAIState
from pyrai3md.data.runtime.state import RuntimeState
from pyrai3md.schema.config import PyRAIConfig, DEFAULT
from pyrai3md.utilities.units import AMU_TO_AU, EV_TO_HARTREE
from pyrai3md.workflow.integrator import surface_hopping
from pyrai3md.workflow.integrator.electronic import ElectronicStepper
from pyrai3md.workflow.integrator.fssh import (
    run_fssh, _density_to_list, _density_from_list,
    FsshIntegrator, _initial_density, WARMUP_STEPS,
    _pair_gap_valid, _pair_gap_valid_soc,
    _complex_matrix_text, FSSH_DUMP_HEADER,
)
from pyrai3md.workflow.integrator.fssh_numpy import (
    dPdt_numpy, rk4_numpy, matB_numpy, ktdc_numpy, decoherence_numpy,
    avoid_singularity,
)
from pyrai3md.workflow.integrator.fssh_numba import (
    NUMBA_AVAILABLE,
    dPdt_numba, rk4_numba, matB_numba, ktdc_numba, decoherence_numba,
)
from pyrai3md.workflow.integrator.fssh_profile import FsshProfiler

# Test system constants
MASS = [12.0, 1.0]                       # amu
HOP_VELOS = [[0.003, 0.0, 0.0], [0.0, 0.003, 0.0]]
COORDS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]   # Angstrom
VELOS = [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]]  # Bohr/au
DT = 20.67                               # au


# ---- Legacy scalar-loop reference kernels (regression baseline) ----

def _ref_avoid(energy_i, energy_j, state_i, state_j):
    """Legacy avoid_singularity (tsh_helper)."""
    sign = -1.0 if state_i < state_j else 1.0
    gap = abs(energy_i - energy_j)
    return sign * max(gap, 1e-16)


def _ref_dpdt(A, H, D):
    nstate = len(A)
    dA = np.zeros((nstate, nstate), dtype=complex)
    for k in range(nstate):
        for j in range(nstate):
            for l in range(nstate):
                dA[k, j] += (A[l, j] * (-1j * H[k, l] - D[k, l])
                             - A[k, l] * (-1j * H[l, j] - D[l, j]))
    return dA


def _ref_rk4(A, H, D, dHdt, dDdt, delt):
    k1 = _ref_dpdt(A, H, D)
    k2 = _ref_dpdt(A + k1 * delt / 2, H + dHdt / 2, D + dDdt / 2)
    k3 = _ref_dpdt(A + k2 * delt / 2, H + dHdt / 2, D + dDdt / 2)
    k4 = _ref_dpdt(A + k3 * delt, H + dHdt, D + dDdt)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _ref_matb(A, H, D):
    nstate = len(A)
    B = np.zeros((nstate, nstate))
    for k in range(nstate):
        for j in range(nstate):
            B[k, j] = (2 * np.imag(np.conj(A[k, j]) * H[k, j])
                       - 2 * np.real(np.conj(A[k, j]) * D[k, j]))
    return B


def _ref_ktdc(s1, s2, E, Ep, Epp, dt, gap):
    dVt = _ref_avoid(E[s1], E[s2], s1, s2)
    dVt_dt = _ref_avoid(Ep[s1], Ep[s2], s1, s2)
    gap = gap / 27.211396132
    if abs(dVt) > gap or abs(dVt_dt) > gap:
        return 0.0
    dVt_2dt = _ref_avoid(Epp[s1], Epp[s2], s1, s2)
    d2Vdt2 = (dVt - 2 * dVt_dt + dVt_2dt) / dt ** 2
    ratio = d2Vdt2 / dVt
    if ratio > 0:
        return -(ratio ** 0.5) / 2
    return 0.0


def _ref_decoherence(A, H, delt, state, deco, ekin):
    nstate = len(A)
    tau = np.zeros(nstate)
    for k in range(nstate):
        if k != state:
            tau[k] = abs(1.0 / _ref_avoid(
                np.real(H[state, state]), np.real(H[k, k]), state, k)
            ) * (1 + deco / ekin)
    # Half exponents: tau is the population decoherence time
    # (OpenMolcas tully.F90 convention).
    for k in range(nstate):
        for j in range(nstate):
            if k != state and j != state:
                A[k, j] *= (np.exp(-0.5 * delt / tau[k])
                            * np.exp(-0.5 * delt / tau[j]))
    Asum = 0.0
    for k in range(nstate):
        if k != state:
            Asum += np.real(A[k, k])
    Amm = np.real(A[state, state])
    A[state, state] = 1 - Asum
    for k in range(nstate):
        for j in range(nstate):
            if k == state and j != state:
                A[k, j] *= (np.exp(-0.5 * delt / tau[j])
                            * (np.real(A[state, state]) / (Amm + 1e-16)) ** 0.5)
            elif k != state and j == state:
                A[k, j] *= (np.exp(-0.5 * delt / tau[k])
                            * (np.real(A[state, state]) / (Amm + 1e-16)) ** 0.5)
    return A


def _rng():
    return np.random.RandomState(7)


def _sample_matrices(nstate=3):
    """Random but physically shaped test matrices."""
    rng = _rng()
    A = rng.uniform(0, 1, (nstate, nstate)) \
        + 1j * rng.uniform(-1, 1, (nstate, nstate))
    A = (A + A.conj().T) / 2
    H = np.diag(rng.uniform(0, 0.2, nstate)).astype(complex)
    H[0, 1] = H[1, 0] = 1e-3
    D = np.zeros((nstate, nstate), dtype=complex)
    D[0, 1], D[1, 0] = 0.01, -0.01
    return A, H, D


# ---- State builders ----

def _make_state(nstate=2, current_state=0, velocities=VELOS,
                nac_pairs=None, soc_pairs=None, population=None,
                **dynamics_overrides):
    """Create a minimal PyRAIState for FSSH tests."""
    config = PyRAIConfig()
    config.system.atoms = ['C', 'H']
    config.system.coordinates = [list(row) for row in COORDS]
    config.system.velocities = [list(row) for row in velocities]
    config.system.mass = list(MASS)
    config.system.natom = 2
    config.system.nstate = nstate
    if nac_pairs is not None:
        config.system.nac_pair_map = [list(p) for p in nac_pairs]
    if soc_pairs is not None:
        config.system.soc_pair_map = [list(p) for p in soc_pairs]

    state = PyRAIState(runtime=RuntimeState(total_steps=10))
    state.pmd = config
    state.update_system({'current_state': current_state})
    if population is not None:
        state.update_dynamics({'state_population': list(population)})
    for key, value in dynamics_overrides.items():
        setattr(config.dynamics, key, value)
    return state


def _push_step(state, energy, nac=None, soc=None):
    """Snapshot one recorded step through the official history API."""
    result = {'energy': list(energy)}
    if nac is not None:
        result['nac'] = np.asarray(nac).tolist()
    if soc is not None:
        result['soc'] = list(soc)
    state.set_result(result)
    state.update_history()


def _prime_history(state, gap=0.05, steps=None, nac=None):
    """Fill the history with constant-gap warm-up steps and set the
    identical current-step result."""
    steps = steps if steps is not None else WARMUP_STEPS
    energy = [0.0, gap]
    for _ in range(steps):
        _push_step(state, energy, nac=nac)
    result = {'energy': list(energy)}
    if nac is not None:
        result['nac'] = np.asarray(nac).tolist()
    state.set_result(result)


def _prime_history_states(state, energy, steps=None, nac=None,
                          soc=None):
    """Same as _prime_history with an explicit energy list (any
    number of states) and optional SOC values (cm-1)."""
    steps = steps if steps is not None else WARMUP_STEPS
    for _ in range(steps):
        _push_step(state, energy, nac=nac, soc=soc)
    result = {'energy': list(energy)}
    if nac is not None:
        result['nac'] = np.asarray(nac).tolist()
    if soc is not None:
        result['soc'] = list(soc)
    state.set_result(result)


def _density(state):
    return _density_from_list(state.pmd.system.state_density)


class TestKernelsLegacyRegression(unittest.TestCase):
    """Kernel-by-kernel regression against the legacy formulas."""

    def test_dpdt_matches_legacy_loops(self):
        A, H, D = _sample_matrices()
        np.testing.assert_allclose(dPdt_numpy(A, H, D),
                                   _ref_dpdt(A, H, D), atol=1e-12)

    def test_rk4_matches_legacy_loops(self):
        A, H, D = _sample_matrices()
        dHdt = 0.001 * H
        dDdt = 0.002 * D
        np.testing.assert_allclose(
            rk4_numpy(A, H, D, dHdt, dDdt, 0.5),
            _ref_rk4(A, H, D, dHdt, dDdt, 0.5), atol=1e-12)

    def test_matb_matches_legacy_loops(self):
        A, H, D = _sample_matrices()
        np.testing.assert_allclose(matB_numpy(A, H, D),
                                   _ref_matb(A, H, D), atol=1e-12)

    def test_ktdc_matches_legacy(self):
        E = np.array([0.0, 0.02])
        Ep = np.array([0.0, 0.04])
        Epp = np.array([0.0, 0.09])
        dt = 20.67
        for gap in (0.2, 1.0, 5.0):
            expected = _ref_ktdc(0, 1, E, Ep, Epp, dt, gap)
            got = ktdc_numpy(0, 1, E, Ep, Epp, dt, gap * EV_TO_HARTREE)
            self.assertAlmostEqual(got, expected, places=12)

    def test_ktdc_baeck_an_threshold(self):
        # Gap above the Baeck-An threshold suppresses the coupling.
        E = np.array([0.0, 0.2])
        self.assertEqual(ktdc_numpy(0, 1, E, E, E, 20.67,
                                    0.1 * EV_TO_HARTREE), 0.0)

    def test_decoherence_matches_legacy(self):
        rng = _rng()
        A = np.diag([0.7, 0.2, 0.1]).astype(complex)
        A[0, 1], A[1, 0] = 0.05 + 0.01j, 0.05 - 0.01j
        A[0, 2], A[2, 0] = 0.02, 0.02
        H = np.diag([0.0, 0.05, 0.12]).astype(complex)
        expected = _ref_decoherence(A.copy(), H, 0.5, 0, 0.1, 0.03)
        got = decoherence_numpy(A.copy(), H, 0.5, 0, 0.1, 0.03)
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_avoid_singularity_sign(self):
        self.assertLess(avoid_singularity(0.0, 0.05, 0, 1), 0)
        self.assertGreater(avoid_singularity(0.05, 0.0, 1, 0), 0)
        self.assertEqual(abs(avoid_singularity(0.0, 0.0, 0, 1)), 1e-16)


@unittest.skipUnless(NUMBA_AVAILABLE, 'numba is not installed')
class TestKernelsNumbaNumpy(unittest.TestCase):
    """Numba kernels must reproduce the NumPy reference numerically."""

    def test_all_kernels_match_numpy(self):
        A, H, D = _sample_matrices()
        dHdt, dDdt = 0.001 * H, 0.002 * D
        np.testing.assert_allclose(dPdt_numba(A, H, D),
                                   dPdt_numpy(A, H, D), atol=1e-12)
        np.testing.assert_allclose(rk4_numba(A, H, D, dHdt, dDdt, 0.5),
                                   rk4_numpy(A, H, D, dHdt, dDdt, 0.5),
                                   atol=1e-12)
        np.testing.assert_allclose(matB_numba(A, H, D),
                                   matB_numpy(A, H, D), atol=1e-12)
        E = np.array([0.0, 0.02, 0.05])
        Ep = np.array([0.0, 0.04, 0.05])
        Epp = np.array([0.0, 0.09, 0.05])
        self.assertAlmostEqual(
            ktdc_numba(0, 1, E, Ep, Epp, 20.67, 1.0 * EV_TO_HARTREE),
            ktdc_numpy(0, 1, E, Ep, Epp, 20.67, 1.0 * EV_TO_HARTREE),
            places=12)
        Hdec = np.diag([0.0, 0.05, 0.12]).astype(complex)
        Adec = np.diag([0.7, 0.2, 0.1]).astype(complex)
        Adec[0, 1], Adec[1, 0] = 0.05 + 0.01j, 0.05 - 0.01j
        np.testing.assert_allclose(
            decoherence_numba(Adec.copy(), Hdec, 0.5, 0, 0.1, 0.03),
            decoherence_numpy(Adec.copy(), Hdec, 0.5, 0, 0.1, 0.03),
            atol=1e-12)

    def test_propagation_preserves_invariants(self):
        A = np.diag([1.0, 0.0]).astype(complex)
        H = np.diag([0.0, 0.05]).astype(complex)
        D = np.zeros((2, 2), dtype=complex)
        D[0, 1], D[1, 0] = 0.02, -0.02
        A1 = A + rk4_numba(A, H, D, np.zeros_like(H),
                           np.zeros_like(D), 0.5) * 0.5
        self.assertAlmostEqual(np.trace(A1).real, 1.0, places=10)
        np.testing.assert_allclose(A1, A1.conj().T, atol=1e-10)


class TestDensitySerialization(unittest.TestCase):
    """Density matrix [re, im] persistence helpers and schema field."""

    def test_round_trip(self):
        A = np.array([[0.7, 0.1 + 0.2j], [0.1 - 0.2j, 0.3]],
                     dtype=complex)
        restored = _density_from_list(_density_to_list(A))
        np.testing.assert_allclose(restored, A)

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(_density_from_list([]))
        self.assertIsNone(_density_from_list([[1.0, 2.0]]))
        self.assertIsNone(_density_from_list('bad'))

    def test_schema_field_round_trip(self):
        config = PyRAIConfig()
        stored = _density_to_list(np.diag([1.0, 0.0]).astype(complex))
        config.system.state_density = stored
        dumped = config.model_dump()
        restored = PyRAIConfig(**dumped)
        np.testing.assert_allclose(
            _density_from_list(restored.system.state_density),
            np.diag([1.0, 0.0]).astype(complex))

    def test_initial_density_seeds_population(self):
        A = _initial_density(2, 0, [0.6, 0.4])
        np.testing.assert_allclose(np.diag(np.real(A)), [0.6, 0.4])
        # Length mismatch falls back to unit occupation of the state.
        A = _initial_density(2, 1, [1.0])
        np.testing.assert_allclose(np.diag(np.real(A)), [0.0, 1.0])






class TestFsshNumericalProtection(unittest.TestCase):
    """Phase 6 pair validity, isotropic rescaling and frustrated
    hopping policies."""

    def test_degenerate_pair_skipped(self):
        np.random.seed(3)
        nac = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        state = _make_state(nstate=2, current_state=0,
                            nac_pairs=[[0, 1]],
                            population=[1.0, 0.0],
                            substep=1, integrate=1, fssh_force_hop=1,
                            deco='OFF',
                            nactype='nac', adjust=1, reflect=1, maxh=1)
        # Zero gap: the pair is degenerate and must not hop.
        for _ in range(WARMUP_STEPS):
            _push_step(state, [0.0, 0.0], nac=nac)
        state.set_result({'energy': [0.0, 0.0],
                          'nac': np.asarray(nac).tolist()})

        run_fssh(state)

        self.assertEqual(state.current_state, 0)
        self.assertIsNotNone(_density(state))

    def test_inverted_gap_pair_invalid(self):
        np.random.seed(3)
        nac = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        state = _make_state(nstate=2, current_state=0,
                            nac_pairs=[[0, 1]],
                            population=[1.0, 0.0],
                            substep=1, integrate=1, fssh_force_hop=1,
                            deco='OFF',
                            nactype='nac', adjust=1, reflect=1, maxh=1)
        # E[1] < E[0]: NN identity order disagrees with energy order;
        # the pair stays invalid for the step (no energy resorting).
        for _ in range(WARMUP_STEPS):
            _push_step(state, [0.05, 0.0], nac=nac)
        state.set_result({'energy': [0.05, 0.0],
                          'nac': np.asarray(nac).tolist()})

        run_fssh(state)

        self.assertEqual(state.current_state, 0)

    def test_ktdc_hop_uses_isotropic_rescaling(self):
        np.random.seed(5)
        state = _make_state(nstate=2, current_state=0,
                            nac_pairs=[[0, 1]],
                            population=[1.0, 0.0],
                            substep=1, integrate=1, fssh_force_hop=1,
                            deco='OFF',
                            nactype='ktdc', adjust=1, reflect=1, maxh=1,
                            gap=2.0)
        # Shrinking gap sequence drives a nonzero kTDC curvature:
        # the last two pushed steps are E(t-dt) and E(t-2dt).
        for gap_t in (0.02, 0.02, 0.09, 0.04):
            _push_step(state, [0.0, gap_t])
        state.set_result({'energy': [0.0, 0.02]})

        kinetic_before = 0.5 * np.sum(
            np.asarray(MASS)[:, None] * AMU_TO_AU
            * np.asarray(VELOS) ** 2)
        run_fssh(state)

        self.assertEqual(state.current_state, 1)
        new_velo = np.asarray(state.velocities)
        kinetic_after = 0.5 * np.sum(
            np.asarray(MASS)[:, None] * AMU_TO_AU * new_velo ** 2)
        np.testing.assert_allclose(kinetic_after,
                                   kinetic_before - 0.02, rtol=1e-10)
        # Isotropic rescaling keeps the velocity direction.
        direction_old = np.asarray(VELOS).reshape(-1)
        direction_new = new_velo.reshape(-1)
        np.testing.assert_allclose(
            direction_new / np.linalg.norm(direction_new),
            direction_old / np.linalg.norm(direction_old), atol=1e-12)

    def test_frustrated_hop_keeps_state_and_velocity(self):
        np.random.seed(7)
        nac = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        state = _make_state(nstate=2, current_state=0,
                            nac_pairs=[[0, 1]],
                            population=[1.0, 0.0],
                            substep=1, integrate=1, fssh_force_hop=1,
                            deco='OFF',
                            nactype='nac', adjust=1, reflect=1, maxh=1)
        _prime_history(state, gap=0.05, nac=nac)
        # Remove essentially all kinetic energy: the upward hop must
        # be frustrated (no reflect, velocity and state kept).
        state.update_system({'velocities': [[1e-9, 0.0, 0.0],
                                            [0.0, 1e-9, 0.0]]})

        run_fssh(state)

        self.assertEqual(state.current_state, 0)
        np.testing.assert_allclose(state.velocities,
                                   [[1e-9, 0.0, 0.0], [0.0, 1e-9, 0.0]])
        self.assertIsNotNone(_density(state))

    def test_ktdc_previous_step_uses_three_point_window(self):
        """The previous-step D reconstruction must feed the E(t-3dt)
        history point into the kTDC curvature window (the legacy
        propagator keeps the computed last_d; the rebuilt matrix
        needs the genuine third point instead of reusing E(t-2dt))."""
        densities = []
        for gap_t3 in (0.15, 0.06):
            np.random.seed(5)
            state = _make_state(nstate=2, current_state=0,
                                nac_pairs=[[0, 1]],
                                population=[1.0, 0.0],
                                substep=2, integrate=1,
                                deco='OFF',
                                nactype='ktdc', adjust=1, reflect=1,
                                maxh=0, gap=4.0)
            # Pushed frames become E(t-3dt) = gap_t3, E(t-2dt) =
            # 0.09 and E(t-dt) = 0.04; 0.15 keeps the curvature
            # positive while 0.06 flips it negative (kTDC zeros the
            # previous-step coupling), so the two reconstructions
            # must produce different D(t-dt) matrices. maxh=0 blocks
            # hopping and substep=2 keeps the first substep inside
            # the Dp -> Dt ramp, where Dp stays observable (a single
            # substep lands exactly on Dt in both cases).
            for gap_t in (0.20, gap_t3, 0.09, 0.04):
                _push_step(state, [0.0, gap_t])
            state.set_result({'energy': [0.0, 0.02]})
            run_fssh(state)
            densities.append(_density(state))

        # The propagated density matrices differ beyond floating-
        # point noise once the third curvature point is wired in.
        self.assertGreater(
            np.abs(densities[0] - densities[1]).max(), 1e-6)

    def test_invalid_nactype_raises(self):
        state = _make_state(nstate=2, current_state=0,
                            nac_pairs=[[0, 1]],
                            population=[1.0, 0.0],
                            substep=1, integrate=1,
                            deco='OFF',
                            nactype='ktcd', adjust=1, reflect=1,
                            maxh=1)
        with self.assertRaises(ValueError):
            run_fssh(state)




class TestFsshSchemaAndSelection(unittest.TestCase):
    """fssh_* keywords, adapter types and kernel selection."""

    def test_schema_defaults(self):
        dynamics = PyRAIConfig().dynamics
        for key in ('fssh_rk4', 'fssh_dpdt', 'fssh_matb', 'fssh_ktdc',
                    'fssh_decoherence'):
            self.assertEqual(getattr(dynamics, key), 'numpy')
            self.assertEqual(DEFAULT[key], 'numpy')
        self.assertEqual(dynamics.fssh_profile, 0)
        self.assertEqual(dynamics.fssh_force_hop, 0)

    def test_adapter_registers_fssh_types(self):
        from pyrai3md.schema.adapter import MD_TYPES
        for key in ('fssh_rk4', 'fssh_dpdt', 'fssh_matb', 'fssh_ktdc',
                    'fssh_decoherence'):
            self.assertEqual(MD_TYPES[key], 's')
        self.assertEqual(MD_TYPES['fssh_profile'], 'i')
        self.assertEqual(MD_TYPES['fssh_force_hop'], 'i')

    def test_update_dynamics_official_path(self):
        state = _make_state()
        state.update_dynamics({'state_population': [0.5, 0.5]})
        self.assertEqual(state.pmd.dynamics.state_population, [0.5, 0.5])

    @unittest.skipIf(NUMBA_AVAILABLE, 'requires a numba-free run')
    def test_numba_request_falls_back_with_warning(self):
        state = _make_state(nstate=2, nac_pairs=[[0, 1]],
                            fssh_dpdt='numba')
        state.set_result({'energy': [0.0, 0.05]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            run_fssh(state)
        self.assertTrue(any('numba' in str(item.message)
                            for item in caught))
        self.assertIsNotNone(_density(state))


class TestFsshProfiler(unittest.TestCase):
    """Function-level timing accumulator."""

    def test_record_and_summary(self):
        profiler = FsshProfiler()
        profiler.record('dPdt', 0.25)
        profiler.record('dPdt', 0.25)
        profiler.set_implementations({'dPdt': 'numpy'})
        text = profiler.summary()
        self.assertIn('FSSH Performance', text)
        self.assertIn('dPdt', text)
        self.assertIn('2', text)
        self.assertIn('Implementation', text)

    def test_wrap_times_calls(self):
        profiler = FsshProfiler()
        wrapped = profiler.wrap('matB', lambda a: a * 2)
        self.assertEqual(wrapped(3), 6)
        self.assertEqual(profiler.stats['matB'][0], 1)
        self.assertGreaterEqual(profiler.stats['matB'][1], 0.0)


class TestSurfaceHoppingFactory(unittest.TestCase):
    """surface_hopping initializes the algorithm once and returns
    the callable state -> state integrator."""

    def test_nosh_returns_pass_through(self):
        state = _make_state()
        surfacehop = surface_hopping('nosh', state)
        self.assertIs(surfacehop(state), state)

    def test_gsh_placeholder_pass_through(self):
        state = _make_state()
        surfacehop = surface_hopping('gsh', state)
        self.assertIs(surfacehop(state), state)

    def test_fssh_returns_integrator(self):
        state = _make_state(nstate=2, nac_pairs=[[0, 1]])
        surfacehop = surface_hopping('FSSH', state)
        # The factory dispatches through the ElectronicStepper; C=1
        # stays a transparent binding wrapping the unchanged core.
        self.assertIsInstance(surfacehop, ElectronicStepper)
        self.assertTrue(surfacehop.transparent)
        self.assertIsInstance(
            surfacehop.binding(0).algorithm, FsshIntegrator)

    def test_unknown_method_raises(self):
        state = _make_state()
        with self.assertRaises(ValueError):
            surface_hopping('xxx', state)

    def test_fssh_requires_planning_context(self):
        state = PyRAIState(runtime=RuntimeState(total_steps=1))
        with self.assertRaises(ValueError):
            FsshIntegrator(state)




class TestFsshDebugDump(unittest.TestCase):
    """The verbose >= 4 debug dump persists the per-step FSSH
    intermediate data into {title}.fssh next to the trajectory
    output; below verbose 4 nothing is written."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_dump_state(self, verbose, title):
        nac = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        state = _make_state(nstate=2, current_state=0,
                            velocities=HOP_VELOS,
                            nac_pairs=[[0, 1]],
                            population=[1.0, 0.0],
                            substep=10, integrate=1,
                            fssh_force_hop=1, deco='OFF',
                            nactype='nac', adjust=1, reflect=1,
                            maxh=1, rk4='1')
        state.pmd.control.verbose = verbose
        state.pmd.control.title = title
        _prime_history(state, gap=0.05, nac=nac)
        return state

    def test_verbose_four_dumps_the_step_matrices(self):
        np.random.seed(11)
        state = self._make_dump_state(4, 'dump_probe')
        run_fssh(state)

        # The forced hop lands on state 2 (same scenario as
        # TestFsshHop) and the dump records the transition.
        self.assertEqual(state.current_state, 1)
        dump_path = os.path.join(self.tmpdir, 'dump_probe.fssh')
        self.assertTrue(os.path.isfile(dump_path))
        with open(dump_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertTrue(content.startswith(FSSH_DUMP_HEADER))
        self.assertIn('&fssh step        1', content)
        self.assertIn('state 1 -> 2', content)
        for marker in ('A density matrix', 'D coupling matrix',
                       'B hopping matrix',
                       'g probabilities per substep'):
            self.assertIn(marker, content)
        # One g row per electronic substep of the step.
        self.assertEqual(content.count('g probabilities'), 1)
        g_block = content.split('g probabilities per substep:')[1]
        self.assertEqual(len([line for line in g_block.strip().split('\n')
                              if line.strip()]),
                         int(state.pmd.dynamics.substep))
        # The dumped density matrix matches the integrator's final
        # A written back through the official state field.
        A = _density(state)
        self.assertIn(_complex_matrix_text(A).split('\n')[0], content)

    def test_verbose_three_writes_no_dump(self):
        np.random.seed(11)
        state = self._make_dump_state(3, 'quiet_probe')
        run_fssh(state)
        self.assertFalse(os.path.isfile(os.path.join(
            self.tmpdir, 'quiet_probe.fssh')))

    def test_default_verbose_writes_no_dump(self):
        np.random.seed(11)
        state = self._make_dump_state(2, 'default_probe')
        run_fssh(state)
        self.assertFalse(os.path.isfile(os.path.join(
            self.tmpdir, 'default_probe.fssh')))


if __name__ == '__main__':
    unittest.main()
