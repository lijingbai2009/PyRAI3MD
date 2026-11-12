"""
======================================================================
This module provides the optional Numba accelerated kernels of the
FSSH electronic propagator (dev/methods/FSSH_DEVELOPMENT.md).

Each kernel mirrors its NumPy reference counterpart in
fssh_numpy.py bit-for-bit in algorithm; the main flow (fssh.py)
selects the implementation per function at initialization. Numba is
an optional dependency: when it is not installed NUMBA_AVAILABLE is
False and the selection mechanism falls back to the NumPy reference,
which always stays runnable.

Kernels:
    dPdt_numba          time derivative of the density matrix
    rk4_numba           RK4 propagation fused into a single kernel
    matB_numba          transition probability matrix B
    ktdc_numba          curvature-driven time-dependent coupling
    decoherence_numba   OpenMolcas-style decoherence renormalization

Aug 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    NUMBA_AVAILABLE = False

# Energy gap cutoff, identical to fssh_numpy.GAP_CUTOFF.
GAP_CUTOFF = 1e-16

if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _avoid_singularity_numba(energy_i, energy_j, state_i, state_j):
        """Signed, regularized energy gap (Numba helper)."""
        sign = -1.0 if state_i < state_j else 1.0
        gap = abs(energy_i - energy_j)
        if gap < GAP_CUTOFF:
            return sign * GAP_CUTOFF
        return sign * gap

    @njit(cache=True)
    def _dPdt_core(A, H, D):
        """dA/dt = K A - A K with K = -i H - D (explicit loops)."""
        nstate = A.shape[0]
        dA = np.zeros((nstate, nstate), dtype=np.complex128)
        for k in range(nstate):
            for j in range(nstate):
                acc = 0.0 + 0.0j
                for l in range(nstate):
                    acc += (A[l, j] * (-1j * H[k, l] - D[k, l])
                            - A[k, l] * (-1j * H[l, j] - D[l, j]))
                dA[k, j] = acc
        return dA

    @njit(cache=True)
    def dPdt_numba(A, H, D):
        """Time derivative of the electronic density matrix.

        Numba counterpart of dPdt_numpy (Tully's FSSH).
        """
        return _dPdt_core(A, H, D)

    @njit(cache=True)
    def rk4_numba(A, H, D, dHdt, dDdt, delt):
        """4th-order Runge-Kutta derivative fused into one kernel.

        The four dPdt evaluations run inside a single Numba kernel
        to avoid four Python -> kernel round trips per electronic
        substep. Mirrors rk4_numpy (legacy RK4dPdt interpolation).
        """
        k1 = _dPdt_core(A, H, D)
        k2 = _dPdt_core(A + k1 * delt / 2, H + dHdt / 2, D + dDdt / 2)
        k3 = _dPdt_core(A + k2 * delt / 2, H + dHdt / 2, D + dDdt / 2)
        k4 = _dPdt_core(A + k3 * delt, H + dHdt, D + dDdt)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @njit(cache=True)
    def matB_numba(A, H, D):
        """Transition probability matrix B (explicit loops).

        B_ij = 2 Im(A_ij* H_ij) - 2 Re(A_ij* D_ij). Numba
        counterpart of matB_numpy.
        """
        nstate = A.shape[0]
        B = np.zeros((nstate, nstate), dtype=np.float64)
        for k in range(nstate):
            for j in range(nstate):
                B[k, j] = (2.0 * (np.conj(A[k, j]) * H[k, j]).imag
                           - 2.0 * (np.conj(A[k, j]) * D[k, j]).real)
        return B

    @njit(cache=True)
    def ktdc_numba(s1, s2, E, Ep, Epp, dt, gap):
        """Curvature-driven time-dependent coupling (kTDC).

        Numba counterpart of ktdc_numpy; identical Baeck-An
        threshold and sign conventions.
        """
        dVt = _avoid_singularity_numba(E[s1], E[s2], s1, s2)
        dVt_dt = _avoid_singularity_numba(Ep[s1], Ep[s2], s1, s2)
        if abs(dVt) > gap or abs(dVt_dt) > gap:
            return 0.0

        dVt_2dt = _avoid_singularity_numba(Epp[s1], Epp[s2], s1, s2)
        d2Vdt2 = (dVt - 2 * dVt_dt + dVt_2dt) / dt ** 2
        ratio = d2Vdt2 / dVt
        if ratio > 0:
            return -(ratio ** 0.5) / 2
        return 0.0

    @njit(cache=True)
    def decoherence_numba(A, H, delt, state, deco, ekin):
        """OpenMolcas-style decoherence correction.

        Numba counterpart of decoherence_numpy; identical decay,
        renormalization and rescaling sequence (half exponents:
        tau is the population decoherence time, matching
        OpenMolcas tully.F90). A is modified in-place and
        returned.
        """
        nstate = A.shape[0]
        tau = np.zeros(nstate, dtype=np.float64)
        for k in range(nstate):
            if k != state:
                tau[k] = abs(1.0 / _avoid_singularity_numba(
                    H[state, state].real, H[k, k].real, state, k)
                ) * (1.0 + deco / ekin)

        for k in range(nstate):
            for j in range(nstate):
                if k != state and j != state:
                    A[k, j] *= (np.exp(-0.5 * delt / tau[k])
                                * np.exp(-0.5 * delt / tau[j]))

        Asum = 0.0
        for k in range(nstate):
            if k != state:
                Asum += A[k, k].real
        Amm = A[state, state].real
        A[state, state] = 1 - Asum

        scale = (A[state, state].real / (Amm + 1e-16)) ** 0.5
        for k in range(nstate):
            for j in range(nstate):
                if k == state and j != state:
                    A[k, j] *= np.exp(-0.5 * delt / tau[j]) * scale
                elif k != state and j == state:
                    A[k, j] *= np.exp(-0.5 * delt / tau[k]) * scale
        return A

else:
    # Numba is not installed: expose the names so the selection
    # mechanism can introspect them uniformly; they raise if ever
    # called directly (the selector never picks them without Numba).

    def _numba_missing(*args, **kwargs):
        raise RuntimeError(
            "Numba kernels are unavailable: install the 'numba' "
            "extra or select the numpy implementation")

    dPdt_numba = _numba_missing
    rk4_numba = _numba_missing
    matB_numba = _numba_missing
    ktdc_numba = _numba_missing
    decoherence_numba = _numba_missing
