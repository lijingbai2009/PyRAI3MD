"""
======================================================================
This module provides the NumPy reference kernels of the FSSH
electronic propagator (dev/methods/FSSH_DEVELOPMENT.md).

Every mathematical function of the FSSH algorithm has exactly one
NumPy implementation here and (optionally) one Numba implementation
in fssh_numba.py; the main flow (fssh.py) selects between them once
at initialization. NumPy is the reference implementation: it keeps
the algorithm runnable without Numba and serves as the numerical
regression baseline.

Kernels:
    dPdt_numpy          time derivative of the density matrix
    rk4_numpy           4th-order Runge-Kutta propagation step
    matB_numpy          transition probability matrix B
    ktdc_numpy          curvature-driven time-dependent coupling
    decoherence_numpy   OpenMolcas-style decoherence renormalization

All kernels are pure functions of their array arguments; pair-level
validity decisions and state bookkeeping belong to the main flow,
never to a kernel.

Aug 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

# Energy gap cutoff used to regularize divisions by state gaps
# (identical to the legacy avoid_singularity cutoff).
GAP_CUTOFF = 1e-16


def avoid_singularity(energy_i, energy_j, state_i, state_j):
    """Return the signed energy gap regularized against zero gaps.

    The sign convention follows the legacy FSSH: for state_i <
    state_j the gap is returned negative, so the kTDC curvature
    ratio keeps a well-defined sign. Gaps smaller than GAP_CUTOFF
    are clamped to the cutoff magnitude.

    Parameters
    ----------
    energy_i : float
        Energy of state i (Hartree).
    energy_j : float
        Energy of state j (Hartree).
    state_i : int
        Index of state i.
    state_j : int
        Index of state j.

    Returns
    -------
    float
        Signed, regularized energy difference E_i - E_j (Hartree).
    """
    sign = -1.0 if state_i < state_j else 1.0
    gap = abs(energy_i - energy_j)
    if gap < GAP_CUTOFF:
        return sign * GAP_CUTOFF
    return sign * gap


def state_gap(energy, s1, s2):
    """Return the regularized literature gap E_j - E_i.

    Used to recover the standard derivative coupling from the
    stored CI-derivative NAC vector <i| dH/dR |j> = d_ij * (E_j -
    E_i): d_ij = <i| dH/dR |j> / (E_j - E_i). Gaps smaller than
    GAP_CUTOFF are clamped to the cutoff magnitude, keeping the
    sign of the original gap.

    Parameters
    ----------
    energy : np.ndarray
        State energies (Hartree).
    s1 : int
        Index of state i.
    s2 : int
        Index of state j.

    Returns
    -------
    float
        Regularized signed energy difference E_j - E_i (Hartree).
    """
    gap = energy[s2] - energy[s1]
    if abs(gap) < GAP_CUTOFF:
        return GAP_CUTOFF if gap >= 0 else -GAP_CUTOFF
    return gap


def dPdt_numpy(A, H, D):
    """Time derivative of the electronic density matrix.

    dA/dt = K A - A K with K = -i H - D (Tully's FSSH, J. Chem.
    Phys. 93, 1061 (1990)).

    Parameters
    ----------
    A : np.ndarray
        Density matrix (nstate, nstate), complex.
    H : np.ndarray
        Model Hamiltonian (nstate, nstate), complex (Hartree).
    D : np.ndarray
        Nonadiabatic coupling matrix (nstate, nstate), complex.

    Returns
    -------
    np.ndarray
        dA/dt of the same shape, complex.
    """
    K = -1j * H - D
    return K @ A - A @ K


def rk4_numpy(A, H, D, dHdt, dDdt, delt):
    """Classical 4th-order Runge-Kutta derivative for the density.

    H and D are linearly interpolated across the electronic substep
    (legacy RK4dPdt form): k2/k3 evaluate at H + dHdt/2, k4 at
    H + dHdt. Returns dAdt (not the propagated A).

    Parameters
    ----------
    A : np.ndarray
        Density matrix at the substep start, complex.
    H : np.ndarray
        Hamiltonian at the substep start, complex.
    D : np.ndarray
        NAC matrix at the substep start, complex.
    dHdt : np.ndarray
        Hamiltonian increment per substep, complex.
    dDdt : np.ndarray
        NAC matrix increment per substep, complex.
    delt : float
        Electronic substep size (au).

    Returns
    -------
    np.ndarray
        dAdt of the same shape, complex.
    """
    k1 = dPdt_numpy(A, H, D)
    k2 = dPdt_numpy(A + k1 * delt / 2, H + dHdt / 2, D + dDdt / 2)
    k3 = dPdt_numpy(A + k2 * delt / 2, H + dHdt / 2, D + dDdt / 2)
    k4 = dPdt_numpy(A + k3 * delt, H + dHdt, D + dDdt)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


def matB_numpy(A, H, D):
    """Transition probability matrix B of the FSSH algorithm.

    B_ij = 2 Im(A_ij* H_ij) - 2 Re(A_ij* D_ij).

    Parameters
    ----------
    A : np.ndarray
        Density matrix (nstate, nstate), complex.
    H : np.ndarray
        Model Hamiltonian (nstate, nstate), complex.
    D : np.ndarray
        NAC matrix (nstate, nstate), complex.

    Returns
    -------
    np.ndarray
        B matrix (nstate, nstate), real.
    """
    return (2 * np.imag(np.conj(A) * H)
            - 2 * np.real(np.conj(A) * D))


def ktdc_numpy(s1, s2, E, Ep, Epp, dt, gap):
    """Curvature-driven time-dependent coupling (kTDC).

    Based on Truhlar et al., J. Chem. Theory Comput. 2022,
    DOI: 10.1021/acs.jctc.1c01080. The Baeck-An threshold suppresses
    the coupling when either instantaneous gap exceeds the 'gap'
    threshold. Pair-level validity (degenerate or inverted gaps) is
    decided by the main flow before calling this kernel.

    Parameters
    ----------
    s1 : int
        Lower state index (0-based, s1 < s2).
    s2 : int
        Upper state index (0-based).
    E : np.ndarray
        State energies of the current step (Hartree).
    Ep : np.ndarray
        State energies one step back (Hartree).
    Epp : np.ndarray
        State energies two steps back (Hartree).
    dt : float
        Full timestep (au).
    gap : float
        Baeck-An gap threshold (Hartree, converted by the caller).

    Returns
    -------
    float
        Time-dependent nonadiabatic coupling element (a.u.).
    """
    dVt = avoid_singularity(E[s1], E[s2], s1, s2)
    dVt_dt = avoid_singularity(Ep[s1], Ep[s2], s1, s2)
    if abs(dVt) > gap or abs(dVt_dt) > gap:
        return 0.0

    dVt_2dt = avoid_singularity(Epp[s1], Epp[s2], s1, s2)
    d2Vdt2 = (dVt - 2 * dVt_dt + dVt_2dt) / dt ** 2
    ratio = d2Vdt2 / dVt
    if ratio > 0:
        # Flip the sign to negative because s1 < s2.
        return -(ratio ** 0.5) / 2
    return 0.0


def decoherence_numpy(A, H, delt, state, deco, ekin):
    """OpenMolcas-style decoherence correction.

    Off-active-state coherences decay as f_i f_j A_ij with
    f_i = exp(-delt / (2 tau_i)) and tau_i = |1/(E_state - E_i)|
    (1 + deco/Ekin); the active-state population is then
    renormalized to 1 - sum(other populations) and the
    active/inactive coherences are rescaled accordingly.

    The half factors in the exponents follow OpenMolcas
    (src/surfacehop/tully.F90): eq. 17 of Granucci & Persico,
    J. Chem. Phys. 126, 134114 (2007), doi:10.1063/1.2715585,
    damps amplitudes with the full exponent, which would decay
    populations twice as fast; the half factor makes tau the
    population decoherence time, matching the OpenMolcas
    implementation.

    Parameters
    ----------
    A : np.ndarray
        Density matrix (nstate, nstate), complex. Modified
        in-place and returned.
    H : np.ndarray
        Model Hamiltonian (nstate, nstate), complex (Hartree).
    delt : float
        Electronic substep size (au).
    state : int
        Active state index (0-based).
    deco : float
        Decoherence parameter (parsed from the 'deco' keyword).
    ekin : float
        Nuclear kinetic energy (Hartree).

    Returns
    -------
    np.ndarray
        The decohered density matrix (same object as A).
    """
    nstate = A.shape[0]
    tau = np.zeros(nstate)
    for k in range(nstate):
        if k != state:
            tau[k] = abs(1.0 / avoid_singularity(
                np.real(H[state, state]), np.real(H[k, k]), state, k)
            ) * (1.0 + deco / ekin)

    # Decay the off-active-state coherences (half exponents: tau is
    # the population decoherence time, see the docstring).
    for k in range(nstate):
        for j in range(nstate):
            if k != state and j != state:
                A[k, j] *= (np.exp(-0.5 * delt / tau[k])
                            * np.exp(-0.5 * delt / tau[j]))

    # Renormalize the active-state population.
    Asum = sum(np.real(A[k, k]) for k in range(nstate) if k != state)
    Amm = np.real(A[state, state])
    A[state, state] = 1 - Asum

    # Rescale the active/inactive coherences.
    scale = (np.real(A[state, state]) / (Amm + 1e-16)) ** 0.5
    for k in range(nstate):
        for j in range(nstate):
            if k == state and j != state:
                A[k, j] *= np.exp(-0.5 * delt / tau[j]) * scale
            elif k != state and j == state:
                A[k, j] *= np.exp(-0.5 * delt / tau[k]) * scale
    return A
