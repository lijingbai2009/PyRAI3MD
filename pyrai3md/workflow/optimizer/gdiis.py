"""
======================================================================
This module provides the GDIISOptimizer for geometry optimization
using the Gradient DIIS (GDIIS) algorithm. GDIIS is an alternative
optimization strategy that uses a subspace of previous gradients
and coordinates to construct an improved geometry through a
least-squares extrapolation.

The DIIS matrix is built from gradient inner products with a
constraint row enforcing coefficient normalization. The extrapolated
geometry combines the coordinate history with a preconditioned
gradient correction.

Aug 07 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from .base import Optimizer, usable_curvature_product


class GDIISOptimizer(Optimizer):
    """Optimizer for energy minima using the GDIIS algorithm.

    Maintains a sliding window of previous coordinates and gradients.
    The DIIS coefficients are obtained by solving the constrained
    least-squares problem. The extrapolated geometry is:

        x_new = sum_i c_i * x_i  -  H^{-1} * sum_i c_i * g_i

    where c_i are the DIIS coefficients and H is the (approximate)
    Hessian (identity by default, or BFGS-updated).

    Parameters
    ----------
    maxiter : int
        Maximum optimization iterations.
    energy_threshold : float
        Energy convergence threshold.
    gradient_threshold : float
        Gradient RMS convergence threshold.
    displacement_threshold : float
        Displacement RMS convergence threshold.
    trust_radius : float
        Trust radius for step scaling.
    window_size : int
        Number of history points used for the DIIS extrapolation.
    use_rfo : bool
        Accepted for interface parity with MinimaOptimizer but
        ignored: GDIIS builds its step from the DIIS extrapolation
        and has no RFO variant.
    """

    def __init__(self, maxiter=100, energy_threshold=1e-6,
                 gradient_threshold=3e-4, displacement_threshold=3e-4,
                 trust_radius=0.3, window_size=6,
                 use_rfo=False,
                 **kwargs):
        del use_rfo  # GDIIS has no RFO variant
        super().__init__(
            maxiter=maxiter,
            energy_threshold=energy_threshold,
            gradient_threshold=gradient_threshold,
            displacement_threshold=displacement_threshold,
            trust_radius=trust_radius,
            **kwargs,
        )
        self.window_size = int(window_size)
        # Inverse Hessian for preconditioning (None = identity)
        self._inv_hessian = None

    # ---- Algorithm implementation ----

    def _compute_step(self, gradient):
        """Compute the GDIIS step.

        On the first call (no history) returns a steepest descent
        step. On subsequent calls builds the DIIS matrix from the
        gradient history, solves for the coefficients, and
        constructs the extrapolated step.

        Parameters
        ----------
        gradient : np.ndarray
            Current flattened gradient (3N,).

        Returns
        -------
        np.ndarray
            Displacement vector (3N,).
        """
        n = len(gradient)

        # First step: steepest descent within trust radius
        if len(self.history['gradients']) < 2:
            return -gradient * self.trust_radius

        # Collect the gradient history window
        grads = self.history['gradients'][-self.window_size:]
        coords = self.history['coordinates'][-self.window_size:]

        # Solve for DIIS coefficients
        coefficients = self._solve_diis(grads)

        # Build the extrapolated step
        # x_extrapolated = sum_i c_i * x_i
        x_extrap = np.zeros(n)
        g_extrap = np.zeros(n)
        for c_i, x_i, g_i in zip(coefficients, coords, grads):
            x_extrap += c_i * x_i
            g_extrap += c_i * g_i

        # Preconditioned gradient correction
        if self._inv_hessian is not None:
            correction = self._inv_hessian @ g_extrap
        else:
            correction = g_extrap

        # Step = extrapolated coordinates - current coordinates
        # minus the preconditioned gradient correction
        current_coords = self.history['coordinates'][-1]
        step = x_extrap - current_coords - correction

        # Update the inverse Hessian with BFGS if enough history
        if len(self.history['gradients']) >= 2:
            s = (self.history['coordinates'][-1]
                 - self.history['coordinates'][-2])
            y = (self.history['gradients'][-1]
                 - self.history['gradients'][-2])
            self._bfgs_update(s, y)

        return step

    def _step_gradient(self, gradient):
        """Extrapolate and precondition on the tangential gradient.

        Every quantity this optimizer builds is a gradient combination - the
        DIIS matrix `B_ij = g_i . g_j`, the extrapolated `g_extrap` it inverts,
        and the `s . y` product of its own BFGS update - and all of them are read
        out of the history, which `Optimizer.run` now fills with the tangential
        vector the step is solved against. That alignment is the whole reason:
        an extrapolation across a window that includes constraint reactions on
        pinned coordinates is an extrapolation of the constraint, not of the
        surface, and the coefficients follow it.
        """
        return self._tangent_of(gradient)

    def _solve_diis(self, gradients):
        """Solve the DIIS linear system for coefficients.

        Constructs the DIIS matrix B where B_ij = g_i . g_j,
        augmented with a constraint row (sum c_i = 1). Solves the
        bordered linear system:

            [[B, -1]] [[c]]   [[0]]
            [[-1, 0]] [[L]] = [[-1]]

        Parameters
        ----------
        gradients : list of np.ndarray
            Gradient history window.

        Returns
        -------
        np.ndarray
            DIIS coefficients (length = len(gradients)).
        """
        m = len(gradients)
        if m == 0:
            return np.array([])
        if m == 1:
            return np.array([1.0])

        # Build the DIIS matrix
        B = np.zeros((m + 1, m + 1))
        for i in range(m):
            for j in range(m):
                B[i, j] = float(np.dot(gradients[i], gradients[j]))
            B[i, m] = -1.0
            B[m, i] = -1.0
        B[m, m] = 0.0

        # Right-hand side
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0

        # Solve with regularization for numerical stability
        try:
            # Add small regularization to the diagonal
            B_reg = B.copy()
            for i in range(m):
                B_reg[i, i] += 1e-8
            solution = np.linalg.solve(B_reg, rhs)
        except np.linalg.LinAlgError:
            # Fallback: equal weights
            self.logger.warning(
                "GDIIS: DIIS matrix singular, using equal weights.")
            return np.ones(m) / m

        coefficients = solution[:m]
        return coefficients

    def _bfgs_update(self, s, y):
        """Update the inverse Hessian with the BFGS formula.

        H_new = (I - rho s y^T) H (I - rho y s^T) + rho s s^T

        Preserves symmetry and positive definiteness; skipped when
        s . y is not positive.

        Parameters
        ----------
        s : np.ndarray
            Displacement vector (3N,).
        y : np.ndarray
            Gradient change (3N,).
        """
        sy = usable_curvature_product(s, y)
        if sy is None:
            return

        n = len(s)
        if self._inv_hessian is None:
            self._inv_hessian = np.eye(n)

        H = self._inv_hessian

        rho = 1.0 / sy
        # The rank-2 form of `(I - rho s y^T) H (I - rho y s^T) + rho s s^T`:
        # one matrix-vector product and three outer products instead of two dense
        # n by n multiplications. The two are the same matrix; the product form
        # costs O(n^3) on every step, which on a 1575-atom box (n = 4731) is
        # seconds against the tens of milliseconds a force-field evaluation of
        # the same step takes. `minima._bfgs_update` carries the same form.
        Hy = H @ y
        yHy = float(np.dot(y, Hy))
        self._inv_hessian = (H
                             - rho * (np.outer(s, Hy) + np.outer(Hy, s))
                             + (rho + rho * rho * yHy) * np.outer(s, s))
