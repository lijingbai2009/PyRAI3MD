"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

Tully Model 2 (dual avoided crossing), Tully 1990 example II.

Diabatic Hamiltonian (x in Bohr, values in Hartree):
    V11 =  0
    V22 = -A * exp(-B x^2) + E
    V12 =  C * exp(-D x^2)

The Gaussian barrier on V22 crosses the flat V11 twice,
producing the two avoided crossings of this model.

Analytic derivatives:
    dV11/dx = 0
    dV22/dx = 2 A B x * exp(-B x^2)
    dV12/dx = -2 C D x * exp(-D x^2)

"""

import numpy as np

from .base import AnalyticModel


class Tully2(AnalyticModel):
    """Dual avoided crossing benchmark model.

    Standard parameters from Tully 1990 example II.
    """

    name = 'tully2'
    PARAMETER_NAMES = ('tully_a', 'tully_b', 'tully_c', 'tully_d',
                       'tully_e')

    tully_a = 0.1     # Hartree, Gaussian barrier height
    tully_b = 0.28    # 1/Bohr^2, barrier width
    tully_c = 0.015   # Hartree, coupling strength
    tully_d = 0.06    # 1/Bohr^2, coupling width
    tully_e = 0.05    # Hartree, asymptotic energy shift of V22

    def hamiltonian(self, x):
        """Evaluate the diabatic Hamiltonian at x.

        Parameters
        ----------
        x : float
            Model coordinate in Bohr.

        Returns
        -------
        numpy.ndarray
            (2, 2) Hamiltonian matrix (Hartree).
        """
        x = float(x)
        v22 = (-self.tully_a * np.exp(-self.tully_b * x * x)
               + self.tully_e)
        v12 = self.tully_c * np.exp(-self.tully_d * x * x)
        return np.array([[0.0, v12], [v12, v22]])

    def hamiltonian_gradient(self, x):
        """Evaluate the analytic dH/dx at x.

        Parameters
        ----------
        x : float
            Model coordinate in Bohr.

        Returns
        -------
        numpy.ndarray
            (2, 2) derivative matrix (Hartree/Bohr).
        """
        x = float(x)
        dv22 = (2.0 * self.tully_a * self.tully_b * x
                * np.exp(-self.tully_b * x * x))
        dv12 = (-2.0 * self.tully_c * self.tully_d * x
                * np.exp(-self.tully_d * x * x))
        return np.array([[0.0, dv12], [dv12, dv22]])
