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

Tully Model 3 (extended coupling with reflection), Tully 1990
example III.

Diabatic Hamiltonian (x in Bohr, values in Hartree):
    V11 =  A
    V22 = -A
    V12 =  B * exp(C x)         (x < 0)
           B * (2 - exp(-C x))  (x >= 0)

The coupling approaches 2B far to the right, keeping the
nonadiabatic region extended on that side.

Analytic derivatives:
    dV11/dx = 0
    dV12/dx =  C * B * exp(C x)    (x < 0)
               C * B * exp(-C x)   (x >= 0)

Both branches meet at x = 0 with the same value B and the
same derivative C*B, so V12 is continuously differentiable.

"""

import numpy as np

from .base import AnalyticModel


class Tully3(AnalyticModel):
    """Extended coupling / reflection benchmark model.

    Standard parameters from Tully 1990 example III.
    """

    name = 'tully3'
    PARAMETER_NAMES = ('tully_a', 'tully_b', 'tully_c')

    tully_a = 0.0006  # Hartree, constant diabatic splitting
    tully_b = 0.1     # Hartree, coupling strength
    tully_c = 0.9     # 1/Bohr, coupling decay rate

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
        if x < 0.0:
            v12 = self.tully_b * np.exp(self.tully_c * x)
        else:
            v12 = self.tully_b * (2.0 - np.exp(-self.tully_c * x))
        return np.array([[self.tully_a, v12],
                         [v12, -self.tully_a]])

    def hamiltonian_gradient(self, x):
        """Evaluate the analytic dH/dx at x.

        Both branches meet at x = 0 with the same derivative
        C*B, so the coupling is continuously differentiable and
        the branch choice at x = 0 is immaterial.

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
        if x < 0.0:
            dv12 = (self.tully_c * self.tully_b
                    * np.exp(self.tully_c * x))
        else:
            dv12 = (self.tully_c * self.tully_b
                    * np.exp(-self.tully_c * x))
        return np.array([[0.0, dv12], [dv12, 0.0]])
