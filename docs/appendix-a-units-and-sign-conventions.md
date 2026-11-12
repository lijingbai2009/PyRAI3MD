---
title: "A. Units and sign conventions"
nav_order: 1
parent: "Appendices"
---

The framework converts every quantity once, at its boundary, into the atomic unit set the workflows integrate in: Bohr, Hartree, the atomic unit of time and the electron mass. An input is written in the units this appendix lists, a skill may compute in its own units if its manifest says so, and the log reports in the conventional unit named in each block title. The table below is the whole contract; where an input unit differs from the boundary unit, the conversion and its direction are named.

| Quantity | Unit at the boundary | Input unit | Note |
| :--- | :--- | :--- | :--- |
| Cartesian coordinates | Angstrom | Angstrom | Input, xyz files and every trajectory record use Angstrom; the geometry is converted to Bohr once when the state is built. |
| Cell parameters | Angstrom and degree | Angstrom and degree | cell is a, b, c, alpha, beta, gamma. lattice_vector is Angstrom and wins over cell when both are written. |
| Energy | Hartree | Hartree | Skills that declare eV or kcal/mol in their manifest are converted by the framework, not by the caller; the value the workflow sees is always Hartree. |
| Energy threshold | Hartree | Hartree | maxenergy, minenergy and the md md_max\* thresholds are Hartree. The one exception is adaptive maxmae, which is a validation energy MAE in eV. |
| Gradient | Hartree/Bohr, dU/dR | Hartree/Bohr | The stored array is the derivative of the energy with respect to the coordinate, not a force. An engine that returns forces has them negated and converted once, so a downhill direction is a negative gradient. |
| Force (engine native) | Hartree/Bohr after conversion | engine native | A classical engine returns force, not gradient. The manifest unit table converts the energy-per-length factor and the sign together; a caller that also negates a converted array inverts the dynamics. |
| Stress tensor | Hartree/Bohr^3 | bar, as a criterion | The full 3x3 tensor is Hartree/Bohr^3. opt_pressure and md pressure are read in bar, and the optimizer and barostat compare against a pressure derived from the tensor. Engines that print GPa are converted centrally. |
| MD time | atomic unit of time | atomic unit of time | md size is the timestep in au and the trajectory time column is au. One au is 0.02418884254 fs. |
| Thermostat and barostat time | atomic unit | fs | thermostat_tau and barostat_tau are written in fs, which is the literature unit, and converted to au. The legacy constant is about 41.34 au per fs. |
| Temperature | K | K | temperature is kelvin at the boundary and in the log; the thermostat and the Boltzmann factors work in Hartree internally. |
| Pressure | bar | bar | pressure, opt_pressure and every pressure column of md.ensemble are bar. One Hartree/Bohr^3 is 2.9421e7 bar, so a barostatted run reads very small numbers in atomic units. |
| Mass | amu | amu | mass is amu in the input and in the log. Internally the mass is in electron masses; one amu is 1822.8885 electron masses. |
| Velocity | Bohr per atomic time | Bohr per atomic time | velocities in the input, in &lt;title&gt;.init.xyz and in md.velo are the same unit, which is what lets a sampled batch be re-read without rescaling. |
| Helicity and spin-orbit coupling | cm-1 | cm-1 | The &spin-orbit coupling block, the soc array and the soc_pair_map are in cm-1 at every layer, so no conversion is applied. |
| State gap | Hartree | Hartree | State energies, gaps and the analysis gap records are Hartree. Neural-network error metrics, including maxmae, are reported in eV. |
| Dipole moment | Debye in the log | not an input | One atomic unit of dipole is 2.541746 Debye. The log prints Debye; the result array keeps the skill value the log was formatted from. |
| Transition dipole | Debye in the log | not an input | Ordered by transition_pair, which is 1-based in the input and printed 1-based. The oscillator strength beside it is dimensionless. |
| Frequency | cm-1 | not an input | Vibrational frequencies are cm-1 and an imaginary mode is printed as a negative number, in the log, in the molden record and in the frequency array. |
| Hessian | Hartree/Bohr^2 | Angstrom, as a step | The matrix is Hartree/Bohr^2. hess_step is a displacement in Angstrom, so the difference quotient carries the Bohr conversion and a step written in Bohr gives a matrix off by 1.89. |
| Nonadiabatic coupling | Hartree/Bohr | not an input | The nac array and the &nonadiabatic coupling block use the energy derivative coupling convention, in Hartree/Bohr, not the wavefunction-overlap convention. |

Two sign conventions are worth stating separately because they are the two that produce silently wrong physics rather than a refused input. The first is the gradient sign: the framework stores dU/dR, so the force an engine reports is negated on the way in and never again. The second is the stress sign: the tensor is the derivative of the energy with respect to strain, and the pressure a barostat targets is minus one third of its trace, so a positive pressure compresses the cell. A run that moves the box the wrong way usually has a sign flipped on one of these two, not a unit error.

Masses, charges and the electron count follow the same rule. A charge is in units of the elementary charge everywhere; multiplicity is a positive integer, never a spin; and the total electron count that the state derives is the number the check compares against the multiplicity. [Section 5.2](05-02-system-molecule.html) lists the fields and Section B lists the quantities the state derives from them.
