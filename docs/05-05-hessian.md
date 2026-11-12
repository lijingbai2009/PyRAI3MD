---
title: "5.5. HESSIAN"
nav_order: 5
parent: "5. Keyword sections"
---

The hessian section configures the curvature request of jobtype hessian, frequency, num_hess and num_freq. Two questions are answered separately: hess_driver says who produces the matrix, the framework displacing the geometry or the backend computing it for itself, and hess_mode says how it is produced, by an analytical second derivative or by a finite difference. The section also decides whether the matrix is read into normal modes and thermochemical corrections and which atoms are displaced.

The section is always present with its defaults, so hess_driver and hess_mode are unset by default. An unset value lets the jobtype token choose the route, which is what keeps a backend matrix (hessian, frequency) and a framework displacement batch (num_hess, num_freq) apart without a second keyword. The pairings that have no producer behind them are refused where the input is still cheap to change.

The keywords, default values, and short descriptions are listed below.

**&HESSIAN**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| multitask | None | Runs the independent displaced evaluations of the batch concurrently when true and one at a time when false. Unset leaves the choice to the workflow… |
| remotetask | None | Submits each displaced evaluation to the cluster through the slurm section of the environment when true. Unset leaves the choice to the workflow… |
| multipool | 1 | Sets the number of workers of the dispatch fan-out of this workflow node. A numerical Hessian is 6N independent displaced evaluations, which is the… |
| hess_driver | None | Chooses who produces the force constant matrix. main makes the framework displace the geometry and difference the gradients, and skill asks the… |
| hess_mode | None | Chooses how the force constants are produced. analytic asks for the analytical second derivative of the backend, numerical displaces the coordinates… |
| hess_step | 0.005 | Sets the displacement of each mobile Cartesian coordinate, in Angstrom. The gradient is differenced in Hartree per Bohr, so the quotient carries the… |
| hess_vibration | true | Reads the matrix into vibrational frequencies and normal modes. False leaves the run with the force constants alone. The default is true. |
| hess_thermo | true | Carries the modes on into the thermochemical corrections at the declared temperature and pressure. It is read only when the vibrational analysis… |
| hess_molden | false | Writes the frequencies and normal modes to &lt;title&gt;.freq.molden in the run directory. That name is the record the initial-condition sampler looks for… |
| hess_atoms | [] | Selects the mobile atoms of a partial Hessian, counted from one; the atoms it does not name are held. An empty list displaces every atom. The value… |
| hess_temperature | 298.15 | Sets the temperature of the thermochemical corrections, in kelvin. The default is 298.15. |
| hess_pressure | 1 | Sets the standard-state pressure of the thermochemical corrections, in bar. It enters the translational partition function through the factor kT over… |
| hess_scale | 1 | Sets the scaling factor applied to every harmonic frequency before the modes are read or written out. The default is 1.0. |
| hess_cutoff | 100 | Sets the wavenumber, in cm-1, below which the vibrational entropy starts to blend into the free-rotor form. Zero turns the blending off and leaves… |
| hess_alpha | 4 | Sets the exponent of the quasi-harmonic blend below hess_cutoff. The default is 4.0. |
| hess_symmetry | 1 | Sets the rotational symmetry number of the molecule in the thermochemical corrections. The default is 1, so an uncorrected symmetry leaves the… |
| hess_linear | None | Declares the structure collinear, which costs the rotor one degree of freedom. Unset lets the analysis detect it from the geometry. |

Full descriptions for all available keywords are summarized below.

**multitask** - Runs the independent displaced evaluations of the batch concurrently when true and one at a time when false. Unset leaves the choice to the workflow class default.

**remotetask** - Submits each displaced evaluation to the cluster through the slurm section of the environment when true. Unset leaves the choice to the workflow class default.

**multipool** - Sets the number of workers of the dispatch fan-out of this workflow node. A numerical Hessian is 6N independent displaced evaluations, which is the case this pool exists for. The default is 1, which keeps execution serial.

**hess_driver** - Chooses who produces the force constant matrix, one of main and skill. main displaces the geometry and differences the gradients the framework is already entitled to, one child runtime per displaced point; skill asks the backend for the matrix it computes for itself, and whether that backend answers with an analytical second derivative or with displacements of its own is the backend's keyword, so from this side the matrix simply arrives in the result. An unset value lets the jobtype token decide, num_hess and num_freq naming the framework and hessian and frequency naming the backend. main with hess_mode analytic or auto is refused, because the framework has no analytical second derivative and auto is the statement that no producer has been settled on.

**hess_mode** - Chooses how the force constants are produced, one of analytic, numerical and auto. analytic asks the backend for its own second derivative, numerical displaces each mobile coordinate by hess_step and differences the gradients, and auto takes the analytical matrix when the evaluation returns one and falls back to the displacement route when it does not, which the run log states. Two pairings are refused: main with analytic and main with auto. skill with numerical is legal and means the engine displaces its own atoms. An unset value lets the jobtype token choose.

**hess_step** - Sets the displacement of each mobile Cartesian coordinate, in Angstrom. The numerical route evaluates the gradient at plus and minus this displacement and forms the central difference, which is 2 times 3N evaluations for a full Hessian, each in its own child runtime. The gradients are Hartree per Bohr, so the difference quotient carries the factor between the two units. The default is 0.005.

**hess_vibration** - Reads the matrix into vibrational frequencies, normal modes, reduced masses and rotational constants. The vibrational problem is built from the atoms the batch displaced, the six rigid directions are removed by rank, and an imaginary mode is reported as a negative wavenumber. Setting it false leaves the run with the force constants alone, which is what a search that only wants the curvature asks for. The default is true.

**hess_thermo** - Carries the modes on into the thermochemical corrections: zero point energy, internal energy, enthalpy, entropy, free energy and the two heat capacities, all in Hartree or Hartree per kelvin, with the pieces they were summed from kept beside them. The analysis is a rigid-rotor harmonic-oscillator sum whose vibrational entropy blends into a free-rotor form below hess_cutoff, damped by hess_alpha. It is read only when the vibrational analysis runs. The default is true.

**hess_molden** - Writes the frequencies and normal modes to &lt;title&gt;.freq.molden in the run directory. The exact name matters: it is the record the initial-condition sampler reads when a dynamics run asks for frequencies. The file is written only when this keyword is true. The default is false.

**hess_atoms** - Selects the atoms of a partial Hessian, counted from one as every atom selection in an input is; the atoms it does not name are held. The analysis re-reads the mobile set from the displacement rows, so a held atom does not appear as a soft mode. During a search the frozen set belongs to the constraint plan: a hess_atoms set that matches the plan is accepted, one that clashes with it is refused, and one written with no fix behind it is refused because the matrix would hold rows of zeros on atoms the run may move. An empty list displaces every atom.

**hess_temperature** - Sets the temperature of the thermochemical corrections, in kelvin. The default is 298.15.

**hess_pressure** - Sets the standard-state pressure of the thermochemical corrections, in bar. It enters the translational partition function through the factor kT over p. The default is 1.0.

**hess_scale** - Sets the scaling factor applied to every harmonic frequency before the modes are read or written out. The default is 1.0.

**hess_cutoff** - Sets the wavenumber, in cm-1, below which the vibrational entropy starts to blend into the free-rotor form. Zero turns the blending off and leaves the plain harmonic oscillator. The default is 100.0.

**hess_alpha** - Sets the exponent of the quasi-harmonic blend below hess_cutoff. The default is 4.0.

**hess_symmetry** - Sets the rotational symmetry number of the molecule in the thermochemical corrections. The default is 1, so an uncorrected symmetry leaves the entropy high by k ln sigma.

**hess_linear** - Declares the structure collinear, which costs the rotor one degree of freedom. Unset lets the analysis detect it from the geometry.

A vibrational analysis and a variable cell are refused together, because the six lattice entries have no mass to be weighted by, and a backend matrix is refused for a queue of more than one backend. The matrix returned by a backend and the matrix assembled by the framework are written to the same result key, so a reader cannot tell which producer answered.
