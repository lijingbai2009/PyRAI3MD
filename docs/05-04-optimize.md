---
title: "5.4. OPTIMIZE"
nav_order: 4
parent: "5. Keyword sections"
---

The optimize section configures the geometry search started by jobtype optimize. It names the algorithm, the five thresholds that judge the atomic step, the step bound, and the options that make the lattice a variable. The section is read only by an optimize run: the other workflows keep their own sections.

The path methods mep, irc and neb are written as values of opt_method, but no optimizer steps them. A run that names one is routed to the Path workflow, which walks a sequence of constrained points or relaxes a band of images between two endpoint files, and the path keywords of this section are read there. The single-structure algorithms are minima, gdiis, ts, meci and mecp.

The keywords, default values, and short descriptions are listed below.

**&OPTIMIZE**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| multipool | 1 | Sets the number of workers of the dispatch fan-out of this workflow node. A NEB band fans its image evaluations out over this pool, while the… |
| opt_method | minima | Sets the search algorithm. minima descends to the nearest minimum, gdiis accelerates the same descent with a DIIS extrapolation, ts walks to a… |
| opt_maxit | 100 | Sets the maximum number of optimization steps. A run that reaches this count without meeting the thresholds is reported as failed with the message… |
| opt_energy | 1e-06 | Sets the energy convergence threshold, in Hartree. The step is converged when the change in energy between two successive steps is smaller than this… |
| opt_grms | 0.0003 | Sets the convergence threshold of the RMS gradient, in Hartree per Bohr. The five thresholds of the atomic step must all be met at once. The default… |
| opt_gmax | 0.00045 | Sets the largest allowed component of the gradient, in Hartree per Bohr. The default is 4.5e-4. |
| opt_drms | 0.0012 | Sets the convergence threshold of the RMS displacement of the last step, in Angstrom. The default is 1.2e-3. |
| opt_dmax | 0.0018 | Sets the largest allowed component of the displacement of the last step, in Angstrom. The default is 1.8e-3. |
| opt_csrms | 40 | Sets the convergence threshold of the RMS cell stress, in bar. It is judged only when optimize_cell is true, and the cell half of the run is… |
| opt_csmax | 60 | Sets the largest allowed component of the cell stress, in bar. It is read only with optimize_cell. The default is 60.0. |
| opt_cerms | 5e-05 | Sets the convergence threshold of the RMS cell strain, which is the displacement of a lattice row divided by the length of that row and is therefore… |
| opt_cemax | 7e-05 | Sets the largest allowed component of the cell strain, dimensionless. It is read only with optimize_cell. The default is 7e-05. |
| opt_pressure | 0 | Sets the external pressure the variable-cell search works against, in bar and tensile positive. The objective becomes the enthalpy E plus pV, so the… |
| opt_cell_coupling | anisotropic | Chooses how many lattice directions one pressure value may move. isotropic scales the box as a whole and anisotropic lets each of the six relaxed… |
| opt_trust_radius | None | Sets the step bound of the search, in Bohr. The bound clips the joint nuclear and lattice step, so it decides whether a pressure band is reachable… |
| opt_use_rfo | false | Requests a rational function optimization step from the augmented Hessian instead of the quasi-Newton step. The minima search honours it, gdiis has… |
| opt_state_i | 0 | Sets the electronic state the search follows, counted from zero. A multi-center run may give one index per center, and a single value is read by… |
| opt_state_j | None | Sets the second surface of a two-state search, counted from zero, for meci and mecp. Unset means there is no pair and the run follows one surface… |
| optimize_cell | false | Makes the six lattice components part of the optimized vector, so the box relaxes together with the atoms. The run needs a backend that reports… |
| opt_path_constraint | sphere | Chooses the algorithm that holds a path point on its hypersphere. sphere takes the mass-weighted Gonzalez-Schlegel constrained step and relu… |
| opt_path_radius | 0.1 | Sets the arc length between two path points, in Bohr, which is also the radius of the hypersphere a point is found on. The value must be positive… |
| opt_path_inner_maxit | 20 | Sets the number of probe steps the constrained minimization of one path point may take. The value must be at least 1. The default is 20. |
| opt_path_inner_tol | 0.0003 | Sets the convergence tolerance of the probe that finds one path point. Left at its default the run uses opt_grms instead, so one keyword drives a… |
| opt_irc_direction | both | Chooses the side of the saddle an intrinsic reaction coordinate walks: backward, forward, or both, which runs one walker per side from the same… |
| opt_irc_mode | hessian | Chooses where the imaginary mode that seeds the IRC comes from. hessian computes it with the framework Hessian and molden reads it from a frequency… |
| opt_irc_mode_file | '' | Names the frequency file the imaginary mode is read from when opt_irc_mode is molden. It is required in that case and is not read otherwise. The… |
| opt_neb_reactant | '' | Names the file that holds the starting structure of a band. It is required for neb, together with opt_neb_product, and the atom order of the two… |
| opt_neb_product | '' | Names the file that holds the final structure of a band. It is required for neb, together with opt_neb_reactant. The default is empty. |
| opt_neb_ts | '' | Names an optional file that holds an intermediate guess, which is placed in the band between the two endpoints. It is read only for neb. The default… |
| opt_neb_images | 8 | Sets the number of interior images of a band. The band itself carries this number plus the two endpoints, and the value must be at least 1. The… |
| opt_neb_interpolate | linear | Chooses how the interior images are placed. linear uses fractional coordinates for a periodic band, and geodesic measures distances in a Morse scaled… |
| opt_neb_geodesic_threshold | 3 | Sets the pair cutoff radius of the geodesic metric, in Angstrom. It is read only with geodesic interpolation and must be positive. The default is 3.0. |
| opt_neb_geodesic_scaler | 1.7 | Sets the alpha parameter of the Morse metric that scales the geodesic distances. It is read only with geodesic interpolation and must be positive… |
| opt_neb_geodesic_min_neighbors | 4 | Sets the minimum coordination number of an atom in the pair list the geodesic metric is built from. The value must be at least 1. The default is 4. |
| opt_neb_spring | 0.01 | Sets the spring constant between neighbouring images, in Hartree per Bohr squared. A value of zero is allowed. The default is 0.01. |
| opt_neb_climb | false | Turns the highest image into a climbing image once the band is relaxed enough, so that the image converges on the saddle instead of on the chord… |
| opt_neb_climb_threshold | 0.01 | Sets the largest perpendicular force below which the climbing image is armed. It is read only with opt_neb_climb and is refused without it. The… |
| opt_neb_step | 0.1 | Sets the initial timestep of the FIRE relaxation of the band, in atomic units. The value must be positive. The default is 0.1. |

Full descriptions for all available keywords are summarized below.

**multipool** - Sets the number of workers of the dispatch fan-out of this workflow node. A NEB band fans its image evaluations out over this pool, while the single-structure loop steps one geometry at a time. The default is 1, which keeps execution serial.

**opt_method** - Sets the search algorithm, and with it the loop the run takes. minima is an inverse-Hessian BFGS descent with a trust radius, gdiis adds a DIIS extrapolation on top of a BFGS preconditioner, ts follows the lowest mode from a partitioned RFO step to a first-order saddle, meci locates a minimum energy conical intersection, and mecp locates a minimum energy crossing point. The last two read the state pair from opt_state_i and opt_state_j, and meci also uses the derivative coupling. The three path names mep, irc and neb are values of this keyword, but the framework routes them to the Path workflow rather than to an optimizer, and the Optimize workflow refuses a path name that reaches it.

**opt_maxit** - Sets the maximum number of steps of the search. The loop stops at this count whether or not the thresholds were met, and a run that ends there is marked failed with Maximum optimization steps reached rather than reported as converged. Only a resumed runtime file that carries its own step budget can lower the bound; a run that says nothing else is bounded by this keyword alone. The default is 100.

**opt_energy** - Sets the energy convergence threshold, in Hartree. The step is converged when the change in energy between two successive steps is smaller than this value. The default is 1e-6.

**opt_grms** - Sets the convergence threshold of the RMS gradient, in Hartree per Bohr. The five thresholds of the atomic step must all be met at once. The default is 3e-4.

**opt_gmax** - Sets the largest allowed component of the gradient, in Hartree per Bohr. The default is 4.5e-4.

**opt_drms** - Sets the convergence threshold of the RMS displacement of the last step, in Angstrom. The default is 1.2e-3.

**opt_dmax** - Sets the largest allowed component of the displacement of the last step, in Angstrom. The default is 1.8e-3.

**opt_csrms** - Sets the convergence threshold of the RMS cell stress, in bar. It is judged only when optimize_cell is true, and the cell half of the run is converged only when all four of its thresholds are met. The default is 40.0.

**opt_csmax** - Sets the largest allowed component of the cell stress, in bar. It is read only with optimize_cell. The default is 60.0.

**opt_cerms** - Sets the convergence threshold of the RMS cell strain, which is the displacement of a lattice row divided by the length of that row and is therefore dimensionless. It is read only with optimize_cell. The default is 5e-05.

**opt_cemax** - Sets the largest allowed component of the cell strain, dimensionless. It is read only with optimize_cell. The default is 7e-05.

**opt_pressure** - Sets the external pressure of a variable-cell run, in bar and tensile positive like the stress it is added to. The objective becomes the enthalpy E plus pV, so the relaxed box is the one whose stress equals minus this pressure. The keyword is refused without optimize_cell, because a held box has no lattice derivative for a pressure to act on; a box under pressure without a relaxation is a dynamics run with a barostat. The default is 0.0, which asks for a stress-free box.

**opt_cell_coupling** - Chooses how many lattice directions one pressure value may move. anisotropic, the default, lets each of the six relaxed components answer for itself. isotropic scales the box, so only the mean normal stress and the single scale of the step are judged, and the deviatoric and shear readings are reported but not held to a threshold, because a shape-preserving step has no direction to relieve them. The two names are the names of the dynamics barostat modes. The keyword needs optimize_cell.

**opt_trust_radius** - Sets the step bound of the search, in Bohr. The radius clips the joint nuclear and lattice step, so the pressure band a box is judged against is reachable or not depending on this number rather than on opt_maxit. An unset value is not a number: it leaves each algorithm on its own starting radius, 0.3 Bohr for minima and gdiis and 0.05 Bohr for the transition-state search, which is what the shipped examples run. A value of zero is refused, because it would freeze a search that then reports that it moved nothing.

**opt_use_rfo** - Requests the rational function optimization step from the augmented Hessian in place of the plain quasi-Newton step. Only the minima search honours the flag: gdiis has no RFO variant and discards it, and the transition-state search always forms a partitioned RFO step because mode following needs the partitioned curvature. The step is clipped by the same trust radius as the quasi-Newton step. The default is false.

**opt_state_i** - Sets the electronic state the search follows, counted from zero. A multi-center run may give one index per center, and a single value is read by every center. The default is 0.

**opt_state_j** - Sets the second surface of a two-state search, counted from zero, for meci and mecp. Unset means there is no pair and the run follows one surface. The default is unset.

**optimize_cell** - Makes the box a variable: the six movable lattice components join the optimized vector, the stress conjugate to them joins the gradient, and the atoms ride the box with their fractional coordinates held. The planning layer derives the stress capability from this keyword and passes the whole-system stress tensor to the skill, so a backend without that capability is refused rather than relaxed in the box it was given. The keyword is refused with a driver other than main, because an engine that owns the steps commits no lattice, and with a vibrational analysis, because the lattice entries have no mass to be weighted by.

**opt_path_constraint** - Chooses the algorithm that holds a path point on the hypersphere of radius opt_path_radius around the accepted point. sphere is the mass-weighted Gonzalez-Schlegel constrained step, whose rotation shrinks when the probe energy fails to fall. relu minimizes the energy plus a one-sided penalty that acts only outside the shell. The keyword belongs to mep and irc and is refused on a band, which is a chain of images rather than a walk on a sphere.

**opt_path_radius** - Sets the arc length between two path points, in Bohr, which is also the radius of the hypersphere a point is found on. The value must be positive. The default is 0.1.

**opt_path_inner_maxit** - Sets the number of probe steps the constrained minimization of one path point may take. The value must be at least 1. The default is 20.

**opt_path_inner_tol** - Sets the convergence tolerance of the probe that finds one path point. Left at its default the run uses opt_grms instead, so one keyword drives a point and the walk alike. The value must be positive and defaults to 3e-04.

**opt_irc_direction** - Chooses the side of the saddle an intrinsic reaction coordinate walks: backward, forward, or both, which runs one walker per side from the same saddle. It is read only for irc. The default is both.

**opt_irc_mode** - Chooses where the imaginary mode that seeds the IRC comes from. hessian computes it with the framework Hessian and molden reads it from a frequency file named by opt_irc_mode_file. It is read only for irc and defaults to hessian.

**opt_irc_mode_file** - Names the frequency file the imaginary mode is read from when opt_irc_mode is molden. It is required in that case and is not read otherwise. The default is empty.

**opt_neb_reactant** - Names the file that holds the starting structure of a band, one frame in the format the molecular section accepts. A neb run is refused without both endpoints, because the band is interpolated between them and a run that names none has nothing to relax. The atom count and atom order must agree with the product file, and a periodic endpoint file must carry its cell comment.

**opt_neb_product** - Names the file that holds the final structure of a band. It is read together with opt_neb_reactant when the band is built, and a neb run that names only one endpoint is refused. The atom order and, for a variable-cell run, the lattice of the two files are checked at that point, because the section validator has no file to look at.

**opt_neb_ts** - Names an optional file that holds an intermediate guess for the band. When it is given, the guess becomes an interior image instead of a point the interpolation invents, which is how a band is kept from crossing a barrier the endpoints alone would not suggest. It is read only for neb and defaults to empty, which leaves the band to the interpolation.

**opt_neb_images** - Sets the number of interior images of the band. The band carries this number plus the two endpoints, so the default of 8 means ten frames, and the count also keeps every band dispatch above the single-unit identity allocation. The value must be at least 1, and geodesic interpolation asks for at least 2, because one interior image is a chord with nothing between it and its neighbours to smooth.

**opt_neb_interpolate** - Chooses how the interior images are placed between the two endpoints. linear, the default, is a straight interpolation in Cartesian coordinates and uses fractional coordinates for a periodic band. geodesic smooths the band in a Morse scaled internal-coordinate metric, which keeps a rotation from being read as a barrier. It needs at least two interior images and is refused for a band above 200 atoms.

**opt_neb_geodesic_threshold** - Sets the pair cutoff radius of the geodesic metric, in Angstrom. Every atom pair closer than this radius enters the internal coordinate list the metric is measured on. The value must be positive and is read only when opt_neb_interpolate is geodesic. The default is 3.0.

**opt_neb_geodesic_scaler** - Sets the alpha parameter of the Morse potential whose distances the geodesic metric uses, so that a stretched pair contributes less than a compressed one. It must be positive and is read only with geodesic interpolation. The default is 1.7.

**opt_neb_geodesic_min_neighbors** - Sets the minimum coordination number an atom keeps in the pair list of the geodesic metric, which stops a sparse or unusual structure from leaving an atom with no distance to anything. The value must be at least 1 and is read only with geodesic interpolation. The default is 4.

**opt_neb_spring** - Sets the spring constant that keeps neighbouring images apart, in Hartree per Bohr squared, so that the constant times a segment length in Bohr is a force in the unit the true force is read in. The springs act along the band only, and the true force acts perpendicular to it, which is what lets an image relax without sliding into a neighbour. A value of zero is allowed and leaves the band to the perpendicular force. The default is 0.01.

**opt_neb_climb** - Turns the highest image into a climbing image once the band is relaxed enough. The climbing image inverts the component of the true force along the band, so it climbs to the saddle instead of settling at the middle of the chord, and the latch is one-way: once armed, the image stays a climbing image for the rest of the run. The default is false.

**opt_neb_climb_threshold** - Sets the largest perpendicular force, in Hartree per Bohr, below which the climbing image is armed. The keyword is read only with opt_neb_climb and is refused without it, because the threshold is the reading that arms an image the run did not ask for. The default is 0.01.

**opt_neb_step** - Sets the initial timestep of the FIRE relaxation of the band, in atomic units. It is the timestep of the integrator that moves the images, not a step bound, and FIRE adjusts it as the band settles. The value must be positive. The default is 0.1.

Each path keyword answers a question exactly one of the three walks asks, so a key declared on a method that cannot read it is refused rather than carried: the hypersphere keys opt_path_constraint, opt_path_radius, opt_path_inner_maxit and opt_path_inner_tol belong to mep and irc, the IRC keys belong to irc, and the band keys belong to neb. A band also needs both endpoints and a climbing threshold only with a climbing image.
