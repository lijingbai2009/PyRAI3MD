---
title: "8. Periodic systems"
nav_order: 8
---

A periodic run adds a cell to every evaluation. The cell is declared once on the molecule section and travels with the coordinates of that evaluation, so a force, a virial and a recorded frame always describe the same box. The workflow does not change: the optimizer, the integrator and the composite layer ask for an energy and a gradient for a system that has a lattice.

## 8.1. Cell, lattice and periodic flags

pbc holds exactly three 0 or 1 flags and is the single periodicity authority. A partial spelling such as [1, 1, 0] is refused at the framework layer and again by the CP2K input writer, so only full 3D periodicity or a molecular run exists. A non-zero lattice no longer implies a periodic run, and a periodic run needs a real cell: with pbc on and no usable lattice, validation refuses the input.

cell takes six parameters, a b c alpha beta gamma in Angstrom and degrees, or nine matrix components. lattice_vector is the runtime authority, a 3x3 matrix of row vectors with R = S @ B between Cartesian and fractional coordinates, and lattice is the classic six-parameter spelling of the same thing. The six-parameter default 99 99 99 90 90 90 is the not-filled sentinel: no lattice is derived from it, so a molecular run never carries a fake 99 Angstrom cube, and a periodic run that leaves it has no cell. A cell and a lattice_vector that disagree are an error rather than a silent overwrite.

Minimum-image work is derived, not stored. Wrapping happens after position propagation and accumulates the crossings in image, and the continuous view is rebuilt from those counts, so a molecule that straddles a face is cut out whole for a region task. The minimum-image displacement is a certified search: it widens beyond the 27 nearest cells when a tilted shape demands it and raises rather than shortening the vector.

kpoint sets the k-point grid, which the backend reads. Every evaluation receives one frozen snapshot of coordinates, crossing counts and cell, and a region task is evaluated without the cell because a molecular engine gets a cluster.

## 8.2. Stress and variable-cell optimization

stress is a derived capability, asked for only in two situations: with ensemble npt, where the barostat needs the internal pressure, and with optimize_cell true, where the search needs the cell conjugate. Only the whole-system provider has to answer it. A backend reports its own declared unit, and the framework converts once to Hartree/Bohr^3 with tension positive; md.stress writes the nine components in that unit with the derived pressure in bar. The optimizer criteria are the pressure pair opt_csrms and opt_csmax in bar and the strain pair opt_cerms and opt_cemax, dimensionless.

opt_pressure, in bar, makes the optimizer objective an enthalpy, and opt_cell_coupling says how many directions that pressure may move: isotropic projects the lattice half of each step onto the shape-preserving direction and judges the mean normal stress, while anisotropic, the default, leaves all six relaxed components free. The conjugate is V sigma^T H^-T and not the volume times the stress, and the atoms ride the box: the fractional coordinates are held while the cell stretches.

Several combinations are refused rather than relaxed silently. opt_pressure or opt_cell_coupling without optimize_cell is refused. A backend that declares no stress raises by name, so a run cannot report cell rows for a box that never moved. A periodic run whose outermost layer is a molecular program is refused at validation. optimize_cell with a vibrational analysis is refused, because a lattice entry has no mass; a saddle with a backend Hessian is refused, because six zero rows are six directions of zero curvature. A deformation beyond what a lattice can carry raises instead of drifting into a NaN box, the cumulative strain bound is ln 1.5, and an anisotropic commit re-triangularizes the deformed lattice.

## 8.3. Dynamics under PBC

NVE and NVT run at a fixed cell, and NPT adds a barostat that integrates the cell. The thermostat and barostat are a literature pairing, not a free grid: parrinello-rahman and mttk accept only the Nose-Hoover family, because the chain on the particle momenta is part of their derivation, while berendsen and scr accept any thermostat other than off. scr is isotropic only, since its tensor form injects shear strain that a liquid cannot store.

The run writes md.stress only when a backend supplied a stress tensor, and md.ensemble only for an extended ensemble. The cell of a frame is written in the Lattice attribute of the md.xyz comment, while image and pbc are not written to xyz at all: a reader that needs them reads the extended record or the checkpoint. An engine that integrates its own box reports it back through fin_cell, and the framework re-derives cell, lattice and volume from it so the next single point is not run on a stale box.

The cell history is recomputed from the products by tools/analyze_npt_frames.py, which reports the frame count, the volume excursion against the first frame, the accumulated rotation of the simulation frame, the extremes of the strain and its momenta, the kinetic, virial and total pressure and the instantaneous temperature, naming the file each number came from. It reads products only.

## 8.4. Which engines support PBC

Support is declared in each manifest as supports_pbc, with the stress unit beside it. A method that only serves a centre region may be molecular even when the environment layer is periodic, because the runtime cuts that task out of the continuous view and clears its cell.

| Engine | Supports PBC | Stress | Notes |
| :--- | :--- | :--- | :--- |
| pyscf | No | No | Molecular only; frequency capability but no box |
| molcas | No | No | Molecular only; ESPF embedding is a cluster run |
| bagel | No | No | Molecular only |
| orca | No | No | Refused for periodic cells; molecular only |
| cp2k | Yes | GPa, converted | Analytical tensor, plus a numerical strain route for excited states |
| xtb | GFN-FF only | Hartree/Bohr^3 | gfn0, gfn1 and gfn2 are molecular; GFN-FF reads gradlatt |
| lammps | Yes | bar, converted | Classical provider and the outermost layer of periodic composites |
| qdm | No | No | Molecular only |
| nn_pes | No | No | Refuses any pbc: features are molecular |
