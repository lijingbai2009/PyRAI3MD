---
title: "H. Glossary"
nav_order: 8
parent: "Appendices"
---

The terms below are used with this meaning throughout the manual.

| Term | Meaning |
| :--- | :--- |
| ab initio dynamics | molecular dynamics whose energies and gradients come from an electronic-structure calculation at every step |
| adiabatic state | an eigenstate of the electronic Hamiltonian at the current nuclear geometry |
| adaptive sampling | an iterative loop that runs trajectories on the current model, selects uncertain structures, labels them with a reference method and retrains |
| barostat | the part of an NPT integrator that changes the cell to hold a target pressure |
| capability | a property or operation an engine declares, such as energy, gradient, nac, soc, stress or field; the planner reads these declarations |
| child runtime | a serialized run that the parent launches as a separate process or scheduler job |
| CI space | the configuration-interaction ladder given by ci: how many states each spin multiplicity contains |
| composite | a method queue of two or three engines whose results are combined, as in ONIOM or QM/MM |
| constraint | an imposed geometric condition: a bias, a wall, a fixed atom, SHAKE or a metadynamics bias |
| decoherence | a correction that removes unphysical population transfer between trajectories in surface hopping |
| driver | the owner of an iteration: main for the framework, skill for the engine itself |
| ensemble | the statistical-mechanical setting of a trajectory: NVE, NVT or NPT |
| ESPF | electrostatic potential fitting, the way the classical environment is passed to the quantum region in QM/MM |
| FSSH | fewest-switches surface hopping, a stochastic algorithm that distributes trajectories over electronic states |
| GSH | the surface-hopping variant that propagates a pure-state density and needs all-state gradients |
| kTDC | curvature-driven time-derivative coupling, a gap-based approximation to the nonadiabatic coupling |
| LZSH | Landau-Zener surface hopping, which needs state energies rather than couplings |
| MECI | minimum-energy conical intersection, the lowest point of a crossing seam between two states |
| MECP | minimum-energy crossing point between two spin states |
| minimum image | the convention that a pair interaction uses the closest periodic copy of the partner |
| multiscale | a calculation that combines methods of different cost on different regions of one system |
| ONIOM | a subtractive composite scheme in which a small region is treated at a high level and the rest at a lower level |
| PBC | periodic boundary conditions |
| plan | the resolved list of tasks, engines and resources that a request produces before any calculation starts |
| profile | the computation chain an engine runs, for example a gradient profile or an optimization profile |
| reference queue | the method queue used to label structures during adaptive sampling |
| skill | a plugin that owns one engine: its schema, input builder, runner and parser |
| surface hopping | a mixed quantum-classical dynamics method in which nuclei follow one adiabatic surface and occasionally switch |
| thermostat | the part of an NVT integrator that holds a target temperature |
| TS | transition state, a first-order saddle point of the potential energy surface |
| UDD | undersampling-driven deviation, a bias added to a region the training data covers poorly |
