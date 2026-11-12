---
title: "B. Index bases and derived quantities"
nav_order: 2
parent: "Appendices"
---

Two different questions are answered here. The first is which index base a keyword uses, because the framework accepts atom and state numbers in more than one spelling and a reader who assumes one base for all of them will be wrong about at least one keyword. The second is which fields of the state are derived rather than read, because a derived field written into an input is either ignored or refused, and the number the run reports is the one it computed.

The rule for atom numbers is 1-based outside and 0-based inside. Every atom selection in an input counts from one, is converted once when the state is built, and is printed 1-based again. The two documented exceptions are center_groups, whose numbers have always been 0-based, and the molecule list of a wall, whose parsed form holds 0-based atom numbers. State numbers follow the opposite convention for the optimizer: opt_state_i and opt_state_j are 0-based in the input itself, while root, coupling and every state number in a log line are 1-based.

| Keyword | External base | Internal base | Note |
| :--- | :--- | :--- | :--- |
| highlevel, midlevel | 1-based | 0-based region index | The QM and mid regions of a layered run. Both accept a flat list or one list per center; the per-center form keeps the same base. |
| highref, midref | 1-based | 0-based region index | The reference regions used by adaptive sampling. They are parsed like highlevel and midlevel but never enter the theory center plan. |
| coupling | 1-based | 0-based pair map | Each pair is two state numbers written as i j, separated by commas. The pair maps nac_pair_map and soc_pair_map carry the converted pairs. |
| root | 1-based | 0-based active state | The starting state of a trajectory. The external spelling is 1-based, so root two is the second state; the state field carries the decremented value. |
| current_state | no external spelling | 0-based | The active state the run is on. It is set by root or by a hop, printed 1-based in the log and in the trajectory comment, and is never an input keyword. |
| opt_state_i, opt_state_j | 0-based | 0-based | The optimized surface and, for a crossing search, the second surface. Zero is the ground state, so naming an excited state means writing one, not two. |
| track_index | 1-based | 0-based pair list | The atom pairs or fragment groups of the geometry tracker. Each number is an atom index in the input and is decremented once. |
| center_groups | 0-based | 0-based | Fragment groups of a multi-center plan. This keyword is the inherited exception: its numbers were 0-based in the legacy input and remain so. |
| hess_atoms | 1-based | 0-based hessian_atoms | The mobile atoms of a partial Hessian in the input. The result field hessian_atoms is the converted 0-based list, and the matrix keeps zero rows and columns for the held atoms. |
| constraint atoms | 1-based | 0-based | groups atoms, bias atoms, wall atoms and ref_atoms, fix atoms and every atoms list of a hard constraint or a metadynamics entry count from one. |
| wall groups (molecules) | 1-based selection, 0-based molecules | 0-based molecules | A wall selection is 1-based like every other selection. The parsed molecules list, which the legacy wall_groups line fills, holds 0-based atom numbers and a molecule count per group. |
| permutation_map | 1-based file | 0-based expansion | A training permutation map. The file holds one atom order per line in the 1-based numbering of the structure, and the expansion converts it to the internal order. |

The derived table below is generated from the System model, so every row names a field that exists in this version. A field listed here is not an input keyword: writing one is either absorbed and overwritten or refused, and neither outcome is what an input author wanted. Persisted fields are the ones a runtime file restores; transient fields are recomputed inside the step that uses them.

| Derived field | What it holds |
| :--- | :--- |
| total_electron | Total electron count, the sum of the atomic numbers minus the total charge. It is the number the multiplicity check compares against. |
| nalpha | Number of alpha electrons, (total_electron + 2S) / 2, derived from the total electron count and the aggregate multiplicity. |
| nbeta | Number of beta electrons, (total_electron - 2S) / 2. |
| nstate | Number of electronic states on the state ladder, the sum of the ci entries across the spin multiplicities. |
| state_label | Per-state label, for example S0 or T1, built from the spin block and the position of the state inside it. |
| state_spin | Per-state 2S, so 0 is a singlet and 2 is a triplet. It is the value that decides which states may couple by SOC. |
| state_density | Electronic state density matrix, (nstate, nstate, 2) with the last axis carrying the real and imaginary parts. It is propagated by the hopping integrator and persisted across steps. |
| npair | Number of state pairs the input requested, the length of transition_pair. |
| nnac | Number of requested pairs that ask for a nonadiabatic coupling, the length of nac_pair_map. |
| nsoc | Number of requested pairs that ask for spin-orbit coupling, the length of soc_pair_map. |
| transition_pair | The requested pairs as they were written, as 0-based state indices in a (npair, 2) array. |
| nac_pair_map | Row index to state pair for the nac array, (nnac, 2). The coupling row of a pair is found here, never by guessing the pair order. |
| soc_pair_map | Row index to state pair for the soc array, (nsoc, 2). |
| target_coord | A target structure carried with the state for a workflow that needs one. It is empty unless a target was supplied, and it is never a second copy of the geometry. |
| boundary | Atom groups at a partition boundary, carried with the state so a layered run can name the atoms on the edge of a region. It is empty for an unpartitioned system. |
| volume | Cell volume in cubic Angstrom, derived from the lattice. Zero when the run is not periodic. |
| image | Per-atom count of cell lengths crossed by the coordinate wrap, (natom, 3). The coordinates stay wrapped, so this array is what makes a continuous trajectory and a cross-frame displacement recoverable. |
| thermostat_dof | Particle thermostat chain state, the chain positions followed by their momenta. Persisted across steps and restored by a runtime file; never sent to a skill. |
| barostat_strain | Full-tensor cell strain, (3, 3). Persisted, and the quantity the cell drift integrates. |
| barostat_momentum | Conjugate cell momentum, (3, 3), the variable the cell equation of motion advances. |
| barostat_velocity | Instantaneous cell velocity, momentum divided by the cell mass. It is transient inside one step and is not a checkpoint field. |
| barostat_thermostat_dof | Chain state of the thermostat that acts on the cell momentum, in the same layout as thermostat_dof. |

The pair maps deserve one sentence of emphasis. The nac and soc arrays are ordered by the maps, not by the order the pairs were typed, and a run that requests a subset of pairs has a shorter array than the state ladder suggests. Reading the coupling of a pair from an index into the array without going through nac_pair_map is therefore wrong as soon as the subset is not the full ladder. [Section 5.2](05-02-system-molecule.html) lists the input fields these quantities are derived from, and [Section 12.1](12-output-and-analysis.html#121-the-run-log) shows where each one is printed.
