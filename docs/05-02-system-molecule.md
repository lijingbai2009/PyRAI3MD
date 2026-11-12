---
title: "5.2. SYSTEM (MOLECULE)"
nav_order: 2
parent: "5. Keyword sections"
---

The system section describes the molecule: the atoms and their geometry, the electronic states a run computes, the regions of a multiscale calculation, and the periodic cell. The classic spelling is &amp;MOLECULE; in YAML and agent JSON it is the system mapping, and the historical name molecule selects the same model.

The structure may be written in the section or read from a file: the coord key names a coordinate file, a same-basename .xyz beside the input is used when no coordinate file is named, and a top-level atoms list in YAML or agent JSON is taken as written. [Section 4](04-input-files-and-conventions.html) compares the three carriers and [Section 7](07-multiscale-calculations.html) explains how the region keys divide a multiscale system.

The keywords, default values, and short descriptions are listed below. A † marks a keyword that is accepted but has no effect in v3.0.0; ‡ marks one value that is honoured; § marks an alias; \* marks a derived quantity.

**&MOLECULE**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| atoms | [] | Lists the element symbols, one per atom, in the framework atom order that every other per-atom array follows. The value is normally resolved by the… |
| coordinates | [] | Gives the Cartesian position of every atom in Angstrom, one row per atom in the order of atoms. It is filled from the coord key or from the… |
| velocities | [] | Gives the Cartesian velocity of every atom in Bohr per atomic unit of time, one row per atom. It is read from the velo key or from the &lt;coord&gt;.velo… |
| mass | [] | Gives the atomic mass of every atom in amu. When the list is empty or its length disagrees with the atom count, the masses are derived from the… |
| natom | 0 | Records the number of atoms. It is derived from the atom list whenever the key is absent, and a value that disagrees with the length of atoms… |
| charges | 0 | Sets the total charge of the system in units of the elementary charge. A scalar is the single-center value and applies to every center, while a list… |
| multiplicity | 1 | Sets the spin multiplicity 2S+1 and must be at least 1 for every center. A scalar applies to every center and a list gives one value per center. The… |
| ci | [1] | Sets the number of electronic states computed in each spin multiplicity. One entry per multiplicity: ci 1 asks for one singlet and ci 1 1 asks for… |
| spin | [0] | Sets the spin of each multiplicity as its number of unpaired electrons, so 0 is a singlet, 1 a doublet and 2 a triplet. The entries run parallel to… |
| coupling | [] | Lists the state pairs whose couplings are computed. Each pair is two state numbers written 1-based, pairs are separated by a comma and centers by a… |
| current_state | 0 | Sets the electronic state the run is on, counted from 0. It is the ledger a surface hop rewrites during a trajectory. When it is left at its default… |
| highlevel | [] | Selects the atoms of the high level, which is the QM region of a QMMM run and the innermost layer of an ONIOM run, as 1-based atom numbers. Ranges… |
| midlevel | [] | Selects the atoms the middle level adds in an ONIOM3 run, as 1-based atom numbers in the same forms highlevel accepts. The middle layer covers… |
| highref | [] | Selects the high layer of the reference partition that adaptive sampling labels with, as 1-based atom numbers. It is parsed exactly like highlevel… |
| midref | [] | Selects the middle layer of the adaptive reference partition, as 1-based atom numbers. It is disjoint from highref and is read only when the… |
| embedding | false | Turns on charge embedding: the high level is computed in the field of the charges of the rest of the system instead of in a mechanical, isolated… |
| read_charge | false | Turns on reading {title}.charge as a background field. The first line of the file declares what its rows are: point angstrom external for arbitrary… |
| q_qm | [] | Sets the zero-order charge of each QM atom, one value per QM atom in the global order of the QM set. The classical stage of a QMMM run seeds its… |
| qm_contains_field_gradient | true | Declares who owns the zero-order field gradient term, the charge times the field. The default true is the ESPF contract, in which the QM analytic… |
| sc_charge | false | Turns on charge self-consistency for a multi-center QMMM run: the charges of the other centers are converged in the field before the wavefunction is… |
| sc_charge_maxiter | 10 | Sets the maximum number of charge rounds of the self-consistent multi-center QMMM iteration and must be at least 1. The iteration stops at the first… |
| sc_charge_threshold | 0.01 | Sets the convergence threshold of the charge iteration in units of the elementary charge, as a positive number. The default 0.01 accepts a round in… |
| sc_charge_damping | 1 | Sets the weight of each round fresh charge fit in the charges the next round is solved in, in the interval from above zero to 1 inclusive. A value of… |
| qmmm_key | None | Carries the retired QMMM declaration of the archive. A non-empty value is refused unless multiscale_type is qmmm, and under QMMM the key no longer… |
| qmmm_xyz † | Input | Records the legacy name of the QMMM coordinate input, Input by default. The value is carried with the system block; the region and the embedding are… |
| point_charges | [] | Supplies point charges as rows of q x y z, one row per atom in framework atom order, with the charge in units of the elementary charge and the… |
| external_charges | '' | Names a legacy .charge file of arbitrary external charges, whose layout is a charge count, a comment line and then q x y z rows. The rows are a… |
| external_potential | [] | Holds the external electrostatic potential and field as rows of V Fx Fy Fz, one row per QM atom, in atomic units. It is the channel an ESPF potential… |
| constraints | None | Holds the structured constraint block: named groups, collective variables, harmonic biases, walls, frozen Cartesian components, strict geometric… |
| center_groups | [] | Partitions the system into fragment groups, one group of atom numbers per fragment. The atom numbers here are 0-based, unlike every other atom… |
| track_type | None | Selects the geometry tracking of an MD trajectory: dist compares each atom pair named in track_index and frag compares the centres of exactly two… |
| track_index | [] | Lists the atom pairs of a dist declaration or the two atom groups of a frag declaration whose distance is tracked. Atom numbers are 1-based in the… |
| track_thrhd | [] | Sets the distance bound of each tracked object in Angstrom. The comparison reports when the value is greater than the bound, so equality does not… |
| track_stop | 0 | Chooses what a crossed bound does: 0 reports in the run log and 1 ends the trajectory after the offending frame is written. The legacy value 2, which… |
| solvent † | gas | Records the legacy environment label of the run, gas by default. It is a label of the archive input and names no solvation model in the framework; a… |
| pbc | [0, 0, 0] | Sets the periodicity of the three axes as three flags, each 0 or 1. Only full three-dimensional periodicity, 1 1 1, or none, 0 0 0, is accepted, and… |
| cell | [99, 99, 99, 90, 90, 90] | Sets the simulation cell as six parameters, a b c alpha beta gamma in Angstrom and degrees, or as the nine components of the lattice matrix. The… |
| lattice_vector | [[0, 0, 0], [0, 0, 0], [0, 0, 0]] | Sets the lattice as three row vectors in Angstrom, which is the form the runtime reads. The legacy spelling lattice takes the six cell parameters and… |
| kpoint | [1, 1, 1] | Sets the k-point sampling grid of a periodic calculation as three integers, one per reciprocal axis. The default 1 1 1 is the Gamma-point… |
| multitask | None | Sets the molecule-scope dispatch of backend terms: true fans the terms of one evaluation out to a multiprocessing pool and false keeps them in… |
| remotetask | None | Sets the molecule-scope dispatch of backend terms to Slurm: true submits each term as its own batch job. An empty value inherits the declarations of… |
| multipool | None | Sets the pool size of the molecule-scope term fan-out, an integer of at least 1. An empty value inherits, which is distinct from the serial default… |

Full descriptions for all available keywords are summarized below.

**atoms** - Lists the element symbols, one per atom, in the framework atom order that every other per-atom array follows. The value is normally resolved by the input parser from a coordinate file rather than written in the section, and an isotope label is stripped when masses are derived from it. An input with no atom list and no resolvable coordinate file is refused before any backend is started.

**coordinates** - Gives the Cartesian position of every atom in Angstrom, one row per atom in the order of atoms. It is filled from the coord key or from the same-basename .xyz file when the key is absent. A row count that disagrees with atoms is refused.

**velocities** - Gives the Cartesian velocity of every atom in Bohr per atomic unit of time, one row per atom. It is read from the velo key or from the &lt;coord&gt;.velo or &lt;coord&gt;.velocity.xyz file beside the geometry, and it is filled with zeros with an informational message when no velocity file exists.

**mass** - Gives the atomic mass of every atom in amu. When the list is empty or its length disagrees with the atom count, the masses are derived from the periodic table; an explicit list of the right length always wins. The integrators, thermostats and barostats read this array, and a run without masses never moves.

**natom** - Records the number of atoms. It is derived from the atom list whenever the key is absent, and a value that disagrees with the length of atoms, coordinates or mass is refused.

**charges** - Sets the total charge of the system in units of the elementary charge. It is not an atom index and is never decremented. A scalar is the single-center form and applies to every center; the per-center list is the authoritative spelling of a multi-center run, and the full-system charge that the low layer sees is the sum of the center charges. The derived electron count is built from this value together with multiplicity, and a run whose unpaired electrons exceed the electron count is refused at validation.

**multiplicity** - Sets the spin multiplicity 2S+1, at least 1 for every center. A scalar is the single-center form and applies to every center, while a list gives one multiplicity per center. The per-center value is the electronic structure of that center region, and the full-system multiplicity is the paired aggregate the lowest layer sees. The value is read together with ci and spin: the state ladders are checked against it, and a disagreement in length is refused.

**ci** - Sets the number of electronic states computed in each spin multiplicity, one entry per multiplicity. Thus ci 1 with spin 0 asks for one singlet, and ci 1 1 with spin 0 1 asks for one singlet plus one triplet. The flat form is the single-center spelling and the nested form carries one such list per center. The state count, the state labels, the pair maps of the couplings and the range that the root and the optimize state indices are checked against are all derived from ci and coupling.

**spin** - Sets the spin of each multiplicity as its number of unpaired electrons: 0 is a singlet, 1 a doublet and 2 a triplet. The entries sit beside the entries of ci and must agree with them in length, in the flat and in the per-center nested form. Together with multiplicity the list decides how many electrons are unpaired in each ladder, and a ladder that asks for more unpaired electrons than the system has is refused.

**coupling** - Lists the state pairs whose couplings are computed. State numbers are written 1-based, a comma separates the two states of a pair, a semicolon separates the pairs of one center from the next, and the word none gives a center no pairs at all. Each pair must name two distinct states inside the ladder, and the same unordered pair may not appear twice. The NAC and SOC pair maps, and therefore what each step asks a backend to compute, are derived from this list.

**current_state** - Sets the electronic state the run is on, counted from 0. It is the ledger a surface hop rewrites during a trajectory. When it is left at its default it is filled from the dynamics root, and an explicitly declared non-default value is never overridden.

**highlevel** - Selects the atoms of the high level, as 1-based atom numbers. Ranges such as 1-3 are accepted, a semicolon separates the region of one center from the next, and a nested list per center is the multi-center spelling. The region must be non-empty, free of duplicates and inside the system, and the high regions of two centers may not overlap. For QMMM the high level is the QM region and must leave a non-empty classical region; for ONIOM it is the innermost layer, and the middle layer of a three-layer run is described by midlevel.

**midlevel** - Selects the atoms that the middle level adds in an ONIOM3 run, as 1-based atom numbers in the flat or the per-center nested form. The middle layer covers midlevel and highlevel together, so midlevel is disjoint from highlevel while the middle region is a superset of the high one. A three-layer run refuses an empty, duplicate or out-of-range middle region, and a two-layer run has no middle level to read it with.

**highref** - Selects the high layer of the reference partition that adaptive sampling labels with, as 1-based atom numbers. It is parsed exactly like highlevel but never enters the theory center plan. When it is empty and the theory queue is layered, highlevel supplies it.

**midref** - Selects the middle layer of the adaptive reference partition, as 1-based atom numbers. It is disjoint from highref and is read only when the reference queue has three layers; the reference partition is single-center only.

**embedding** - Turns on charge embedding, so that the high level is computed in the field of the charges of its surroundings rather than in an isolated mechanical calculation. Classic text reads true or false, and because a shared namespace can also hand the keyword a skill word, charge, field and legacy fold to true while none folds to false; potential is refused with a pointer to field. An ONIOM run with the switch on needs a highlevel region, and the three-layer form a midlevel as well, so an empty pair is refused. ONIOM2 with a classical outer method refuses the switch outright and points to multiscale_type qmmm, which is the additive combination it describes.

**read_charge** - Turns on reading {title}.charge as a background field, with no computed field at all in a standalone run. The first line of the file declares its rows, and one file carries one kind: point angstrom external for arbitrary external points, point angstrom atom for one row per mid-shell atom of an ONIOM run, or field au for one potential and field row per QM atom. A standalone run refuses the atom-indexed kind because it has no mid shell, an ONIOM run refuses a field document, and a QMMM run clears the keyword because the composite supplies the field every step. The switch is mutually exclusive with external_charges.

**q_qm** - Sets the zero-order charge of each QM atom, one value per QM atom in the global order of the QM set. The classical stage of a QMMM run seeds its field from these charges, and a QM backend that returns computed charges overrides them. The length must equal the QM set, and only a single-center QMMM run may pin them.

**qm_contains_field_gradient** - Declares who owns the zero-order field gradient term, the charge times the field. The default true is the ESPF contract, in which the QM analytic gradient already contains the term and the composite removes it from the classical correction. False declares that the QM provider does not contain it, so the term stays on the classical side; the switch never changes an energy.

**sc_charge** - Turns on charge self-consistency for a multi-center QMMM run: the charges of the other centers are converged in the field before the wavefunction is solved. It needs at least two centers and embedding true, because a single center has no other charge in its field to iterate over.

**sc_charge_maxiter** - Sets the maximum number of charge rounds of the self-consistent multi-center QMMM iteration and must be at least 1. The iteration stops at the first round that meets sc_charge_threshold or at this bound, and the run log records which of the two stopped it.

**sc_charge_threshold** - Sets the convergence threshold of the charge iteration in units of the elementary charge, as a positive number. The default 0.01 accepts a round in which no center charge moved by more than that amount since the previous round.

**sc_charge_damping** - Sets the weight of each round fresh charge fit in the charges the next round is solved in, in the interval from above zero to 1 inclusive. A value of 1.0 accepts the fit as it is and a smaller value damps an iteration that oscillates instead of settling. The convergence test is taken on the raw fit either way.

**qmmm_key** - Carries the retired QMMM declaration of the archive. A non-empty value is refused unless multiscale_type is qmmm, and under QMMM the key no longer selects anything: the scheme comes from multiscale_type and the region from highlevel.

**qmmm_xyz** - Records the legacy name of the QMMM coordinate input, Input by default. The value is carried with the system block; the region and the embedding are declared by highlevel and embedding. This keyword is accepted but has no effect in v3.0.0.

**point_charges** - Supplies point charges as rows of q x y z, one row per atom in framework atom order, with the charge in units of the elementary charge and the position in Angstrom. The array is whole-system, so row i is atom i, and a composite task reads its own region slice rather than the whole array. A row count that disagrees with the atom count is refused.

**external_charges** - Names a legacy .charge file of arbitrary external charges, whose layout is a charge count, a comment line and then q x y z rows. The rows are a background field for the first theory task and the first reference task; they are never sliced and carry no periodic images. It is refused together with highlevel or midlevel and with any periodic run, and it may not be combined with read_charge.

**external_potential** - Holds the external electrostatic potential and field as rows of V Fx Fy Fz, one row per QM atom, in atomic units. It is the channel an ESPF potential embedding reads in place of the point-charge channel. A standalone read_charge file and the QMMM field handoff fill it, and a task that carries both this array and external charges is refused.

**constraints** - Holds the structured constraint block: named groups, collective variables, harmonic biases, walls, frozen Cartesian components, strict geometric constraints and metadynamics. In classic text the block is its own &constraint section, one keyword per line, and in YAML and agent JSON it is the constraints mapping of the system section. An empty value means no constraint plan and no change of behaviour; the keyword inventory is in [Section 9](09-constraints.html).

**center_groups** - Partitions the system into fragment groups, one list of atom numbers per fragment, and is the cue the composite planner reads for a fragment task queue. The atom numbers are 0-based, deliberately unlike every other atom selection in the input, because the planner has always read them that way; center_groups 0 1 followed by 2 makes group 0 out of atoms 1 and 2 and group 1 out of atom 3. The name keeps the center_ prefix because the field once shared the name groups with the constraint block and the two were read as each other. A multi-center run refuses center_groups beside the legacy qmmm_key.

**track_type** - Selects the geometry tracking of an MD trajectory: dist compares each atom pair named in track_index and frag compares the centres of exactly two groups. It needs track_index and at least one track_thrhd, and leaving it empty monitors nothing.

**track_index** - Lists the atom pairs of a dist declaration or the two atom groups of a frag declaration whose distance is tracked. Atom numbers are 1-based in the input and decremented once; a frag entry that does not name exactly two groups is refused.

**track_thrhd** - Sets the distance bound of each tracked object in Angstrom. The comparison reports when the value is greater than the bound, so equality does not report. A list shorter than the number of tracked objects is broadcast with its last value, and an empty list is refused.

**track_stop** - Chooses what a crossed bound does: 0 reports in the run log and 1 ends the trajectory after the offending frame is written. The legacy value 2, which suppressed surface hopping, is not implemented and is refused.

**solvent** - Records the legacy environment label of the run, gas by default. It is a label of the archive input and names no solvation model in the framework; a continuum solvent is requested by the solvent keyword of the skill section. This keyword is accepted but has no effect in v3.0.0.

**pbc** - Sets the periodicity of the three axes as three flags, each 0 or 1, and requires exactly three of them. Only full three-dimensional periodicity, 1 1 1, or none, 0 0 0, is supported in this version; a mixed setting is refused because the low-dimensional solvers are not defined. pbc is the single periodicity authority after normalization, so a non-zero lattice alone no longer implies a periodic run, and a periodic run needs a real cell and lattice. Every provider of a periodic task must declare that it supports periodic boundaries.

**cell** - Sets the simulation cell either as six parameters, a b c alpha beta gamma in Angstrom and degrees, or as the nine components of the lattice matrix. The default 99 99 99 90 90 90 is a sentinel meaning not filled in: no lattice is derived from it, so a non-periodic run never carries a fake 99 Angstrom cube. The six-parameter form is triangularized into the row-vector lattice that the runtime reads, a nine-component form and a lattice_vector must agree or the input is refused, and the cell volume is derived from the matrix unless it is given.

**lattice_vector** - Sets the lattice as three row vectors in Angstrom, which is the form the runtime reads. The legacy spelling lattice takes the six cell parameters and is converted to this matrix. A matrix that disagrees with cell is refused, and the cell volume is derived from it.

**kpoint** - Sets the k-point sampling grid of a periodic calculation as three integers, one per reciprocal axis. The default 1 1 1 is the Gamma-point calculation, and the value is carried with the system block for the periodic backend.

**multitask** - Sets the molecule-scope dispatch of backend terms: true fans the terms of one evaluation out to a multiprocessing pool and false keeps them in process. The value is tri-state, so an empty value inherits the declarations of the skills in the queue and an explicit value overrides them. It is refused together with a composite executor of thread when it is false.

**remotetask** - Sets the molecule-scope dispatch of backend terms to Slurm: true submits each term as its own batch job. An empty value inherits the declarations of the skills in the queue. Per-term remote execution is refused for QMMM, which runs its terms in process.

**multipool** - Sets the pool size of the molecule-scope term fan-out, an integer of at least 1. An empty value inherits, which is distinct from the serial default of 1 that the workflow sections use. A value that disagrees with an explicit composite max_workers is refused.

The per-atom arrays are whole-system arrays: row i is atom i in the order of atoms, in the input file, in every trajectory frame and in every output table. A composite task reads the slice of its own region, so an array that is not sized to the whole system is refused rather than guessed at.

The derived quantities of this section, among them the electron count, the state labels, the lattice volume and the cell-crossing ledger, are not input keywords; [Appendix B](appendix-b-index-bases-and-derived-quantities.html) lists them with the runtime fields of the state.
