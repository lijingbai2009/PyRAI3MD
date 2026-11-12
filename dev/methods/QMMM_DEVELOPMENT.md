# PyRAI3MD QM/MM (QMMM) Development Guide

Last updated: 2026-11-07 08:39 (only the field-handoff paragraph of section 4 was touched: it now states that `v_mm_native`/`e_mm_native` are converted to atomic units with the skill manifest factors at the handoff, the same arithmetic the `.esfp` channel applies, and names the #465 regression that read them raw. Everything else was not re-audited. Earlier: 2026-11-06 13:30 (only the adaptive paragraph of section 4 was touched: it now names its shipped carrier, `examples/adaptive_qmmm_local.yaml`, and the two measured readings that carrier leaves - an `&method alignment` record with `aux: [mm_energy, q_qm]` and no `field`, and sidecar rows whose `mm_energy` sits at about -8.270 Hartree for all five frames. Sections 2, 5, 6, 8, 9 and 10 were not re-audited. Earlier: 2026-11-04 15:20 (#452 tiers the five shipped cases by charge question and re-measures every reading in section 9; #451 multi-center QMMM. Sections 2, 4, 5, 6, 8, 9 and 10 were rewritten against the code and against the runs this round produced, and the readings in §9 are the local host's own (Molcas v26.10 + LAMMPS 22 Jul 2025 Update 6): §2 carries the three `sc_charge` keys and the nested-`highlevel` acceptance with the `q_qm` refusal under several centers; §4 the per-center duties and the charge round that exists only when `sc_charge: true`; §5 the C-center identity and the id-addressing rule; §6 the coupling refusal, the opened curvature searches with their `ts` readings and the narrowest-ladder bound; §8 what one evaluation does (cold start, Jacobi snapshot, no fallback inside a round, `CenterSpec.charge` as the balancing target) and what is deferred inside multi-center; §9 the three shipped cases, the LAMMPS-side reciprocity numbers, the resident-session readings and the periodic-resident segmentation fault with its three controls. Sections 1, 3 and 7 kept their earlier text. Earlier: 2026-11-01 11:00 (#446. §2 now says the charge readback is one row per state and that the framework selects the row; the coupling half of the same change is in `dev/skills/MOLCAS_SKILL_DEVELOPMENT.md` sections 5.4 and 6.2. Before it, 2026-10-31 23:20 (the §2 readback paragraph now carries the test environment's own numbers from round `tmp_20261031_220625`: the printed `-0.5994 / 0.2997 / 0.2997` traced to the `rasscf` section's first block (`Charge on O1 = -0.5990`, `Total ESPF charge = 0.0012`) minus `-0.0004` per atom, with the `caspt2` block at `-0.5730` and the second root at `+0.1215` left unread, and the unpinned `multiscale_qmmm_ts` printing the averaged thirds. Before it, 2026-10-31 18:20 (later the same day: §2's `q_qm` paragraph was
corrected against `skills/molcas_skill/backend.py` — the ESPF readback it
describes had never been reachable on the QMMM path, and the two field-probe
paragraphs now point at the accuracy that makes the A/B difference converge
rather than at an equalisable grid. Written this round from
`dev/methods/QMMM_drafts.md` and re-read against the code: the planner
(`planning/task.py::_expand_qmmm`), the calculator's three stages
(`workflow/calculator.py::_run_qmmm`, `_qmmm_atomic_block`), the composite
stitch (`workflow/multiscale/composite.py::assemble_qmmm`), the LAMMPS skill's
two QM/MM profiles (`skills/lammps_skill/service/parser/qmmm_output.py`) and
the validation surface (`schema/validate.py`). The physical derivations that
the draft argued for are kept where they are implemented and are marked
"not implemented" where they are not; nothing here is a plan for future work
unless it says so.

---

## 1. Scope and boundary

QM/MM in this package is the **additive**, non-polarizable scheme: one
electronic region (`molecule.highlevel`) computed in the electrostatic field
of the rest of the system, and one classical evaluation of the whole system
with the QM region represented by its zero-order (point) charges. It is
selected by `control.multiscale_type: qmmm` with exactly two theory methods,
QM first and the classical provider last (today always LAMMPS).

What this scheme is not:

- it is not a polarization model. The QM calculation is embedded in a fixed
  field built from the classical charges; the classical side never responds
  to the QM density beyond the fixed `q_qm` charges it is given. The
  polarization-only / NN extension is a separate design
  ([QMMM_NN_drafts.md](QMMM_NN_drafts.md), not implemented);
- it is not charge embedding. ONIOM passes point charges to its high-level
  skill; QMMM passes the MM potential/field sampled at the QM nuclei, and a
  QMMM run never reads a charge file (section 2);
- it is not multi-center (section 8).

## 2. Input and planning

The QM/MM plan is `qmmm_field -> QM -> qmmm_energy` (with `embedding: true`)
or `QM -> qmmm_energy` (with `embedding: false`). Under several centers each
of the first two duties is expanded once per center, and `sc_charge: true`
puts one energy-only charge round per center between them (section 8). The
three keys that look alike answer one question each:

| key | question it answers | QMMM behaviour |
|---|---|---|
| `multiscale_type: qmmm` | is there a composite, and which one | selects this scheme; exactly two theory entries, `highlevel` required |
| `embedding` | does the QM calculation see the environment electrostatically | `true` adds the field stage and the V/E handoff; `false` runs an isolated QM and leaves the QM–MM electrostatics in the classical term |
| `read_charge` | does the embedding data come from `{title}.charge` | **not read at all**: the MM stage produces the field every step, so both values are cleared without a warning |

The charge iteration is one further question with two companions:

| key | question it answers | QMMM behaviour |
|---|---|---|
| `sc_charge` | does each center's field carry the other centers' current charges | `false` (default) probes once with the centers uncharged and solves once; `true` repeats the probe/charge cycle until every center's 0-order charges stop moving |
| `sc_charge_maxiter` | how many rounds that may take | only read when `sc_charge: true`; the default 10 bounds one evaluation's charge solves at 2 x 10 per two-center run. `1` is a legitimate declaration, not a stall: the log then reads `stopped by single round`, and that is how `multicenter_qmmm_sp.yaml` asks for exactly one 0-order fit |
| `sc_charge_threshold` | how small a charge move counts as stopped | max-norm over the centers, in elementary charge; default 1e-2 |

Validation (`schema/validate.py`, QMMM branch) also enforces: `highlevel` is
a duplicate-free index list, flat for one center or one list per center for
several; the union of the regions is a strict subset of the system (the MM
region must be non-empty); `q_qm`, when written, has one finite value per
high-level atom and is **refused outright under several centers**, because the
classical stage there is fed by what each center's own fit reported
(section 8); `sc_charge: true` needs `embedding: true` and at least two
centers; `center_groups` may not accompany QMMM; and the
`&system_constraint` block **is accepted**, exactly as for ONIOM, because
constraints name global atoms and their corrections are applied by the
framework after assembly (section 6).

`q_qm` is the QM 0-order charge set in `highlevel` order. It seeds the
classical evaluation; a QM backend that returns `computed_charges` (the
Molcas ESPF fit) overrides it at run time, and the charge store is the single
home of the value the classical stage actually receives
(`q_qm = point_charges[highlevel]`).

That override was unreachable until this round, and the sentence above
described an intention rather than a path - for two independent reasons, both
measured on round `tmp_20261031_183037`. First, the backend read the post-ESPF
charges only for the embedding words `charge` and `potential`, while a QMMM
task arrives with `field` (`potential` being refused by the skill's own
schema), so the reader was never even called. Second, the reader it would have
called opened `ESPF.DATA`, a sidecar no deployed calculation directory ever
contains (it is opened with `Molcas_Open` into the per-job run directory,
which the generated script deletes on exit): the charges are printed in the
log's `++ ESPF analysis` block, e.g.

```text
        Charge on O1          =    -0.7164
        Charge on H2          =     0.3605
        Charge on H3          =     0.3587
      Total ESPF charge     =     0.0028
```

A third thing had to be fixed before the readback could be used at all, and
it is the one that made round `tmp_20261031_195117` fail on all eight QMMM
examples: **a run prints many such blocks**, one per root inside `&RASSCF`
plus more from `&CASPT2` (twelve on a two-root XMS job, one on a single-root
job, three on a three-pass RASSCF), and no block carries a state label.
Reading whichever block comes last hands the classical stage another root's
response charges - the two-root case measured 0.0276 there against 0.0012
for the active state - and `qmmm_energy` rightly refuses a QM partition that
is not neutral. The reader therefore takes the k-th block **of the `rasscf`
section** as state k, asked for by `target_states[0] + 1`, and ignores
`caspt2` blocks; a state with no block of its own yields no charges rather
than a neighbour's.

Even the right state's charges leave a residual, because the fit is a
projection onto a shell grid and not a constrained sum: 1.2e-3 on a neutral
water at the default `PNT 4 1 Å`, and denser sampling does not fix it
(1.5e-3 at 56,541 points, 5e-4 at 8 shells). So the framework spreads what
is left evenly over the QM region against the declared `molecule.charges`
before the classical stage sees it - shape of the fit kept, sum admissible -
and logs both the amount removed and the unadjusted values. Only the readback
is balanced: a `q_qm` the input spells out is a declared number and stays as
written, and a per-center `charges` list is left alone because which entry
owns the region is not defined.

The readback is one row **per state** - `(nstate_read, natom)`, in the order the
engine prints its roots - and the framework picks the row (`_charge_embedding_state`,
shared with the ONIOM charge scatter since #450, read from the step's `target_states`
else the system's current state), because the
0-order fit is a property of the density being embedded: taking row 0 whatever
the step runs on leaves an excited surface embedded in the ground state's
charges. A state index the backend did not print is refused in the log rather
than answered with a neighbour's row, and only the chosen row is balanced
against `molecule.charges`.

All three fixes together are what let `multiscale_qmmm_ts` drop its pin.
Round `tmp_20261031_220625` shows the two halves working in the test environment, in
the printout rather than only in the resolver: `multiscale_qmmm_embedding_sp`,
whose input still pins `[-0.83, 0.415, 0.415]`, prints `-0.5994 / 0.2997 /
0.2997`. Its `MOLCAS.log` has the arithmetic in it: the `rasscf` section's first
block reads `Charge on O1 = -0.5990`, `Total ESPF charge = 0.0012`, and the
delivered vector is that block with `-0.0012 / 3` taken off each atom, so the
partition is neutral and the shape is the fit's. The same log also carries a
`caspt2` block at `-0.5730` and a second-root `rasscf` block at `+0.1215`
(`Total ESPF charge = 0.0255`), either of which a positional reader would have
delivered. `multiscale_qmmm_ts`, which declares no `q_qm` at all, prints
`-0.7143333 / 0.3571667 / 0.3571667` on its own geometry: the repeating thirds
are the averaged residue, and the vector is the engine's, not the file's.

Both had to be fixed before `q_qm` could be derived, and the reading also
shows why the pinned values were a substitute rather than a copy: the
`multiscale_qmmm_ts` case pinned `[-0.83, 0.415, 0.415]` where the fit gives
`[-0.7164, 0.3605, 0.3587]`, and the six-atom `mep` case pins two identical
waters while the fit resolves them differently
(`-0.7149/0.4485/0.2703` against `-0.6276/0.3304/0.2979`). `multiscale_qmmm_ts`
now declares no `q_qm` at all and takes the derived vector; the other three
keep theirs as the fallback the resolver reaches only when the reader finds
nothing. Note the log carries four decimals - that is the resolution of the
derived value, and a case needing more has to make Molcas write the sidecar.

## 3. Physical definitions

### 3.1 The two flows

The classical provider splits one full-force-field evaluation (`U_main`)
into the pieces the composite needs:

```
qmmm_el   = sum_i q_qm_i V_i          (QM-MM electrostatics, from the field)
U_gas     = in-cell bare Q-Q Coulomb   (computed in Python, same minimum image)
E_classic = U_main - U_gas - qmmm_el
```

`V` and `E` are produced by the `qmmm_field` task, an A/B Coulomb probe:
run A sets the QM atoms to charge +1 and keeps the MM charges, run B zeroes
them, and `V_i = 2(pe_A,i - pe_B,i)`, `E_i = F_A,i - F_B,i`. The factor 2 is
`compute pe/atom`'s half-pair allocation; the force needs no factor. LAMMPS
sizes the reciprocal sum from the system's squared charge at every init, so
zeroing the MM charges moves that extent: the two runs must **converge to the
same sum**, not merely be told to use the same grid - `kspace_style` carries
`ewald_accuracy`, `kspace_modify gewald 3.2/rc` pins the split, and ewald
ignores an explicit `mesh` outright. The probe then checks `sum V` against
the Q-M `group/group` energy before anything is handed over. See
`skills/LAMMPS_SKILL_DEVELOPMENT.md` §12.2 for the measurements and for what
the accuracy costs a periodic trajectory.

### 3.2 The field gradient term and its owner

`f_el = q_qm * E` is a force; as a gradient it is `-q*E`. The LAMMPS skill
reduces the classical surface before returning it: the QM rows of its
`gradient` are `d(U_main - U_gas - qmmm_el)/dR`, every other row is
`dU_main/dR`, and the removed pieces are reported beside it
(`u_gas_gradient_native`, `qmmm_el_gradient_native`; the reduced array is
also written as `{title}.grad`). The engine's own `.energy`/`.force` files
stay the unreduced `U_main`.

Who carries `-q*E` is the `System.qm_contains_field_gradient` switch:

- `true` (default): the QM provider's analytic gradient already contains the
  zero-order field term (the Molcas ESPF contract), so the classical side must
  not, and it stays removed;
- `false`: the QM provider reports only the fixed-potential derivative (a
  polarizable NN), the field term was removed from the classical side for a
  provider that does not have it, and the composite adds
  `qmmm_el_gradient` back onto the QM rows.

Either branch holds exactly one copy of the term, and the switch moves no
energy and no MM row.

## 4. Task queue and contracts

| task | contribution | profile | returns |
|---|---|---|---|
| `qmmm_field` | 0 (auxiliary) | `qmmm_field` | `qmmm_field` block: `qm_atoms`/`qm_atom_ids`, `v_mm_native`, `e_mm_native`, `u_gg_native`, `f_gg_native`, the probe charge and the group-check residuals; its `energy`/`gradient` are probe-only zeros |
| QM | +1 | provider's own | energy and gradient of the high-level region, plus `computed_charges` (ESPF fit) and, for multi-state work, `nac`/`soc` |
| `qmmm_energy` | +1 | `qmmm_energy` | `U_main` energy/force/stress, the `qmmm` block (`u_main_native`, `u_gas_native`, `qmmm_el_native`, `e_classic_native`, `q_qm`, the two removed gradients), the reduced `gradient`, and the audit files |

The field handoff has one unit rule: `v_mm_native` / `e_mm_native` are
native LAMMPS numbers (for the `real` unit set, kcal/mol and
kcal/mol/Angstrom), and the framework converts them to Hartree / Hartree per
Bohr with the skill manifest's `conversion` factors before they reach the QM
calculation. The `.esfp` file is the same conversion written once by the
skill, so the two channels must agree to 1e-12. Reading the block raw
embeds the QM in a 627x potential and a 1186x field; that regression
(`#465`) showed up as QM-row gradients near 600 Hartree/Bohr and was fixed
at the handoff, not in the skill.

Under several centers the same three duties appear once per center, plus a
fourth duty that exists only there:

| task | contribution | profile | returns |
|---|---|---|---|
| `qmmm_field` x C | 0 | `qmmm_field` with `qm_probe` narrowed to that center | the center's own rows, addressed by atom id; the other centers sit in the probe's environment at their current charges |
| `qm_charge` x C | 0 | provider's energy-only jobtype (`&ESPF`, no `&ALASKA`) | the center's 0-order charges and nothing else |
| `qm` x C | +1 | provider's own | each center's energy, gradient and charges |
| `qmmm_energy` x 1 | +1 | `qmmm_energy` with `qm_centers` | one whole-system surface, with the in-cell Q-Q removal split per center |

The charge rounds are a separate duty because a skill compiles its module
chain once per task: the same QM task cannot be an energy-only solve in one
round and a gradient solve in the next. They exist only when `sc_charge: true`;
with the switch off, the property solve is the one and only solve, and its
charges are the ones the classical stage receives. The queue order is fixed by
the plan (all probes, then all charge rounds, then all property rounds, then
the classical one), and every round reads one charge snapshot, so the order the
centers were declared in cannot change the answer.

In adaptive sampling this same three-stage queue is the reference plan when
`control.reference` names two methods and `multiscale_type=qmmm`. The QM
region is `molecule.highref` (inherited from `highlevel` when the theory
queue is layered), and the QM task's raw energy/gradient -- not the
stitched total -- is the neural-network label. `assemble_qmmm` also
publishes `label_channels` with the QM energy/gradient, the V/F field
rows, `E_classic` and `q_qm`; the MM rows stay out of the label. A
reference list with no MM provider is allowed but records a warning that
the labels are unpolarized and carry no field. The shipped carrier of that
shape is `examples/adaptive_qmmm_{local,slurm}.yaml`; its recorded note is the
one above, and its sidecar rows carry `mm_energy` and six `q_qm` with no
`field` row.

The plan appends the `field`/`qmmm` capability tokens to the two LAMMPS tasks
so the skill resolves its own profile; `_expand_qmmm` passes the QM skill's
declared electronic capabilities (minus the whole-system-only `stress`) to the
QM task, so a surface-hopping or crossing run requests what the provider can
give. Per-term remote execution is refused for QMMM: the field → QM handoff
is local to the evaluation host.

A region provider answers a derivative coupling in region coordinates
(`(npair, nlocal, 3)`) and the composite scatters those rows to the system
the way it already scatters the region's gradient
(`_project_region_coupling`); unprojected, a crossing search dies on
`shapes (18,) and (4743,) not aligned`.

## 5. Result stitching

`Calculator._run_qmmm` executes the stages, converts each task's native
numbers to Hartree / Hartree-per-Bohr through the skill's declared units, and
`composite.assemble_qmmm` produces:

```
E         = E_QM + E_classic
gradient  = [G_QM + classical Q-rows] on the QM rows
            [classical rows]           elsewhere
```

For C centers the same two lines hold with the QM side summed per center and
the classical side still single:

```
qmmm_el   = sum_c  q_c . V_c(total)     V_c = MM field + the other centers
U_gas     = sum_c  intra-center pairs
E_classic = U_main - U_gas - qmmm_el
E         = sum_c  E_QM(c) + E_classic
```

The inter-center 0-order pair is then counted once: it enters `qmmm_el` for
both centers (2 terms) and leaves `U_gas` for neither (the classical surface
removes only what is inside one center), and the residual guard reads the one
identity `U_main - U_gas - qmmm_el - E_classic = 0` over the whole system.
Every gradient row is addressed by atom id, never by concatenation order: the
union array is what the classical side reduces once, and a center's rows are
picked out of it by id
(`_qmmm_external_potential`, `workflow/multiscale/composite.py::assemble_qmmm`).

with the reduction (and, for `qm_contains_field_gradient: false`, the field
term) as described in section 3.2. Non-aggregated keys (`nac`, `soc`,
`dipole`, ...) pass through from the QM task. The report carries
`qmmm_components`: `energy_semantics`, `embedding`, the energy decomposition,
`q_qm`, the QM atom list, the field rows actually handed to the QM stage, and
an `approximations` list that states, among other things, that the MM rows
keep the full `U_main` force (including the Q–M Coulomb reaction force).

**Stress.** The reported tensor is the unreduced classical `U_main` virial
from the `qmmm_energy` task, converted once by the framework. The terms
removed from the QM rows are restored to atoms by the QM side but not to the
cell, so the cell response of `qmmm_el` and `U_gas` stays at the point-charge
level. This is a model difference, not a numerical error, and it is stated in
the result's own `approximations`.

## 6. Capabilities and jobtypes

- `energy`, `gradient`, `optimize` (including `optimize_cell`), `md`
  (`nve`/`nvt`/`npt`): supported. The framework drives the loop and asks the
  three stages for one evaluation per geometry; no new execution path is
  needed.
- `surface_hopping`: supported. The QM task is the electronic provider and
  the field is inside its Hamiltonian; `nac`/`soc` pass through the
  composite. The classical term is state-independent, so `q_qm` is taken from
  the first state's charges — the one approximation of this path.
- `&system_constraint`: supported as of this round, identically to ONIOM.
  Constraints act on the whole system and only on the atoms they name; their
  energy and gradient corrections are added by
  `workflow/constraints/runtime.py::finalize` after the composite assembled
  its energy, they appear in `energy_contributions` and never in
  `composite_terms`, and a `fix` reaches the same mask the optimizer and
  integrator use. A `fix` on the MM atoms is therefore the way a QMMM
  saddle/IRC run holds its environment: `hess_atoms` derives from that mask
  and the numerical Hessian displaces only the QM atoms.
- `hessian`/`frequency`: only the numerical form is available for a
  multi-scale run (`hess_driver: main` or `num_hess`/`num_freq`). An analytic
  matrix (`hess_driver: skill`) is refused: one geometry has one composite
  curvature and a single backend's matrix is only one of its terms.

Under several centers two of those lines change and one is added:

- `nac`/`soc` and the crossing searches are **per center**, not refused. Each
  center answers its own coupling between its own states, the carrier keeps the
  blocks in `state.electronic[c]`, and `assemble_qmmm` scatters each center's
  rows into its own region and concatenates them along the pair axis - nothing
  is summed or averaged across centers, and center 0 lends its coupling to
  nobody. A crossing search names one pair **per center** (`opt_state_i` /
  `opt_state_j` as lists), and the composite then outputs two synthetic surfaces
  (each center at its own first/second member, the classical term identical on
  both), which is what lets `meci`/`mecp` follow one gap that is the sum of the
  centers' gaps. The refusal that survives is about the declaration, not the
  count: asking for `nac`/`soc` without naming the state pairs in
  `molecule.coupling` is stopped at planning, because an undeclared pair is
  shielded and there would be nothing to read back.
- Curvature-following searches (`ts`, and the crossing searches that need an
  initial Hessian) are searchable: the composite is one surface over the
  whole system's 3N coordinates, `hess_atoms` derives from the global
  constraint mask, and nothing is split per center. Measured with two water
  centers on the nine-atom cluster: two iterations of `method: ts` report
  `Negative Modes (model)` 4 and 5 with mode eigenvalues -0.00286651 and
  -0.00282293 Hartree/Bohr^2, and the run stops on the declared
  `maxit: 2` (5 min 30 s of wall on this environment, one 3N=27-displacement matrix
  per iteration).
- `opt_state_i`/`opt_state_j` are bounded by the **narrowest** center ladder,
  because the search names one index and every center is solved at it; a
  multi-center run also does not write `current_state`, since it has no such
  global scalar (the per-center active roots belong to the plan). Both fields
  take a scalar or one index per center (`state_i: [0, 0]`), and a per-center
  list does not travel into the skill configs: a skill solves one region and
  receives its states through `target_states`, so the framework drops the
  sequence there (`schema/adapter.py::_merge_one_skill_config`) while scalars
  keep being forwarded (#460).
- `surface_hopping` over C>1 runs one ledger per center: `root: "2; 2"` names
  each center's starting state, `nactype: nac` makes each center answer its own
  coupling, and a hop is confined to the center it belongs to
  (`TestMultiCenterHopIsolation` in `test_electronic.py`). `multicenter_qmmm_md_fssh.yaml` is
  the engine carrier.

## 7. Periodic and charge limits

- Neither partition has to be neutral. The QM charge is declared per center in
  the input (`molecule.charges`, beside `ci`/`coupling`) and the MM charges
  come from the user's topology; the framework balances each center's fitted
  0-order charges to that declared value, so each center's in-cell removal is
  taken with the charges it actually carries. The old rule refused a charged
  partition on the theory that the probe's two Ewald backgrounds would not
  cancel; measured on the 60-atom box with the MM cell charged by `-0.4 e`,
  the A/B identity holds to `2.0e-5` against a `-8.156` kcal/mol tally, the
  same order as the neutral cell - a uniform background is the same level in
  both legs of one evaluation, and any offset that did survive would cancel
  between `E_QM` (linear in V with the QM net charge) and
  `qmmm_el = sum q V`. `validate_partition_charges` now checks the vector's
  shape only.
- The Q–Q interaction inside the cell is removed analytically
  (`in_cell_qq`, minimum image) and the Q–Q image Coulomb stays in
  `E_classic`; v1 clones the QM LAMMPS types so the Q–Q non-Coulomb terms are
  zero without touching Q–M/M–M terms.
- `qmmm_energy` needs the cell of a periodic run, because the in-cell Q–Q
  removal is defined against it.
- The field probe is an approximation to the periodic QM operator; its
  residual is reported, not hidden (`approximations`).

## 8. Multi-center

Several QM regions are supported: `molecule.highlevel` written as one list per
center builds C centers, and `resolve_composite_type` distinguishes the two
spellings (`'qmmm'` for a flat region, `'multi_center_qmmm'` for nested) so the
single-center task identifiers and calculation directories do not move.

What one evaluation does:

1. One field stage per center, with every QM center's charge set to zero for
   the first round. The probe is narrowed to its own center by `qm_probe`, so
   the other centers fall on the environment side of the A/B difference and
   share the MM's Ewald and images - the inter-center field is not a second
   electrostatic route.
2. If `sc_charge: true`, one energy-only charge round per center in that
   field (its `&ESPF` block, no `&ALASKA`), then the charges are collected and
   the probes are rebuilt from them. The rounds repeat until the largest
   per-center max-norm move is under `sc_charge_threshold` or
   `sc_charge_maxiter` is reached. All centers read one snapshot per round
   (Jacobi), so declaring order cannot change the answer.
3. Then, and only then, the property round with gradients and the single
   whole-system classical evaluation.
4. `E_classic = U_main - U_gas - qmmm_el` and `E = sum_c E_QM(c) + E_classic`
   with `U_gas` the sum of the intra-center pair removals (section 5).

Three rules the reading depends on. **Cold start**: every step's iteration
begins from zero charges and only the accepted round is written to the charge
store, so a frame's energy and forces answer to that frame alone. **No
fallback inside a round**: a charge round whose provider returns no
`computed_charges` stops the run instead of falling back to the store or to
`molecule.q_qm`, because the number it would invent is the number the next
round's field is built from. **id addressing**: the `.esfp` handoff carries an
atom-id column and the framework binds rows by id, so a center can never be
embedded in another center's field by an accident of ordering.

A composite evaluation reports **one assembled surface**, chosen by the
per-center `active_configuration` and recorded in the same result. That has one
consequence every search has to respect: the result's energy is a single number
that already IS the requested state, so a search must not index it by the
scalar state id. Multi-center excited-state minimization (`opt_state_i: 1` with
the same index in every center) hit exactly that and raised
`index 1 is out of bounds for axis 0 with size 1` before #459; the optimizer now
reads the assembled value when the result carries the marker and keeps the loud
failure for a plain run whose provider returned fewer states than it asked for.

One optional knob, `sc_charge_damping` (default 1.0, range (0, 1]): the weight
of each round's fresh fit in the charges the next round is solved in. Two
centers polarizing each other make the undamped map non-contractive in general,
and a weight below 1 is the standard cure; the convergence test stays on the raw
fit against the previous raw fit, so damping changes the path and never what
counts as converged (`workflow/multiscale/charges.py::damp_charges`, and the
criterion that a damped weight turns a diverging map into a convergent one is
`TestChargeDamping` in `test_multicenter_qmmm.py`). On the shipped
`multicenter_qmmm_sc_charge.yaml` it changes nothing measurable - the map there
converges in two rounds either way (`9.140667e-1` then `7.000000e-4` against
1e-3) - so no case in the tree motivates it yet; it is carried as the escape
hatch for a strongly coupled box, not as a new default.

The charge a center is fed is balanced to that center's declared charge
(`CenterSpec.charge`), not to the whole-system `molecule.charges`, so a center
is balanced to its own declared charge round by round.

Charged centers are admissible (section 7): a center's QM charge is its own
`molecule.charges` entry, the MM charges come from the topology, and the
per-center gas removal is taken with that center's fitted 0-order charges, so
`U_gas` is the removal of the charge set that center actually carries.

Deferred within multi-center: overlapping centers, link atoms for a bond that
crosses a partition, and mixed QM providers across centers (one level per run
today). Per-center couplings and per-center crossing pairs are not deferred -
they are the reading section 6 describes, with
`multicenter_qmmm_nac_sp.yaml` as the shipped carrier. The N-center machinery
this reuses is documented in
[MULTICENTER_DEVELOPMENT.md](../MULTICENTER_DEVELOPMENT.md) section 15.6.

## 9. Acceptance

Shipped cases (`pyrai3md/examples/`, family `multiscale_test`):

- `multiscale_qmmm_{embedding,no_embedding}_{sp,md}`: the field-embedded and
  the isolated forms, energy + SOC and a 5-step NAMD, on the 60-atom water box.
- `multiscale_qmmm_{meci,mecp}`: crossing searches on the 1581-atom
  ethylene-in-water box, Molcas CAS-SCF + LAMMPS, `maxit: 2`; the meci case is
  the one that also proves the derivative coupling reaches the branching
  plane (`branch_dirs == 2`, `coupling_norm > 0` in `opt_status.txt`).
- `multiscale_qmmm_mecp_cluster`: a crossing on the six-atom cluster, two
  singlets of one CAS(2,2) space. It was authored as a singlet/triplet pair and
  the deployment refused it: see the mixed-spin entry in Section 10.
- `multiscale_qmmm_ts`: a saddle search on the six-atom cluster with the MM
  atoms held by a `fix` and the curvature from the numerical composite
  Hessian.
- `multiscale_qmmm_mep`: a three-image band on the 60-atom box.

Multi-center cases (`family multiscale_test`, ported from
`multicenter_pbc_nve.yaml`: the same 20-water TIP3P box and the same two
face-straddling waters, with Molcas for each center). They divide the charge
question three ways, and the division is what the queue shape shows:

- `multicenter_qmmm_sp.yaml` - **one fitted round**. `sc_charge: true` with
  `sc_charge_maxiter: 1`: the cheap energy-only solve runs once per center, the
  accepted charges are probed a second time, and the gradient solve is embedded
  in that polarized field. Seven duties (`field, field, qm_charge, qm_charge,
  qm, qm, energy`), and the log reads `stopped by single round`.
  Read on the local host (Molcas v26.10 + LAMMPS 22 Jul 2025 Update 6, cold
  stage, 26 s): `E_QM -152.3057801400` = center 1 `-76.1563741400` + center 2
  `-76.1494060000`, `E_classic -0.2230559853`, total `-152.5288361253`.
  What one round moves is mostly inside the two terms cancelling each other:
  `E_QM` shifts `-152.2976518400 → -152.3057801400` (8.13e-3) while `qmmm_el`
  takes the same money back out of the classical surface, and the total moves
  1.23e-4 Hartree.
- `multicenter_qmmm_sc_charge.yaml` - **the converged fixed point**. Same input
  with `maxiter 5` at `1e-3`; two rounds (`max|dq|` 9.140667e-1 then
  7.000000e-4), `stopped by tolerance`, total `-152.5288369992`, 29 s. The
  second round is worth 8.7e-7 Hartree on this geometry, so on this box one
  fitted round already carries 99.3% of the inter-center polarization.
- `multicenter_qmmm_md.yaml` and `multicenter_qmmm_mep.yaml` - **one fitted
  round**, like `sp`. Anything that integrates or walks the surface has to
  name one electronic structure per evaluation: the zero tier below leaves the
  probe's field without the centers' charges while the classical stage
  subtracts the charges this evaluation fitted, so the two halves answer
  different electronic structures. `md` is 3-step NVE in 43 s with seven
  duties per step and its own `&qmmm charge iteration` block; `mep` is 2 steps
  with `path_status.txt` starting at the unpolarized single point
  (`-152.5287129798`) and the grown image following it.
- `multicenter_qmmm_num_freq.yaml` - **no charge round**. `sc_charge` off, so
  the plan expands five duties (`field, field, qm, qm, energy`) and every
  evaluation probes from zero center charges. The displacement batch is not
  integrated, so there is nothing for a coherent charge set to protect here,
  and the batch stays half the cost of the fitted tier. On the corrected mask
  (the block is the two centers, not six atoms beside them - see #458) the
  deployment measured `10:47` for both carriers against the `2400` s budget
  this round raised `_timeout` to.
- `multicenter_qmmm_nac_sp.yaml` - **one coupling per center**. Two states and
  one declared pair on each center, so the run asks `energy gradient nac` of a
  composite and gets two blocks, not one averaged matrix: each center's rows land
  on its own atoms (`nac` shape `(2, 60, 3)`, pair 1 non-zero on rows 10-12, pair
  2 on rows 22-24 and nowhere else) and the carriers stay separate in
  `state.electronic[c]`. 19 s on the local host, `Status: completed`.
- `multicenter_qmmm_npt.yaml` - **the box under a barostat**. The stress comes
  from the whole-system classical duty alone (`stress` is in
  `planning/task.py::_SYSTEM_ONLY_CAPABILITIES`, so no center's cluster solve
  carries it), the barostat reads it through `state.active_stress()`, and each
  step hands the new cell to both probes and the classical surface. Three steps,
  41 s, `Status: completed`: first step `P = -19080.66 bar`, edge
  `8.39716243 → 8.39429460` Angstrom.
- `multicenter_qmmm_cell.yaml` - **the box as an optimizer variable**. Same
  composite, framework `minima` search, `optimize_cell: true` with
  `opt_cell_coupling: anisotropic`, `maxit: 2` (so it ends in the expected
  truncation, `failed: Maximum optimization steps reached`); 34 s, energies
  `-152.52883613` then `-152.54156374`, box strained to
  `a: 8.395978182473 / b: 8.395517681227 / c: 8.397996023406`, volume
  592.51160370 → 591.96284210. It also caught a spelling defect rather than a
  physics one: a variable-cell step wrote `lattice_vector` and left the state's
  other two readings of the same box (`cell`, `volume`) at the pre-strain
  numbers, so the run's own `&cell` block showed two boxes at once (#456, and
  its criterion `test_optimizer.TestVariableCellStepKeepsTheCellSpellings`).
- `multicenter_qmmm_neb.yaml` - **the band over the centers the endpoints move**.
  Three images interpolated between the same pair of structures the single-center
  carrier uses, each image one multi-center evaluation; the two centers are the two
  waters the endpoints displace (rows 1-3 and 4-6), so a per-center `E_QM` moves
  with the reaction coordinate. 2 min 14 s on the local host, two NEB steps to the
  declared `maxit: 2`: five profile points (`-152.5281727945` reactant,
  `-152.4629558045 / -152.3878677587 / -152.4558203436` images,
  `-152.5132112872` product) and `barrier` 0.1272069139 → 0.1253435285 Hartree.
- `multicenter_qmmm_mecp.yaml` - **each center's own pair under a crossing
  search**. `optimize.method: mecp` follows one gap, and over C>1 that gap is the
  sum of the centers' gaps: the composite's pair axis hands the search two
  synthetic surfaces, so the single-surface machinery works unchanged. The pair
  is named per center (`state_i: [0, 0]`, `state_j: [1, 1]`), which the input now
  parses - `Optimize.opt_state_i/j` take a list as well as a scalar (#460). 20 s
  on the local host, ending at `maxit: 1`: `cross_gap 0.57950573`,
  `gap_norm 0.20553932`, each center's own `E_QM` `-76.11922511 / -76.11188854`.
  The carrier is MECP and not MECI on purpose: a MECI's branching plane needs the
  derivative coupling between the two surfaces it minimizes over, and those two
  surfaces are sums of different centers' states - that coupling is not a
  per-center matrix element. The MECI spelling over C>1 reports
  `Branching Directions 1` and `Coupling Norm: Not measured` by construction.
- `multicenter_qmmm_md_fssh.yaml` - **one hopping ledger per center**.
  `sfhp: fssh` with `nactype: nac`, `root: "2; 2"` giving each center its own
  starting state, two steps, 42 s, `completed`. Every frame's `.md.pop` carries
  the per-center occupations and active states, and no hop occurs in those two
  frames (`fssh_force_hop 0`) - the reading is that the two ledgers run side by
  side on the composite's per-center coupling blocks.
- `multicenter_qmmm_ex_opt.yaml` - **excited-state minimization on the per-center
  ladder**. `state_i: [1, 1]` asks every center for its first excited state, the
  composite assembles exactly that surface, and the search reads that single
  energy as the requested state. 18 s to the declared `maxit: 1`:
  `Energy -152.45222204`, `Gradient RMS 0.01630798` (the ground-state `sp` on the
  same geometry is `-152.52883613`, so this is the excited surface).
- `multicenter_qmmm_md_restart.yaml` against
  `multicenter_qmmm_md_restart_base.yaml` - **the checkpoint pair**. Two
  processes (2 steps, resume the `<title>.rt` for 2 more) against one process of
  4 steps, same seed, same cadence. Every frame of the four-frame `.md.xyz`, the
  per-frame fitted charges `.md.charges`, the velocities, the populations and the
  energy table including both `E_QM` columns are equal digit for digit; the
  resumed process took 22 s against the baseline's 56 s, so no warm-up evaluation
  is due. The engine-free half of the same claim is
  `test_core.TestR4eSerializerVersioning.test_the_charge_store_and_each_centers_coupling_travel`,
  which puts the three objects a composite adds - the charge store, each center's
  coupling block, and the per-center active configuration - through the save-load
  path and checks what the restored run actually reads back.
- The Etot criterion does not judge a multi-center embedded composite
  (`runner._embedding_breaks_conservation`): the ESPF fit is re-done every
  frame and the force is the gradient at fixed charges, so the term coupling
  the centers is missing from it and the total is not a constant of motion by
  construction. Measured bands on `md`: 8.4e-3 (zero tier) and 9.7e-3 (one
  fitted round) against a 1e-3 budget, while the same record's kinetic energy
  alone swings 6.5e-3 over its three frames. Components, the composite
  identity and the per-center bookkeeping stay judged; single-center embedded
  cases and ONIOM keep the criterion.

The same five cases in the test environment host (round `tmp_20261104_124614`, both
carriers), read from the round's own logs:

- the three single-evaluation legs finished, and every term agrees with the
  local host digit for digit - `sp`'s five components, and `sc_charge`'s
  converged `E_QM -152.3057825700` with the same charge set, the same residual
  sequence and both centers' V/E rows. An earlier local reading of `-152.3063674100`
  was an artifact of a run made before the last round of fixes, not a
  cross-host difference: re-run with the current code the same input answers
  the test environment's number exactly, so the 5.848e-4 apparent split recorded here
  first is withdrawn.
- `num_freq` finished in `18:39` (yaml) and `17:28` (input) against the
  per-example `_timeout: 1200` budget - 93% of it - on the mask that measured a
  block containing no QM atom (#458). On the corrected mask and the budget this
  round raised to `2400` s it reads `10:47` for both carriers, i.e. 27% of the
  cap. The old number stands only as the reason the budget was raised.
- `md` and `mep` did **not** finish. Both carriers, both variants, the second
  composite evaluation: `Status: error: Composite evaluation failed (1
  term(s)): task 'field' (evaluation 2): qmmm_field V does not sum to the
  group/group Q-M energy`, residual 2.6454 against scale 249.11 for `md` and
  23.0140 against 159.6025 for `mep` (the gate is 1e-4 of the scale). Their
  probe directories carry `lammps_resident.perf` with `phase=evaluate steps=3`
  and `per_step_line='run 0'`, so the deployment ran them as a resident session
  - which is the path the test environment cannot exercise on a periodic box at all (it
  segfaults there, see below). The reciprocity gate has failed on that box
  before: rounds 3/4 reported residual 0.13034 on scale 270.236 for three
  `lammps_skill_qmmm_periodic_*` carriers, and #23 is where it stopped
  appearing anywhere in a round log.
  What has never been measured is the control that says whether this belongs to
  multi-center or to any multi-step periodic field leg on that build:
  `multiscale_qmmm_{embedding,no_embedding}_md`, `_mep` and `_neb` - the four
  single-center multi-step periodic legs - were all closed in this round
  (`disabled by`), so the next round should open them before this is called a
  multi-center defect.
  One more reading comes free from that round: the legs above ran the
  pre-division inputs, where every evaluation probes twice (once from zero,
  once with the accepted charges). At the new setting `md` and `mep` carry five
  duties and probe once per evaluation, so if the second probe of an evaluation
  is what the reciprocity gate cannot reproduce on that build, they go green
  without anything else changing.
  **What actually closed it was the rank count, not the probe count.** The same
  two legs failed again in the test environment at five duties and one probe per
  evaluation (residual 2.6482 against scale 249.32), which rules the second
  probe out. The next three readings localized it: the same input on the same
  host with `python_enabled: false` finished with energies equal to the local
  host digit for digit; the resident run, with the guard's operands printed,
  gave `sum(V) = -27.967417` against `group/group = -30.615639` out of **one
  and the same A pass**; and between the failing frames no atom crossed a face
  and the largest displacement was 0.048 A. The resident probe was inheriting
  `lammps_nproc: 2` from `pmd_env.yaml`, and on that build a per-atom kspace
  allocation summed across two ranks is not the group/group energy - the atom
  list exposed it, coming back as 29 + 31 = 60. Pinned to one rank the same
  session satisfies the identity, and `backend.py::_start_persistent_service`
  now pins the `qmmm_field` session to one rank (a probe that reads a handful of
  atoms' potentials buys nothing from a second rank); the re-verified round
  `tmp_20261104_205001` reports 156 PASS / 0 FAIL with every leg's own end line
  read back, and the probe log now says `1 by 1 by 1` while the energy leg keeps
  `1 by 1 by 2`.
  The four single-center multi-step periodic legs
  (`multiscale_qmmm_{embedding,no_embedding}_md`, `_mep`, `_neb`), which were
  the planned control for "multi-center-specific", are still unopened: the rank
  finding makes their answer unnecessary for this fix, and they remain owed as
  the regression arm for that build's periodic resident path.

The engine-level contract behind those numbers is tested directly, without
Molcas, against the LAMMPS binary in
`skills/lammps_skill/tests/test_qmmm_engine.py` on a nine-atom three-water
cluster (two centers plus a deliberately uncharged classical water, which makes
any nonzero probe potential attributable to the other center and to nothing
else):

- the two narrowed probes report the same inter-center pair from either side:
  `q_A . V(A) = q_B . V(B) = 0.01200838` Hartree;
- narrowing changes the answer, as it must: center A's rows read
  `0.552490 / 0.627721 / 0.608354` Hartree per element charge against
  `0.000000` for the same atoms measured over the union partition;
- the per-center in-cell removals sum to the total and each equals the
  standalone one-center run of that same water (`-9.17466068`, relative 1e-9);
- and the same three readings hold through a **resident** session
  (`python_enabled: true`), which feeds the environment charges by per-atom
  `set` instead of by a rewritten data file.

Host boundary found while running those: on the test environment's LAMMPS build, a
**periodic** QMMM field stage under a resident session dies inside the per-atom
`set ... image` restoration with a segmentation fault. It is not a multi-center
matter - the identical crash reproduces with one center (no `qm_probe` at all,
the pre-existing spelling), a direct backend call with the same deck and box
survives three frames, and a periodic ONIOM resident run on the same box
completes. The periodic multi-center readings above are therefore cold-stage.
`python_enabled: true` stays as the shipped examples set it because it is the
test environment's route, but that route is not yet shown to work for a
multi-step periodic run of either center count: the test environment's `md` and `mep`
legs reach a resident session and stop at the reciprocity gate on the second
evaluation (above), and the single-center multi-step controls that would say
whether the count matters have not been run.

The two channels built on top of that evaluation were also run here.
`multicenter_qmmm_num_freq.yaml` takes the framework's numerical Hessian of the
composite: the fix mask holds every atom outside the two centers, so the batch is
6 atoms x 3 = **18 gradient jobs**, and the analysis diagonalizes the mobile
block it actually moved. The block is the centers themselves (rows 10-12 and
22-24); an earlier revision of the file froze `7-60` and left atoms 1-6 mobile,
so the twelve modes recorded there (`-87.3564 cm-1` up to `3371.05 cm-1`, ZPE
`0.0469868196`) belonged to six atoms that are in neither QM region, and they are
withdrawn (CHANGELOG #458). Re-measured on the corrected mask: 7 min 31 s,
twelve modes `-2610.72 / -1717.53 / -190.04 / -87.89 / 112.35 / 142.05 / 207.24 /
329.83 / 1415.59 / 1933.79 / 2478.44 / 11388.92 cm-1`, zero-point energy
`0.0410257424` Hartree. **These numbers are not trusted yet** (#460): the batch
route that produced them disagrees with a direct +/-0.005 A difference on the
same geometry - `H_xx` of composite atom 9 reads 3.2183 from the batch against
0.4011 from the single points, and the C=1 arm's two hydrogen diagonals read
~1e-4 against 0.23 - so the case currently stands as a channel carrier, not as a
curvature reading. A partial Hessian of six atoms inside a frozen periodic
box is not a spectrum of the water - the four imaginary modes and the 11389 cm-1
mode are what a block that cannot relax its neighbours reads like - so what this
case establishes is the per-center assembly of the curvature, not the
frequencies.

**The C=1 paired reference** (CHANGELOG #458) puts three arms on the same
geometry (**withdrawn pending #460**: its matrices come from the batch route
above, so the relative differences it reported are not a reading yet): each center alone (9 mobile rows) and both together (18). Every arm is
reproducible from a clean directory bit for bit (`max|diff| = 0.0`). Read
element by element, the composite's intra-center block differs from the
independent single-center block by `max|diff| 2.412209` for center 1 (against a
block whose own largest element is 3.407439) and by `1.812208` for center 2
(against 0.02968973), while the inter-center block - the rows that exist only in
the composite - is `5.478333e-03` in Frobenius, **0.1%** of the mobile block. So
the composite's curvature is not the sum of two single-center curvatures: the
intra-center blocks move by 22.7% in norm overall (and much more for center 2),
while the coupling across centers is a tenth of a percent. One part of that
reading is not attributed yet: in the center-2 arm only the oxygen's three rows
carry magnitude (0.0152 / 0.0297 / 0.0189 Hartree/Bohr^2) while its two
hydrogens read 1e-4, and the same six atoms are all non-zero in the composite.
Both arms solve the same electronic state by energy (`E_QM -76.1561091900` and
`-76.1489616400` against the composite's `-76.1563741400 / -76.1494060000`), so
this is an open question about the measuring arm, recorded in `dev/TODO.md`,
not a defect of the composite evaluation. Before this round the
analysis kept the 54 held atoms in the vibration problem and reported 174 modes
whose first 168 read `-0.0000 cm-1 imaginary` (see
[methods/NUM_HESS_DEVELOPMENT.md](NUM_HESS_DEVELOPMENT.md)'s mobile-set bullet).
`multicenter_qmmm_mep.yaml` grows a band on the same carrier with the posture
its single-center twin ships (`maxit: 2`, `path_inner_maxit: 2`,
`path_radius: 0.15`): 23 s, two steps, and `path_status.txt` carries the band's
own columns for both - start image at `-152.5287129798`, which is the
unpolarized single point of the same geometry digit for digit, then a grown
forward image at `-150.5222703448` with `shift 2.0064`, `radius 0.15`,
`rms_g 4.659`. Each of those steps is one five-duty multi-center evaluation
with no charge round in it, which is what the division above asks of a band.

A later multi-center-only round (`tmp_20261105_152741`) repeated the fourteen
QMMM carriers in both modes: 28 example rows, all pass, and every printed
total equal between the yaml and input carriers. The readings were
`sp`/`sc_charge` `-152.5288361253` / `-152.5288369992`, `md`
`-152.5271661113`, `npt` `-152.5273886334`, `cell` `-152.5415637330`,
`mep` `-152.5288361253`, `md_fssh` `-151.8810699187`, and
`mecp`/`ex_opt`/`nac_sp` `-152.4540044312`; the `mecp`/`ex_opt` runs end on
the declared `maxit`, as the single-center carriers do.

Tests: `qualification/verification/test_qmmm_composite.py` (stitch, switch,
native-unit conversion, the two-center assembly, the charge iteration and the
`sc_charge` key face), `test_multicenter_qmmm.py` (the shipped multi-center
inputs through the real load pipeline, the coupling refusals and the search
state axis), `test_constraint_qmmm.py` (the constraint block on a QMMM run),
and the LAMMPS skill's own QM/MM tests
(`skills/lammps_skill/tests/test_qmmm*.py`).

## 10. Not implemented

- Polarization-only energy and its response contract: see
  [QMMM_NN_drafts.md](QMMM_NN_drafts.md). The current scheme is the additive,
  non-polarizable one.
- Charge embedding for QMMM, an OpenMM classical provider, and analytic
  multi-scale Hessians. (Multi-center QMMM and charged QM regions are
  implemented; what is still deferred is listed in section 8.)
- A QM provider without a V/E channel: today only Molcas (ESPF, CASSCF/CASPT2)
  can be the QM side of an embedded QMMM run; ONIOM charge embedding is the
  route for providers that take point charges instead.
- A crossing pair that spans two spin manifolds. Molcas under ESPF returns one
  state per spin, so a singlet/triplet pair arrives with a surface missing and
  the parser refuses it rather than guessing which one
  (`the log carries 1 state energy(ies) while the run declares 2 states`).
  Serving it needs the pass groups regrouped by spin, a non-SOC source for
  `singlet/triplet_ci_roots`, per-block `&ESPF` ordering, and
  `log_parser.blocks_per_iteration` ungated from
  `casscf_geometry_optimization`. Until then a crossing case names two states of
  one spin, which is what the shipped `mecp_cluster` does.
