# PyRAI3MD ONIOM (Two- and Three-Layer) Development Guide

Last updated: 2026-11-06 13:30. Section 4's adaptive paragraph was re-read
against the shipped carrier and extended with what that run records
(`examples/adaptive_oniom2_local.yaml`: `source: layer`, `aux: []`, no
sidecar, and labels bit-identical to the QMMM spelling). Sections 5-9 were not
re-audited. Earlier: 2026-11-05 15:25. Sections 5-9 were added with the true-embedding
test set (ORCA/BAGEL/Molcas high-level consumers, PySCF/xTB charge sources) and
the multi-center ONIOM2 outer-charge semantics; the charged ONIOM2/ONIOM3
carriers and their per-layer charge rule were added to sections 7 and 8 the
same day. Sections 1-4 were re-read against
`workflow/multiscale/charges.py`, `workflow/calculator.py` and
`schema/validate.py` this round. Earlier: 2026-10-31 23:20 (#445). Section 3.1 gained the two-phase ownership rule, checked against `workflow/calculator.py:1249` and `workflow/multiscale/charges.py:112`: the excluded rows are derived from the whole queue and handed to `update_from_outcomes` as `sources`, because deriving them from the phase list is what let theory[1] overwrite the high-level rows. Before it, 2026-10-30 22:46. Sections 2 and 3 were re-read against the code
this round: the two/three-layer contribution formulas against
`workflow/multiscale/composite.py::assemble_results` and
`planning/task.py::_expand_oniom_centers`, and the charge-routing rules
against `workflow/multiscale/charges.py`, `data/charges.py` and the
calculator's store wiring. Section 1 is a placement statement, not an audit.
The generic aggregation machinery (planning, execution, scatter-add,
serialization, `&composite` breakdown) stays in
[COMPOSITE_DEVELOPMENT.md](../COMPOSITE_DEVELOPMENT.md); the N-center
extension stays in [MULTICENTER_DEVELOPMENT.md](../MULTICENTER_DEVELOPMENT.md).

---

## 1. Scope and placement

ONIOM is one scheme of the multiscale composite layer. A run declares two or
three theory methods and a partition (`highlevel`, `midlevel`); the planner
expands the queue, the calculator executes the terms, and the composite
module combines them by the contribution coefficients. Nothing in this
document is a separate code path: the ONIOM-specific part is (a) which terms
are queued and with what sign, and (b) which term is allowed to write
environment charges for an electrostatic embedding.

Three-layer runs additionally support electrostatic embedding of the
high-level model in the midlevel's charges; two-layer runs support it when
the outer method is quantum (its computed charges fill the outer region).
A two-layer run with a classical outer method and `embedding: true` is
refused with a pointer to QMMM, whose field channel replaces it
([QMMM_DEVELOPMENT.md](QMMM_DEVELOPMENT.md) section 3).

## 2. Contribution formulas

### 2.1 Two-layer ONIOM

For N non-overlapping centers, the two-layer energy is

```
E = E_L(real) + SUM_c [E_H(M_c) - E_L(M_c)]
```

and the gradient scatters each term's local gradient to its
`region_indices` before multiplying by the contribution coefficient:

```
G[:, i, :] += coeff * g[:, j, :]
```

The contribution pattern of one center is therefore `+1, +1, -1`: the
whole-system low-level term, the high-level model term and its low-level
correction. `resolve_task_queue` / `_expand_oniom_centers` build the queue
with those coefficients; `composite.assemble_results` only sums.

### 2.2 Three-layer ONIOM

For N centers with intermediate regions,

```
E = E_L(real) + SUM_c [E_M(I_c) - E_L(I_c) + E_H(M_c) - E_M(M_c)]
```

which is the pattern `+1, +1, -1, +1, -1`: real(basis), inter(+1), inter
correction(-1), model(+1), model correction(-1). Each term keeps its own
`region_indices`; the scatter-add form is identical to the two-layer case.

The region derivation is documented once, in
[ONIOM region semantics](MULTICENTER_DEVELOPMENT.md): `highlevel` and
`midlevel` are disjoint input lists, `region_h = highlevel`,
`region_m = highlevel ∪ midlevel`, `region_l = the whole system`.

## 3. Charge management and electrostatic embedding

A three-layer ONIOM run may embed the high-level model in the electrostatic
field of its surrounding midlevel. Charge management is switched on by
`embedding` (compute and embed) or `read_charge` (load a static field); the
computed route requires a term that produces charges, and the static route
requires `{title}.charge`. `highlevel` and `midlevel` are disjoint input
regions, so the midlevel is exactly the embedding source, and the mid-level
tasks cover `region_m = highlevel ∪ midlevel`.

### 3.1 Charge routing rules

Per-atom partial charges are routed by geometric region and by the theory
method that produces them (`workflow/multiscale/charges.py`):

- high-level charges come only from theory[0], the high-level provider
  (`provides_electronic=True`); its whole region H_c is updated;
- mid-level charges come only from theory[1]'s inter term (`layer_id==1`,
  `region=='inter'`); that task covers `region_m` and only the disjoint I_c
  part is written, while its high-level rows are ignored;
- the low level (theory[2]) and every correction term return no charge
  update.

A source term whose backend emits no `computed_charges` is skipped with an
info log, never an error. `computed_charges` stays an `IGNORED_KEYS`
byproduct for multi-center aggregation, so the two consumers never conflict.
Which skills can be a source is a skill-side fact: the molcas, pyscf and xtb
skills return it; openqp does not. The consumer side is the same kind of fact:
ORCA writes `%pointcharges` beside a `<project>.pc`, BAGEL appends the rows as
`Q` atoms, and Molcas folds `embedding: true` to `'charge'` and writes
`&ESPF External = N Angstrom`; openqp writes none.

`computed_charges` is `(nstate, nlocal)`, one row per state, and since #450 the
row written is the state the task itself ran (`Calculator.
_charge_embedding_state`: the task's own target states, else the system's
`current_state`), the same rule the QM/MM stage uses. Row 0 is therefore not a
standing rule: a payload that carries fewer rows than the requested state
writes **no** row and logs a warning, because embedding one surface's charges
under another surface's name is the defect this channel exists to prevent
(#36 fixed the same thing on the QM/MM side).

An embedded run executes in two phases - the non-provider terms first, then
the providers, whose embedding needs the charges the first phase produced -
and only the providers appear in the second phase's list. Which rows belong
to H_c is a property of the partition, so it is read off the whole queue and
handed to `update_from_outcomes` as `sources`; deriving it from the phase
being scattered answers "none" in phase one, theory[1] then writes its own
H_c part over the high-level rows, and for a provider that emits no charges
(bagel, openqp) those are the rows the run keeps and prints. The embedding
handed to the provider was never wrong that way - it comes from
`embedding_regions`, not from the store - which is why the energies of the
affected cases stayed right while `&highlevel point charges` and
`has_values` were not.

Two-layer runs route the same way with only theory[0] and theory[1]: the
outer region (every atom outside `highlevel`) is the embedding source, filled
by theory[1]'s computed charges in the normal two-phase order.

### 3.2 ChargeStore and the managed region

`data/charges.py::ChargeStore` owns one global per-atom array
`atom_charges (natom,)`. Only the managed region is ever updated — for
three layers `region_m`, for two layers the whole system — and every other
atom stays at 0. It derives whole-system `[q, x, y, z]` rows (row i is atom
i) from a coordinate array handed to it, so it imports nothing from the
workflow layer. `embedding_region_by_center` (the disjoint midlevel) and
`managed_region_indices` are pure set helpers shared by the data and
workflow layers. A supplied `system.point_charges` seeds the store;
`read_charge` then fills the midlevel and the two-phase run fills the high
and mid level from the computed charges.

The store is built by `build_state`, persisted through `PyRAIState.to_dict`
and restored on restart; a legacy `.rt` migrates to `charge_store=None`.

### 3.3 `read_charge` without a partition

The same keyword has a second meaning, decided by one predicate
(`schema/validate.py::is_standalone_queue`, read by both the validator and the
task queue so the two cannot disagree): one electronic provider, at most one
reference, and no `highlevel`/`midlevel`. There is no environment for
`embedding` to steer, so `read_charge: true` is the whole embedding decision:
it reads `{title}.charge` and distributes what the file declares.

The file's first line is the declaration, which keeps one keyword from
meaning three things (`data/charges.py::read_embedding_file`):

```
point angstrom external   # rows q x y z: arbitrary external points -> point_charges
point angstrom atom       # rows q x y z: one per mid-shell atom (ONIOM only)
field au                  # rows V Fx Fy Fz: one per QM atom -> external_potential
```

A point document lands on the charges channel of the first electronic theory
task and the first reference task; a field document lands on the potential
channel of the same tasks. One file carries one kind, so a run can never hold
both, the calculator refuses to assemble both, and the log prints whichever
arrived — never two blocks. `point ... atom` is refused here: a standalone
run has no mid shell to map the rows onto.

Under a partition the switch is read differently: ONIOM consults it only when
`embedding` is on (it then replaces the computed mid-shell charges with the
file's, which must be `point ... atom`), and QMMM does not read it at all —
the MM stage supplies the field every step, so both values are cleared
without a warning and the field always comes from the composite. A `field`
document under ONIOM is refused.

## 4. Where the code lives

| Concern | Location |
|---|---|
| queue expansion and contribution coefficients | `planning/task.py::_expand_oniom_centers` |
| region derivation | `planning/task.py`, `data/center.py` |
| aggregation and gradient scatter | `workflow/multiscale/composite.py::assemble_results`, `multiscale/regions.py` |
| charge routing | `workflow/multiscale/charges.py` |
| charge store | `data/charges.py::ChargeStore` |
| input validation (embedding vs partition shape) | `schema/validate.py` |
| outer-reference capture (ONIOM2) | `workflow/multiscale/charges.py::capture_outer_reference` |
| two-phase execution | `workflow/calculator.py::_run_two_phase` |

Adaptive sampling reuses the same center-based expander for its
reference queue, but with `molecule.highref/midref` instead of
`highlevel/midlevel`; the reference list length is its own layer count.
The training label is the reference layer whose region equals the
neural-network surface (or the composite total when the reference spans
the whole system), never a positional rule. See
`dev/methods/ADAPTIVE_DEVELOPMENT.md` section 14. The mechanical spelling is
shipped as `examples/adaptive_oniom2_{local,slurm}.yaml`, where the reference
high layer (`pyscf`) covers the same six atoms as `highref`, so the label is
that layer's raw term and no auxiliary channel exists to write: the
`&method alignment` record reads `source: layer`, `aux: []`,
`reference_type: oniom2`, and `iter-2/reference/` holds no `aux-labels.json`.
`schema/validate.py` warns on this input - a classical outer method with
`embedding: false` is the same additive combination as QMMM without embedding -
and on the two shipped cases the warning shows up as measured agreement: the
ONIOM2 and QMMM local carriers returned bit-identical labels (`max
|difference| = 0.0` on both energy and gradient, five frames of two states).

## 5. Capabilities and jobtypes

`energy`, `gradient`, `optimize` (including `optimize_cell`), `md`
(`nve`/`nvt`/`npt`) and the crossing/path searches (`meci`, `mecp`, `ts`,
`mep`, `neb`) run through the same composite evaluation; the framework drives
the loop and asks each layer for one evaluation per geometry. `nac` is a
provider property: the high-level block answers its own pair.

A run is **electrostatically embedded** only when its high-level carrier
consumes point charges. The shipped consumers are ORCA (`%pointcharges` plus
`<project>.pc`), BAGEL (background rows as `Q` atoms) and Molcas
(`&ESPF External = N Angstrom`, folding `embedding: true` to `'charge'`).
openqp writes none, so an openqp-high run with `embedding: true` is a
bookkeeping control, not an electrostatic measurement. The charge sources are
PySCF, xTB (including GFN-FF) and Molcas under ESPF; ORCA returns no
`computed_charges` and cannot be an outer/mid source.

Two carrier limits are measured, not inferred. ORCA refuses NACME while point
charges are embedded and its input builder drops the block, so the embedded
`nac` carrier is Molcas (its two-pass ESPF + field-free deck). And a computed
ONIOM field moves with the geometry: the single-center ORCA/PySCF MD case
printed a 2.45e-3 Hartree three-frame Etot band against the 1e-3 budget, which
is why the qualification criterion reads a computed embedded ONIOM run as
"not judged" (a static `read_charge` field stays judged).

## 6. Periodic and charge limits

The three-layer periodic carriers put the whole system, and the stress, on the
LAMMPS low layer; the high-level model and the mid-level source are molecular
clusters cut from the frame (`cell=None` for a region task), the same split the
QMMM periodic cases use. `pbc` and `cell` are declared in the input; a
region-sliced task never inherits the global cell implicitly. The two-layer
shipped carriers are non-periodic clusters; a periodic ONIOM2 with a quantum
outer is not shipped.

## 7. Multi-center

Two forms are supported.

**ONIOM3 (three layers, C centers).** `_expand_oniom_centers` builds
`1 + 4C` duties: one whole-system `L(real)` and, per center, `+M(I_c)`,
`-L(I_c)`, `+H(M_c)`, `-M(M_c)`. Each high-level model is embedded only in its
own center's midlevel, and `_embedding_regions_by_center` returns exactly that
disjoint mid shell. No center sees another's mid shell.

**ONIOM2 (two layers, C centers).** The queue is `1 + 2C`: `L(real)` and, per
center, `+H(M_c)`, `-L(M_c)`. Center c's environment is every atom outside its
own `highlevel`, which includes another center's high-level atoms; the
environment charges are that evaluation's whole-system `L(real)` low-level row
(`capture_outer_reference`). One row fills every environment — there is no
cross-center charge iteration and no second quantum density in the field. The
capture is a per-evaluation array, not the persistent store: the store's
high-level rows still belong to the providers, so `&highlevel point charges`
never shows theory[1]'s numbers under the high-level name (#450).

A static `read_charge` file under several ONIOM2 centers is refused: the rows
would have to be bound per center, and one file has no such binding. A
classical (LAMMPS) outer layer is refused for ONIOM2 embedding and points to
QMMM, whose field channel replaces it.

**Charged centers.** `molecule.charges` and `molecule.multiplicity` are read
per center (`CenterSpec`), and the whole-system `L(real)` uses their sum and
their unpaired-electron pairing (`system_charge` / `system_multiplicity`).
Every region task of a center carries that center's charge, and the
full-system task carries the sum. A three-layer center's midlevel must be
neutral: the mid region `H ∪ I` inherits the center's charge one-to-one, so
the net charge belongs in the high region and the mid shell only supplies the
field. Shipped charged carriers:

- `multicenter_oniom2_orca_charged_sp` / `..._no_embedding_sp` / `..._md`: an
  ammonium centre (+1) and a neutral water; the water centre's `.pc` carries
  the ammonium's five L(real) rows (sum +1), the ammonium centre's carries the
  neutral water's three.
- `multicenter_oniom3_orca_charged_sp` / `..._no_embedding_sp` / `..._md`:
  ammonium high (+1) with a neutral water midlevel, a neutral water centre,
  and an outer water; ORCA is the high and low method, PySCF the midlevel
  (`theory: [orca, pyscf, orca]`). Each high-level model embeds in its own
  neutral mid shell; the real layer runs at +1.

Shipped multi-center carriers (`family` = the individual `multicenter_*`
switches, no group key):

- `multicenter_oniom2_orca_embedding_sp` / `..._no_embedding_sp`: two ORCA
  high-level waters, each in the other's PySCF HF charges; the control.
- `multicenter_oniom2_orca_embedding_md`: the same pair for three NVE steps;
  the Etot criterion is not judged (computed embedded composite).
- `multicenter_oniom2_molcas_embedding_nac_sp`: each center's own singlet pair;
  the Molcas two-pass deck; engine-gated (Molcas).
- `multicenter_oniom3_orca_embedding_sp` / `..._npt`: two ORCA/PySCF centres
  in the 60-atom LAMMPS box, each embedded in its own mid shell.

## 8. Acceptance

Read on this environment (ORCA 6.1.1 local build, PySCF 2.14, LAMMPS
Python). Molcas, xTB and BAGEL cases are engine-gated and **not run** here;
their deck-level checks are the skill tests.

| case | jobtype | reading |
|---|---|---|
| `multiscale_oniom2_orca_embedding_sp` | energy gradient | total `-150.2134569082`; high `-75.2519928809`; `.pc` 3 rows |
| `multiscale_oniom2_orca_no_embedding_sp` | energy gradient | total `-150.2133745069`; high `-75.2519104796`; no `%pointcharges` |
| A/B difference | | `-8.24013e-5` Hartree |
| `multiscale_oniom2_orca_embedding_md` | md nve 3 | `-150.2134569082`, `-150.2143281852`, `-150.2159114076`; band `2.45e-3` (not judged) |
| `multiscale_oniom2_orca_embedding_meci` | optimize meci | gap `0.41192687` then `0.35736808`; stops at `maxit 2`; ORCA omits NACME under charges |
| `multiscale_oniom2_orca_embedding_neb` | optimize neb | five profile points, `-150.2134569` (reactant) to `-150.2134523` (product); 3 images; stops at `maxit 2` |
| `multiscale_oniom3_orca_embedding_npt` | md npt 3 | totals `-150.5425338625`, `-150.5533109724`, `-150.5624113431`; edge `8.39882808 → 8.39857145 → 8.39794614`; high `.pc` = mid 3 rows |
| `multiscale_oniom3_orca_embedding_cell` | optimize minima + cell | volume `592.51160370 → 592.33754479`; edge `a 8.39909100 → 8.39784563`; expected `Maximum optimization steps reached` |
| `multicenter_oniom2_orca_embedding_sp` | energy gradient | total `-150.5041911616`; each high `.pc` carries the other centre's 3 atoms |
| `multicenter_oniom2_orca_no_embedding_sp` | energy gradient | total `-150.5040249928`; no `%pointcharges` |
| A/B difference | | `-1.661688e-4` Hartree |
| `multicenter_oniom2_orca_embedding_md` | md nve 3 | `-150.5041911616`, `-150.5051273404`, `-150.5067411925`; completed |
| `multicenter_oniom3_orca_embedding_sp` | energy gradient | total `-300.7528934636`; each high `.pc` = its own mid 3 rows |
| `multicenter_oniom2_orca_charged_sp` | energy gradient | total `-131.3503737228`; water centre `.pc` 5 rows sum `+1`; ammonium centre `.pc` 3 neutral rows |
| `multicenter_oniom2_orca_charged_no_embedding_sp` | energy gradient | total `-131.3464247919`; no `%pointcharges` |
| charged ONIOM2 A/B difference | | `-3.94893e-3` Hartree |
| `multicenter_oniom2_orca_charged_md` | md nve 3 | `-131.3503737228`, `-131.3594691111`, `-131.3734648577`; completed |
| `multicenter_oniom3_orca_charged_sp` | energy gradient | total `-356.3586172761`; 9 tasks; each high `.pc` 3 neutral mid rows |
| `multicenter_oniom3_orca_charged_no_embedding_sp` | energy gradient | total `-356.3578010763`; no `%pointcharges` |
| charged ONIOM3 A/B difference | | `-8.161998e-4` Hartree |
| `multicenter_oniom3_orca_charged_md` | md nve 2 | `-356.3586172761`, `-356.3818173232`; completed |
| `multiscale_oniom2_molcas_{embedding,no_embedding}_sp` | energy | not run (no Molcas here) |
| `multiscale_oniom2_molcas_embedding_nac` | energy gradient nac | not run; ORCA refused this leg |
| `multicenter_oniom2_molcas_embedding_nac_sp` | energy gradient nac | not run (Molcas) |
| `multicenter_oniom3_orca_embedding_npt` | md npt 2 | not run locally; same queue as the sp carrier |

The A/B pairs share atoms and coordinates exactly
(`test_oniom_embedding_examples`); the deck check is `%pointcharges` present
versus absent. The Molcas pair's deck-level proof is
`test_caspt2_espf.py::test_espf_nac_uses_two_passes` and
`test_espf_charge_and_potential_blocks`.

**Test round `tmp_20261105_152741` (host 192.0.2.10).** The round ran
the multi-center set only (26 ONIOM2/ONIOM3/QMMM cases in both carriers): 159
active tasks, **155 pass / 0 fail / 0 error**, all 52 example rows pass, and
every case that prints a composite or QMMM total has the yaml and input
carriers equal to the printed digit (`d = 0.000e+00`). The ONIOM A/B splits
there are neutral ONIOM2 `-1.721002e-4`, charged ONIOM2 `-3.953652e-3` and
charged ONIOM3 `-8.213328e-4` Hartree; the QMMM `sp`/`sc_charge` pair reads
`-152.5288361253` / `-152.5288369992`, and its band, NPT, cell and crossing
cases carry their usual readings. The cases that end on `maxit` (`cell`,
`mecp`, `ex_opt`, `neb`) are the intended truncations.

The round also exposed four main-log gaps, three now fixed and one made
explicit: the pre-fix logs had no per-center ONIOM field/state/region and no
per-center QMMM state (fixed by `composite_scheme` in the multi-center
assembler, `oniom_outer_charges`, and `logger._result_centers`); no per-center
NAC/SOC (fixed); no final NEB band summary (fixed with `&path summary` from
the optimizer's own report); and no reference electronic energy in a
`num_freq` result (the workflow does not evaluate that surface, so the log
now says the G/H/S rows are thermal corrections alone; computing it would be
a workflow change, not a formatter one). The remote artifacts predate the
formatter fixes; the new blocks were verified on the same ONIOM2/ONIOM3 and
NEB cases locally and with the QMMM/NAC mock tests.

**Log blocks.** The multi-center assembler now writes `composite_scheme`, so
the log selects the ONIOM view instead of the plain whole-system `&point
charges` block. For C>1 the result block prints, per center,
`&center N highlevel point charges`, `&center N midlevel point charges`
(three-layer) or `&center N embedding point charges` (two-layer, from the
recorded `oniom_outer_charges` L(real) row); the final non-step block also
carries the per-center `&electronic state` and
`&region charge & spin multiplicity` rows (`logger._result_centers`). The
QMMM 0-order charge block is per center as well. This is what makes a charged
or two-center run's main log self-contained: the high-level provider's own
rows (zeros for ORCA) and the field each centre was embedded in are separate
blocks.

## 9. Not implemented

- openqp high-level charge embedding (no point-charge writer); the openqp
  `embedding: true` cases stay as layered-bookkeeping controls.
- Multi-center ONIOM2 `read_charge` (one static file has no per-center
  binding), and classical (LAMMPS) outer ONIOM2 embedding (use QMMM).
- Cross-center charge iteration for ONIOM2: each evaluation embeds in the
  current `L(real)` row; there is no fixed-point solve like QMMM `sc_charge`.
- ORCA NACME under embedded point charges; the embedded `nac` carrier is
  Molcas.
- Link atoms for a bond crossing a partition, and overlapping centers.
