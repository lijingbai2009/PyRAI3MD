# Numerical Hessian and vibrational analysis

Last updated: 2026-11-05 11:45 (#460 adds the open item above: the batch route on a multi-center composite disagrees with a direct single-point difference, with the four excluded causes listed; the single-center verification in section 6 stands. Earlier: 2026-11-04 12:08 (#451: the mobile-set bullet above is now what the code does - `Hessian.run` re-reads `mobile` from the displacement rows on the numerical route - with the 60-atom two-center symptom measured. The rest was not re-audited this round. Earlier: 2026-10-22 22:20 (section 3 now says the batch hooks are this workflow's own overrides rather than inherited from `Dispatcher`. The rest was not re-audited; section 4's freeze rule and the rest of the file stand as the 2026-10-12 pass left them.

Before it, 2026-10-12 23:25 (section 4's freeze rule checked against #338; the rest was not re-audited)

## 1. What the link is

One workflow owns the force constants of a structure and everything read out of
them: the vibrational frequencies and normal modes, and the harmonic
thermochemical corrections built from those frequencies. It replaced two files
that overlapped without either finishing the job: `workflow/hessian.py`
assembled the numerical matrix and stopped there, while `workflow/frequency.py`
re-asked for a Hessian and diagonalized a placeholder that returned
`(None, None)` and wrote a `normal_modes` key the result model never declared.

`workflow/hessian.py` now holds the workflow; `workflow/frequency.py` holds the
analysis functions and no workflow at all. A matrix that arrived analytically
from a backend and a matrix assembled from displaced gradients go through the
same `vibration`/`thermo_analysis` calls, so there is one arithmetic to be wrong
in.

## 2. Two decisions, taken independently

    which matrix   analytic   the backend returns `hessian`
                 numerical  this workflow displaces coordinates and differences gradients
    who computes it  driver=main   the framework, through the batch below
                     driver=skill  the engine, and the run never reaches this workflow

`jobtype` spells the first, `control.driver` the second:

| token | matrix | analysis | route default |
|---|---|---|---|
| `hessian` / `hess` | yes | no | `auto` |
| `frequency` / `freq` / `frequence` | yes | yes | `auto` |
| `num_hess` | yes | no | `numerical` |
| `num_freq` | yes | yes | `numerical` |

`auto` means: use the backend's matrix if the evaluation returned one, otherwise
displace. The switch is written into the run log, because silently reading a
different force-constant matrix than the input asked for is the failure mode
this table exists to prevent. `&hessian mode` overrides the token, and only a
value the input actually declared wins: a model default of `auto` in that field
would outvote `jobtype: num_hess` on every run, since by the time the workflow
reads the section every keyword is present.

The numerical tokens request no `hessian` capability from planning - only
`energy` and `gradient` - which is what lets a multi-backend run differentiate
gradients. A backend-owned matrix is refused there: an additive composite has
no single backend that owns its force constants. The refusal is keyed on the
resolved `hess_driver` (`schema.config.resolve_hessian_route`), so an explicit
`&hessian hess_driver: main` with `hess_mode` unset counts as numerical and is
allowed; only `hess_driver: skill` is refused (`planning/resolver.py`).

## 3. The batch

`generate_batch` -> `dispatch(units, child_jobtype='energy gradient')` ->
`collect` -> `postprocess`. The three hooks are the `Dispatcher` convention, and this
workflow owns their bodies: `workflow/hessian.py:265`, `:355`, `:484` (the base class
declares them abstract, `workflow/base.py:14`); `child_jobtype = 'energy gradient'` is
the class attribute at `hessian.py:163`.

* one task per sign and Cartesian component of every **mobile** atom: 2*3n
  evaluations, each materialized as its own child state with its own
  `runtime_NN/` directory (`runtime_isolated_workdir()` returns True, because
  6N calculations of one molecule sharing one scratch directory leave a trace of
  one of them);
* the child entry is a module-level function
  (`_EXECUTOR_MODULE`, `_child_entry`), because a bound method would pickle the
  workflow, its Calculator and its live backends into any payload that crosses
  a process;
* the difference quotient is `(g(+h) - g(-h)) / (2 h)` with **h in Bohr**: the
  displacement keyword is a distance in Angstrom while gradients are Hartree per
  Bohr, and the factor is what makes the result Hartree per Bohr^2;
* the matrix is embedded as the mobile block only, symmetrized there, and the
  rows and columns of held atoms stay zero. Averaging the whole padded matrix
  would smear a mobile-to-held coupling - a real number of this finite
  difference and a fiction as a force constant - into the held rows;
* a missing gradient marks the run failed and raises, naming the displaced
  atom and axis. It used to return None, write nothing, and still report
  `completed`.

`multitask`/`multipool` fan the batch out over a spawn pool; `remotetask`
submits each displacement as its own job and reads `result.json` back. Both arms
reach `collect` through the same `result['gradient']` access.

## 4. The analysis, and the units it is written in

`vibration(coord_bohr, mass_amu, hessian, mobile=None)`:

* mass weighting by `sqrt(m_i m_j)` with masses in amu;
* the rigid directions are **removed by rank**, not by count: six candidate
  columns (three translations, three rotations about the center of mass in the
  lab frame) are SVD'd and the complement of whatever is linearly independent is
  the vibration space. A linear molecule drops its axial rotation to zero and
  keeps four modes where the fixed-six arithmetic kept three; a single atom
  keeps none. The legacy OQP `frequency.normal_mode` this came from took
  `u[:, 6:]` unconditionally, which is the bug the CO2 test pins down;
* frequencies are `sign(lambda) * sqrt(|lambda|) * VIBRATION_TO_CM`, imaginary
  modes negative, the same convention the ORCA and OpenQP parsers use;
* modes are returned unmass-weighted and normalized so `sum(m x^2) = 1`, which
  is the convention `workflow/sampling.read_molden` assumes and
  `norm_mode` is declared with;
* rotational constants in GHz (`h / (8 pi^2 I)` is already an inverse second;
  multiplying by a speed-of-light factor, as the legacy module did, leaves every
  constant thirty times too small);
* a partial Hessian projects the rigid motions of the **mobile set about its
  own center of mass**: removing the whole molecule's translations from a
  fragment's block leaves the fragment's own drift in the answer as if it were
  a vibration;
* an atom held in all three components leaves the vibration problem entirely:
  the analysis diagonalizes the atoms the batch actually displaced (`mobile` is
  re-read from the displacement rows), because holding that set open instead
  adds a row of zeros per held atom and every one of them comes back as a soft
  mode - measured on the 60-atom two-center QMMM case, where 54 held atoms
  turned a 12-mode mobile block into a 174-entry table whose leading 168 rows
  read `-0.0000 cm-1 imaginary` (#451).
* when a `fix` plan freezes components instead of taking atoms out of the
  vibration problem, `vibrational_basis(mask=...)` removes only the whole-body
  motions the anchors **absorb** - the combinations whose frozen rows are zero,
  three of six for one pinned atom and none once two atoms pin an axis. A
  motion an anchor opposes is no longer a free motion: it is an ordinary
  deformation with an ordinary curvature, and a search that may not use it holds
  at a nonzero gradient forever (measured on the six-atom saddle in
  [OPTIMIZE_DEVELOPMENT.md](OPTIMIZE_DEVELOPMENT.md) section 15.2). `rank` in
  the returned tuple counts what was taken out, so with nothing frozen it is the
  rank of the rigid block itself and with a freeze it is six minus the rank that
  block reaches on the frozen rows.

`thermo_analysis(...)` is RRHO per molecule in Hartree, with the Grimme
quasi-RRHO free-rotor blend below `hess_cutoff` (alpha 4, both keywords
exposed). Standard state is the declared `hess_pressure`, 1 bar by default;
the rotational symmetry number `hess_symmetry` defaults to 1 and the run log
says so, because an uncorrected sigma leaves the entropy high by exactly
`k ln sigma`. Electronic entropy reads `system.multiplicity`, and 0 or absent
is treated as 1: the legacy default took the logarithm of zero.

The synthetic keys are the ones the result model declares and the analytical
backend fills, so the two routes can be compared entry by entry:
`zero_point_energy`, `internal_energy`, `enthalpy`, `entropy`, `free_energy`,
`heat_capacity_cv`, `heat_capacity_cp`. The pieces they were summed from go in
`terms`, and `utilities/print_results.py` prints
`&vibrational frequencies in cm-1`, `&normal modes`, `&hessian` (verbose 3)
and `&thermodynamic corrections`. None of those sections appears when the run
produced nothing, because the same formatter writes the per-step block of every
dynamics run.

`<title>.freq.molden` is written when `hess_molden: true`, into the run's own
directory, under exactly the name `sampling.load_frequency_data` looks for.

## 5. Keywords

`&hessian`: `hess_driver`, `hess_mode`, `hess_step`, `hess_atoms`,
`hess_vibration`,
`hess_thermo`, `hess_molden`, `hess_temperature`, `hess_pressure`,
`hess_scale`, `hess_cutoff`, `hess_alpha`, `hess_symmetry`, `hess_linear`,
plus the dispatch trio `multitask`/`remotetask`/`multipool`.

`hess_driver` and `hess_mode` are two questions and are cross-checked, not
independently free: the driver says *who* produces the matrix (`main` the
framework's own displacement batch, `skill` the backend computing it), the mode
says *how* (`analytic` a second derivative of the engine, `numerical` a finite
difference, `auto` either). Two of the six pairings have no producer behind them
and are refused where the input can still be changed: `main` with `analytic`
(the framework has no analytical second derivative) and `main` with `auto`
(`auto` is the statement that no producer has been settled on). `skill` with
`numerical` is legal and means the engine displaces its own atoms (ORCA's
`NumFreq`, Molcas' MCKINLEY): from this side that is still one evaluation
returning a matrix, which is why the framework never sees the difference.

The prefix is not decoration: `step`, `temperature` and `pressure` are also flat
names owned by `&md`, and the flattened routing handed the Hessian section's
values to the dynamics submodel (`step` reached it as a float where an integer
was expected, and the rest were silently ignored). Registering a keyword of
this section means four places: the model in `schema/config.py`, the type table
in `schema/adapter.py`, the whitelist property `PyRAIState.hessian_params`, and
the read in `workflow/hessian.py`. A missing entry does not raise; it reads the
default.

`hess_atoms` means something different during a search (#336). There the frozen
set belongs to the constraint plan - `fix` is the one statement about what may
move - and the atom set this batch displaces is derived from it
(`data/core.py::_derive_curvature_atoms`): written and matching is accepted,
written and clashing is refused, written with no `fix` behind it is refused
because the matrix would hold zero rows on atoms the run can move. The batch
itself now works per component (`_active_rows`), so freezing `x` of an atom
drops one row and not three, and a search refuses that granularity outright for
the rotational reason written out in `OPTIMIZE_DEVELOPMENT.md` section 14.

## 6. Verification, and what is not verified

Measured on B3LYP/6-31g* water (`qualification/verification/test_hessian_link.py`
carries the engine-free part, the numbers below come from a live PySCF run
through `python -m pyrai3md`):

| route | frequencies (cm-1) |
|---|---|
| PySCF analytical, direct | 1670.70, 3889.17, 4021.50 |
| framework, `hessian` (analytical) | 1670.67, 3888.97, 4021.33 |
| framework, `num_freq`, h = 0.01 A | 1669.72, 3889.37, 4021.78 |
| framework, `num_freq`, h = 0.005 A | 1670.40, 3888.92, 4021.55 |
| framework, `num_freq`, `multitask` pool of 3 | 1669.73, 3889.46, 4021.67 |

Halving the displacement moves each entry 0.7, 0.45 and 0.24 cm-1 toward the
analytical matrix, which is the O(h^2) behaviour the central difference
promises. Engine-free, the same code reproduces a known force-constant matrix to
1e-9 (a central difference of a quadratic is exact) and is provably independent
of the displacement; the negative control - scaling the reported gradient by the
Bohr factor - moves the assembled matrix by 1e-3, so the agreement is evidence
and not a comparison of a thing with itself.

The shipped `examples/hessian_num.yaml` is the route's only real-engine
qualification case (`jobtype: freq` takes the analytical matrix whenever PySCF
can serve one, so without a `num_freq` case the numerical link has no remote
reading at all): HF/sto-3g water, 19 displaced gradient jobs, 3.5 s on one host.
Measured there: numerical **2043.0734 / 4488.0646 / 4790.2488 cm-1** against the
analytical sibling's **2043.3249 / 4488.0061 / 4790.1960 cm-1** at the same
geometry -- a largest difference of 0.26 cm-1 at the 0.005 A step -- and the
`.input` carrier reproduces the numerical triple **digit for digit**, which is
the point #325 added, because the carrier used not to reach the engine at all:
the classic-input writer kept its own older list of `&HESSIAN` keywords and
refused the section at conversion time (`converter.WORKFLOW_SECTION_KEYS` knew
only the three dispatch words), while the yaml carrier converted and ran. The
writer now derives its list from the reader's `schema/adapter.HESSIAN_TYPES`,
and `TestHessianSectionCarrier` pins both the round trip and the refusal of an
unknown keyword.

Not verified anywhere: `remotetask` against a real SLURM controller, the
ORCA/Molcas/BAGEL/OpenQP analytical matrices through this analysis (they are
parsed by the skills and this document assumes only the result keys), PBC
(supported backends have no frequency profile), and multi-center runs (the
capability is refused for more than one backend, and `composite` refuses
`hessian` outright).

## 7. Open items

* The two sampling branches used to disagree about the mode normalization: the
  `boltzmann` branch divided the summed displacement and velocity by
  `sqrt(mass)` while `wigner` used the modes as read. Settled after this
  document was first written: the extra factor was right for the unnormalized
  Molcas vectors its legacy source read and wrong for every reader here, which
  normalizes to `sum(m x^2) = 1`. Both branches now convert through one
  `_mode_scales`, and equipartition is asserted on the sampled ensemble
  (`qualification/verification/test_sampling.py::TestEquipartition`,
  CHANGELOG #324).
* A partial Hessian has no thermodynamic meaning of its own - the corrections
  of a frozen fragment are not the corrections of a molecule - and the run
  reports them anyway, with the mobile set named in the same section.
* **Open (#460): the batch route on a multi-center composite does not agree
  with a direct single-point difference.** On the same geometry,
  `H_xx` of one QM-region atom reads 3.2183 from the 18-job batch and 0.4011
  from +/-0.005 A single points evaluated one at a time; a single-center arm's
  two hydrogen diagonals read ~1e-4 against 0.23. Excluded already: the batch
  displacements themselves (`generate_batch` emits +/-0.005 A and the row order
  matches `_batch_rows`), the electronic surface (that same step moves `E_QM` by
  6e-4 Hartree), the constraint block (0.4010 with it, 0.4011 without), and
  run-to-run noise (clean-directory repeats are bit-identical). The batch route
  itself is verified on a non-composite system (section 6's PySCF water, where a
  central difference of a quadratic is exact to 1e-9), so what is open is what a
  *composite* child evaluates, not the differencing. Until it is closed, the
  shipped `multicenter_qmmm_num_freq` case is a channel carrier and its
  frequencies are not a curvature reading.
* The transition-state search asks for its initial curvature from either
  producer through `hess_driver`, and both routes were run to convergence
  against a live PySCF backend (2026-10-12, CHANGELOG #335). The `skill` route
  is a single dispatched child job asking the backend for its matrix, so it
  works inside a run whose loop is otherwise evaluated for energy and gradient
  alone; the child's own mode is pinned to `analytic` so an engine that cannot
  answer fails by name instead of starting a displacement batch inside the
  worker. A multi-backend run is refused while naming the skill as the producer
  (one geometry has one force-constant matrix), and so is a partial Hessian,
  whose held atoms come back as exactly zero curvature.

---

See also: [ARCHITECTURE.md](../ARCHITECTURE.md),
[WORKFLOW_DEVELOPMENT.md](../WORKFLOW_DEVELOPMENT.md) section 7.3,
[OPTIMIZE_DEVELOPMENT.md](OPTIMIZE_DEVELOPMENT.md) section 15.2,
[ENSEMBLE_DEVELOPMENT.md](ENSEMBLE_DEVELOPMENT.md) section 8 for the barostat's
use of the same stress tensor.
