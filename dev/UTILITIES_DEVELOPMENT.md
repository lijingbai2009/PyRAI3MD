# PyRAI3MD Utilities Development Guide

Last updated: 2026-11-12 19:20 (rule 6 records the import-free schema/reference and the schema/__init__ unit re-export. Earlier: 2026-10-24 09:40 (the utilities refactor landed: geometry/cell/topology/CV split, structure formatting, and the `opt_status.txt` write/read split; sections 2-9 and 12 were read against code this round, sections 10-11 were re-run, section 13 is a list of deliberately deferred items. The old `dev/utilities_refactor.md` was deleted and this document supersedes it.)

---

## 1. What this document is, and what it is not

This is the design authority for `pyrai3md/utilities/`: module boundaries, import directions, public names, contracts and verification gates. It records how the package is put together, not what a workflow or a skill decides.

It does not own:

- the constraint semantics, CV spelling in an input, or the image rule a constraint applies — `dev/methods/CONSTRAINT_DEVELOPMENT.md`;
- the run-log grammar and verbose levels — `dev/LOG_DEVELOPMENT.md`;
- the PBC derivation and the image-flag seed — `dev/methods/PBC_DEVELOPMENT.md`;
- the optimizer's columns or the status record's consumers — `dev/methods/OPTIMIZE_DEVELOPMENT.md` and `dev/methods/CELL_OPT_DEVELOPMENT.md`;
- qualification verdicts and round bookkeeping — `dev/QUALIFICATION_DEVELOPMENT.md`.

`dev/ARCHITECTURE.md` remains the only maintainer of the module tree and the document index. This document records the semantics behind that tree. `dev/utilities_refactor.md` is retired: its D1-D8 decisions were carried into section 2, and its statements that are no longer true were replaced rather than kept in parallel.

## 2. Decisions

| ID | Decision |
| --- | --- |
| D1 | The topology half of `image_flags.py` moved to `utilities/bonds.py`; `image_flags.py` was deleted with no shim. |
| D2 | `geometric.expand_components` became `bonds.unfold_components`; `geometric.py` is numpy-only again. |
| D3 | Bond perception uses the certified nearest-image search in `lattice.min_image_displacement`. A cell the certificate cannot resolve is refused by `ValueError`; the configuration path warns and returns no counts, while the constraint path refuses a multi-atom CV by name. |
| D4 | `pbc=None` keeps one meaning per entry point: `lattice.normalize_pbc(None)` is non-periodic, `bonds.periodic_flags(None)` is fully periodic, and `min_image_displacement` never accepts `None` (its default is `(1, 1, 1)`). |
| D5 | The CV kind vocabulary is `'bond'`, `'angle'`, `'dihedral'`; `'torsion'` is no longer a dispatch spelling. |
| D6 | Cell CVs stay in `cv`; `group_center` and `_as_coords` live in `geometric`; `evaluate_on_images` lives in `cv`; the utility copy of `require_same_component` was deleted, and the live refusal is `workflow/constraints/images.py::require_same_component`. |
| D7 | `cv.cell_volume` and `lattice.cell_volume` are different questions with different degeneracy contracts and stay separate. |
| D8 | No new switches or modes. The only signature extensions are `pbc=(1, 1, 1)` on `min_image_displacement` and `indent='  '` on `structure_formatter.print_coord`. |
| D9 | Bond data and bond policy are separate. `schema.reference.BOND_LENGTHS` owns the table; `utilities/bonds.py::bond_lib` owns the lookup and the 1.2 Angstrom default for an unlisted pair. `bonds.py` is kept. |
| D10 | `utilities/structure_formatter.py` is the single home of structure-table text used for internal data communication: atom/vector rows, charge rows, matrices, cells, PBC flags and the TurboMole `$coord` block. `coordinates.py` was deleted; `io.py` keeps parsing only. |
| D11 | `utilities/opt_formatter.py` owns only the `opt_status.txt` header and row format. `utilities/print_results.py` owns the artifact's read side: `OptStatusRecord`, `read_status`, `classify_header`, `assert_writable`, `format_malformed`. The artifact name does not change. |
| D12 | `ComponentError` lives in `bonds.py`, the only remaining raiser (`unfold_components`). `cv.evaluate_on_images` raises builtin `ValueError` for an unknown kind or wrong index count, so `cv` never imports `bonds`. |
| D13 | No compatibility aliases. `utilities/__init__.py` no longer exports `print_charge`; `print_charges` is the exported name; `print_coord` resolves to the indented structure formatter. `image_flags`, `coordinates` and `opt_status` module paths are gone. |
| D14 | `_as_coords` moved from `cv` to `geometric` because it raises `CVSingularError`, which belongs there; `cv` imports it privately. `cv._acos` was deleted; `cv.cell_angle` uses `geometric._acos`. |
| D15 | `test_image_flags.py` became `test_bonds.py`; `test_opt_status_record.py` keeps its name because it tests the artifact; `test_structure_formatter.py` was added. |

The one accepted upward edge is `X -> pyrai3md.schema.reference`. `reference.py` contains no import statement, `io.py` and `geodesic.py` already used it, and the bond table's data does not belong to an algorithm module. The topology module therefore imports `numpy`, `.lattice` and `..schema.reference`, and nothing else.

## 3. Module map and import graph

`utilities/` has 18 files: `__init__.py` plus 17 modules.

| Module | Duty |
| --- | --- |
| `__init__.py` | Package public API: parsers, structure formatters, bond lookup/topology, dense geometry adapters, lattice, units, timer, filesystem, logo. |
| `bonds.py` | Bond-length lookup, perception, bond graph, connected components and offsets, unfolding, initial image counts. `bond_lib` reads `schema.reference.BOND_LENGTHS`. |
| `cv.py` | Constraint-facing CVs, compact Jacobians, cell CVs, periodic differences, Jacobian scattering, image-space CV evaluation. |
| `filesystem.py` | Directory creation, safe copy, temporary working-directory context. |
| `geodesic.py` | Riemannian interpolation over Morse-scaled internal distances. |
| `geometric.py` | Bond/angle/torsion values with exact first and second derivatives, dense adapters, mass-weighted group centers. Numpy-only. |
| `io.py` | Input parsing only: coordinate/charge/initcond readers, file-frame readers, symbol/number conversion. |
| `lattice.py` | Every pure cell operation: 6-parameter/matrix, volume, pbc, fractional/Cartesian, wrap/unwrap, certified nearest image, interplanar half-bound, stress conjugates. |
| `logo.py` | Startup banner and version. |
| `opt_formatter.py` | The `opt_status.txt` header and row format: columns, widths, `status_header`, `status_line`. No imports. |
| `print_results.py` | Run-log result blocks at verbose 0-3, and the `opt_status.txt` read side. |
| `read_tools.py` | `ReadVal`/`ReadIndex` value and index parsing. |
| `rmsd.py` | Aligned RMSD as a CV: weighted Kabsch, D^2 and its derivative. |
| `selection.py` | Atom selection: indices, ranges, elements, named groups. |
| `serialization.py` | `json_safe` and atomic `write_json` for result documents. |
| `structure_formatter.py` | Structure-table primitives for internal data communication. |
| `timer.py` | Timestamp/elapsed formatting, `Timer`, `PhaseProfiler`. |
| `units.py` | Unit conversion constants and helpers. |

```text
lattice              -> numpy
geometric            -> numpy
structure_formatter  -> numpy
opt_formatter        -> (nothing)
bonds                -> numpy, .lattice, ..schema.reference
cv                   -> numpy, .geometric
print_results        -> numpy, .units, .structure_formatter, .opt_formatter
io                   -> numpy, ..schema.reference
geodesic             -> numpy, ..schema.reference, .lattice
other modules        -> unchanged
```

Rules:

1. `lattice`, `geometric`, `structure_formatter` and `opt_formatter` import no other utilities module.
2. `bonds` imports `lattice` and `schema.reference` only; `cv` imports `geometric` only.
3. `print_results` imports `structure_formatter`, `opt_formatter` and `units`; `opt_formatter` never imports `print_results`.
4. The only permitted package-level upward edge is `X -> pyrai3md.schema.reference`, and that file must stay import-free.
5. No backend-specific code lives in utilities (`ARCHITECTURE.md` section 3.6).
6. `schema/reference.py` is import-free; the public unit constants are
   re-exported by `schema/__init__.py` directly from `utilities.units`.

`print_results` still has two lazy imports that predate this refactor: `data.center.aggregate_system_multiplicity` for the region charge/spin block, and `workflow.records.deviation.DEVIATION_KEYS` for the prediction-error section. They are recorded in section 13 as deferred extractions rather than silently claimed to be gone.

## 4. Local geometry, centers and CVs

- `geometric.py` owns `bond_core`, `angle_core`, `torsion_core`, their Hessian cores, `cv_hessian_block`, the dense adapters `BND`/`AGL`/`DHD`, and `_scatter`. Angles are radians. Degenerate geometry raises `CVSingularError`; no path returns nan.
- `_as_coords` is the shared input normalizer. It raises `CVSingularError` for a shape other than `(natom, 3)` and is imported privately by `cv`.
- `_acos` is the one checked arccos. `cv.cell_angle` uses it; `cv` no longer defines its own.
- `group_center(coords, indices, masses=None)` moved from `cv` verbatim. Its weighted sum is divided once, and `weights` is the derivative of the center, normalized to one. The bit-stability requirement is measured: the new function returns the same center and weights as the old one (`array_equal`, not `allclose`), and `operations.velocity_correction` reads it.
- `cv.distance/angle/dihedral` return compact Jacobians in the atom order the variable names; `cell_length/cell_angle/cell_volume/strain_components/strain` are the cell CVs. `cv.cell_volume` raises `CVSingularError` on a singular cell, while `lattice.cell_volume` returns `0.0`; those are different questions, not a mode.
- `cv.evaluate_on_images(expanded, var, kind, grad=False)` evaluates `BND`/`AGL`/`DHD` on unfolded coordinates. Kinds are `'bond'`, `'angle'`, `'dihedral'`; an unknown kind or a wrong index count raises builtin `ValueError`. It keeps the dense cores rather than re-deriving through the compact front doors, because duplicated indices behave differently there.
- `cv` re-exports the geometric constants and cores the constraint layer already reads; `cv.torsion_core is geometric.torsion_core` remains an AST-pinned identity.

## 5. Cell math

- `lattice.py` owns six-parameter/matrix conversion, volume, boundary classification, pbc normalization, Cartesian/fractional conversion, wrap/unwrap/continuous views, the certified nearest image, `half_box_bound`, and the stress/lattice conjugates.
- `normalize_pbc(None)` means non-periodic (`[0, 0, 0]`). `bonds.periodic_flags(None)` means fully periodic (`[1, 1, 1]`). These are two entry contracts, each with one implementation.
- `min_image_displacement(displacements, lattice, pbc=(1, 1, 1))` searches the candidate images around the component-wise value and certifies the result. `pbc` is an explicit tuple; `None` is refused because omitting the argument is the all-periodic request. A non-periodic axis is never shifted, is excluded from the candidate set, and contributes no reach to the certificate, so it cannot widen the shell or force a refusal.
- The shell starts with the zero shift first. An exact half-cell tie therefore keeps the zero shift; a widened shell also puts zero first. This is a deliberate rule and is pinned by `test_lattice`.
- On a cell that is not size-reduced, the certificate raises `ValueError` naming "size-reduced" and "certifiable"; it does not return a longer, non-shortest image.
- `half_box_bound(lattice)` is half of the shortest interplanar spacing. Its result carries the input length unit; the docstring says so because most callers pass Angstrom but the formula is unit-agnostic.

## 6. Topology and images

- `schema.reference.BOND_LENGTHS` is data: `'A-B'` keys, Angstrom values, the heavy-atom entries at their single-bond lengths, and the D-28 rationale in the comment. `bond_lib` is policy: an exact key lookup, and `1.2` Angstrom for anything else. The default stays in the function because `bonds._listed_pairs` uses it as the "not listed" sentinel.
- `perceive_bonds(atoms, coordinates, lattice_vector, pbc=None, scale=BOND_SCALE)` returns `(i, j, delta)` rows. With no cell it still returns rows, with zero deltas; that is the topology answer for a molecule. With a cell it converts once (a singular conversion returns `[]`), then for each atom pair asks `lattice.min_image_displacement` for the nearest image and recovers `delta = rint((reduced - difference) @ inverse_lattice)`. The certificate's refusal propagates.
- `perceive_bonds_focus` is the same rule scanned outward from the focal atoms, with the lower index first in every row. It returns `[]` when there is no cell.
- `bond_graph` returns `{'bonds', 'atoms', 'components'}`. It refuses an element pair the table does not list when that pair sits inside the default cutoff, and its message now points at `schema/reference.py::BOND_LENGTHS`. Its components come from the same `component_offsets` traversal as unfolding. A singular cell raises `ValueError` from the nearest-image search instead of `LinAlgError` from an inline inverse.
- `component_offsets(neighbours, natom, flags)` is the one traversal: roots are lowest atom indices, breadth-first order follows the adjacency, and it returns `(offsets, components, fragments)`. `derive_initial_image` adds the per-fragment rounded-mean shift and then measures every perceived bond against `half_box_bound`; `unfold_components` uses the offsets directly; `bond_graph` takes the components.
- `unfold_components(coordinates, lattice, bonds, pbc=None)` returns `(expanded, offsets, components)` with `expanded = coordinates + offsets @ lattice`. The no-cell path is the identity, to the last bit. `ComponentError` is raised for a bond index outside the structure.
- `derive_initial_image` returns `[]`, with one warning where a warning is the contract, for an unreadable cell, a non-periodic request, no perceived bonds, an oversized system, or counts that cannot satisfy every bond. The water box's 40 O-H crossings and the torn-water control remain the measured fixtures.

## 7. Structure formatting

`structure_formatter.py` is the single home of the table text that the framework passes between its own components:

- `print_coord(atoms, vectors, indent='  ')`: the run log uses the default two spaces; a trajectory frame passes `indent=''`. Both are byte contracts.
- `print_charges(charges)`: four numeric columns with the two-space lead the trajectory files use.
- `print_matrix`, `print_cell`, `print_pbc`, `print_turbocoord`: the matrix, cell, PBC and TurboMole layouts. They have no internal caller today beyond the package re-export, but they are the only copy of those layouts and stay public.
- `io.py` owns no printer. `coordinates.py` is gone. The three live consumers are `print_results` (run-log sections), `kernel/logger` (input echo) and `workflow/records/output` (trajectory frames); `output._print_coord` now delegates with `indent=''`.

## 8. Result and record formatting

- `print_results.format_result_block(system, results, step=None, runtime=None, thresholds=None, verbose=2, centers=None, dynamics=None)` is unchanged in responsibility. Verbose levels are cumulative: 0 progress/energy, 1 structure plus `&cell`, 2 property arrays, 3 the Hessian and every missing marker. The input-header echo is not gated.
- `print_results` also owns `opt_status.txt`: `WRITER_ATOM/CELL/FOREIGN`, `OptStatusRecord`, `read_status`, `classify_header`, `assert_writable` and `format_malformed`. The record is a namedtuple; malformed lines are counted, not dropped; a foreign header is refused before a write.
- `opt_formatter.py` owns only `STATUS_ATOM_COLUMNS`, `STATUS_CELL_COLUMNS`, `STATUS_STEP_WIDTH`, `STATUS_FIELD_WIDTH`, `status_header(cell=False, terms=(), method=())` and `status_line(step, values, extra=0, method=0)`. The column order is atom + cell + method + terms; the fixed prefix is what `classify_header` reads; a row whose width disagrees with its declaration raises `ValueError`.
- The artifact name `opt_status.txt` is unchanged. The write and read halves were split by responsibility, not by file name.

## 9. Input parsing

`io.py` keeps `string2float`, `reverse_string2float`, `complex2string`, `string2complex`, `verify_xyz`, `read_coord`, `read_xyz_frame`, `read_lattice_comment`, `read_charge`, `read_initcond`, `read_float_from_text`, `lookup_amber`, `atomic_number`, `numerize_xyz` and `mark_atom`. It imports `schema.reference` for the periodic table and `ELEMENT_SYMBOLS`. It owns no output layout; sections 7 and 8 point to the two homes that do.

## 10. Call-site and public-API map

The geometry/CV moves:

- `schema/normalize.py` imports `bonds` instead of `image_flags`.
- `workflow/constraints/images.py` imports `bonds`, `geometric` and `lattice`; it calls `bonds.perceive_bonds*`, `bonds.unfold_components`, `lattice.half_box_bound` and `geometric.group_center`.
- `workflow/constraints/wall.py` and `workflow/integrator/operations.py` call `geometric.group_center`.
- `workflow/records/diagnostics.py` takes `BOND_SCALE` from `bonds` and `half_box_bound` from `lattice`.
- `constraints/state_adapter.py` keeps calling the constraint-side `images.require_same_component`; that refusal was not moved or deleted.

The formatting/record moves:

- `kernel/logger.py` and `workflow/records/output.py` import `structure_formatter`.
- `workflow/optimize.py` takes `status_header`/`status_line` from `opt_formatter` and `assert_writable` from `print_results`.
- `qualification/runner.py` and `tools/verify.py` take `STATUS_*` from `opt_formatter` and `read_status`/`WRITER_FOREIGN`/`format_malformed` from `print_results`.
- `test_opt_status_record.py` imports the two halves separately; `test_workflow.py` and `test_cell_optimization_check.py` follow.

Public API consequences:

- `print_charge` is gone; `print_charges` is exported.
- `print_coord` now resolves to the indented structure formatter (the old package-level name resolved to an unused, unindented copy in `io`).
- `utilities.image_flags`, `utilities.coordinates` and `utilities.opt_status` module paths are gone; `utilities.bonds` now means the topology service.
- `min_image_displacement` has one new optional argument; all existing calls keep their old behavior.
- `geometric.evaluate_on_images`, `geometric.expand_components`, `geometric.require_same_component`, `geometric.ComponentError`, `cv.group_center`, `cv._acos` and `cv._as_coords` are gone under those names.

## 11. Verification gates

The refactor was accepted on these readings, measured in the working tree:

- **Byte equality.** The six structure primitives on a fixed fixture are byte-identical to their pre-refactor outputs; `print_coord(..., indent='')` is byte-identical to the old `records/output._print_coord`. `status_header`/`status_line` are byte-identical for atom, cell, terms, method and cell+terms+method combinations; `read_status` returns the same record fields on the atom, cell and engine fixtures. One `format_result_block` fixture at verbose 0/1/2/3 is unchanged; the `test_md_output` and `test_charges` suites pass.
- **Data equivalence.** `bond_lib` returns the same value for 484 element pairs (both orders plus unknown symbols). `group_center` returns the same center and weights as the old implementation (`array_equal`). The water box keeps its 40 O-H crossings; the torn-water unfolding offsets and the bond-graph components are unchanged; the exact half-cell tie chooses the zero shift.
- **Refusals.** A non-size-reduced cell raises `ValueError` naming "size-reduced" and "certifiable" from `min_image_displacement`; `perceive_bonds` returns `[]` on a singular cell; `derive_initial_image` returns `[]` with one warning for a non-reduced cell; `bond_graph` propagates the certificate refusal.
- **Structure.** A fresh process imports all 17 non-init utilities modules and `pyrai3md`. `schema/reference.py` has no import statement. `geometric` imports numpy only; `lattice` imports numpy only; `bonds` imports numpy, `.lattice` and `..schema.reference`; `cv` imports numpy and `.geometric`; `structure_formatter` imports numpy; `opt_formatter` imports nothing; `print_results` does not import `bonds` or `cv`.
- **Grep.** No `image_flags`, `utilities.coordinates`, `utilities.opt_status`, `utilities_refactor`, `geometric.evaluate_on_images`, `geometric.require_same_component`, `geometric.expand_components`, `geometric.ComponentError`, `cv.group_center` or `cv._acos` remains in the package or `tools/` (the one hit for `image_flags` is a LAMMPS test method name, not an import).
- **Tests.** The affected set ran as **852 tests in 63.921 s, OK** (`.venv`, Python 3.11). It covers the five geometry suites, the renamed bond suite, the new structure-formatter suite, the optimizer status/record suites, the workflow/kernel/output suites, the constraint suites and the path suite. The full gate then ran in two pytest groups: `qualification/verification` **3483 passed / 1 skipped / 3933 subtests in 807.75 s**, and `skills` **1506 passed / 18 subtests in 121.28 s**; total **4989 passed / 1 skipped / 0 failed / 3951 subtests**. The one skip is the known `test_fssh.py::test_numba_request_falls_back_with_warning` case, which is skipped because numba is installed.

## 12. Migration record

- **`image_flags.py` retired.** Its topology half became `bonds.py`; `half_box_bound` became a lattice function; `ComponentError` moved with the unfolding code; `unfold_components` replaced `expand_components`; `periodic_flags` replaced the private `_normalize_flags`; `_candidate_shifts`, `_SHIFT_RANGE` and `_to_fractional` were deleted with the 27-shift search. The module had no compatibility shim.
- **`coordinates.py` retired.** `print_coord` and `print_charges` moved to `structure_formatter.py`; `print_coord` gained the explicit indent. `io.py`'s dead `print_coord`/`print_charge` copies and the four unused layout functions were removed from io (the layouts themselves moved); `io.py` is input-only.
- **`opt_status.py` retired.** Its write format became `opt_formatter.py`; its record, reader, classifier and write guard became part of `print_results.py`. The artifact name stayed `opt_status.txt`.
- **Deliberate behavior changes.**
  - An exact half-cell tie in the nearest-image search keeps the zero shift. Distances are unchanged; the returned `delta` can differ from the old 27-shift choice.
  - Perception on a cell the certificate cannot resolve now refuses by `ValueError`; the configuration path warns and returns no counts.
  - `perceive_bonds` on a singular cell still returns `[]`; `bond_graph` now raises `ValueError` from the conversion instead of `LinAlgError` from an inline inverse.
  - `cv.evaluate_on_images` refuses an unknown kind or wrong arity with builtin `ValueError` (the old `ComponentError` was also a `ValueError`, so callers catching `ValueError` are unaffected; the test that pinned the subclass was updated).
  - `'torsion'` is no longer a CV dispatch spelling; `'dihedral'` is the one word.
  - `print_charge` is no longer a package export.
- **No shims.** Every deleted name was removed from `utilities/__init__.py` and every call site in the same change.

## 13. Deferred items

- `_ELEMENT_SYMBOLS` in `bonds.py` is still an explicit element list. `_listed_pairs` could derive it from the keys of `BOND_LENGTHS`, but that changes the probe's byte-for-byte input set; it is a separate cleanup.
- `print_results` still has the two pre-existing lazy upward imports (`data.center.aggregate_system_multiplicity`, `workflow.records.deviation.DEVIATION_KEYS`). Extracting them into a neutral leaf is a later change, not a hidden claim of this refactor.
- The four public structure layouts with no internal caller (`print_matrix`, `print_cell`, `print_pbc`, `print_turbocoord`) are kept because they are the only copies of those layouts. Whether to drop them from the package API is a separate decision.
- `print_results` now holds the status record lifecycle and is the longest utilities module. If that ages badly, the natural extraction is a `status_record.py` for the read side; no such module is created now.
- No timing pair was recorded for the certified perception search. The refactor does not claim a speedup; a measurement on the same fixture before and after would be the way to price it.
- `geodesic.py` remains outside the geometry refactor: it is a port with its own solver and alignment rules, and it is not merged into `geometric`.