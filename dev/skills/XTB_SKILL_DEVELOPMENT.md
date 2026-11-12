# xTB Skill Development Guide

Last updated: 2026-10-22 23:30（§1 的拓扑沿用改为按 `keep_topology` 的条件语义并给出实测句；§8 的 `JOBTYPE_CAPABILITY_MAP` 改为实际的 `_TOKEN_CAPABILITY_MAP`；`optimize_cell` 那半的出处从 `workflow/optimize.py:196-223`（那是单位换算辅助）改到 `workflow/hessian.py:749-775` 与被拒配对所在的 `workflow/optimize.py:309-345`；§6 的两个计数改为盘上实数：9 个 YAML 用例（含 `gfn_ff_stress`）与 4 个无需 xTB 安装的单元测试模块（补 `test_keep_topology.py`，出处 #378）；其余章节本轮未重读）。

此前 2026-10-19 12:10（新增 §8：引擎侧 Hessian（`--ohess`）的设计与输入骨架——本轮只写请求形状与真产物要答的四件事，解析器、profile、`resolve_profile` 分派与任何数值一律不做（仓库内无 xtb Hessian 样本，本机无 xtb，#362 的规矩）。同轮落地的是拒绝：`data/core.py::MATRIX_CAPABILITIES` 让 `hess_driver: skill` 在声明里没有矩阵能力时建状态前具名拒绝（xtb 声明 energy/gradient/stress，判据 `test_core.py::TestCurvatureProducerScope`）。§3.2 的周期边界规则与代码逐条核对并补实测；其余段落沿用 2026-10-05 的核对）

This document covers **xTB-specific** implementation details for xtb_skill.
For the abstract skill development specification (directory structure, schema conventions,
backend architecture, profile design, etc.), see [SKILL_APP_DEVELOPMENT.md](../SKILL_APP_DEVELOPMENT.md).

---

## 1. xTB-Specific Design Considerations

### 1.1 Ground-State Single-Reference Program

xTB is an external command-line program. A single point calculation produces exactly
one ground-state energy and gradient — there are no excited states, NAC, or SOC.
Consequences in the skill design:

- `energy` is always a list of length 1; `gradient` has shape `(1, natom, 3)`.
- `normalize.py` enforces `nstate = 1`, `target_states = [0]`, `state_spin = [multiplicity - 1]`,
  and clears any NAC/SOC pair maps.
- No orbital management: there is no orbital-guess or orbital-file protection logic.
  The fixed calcdir per backend instance serves a different purpose — GFN-FF topology
  reuse, and that reuse is **conditional**: the backend deletes a leftover `gfnff_topo`
  before every evaluation unless the framework sets `keep_topology`
  (`backend.py:227-240`; `schema/config.py:186-191` - the framework sets it only for a
  dynamics run). The measurement behind the switch: carrying a clashed run's
  `gfnff_topo` onto a sane geometry moved its energy by +0.3508098 Hartree, and round
  `tmp_20261017_234656`'s variable-cell rows 6-21 read +0.20 while those same frames
  evaluate fresh to -0.929 (`tests/test_keep_topology.py`, #378).

### 1.2 Method to CLI Flag Mapping

| Method | CLI Flag | Roles | PBC Support |
|---|---|---|---|
| `gfn2` | `--gfn 2` | qm, qm2 | No |
| `gfn1` | `--gfn 1` | qm, qm2 | No |
| `gfn0` | `--gfn 0` | qm, qm2 | No |
| `gfn_ff` | `--gfnff` | mm | Yes |

Command assembly in `service/builder/input_builder.py::build_xtb_args()`:

```
xtb <cli_flag> --grad [--chrg N] [--uhf M] -I <project>.inp <geometry_file>
```

- `--grad` is **always appended** (every profile, including `*_energy`):
  the output parser reads the total energy from the `engrad` file
  unconditionally, and xTB only writes that file when `--grad` is
  requested, so an energy-only run without the flag would fail at
  parse time.
- `--chrg N` only when the total charge is nonzero; `--uhf M` with `M = multiplicity - 1`.
- Parallelization is controlled via `OMP_NUM_THREADS` in the run script template
  (consistent with legacy `qc_xtb.py`); `--parallel` is intentionally not used.

### 1.3 Geometry File Formats

- **Molecular systems**: standard XYZ file `<project>.xyz` (Angstrom).
- **Periodic systems** (gfn_ff with `cell`/`pbc` set): Turbomole coord format
  `<project>.coord` with `$coord` (Bohr), `$periodic <ndim>`, and
  `$lattice angstrom` blocks. Lattice vectors are filtered by the nonzero `pbc` flags.
  This logic is implemented natively in `write_coord_file()` (the shared
  `utilities/structure_formatter.py::print_turbocoord` layout, no legacy dependency).

Periodicity detection uses the `pbc` flags (`any(int(x) != 0)`), **not** the `cell`
matrix, because the schema `cell` default is non-empty (99 Å box). This matches
the framework-wide rule that `pbc` is the single periodicity authority
(CHANGELOG #216, `dev/methods/PBC_DEVELOPMENT.md` §3.2). The per-step
`lattice_vector`/`pbc` reach the coord file through the `calculate(cell=...)`
snapshot applied to `config.system`, so a variable-cell step writes its own
deformed lattice (only GFN-FF declares `supports_pbc`; other methods reject
`pbc` in `schema/validate.py`).

### 1.4 User Input Template Merging

If a file named `<project>.inp` exists in the current working directory, its content is
read first and used as the base of the generated input; skill-generated `$embedding`
blocks are appended after it. This mirrors the legacy template mechanism and lets users
inject advanced xTB settings (e.g. `--opt` auxiliary groups) without schema changes.

---

## 2. Calculation Profiles

### 2.1 Complete Profile List (8 profiles)

| Profile | Capability | Steps | xTB CLI |
|---|---|---|---|
| `gfn2_energy` | energy | energy | `xtb --gfn 2 --grad` |
| `gfn2_gradient` | gradient | energy, gradient | `xtb --gfn 2 --grad` |
| `gfn1_energy` | energy | energy | `xtb --gfn 1 --grad` |
| `gfn1_gradient` | gradient | energy, gradient | `xtb --gfn 1 --grad` |
| `gfn0_energy` | energy | energy | `xtb --gfn 0 --grad` |
| `gfn0_gradient` | gradient | energy, gradient | `xtb --gfn 0 --grad` |
| `gfn_ff_energy` | energy | energy | `xtb --gfnff --grad` |
| `gfn_ff_gradient` | gradient | energy, gradient | `xtb --gfnff --grad` |

### 2.2 Profile Task Extraction

`gfn_ff` contains an underscore, so profile task extraction must use
`profile.rsplit('_', 1)` (e.g. `gfn_ff_energy` -> `gfn_ff` + `energy`),
not `split('_', 1)`.

---

## 3. Point Charge Embedding and PBC Rules

### 3.1 Point Charge Embedding (qm/qm2 scenarios)

Background point charges `[charge, x, y, z]` (Angstrom) are written to `<project>.pc`:

```
<number of point charges>
<charge> <x> <y> <z>      # coordinates converted to Bohr
...
```

and referenced from the input file via:

```
$embedding
input=<project>.pc
$end
```

### 3.2 Mutual Exclusion Rules (enforced in schema/validate.py)

- `point_charges` and `cell`/`pbc` are mutually exclusive — xTB cannot combine
  embedding with periodic boundary conditions.
- Non-empty `pbc` requires `method == gfn_ff` — only GFN-FF supports PBC. The
  capability lives in one predicate, `method_supports_pbc()`
  (`schema/method.py`), which both `validate_system()` and `write_input_file()`
  consult; the writer needs it because the schema can only see a `pbc` written
  on the skill, while a framework run carries the cell on the system and hands
  it over per evaluation. Measured against xtb 6.7.1 in the test environment host for
  the same ethylene box in a 10 Å cell: `--gfn 2` stops in its own SCF
  (`scf: Multipoles not available with PBC`), `--gfn 1 --grad` completes and
  writes `gradlatt`, but the lattice-gradient rows are nine exact zeros
  (`|dE/dlatt| =  0.000000`), and only GFN-FF writes a nonzero block
  (`|dE/dlatt| =  0.008680`). A periodic variable-cell run therefore has one
  producer, and the parser will not read an unwritten gradient as a zero-stress
  box (`service/parser/output_parser.py`, CHANGELOG #356).

---

## 4. Output Parsing

### 4.1 engrad File

Energy and gradient are parsed from `<project>.engrad`:

```
The current total energy in Eh
                              <energy>        <- title line + 2

The current gradient in Eh/bohr
                              <g1x>           <- natom * 3 values follow
                              <g1y>
                              ...
```

- The value sits two lines below the section title line.
- Gradient values are collected into a `(natom, 3)` array and wrapped to `(1, natom, 3)`.
- A missing `engrad` file raises `RuntimeError` pointing to `<project>.out` for diagnostics.

### 4.2 Atomic Charges

- GFN-x methods: read from the `charges` file in calcdir.
- GFN-FF: read from the `gfnff_charges` file.
- Exposed through the framework **`computed_charges`** channel with
  shape `(1, natom)` (state-resolved for the single ground state),
  not `atom_charges`.
- When xTB wrote no charge file the key is **omitted** from the result
  dictionary (matching the empty-default "not computed" schema contract).

### 4.3 Units

All results stay in atomic units: energy in Hartree, gradient in Hartree/Bohr
(declared in `manifest.yaml` units section).

---

## 5. Environment Resolution

### 5.1 xtb_path Semantics

`xtb_path` follows the legacy convention with a three-level fallback
(`service/runner/env_check.py::validate_environment` and
`service/templates/run_templates.py::_resolve_xtb_binary`):

1. Empty: search `xtb` in PATH (`shutil.which`).
2. Directory: use `$xtb_path/bin/xtb`.
3. Otherwise: treat as a direct binary path.

### 5.2 Run Script Environment

Local and slurm run script templates export:

```
XTB_PROJECT             # project name
XTBHOME                 # xTB installation root, derived from
                        # xtb_path (omitted for a bare PATH-resolved
                        # 'xtb' command)
XTBPATH                 # $XTBHOME/share/xtb parameter/data directory
                        # (legacy runscript contract)
OMP_NUM_THREADS         # = thread (the framework forwards the same
                        # count as xtb_thread; the larger of a pair
                        # wins, and the old xtb_nproc counted threads
                        # under a process name -- retired, since xtb has
                        # no MPI arm, #306)
OMP_STACKSIZE           # = xtb_mem MB (KMP_STACKSIZE mirrored)
OMP_MAX_ACTIVE_LEVELS=1
```

The skill-level shell commands (config `shell` key, module loads from
`skills.xtb_skill.shell` of pmd_env.yaml) are copied ahead of the
exports when configured; the global `environment.shell` is reserved
for the main framework (#134).

and `cd` into the calcdir before executing the xTB command, with stdout redirected
to `<project>.out`.

---

## 6. Test Cases

`xtb_skill/tests/` ships 9 YAML cases (gfn0/gfn1/gfn2/gfn_ff × energy/gradient,
plus the force field's stress case):

`gfn0_energy`, `gfn0_gradient`, `gfn1_energy`, `gfn1_gradient`,
`gfn2_energy`, `gfn2_gradient`, `gfn_ff_energy`, `gfn_ff_gradient`,
`gfn_ff_stress`.

In addition, four unit test modules run without an xTB installation:

- `test_input_builder.py`: command-line assembly (`build_xtb_args`) and
  input file generation.
- `test_output_parser.py`: engrad-based energy/gradient extraction and
  the `computed_charges` channel contract ((1, natom), omitted when
  xTB wrote no charge file).
- `test_run_script.py`: run script environment exports and command.
- `test_keep_topology.py`: the GFN-FF topology-reuse switch (#378).

The `fin_coord` / `fin_velocity` channels are reserved for contract
uniformity: the xtb skill performs no skill-internal iteration, so
`fin_coord` stays empty (no internal iteration) and `fin_velocity` is
always returned empty (no velocity output).

---

## 7. Known Limitations

- No hessian/frequency, NAC, SOC, or geometry optimization profiles (outside both
  the current requirement scope and the skill's ground-state single-reference design).
- Method-role compatibility (gfn_ff ↔ mm, others ↔ qm/qm2) is declared in the manifest
  and `schema/method.py::METHOD_ROLE_MAP` and documented, but not enforced at runtime
  when the framework assigns roles by theory order (consistent with existing skills).

---

## 8. Design: An Engine-Side Hessian (`--ohess`), Not Implemented This Round

**Ruled 2026-10-20: this is not an owed variable-cell item.** A formal skill
backend cannot supply the lattice rows a variable-cell TS needs; the framework's
strained batch is the supported route (`optimize_cell_ts`, `hess_driver: main`).
The design below stays as a future note for the atomic half, not as a blocker
for the cell search.

Section 7's first line is still the fact on disk: `xtb_skill` declares
`capabilities: [energy, gradient, stress]` and offers no force constants. What
this section fixes is the *shape* of the profile that would change that, and what
it deliberately leaves unwritten.

**Why it is design-only.** No output of `xtb --ohess` exists in this repository
(`find` over the tree returns no `hessian`/`*.hess` sample) and xtb is not
installed on this environment, so a parser written now would be a parser written against
a guess. #362 set the rule on the CP2K reader: capture a real product first, then
write what reads it. The input side is different - a command line is a request,
not a claim about an answer - so the request is specified here and the answer is
owed to the deployment.

**The manifest chain to add, in the order the other skills spell it.**

| Layer | What a Hessian profile needs | Today |
| --- | --- | --- |
| `manifest.yaml:capabilities` | `vibrational_frequency`, the word the ORCA and MOLCAS manifests already use (`orca_skill/manifest.yaml`, `molcas_skill/manifest.yaml`) | absent |
| `manifest.yaml:profiles` | `gfn2_hessian` / `gfn_ff_hessian`, `capability: vibrational_frequency`, steps `[energy, gradient, hessian]` | absent |
| `schema/method.py` | the method-to-profile rows for those names | absent |
| `query.resolve_profile` | jobtype token `hessian`/`frequency` → the new profile | falls through |
| `planning/resolver.py` | nothing: `_TOKEN_CAPABILITY_MAP` (`planning/resolver.py:19-24`) already routes `frequency` to the `hessian` capability | in force |

The capability *word* is a ruling, not an accident: `data/core.py` now accepts any
of `MATRIX_CAPABILITIES = ('hessian', 'frequency', 'vibrational_frequency')` when
it decides whether a skill can answer `hess_driver: skill`, and the two
ground-state-adjacent spellings already divide between skills. A periodic engine
whose matrix is a vibrational analysis - xtb and CP2K both iterate to it - takes
`vibrational_frequency`, the majority spelling of the pair that already has it.

**What the run asks.** One command per structure, the box a variable:

```
xtb <project>.xyz --gfn <2|ff> --ohess [extra flags]
```

The four things the first real product has to answer before a line of parser is
written, each of which the design currently only names:

1. the file the matrix lands in, and whether `--ohess` writes it next to the
   input or into the run directory;
2. its units and its shape (3N x 3N, and whether xtb prints it mass-weighted -
   the framework's own numerical batch returns the unmixed Cartesian matrix, and a
   mass-weighted row silently changes every frequency);
3. which line the frequencies themselves follow, since the `&frequency` jobtype
   reports them and the framework does not re-derive them;
4. whether GFN-FF is accepted at all - the stress route that variable-cell runs
   use today is GFN-FF's, and if `--ohess` refuses it, the profile is
   `gfn2_hessian` only and `optimize_cell` + engine Hessian has no combination on
   this engine.

**The `optimize_cell` half, which is already built.** A variable-cell run does not
need the engine for its lattice rows: `workflow/hessian.py` strains the box twelve
times and joins those differences to the atomic block
(`workflow/hessian.py:749-775` `_strain_cell_block_onto` lays the six strained rows beside
the engine's atomic block, and the pairing it refuses is named at
`workflow/optimize.py:309-345`, #365). So the
engine profile buys the *atomic* half, and the pairing that is refused today -
`hess_driver: skill` on an excited state - stays refused for the same reason
after the profile lands: an xtb matrix is its ground state's.

**What is not done this round, stated so it is not read as done:** no
`vibrational_frequency` capability, no profile entry, no `resolve_profile`
branch, no parser, no number. `qualification/verification/test_core.py::
TestCurvatureProducerScope::test_a_backend_without_force_constants_is_refused_by_name`
is the criterion that keeps the gap honest: it names xtb, cp2k and lammps as
skills that must be refused *before* an evaluation is paid for, and it will start
failing for xtb the day the capability is added - at which point the profile, the
parser and this section's four answers arrive together.
