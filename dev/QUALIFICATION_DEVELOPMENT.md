# PyRAI3MD Qualification Framework Design

Last updated: 2026-11-12 21:05 (7.1.1 records truthful verdicts, zero-case errors, all-skipped rows, the shared pytest phase and the 153/78 plan coverage. Earlier: 2026-11-12 19:20 (7.1.1 records truthful verdicts, zero-case errors, the shared pytest phase and the 153/78 plan coverage. Earlier: 2026-11-12 18:10 (the shared `skills/tests/` pytest group now follows the aggregate `skills.<name>.runtime_tests` phase instead of running unconditionally; `_runall: true` opens the phase. Earlier: 2026-11-11 17:50 (the grouped-round in-flight cap is now `environment.max_in_flight`, default `0` = submit all; section 3's scheduler bullet and the environment key list name it. The rest was not re-audited.)

Before it, 2026-11-06 13:10. The adaptive band landed here: `examples.adaptive_test` joins `manager.EXAMPLE_FAMILIES` as the sixth family (measured over the 201 shipped YAMLs: `md_lammps_` 12, `constraint_md_` 33, `optimize_` 25, `path_` 32, `multiscale_` 37, `adaptive_` 6, no stem matching two prefixes - the pairwise-disjointness and row-order pins are `integration/test_integration_manager_planning.py::test_optimize_prefix_shadows_no_other_family`, which enumerates the table, and the new family's own three-valued decision plus its six stems are `verification/test_manager.py::TestAdaptiveFamilySwitch`); the section 4.2 posture was re-measured rather than inherited (**610 rows planned - 111 main + 248 example + 226 skill + 25 benchmark - 384 closed by switches, 230 printed**, 47 case keys `true` against 154 `false`); section 6.3 gained the adaptive converter paragraph (`control.reference`, `&ADAPTIVE` via `_build_adaptive_pairs`, `highref`/`midref` in `&MOLECULE`); section 6.6 gained the two-source fixture staging (`.json`/`.charge`/`.xyz`/`.freq.molden`, the bare-name-in-text rule, `modeldir:` copied as a directory) and the `_check_adaptive_products` verdict; and the `tools/verify.py` enumeration now runs through J, with the round-`tmp_20261106_102745` reading (`unexpected_aux 2` on the pre-fix ONIOM artifacts, 0 on the post-fix local rerun). Three names this file pointed at were corrected against where they actually live: `TestPathFamilySwitch` and `test_optimize_prefix_shadows_no_other_family` are in `qualification/integration/test_integration_manager_planning.py`, not `verification/test_manager.py`. Both suite-size readings in section 6.5 were re-measured on this checkout: verification **3332 OK (skipped=1) in 68 s**, integration **123 passed with 7218 subtests in 1578 s** under `-m integration` - the second reading taken after two integration criteria that the new examples legitimately broke were corrected (the shipped-ensemble check compared every frame against a hardcoded six atoms, so the 1581-atom adaptive box failed it; the dispatch-pool membership list enumerates the sections that state a trio, and the three `adaptive_*_slurm` carriers state one in `&ADAPTIVE`). Still owed and not touched here: the "one engine certification" bullet cites `test_manager.py::test_the_shipped_stage_opens_the_three_engines_and_the_code_phase_rest`, which does not exist in this repository, and the section 4.2 bullet's older path-round readings are quoted as history rather than re-measured. Earlier: 2026-11-02 22:42 (#450: section 5.1's suite-size row re-measured to 3205 OK, and the two multiscale example counts this round had to move are named (27 in `test_manager`, 41 in `test_integration_multicenter_converter`, both from #449's new example). Before it, 2026-11-02 11:50 (#448: section 5.1's suite-size row was re-measured rather than inherited - verification 3196 OK (skipped=1), integration 123 OK (skipped=4), skill suites 1567 passed / 2 skipped / 18 subtests on the repository `.venv`, plus the reading that the same tree gives a different nn count on the TensorFlow-less interpreter. The converter paragraph (section 6.5) still states the #447 rule and stays correct: the declaration is rendered into `coupling` and nothing is invented for a case that declares none; #448 moved the same rule onto the receiving side, which is documented in SKILL_TEST_DEVELOPMENT.md 4.5/4.6. The rest was not re-audited. Before it, 2026-10-27 - the main-test section 6.5 now separates the verification unit suite from the new `qualification/integration/` package and records the `test_integration` switch and the measured 4495-unit / 123-integration split; the rest was not re-audited. Earlier, 2026-10-22 22:00 - one test name in the verification table corrected to the current `test_manager.py::test_the_shipped_stage_opens_the_three_engines_and_the_code_phase_rest`; nothing else was re-read. Earlier, 17:10 - round `tmp_20261022_084519` collected: 289 rows / 286 pass / 0 fail / 213 closed, all 31 `path_*` examples green in both carriers (62 rows), and `tools/verify.py` section D re-read the 10 variable-cell path legs against their own frames (550 rows, 0 disagreements). The review found two defects, both fixed and both needing the next round to certify: `irc_direction: both` walked its second branch from the first one's endpoint (F now pins the two saddle rows), and `path_irc_pbc` was a hollow path (the fixture's mode climbs on gfn-FF; it now walks the descending side). Earlier, 2026-10-21 23:48 - the path coverage audit is closed: ten examples added for the combinations that had none, the path round's post-checks extended (D now re-reads a path leg's cell record and cross-checks its volume against the trajectory's `Lattice=` frames, which found a variable-cell band writing the parent box into every frame; F now reads the frozen atoms), and the numbers in section 4.2 re-measured for the thirty-one-case posture (498 rows / 213 closed / 288 printed). Before it, 21:55 - the path round is collected: round `tmp_20261021_195813` printed 269 rows (266 pass / 0 fail / 0 error / 213 closed) with the twenty-one `path_*` examples green in both carriers, the four switch families and their counts were corrected in section 4.2 (four families, not three; the path family's twenty-one cases), the shipped posture and its plan reading were re-measured for this file (478 rows / 213 closed / 268 printed), and section 2.1 now enumerates `tools/verify.py`'s seven sections including the new F. Before it, 2026-10-20 07:50 - the `_timeout` section now records that `_example_timeout` and `manager._resolve_example_timeout` were deleted by ruling on 2026-10-20, leaving `_timeout` as the only budget; the rest was not re-audited. Before it, 2026-10-19 19:52 - #395 put the three dispatch keywords on the classic-input writer, so the cp2k strain passage now carries both carriers' readings: round 19 measured 589 s (yaml, pool 2) against an 821 s serial baseline for `optimize_cell_cp2k` while its `.input` leg sat at 815 s against 817 s because the carrier stated no pool, and the fix shows byte-identical for 92 of the 96 generated classic inputs with 4 changed; the six lines those two cp2k carriers give up were checked to be either duplicates `&MOLECULE` owns or already absent on the classic side before the fix. Before it, 16:10 - #393 collected round 18 (`tmp_20261019_122659`: 251 rows / 244 pass / 4 fail / 0 error, `tools/verify.py` clean over the downloaded directory), narrowed the pressure verdict to the two asks that still decide pass or fail, and put the strain-pool keywords of the two cp2k examples under a criterion that reads them through the skill's own loader. Before it, 12:10 - the three postures re-measured on this checkout through the same probe `tools/deploy.py` sends to the server, after #392 put three more inputs on the plan (the isotropic zero-pressure twin and the two crossing legs on a force-field environment): **445 planned / 197 closed / 248 working / 251 printed** for the optimize round; **676 rows, 670 of them working** under `_runall: true` (the six are capability closures - two want `sbatch`, four want TensorFlow); **399 / 232 / 167**, printing 170, under `_runall: false`. Both are host readings: on this `python3` (3.9.6) the four `nn_*` examples plan as named skips, and the repository `.venv` (3.11, TensorFlow 2.21.0) prints 715 for the same full-round posture (example_tests 190, skill_tests 400). Family counts re-measured: 96 qualified YAMLs, 24 `optimize_*` with 16 `optimize_cell_*` among them, `md_lammps_` 12 and `constraint_md_` 33 with no overlap; the `examples` section ships 21 `true` keys (the family key and twenty named cases) against 76 `false` and two `condition`. The 2026-10-19 00:11 pass recorded 439/197/242/245, 670 and 396/229/167 over 93 YAMLs and 18 `true`, and rounds 15, 16 and 17 (243/232/8, 243/241/2, 245/238/4, `verify.py` clean on the last two) stay collected. What moved this pass on the judging side: the pressure direction clause pairs against a zero-pressure twin matched on engine list, optimizer, declared coupling and starting volume within 1e-6, with two modes of one example merging into one reference (#392, on top of #391); `tools/verify.py` section D got a second source for `shape_ratio_change` and a PySCF engine reader, and its selftest now plants the pairing in both directions. The engine-side reading that pass left owed - the 1700 bar pair and the isotropic twin's own drift - was paid by round 18 and is recorded in the two pressure examples' headers and in `dev/methods/OPTIMIZE_DEVELOPMENT.md` section 15 (15.195667 A^3 loaded against 17.614350 unloaded on the isotropic side, 14.748035 against 17.087668 on the anisotropic one). The rest of the document was not re-audited line by line.

---

> **Multi-center acceptance (refactor R0-R6, 2026-09-08).** The multi-center
> unified base is qualified by dedicated verification modules
> (`qualification/verification/`): `test_multicenter_baseline.py` (R0 frozen
> single-center behavior + the three pinned legacy defects),
> `test_multicenter_schema.py` (R1 canonical center plan, docs §13.1 items 1-5),
> `test_task.py` / `test_core.py` / `test_calculator.py` (R2 identity + property
> requests, R3 dispatch/executor, R4 `TestR4fMultiCenterGate` finite-difference +
> serialize round-trip), `test_dispatch.py::TestTermWorker` (R3 remote per-term
> worker), `test_electronic.py` (R5 dispatch layer, r5f gate),
> `test_multicenter_md.py` (R6 assembled-gradient verlet, center-aware target
> states, sync guard, partitioned output matrix, run-log center protocol, full
> C>1 AIMD loop). The docs §13 acceptance matrix (items 1-22) maps onto these
> modules; see the coverage cross-reference below. A real audited backend
> end-to-end concurrent qualification is deferred (the current gates use
> mock/analytic backends, which prove framework scheduling, not runner
> thread-safety). What #301 does prove about the runners is checked without
> them: `test_thread_anchoring.py` runs real children of two terms in one
> thread pool and scans every skill runner for process-wide state.

## 1. Overview

The Qualification framework is the software quality assurance infrastructure of PyRAI3MD. It qualifies the correctness, compatibility, and robustness of the PyRAI3MD Runtime and its ecosystem across different computing environments.

Unlike a simple test driver, the Qualification framework is a Software Qualification System covering:

- Runtime verification;
- Workflow verification;
- Calculator verification;
- Skill qualification;
- Example qualification;
- Algorithm benchmarking;
- Integration testing;
- Regression testing;
- Execution planning for automated tests.

This naming forms a clear responsibility split with `validation/` (input/schema validation) and unit tests: Qualification verifies complete software behavior, not isolated units. The three-layer quality model (skill qualification, framework verification, application-level QA) is summarized in Section 6.9.

The design follows these principles:

1. **Real execution path verification**

   All qualification tasks must verify the real software execution flow. Qualification should not bypass core pipelines by directly calling internal functions.

2. **Separation of runtime and qualification infrastructure**

   The core computation framework (`pyrai3md/`) and the qualification system (`qualification/`) remain independent. Qualification dependencies and logic do not enter the core runtime modules. Qualification never participates in calculations; it only verifies calculations.

3. **Deployment and qualification lifecycle separation**

   Deployment prepares the execution environment; qualification verifies the deployed software. The deployment tool (`tools/deploy.py`) is an independent development utility, not part of the qualification framework.

4. **Single configuration source**

   All environment and deployment configuration is managed through a single file: `qualification/pmd_env.yaml`.

5. **One-way calls**

   Qualification calls the runtime only through its public entry points (CLI commands, Interface layer parser, converters). The runtime never depends on qualification.

---

# 2. System Architecture

```text
                 PyRAI3MD Repository

        +------------------------------+
        |                              |
        v                              v

      tools                        qualification

   deploy.py                  Software Qualification
                                   Framework

        |                              |
        v                              v

 Remote Environment           pyrai3md test

                                     |
                                     v

                            Qualification Manager
                               (manager.py)
                                     |
                                     v

                     pyrai3md / pyrai3md_agent / pyrai3md_skill
```

## 2.1 Component Responsibilities

### pyrai3md (Runtime System)

Responsible for:

- Software execution logic;
- Data structures;
- Computation pipelines;
- Capability invocation;
- Skill registry querying (`pyrai3md/skills/query.py`);
- Skill wrapper for independent testing (`pyrai3md/skills/wrapper.py`).

CLI entry points:

```bash
pyrai3md <input>.yaml         # Traditional CLI (yaml / classic input)
pyrai3md test                 # Forward to qualification.runner
pyrai3md_agent <input>.config # Agent CLI (sectioned dict JSON)
pyrai3md_skill <input>.json   # Skill independent test (wrapper, flat JSON)
```

The CLI does not implement qualification logic; it only forwards the `test` command to `qualification.runner`.

---

### qualification (Software Qualification Infrastructure)

`qualification` represents:

> A software qualification infrastructure that prepares, schedules, executes, and evaluates software verification tasks.

Responsible for:

- Qualification configuration parsing (`pmd_env.yaml`);
- Skill registry querying and input file generation;
- Example qualification (official workflow YAMLs);
- Qualification plan generation;
- Qualification execution management (all modes);
- Result collection and reporting.

Not responsible for:

- Environment deployment;
- Remote login;
- Component implementation.

---

### tools/deploy.py (Deployment Tool)

Responsible for:

- Remote server connection;
- Source code synchronization;
- Software installation;
- Test triggering on remote;
- Periodic result collection.

Not responsible for:

- Qualification planning;
- Qualification task generation;
- Qualification result analysis.

---

# 3. Directory Structure

The complete project directory structure is defined in [ARCHITECTURE.md](ARCHITECTURE.md).

Qualification-related modules:

- `qualification/` — Qualification infrastructure (manager, runner, pmd_env.yaml) plus three flat pillars at the same directory level: `verification/` (the **only** home of the main framework unit tests, plus their planner; the utilities suites that used to sit in `pyrai3md/utilities/tests/` moved here under CHANGELOG #299), `certification/` (skill tests: planner, handler, converter) and `benchmark/` (algorithm benchmarks: planner + one task module per task type + unified `<task>.yaml` case definition with optional dedicated `<task>_<method>.yaml` overrides)
- `qualification/manager.py` — Config parsing and plan aggregation: it delegates planning to `verification/planner.py`, `certification/planner.py` and `benchmark/planner.py` and owns only the shared script writers and flag coercion
- `qualification/jobs.py` — Grouped Slurm round: reads the `job_group` each planner stamps on its tasks, writes one spec and one `.sbatch` per case group under `tmp_*/jobs/`, submits and polls them (bounded in-flight window), reads each task's row fragment, and hands the rows back to the runner in plan order. The job entry (`python -m pyrai3md.qualification.jobs <job.json>`) executes one group's tasks on its node with the round's config snapshot
- `qualification/certification/handler.py` — Skill test file discovery (`<skill>/tests/` YAML cases and `test_*.py` runtime scripts) and the generic classic-input skill section builder (dev_tool functionality fully absorbed; the per-skill `dev_tool.py` modules no longer exist)
- `qualification/certification/converter.py` — Unified test YAML parsing (`parse_test_yaml` / `generate_test_configs`, the former per-skill dev_tool generation), example YAML to flat .json conversion, and flat .json to legacy input / pyrai3md agent config conversion (shared sectioned dict). It renders `&constraint` from a config's `constraints` block through `schema.adapter.write_constraint_section` rather than spelling the grammar out here, so the generated classic input and the YAML it came from read back as the same run; a block the flat text cannot state raises during conversion instead of emitting a file that means something else (CHANGELOG #271)
- `qualification/benchmark/` — Flat benchmark layout: `planner.py` pairs every task module (`fssh.py`, `ktdc.py`, `gsh.py`, `lzsh.py`, `perf.py`) with its unified case YAML (`fssh.yaml` / `ktdc.yaml` / `gsh.yaml` / `lzsh.yaml` / `perf.yaml`; the Tully model is selected through the dynamically injected `method`, dedicated `<task>_<method>.yaml` files may override it); the task module owns the complete calibration task (input builders or the `build_legs` contract, reference handling, evaluation). `timing.py` and `plotting.py` are infrastructure only (the wall-clock wrapper of the perf legs and the S0 population plot helper, never task modules)
- `tools/deploy.py` — Remote deployment tool (SSH + rsync). Its task denominator comes from the running round's own log, not from a plan it runs itself (see the polling section); in a slurm round the newest round's `tmp_*/jobs/jobs.json` is the ledger `--force` reads to `scancel` the jobs before stopping the orchestrator
- `tools/verify.py` — Post-round verifier (sibling of `deploy.py`, one file, one entry point `python tools/verify.py <work_dir>`). It re-derives what a finished round reported from the files that round left behind, using the framework's own readers (`read_md_energies`, `read_md_lattice`, `energy_conservation_metrics`, `cell_frame_metrics`) so it cannot drift from the writers: A recomputes every recorded example metric from its frame records and flags any record older than the round; B compares each logged step's electronic and kinetic energy against the same step's record row; C reads a surviving engine file back and requires the recorded potential energy to be a number that file printed; C2 does the same for an optimize leg's `opt_status.txt` and counts the legs whose own log closed on an optimization budget stop; C3 requires a composite row's term columns to add to the energy beside them (a deviation of either sign is named, and a crossing's gap term is read as an offset rather than judged); D re-derives a variable-cell verdict from its three sources (the status record, the recorded `cell_metrics`, the printed `[CELL]` sentence) and, for a path round, re-reads each variable-cell path leg's own `path_status.txt` by column name and matches its volume against the leg's `Lattice=` frames; E compares one example's two carriers, on the recorded metrics and on each record's term decomposition; F re-derives the path family's own claims - the held internal coordinate of every committed frame of a leg that declares a strict condition, the coordinates of every atom a leg freezes (`fix_atoms`, over the walkable frames: the band endpoints are inputs), and the energy a composite leg and its single-layer counterpart record for the geometry both declare; G reads one molcas case's three carrier logs with the skill's own parser and requires the four arrays to agree, then asserts each property's shape and provenance so a zero-filled array of the right length cannot clear a tolerance; H re-derives the periodic field probe's cross energy with a converged Ewald sum, which is the only reading that names a deck having lost its pinned `kspace_modify gewald/mesh` line, because the A−B identity holds only when both runs share one split; I reads the four QM/MM force/gradient files back and checks per atom `grad = -force + u_gas.force + qmmm_el.force` on the QM rows and `grad = -force` on every other row, with the tolerance set by the least precise file in the comparison (a cold dump keeps six significant digits, recognised by the missing id column in its header, so the loose line is never applied to a high-precision file); J audits adaptive sampling's two-round product tree - two rounds of five trajectories each, `selected` against its own xyz, the population list length, the label count against the dataset frames and against the initial set, a servable model folder, the `&method alignment` record, the run-log markers, `reference_failed`/`verified`/finite values, and the two carriers of one case agreeing on every structural count. J deliberately does not compare the reference energies across carriers: those are calculations, not a serialization, and a QM label may ride a different SCF path. Its auxiliary reading is the sidecar rule stated in `dev/methods/ADAPTIVE_DEVELOPMENT.md` section 14: a QMMM case must have `iter-2/reference/aux-labels.json` and a non-QMMM case must not, so `unexpected_aux` is a counted problem. K compares every skill case's three carriers file-only - the engine deck each carrier wrote (per-carrier paths masked) and the final energy each engine printed where it writes one - and never runs a skill; it was introduced after round `tmp_20261110_093321`, where OpenQP `tddft_gradient` asked the skill carrier for `grad=2,3` and both framework carriers for `grad=1` and every leg passed its own status. Measured on round `tmp_20261106_102745` (six adaptive slurm carriers, 113 rows / 109 pass / 0 fail / 0 error / 453 skipped, the skip rows being the ones the temporary adaptive-only posture closed by switch): J read 6 carriers over 3 cases with every counter 0 except `unexpected_aux 2`, the two ONIOM carriers of the pre-fix build that wrote a sidecar carrying only `region_indices`; the same audit over a post-fix local ONIOM rerun reports `unexpected_aux 0` and writes no sidecar at all, and the two flagged rows are what prove the check is load-bearing rather than inert. Coverage is stated, not assumed: the output lists which skills have readers, which rows it refused to compare (layered aggregates, runs reporting energy contributions, a path leg's variable-cell verdict, whose record uses a different reader) and that `md.pop` populations have no engine artifact to check against. `--selftest` plants a wrong energy, a wrong cell metric, a stale record, a broken hold and a composite divergence, and fails unless every one is caught; the path arm was proven load-bearing by reverting its condition-key reader to the kind-named form, which makes the selftest fail exactly the way that defect read on a real round (six of ten legs audited, no problem reported).
- `<skill>/tests/` — Skill test case YAML definitions and runtime test scripts (discovered by the certification handler)
- `pyrai3md/examples/` — Official workflow examples (single source of truth for example qualification)

---

# 4. Configuration Management

## 4.1 Single Configuration File

All environment and deployment configuration is stored in:

```text
qualification/pmd_env.yaml
```

Format: YAML

Structure:

```yaml
deploy:
  host: ""
  username: ""
  port: 22
  install_path: ""
  jump_host: ""
  jump_username: ""

environment:
  python: "python3"
  conda: ""
  scheduler: ""
  dependencies: []
  shell:
    - "module load gcc/11.3.0"
  export:
    OMP_NUM_THREADS: [4]
    MPI_NP: [4]

examples:
  energy: true
  gradient: true
  optimize_minima: true
  optimize_gdiis: true
  optimize_tda: true

slurm:
  partition: ""
  nodes: 1
  ntasks: 1
  cpus_per_task: 1
  mem: ""
  time: "24:00:00"
  job_name: pyrai3md
  output: slurm-%j.out
  error: slurm-%j.err
  exclusive: false
  # ... (see INPUT_SYSTEM_DEVELOPMENT.md for full list)

skills:
  <skill_name>:
    <env_var>: <value>
    ...
```

## 4.2 Configuration Sections

### deploy

Deployment target and connection parameters:

- `host`: Target server address;
- `username`: Login username;
- `port`: SSH port (default 22);
- `install_path`: PyRAI3MD installation path on remote;
- `jump_host`: Jump/bastion host tunneled via `ssh -J` (empty = no jump host, direct connection);
- `jump_username`: Login username on the jump host (empty = automatically falls back to `deploy.username`; the local username is used only when both are empty).

### environment

Runtime requirements of PyRAI3MD:

- `python`: Python interpreter command;
- `conda`: Conda environment name (empty = use current environment);
- `scheduler`: Round execution mode of the qualification harness (`''` or `local` = one local process; `slurm` = the plan is grouped into independent Slurm jobs and submitted). It belongs to the **main framework** and says nothing about how a skill computes:
| environment.scheduler | how the round runs | how a skill inside it runs |
|---|---|---|
| `''` / `local` | every task in one local process, sequentially | `skills.<name>.execution_mode` decides (local, or the skill's own `sbatch -W`) |
| `slurm` | one job per independent carrier: verification = 1, integration = 1, each example carrier (yaml/input separately; a chain's members share the mode's job) = 1, each skill carrier leg (a chain's cases stay together in one leg) = 1, each skill's runtime pytest phase = 1, each benchmark case = 1 (after every other category; the physics calibrations run together, the perf timing case alone and serial) | `execution_mode` still decides; `slurm` there is a deliberate nested submission and is subject to the cluster's compute-node submission policy |

  The orchestrator is still the `pyrai3md test` process: it writes one spec and one `.sbatch` per group under `tmp_*/jobs/`, submits them (`environment.max_in_flight`, default `0` = all at once; `N > 0` caps unfinished jobs), polls `squeue`/`sacct`, reads each task's row fragment and prints the same report in plan order. `slurm.time` is the per-job limit and is not scaled automatically: a group with several tasks prints a plan-time reminder, and a job that hits its limit keeps the rows its finished tasks already wrote while the unfinished ones become error rows.
- `planning_workers`: Worker processes for the plan phase. Planning parses every example YAML and renders its carriers; the work is independent per example, so a spawn pool shortens a full round's startup. `0` = auto (at most 8, never more than the host has), `1` = serial. It changes only how fast the plan is built, not what the plan contains or how tests execute; a chain family stays in one worker because its members share a directory.
- `max_in_flight`: Grouped-round in-flight cap. `0` (default) submits every group at once and lets the scheduler queue it; `N > 0` keeps at most N submitted-and-unfinished jobs and tops the count back up as they finish. Read by the qualification orchestrator only; it is not a `#SBATCH` directive, so it lives here rather than in the `slurm:` section (which refuses keys that map to no directive).
- `dependencies`: Required software dependencies;
- `shell`: Raw shell commands (module loads, activation lines, ...) copied verbatim into every generated .sh and .sbatch script before the export section;
- `export`: Environment variables to export before execution. Each variable maps to a list of values. The runner merges these values when loading.

### _timeout (per-test execution timeout)

Top-level per-test execution timeout in seconds (default `300`):

- The runner's `subprocess.run` uses it as the backstop timeout for every local test task (skill/example modes); benchmark tasks carry their own `DEFAULT_TIMEOUT` (3600 s);
- Each generated execution script embeds `timeout -k 10s (_timeout - 5)s bash -c '<command>'` (the margin is `manager.SCRIPT_TIMEOUT_MARGIN`), so the script's own `timeout` fires 5 s before the runner backstop and kills the whole engine process group (`pyrai3md -> mpirun -> cp2k` share one process group — no skill/workflow subprocess uses `setsid`/`start_new_session`). This guarantees a timed-out calculation leaves no residual processes, which would otherwise keep consuming cores and slow every subsequent test;
- The runner maps the `timeout` exit codes (124 = timed out, 137 = SIGKILL escalation) to an `error` verdict `Test timed out (Ns)`, distinct from a calculation `fail`;
- `deploy.py`'s polling idle timeout is its own base (300 s) plus `_timeout`, so it is always >= the remote per-test timeout and a task running its full budget is never misjudged as a hung remote;
- `deploy.py`'s task count has one source: the running round's own log. Its header prints `Total tests: N` after planning, every progress line carries `[i/N]`, and the poller adopts that number as soon as it appears. The tool no longer plans the round before triggering: that ran the full planner a second time (measured 3 m 52 s and 3404 staged files on a full round) for a number the round is about to publish itself;
- completion is decided by `status.json`, never by the denominator: the count only labels progress.
- Slurm paths (`environment.scheduler: slurm` or `skills.<name>.execution_mode: slurm`) are unaffected: their duration is bounded by the slurm time directive, so neither the subprocess timeout nor the script wrapper applies. In a grouped round the bound is the job's `slurm.time`, not a per-task budget.

Every example answers to this one budget. The per-example `_example_timeout: {<workflow>: <seconds>}` map (#226, resolved by `manager._resolve_example_timeout`) was deleted on 2026-10-20 by ruling, together with the one entry it still carried. Those twelve LAMMPS runs never needed a budget of their own - the deployment finishes one in **12 s** (`PyRAI3MD End: … Total: 00:00:00:12`, read from `tmp_20260928_184646`) against the 1200 s backstop, and the override entries they used to carry (14400 s each, sized for a cold-path 10 ps run at 950 ms/step) were retired in #245. The consequence is deliberate - an example that silently falls back to the subprocess path needs ~1900 s for the same 2000 steps and is stopped at 1195 s, so a lost resident service reports a timeout instead of passing hours later. The strained-box example that #385 gave a 2400 s entry kept it only until round `tmp_20261019_200652` measured both of its legs at **735 s and 728 s** (61% of 1200 s) once each file declared the two workers that `multitask: true`/`multipool: 2` describe; with `maxit: 2` the room was no longer needed, so the entry and the map went together. The single budget is pinned where it is planned: `test_manager.py::TestEveryExampleAnswersToOneBudget` asserts the shipped file has no `_example_timeout`, that `_resolve_timeout` answers 1200, and that both planned `optimize_cell_gap_cp2k` tasks carry exactly 1200 s.
Round 18 (`tmp_20261019_122659`, 251 rows, 42 min to row 104) then attributed its wall clock to this family and two rulings followed. `optimize_cell_cp2k`'s two carriers took **13 m 41 s and 13 m 37 s** - 821 s, 68% of the 1200 s backstop - so the strained-box example that was still running at collection time had its `maxit` cut from 5 to 2 by ruling (the 2400 s entry it kept at that point was later deleted with the map itself, once round 20 measured the two-step legs inside the shared budget). And neither cp2k example declared `multitask`/`multipool`, which `service/stress/num_stress.py:382-385` reads as "undeclared means serial": the twelve strain points of a step were being walked one after another. Both files now carry `multitask: true` with `multipool: 2` - the node budget is `workers x cp2k_nproc x thread`, the test environment exports `MPI_NP 1` and `OMP_NUM_THREADS 1` and runs `cp2k_nproc: 4` on a multi-core node, so 2 is what fits without tripping `_warn_oversubscription`. The pair is pinned by `test_example_dispatch.py::TestTheStrainedBoxExamplesAskTheSkillsOwnStrainPool` through two chains, because they answer different questions: the framework's task queue must still be carrying `multitask=True`/`multipool=2` when it hands the skill its flat config, and the skill's own schema pipeline (`skills/cp2k_skill/schema/adapter.build_config` + `normalize`) must then resolve the two workers the strain route asks for - with `cp2k_nproc` read back as 4. Constructing the skill's config class straight from that flat dict is not that path and gives a different answer (measured: `SkillConfig(**flat)` leaves `environment` None, so such a probe reports the pool serial while the run asks for two workers), which is why the criterion walks the skill's own loader instead of the shortcut. The third criterion pins the other layer's reading rather than claiming it as design: the framework's own graph node for these two examples says `multitask=False`, `multipool=1`, so the fan-out being asked for is the skill's over its own perturbed points and a second fan-out at the framework level would multiply against it (2 x 2 workers, each 4 ranks). The saving itself is a round reading, not a promise: pool 4 is what the skill's own certification case (`tests/dft_stress_num.yaml`) declares and 2 is what this node's arithmetic allows.
Round 19 settled both halves of that sentence, and exposed a third thing. `optimize_cell_cp2k` finished its two steps in **589 s in yaml mode against 821 s serial on round 18**, and in **815 s in `.input` mode against 817 s** - so the pool paid off in one carrier and not in the other, because the classic writer never stated it: a skill's three dispatch fields stay out of its `reference.DEFAULT` by design (a DEFAULT entry would write them into every generated input) and nothing else emitted them. #395 put the lines on the writer - presence-driven, `&MD` echoing what its own section declared and a skill block stating its own trio - and the same comparison over all 96 shipped examples came back with **92 generated classic inputs byte-identical and 4 changed**, which is the whole blast radius, priced rather than argued (`qualification/verification/test_example_dispatch.py::TestTheDispatchPoolReachesTheClassicCarrier`, 14 → 18 criteria). What the changed cp2k carriers give up is six duplicated lines (`charges`, `ci`, `multiplicity`, `pbc`, `spin`, `opt_maxit`): reading both modes back into a task queue shows the first five arriving in the skill config identically, and `opt_*` was already absent on the classic side before this fix (round 19's `.input` leg reported two steps and correct `[CELL]` thresholds without it). `optimize_cell_gap_cp2k` likewise completed two steps and passed in both modes (12 m 13 s / 15 m 58 s), so its `maxit: 2` ruling holds inside the shared backstop in the yaml carrier. The `.input` legs' first pool-enabled timings are round 20's reading, and it says the penalty is gone: on `tmp_20261019_200652` the same two carriers finished in **581 s (yaml) and 584 s (`.input`)** against round 19's 589 s and 815 s, and the engine's own `PROGRAM STARTED/ENDED` stamps show the twelve strain points walking as **six simultaneous pairs** (paired starts within 0.01 s), so the classic carrier really runs the pool instead of only declaring it.

Read by the three qualification planners (via `manager._resolve_timeout`) and by `tools/deploy.py`.

### _runall (master switch)

Top-level master switch governing every per-item test/calibration switch documented below (`examples.<workflow>`, `skills.<name>.test`, `benchmark.<task>[].enabled`):

- Three values, and the family-level and skill-level switches take the same three: `true` forces everything below it on, `false` forces everything below it off, `condition` leaves the switch one level down to answer. A **missing key means `condition`**. The shipped file reads `_runall: condition` for the length of the optimize round (see "What a round is right now" below); a full round is one key away, since `true` overrides everything under it. `false` used to mean "do not force anything" and now means "close everything" - that inversion is the one thing this switch changed for an existing file, and it is decided **above** the families: `_runall: false` with `optimize_test: true` runs no example at all (pinned by `test_manager.py::test_runall_false_defeats_the_optimize_family`). "Everything" is the engine work: the pytest phases that need no engine answer to `runtime_tests` and to nothing else, so `false` at the master still runs the framework tests and the 64 engine-free code checks (see the skill switch section);
- It only overrides switches already present in `pmd_env.yaml`: the config sections remain the sole source of truth, so `_runall` never enumerates skills or benchmark cases absent from the file (a missing skill/case is completed by adding it to `pmd_env.yaml`, not by the switch);
- It shares the example flag coercion, so an unrecognized value warns and never silently disables the run-all intent; a per-item switch is a leaf, and `_coerce_flag` there reports `condition` as the misplaced value it is and keeps the item open;
- Whatever a switch closes is still **listed in the plan**: one skipped task per closed item, whose `skip_reason` names the key that closed it (`disabled by _runall`, `disabled by examples.md_lammps_test`, `disabled by examples.optimize_test`, `disabled by skills.molcas_skill.tests.casscf_gradient`). What the **run prints** is one row per bucket for everything its switches closed (#360): the row is named `<bucket>_closed`, carries the number of tasks it stands for in `skip_count` and the per-key counts in `closed_reasons`, and lands in `results.json` in place of the individual rows. The distinction is what makes a stage round readable - 415 planned rows against 221 printed - while the plan stays enumerable, which is the only way a per-example switch can be audited at all. A skip that is not a closed switch (a postponed example, a host with no `sbatch`) keeps its own row, and the final report still groups by reason for those;
- Orthogonal to two policies it never touches: the `POSTPONED_EXAMPLES` postponement (a YAML outside the list is planned as a skipped task carrying the postpone reason whatever `_runall` says; no shipped example is in that branch now, and `test_manager.py::test_an_unqualified_example_is_still_postponed` pins the mechanism with a planted file while `test_every_shipped_example_is_qualified` fails if a new example ships unlisted) and the main framework tests (run in-process in a local round and as one job in a slurm round, no switch). Note that the qualified dynamics examples (`md_*_fssh`, `md_lammps_*`, `multicenter_md`) are ordinary switches: `_runall: true` forces them on together with everything else.

Read by the three qualification planners (example, skill certification, benchmark) and mirrored by the deploy task-count estimate.

### examples

Per-example switches for the official workflow qualification tests (one entry per file in `pyrai3md/examples/`), with six family switches sitting above them and below `_runall`:

- Key: example workflow name (file name without extension, e.g. `energy`, `optimize_minima`);
- Value: `true` = run the example test, `false` = skip it; the skip reason names the switch that closed it (`disabled by examples.energy`), so a case reachable from three different keys says which one to edit;
- Family switches, three-valued like the master: `md_lammps_test` covers every `md_lammps_*` example, `constraint_md_test` every `constraint_md_*` one, `optimize_test` every `optimize_*` one (twenty-five cases, seventeen `optimize_cell_*` among them), `path_test` every `path_*` one (the thirty-two `path_*` examples, asked for as `optimize.method` mep / irc / neb), `multiscale_test` every `multiscale_*` one (thirty-seven composite cases) and `adaptive_test` every `adaptive_*` one (the six adaptive carriers: single method, ONIOM2 mechanical and QMMM mechanical, each local and slurm) - which is what lets an optimization round, a path round, a multi-scale round or an adaptive round be asked for by one key instead of by the master. The prefixes are pairwise disjoint so that no key opens another's rows: measured on the shipped set of 201 YAMLs, `md_lammps_` 12, `constraint_md_` 33, `optimize_` 25, `path_` 32, `multiscale_` 37, `adaptive_` 6 and no stem matches two of them; the A/B reading on round `tmp_20261021_195813` - a path round closes all twenty-five `optimize_*` cases and an optimize round closes the `path_*` ones - is pinned by `integration/test_integration_manager_planning.py::TestPathFamilySwitch`, and the six-carrier adaptive family by `verification/test_manager.py::TestAdaptiveFamilySwitch`. `true` opens the family whatever its cases say, `false` closes it, `condition` (also the value when the key is absent) reads the cases one by one. Membership is the **name**, not the physics: `constraint_opt_periodic` and `constraint_opt_surface` do relax a geometry and stay individual on purpose, so an optimize round does not certify the constraint subsystem along the way, and `energy`, `gradient`, `multicenter_*` and the `md_*_fssh` set have no family either. The table is `manager.EXAMPLE_FAMILIES`, it is matched by `startswith` with the first hit winning, and `integration/test_integration_manager_planning.py::test_optimize_prefix_shadows_no_other_family` pins that the prefixes stay pairwise disjoint and that no ordering of the table changes any stem's decision;
- Missing section or missing key defaults to `true` (backward compatible);
- The qualified-list policy takes precedence: an example named in `POSTPONED_EXAMPLES` stays postponed regardless of any switch. Nothing is named there today -- all shipped YAMLs are planned, so the postponement branch exists only to catch a YAML added without a decision (pinned by `test_manager.py::test_an_unqualified_example_is_still_postponed`).
- **What a round is right now.** Re-measured on 2026-11-06 against the shipped file, through the runner's own counters (`_is_closed`, `_folded_count`) on this checkout with the repository `.venv`: `_runall: condition`, six family keys of which `md_lammps_test`/`constraint_md_test`/`optimize_test`/`path_test` are `false` and `multiscale_test`/`adaptive_test` are `condition`, and 47 case keys `true` against 154 `false`. That plans **610 rows - 111 main + 248 example + 226 skill + 25 benchmark - 384 closed by switches and 230 printed** (101 + 95 + 33 + 1, each bucket folding its closed rows into one). The adaptive six are inside that reading as three open local carriers and three closed slurm carriers. What follows belongs to the path posture the file carried on 2026-10-21 (`_runall: condition`, `examples.path_test: true`, `examples.optimize_test: false`, the other two family keys then at `condition`, and 101 case keys `false` against 27 `true` - every key outside the path route switched off with its justification left in place, deferred rather than withdrawn, so flipping the two family keys back reopens an optimize round). Measured on this checkout through the runner's own counters (`_is_closed`, `_folded_count`), by the same probe `tools/deploy.py` sends to the server: this posture plans **478 rows - 103 main + 144 example + 206 skill + 25 benchmark - 213 of them closed by switches and one more by a capability, and 268 printed** (the example, skill and benchmark buckets fold their switch-closed rows into one row each). Round `tmp_20261021_195813` ran it in the test environment host: **269 rows printed, 266 pass, 0 fail, 0 error, 213 closed**, of which the examples are 55 (the 54 above plus the folded row) - and 42 of those example rows are the twenty-one `path_*` cases in both carriers, all passing. **Round `tmp_20261022_084519` then ran all thirty-one**: 62 path rows, 62 pass, in a round of 289 rows (286 pass / 0 fail / 0 error / 213 closed); **round `tmp_20261022_151713` ran the thirty-second** (`path_cell_mep_multiscale`, a PySCF centre in a GFN-FF box): 64 path rows, 64 pass, 291 rows (288 pass / 0 fail / 0 error / 213 closed), and `tools/verify.py` clean over its artifacts. Ten more path cases were added after that round (the coverage audit's list: the penalty model on a cell and on an IRC, the periodic and variable-cell IRC, the geodesic periodic and variable-cell bands, the band holding an angle or a torsion, and the two frozen-atom cases), so the same posture now plans **500 rows - 103 main + 166 example + 206 skill + 25 benchmark - the same 213 closed by switches and one more by a capability, and 290 printed** (the example bucket's 76 is 64 path rows, ten other open rows, the one that names a capability, and the folded one). The readings below belong to the optimize posture this file carried before it (the file-as-it-stood readings for that posture, not re-measured here);
- **One engine certification stays open across both postures**: ruled on 2026-10-17, `skills.lammps_skill.test: true`, because #366's engine-side cell profile is a claim about a particular LAMMPS build (the case moved a box on a local `lmp_serial`, while `dev/TODO.md` records the test environment's byte-level parity diverging in the last digit where the test environment's was identical, so a local run is no substitute); `true` rather than `condition`, since the skill's eleven cases go together. `test_manager.py::test_the_shipped_stage_opens_the_three_engines_and_the_code_phase_rest` pins it, and pins the example family keys with it, reading the case and pytest lists off the disk rather than restating them here. Every other skill is closed at the skill level, which since #359 no longer hides that skill's pytest phase: `runtime_tests` is the only key that speaks for it. Re-measured for the optimize posture through the runner's own counters,
plan lists 100 main + 120 example + 200 skill + 25 benchmark = **445 rows, 197
closed by switches, 248 that do work**, and prints **251 rows** (248 + one
folded row for each of the example, skill and benchmark buckets); the 248 are
the 100 framework tests, the twenty-five `optimize_*` examples on two carriers
each
(the seventeen `optimize_cell_*` among them), the eleven LAMMPS cases on three
carriers (33 rows), and 67 engine-free pytest files - 64 in the ten skills' own
`tests/` directories and 3 in `pyrai3md/skills/tests/`, the layer the skills
share. Four optimize cases (`optimize_gdiis`, `optimize_minima`,
`optimize_partial_ts`, `optimize_tda`) read `false` on their own key and still
run, because the family says `true` and the family is consulted first. The
additions since the 421-row reading are accounted for row by row and no further:
five variable-cell examples on two carriers (+10, the example rows 102 -> 112)
and two LAMMPS `box_relax` cases on three carriers (+6, the skill rows 194 ->
200). The same file with `_runall: true` plans **668 rows - 100 main + 179
example + 364 skill + 25 benchmark - and every one of them works**; that reading
does not reconcile with the 691 of the previous pass in either direction (this
round's own additions would put it at 707), so the 691 is left as the file-as-
it-stood reading it was rather than re-derived, and what is quoted from it here is
only the part that still holds: two rows are host-capability skips, the examples
that submit to Slurm on a machine with no `sbatch`, which is what
`test_manager.py::test_runall_true_restores_the_full_round_from_the_stage_file`
asserts. On this interpreter the full-round figure is also a capability reading:
the four `nn_*` examples plan as named skips without TensorFlow (3 of the 4 rows
name the package), so a checkout with the ML extra plans more than 668. With
`_runall: false` the same stage file plans **395 rows, closes 228, and still
runs the 167 that need no engine** (100 framework tests + 67 pytest files),
printing 170 - the master closes the certification, not the code checks, and only
`runtime_tests: false` closes those. The stage buys a round that prints 243 rows
instead of 668 and ran 40 minutes on the cluster against the full pass's 4.5
hours; it costs the NN route every case reading it had, and that is recorded as a
coverage gap rather than argued away; a dispatched round on the cluster reads one
row more per Slurm-armed example than this environment does, so the difference is host capability, not the switchboard.
- An `examples` key that names neither a shipped example nor a family switch **warns at plan time**. It used to be ignored, and an ignored `md_lammps_tes: true` is the shape of mistake that reads as "I opened the family" while closing nothing.

This section is read only by the qualification manager. It also flows through the runtime config merge harmlessly: unknown sections are flattened and silently dropped by the schema adapter, so it does not affect the input system.

### energy

Tolerances (Hartree) for the NVE total-energy verdict of Section 6.6:

- `etot_drift`: largest permitted spread of the recorded Etot column. This is the verdict only for a record shorter than `RATE_MIN_FRAMES`, where no slope can be fitted; from that length up it is measured and reported but does not decide, because the spread of a long NVE record is dominated by the bounded kinetic/potential exchange a symplectic integrator always shows and cannot separate it from energy leaving the system;
- `etot_drift_rate`: largest permitted least-squares slope per frame, judged from `RATE_MIN_FRAMES` (50) frames up, where it decides;
- `etot_center_drift`: no tolerance, and not a verdict -- the widest band among a multicentre run's per-center records, reported so the amount a center exchanges with the box is readable (section 6.6);
- Missing section or key falls back to the shipped `runner.ENERGY_TOLERANCES` defaults (1e-3 / 1e-5), so an older pmd_env.yaml still gets judged.

### box

Thresholds for the NPT cell-stability verdict of Section 6.6 (defaults in
`runner.BOX_TOLERANCES` / `runner.BOX_NOTES`; each section overrides its own keys):

- `volume_drift` (fail 0.25, note 0.10): max |V/V0 - 1| over the frames written to
  `md.xyz`, V0 being the first recorded cell;
- `temperature_drift` (fail 3.0, note 0.5): max |T_inst/T_target - 1| from the `md.ensemble`
  `T_inst(K)` column, first frame excluded because the velocities have not settled;
- `cell_rotation_rad` (note 0.15, never a failure): the accumulated polar turn of
  the simulation frame. It is reported and not judged for the reason in
  `dev/methods/ENSEMBLE_DEVELOPMENT.md` section 8.2 -- the increments are real
  (5.7e-5 rad/step) and they do not cancel, so the accumulated angle measures how
  long the run was, not whether the cell was well behaved.

`temperature_drift` was 1.0 until tmp_20260929_091941 failed two berendsen
examples at 1.754/1.741 on boxes that stayed within 8-16% of where they
started. Those runs are not a broken integrator: a cell that moves does work
on the particles, and the number says the ensemble is not where the user
pointed it, which is a case-design finding. 3.0 keeps the code-hole signal --
the same round's csvr_pr reached 3.1e6 K and still fails (that pairing is
no longer shipped at all: schema/validate.py refuses csvr with any
extended-cell barostat, and dev/methods/ENSEMBLE_DEVELOPMENT.md 8.13 says
what replaced it) -- and leaves the
drift in `details` as a note instead of a verdict. `volume_drift` stayed 0.25
because it is the only number that ever caught a run that passed while
deforming (nh_mttk 0.575), and the healthiest finished run of that round
reaches 0.159: a 1.6x margin, so raising it buys nothing except blindness.

### monitor findings (`[DIAG`)

A run-health finding is not the same thing as a failed test. With
`control.debug` off -- the framework default -- the accumulated-strain bound
is reported and the trajectory continues, so the pass/fail verdict would be
silent about the only signal that says the box was drifting.
`runner._check_run_diagnostics` therefore reads the `[DIAG <check>]` lines the
monitor wrote into `<title>.log` and appends them to `details`, capped at
eight lines with a count of what was left behind. It never changes `status`:
watching is not judging. The closing summary line (`<n> finding(s), first
step ..., last step ..., worst ...`) is what distinguishes "the box held"
from "nobody looked" in a passing record.

### who prints, and in what order

The transcript used to arrive out of order: the banner and `Total tests: N`
sat below the first unit-test lines. Nothing wrote them late -- stdout is block
buffered once redirected to a file while `TextTestRunner` writes to stderr, so
the first stdout flush (a task line) dragged the buffered header after output
that had already landed. Three layers keep the order now: every header line is
flushed as it is printed (`_print_run_header`), stdout is switched to line
buffering for the run (`_stdout_line_buffered`), and the trigger runs the remote
test under `python -u`.

A task also owns what its children write. `_OwnedStream` duplicates fds 1 and 2
into a temporary file for the duration of one task, so stray output from a
subprocess -- logger lines, engine chatter, a bare `print` inside a test --
cannot land in the report between the counter line and its detail lines. On PASS
that text is discarded; on FAIL it is kept and printed with the traceback,
because it is often the only explanation of the failure. The library itself is
untouched: the run log of each calculation still receives everything.

Unit tests print one line on PASS (`[UNIT] ran=16 failures=0 errors=0`, or the
pytest tail for a `skill_runtime` task) and the whole captured block on FAIL.
Everything else that can appear on a PASS block is a qualifying note -- the
`_NOTE_TAG` lines such as `[ENERGY CONSERVATION]` -- which is the deliberate end
of the "silently green" class: a verdict that did not judge says so even when it
passes.

### result-line layout

The counter column is the layout anchor of the whole transcript, so it is
fixed width: `[%0*d/%d]` pads `idx` to the digit width of `total`, and every
detail line -- on a failure the clipped engine tail and captured block, on a pass only its qualifying notes, the `[NPT STABILITY]`, `[ENERGY
CONSERVATION]`, `[PBC]` and `[DIAG` traces, a skip's input path, a benchmark
metric row -- is indented by `_detail_indent(total)`, which is exactly the
counter's width plus its left margin. For a 38-task round that is the 10
spaces the shipped logs already show; a 107-task round becomes 11. Two
consequences are load-bearing and tested:

- the layout lives in the printer, not in `details`, which stays unindented
  data (six assertions in `test_npt_stability` select on its substrings);
- padding is with zeros. `tools/deploy.py._count_log_progress` parses
  `r'\[(\d+)/(\d+)\]'` from the captured remote log to decide whether a run is
  still advancing, and `'[ 1/38]'` does not match -- the same class of bug as
  #244's `pgrep` self-match, a probe that silently stops measuring the thing
  it was written to measure.

Engine captures are clipped on line boundaries (`_clip_to_lines`) rather than
at a character count, because a raw `[-500:]` began mid-word and the 09:19
record showed a divergence as `ce, rotation)`. The three verdict traces that
used to ASSIGN `details` (`[LOG STATUS]`, `[ENERGY CONSERVATION]`, `[NPT
STABILITY]` exceeding a tolerance) now append, so the captured stream that
explains the verdict survives next to it.

### slurm

SLURM job parameters for test submission. Common fields:

- `partition`: Queue/partition name;
- `nodes`, `ntasks`, `cpus_per_task`: Resource allocation;
- `mem`, `mem_per_cpu`: Memory settings (mutually exclusive);
- `time`: Wall time limit;
- `job_name`, `output`, `error`: Job identity;
- `account`, `qos`: Billing and service tier;
- `exclusive`: Exclusive node access (boolean flag);
- `constraint`, `gres`: Node constraints and generic resources;
- `mail_type`, `mail_user`: Email notifications;
- `array`: Job array specification;
- `extra`: Additional raw sbatch options.

Empty string values are omitted from the generated script. Boolean flags generate `#SBATCH --exclusive` when true, omitted when false.

Two consumers read this section, in two dialects: `manager.build_slurm_script` wraps a qualification **task** (skill tests in `pyrai3md_input`/`pyrai3md_agent` modes, and the benchmark compute tasks) in a batch script of its own, while `workflow/dispatch/slurm_run._build_script` builds the script of a **child runtime of a run** that asked for `remotetask`, mapping a named set of keys onto canonical directives and refusing a key that maps to nothing. Main framework tests do not submit at all. An example task is started locally like any other, but the run it starts may submit its own children -- `examples/nn_grid_search.yaml` and `examples/md_batch.yaml` do -- so a host with no `sbatch` skips those examples by name rather than failing them (Section 6.6).

A run may restate any of these keys in its own input (`slurm:` block in a YAML, `&SLURM` in a classic input, the `slurm` entry of an agent config). The merge is per key, applies to that run alone, and is read even when the input declares no `&environment` section; see [INPUT_SYSTEM_DEVELOPMENT.md](INPUT_SYSTEM_DEVELOPMENT.md) section 7.5.

For the full slurm configuration reference, see [INPUT_SYSTEM_DEVELOPMENT.md](INPUT_SYSTEM_DEVELOPMENT.md).

### benchmark

Per-task settings for the benchmark calibration tests: one subsection per benchmark task module (keyed by the module stem, e.g. `fssh`; the shipped `ktdc`, `gsh` and `lzsh` algorithms own a sibling subsection each, and future algorithms such as `znsh` add one the same way). The subsection is either a single parameter dict (one calibration case) or a list of parameter dicts (several cases of the same task calibrated in one pass). Every entry selects the Tully model through its `method` key and carries the physical parameters that override the shipped case YAML:

```yaml
benchmark:
  # Multi-method calibration: one independent case per list entry.
  fssh:
    - method: tully1     # model injected into the unified fssh.yaml
      ninitcond: 100     # trajectories (parameter pass-through)
      nstep: 300         # md steps (T = 6000 au)
      dt: 20             # md time step (au)
    - method: tully2
      v0: 0.005          # per-method wavepacket parameters
    - method: tully3
  # The single parameter dict (one case) stays supported:
  # fssh:
  #   enabled: true      # optional switch (example flag coercion)
  #   method: tully1
  #   nactype: nac       # md-schema keyword (nac: qdm NAC, ktdc: energies only)
  #   deco: '0.1'        # decoherence correction (md-schema deco keyword)
  #   x0: -20            # initial coordinate (Bohr)
  #   v0: 0.005          # initial velocity (Bohr/au)
  #   sigma: 0.5         # wavepacket width (Bohr)
  #   model_mass: 2000   # model mass (au)
  #   initial_state: 0   # initially populated diabatic state
  #   ngrid: 2048        # reference grid points
  #   substeps: 10       # reference inner steps per md step
```

- `method` resolves the case YAML in two layers: a dedicated `<task>_<method>.yaml` (model-specific case) wins over the unified `<task>.yaml` (e.g. `fssh.yaml`), whose `production.method` the entry overrides -- calibrating another Tully model needs no new YAML file; neither file shipped produces a diagnostic skip;
- every entry expands into its own isolated `benchmark/<task>_<method>/` run directory and compute task (the run directory is keyed by the method, so a duplicated method plans only its first occurrence and reports the rest as diagnostic skips);
- every other field overrides the corresponding case YAML parameter at plan time (production: method/ninitcond/nstep/dt/sfhp/nactype/deco/x0/v0/sigma/model_mass/initial_state; reference: ngrid/substeps), and the merged case YAML is written into the run directory under the case key so the evaluation tool sees the identical settings;
- `enabled` shares the example flag coercion, so a typo never silently disables a task;
- a missing `benchmark` section keeps every shipped case enabled with the case YAML defaults (backward compatible); when the section exists only the configured entries are planned.

### skills

Dict of registered skills and their environment-specific configuration. Each skill entry maps to a set of key-value pairs that the manager and runner use during test generation and execution.

Example:

```yaml
skills:
  molcas_skill:
    molcas_path: ''        # OpenMolcas installation directory
    molcas_mem: 4000       # Memory in MB
    molcas_nproc: 1        # OpenMP processes
    molcas_print: 0        # Print level
    molcas_project: MOLCAS # Default project name
    test: condition          # three-valued, as the master switch is
    runtime_tests: true      # the tests/ pytest scripts, one switch for the phase
    tests:                   # one switch per test case YAML, keyed by its stem
      casscf_energy: true
      casscf_gradient: false

  cp2k_skill:
    test: true             # certifies every case, so the maps below it are not read
    cp2k_path: /opt/cp2k-2025.2
    mpi: 2                 # MPI process count — cp2k uses the mpi key,
    cp2k_project: CP2K     #   not a cp2k_nproc key
```

The skill's own switches are the level below `_runall` and the level above the cases: `test: true` certifies the whole skill and neither `runtime_tests` nor `tests.<case>` is read, `test: false` closes the YAML cases, and `test: condition` (the only setting where the per-case map speaks) lets `runtime_tests` and each `tests.<case>` decide for itself. A case key that is absent means `true`, so the map closes cases rather than whitelisting them, and every closed one is listed as a skipped task naming its own key. `test`, `tests` and `runtime_tests` are planner-only keys and are never injected into a generated skill config (`planner.QUALIFICATION_ONLY_KEYS`).

`test: false` does **not** close the pytest phase, and neither does the master's `false`: what a skill's `tests/test_*.py` files cost is seconds and no engine, so the switch that speaks for them is `runtime_tests` alone (#359's ruling - a local-only code check is a stand-in for an unreachable server, not extra content). `true` from the master or from `test` still overrides `runtime_tests`, because that setting promises everything under the skill runs. The shared `pyrai3md/skills/tests/` group has no key of its own and follows the aggregate of the skills' `runtime_tests` switches: open when any skill's phase is open, closed and listed with `skills.*.runtime_tests` when all are closed. The phase lives in `planner._runtime_tasks`, which both the open and the closed branch of `plan_skill_tests` calls.

One group belongs to no skill: `pyrai3md/skills/tests/` checks the layer the skills share (the agent's `calculate()` forwarding contract, the wrapper's skill-name resolution). `planner._shared_runtime_tasks` plans it once per round under the label `skills`; it has no key of its own but follows the skills' pytest phase: it runs when at least one skill's `runtime_tests` is open (or the master/`test` switch forces it), and it is listed as skipped with `skills.*.runtime_tests` when every skill's phase is closed. The rows arrive as `skill_runtime` like a skill's own pytest files. They deliberately do not go through the main-test phase: that phase discovers with `unittest.TestLoader`, which collects zero cases from a pytest module and would report the silence as a pass (pinned by `test_qualification_slurm.py::TestSharedSkillLayerTestsArePlanned`).

The manager uses the skill names to discover test cases through the certification chain (`certification/planner.py` -> `certification/handler.py` file discovery -> `certification/converter.py` parsing) and generate skill test plans. The environment variables are passed to the skill backend during test execution.

Execution policy keys (`execution_mode` and the `slurm_*` keys of the skill schema Environment) deserve special treatment:

- `execution_mode: slurm` makes the skill backend submit its own blocking `sbatch -W` job in all three skill test modes (`pyrai3md_skill`, `pyrai3md_agent`, `pyrai3md_input`). This is the skill's execution policy and is independent of `environment.scheduler`, which decides only how the harness round is submitted (one local process, or one job per test case group); when the round is grouped, a skill with `execution_mode: slurm` submits a nested job from the compute node, which some sites forbid;
- these keys use override semantics during injection: a non-empty value in pmd_env.yaml replaces the value in the generated config, because the skill test YAMLs hardcode `execution_mode: local` and fill-only injection could never flip the switch. All other keys keep fill semantics (existing non-empty config values are never overwritten);
- `slurm_partition`, `slurm_time`, `slurm_ntasks`, `slurm_memory`, `slurm_command` feed the skill's own sbatch script (key names follow each skill schema); empty values fall back to the skill schema defaults;
- sbatch availability is verified at plan time: a skill with `execution_mode: slurm` on a host without `sbatch` produces a diagnostic skip task instead of failing at runtime;
- skill_slurm tasks run without the `_timeout` subprocess timeout (duration is bounded by the skill slurm time directive), and a passing return code is additionally asserted against the slurm artifacts (`{project}.sbatch` and `{project}_slurm.out` under the test directory) so a silent `run_local` fallback cannot pass as a slurm test.
- Unconverged geometry optimizations are tolerated: qualification runs often truncate the optimization step budget to keep tests cheap, so a non-zero exit whose output or run log (`<title>.log` `Status: failed:` tail) carries an unconverged signature (`Maximum optimization steps reached`, `not converge`, `No convergence`, `_RC_ITERATION_LIMIT_`) is reported as `PASS[NOT CONVERGED]` instead of a failure. The skill backends still raise on abnormal termination; the tolerance lives entirely in the runner verdict. **Zero-exit runs carry the same annotation (#399).** Input/config modes exit 0 while the framework's optimize workflow records the unconverged state in its own log, so that route now appends the identical `[NOT CONVERGED] …` detail and sets `not_converged` on the result instead of staying a bare `pass` — the verdict is still `pass` (one budget rule, two routes, no third outcome), and `tools/verify.py` section C2 counts those legs as `budget_stopped` from their own closing lines, which is how round 20's 32 variable-cell legs became distinguishable from the converged ones.

---

## 4.3 Authentication Design

`pmd_env.yaml` does not store passwords.

Reason:

- Avoid sensitive information entering the code repository;
- Keep configuration file shareable.

Authentication method:

- SSH Key Authentication (preferred): `ssh-copy-id user@host`
- Temporary password input: If SSH key is unavailable, password is requested during deployment only. Not saved, not written to configuration.

---

## 4.4 Interactive Configuration

If `deploy.host` or `deploy.username` is missing, `tools/deploy.py` enters interactive mode:

```text
Deployment configuration incomplete.

Host:
>

Username:
>

Port [22]:
>

Install path:
>
```

If `environment.conda` is empty, the tool asks for a conda environment name:

```text
Conda environment name (press Enter to use default):
>
```

- If the user enters a name, it is saved to `pmd_env.yaml` and used for remote activation.
- If the user presses Enter (empty), the default environment is used, and `_prompts.conda_asked: true` is written to `pmd_env.yaml` so the question is not asked again on subsequent runs.

Jump host interaction flow:

- Connection resolution always probes the direct SSH connection first
  (`_resolve_ssh_base`). When the direct connection fails and no jump
  host is configured, `_ask_jump_config` interactively asks for the
  jump host and jump username (plain `input()` with no timeout; the
  SSH connection timeouts only start when the probe runs after the
  answers are received). Non-empty answers are saved to `pmd_env.yaml`
  and the connection is re-probed through the jump host (`ssh -J`).
- First-use configuration (`_ensure_deploy_config`) asks the direct
  connection information first (host, username, install path, port),
  then asks the jump host once together with the initial setup — the
  jump host is auxiliary and only revisited at runtime when the direct
  connection fails.
- Before any work starts, `_print_config_banner` prints a prominent
  banner with the active SSH connection parameters (host:port,
  username, jump target or "none (direct)") and the conda environment,
  together with instructions on where to modify them.

Use `--help` or `-h` to display the configuration file path and all configurable variables without running the deployment.

---

# 5. Qualification Modes

## 5.1 Local Qualification

Command:

```bash
pyrai3md test
```

Purpose:

Validate the current PyRAI3MD installation.

Workflow:

```text
pyrai3md test
    |
    v
qualification.runner.cli_entry()
    |
    v
manager.load_config() -> qualification/pmd_env.yaml
    |                   + ~/.pyrai3md/pmd_env.yaml (user override layer)
    v
manager.generate_plan()
    |  (queries skill registry via skills/query.py)
    |  (converts example YAMLs to config/input files)
    |  (generates input files in tmp_YYYYMMDD_HHMMSS/)
    |
    v
runner.run_tests()
    |  (executes all tasks + collects results + prints report)
```

This mode is designed for:

- Installation verification;
- Environment diagnosis;
- Routine development checks.

## 5.2 Remote Qualification

Command:

```bash
python tools/deploy.py
```

Purpose:

Deploy code to a remote server and trigger qualification.

Workflow:

```text
tools/deploy.py
    |
    v
Load qualification/pmd_env.yaml
    |
    v
Check deploy section (interactive if incomplete)
    |
    v
rsync source code to remote
    |
    v
pip install on remote
    |
    v
Trigger: pyrai3md test (on remote)
    |
    v
Periodically poll status via SSH, download deploy_test.log on completion
```

---

# 6. Qualification Planning

## 6.1 Manager

The manager (`qualification/manager.py`) parses `pmd_env.yaml` and generates a qualification plan with input files.

`load_config()` loads the package `qualification/pmd_env.yaml` and deep-merges the user override layer `~/.pyrai3md/pmd_env.yaml` on top of it (only when called without an explicit path), matching the runtime `Environment` semantics: user values take priority over package defaults, and empty user values (`None`, `''`, `[]`, `{}`) count as unset and fall back to the package value (`False` and `0` are valid values and never fall back). Machine-specific settings such as `molcas_path` therefore belong in `~/.pyrai3md/pmd_env.yaml` and take effect for both runtime jobs and qualification tests without touching the package file.

The manager determines:

- Which main framework tests to execute (delegated to `verification/planner.py`, which scans `qualification/verification/` for `test_*.py`; when `test_integration` or `_runall` opens them, the same planner appends the cross-mode/all-example files from `qualification/integration/`);
- Which example qualification tasks to run (discovers `pyrai3md/examples/*.yaml`);
- Which skill tests to run (delegated to `certification/planner.py`, based on the `skills` list);
- Which benchmark calibration tasks to run (delegated to `benchmark/planner.py`, based on the `benchmark` section);
- Queries skill registries via `pyrai3md.skills.query` to obtain `test_method` information;
- Generates corresponding input files (`<test>.json`, `<test>.config`, `<test>.input`, `<test>.xyz`);
- Creates a `tmp_YYYYMMDD_HHMMSS/` working directory for all generated files;
- Which slurm settings to apply (only for `pyrai3md_input` and `pyrai3md_agent` skill modes and the benchmark compute tasks).

The planning itself lives in the three pillar planners (`verification/planner.py`, `certification/planner.py`, `benchmark/planner.py`); `generate_plan()` only aggregates their results into the four plan categories (`main_tests`, `example_tests`, `skill_tests`, `benchmark_tests`).

The manager does not:

- Initialize computational objects;
- Execute calculations.

## 6.2 Skill Test Discovery

Skill test discovery lives in `certification/handler.py` (the former skill_input, dev_tool functionality fully absorbed): the handler locates each skill's `tests/` directory and lists the test case YAML files (`list_skill_test_yamls`) and the runtime test scripts (`list_skill_runtime_tests`, the `test_*.py` unittest/pytest files of the skill itself).

Parsing and config generation moved to `certification/converter.py`: `parse_test_yaml(yaml_path, skill_name)` turns one test YAML into the flat skill config dict (name -> title/project, `current_state` nested under `system`, the declared pair maps rendered into `coupling` and nothing invented for a case that declares none - #447 - skill-agnostic), and `generate_test_configs(skill_name, output_dir)` writes the flat `.json` files for every case of the skill, returning `{profile: path}`.

The certification planner (`certification/planner.py`) drives the chain per test-enabled skill: runtime test scripts are planned **ahead of** the YAML task tests (skill-internal regressions surface before the end-to-end modes run), then each YAML case expands into the three classic modes. A skill with no tests directory produces a diagnostic skip task.

## 6.3 Input File Generation

For each skill test case, the manager generates:

- `<test_name>.json` — Flat skill config written by the certification converter (`generate_test_configs`), consumed by `pyrai3md_skill` (skill mode: every parameter is skill-owned, no skill section);
- `<test_name>.config` — Sectioned dict JSON serialized by the converter (`convert_to_pyrai_config`), consumed by `pyrai3md_agent`; skill parameters live in the full-name skill section (e.g. `xtb_skill`), System fields (atoms/coordinates/nstate/state_spin/...) in the molecule section;
- `<test_name>.input` — Legacy text input for `pyrai3md_input` mode (rendered by `qualification/certification/converter.py` from the same sectioned dict);
- `<test_name>.xyz` — Same-basename coordinate file auto-loaded by the classic input pipeline;
- `<test_name>.velo` — Same-basename plain-numeric velocity file (framework `title.velo` naming convention) written when the test YAML carries inline velocities (dynamics tests; read by `read_coord`);

Suffix contract (#140): data flows strictly one way — `tests/*.yaml -> converter flat .json -> converter sectioned dict -> {.input text, .config JSON}` — with no re-parsing of rendered output; both `.input` and `.config` are produced from the same in-memory sectioned dict, so `pyrai3md_input` and `pyrai3md_agent` consume exactly the same effective configuration. The `.json` file contains `skill_name` for `wrapper.py` to locate the skill agent; `resolve_skill_name` reads it top-level first with an `environment` section fallback, and the converter `load_config` flattens the legacy five-section layout (system/properties/dynamics/calculation/environment, keys promoted to top level without overriding existing ones), so JSON files previously rewritten by the backend write-back remain consumable (#142). The input `.json` is never modified by execution: the backend write-back of the validated config was removed because the local/slurm runners never read the input file back — the runner receives the in-memory config object and slurm node hand-off serializes its own copy into the calculation directory (#143). After generation, the manager injects the merged `skills.<name>` section of pmd_env.yaml (machine-specific parameters such as `molcas_path`) into each `.json` so the standalone `skill_agent` mode — which performs no pmd_env.yaml merge at runtime — can find them, and the converter picks the injected values up for the `.input`/`.config` outputs. Existing non-empty values from the skill test YAML are never overwritten, and the `test` switch is never injected. Execution control keys (`execution_mode`, `slurm_*`) use **overwrite** semantics — the test YAML hard-codes `execution_mode=local` and a fill-only injection would not override it — while all other keys use fill-if-empty semantics (#80). After the environment injection the manager forces `driver: skill` into every skill-test `.json` (unconditional override): skill test cases are skill-driven by definition, so the `pyrai3md_input` mode gains a `driver skill` line in `&CONTROL` (written by the converter) and `pyrai3md_agent` routes through the skill driver; only the main framework examples run `driver=main` (#107). The converter keeps the framework-side `&MD` section and the skill-side section independent: every scalar config key matching the skill schema Calculation keywords (resolved dynamically through the skill's public `query.get_defaults()` interface, no hardcoded key list) is synced into the skill section (`&MOLCAS`/`&BAGEL`/`&PYSCF`), so settings such as `maxit_md`/`temperature`/`size` in the test YAML reach the skill instead of falling back to schema defaults; skill-owned flat int lists (e.g. `target_states`) are written as space-separated tokens to the skill section, while molecule-level lists (ci/spin/ci_roots/ci_spin/coupling/state_spin) stay in `&MOLECULE` (#108, #120). The schema adapter converts the text tokens back to int lists on the way in, so the round-trip YAML → config → classic input → parse → skill config preserves the original list values.

The converter resolves the framework jobtype from the config (`_resolve_jobtype`): `*_surface_hopping` profiles map to the dedicated `surface_hopping` advanced jobtype token (accepted by the framework vocabulary `ADVANCED_JOBTYPES` and dispatched to the AIMD workflow via `WORKFLOW_REGISTRY`/`JOBTYPE_MAP`), so `casscf_surface_hopping` configs no longer raise `ValueError: Unknown jobtype`; the token is part of the converter's advanced set, therefore the generated input carries an `&MD` section. `current_state` (the initial state selected by skill md-type tasks, 0-based) travels the same pipeline: the converter `parse_test_yaml` nests it under the config `system` section, the converter writes `current_state   N` into `&MOLECULE` (reading either the nested `system` section or a top-level key), and the framework input parser loads it into `System.current_state` (#110).

Naming source (#114): the test YAML top-level `name` is the single naming source for the generated task. The converter `parse_test_yaml` injects it into the config `title` and into `config['<skill>_project']` (both falling back to the YAML file stem / respecting an explicit project in the YAML input section); the converter then derives both the `&CONTROL title` and the skill project (`molcas_project`/`bagel_project`/`pyscf_project`, `_derive_project_name` priority `title → profile → 'pyrai3md'`) from this field, so tests sharing one profile (e.g. `casscf_geometry_optimization` vs. MECI/MECP) stay distinguishable and keep uniquely named output files. The skill schema default project names (`molcas_calc`/`PYSCF`/...) and the adapter fill-if-empty logic are unaffected — the converter always writes an explicit project for qualification inputs.

Periodic cell transfer: `_build_molecule_pairs` flattens a non-zero 3×3 `lattice_vector` (top-level or nested `system` section) into the classic `cell` keyword — 9 floats, row-major — written into `&MOLECULE`; the framework normalize step rebuilds `lattice_vector` from it, so both input modes deliver the box down to the skill System submodel. An all-zero or missing box writes nothing and molecular systems stay unaffected (`test_qualification_slurm.py::TestPeriodicCellPipeline`, 4 cases: flattened rendering, nested `system` section, zero-box suppression, and a cp2k YAML end-to-end check down to `System.lattice_vector`).

Optimization inputs: when the resolved jobtype is `optimize`, the converter emits a dedicated `&OPTIMIZE` section (`_build_optimize_pairs`, mirrored as the `optimize` key of the sectioned dict) instead of the `&MD` section used by the other advanced jobtypes.

Multi-center inputs (dev/MULTICENTER_DEVELOPMENT.md section 4): the converter round-trips the center-dimensioned molecule/md keywords (`highlevel`/`midlevel`/`ci`/`spin`/`coupling`/`root`) through both output formats. `_is_multicenter` detects a multi-center config (a classic `;`-separated string, or a nested list one depth beyond the field's single-center form -- note `coupling`'s single-center form is already `list[pair]`, so multi-center is `list[center][pair]`), and `_render_center_string` normalizes each center field to the classic `;`-separated string emitted verbatim into `&MOLECULE`/`&MD` and into the sectioned dict (the agent pipeline re-parses it with the same 1-based/0-based adapter contract, so `.input` and `.config` stay equivalent). Because a multi-center single-shot job (e.g. jobtype `gradient`) still carries its per-center initial state in `md.root`, `_needs_md_section` emits an `&MD` section for it even though the jobtype is not an advanced workflow (the workflow is still resolved from jobtype, so the extra section never turns a single-shot job into dynamics). A multi-theory composite (multi-center ONIOM, two or three theory skills) is preserved per skill: `yaml_to_config` stores each skill's parameter block under `skill_sections` (only when more than one skill is present, so single-skill configs stay byte-identical) with the primary skill following `control.theory`, and both `convert` and `convert_to_pyrai_config` render one section per skill (`&OPENQP` + `&XTB`, `openqp_skill` + `xtb_skill`) through `_build_skill_pairs(config, skill_name, section_params)`, so the high and low level keep their own `method`/`basis` instead of colliding in the flat namespace.

Theory method validation is discovered dynamically (#150, `schema/validate.py`): `_valid_theory_methods()` combines the built-in machine-learning methods (`nn`, `e2n2`) with the program labels registered by the installed skills (`skills/query.discover_all`), so newly added skills become valid theory methods without touching the whitelist; a static fallback keeps validation functional when the registry cannot be discovered.

Adaptive inputs: the reference queue is an independent plan, so the converter carries it as its own key (`control.reference` -> `_token_list` -> `config['reference']`, rendered back as the `reference` pair of `&CONTROL`) rather than reading it off `theory`; `_resolve_jobtype` spells `adaptive` from the `adaptive` section, `_build_adaptive_pairs` renders every `&ADAPTIVE` keyword the YAML declared (`md_*`/`reference_*` stage flags and `remote_train` are ordinary adaptive fields, so the classic carrier gets the same flags the YAML carrier did), and `&MOLECULE` round-trips `highlevel`/`midlevel` together with the reference partition `highref`/`midref` (`_build_molecule_pairs` and the sectioned-dict side both list the four keys). A case that declares none of them renders no `&ADAPTIVE` section at all - the intermediate may not invent a stage flag the user never asked for. This is what the two carriers of one adaptive example are: the YAML carrier keeps the queue in memory, the input carrier re-derives the same plan from the text, and round `tmp_20261106_102745` read the same energy and gradient off both (YAML versus input, `max |difference| = 0.0` on all six adaptive cases).

Generated inputs are stored under the `tmp_YYYYMMDD_HHMMSS/` working directory, organized
in skill subfolders: `tmp_<stamp>/<skill>/` holds one case's generated `.json` / `.config`
/ `.input` and its launch scripts, and each carrier leg runs inside its own subfolder
`tmp_<stamp>/<skill>/<gNN>/<mode>/` (`skill_agent`, `pyrai3md_agent`, `pyrai3md_input`), which is
where a run's own products land -- its `<title>.log`, the ensemble folder a relative
`modeldir` names, an `opt.xyz`. `<gNN>` is the case group's ordinal; the descriptive
group name (`gNN_<modeldir or case>`) stays on `job_group` and in the report, because the
long directory form pushed a Molcas calculation directory past the engine's path limit and
the gateway failed to open the coordinates file it had just been handed (round
`tmp_20261111_162136`). The leg split exists because a skill case chain (`00`
trains the folder `01` serves) is a chain *within* one carrier: with all three legs
sharing one directory, the second leg's training request met the ensemble the first leg
trained there and the skill refused to overwrite it, which is the rule #319 added on
purpose -- so six `nn_pes_skill_*_train_*` cases were decided by the collision rather
than by the calculation (round `tmp_20261010_133424`). Examples keep the same shape one
level up: `tmp_<stamp>/examples/<case>/<mode>/`. A leg directory also holds a copy of
everything already staged in the skill directory except the `.sh` / `.sbatch` launchers,
because a backend resolves its topology by name from the *launch* directory
(`<project>.data`, `<project>.in.init`, a same-basename `.xyz`) and a case names its data
file relatively (`train_data: ./egs_small.json`); without that copy the split costs a
LAMMPS case its input (the 24 `lammps_skill_ff_*` failures of round
`tmp_20261010_181721`).

For the complete input file format specification and configuration priority mechanism, see [INPUT_SYSTEM_DEVELOPMENT.md](INPUT_SYSTEM_DEVELOPMENT.md).

## 6.4 Qualification Plan Structure

```text
TestPlan
├── main_tests:           (one job in a slurm round; unit files from qualification/verification/test_*.py,
│                          plus qualification/integration/test_*.py when test_integration is on)
│   ├── test_agent
│   ├── test_calculator
│   ├── test_core
│   ├── test_deploy_log
│   ├── test_dispatch
│   ├── test_flatten
│   ├── test_integrator
│   ├── test_kernel
│   ├── test_manager
│   ├── test_optimizer
│   ├── test_pmd
│   ├── test_property
│   ├── test_qualification_slurm
│   ├── test_skill_packaging
│   ├── test_strorb
│   ├── test_task
│   └── test_workflow
│
├── example_tests:        (official workflow YAMLs from pyrai3md/examples/)
│   ├── example_<name>_yaml    (pyrai3md <yaml>; its own job in a slurm round)
│   ├── example_<name>_input   (pyrai3md <input> + xyz; its own job)
│   └── md_local / md_parallel / md_batch qualified: the three dispatch topologies run for real, fan-out included
│
├── skill_tests:          (per registered skill: runtime test_*.py scripts
│   │                      first, then per test case from the converter; one job
│   │                      per carrier leg in a slurm round, with a chain's
│   │                      cases kept together inside the leg)
│   ├── <skill>_runtime_<test>  (pytest on the skill's own test_*.py; one job per skill)
│   ├── pyrai3md_input    (local inside the case's job, legacy .input via converter)
│   ├── pyrai3md_agent    (local inside the case's job, sectioned .config via converter)
│   └── skill_agent       (skill manages independently, flat .json from converter)
│
├── benchmark_tests:      (per benchmark/<task>.yaml case, one task per
│                          pmd_env benchmark entry: <task>_<method>)
│   └── benchmark_<case>          (compute: reference && production chained
│                                  in ONE .sh; one job per case in a slurm
│                                  round, after every other category (the
│                                  physics calibrations together, perf alone
│                                  and serial), task-level timeout
│                                  3600 s; the runner then calls the task
│                                  module's evaluate_run
│                                  in-process. Tasks implementing the
│                                  build_legs contract, e.g. perf, chain
│                                  their own leg commands instead)
│
├── slurm: {...}
├── export: {...}         (merged environment variables)
└── work_dir: tmp_YYYYMMDD_HHMMSS/
```

## 6.5 Main Tests

Main framework **unit** tests are unittest files in `qualification/verification/`, which is
their home: the utilities suites (`test_geometric.py`, `test_bonds.py`,
`test_lattice.py`, `test_print_results.py`, `test_structure_formatter.py`,
`test_opt_status_record.py`, `test_timer.py`) lived in
`pyrai3md/utilities/tests/` until CHANGELOG #299 moved them alongside the rest. The manager
scans the directory for `test_*.py` files, so new test files are picked up
automatically without code changes. They always run locally and do not depend
on skills.

Integration tests live in the sibling `qualification/integration/` package:
they load every shipped example or skill and compare the carriers against each
other, so they are slower and are not part of the unit posture. The top-level
`test_integration` switch controls them (`true` runs, `false` skips; a skipped
round folds them into one `main_closed` row naming the switch). The `_runall`
master sits above it: `true` forces integration on, `false` forces it off, and
`condition` lets `test_integration` answer. The direct pytest commands are
`pytest pyrai3md --ignore=pyrai3md/qualification/integration` for the unit
suite and `pytest pyrai3md/qualification/integration` for the integration
suite. No pytest config file is needed: qualification runs both suites
through unittest, and the skill runtime files name their file explicitly and
use no custom marker.

Current test items:

| Test file | Coverage |
|-----------|----------|
| `test_pmd.py` | CLI + parser + schema pipeline |
| `test_agent.py` | Agent + loader |
| `test_calculator.py` | Calculator run/sync, composite aggregation (`aggregate_results`) and queue validation (`validate_plan`), PyRAIState single data source (read-only properties, `update_system`, `set_result`), one-direction calculator→composite dependency |
| `test_core.py` | Data core PyRAIState (history API: bounded snapshots, chronological order, slice-to-last-N, copy semantics) |
| `test_deploy_log.py` | tools/deploy.py: log download (ssh-cat primary, scp fallback, jump-consistent), the work-directory wait, the remote plan probe and the single progress denominator, the stop/liveness pattern, the build clean before install, and the four-category summary (ion) |
| `test_manager.py` | Qualification manager switch coercion (boolean/int/string interpretation), config loading and shell timeout; the all-example planning checks moved to `integration/test_integration_manager_planning.py` |
| Suite size, measured on 2026-11-06 14:20: the verification suite is **3332 tests OK (skipped=1)** in 68 s (3205 when #450 measured this row, 3196 under #448; this round's own additions are `test_manager.py::TestAdaptiveFamilySwitch` and the adaptive carrier criteria, and the rest of the growth between the two readings is not attributed here) under `python3 -m unittest discover -s pyrai3md/qualification/verification -t .` (it was 3182 before #448; the 14 new criteria are the four classes in `test_state_declaration.py`, which now collects 32); `qualification/integration` is **123 passed (no skips, `-m integration`) with 7218 subtests** in 1578 s on this environment (2026-11-06 14:30, the reading taken after this round's two criteria were corrected: the shipped-ensemble check no longer hardcodes six atoms, and the pool-membership list grew from eight sections to eleven when the three `adaptive_*_slurm` carriers joined it. Before that, 123 OK in 1755 s with `skipped=4` on 2026-11-02, and 123 OK in 1884 s with no skip clause on 2026-11-03 - the same suite under three postures); the ten skill suites together are **1567 passed, 2 skipped, 18 subtests** under the repository `.venv` (nn_pes needs TensorFlow, so the same tree read 1563 passed with 1 named nn failure on the system `python3` - the count moved because the module skipped, not because it failed). The whole-tree reading of 2026-10-27 (4495 passed / 1 skipped / 123 integration deselected in 68 s, and the 128-test integration figure from before five unit methods were returned to verification) is that date's measurement and was not repeated. |
| `test_optimizer.py` | `control.driver` execution model: driver validation, planning dispatch, workflow resolution, SingleShot workflow |
| `test_skill_packaging.py` | Skill resource packaging guard: parses setup.py `package_data` via AST and asserts every skill with a manifest declares its manifest/skill_guide/tests resources |
| `test_strorb.py` | Automatic initial orbital channel (`title.StrOrb`): file detection and molcas `guess_orbital` injection, 9 cases |
| `test_task.py` | Task Queue expansion (planning/task.py) |
| `test_workflow.py` | Workflow layer (iterative loop contract: `run()` → `update_system` → `sync()`) |
| `test_property.py` | Planning layer (resolver) |
| `test_kernel.py` | Kernel + maestro |
| `test_dispatch.py` | Dispatch subsystem (runtime plans, executors) |
| `test_grid_plan.py` | Grid scan plan (no TensorFlow): skill declaration surface, the training profile a point stamps, Cartesian order and key reversibility, the member-axis table on a real materialized unit, refusal of an empty grid / missing train_data / multi-skill plan, parent state untouched |
| `test_grid_record.py` | Grid scan read-back: the three result carriers, the TF-free folder read with per-member unscaling, half-product refusal, ranking by max/mean with ties and absent points, the plan and summary tables, `summary.csv`, `crashed.txt` |
| `test_ml_example_input.py` | The three machine-learning examples through both user carriers: `&search grid` survives the conversion to classic text (a dropped section would plan an empty scan and still report a completed run), `&TRAIN` / `&PREDICTION` render, an unstated keyword in a workflow section is refused by name, and a machine-learning jobtype emits no default `&MD` block |
| `test_ml_verdict.py` | ML outcome channel and scan registration: a train/prediction call settled by the record its jobtype promises, non-ML jobtypes unchanged, the verdict banner, `GridSearch` registration and `child_jobtype`, `&search multitask` reaching the graph node, `&search` ranking-vocabulary validation |
| `test_flatten.py` | Config flatten / merge |
| `test_integrator.py` | Integrator tests |
| `test_qualification_slurm.py` | Skill Slurm self-submission qualification (injection semantics, converter skill section, runner artifact assertion, periodic cell pipeline) |
| `test_skill_input.py` | Certification handler: skill tests-directory/YAML/runtime-script discovery and the generic classic-input skill section builder |
| `test_benchmark.py` | Benchmark pillar: flat case discovery (unified `<task>.yaml` + layered resolution), single compute-task planning with the run_dir/benchmark_task handoff, the dict/list benchmark section formats (multi-method calibration, diagnostic skips), the reference && production single-script chain, the perf build_legs planning, fssh task-module input builders, the ktdc task module (verbatim fssh reuse: the same module objects, the mirrored ktdc cases pinning nactype ktdc), the gsh task module (fssh reference reuse, the GSH-only keyword table rejecting fssh-only keywords, the dual-family verdict: trajectory hop statistics against the ZN kernel replay for the tully cases, Landau-Zener deviations for the lz family, the wavepacket reference reported but never judged), the lzsh task module (Landau-Zener surface hopping: fssh reference reuse, the LZ-only keyword table without the gradient-only `activestate` key, the gradient-free kernel replay, the matching hop-statistics verdict), the population plotting helper (the pinned goldenrod/royalblue/crimson curves, the calibration method title, the degraded-table tolerance and the evaluator hook) and the evaluate_run verdict protocol with production.pop/reference.pop persistence and the tools wrapper re-export, `\_workflow\_config\_kwargs` &md wiring |
| `test_perf.py` | Perf benchmark task: the per-backend leg builder (fssh_* kernel pinning, gl_seed, timed commands), the numba-missing diagnostic skip, evaluate_run on a synthetic run directory (wall-time/agreement metrics, PASS/FAIL/ERROR protocol, numpy.pop/numba.pop/perf.json persistence, profile-table parsing) and the timing wrapper (mocked subprocess) |
| `_parallel_helper.py` | Parallel execution helper for verification tests |

New test files are picked up automatically by `verification/planner.py`; the table lists representative coverage rather than every file (md workflow tests such as `test_fssh.py`, `test_fssh_regression.py`, `test_initcond.py`, `test_md_output.py`, `test_trajectory.py` and `test_deviation.py` also live here).

## 6.6 Example Qualification

The `pyrai3md/examples/` directory is the single source of truth for official workflow YAMLs. Qualification never maintains its own example inputs: it discovers the example YAMLs and renders the classic input from them through the converter (the same converter the skill tests use):

```text
example YAML
    |  (qualification/certification/converter.py: yaml_to_config,
    |   parsing via the Interface layer parse_input)
    v
flat skill JSON config (.json)
    |  (qualification/certification/converter.py: shared sectioned dict)
    v
classic text input (.input) + same-basename coordinates (.xyz)
```

Each qualified example is then exercised through the two PyRAI3MD entry points that take a user-authored input file:

| Mode | Command | Input File |
|------|---------|------------|
| pyrai3md_yaml | `pyrai3md <name>.yaml` | Original example YAML |
| pyrai3md_input | `pyrai3md <name>.input` | Classic text input + xyz (converter) |

These are PyRAI3MD runtime qualifications, not skill tests: the task `skill` field is `examples`, and the **task** is started locally (`use_slurm: False` on every example task -- the runner never wraps an example in a batch script of its own). What the run does inside that process is its own declaration: an example whose input asks `remotetask` submits its own child runtimes through the framework dispatch layer, which is a different submission from the wrapper's. The `pyrai3md_agent` and `pyrai3md_skill` standalone modes are deliberately NOT planned for examples — an example is a PyRAI3MD computation task, not a simulated agent-calling-skill scenario, and the agent `.config` is a machine-rendered artifact rather than a user input, so planning it would add a third of the example suite for no extra parser coverage. The agent loader pipeline stays qualified through the skill tests, which run all three modes. The flat `.json` artifact is only the converter intermediate feeding `.input`, never executed.

**A submission the host cannot make is a named skip.** Before the requires gate, the planner reads the example's own input (`manager._example_submits_remotely`: any section declaring a truthy `remotetask`) and, when `shutil.which('sbatch')` finds nothing, records the task as skipped with a reason naming `sbatch` and **claims no run directory for it** -- the gate fires before the mode loop, so the remote stem is one skip record rather than two, and no classic input is generated for a run the test environment cannot submit. An example that never says it submits is a local run whether it fails or is skipped, which is why the request is read from the input and not inferred from a file-name convention: reporting a missing scheduler as a failure would say "the framework is broken" where the sentence should be "the test environment has no cluster client". `test_manager.py::TestExampleChains::test_the_remote_arm_is_a_named_skip_without_a_scheduler` pins both halves, including that the local arm of the same scan is planned in both modes. Where more of the arm needs judging, the rehearsal is a directory of `sbatch`/`sacct` shell shims placed first on `PATH` and nothing ships for it: the run then executes the framework's own dispatch and job-script paths and stops where a real cluster would answer, so what is read back is the `slurm.sh` and `.rt` payload each child runtime left, not a claim about the queue.

**Artifact resolution anchors.** Both checks locate the run's artifacts by the pair (`run_dir`, `title`) the task carries, never by the input file path. This distinction is load-bearing: an example task runs as `cd <work_dir>/examples && pyrai3md <source>/pyrai3md/examples/<name>.yaml`, so its log and trajectory land in the run directory under the `control.title`, and the two names differ (`md_orca_fssh.yaml` titles itself `md_orca_fssh_example`). Resolving beside the input instead finds nothing in either directory, and both checks used to return silently, which meant the NVE conservation verdict never actually ran on any example and a `Status: error` run with exit code 0 passed as well. Skill tests stage their artifacts beside their own config under the config stem, so they set neither key and keep the input-relative anchor. A judged YAML/md/NVE example whose records are missing now leaves a visible `[ENERGY CHECK] no <title>.md.energies under <dir>` trace in the result details instead of passing unjudged in silence: a verification step that cannot find its evidence must say so.

**Qualification policy**: `POSTPONED_EXAMPLES` in `qualification/manager.py` is the postponement list, and every YAML the package ships is planned unless it is named there -- the reference dispatcher examples (`md_local`, `md_parallel`, `md_batch`) were the last three held out, and the round that closed #308 qualified them (`test_manager.py::test_every_shipped_example_is_qualified` fails if a new YAML ships unlisted, and accepts only one kind of exception: the named skip of an example that submits and finds no `sbatch`, whose reason must name `sbatch`). Two examples submit Slurm children of their own -- `md_batch`, whose three trajectories are each a submitted runtime, and `nn_grid_search`, whose four grid points are -- so those are the ones to disable in the `examples` section if the cluster client on the host is the thing that fails; the scan keeps a local-pool twin (`nn_grid_search_local`) precisely so the mechanism stays gated on a machine with no scheduler. The qualified dynamics examples carry `false` switches in the pmd_env.yaml `examples` section (they need bagel/molcas/openqp/orca/lammps/xtb on the host), so they run only when enabled individually or through `_runall`. `multicenter_gradient` and `multicenter_md` exercise the multi-center ONIOM base (two openqp MRSF centers in an xtb environment) and require both external programs installed on the execution host; their classic-input mode is produced by the multi-center / multi-skill converter path documented in Section 6.3.

**Skill topology fixtures**: an example whose backend resolves launch-directory topology (the LAMMPS `<project>.data` water box) has its shared fixtures staged by `manager._stage_example_skill_fixtures`: the `.data`/`.in.init`/`.in.settings` files are copied from each theory skill's `tests/` directory into the example run directory, under both the fixture name and the injected `<stem>_project` name, mirroring the skill-certification staging contract. `.init.xyz` is the exception: it comes from the example's own directory (shipped beside the YAML through `setup.py`'s `examples/*.init.xyz`), is staged under its own name, and is looked up as `<title>.init.xyz` by `workflow/initcond.py`, so an example that names such a file has to find its own topology rather than another system's.

Two sources, staged in this order: what sits beside the example YAML first, then the skill's own fixture only when the run directory still lacks that name. The beside-YAML source is keyed on `_EXAMPLE_FIXTURE_SUFFIXES` (`.data`/`.in.init`/`.in.settings` plus `.json`, `.charge`, `.xyz` and `.freq.molden`) and on the bare name before the first dot, matched either against the names the config declares (the example stem, the title, every `*_project` including the ones nested in `skill_sections`) or against the example's own text - `egs_ethylene_100.json` and `adaptive_ethylene.init.xyz` are named by `train_data:`/`initcond:`, not by a project, and the engine reads them from the launch directory under that name. Staging every file in the shared `examples/` directory into every run is what the two-source rule replaced: round `tmp_20261016_075328` reported `[PBC] pair cutoff 10.000 A exceeds half of the 10.000 A cell` inside a passing xtb leg, because the runner's minimum-image check globs `**/*.in.init` and found a LAMMPS deck the run does not have. A pretrained ensemble is a directory rather than a file, so a `modeldir:` value found in the example text is copied whole with `copytree` under the same basename - the six adaptive examples read `./nn_ethylene_init`, and an example naming a model folder it cannot find fails with the skill's own "train the model first" sentence rather than with an `OSError`.

**Adaptive products**: `_check_adaptive_products` is the third post-run judgement beside the skill-log and energy checks, because an adaptive run returns 0 whether or not it selected, labelled or trained anything. It reads `<title>.adaptive/<title>.adaptive.json` and both `status_iter-<k>.json`, and requires `n_iterations` 2, `ntraj` 5 in each round, iter-2 `reference_ran`/`trained` with at least one new frame, an `iter-2/data_iter-2_*.json` snapshot, `iter-2/reference/new-samples.json` and the `iter-2/NN-<title>-iter-2` model folder; any of them missing downgrades the row to `fail` with `[ADAPTIVE]` and the list of what is missing. A non-adaptive example writes no `.adaptive` directory and the check returns without a word. What the runner does not read, `tools/verify.py` section J does on the fetched directory - the counts that must agree across the two carriers, the aux sidecar rule, the finite-value and alignment readings - so the round verdict and the audit are deliberately different depths: in round `tmp_20261106_102745` every adaptive row passed the checks above, and J's `unexpected_aux 2` named a file the runner never looks at.

The generated agent config carries explicit `jobtype` and `theory` fields derived from the example `control` section, so the converter and the agent pipeline behave identically to skill-generated configs. Example runs are isolated in a `work_dir/examples/` subfolder so scratch outputs never pollute the repository or the caller cwd.

A passing exit code is not the whole verdict: `runner._run_skill_test` inspects the artifacts the run left behind. `_check_skill_log_status` downgrades a run whose own log does not end on the one word its agent writes for a finished calculation (`skills/*/service/utils.py::safe_run` answers 'success' or a traceback, and since #321 anything but 'success' is a failure -- the rule used to look for the words 'failed'/'error', which a traceback walks straight past; six cp2k and molcas cases rode that false green in round tmp_20261010_133424), and `_check_energy_conservation(..., mode=)` reads back the `<title>.md.energies` frame records of a microcanonical example and reports `etot_frames`, `etot_drift` (the spread of the recorded total energy) and, once at least `RATE_MIN_FRAMES` frames exist, `etot_drift_rate` (the least-squares slope per frame, which separates a systematic leak from the bounded velocity-Verlet oscillation). Records are discovered in the task's own run directory and one level down under `<title>_*`, because a fan-out run (`ninitcond > 1`) writes its frames inside one directory per trajectory; the search is deliberately not recursive, so a scratch tree or a leftover directory from an earlier round cannot blend its frames into the worst case being reported, and `.sh.*` hop records are excluded because they are sparse by construction. Only YAML examples whose `md.ensemble` is `nve` and whose `thermostat` is off are judged; a run driven by a thermostat or a barostat is not, and since #321 it says so in the record (`[ENERGY CONSERVATION] not judged under {run_dir}: this run is driven by ensemble nvt and thermostat ...`) instead of returning silently -- before that the branch left no trace, so an NVT example read exactly like one whose energy had been checked and passed. A `mode == 'benchmark'` leg is never judged by this rule -- its verdict is the reference-population comparison below, and it says so in the note instead of printing a missing-records line, because a coupling does work on the nuclei and Etot is meant to move there; a multicentre run contributes one record per center as well (`workflow/records/output.py::_write_center_files`), and those are read but **not judged**: what a center's own file holds is that cluster's energy plus its own atoms' kinetic energy, and the composite propagates the cluster with forces coming from the whole system, so a center is not a closed subsystem and only the trajectory total is a constant of the motion. The widest per-center band is reported as `etot_center_drift` for reading, never as a verdict -- the rule before #320 took the worst value over every file, which failed `examples/multicenter_pbc_nve.yaml` on center 1's 8.82e-03 Hartree of exchange while its composite slope was 2.906e-06 Hartree per frame against a 1.0e-05 tolerance. Tolerances come from the pmd_env.yaml `energy` section (defaults `etot_drift` 1e-3, `etot_drift_rate` 1e-5 Hartree) and a violation turns a passing example into a failure.

The other half of that sentence is `_check_npt_stability`, added because the eight NPT failures of `tmp_20260928_184646` were invisible to a suite whose only trajectory judgement deliberately skipped anything with a barostat: four examples passed while their boxes deformed by 8 to 57 percent, one of them ending with a 1.66 Angstrom axis. It reads the per-frame `Lattice="..."` comment of `<title>.md.xyz` and the `T_inst(K)` column of `<title>.md.ensemble` and judges `volume_drift` (max |V/V0 - 1|, fail above 0.25) and `temperature_drift` (max |T_inst/T_target - 1| over every frame but the first, fail above 3.0), while `cell_rotation_rad` -- the accumulated polar turn of the simulation frame -- is reported but does not decide, because a healthy anisotropic run turns 5.7e-5 rad per step and does not cancel, so any fixed bound on the accumulated angle separates long runs from short ones rather than sick runs from healthy ones (`dev/methods/ENSEMBLE_DEVELOPMENT.md` section 8.2). Tolerances come from `box` and note thresholds from `box_notes` in pmd_env.yaml; a violation downgrades a pass, a wide-but-legal run leaves the three numbers in `details`, and a run with fewer than two cell frames or no trajectory at all says so instead of passing silently. Judged against the products of that run, three of the four sick-but-passing examples become failures (`nh_mttk` at volume_drift 0.575, `csvr_berendsen` and `nhc_berendsen` at temperature_drift 1.75/1.74) and `nh_berendsen` (0.116) stays a pass with its numbers written down. The verdict stays quiet for a run that already failed: a dead engine leaves a truncated trajectory, and reading that as a divergence would be double-counting one death as two. Which of the two decides depends on the record length: below `RATE_MIN_FRAMES` there is no slope to judge and the spread carries the verdict, while from 50 frames up the slope decides and the spread is reported beside it (`judged on N frames: etot_drift_rate ... against tolerance ...; recorded etot_drift (max-min) ...`), because the spread of a long run is dominated by bounded oscillation - the shipped 20-water NVE example shows a 2.004e-03 Hartree band that is the same width in its second half as in its first, against a slope of 3.360e-08 Hartree per frame. This is the check that catches a backend returning state energies on an inconsistent reference, which leaves every state energy difference correct and so is invisible to every other test.

**Shipped example inventory (retired list, kept as the record of why each case exists).** The `QUALIFIED_EXAMPLES` tuple below was the policy list before the plan became disk-derived: it was exhaustive, so adding a YAML under `pyrai3md/examples/` needed a Python edit. The policy is now a *postponement* map (`POSTPONED_EXAMPLES`, empty), and every shipped example is planned unless it is named there. The annotations carry the measured readings of the rounds that introduced the cases, so they are kept verbatim here rather than deleted:

```python
QUALIFIED_EXAMPLES = (
    'energy', 'gradient',
    # The framework's own numerical Hessian route end to end: 19 gradient
    # jobs of displaced water, the mobile-block assembly, the rank-based
    # rigid-mode removal, frequencies/molden/thermo. `jobtype: freq` takes
    # the analytic engine route whenever PySCF can serve it, so without this
    # case the numerical path has no real-engine reading at all. Measured
    # against that analytic sibling on one host: 2043.0734 / 4488.0646 /
    # 4790.2488 against 2043.3249 / 4488.0061 / 4790.1960 per cm-1, a
    # largest difference of 0.26 cm-1 at the 0.005 A step.
    'hessian_num',
    'optimize_minima', 'optimize_gdiis', 'optimize_tda',
    # The saddle search on both curvature producers and on a single-center
    # composite: 32 evaluations each on HF/sto-3g HCN, reaching
    # -91.5650982833 / -91.5650982834 / -91.5650982833 Hartree, with the
    # endpoint's own frequency job reporting one imaginary mode (dev/methods/
    # OPTIMIZE_DEVELOPMENT.md section 15.2).
    'optimize_ts', 'optimize_ts_skill_hess', 'optimize_ts_multiscale',
    # A relaxation with a pinned fragment: the step, the recorded gradient the
    # curvature update is fitted to and the trust radius all read the tangential
    # part (#337). Converges in 27 evaluations at -91.6424015 Hartree against 23
    # at -91.6446781 with the constraint block removed; before #337 the pinned
    # run did not converge at all.
    'optimize_partial_minima',
    # A saddle search with one atom held by the anchor: 75 evaluations to
    # -149.8362656686 Hartree on HF/sto-3g water dimer, one imaginary mode
    # (-3064.0 cm-1) on the endpoint's own frequency job, and the held oxygen
    # returned where it was put (#338).
    'optimize_partial_ts',
    'multicenter_gradient', 'multicenter_md',
    # A PySCF center inside a LAMMPS box, on the geometry that puts the water
    # across a box face: the region the center is cut from comes out of the
    # continuous view rebuilt from the frame's crossing counts, and these two
    # are the real-engine reading of that path (dev/methods/
    # PBC_DEVELOPMENT.md section 4.2). They need PySCF *and* LAMMPS on the host,
    # and the shape is the one a single LAMMPS topology can serve: the center
    # covers the whole system, because a LAMMPS low layer cannot be handed a
    # slice smaller than its `<project>.data` template.
    'multicenter_pbc_nve', 'multicenter_pbc_npt',
    # The same two-center box read as QMMM: Molcas CASSCF per center, the
    # field of each center carrying the other center's 0-order charges, one
    # classical LAMMPS surface over all sixty. `sp` is the fixed-round
    # reading, `sc_charge` the iterated one, `md` the per-step rebuild.
    # They need Molcas *and* LAMMPS on the host.
    'multicenter_qmmm_sp', 'multicenter_qmmm_sc_charge',
    'multicenter_qmmm_md',
    # The two channels built on top of that evaluation: a band whose every
    # image is one, and a numerical Hessian whose every gradient job is one.
    'multicenter_qmmm_mep', 'multicenter_qmmm_num_freq',
    # The coupling channel: each center answers its own NAC on its own atoms
    # and the composite keeps the blocks apart (nothing is summed across
    # centers), so this is the per-center reading of a two-ladder run.
    'multicenter_qmmm_nac_sp',
    # The two channels that move the box under those centers: NPT steps it from
    # the whole-system stress inside the engine, `cell` steps its six entries
    # against the same stress from the framework. Both rebuild the composite at
    # the box of the step, so they are the family's variable-cell legs.
    'multicenter_qmmm_npt', 'multicenter_qmmm_cell',
    # The checkpoint pair: `..._restart` runs two steps and writes `<title>.rt`,
    # `..._restart_base` runs the same four in one process. The round itself
    # executes one process per leg; the resume half of the split is the second
    # line of the case's own header, and the parity reading is in
    # `dev/methods/QMMM_DEVELOPMENT.md` section 9.
    'multicenter_qmmm_md_restart', 'multicenter_qmmm_md_restart_base',
    # The band carrier: three images between the endpoints the single-center
    # `multiscale_qmmm_neb.yaml` uses, with each center a water the endpoints
    # actually move.
    'multicenter_qmmm_neb',
    # The three channel carriers that close the matrix: each center's own pair
    # under a crossing search (MECP - a MECI's branching plane is not a
    # per-center object), one FSSH ledger per center with the engine's
    # couplings, and an excited-state minimization on the per-center ladder.
    'multicenter_qmmm_mecp', 'multicenter_qmmm_md_fssh',
    'multicenter_qmmm_ex_opt',
    # The ONIOM counterparts of the multi-center QMMM carriers: two ORCA
    # high-level centers with a quantum outer/mid (PySCF HF). The two-layer
    # form embeds each center in the L(real) row of every atom outside its
    # own highlevel; the three-layer form embeds it in its own midlevel
    # before a LAMMPS low layer. All four are local (ORCA+PySCF, LAMMPS).
    'multicenter_oniom2_orca_embedding_sp',
    'multicenter_oniom2_orca_no_embedding_sp',
    'multicenter_oniom2_orca_embedding_md',
    'multicenter_oniom2_molcas_embedding_nac_sp',
    'multicenter_oniom3_orca_embedding_sp',
    'multicenter_oniom3_orca_embedding_npt',
    # The charged counterparts: an ammonium (+1) and a neutral water centre,
    # with the real layer at +1 and each centre's tasks at its own charge.
    # The two-layer form takes the L(real) row in both environments; the
    # three-layer form keeps ORCA high and low with a PySCF midlevel.
    'multicenter_oniom2_orca_charged_sp',
    'multicenter_oniom2_orca_charged_no_embedding_sp',
    'multicenter_oniom2_orca_charged_md',
    'multicenter_oniom3_orca_charged_sp',
    'multicenter_oniom3_orca_charged_no_embedding_sp',
    'multicenter_oniom3_orca_charged_md',
    'md_bagel_fssh', 'md_molcas_fssh', 'md_openqp_fssh', 'md_orca_fssh',
    # The molcas NAMD loop with the engine answering the couplings
    # (`nactype: nac`) instead of kTDC, on three pairs declared in a non-sorted
    # order. The framework indexes the returned rows by that declaration, so
    # this is the trajectory-level reading of row identity: a single-pair leg
    # cannot tell a remap from an identity map (#448).
    'md_molcas_fssh_nac',
    # The same loop on a ladder that spans two spin manifolds
    # (`ci: "2 1"` / `spin: "0 1"`). Every step's `energy gradient`
    # resolves to `casscf_gradient`, which #449 routes through the
    # grouped builder, so this is the trajectory-level reading of the
    # mixed-spin energy+gradient channel (#21).
    'md_molcas_fssh_twospin',
    # The two-pass QMMM deck (ESPF pass + field-free NAC pass) on the
    # mixed ladder: the framework-native carrier for the shape #449
    # writes `&ESPF` once per manifold in.
    'multiscale_qmmm_embedding_nac',
    # The same NAMD loop with &md dosoc on: each step is a two-manifold
    # runtype=data run that returns energies, gradients and the singlet-
    # triplet couplings together. Opened with a measured 5-step run on
    # OpenQP 1.3.1; it stays switched off by default because 1.2 writes no
    # `socme` for the reader to find.
    'md_openqp_fssh_soc',
    # The three dispatch topologies the runtime was built around: a serial
    # local run, a thread pool on the local host, and a Slurm fan-out. They
    # were held out while the Dispatcher and Workflow runtime settled, and
    # the round that closed #308 took them in -- md_batch is the only one
    # that needs a cluster client on the host.
    'md_local', 'md_parallel', 'md_batch',
    # The initial-velocity control (dev/methods/MD_DEVELOPMENT.md section 14):
    # the excess pair states one kinetic energy two ways, the randvelo case is
    # the temperature-only start that needs no frequency record, and the reset
    # case is the rigid-rotation half that a periodic box must not take. All
    # four need nothing but PySCF on the host.
    'md_velocity_excess', 'md_velocity_excess_base',
    'md_velocity_randvelo', 'md_velocity_reset',
    # The dissipation gate: an NVT run whose thermostat starts only after the
    # trajectory has been on the ground state for `thermodelay` steps.
    'md_dissipate',
    'md_lammps_nve',
    'md_lammps_nvt_nh', 'md_lammps_nvt_nhc', 'md_lammps_nvt_csvr',
    'md_lammps_npt_nh_berendsen', 'md_lammps_npt_nh_pr',
    'md_lammps_npt_nh_mttk', 'md_lammps_npt_nhc_berendsen',
    'md_lammps_npt_nhc_pr', 'md_lammps_npt_nhc_mttk',
    'md_lammps_npt_csvr_berendsen', 'md_lammps_npt_csvr_scr',
    # The constraint subsystem's functional cases (dev/methods/
    # CONSTRAINT_DEVELOPMENT.md section 20): 14 on PySCF, 25 on LAMMPS, over
    # the two boundary conditions, both engines' provider channels, the
    # nve/nvt/npt ensemble axis, the restart pair and the three jobtypes
    # (md, optimize, gradient). Like the ensemble set they are qualified but
    # switched off by default in pmd_env.yaml, because they need the engine
    # the host either has or has not.
    'constraint_md_shake', 'constraint_md_rattle', 'constraint_md_freeze',
    'constraint_md_bias', 'constraint_md_metad', 'constraint_md_metad_rmsd',
    'constraint_md_torsion', 'constraint_md_torsion_intermolecular',
    'constraint_md_wall_nonsphere', 'constraint_md_restart',
    'constraint_md_restart_base', 'constraint_wall_center',
    'constraint_opt_surface', 'constraint_sp_bias',
    'constraint_md_periodic_shake', 'constraint_md_periodic_freeze',
    'constraint_md_periodic_bias', 'constraint_md_periodic_wall',
    'constraint_md_periodic_wall_box',
    'constraint_md_periodic_wall_ellipsoid',
    'constraint_md_periodic_wall_triclinic',
    'constraint_md_periodic_torn_angle',
    'constraint_md_periodic_metad_gaussian',
    'constraint_md_periodic_metad_rmsd', 'constraint_md_periodic_restart',
    'constraint_md_periodic_restart_base',
    'constraint_md_periodic_nvt_shake', 'constraint_md_periodic_nvt_freeze',
    'constraint_md_periodic_npt_shake',
    'constraint_md_periodic_npt_shake_slow', 'constraint_md_periodic_npt_angle',
    'constraint_md_periodic_npt_bias', 'constraint_md_periodic_npt_freeze',
    # The variable-cell family (#355): `optimize_cell` had no shipped run at all,
    # so the cell columns the optimizer prints were never read off a real
    # qualification product. `optimize_cell_lammps` is the acceptance vehicle
    # (a 25 A TIP3P box whose energy is smooth in its own cell, measured at
    # 6.8 s for its 12 recorded steps); `_constrained` adds the held bond, the
    # held angle and the pinned atom riding the same box; `_xtb` and `_cp2k`
    # put an electronic-structure tensor under the same keyword - cp2k on an
    # excited state, which is the case the skill's stress-route resolution
    # exists for. The last two carry no local timing: neither engine is
    # installed on this environment, so their cost is a round reading, and `maxit`
    # stays at 4 until one exists.
    'optimize_cell_lammps', 'optimize_cell_lammps_constrained',
    # The angle-only counterpart of `_constrained`: one held H-O-H angle and a
    # pinned atom, no bond term, so the angle's own cell term is exercised in
    # isolation. Functional test at `maxit: 6`; the converged box position for
    # an angle is not claimed (dev/TODO.md).
    'optimize_cell_lammps_angle',
    'optimize_cell_xtb', 'optimize_cell_cp2k',
    # The pressure arm of the same keyword: `opt_pressure` plus
    # `opt_cell_coupling: isotropic` on the ethylene-in-a-10-A box, so the
    # objective is an enthalpy and the box is allowed only a shape-preserving
    # step. Its cost is a round reading too: the engine is not on this environment.
    'optimize_cell_pressure',
    # The two arms #365 and #373 opened. `_ts` puts a saddle search under the
    # same keyword, which is what makes the numerical batch ask for twelve
    # strained lattice evaluations instead of refusing the cell half by name;
    # `_meci_multiscale` is the gap arm, where the centre (pyscf, an excited
    # manifold) supplies the two surfaces and the outer layer (xtb) supplies the
    # stress that moves the box. Both are xtb-priced on the ethylene-in-a-10-A
    # box, so their cost is a round reading like the four above.
    'optimize_cell_ts', 'optimize_cell_meci_multiscale',
    # The ethylene-in-water box (81 atoms: one C2H4 pinned at the centre of a
    # 10.5 A cube, 25 H2O around it) and the three things it is asked to show.
    # `_gdiis` puts the DIIS search on a lattice (gdiis.py:63, 65-72 passes the
    # cell arguments through; it has no TS counterpart, so `method: ts` and
    # gdiis is not a combination the registry offers). `_meci` and `_mecp` are
    # the crossing pair on that box, centre pyscf / outer xtb, with the
    # centre's basis at 631gs because the same region at sto3g returns its two
    # lowest roots 0.0016 eV apart - the search would start on the crossing it
    # is meant to find. `_aniso` names the coupling the other cases inherit.
    # Costs are round readings: xtb is not on this environment.
    'optimize_cell_gdiis', 'optimize_cell_meci', 'optimize_cell_mecp',
    'optimize_cell_aniso',
    # The same pressure the isotropic arm asks, on a box whose six components
    # are free: under `anisotropic` the two `Cell Stress` rows are the rms and
    # maximum over all six components of `sigma + p I` (the mean-normal-stress
    # overwrite at `optimizer/base.py:533-546` belongs to the coupled branch
    # only), so this is the first case in the set where the off-diagonal part
    # of the pressure residual is judged rather than printed.
    'optimize_cell_pressure_aniso',
    # The same box with no load on it. `_check_cell_pressure`'s second ask reads
    # direction by comparing a pressurized drift with what this box does freely,
    # and the pair is matched on engine list, optimizer, declared coupling and
    # starting volume (`runner.py::_natural_motion_of_the_same_box`). Round
    # tmp_20261018_200317 supplied that reference for the anisotropic arm, out of
    # `optimize_cell_aniso`, and printed `paired against nothing` for the
    # isotropic one; this case is the second half of the pair, so a 1700 bar
    # claim under a shape-preserving step is measured rather than asserted.
    'optimize_cell_iso',
    # The crossing pair on a force-field environment: 1581 atoms of TIP3P water
    # with one ethylene as the centre (pyscf TDA, rows 1576-1581), so the layer
    # that carries the lattice is classical and has a topology of its own - the
    # join no other cell case makes (`optimize_cell_lammps` has no centre, the
    # four electronic gap cases have no force field). Their cost is the one thing
    # measured rather than owed: both legs finish their five steps in under a
    # minute on this environment, which carries `lmp_serial` and PySCF.
    'optimize_cell_meci_lammps', 'optimize_cell_mecp_lammps',
    # The gap arm on ONE program, where the strain batch hands the search a
    # tensor per state and the crossing objective therefore has a lattice
    # derivative of its own: the `state-resolved stress` source label, read off
    # a real engine for the first time. Thirteen CP2K calls per step, yet round
    # tmp_20261019_200652 measured both of its legs at 735 s and 728 s inside
    # the shared 1200 s budget, so it answers to that budget like every other
    # example (the per-example map it once used was removed on 2026-10-20).
    'optimize_cell_gap_cp2k',
    'constraint_md_periodic_npt_wall', 'constraint_md_periodic_dihedral_ethanol',
    'constraint_md_periodic_xh_ethanol', 'constraint_opt_periodic',
    # The machine-learning trio. `nn_train` and `nn_predict` are one chain
    # (see EXAMPLE_CHAINS): the fit writes the model folder the test-set
    # evaluation reads, and no other example carries that state.
    # All three need the optional TensorFlow extra, which is now checked
    # at plan time and reported as a skip naming the package.
    'nn_train', 'nn_predict', 'nn_grid_search', 'nn_grid_search_local',
    # The path family: `optimize.method` mep / irc / neb over the same three-atom
    # HCN system the saddle examples use, plus three composite arms, two periodic
    # bands on LAMMPS, eight xtb cases, and the composite on a moving box. All
    # thirty-one of the first set ran in both carriers on
    # the test environment in round tmp_20261022_084519: **62 example rows, 62
    # pass**. The artifacts were read back afterwards (`tools/verify.py`):
    # section F held 14 condition legs over 148 frames and 4 frozen legs over 34
    # (0 broken), matched 6 composite legs to their single-layer counterparts,
    # and section D matched 10 variable-cell legs' frame boxes against 550 rows.
    # That review also found two defects since fixed (the IRC `both` branches
    # shared one endpoint; `path_irc_pbc` walked the side that climbs), so the
    # next round rereads those rows. `path_cell_mep_multiscale` is the
    # thirty-second: the same composite with an xtb environment, whose
    # whole-system stress is what conjugates the box.
    # A name here is the postponement policy's decision, not a claim about a
    # reading - and an example left out of this list is skipped whatever its
    # switches say, which is why the thirty-two are listed rather than left to
    # the prefix.
    'path_mep', 'path_mep_relu', 'path_mep_bond', 'path_mep_angle',
    'path_mep_torsion', 'path_mep_freeze', 'path_irc', 'path_irc_mode',
    'path_irc_bond', 'path_irc_relu', 'path_neb', 'path_neb_ts',
    'path_neb_geodesic', 'path_neb_bond', 'path_neb_angle',
    'path_neb_torsion', 'path_neb_freeze', 'path_cineb',
    'path_mep_multiscale', 'path_irc_multiscale', 'path_neb_multiscale',
    'path_cell_mep_multiscale',
    'path_mep_pbc', 'path_neb_pbc', 'path_neb_pbc_geodesic', 'path_irc_pbc',
    'path_neb_xtb', 'path_cell_mep', 'path_cell_mep_relu', 'path_cell_irc',
    'path_cell_cineb', 'path_cell_neb_geodesic',
    # Multiscale composite cases (family switch `multiscale_test`).
    # ONIOM2 (openqp high + xtb low): embedding false, computed outer
    # charges, and static read_charge; energy+SOC and 5-step NAMD.
    'multiscale_oniom2_no_embedding_sp',
    'multiscale_oniom2_embedding_sp',
    'multiscale_oniom2_read_charge_sp',
    'multiscale_oniom2_no_embedding_md',
    'multiscale_oniom2_embedding_md',
    'multiscale_oniom2_read_charge_md',
    # ONIOM3 (openqp + xtb + LAMMPS on a periodic water box): same three
    # charge treatments; the midlevel is the embedding source.
    'multiscale_oniom3_no_embedding_sp',
    'multiscale_oniom3_embedding_sp',
    'multiscale_oniom3_read_charge_sp',
    'multiscale_oniom3_no_embedding_md',
    'multiscale_oniom3_embedding_md',
    'multiscale_oniom3_read_charge_md',
    # QMMM (Molcas XMS-CASPT2 + LAMMPS): field V/E embedding and the
    # isolated-QM two-task form; energy+SOC and 5-step NAMD.
    'multiscale_qmmm_embedding_sp',
    'multiscale_qmmm_no_embedding_sp',
    'multiscale_qmmm_embedding_md',
    'multiscale_qmmm_no_embedding_md',
    # QMMM optimize/path copies (Molcas high + LAMMPS low, electrostatic):
    # the same algorithms the ONIOM optimize/path family carries, re-hosted
    # on shipped LAMMPS systems (c2h4_water25, h2o, qmmm_cluster_test).
    'multiscale_qmmm_meci',
    'multiscale_qmmm_mecp',
    'multiscale_qmmm_mecp_cluster',
    'multiscale_qmmm_ts',
    'multiscale_qmmm_mep',
    'multiscale_qmmm_neb',
    # ONIOM point-charge embedding (ORCA / BAGEL high in xTB GFN-FF
    # charges): one shared set of algorithms, split between the carriers.
    'multiscale_oniom2_orca_charge_meci',
    'multiscale_oniom2_orca_charge_mep',
    'multiscale_oniom2_bagel_charge_mecp',
    'multiscale_oniom2_bagel_charge_ts',
    # ONIOM true-embedding cases: ORCA (and one Molcas) high-level consumers
    # with PySCF as the charge source, so the electrostatic shift is
    # measurable. The `no_embedding` twins are the controls.
    'multiscale_oniom2_orca_embedding_sp',
    'multiscale_oniom2_orca_no_embedding_sp',
    'multiscale_oniom2_orca_embedding_md',
    'multiscale_oniom2_molcas_embedding_nac',
    'multiscale_oniom2_orca_embedding_meci',
    'multiscale_oniom2_orca_embedding_neb',
    'multiscale_oniom2_molcas_embedding_sp',
    'multiscale_oniom2_molcas_no_embedding_sp',
    # The three-layer periodic legs: PySCF mid shell, LAMMPS low layer.
    'multiscale_oniom3_orca_embedding_npt',
    'multiscale_oniom3_orca_embedding_cell',
    # Adaptive sampling: one method and the two mechanical multiscale
    # schemes, each in a local-stage and a Slurm-stage carrier. Every case
    # runs two iterations and five trajectories; the pretrained ethylene
    # model and the c2h4 + 25-water box are shared fixtures.
    'adaptive_single_local',
    'adaptive_single_slurm',
    'adaptive_oniom2_local',
    'adaptive_oniom2_slurm',
    'adaptive_qmmm_local',
    'adaptive_qmmm_slurm',

)

```

## 6.7 Skill Tests

For each registered skill the certification planner first plans the skill's runtime test scripts (`test_*.py` under `<skill>/tests/`, executed in place by pytest), then expands each test case (from `certification/converter.generate_test_configs`) into three test modes:

### pyrai3md_input

Traditional CLI test:

```bash
pyrai3md <test_name>.input
```

The `.input` file is generated by `qualification/certification/converter.py` from the test `.json` config.

Verifies: Converter → Schema Processing → State Construction → Kernel Execution

Uses slurm if configured.

### pyrai3md_agent

Agent CLI test:

```bash
pyrai3md_agent <test_name>.config
```

Verifies: Agent Interface → State Construction → Kernel Execution

The `.config` file is the converter's sectioned dict serialized as JSON (same sectioned dict the `.input` text is rendered from), so agent mode consumes the same effective configuration as input mode.

Uses slurm if configured.

### skill_agent

Independent component agent test:

```bash
pyrai3md_skill <test_name>.json
```

The `pyrai3md_skill` command invokes `wrapper.py`, which resolves `skill_name` from the config (top-level first, `environment` section fallback, #142) and executes:

```bash
python -m pyrai3md.skills.<skill_name>.agent <input.json>
```

The skill manages its own execution independently. No slurm involvement. This verifies the skill's standalone execution capability. Because this path performs no pmd_env.yaml merge, machine-specific parameters (e.g. `molcas_path`) are injected into the `.json` file by the planner at plan time (see the generated-file list above).

---

## 6.8 Benchmark Tests

Benchmarks are application-level algorithm calibrations executed after every functional test (the fourth plan category, `benchmark_tests`). The layout is flat, at the same level of directory management as certification and verification: one single-script module per benchmark task type, named after the task (`fssh.py`), paired with the unified case YAML `<task>.yaml` (`fssh.yaml`) plus optional dedicated `<task>_<method>.yaml` files for model-specific cases. The task module owns the complete calibration task; adding a new method of an existing task usually only changes the pmd_env benchmark entries, no new YAML file.

Each discovered case is isolated in `work_dir/benchmark/<case_key>/` and expands into ONE compute task (generated input files + one execution script):

1. **Compute task** — the reference input and the production input (both framework YAMLs, both executed by the regular `pyrai3md` entry point) are generated at plan time through the task module's builders and live in the run directory as `reference.yaml` / `production.yaml`. The reference input uses `control.driver: skill` + `jobtype: md`, which routes through SingleShot to the qdm skill's public `{method}_reference` profile (`qdm_project: reference` names its output files). The two runs are chained into **ONE script** with `&&`, identically in the local `.sh` and the slurm `.sbatch`:

   ```bash
   cd <ref_dir>  && pyrai3md <run_dir>/reference.yaml && \
   cd <prod_dir> && pyrai3md <run_dir>/production.yaml
   ```

   so production can only start after the reference succeeded, whether the script runs locally or inside a job. The script is a local `.sh`; in a slurm round the runner places this case in one job (after every other category; the perf timing case runs alone and serial) and the planner writes no per-case `.sbatch`. The compute task carries a task-level timeout (3600 s by default).

   Run-directory layout (identical for local and slurm dispatch):

   ```text
   work_dir/benchmark/<case_key>/          # run directory
   ├── <case_key>.yaml                     # merged case YAML (evaluation input)
   ├── reference.yaml / production.yaml    # framework YAMLs (pyrai3md entry)
   ├── benchmark_<case_key>.sh             # compute chain script (the only .sh)
   ├── production.pop / reference.pop      # written by the evaluation tool
   ├── population_s0.png                   # rendered by the plotting helper
   ├── reference/
   │   ├── <title>.log
   │   └── tmp_QDM/reference_population.dat (+ norm/grid/density)
   └── production/
       ├── benchmark_<case_key>.log
       └── benchmark_<case_key>_{1..ninitcond}/   # one directory per
           ├── benchmark_<case_key>.log           # trajectory, named
           ├── benchmark_<case_key>.md.* + sh.*   # '{title}_{idx}'
           └── tmp_QDM/                           # trajectory calcdir
   ```

   The trajectory directories are owned by the md dispatch layer: the AIMD fan-out names them `{title}_{idx}` next to the production run files, places each trajectory's calculation directory (`tmp_QDM`, no runtime suffix) inside it, and applies the same layout whether the trajectories run locally (`remotetask: false`) or through slurm (`remotetask: true`).
2. **Evaluation** — a qualification tool, not a framework calculation task: right after the compute task finished, the runner calls the task module's `evaluate_run(run_dir)` in-process. It parses the reference and trajectory populations, ensemble-averages the trajectories, computes the metrics against the tolerances defined in the case YAML, persists the averaged production population and the reference population as `production.pop` / `reference.pop` into the run directory, renders the S0 population plot (`population_s0.png`, see *Population plotting* below) and reports the verdict (PASS/FAIL/ERROR). Missing or broken compute output yields verdict ERROR instead of crashing silent; a non-PASS verdict is recorded as a fail in the qualification report. When `pyrai3md test` executes the calibrations, the runner completes the entire calibration pipeline in-process — planning, compute, analysis and plotting — and never calls `tools/benchmark.py`: that wrapper is the external one-way analysis entry point for secondary verification only (re-analyzing an already-computed run directory; it never launches calculations). It re-exports the same entry points (external tools directory, next to `tools/deploy.py`; import path `pyrai3md.qualification.benchmark.fssh`) and owns the command-line entry point (the task module itself is programmatic only), e.g. `python3 ./tools/benchmark.py <run_dir>`.

Per-task settings live in the `benchmark` section of pmd_env.yaml (one subsection per task module, a single parameter dict or a list of them for multi-method calibration): every entry's `method` key selects the Tully model (injected into the unified case YAML; a dedicated `<task>_<method>.yaml` wins when shipped) and the remaining fields override the case parameters at plan time (the merged case YAML is written into the run directory under the case key so the evaluation tool sees the identical settings). A missing section keeps every shipped case enabled. Currently shipped: the unified `fssh` case (fewest-switches surface hopping on the Tully models tully1/2/3, calibrated against the exact split-operator quantum reference generated independently at benchmark time; the shipped defaults calibrate tully1), the `ktdc` case (the identical FSSH calibration with the energy-only curvature coupling NAC source, see below), the `gsh` case (Zhu-Nakamura global surface hopping on the same Tully models, see below), the `lzsh` case (Landau-Zener surface hopping on the same Tully models plus the dedicated lz family, see below) and the `perf` case (numpy vs numba computational efficiency comparison, see below).

### The unified final-state metric set

Every benchmark evaluator (fssh, ktdc, gsh and lzsh) delegates to the shared `fssh.evaluate_unified` and reports the identical metric layout: five physical quantities plus up to three judged differences, skipping whatever the case does not carry.

- **Reported quantities** — `final_trajectory_population` (the fraction of trajectories whose final active state is state 1), `final_state_population` (the final S1 population of the ensemble-averaged electronic block), `final_reference_population` (the final S1 probability of the exact wavepacket reference), `internal_hopping_probability` (the deterministic kernel replay probability, every gsh/lzsh case: the lz family reports the first non-zero gate probability, the gsh tully cases replay on numerically differentiated recorded gradients, the lzsh tully cases take the maximum per-step probability of the replay window) and `analytical_hopping_probability` (the analytic single-passage Landau-Zener probability, lz model cases only). The probability quantities themselves are reported without a tolerance: the runner prints the bare value, while the judged differences print their tolerance and per-metric verdict.
- **Judged differences** (each against its case YAML tolerance) — `state_reference_difference` (every case), `state_internal_difference` (every gsh/lzsh case, whenever the kernel replay supplies an internal probability) and `state_analytical_difference` (lz cases only). A comparison target the calibrated model or method does not carry produces no metric entry at all, so the runner never prints it.
- **Case matrix** — fssh/ktdc tully: reference difference only; fssh/ktdc lz: reference + analytical; gsh/lzsh tully: reference + internal; gsh/lzsh lz: reference + internal + analytical. FSSH/kTDC carry no kernel replay, and the tully models carry no analytic hopping probability by construction.
- **Tolerances** — all shipped tolerances are placeholder 1.0 (guaranteed PASS) pending calibration from the benchmark results; the verdict is PASS only when every judged difference stays within its tolerance.

### The perf case (numpy vs numba backend comparison)

The `perf` task module (`perf.py` + `perf.yaml`) measures the numpy vs numba computational efficiency of the FSSH electronic propagator on the same Tully model the accuracy case uses. It implements the **`build_legs` contract** instead of the reference/production builder pair: `build_legs(case, run_dir)` returns an ordered list of leg descriptors (`name`, `leg_dir`, `input_path`, `command`) and the planner chains the ready-made commands with `&&` into the same single compute script. Task modules without `build_legs` keep the reference && production path untouched.

- **Legs** — one production ensemble per backend (`run_dir/numpy/production.yaml`, `run_dir/numba/production.yaml`); every `fssh_*` kernel keyword of the leg input is pinned to the leg backend, `fssh_profile` is switched on and `control.gl_seed` is written explicitly, so every trajectory draws the identical random number stream in both legs and the two ensemble averages must agree to floating-point noise. No exact quantum reference is computed: the comparison needs only the two-backend agreement plus the wall-time measurement.
- **Timing wrapper** — each leg command runs through `python3 -m pyrai3md.qualification.benchmark.timing <input.yaml> <leg_dir> <name>` (infrastructure module `timing.py`): it executes `pyrai3md` as a subprocess inside the leg directory, captures the calculation output into `<leg_dir>/<name>.stdout` (the function-level `FSSH Performance` tables), writes the wall time into `<leg_dir>/leg.timing` and exits with the subprocess return code, keeping the `&&` chain fail-fast. The wrapper is portable across the local `.sh` and the slurm `.sbatch`.
- **Evaluation** — `evaluate_run(run_dir)` reads both `leg.timing` files, ensemble-averages both legs' md.pop records (the state count is inferred from the extended electronic + active column layout), judges `backend_population_max_dev` (the maximum absolute deviation between the two ensemble averages) against the case tolerance and reports the wall times (`numpy_wall`, `numba_wall`, `speedup_wall`) plus the best-effort function-level profile speedups parsed from the captured stdout tables (reported, never judged). It persists `numpy.pop` / `numba.pop` / `perf.json` into the run directory; verdict is PASS/FAIL on the agreement metric, ERROR when a leg output is missing or broken.
- **Optional dependency** — numba is optional: when it is not installed `build_legs` raises and the planner reports a diagnostic skip task instead of planning an unrunnable comparison.

### The gsh case (Zhu-Nakamura global surface hopping)

The `gsh` task module (`gsh.py` + `gsh.yaml`) calibrates the Zhu-Nakamura global surface hopping method (`md.sfhp: gsh`, ported from the legacy implementation in #182) on the same Tully models as the fssh case. It reuses the algorithm-independent generic pieces of the fssh module — `load_case`, `apply_overrides`, `build_reference_input` are imported directly, so both calibrations share the identical exact split-operator quantum reference (`{method}_reference` qdm profile, ngrid 2048 / substeps 10) — and its `evaluate_run` delegates to the shared fssh evaluator `evaluate_unified` (the unified final-state metric set, see above), supplying the deterministic ZN kernel replay as the internal hopping probability of every case — `_lz_kernel_probability` for the lz family (the production `internal_conversion_prob` re-evaluated on the first recorded trajectory with the exact model adiabatic gradients) and `_tully_kernel_probability` for the tully cases (the same replay on numerically differentiated recorded state energies, `_numerical_adiabatic_gradients`, since the Tully models carry no closed-form adiabatic gradient interface); `tools/benchmark.py` routes `gsh`/`gsh_*` run directories explicitly to `gsh.evaluate_run` to keep the routing table self-describing.

- **GSH-only keyword table** — `gsh.build_production_input` mirrors the fssh builder structure (initcond '0', multitask/multipool wiring, the same structural keys) but owns a dedicated `PRODUCTION_MD_KEYWORDS` forwarding table: `sfhp` (callable default 'gsh'), the present-only `gap`/`gapsoc`/`adjust`/`reflect`/`maxh`/`activestate` entries (written only when the case or a pmd_env override sets them) and nothing else. Fssh-only keywords (`deco`, `rk4`, `substep`, `integrate`, `nactype`, the `fssh_*` family) fail the unrecognized-keyword guard and surface as a diagnostic skip instead of silently entering a GSH input.
- **Semantic differences** — GSH rebuilds a pure-state density every step (no coherent density, no decoherence/rk4/substep concepts), so the electronic block and the active block are trajectory-wise identical and the trajectory hop statistic is a clean Bernoulli count. The shipped case pins `adjust: 1` / `reflect: 1` (the global-rescale + frustrated-hop reflection policy) and leaves `activestate` unpinned: the schema default 0 requests all-state gradients, the data precondition of the Zhu-Nakamura hopping probability. The shipped `state_reference_difference` / `state_internal_difference` tolerances (1.0) are placeholders to be calibrated from the benchmark results (the pmd_env entries ship disabled, exactly like fssh).

### The lzsh case (Landau-Zener surface hopping)

The `lzsh` task module (`lzsh.py` + `lzsh.yaml` + dedicated `lzsh_lz1/2/3.yaml`) calibrates the Landau-Zener surface hopping method (`md.sfhp: lzsh`, new implementation, see `dev/methods/LZSH_DEVELOPMENT.md`) on the same Tully models as the fssh/gsh cases plus the dedicated Landau-Zener crossing family. It reuses the algorithm-independent generic pieces of the fssh module (imported directly) and shares the LZ-analytic infrastructure (`LZ_METHODS`, `_lz_analytic_probability`) with the gsh module, so all three algorithms calibrate against the identical exact split-operator quantum reference and the identical analytic Landau-Zener probability — and its `evaluate_run` delegates to the shared fssh evaluator `evaluate_unified` (the unified final-state metric set, see above), supplying the deterministic LZSH kernel replay as the internal hopping probability of every case — `_gate_lzsh_kernel_probability` for the lz family (the first non-zero per-step probability of `_replay_lzsh_kernel`) and `_max_lzsh_kernel_probability` for the tully cases (the maximum per-step probability of the replay window); `tools/benchmark.py` routes `lzsh`/`lzsh_*` run directories explicitly to `lzsh.evaluate_run`.

- **LZ-only keyword table** — `lzsh.build_production_input` owns a dedicated `PRODUCTION_MD_KEYWORDS` forwarding table: `sfhp` (callable default 'lzsh'), the present-only `gap`/`gapsoc`/`adjust`/`reflect`/`maxh` entries and nothing else. Fssh-only keywords fail the unrecognized-keyword guard, and the GSH-only `activestate` entry is absent: the LZSH kernels consume the state energies and the timestep only (no gradients, no NAC vectors). `lz_a`/`lz_b` forward into the qdm section of both legs through `PRODUCTION_QDM_KEYWORDS`.
- **Semantic differences** — the IC kernel derives the force difference from the curvature of the recorded avoided-crossing gap (three-point second time derivative at the gate step) and the ISC kernel from the signed-gap central difference, so both probability legs are gradient-free. The lz-family internal probability keeps the first-non-zero-gate semantics of the earlier kernel replay (`_gate_lzsh_kernel_probability`, a thin wrapper of `_replay_lzsh_kernel`) and reads the recorded trajectory energy snapshots only — simpler than the GSH replay, which additionally needs the adiabatic gradients. The tully cases report the maximum per-step probability of the replay window (`_max_lzsh_kernel_probability`): the tully surfaces carry no single-passage crossing of a fixed gate, so no gate value exists. The dedicated lz cases share the gsh phase-aligned parametrization so LZSH and GSH run the identical problem; the pmd_env entries ship disabled, exactly like the gsh entries.

### The ktdc case (kTDC energy-only curvature coupling)

The `ktdc` task module (`ktdc.py` + `ktdc.yaml` + dedicated `ktdc_lz1/2/3.yaml`) calibrates the FSSH algorithm with the energy-only curvature coupling NAC source (`md.nactype: ktdc`, valid fssh `nactype` value) instead of the exact qdm transition gradients (`nactype: nac`) on the same Tully models plus the dedicated Landau-Zener crossing family. It is a pure reuse module: every entry point (`load_case`, `apply_overrides`, `build_reference_input`, `build_production_input`, `evaluate_run`, the keyword tables and the LZ-analytic infrastructure) is imported verbatim from the fssh module — the same module objects, never re-implemented copies — because the production md input is otherwise identical to the fssh calibration and the two entries isolate the NAC source as the single difference. The ktdc case files pin `nactype: ktdc` (the fssh case files pin `nactype: nac`); the dedicated `ktdc_lz1/2/3.yaml` mirror the `fssh_lz1/2/3.yaml` parametrization and metrics entry for entry. `tools/benchmark.py` routes `ktdc`/`ktdc_*` run directories explicitly to `ktdc.evaluate_run`; the runner reaches it through the `benchmark_task: ktdc` handoff like every other task module.

### Population plotting

The benchmark evaluators render the S0 population curves of the compared population tables through the infrastructure helper `benchmark/plotting.py` (`plot_population_s0(run_dir, title)`): after the evaluators persist `production.pop` / `reference.pop` (plus `production.act.pop` when the active-state columns exist) and before the report returns, the helper overlays whichever tables exist — the active-state ensemble in goldenrod, the electronic ensemble in royalblue and the exact reference in crimson — on one figure saved as `population_s0.png` inside the run directory. The figure title follows the `<method>_<model>` case-key convention (`plot_title(prod)` derives it from the case production section: the sfhp algorithm plus the qdm model, for example `fssh_lz1` or `lzsh_tully1`; the ktdc coupling variant reports its nactype instead of the shared fssh sfhp), and the S0 probability axis is fixed to the range 0 to 1 with ticks every 0.2 so the figures of different calibrations stay directly comparable. The hook lives in the fssh, gsh and lzsh evaluators (the ktdc evaluator inherits it through the shared fssh evaluator; the perf backend comparison carries no reference curve and never plots) and records the outcome in `details['population_plot']`. The helper is diagnostic only: the Agg backend keeps it headless-safe, and a missing matplotlib installation, a missing or unreadable population table or any rendering failure degrades to a `skipped` entry and never affects the calibration verdict.

## 6.9 Application-Level Quality Assurance (QA) Design

The benchmark pillar is the application layer of PyRAI3MD's three-layer quality model (merged from the former `QA_DEVELOPMENT.md`). The three layers have strictly separated responsibilities:

```text
Skill Layer       ── Certification pillar   ── is the skill qualified?
Framework Layer   ── Verification pillar    ── is the software correct?
Application Layer ── Benchmark pillar (QA)  ── is the application result qualified?
```

The application layer answers a question the other two layers cannot: **do PyRAI3MD's actual application results meet the expected quality standard** (accuracy, numerical stability, performance, reproducibility, regression consistency)? It is a quality inspection of existing software and methods, not a new computation module: it never implements computational methods itself and never modifies the Workflow/Skill/Test architecture — production runs always go through the real execution path (Section 7.1).

### QA Case Model

The basic unit is the QA Case; each benchmark case YAML describes one explicit quality judgment:

```text
Case
├── Input       (reference + production input builders)
├── Reference   (trusted standard result)
├── Production  (PyRAI3MD result via the real execution path)
├── Metric      (quantitative comparison)
├── Tolerance   (predefined acceptance bound)
└── Verdict     (PASS / FAIL + retained metrics)
```

All cases follow the unified `Reference → Production → Evaluation → Verdict` pipeline (realized in the benchmark two-task chain). Reference results may come from analytic solutions, high-precision numerical computation, independent programs, or literature data. The verdict is not only PASS/FAIL: the actual metric values are always retained (e.g. reference transmission 0.673 vs production 0.668, absolute error 0.005 against tolerance 0.020 → PASS).

### Evaluation Boundary vs Trajectory Analysis

QA evaluation stays **simple, standardized, automatically executable, reproducible, and decidable** (MAE / RMSE / maximum error / relative error / population deviation / energy drift / runtime / memory / scaling). It answers "does the result meet the quality standard?" and does NOT replace complex trajectory analysis tools, which answer the scientific question "what phenomena did the trajectories produce?" (RMSD, MSD, state lifetimes, trajectory distributions, ...). Complex scientific analysis remains outside QA.

### Independence Rules

- FSSH and QDM are combined only at the QA layer: the FSSH workflow does not depend on QDM and QDM does not depend on FSSH; the benchmark case uses QDM for the reference and the normal FSSH workflow for production, with no mutual dependency;
- QA calls PyRAI3MD, never the reverse — the QA/benchmark layer is a verifier, not part of the verified system, so its logic never enters the runtime core (one-way dependency, Overview principle 5);
- `pyrai3md/examples/` stays user-facing application examples and is never reorganized or absorbed into QA; an example that needs testing runs through the existing qualification mechanisms.

### No Premature Abstraction, Natural Extension

The first stage implements only FSSH and deliberately avoids large abstractions (no QAManager/QAWorkflow/QAState/QADispatcher); common components are extracted only after multiple cases produce genuinely repeated code. Future QA dimensions extend the benchmark pillar naturally:

- **Accuracy QA** — PyRAI3MD ↔ reference comparison;
- **Stability QA** — long-time propagation, extreme conditions, numerical stability;
- **Performance QA** — runtime, memory, parallel scaling;
- **Regression QA** — version N vs version N+1;
- **Reproducibility QA** — same input + same seed → consistent result.

---

# 7. Execution Principles

## 7.1 Real Execution Path

All qualification tasks follow the normal PyRAI3MD execution architecture:

```text
Input / Test Configuration
    |
    v
Interface Layer
    |
    v
Schema Validation
    |
    v
State Construction
    |
    v
Kernel Management
    |
    v
Component Execution
    |
    v
Result Collection
```

Qualification should not bypass:

- Input processing;
- State management;
- Registration mechanisms;
- Execution control.

### 7.1.1 Truthful verdicts and test placement (2026-11-12)

- The six post-run artifact checks run in their own `try`; an exception sets
  `status='error'` with the traceback. It can no longer leave the earlier
  PASS in place.
- A unittest file that collects zero cases is an error, not a pass. This is
  the guard against pytest-style modules or helper files certifying
  themselves.
- A file whose every collected case skipped (a module-level
  `unittest.SkipTest` guard, for one) is a `skip` row, not a pass: no assertion
  ran. Files under `verification/` must be unittest-style for the same reason;
  a pytest module there collects nothing, so its checks have to move to
  `integration/` or be converted (pinned by
  `test_diagnostics.py::TestWhatTheQualificationShows`).
- A planner exception produces an error row (`_make_error_task`); only a
  genuinely inapplicable case is a skip.
- The shared `pyrai3md/skills/tests/` pytest group follows the aggregate
  `skills.<name>.runtime_tests` phase. `_runall: true` forces that phase open,
  so the per-skill and shared pytest rows are planned even when the shipped
  skill sections say `runtime_tests: false`.
- The current full plan contains 153 system tasks (136 verification files +
  17 integration tasks) and 78 pytest runtime rows (73 skill files + 5 shared
  files); all package test files are represented. The remote run is a
  file-level plan, not a claim that every case executed: skips, host gates and
  timeouts still have to be read in the report.

## 7.2 Execution Method

The qualification system does not directly call:

```python
kernel.run()
skill.execute()
```

Instead, tasks use subprocess to invoke PyRAI3MD commands:

```text
Qualification -> subprocess -> PyRAI3MD Command
```

> Qualification verifies the software through its own public interfaces.

## 7.3 Command Formats

| Mode | Command | Input File | Slurm |
|------|---------|------------|-------|
| pyrai3md_yaml | `pyrai3md <example>.yaml` | Example YAML | No |
| pyrai3md_input | `pyrai3md <test>.input` | Legacy text (converter) | Yes (skills) |
| pyrai3md_agent | `pyrai3md_agent <test>.config` | Sectioned dict JSON (converter) | Yes (skills) |
| skill_agent | `pyrai3md_skill <test>.json` | Flat skill JSON (converter) | No |
| benchmark compute | `bash benchmark_<case>.sh` (or sbatch) | Chained reference && production script | Yes |

## 7.4 Environment Variable Merging

The `environment.export` section contains variable names mapped to lists of values:

```yaml
export:
  OMP_NUM_THREADS: [4]
  MPI_NP: [4]
```

The runner merges these value lists into environment variables before test execution. For slurm jobs, variables are exported in the batch script.

## 7.5 Raw Shell Commands

The `environment.shell` section holds raw shell commands (typically module loads required by external programs, e.g. MKL for Molcas):

```yaml
shell:
  - 'module load gcc/11.3.0'
  - 'module load intel/mkl-mpi'
```

These commands are reserved for the **main framework** scripts: the examples wrappers and all three skill test wrapper modes (`_skill`/`_agent`/`_input`, local `.sh` and slurm `.sbatch`) copy them after the `#SBATCH` directives and before the `export` section (#136: every qualification wrapper loads the global shell). The skill-level shell only surfaces inside the execution scripts that the skill backend writes at runtime. A single string value is also accepted; an empty or missing section writes nothing.

The same split applies to the inner execution layers:

- **Skill run scripts**: only the skill-level shell reaches the skill configs — `skills.<name>.shell` rides along the machine-parameter injection (fill-if-empty, same semantics as Section 6). Every external-program skill (molcas/bagel/lammps/xtb/orca/openqp) writes its config `shell` into its own local `.sh` and slurm `.sbatch` before the `export` block. The global `environment.shell` is NOT filled into skill configs (#134): skills that need module loads (e.g. xtb) must declare `skills.<name>.shell` explicitly. pyscf is exempt (its local mode runs in-process; its slurm sbatch inherits the submission environment).
- **Dispatch child runtimes**: `slurm_run` copies `state.pmd.environment.shell` verbatim into each child `slurm.sh` (after the `#SBATCH` directives, before the exports) — dispatch serves the main framework and keeps the global shell.

Classic input limitation: the `&environment shell:` entry is single-line; separate multiple commands with `;` (e.g. `shell: module load gcc; module load mkl`). All normalization points (kernel `get_shell`, qualification `_get_shell_commands`, skill template helpers) accept the three forms equivalently: yaml list, multi-line string, single line with `;` separators. The converter enforces the same single-line contract on the way out: a list or multi-line `shell` in the agent config is collapsed into one `;`-separated line in the skill section (`_collapse_shell_line`, #135).

---

# 8. Remote Deployment Workflow

## 8.1 Deploy Tool Location

`deploy.py` is a development utility, located at:

```text
tools/deploy.py
```

It checks its location at runtime. If not in the `tools/` directory, a warning is printed.

## 8.2 Deployment Workflow

```text
Local Development Environment
    |
    v
Read qualification/pmd_env.yaml
    |
    v
Verify deploy configuration (interactive if incomplete)
    |
    v
Resolve SSH connection mode (direct first, then jump host) and print the SSH/conda banner
    |
    v
Resolve conda environment (interactive if empty, verify on remote)
    |
    v
Synchronize source code (rsync, excludes .git/, __pycache__/, *.egg-info/,
.idea/, .vscode/, legacy/, skill_demo/, tmp_*/, test_*.log, build/, *.egg;
--delete-excluded is NOT used, so remote run artifacts survive)
    |
    v
Install PyRAI3MD (rm -rf build *.egg-info, then pip install,
so the server cannot reuse a build tree the sync never prunes)
    |
    v
Plan probe (ssh: generate_plan(load_config()) in the run dir) ->
that total becomes the progress denominator; then touch marker ->
(nohup python -u -m pyrai3md test,
background with output to deploy_test.log) -> foreground loop waits up to
120s with 'find -newer marker' for this run's new tmp_<YYYYMMDD_HHMMSS>
directory -> returns (success, work_dir)
    |
    v
Poll: _collect_results(resolved_dir=work_dir) polls status.json every 30s
with the resolved directory as the direct target (skips the discovery phase)
    |
    v
Fetch results.json via SSH, download deploy_test.log as test_<stamp>.log, print summary with tracebacks
```

## 8.3 Key Implementation Details

**Progress line**: every polling round prints one line built by
`_progress_line`, `[YYYY-MM-DD HH:MM:SS] RUNNING TESTS  34% (043 of 125)
ELAPSED: 34 seconds <detail>`. The completed count is zero padded to the
digit width of the total and the percentage is right aligned to three
columns, so successive rounds line up in a terminal. While the total is
still unknown the percentage and counter are dropped. The round number
(`Check #n`) is no longer printed; the liveness and completion details
(`process alive`, `no new results`, `WARNING: remote process not found!`,
SSH timeouts) move to the trailing `<detail>` field so no diagnostic is
lost.

**Synchronous trigger barrier**: The trigger command is a single remote
shell pipeline: it touches a marker file (`.deploy_trigger_marker`),
launches the test inside an explicit subshell `(nohup ... > deploy_test.log
2>&1 < /dev/null &)`, then runs a foreground loop for up to 120 seconds
(`_TRIGGER_WAIT_SECS`) executing `find . -maxdepth 1 -type d -newer
<marker> -name 'tmp_*'`, keeping only `tmp_<YYYYMMDD_HHMMSS>` matches
(skill calculation directories such as `tmp_MOLCAS` are filtered out),
and returns `(success, work_dir)`. The marker file makes 'newer' refer to
the trigger instant, so leftover timestamped directories from earlier
runs can never be reported; when no directory appears within the window
the trigger still succeeds and the poller falls back to discovery.

**Remote process survival**: The remote test is launched with `nohup` inside an explicit subshell `(nohup ... > deploy_test.log 2>&1 < /dev/null &)` with all stdio redirected. This prevents the process from being killed when the SSH session closes. Backgrounding inside the subshell is essential: a bare `a && b && nohup cmd > log 2>&1 &` backgrounds the whole AND-list, and the implicit subshell inherits the SSH channel stdout/stderr and waits for the test process, keeping the channel open until the trigger timeout expires.

**Result collection**: The trigger normally resolves this run's work
directory synchronously, so `_collect_results(resolved_dir=...)` takes
the resolved directory as the direct polling target and skips the
discovery phase entirely: `status.json` is polled at that exact path
every 30 seconds. `tmp_wait_timeout` (120 s) is retained only as the
resolution-failure fallback path: when `resolved_dir` is empty, the
poller rediscovers the latest `tmp_*` directory on the remote via
`ls -dt tmp_*/` (keeping only `tmp_<YYYYMMDD_HHMMSS>` names) and locks
the first timestamped match, aborting after `tmp_wait_timeout`. Progress
is reported as `X/N tasks completed, +N new, process alive`. On
completion only `deploy_test.log` is downloaded into the current
directory as `test_<stamp>.log` (stamp derived from the tmp directory
name, e.g. `tmp_20260812_092706` -> `test_20260812_092706.log`); the
remote tmp_* work directory with full calculation outputs stays on the
server. The download prefers an `ssh cat` redirection over the
already-verified SSH channel (so it inherits the direct/jump connection
mode exactly) and falls back to scp, whose jump usage must mirror the
actual SSH connection (ssh via jump -> `scp -J jump`; ssh direct -> scp
direct). A best-effort download also runs on the idle-timeout and
tmp-wait-timeout abort paths.

**Error reporting**: When tests complete, the tool fetches `results.json` via SSH (not scp) to avoid path mismatch issues. Its summary sums the same four result categories the runner aggregates (`main`/`example`/`skill`/`benchmark`); a version that left `benchmark_results` out announced 568 tasks for a 593-task round and could not list a failing benchmark at all, which `TestSummaryCountsEveryCategory` now pins. The summary includes full error tracebacks for failed tests.

**Configuration help**: `python tools/deploy.py --help` displays the configuration file path and all configurable variables.

**Force stop**: `python tools/deploy.py --force` stops a running remote test through `REMOTE_TEST_PATTERN` (`python.*pyrai[3]md test`; the bracketed digit keeps the pattern from matching the shell that carries it, which used to kill its own session and report a failed stop for a test that had stopped). The `pkill` exit code decides which of three lines is printed: stopped / nothing was running / host unreachable. The same pattern drives the liveness probe, so a dead run cannot report itself alive.

## 8.4 Guaranteed Exit Policy

The deploy tool **must always exit automatically**, even under total failure conditions. This is a critical design requirement.

The following scenarios must all result in automatic exit (never hang indefinitely):

| Scenario | Exit Mechanism |
|----------|---------------|
| All tests fail | `status.json` reaches terminal state (`pass`/`fail`/`error`/`skip`) |
| Tests cannot start | Trigger barrier 120 s — no new `tmp_<YYYYMMDD_HHMMSS>` directory appears after the marker (fallback discovery is bounded by `tmp_wait_timeout=120s`) |
| Remote process crashes | Process alive check (`pgrep`) detects death, exits after grace period |
| Remote process hangs | `idle_timeout=300s` — no response for 5 minutes |
| SSH connectivity lost | `ConnectTimeout=5` on every ssh/scp command plus local subprocess timeouts: direct connection probe 10 s, jump-host probe 20 s, trigger channel (`_run_remote`) 150 s, status polling 30 s per check; combined with the idle timeout |
| No results written | Process alive check + idle timeout combined |

**Idle timeout mechanism**: Instead of a fixed total timeout, the tool uses an idle (no-response) timeout. The timer resets whenever:

- A tmp directory is discovered on remote;
- `status.json` is successfully read;
- New test results appear in `results.json`;
- The remote process is confirmed alive via `pgrep`.

This ensures long-running tests (e.g., Molcas calculations) are not interrupted, while genuinely hung processes are detected.

**Terminal states**: All possible `status.json` values must be recognized as terminal:

- `pass` — all tests passed;
- `fail` — some tests failed;
- `error` — framework error;
- `skip` — tests were skipped (e.g., missing dependencies).

Any unrecognized status is treated as non-terminal (`running`), but the process alive check provides a secondary safety net.

## 8.5 Boundary

`tools/deploy.py` handles deployment, test triggering, and result collection. It does not handle:

- Qualification planning;
- Qualification task generation;
- Qualification execution logic.

These are handled by `qualification` on the remote server when `pyrai3md test` is executed.

---

# 9. Design Goals

## Reproducibility

The same configuration should produce equivalent qualification procedures across different environments.

## Extensibility

New components can introduce their own tests without modifying the core qualification framework. Skills register in `pmd_env.yaml` and the manager automatically queries their registry to generate corresponding test tasks. Official examples are qualified automatically as soon as they are added to `pyrai3md/examples/` and listed in `QUALIFIED_EXAMPLES`.

## Maintainability

Qualification logic remains separated from runtime logic and development utilities. The reporter functionality is integrated into the runner, reducing module count.

## Automation

A single command (`pyrai3md test` or `python tools/deploy.py`) should validate a complete PyRAI3MD installation.

## Agent Compatibility

The qualification framework provides clear interfaces that allow automated development agents to:

- Deploy code;
- Execute validation;
- Detect failures;
- Assist debugging.

---

# 10. Summary

The PyRAI3MD qualification system separates:

```text
Deployment (tools/deploy.py)
    |
    v
Configuration (qualification/pmd_env.yaml)
    |
    v
Qualification Planning (qualification/manager.py + verification/certification/benchmark planners)
    |
    v
Qualification Execution (qualification/runner.py)
    |
    v
Runtime / Example / Skill Qualification (pyrai3md / pyrai3md_agent / pyrai3md_skill)
```

Final architecture principle:

> PyRAI3MD qualifies itself, but qualification logic does not belong to the PyRAI3MD runtime core.

- `tools/deploy.py` handles deployment and test triggering;
- `qualification/` handles qualification planning, execution, and reporting; planning is delegated to the three pillar planners (`verification/planner.py`, `certification/planner.py`, `benchmark/planner.py`) and aggregated by `manager.py`;
- `qualification/certification/converter.py` parses skill test YAMLs into flat .json configs, converts example YAMLs to flat .json configs, and converts each .json to both the legacy input text and the pyrai3md agent config (one shared sectioned dict);
- `qualification/certification/handler.py` discovers skill test files and builds the generic classic-input skill section;
- `qualification/benchmark/` provides the algorithm benchmarks: `planner.py` chains reference && production into one script per case, `fssh.py` supplies the tully1 calibration task, `ktdc.py` supplies the identical FSSH calibration with the energy-only curvature coupling NAC source (verbatim fssh reuse), `gsh.py` supplies the Zhu-Nakamura calibration, `lzsh.py` supplies the Landau-Zener calibration and `plotting.py` renders the S0 population plot shared by all evaluators;
- `pyrai3md/examples/` provides the official workflow examples qualified in two modes (yaml + classic input);
- `<skill>/tests/` provides the test case YAML definitions and runtime test scripts;
- `pyrai3md/skills/query.py` provides skill registry information to the manager;
- `pyrai3md/skills/wrapper.py` provides the `pyrai3md_skill` command for independent skill testing;
- `pmd_env.yaml` unifies environment and deployment configuration.

---

# 11. Multi-Center Acceptance Coverage (docs §13)

The multi-center unified base (refactor R0-R6) is accepted against the
`dev/MULTICENTER_DEVELOPMENT.md` §13 matrix. Coverage cross-reference (all modules
in `qualification/verification/`, mock/analytic backends):

```text
§13.1 input & schema   1-5   test_multicenter_schema.py, test_multicenter_baseline.py, test_pmd.py
§13.2 backend & sched  6-7   test_calculator.py, test_task.py, test_core.py
                       8-11  test_dispatch.py (TestTermWorker, molecule flags),
                             test_calculator.py (TestTaskExecutor, Barrier concurrency)
§13.3 electronic/phys  12    test_electronic.py (transparent C=1), test_fssh_regression.py
                       13    test_electronic.py (r5f), test_multicenter_md.py (r6e)
                       14    test_fssh.py, test_gsh.py, test_lzsh.py, test_electronic.py (view freshness)
                       15    test_electronic.py (r5f velocity/active locality)
                       16    test_calculator.py (TestR4fMultiCenterGate finite difference)
                       17    test_integrator.py, test_multicenter_md.py (verlet assembled gradient)
                       18    test_calculator.py (commit atomicity), test_multicenter_md.py
§13.4 restart & compat 19    test_calculator.py (TestR4fMultiCenterGate serialize round-trip)
                       20    test_calculator.py (out-of-order == serial), test_dispatch.py
                       21    the whole verification suite (regression)
                       22    test_core.py (C>1 scalar access raises), test_electronic.py
```

Deferred / boundary (documented, not gaps of this round):

- §13.3-17: the "two half-steps same state" / "post-hop force feeds the next
  step" integration timing was WITHDRAWN by the R6 revision (design §10.2/§10.3
  annotations); the verlet integration convergence and the assembled
  active-gradient reading stay covered.
- §13.4-19: full trajectory restart (run N continuously vs save at K, restore,
  continue to N) is deferred together with the checkpoint/resume feature
  (`md.py` L481 TODO, needs full RNG-state serialization); the committed-state
  save/load round-trip is already covered.
- §13.2-11: a real backend end-to-end concurrent qualification is
  deferred (no real backend environment); the mock/analytic gates prove
  framework scheduling, and `test_thread_anchoring.py` proves the runners
  keep the process cwd where a second term can reach it.
