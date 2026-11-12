# PyRAI3MD Architecture Specification

> **A classical low layer can now be handed a center (2026-10-09, #317).** A 2-layer
> multi-center plan is `L(real) + Σ_c [H(c) − L(c)]`, so with a force-field skill in the low
> layer every center slice is also a classical calculation, and such a skill reads its topology
> from a user template it does not generate. The channel that closes this is one keyword:
> `planning/assembly.py` hands a region task the 1-based **row numbers** of its region in the order the
> plan declared them (`atom_selection`, routed through the skill's own
> `schema/reference.py::DEFAULT` like every other new keyword), and the skill cuts and
> compresses the user's `{project}.data` / `.in.init` / `.in.settings` into that task's own
> files at backend initialization (`skills/lammps_skill/service/builder/ff_model.py::_apply_selection`,
> `_degrade_coulomb`, `input_builder._subsystem_box`). A cut-out subsystem is evaluated
> **non-periodically** — `boundary f f f`, no kspace command, `…/coul/long` pair styles mapped
> to their real-space siblings at the template's own cutoffs — because keeping the box would
> make the difference subtract the cluster's Ewald self-image term as well (measured
> 0.280090 kcal/mol = 4.464e-4 Ha per center). A center is a set of atoms, so
> `schema/adapter.py` canonicalises its indices to ascending for every input carrier and the
> skill renumbers by position, never by value, because its writer puts coordinate row `i` on
> the row numbered `i + 1`. `examples/multicenter_pbc_{nve,npt}.yaml` are now two disjoint centers
> inside the shipped 60-atom box rather than one water standing in for the whole system, and
> the recorded-step region watch has run at more than one center against real engines. The
> `stress` routing defect found while certifying it is fixed in #318: one capability set is
> now defined in one place, and a task whose region is a slice cannot be asked for a
> whole-system property, so the box force comes from the whole-system task and the NPT run's
> `Pvir_bar` reads -6050.718 bar at its first step instead of zero. Details:
> `dev/skills/LAMMPS_SKILL_DEVELOPMENT.md` §5.6a, `dev/methods/PBC_DEVELOPMENT.md` §4.2,
> `dev/MULTICENTER_DEVELOPMENT.md` §11.3.

> **MD step watch and the `&md` keyword surface (2026-10-08, #314).** `verification()` is
> deleted from the integrator: the recorded step is now watched in `AIMD.checkpoint`
> (`records/diagnostics.py::check_recorded_step` for the frame's own kinetic energy, crossing
> counts and the region each center is cut from, `integrator/operations.py::verify_recorded`
> for the constraint reading), on the steps `output.record_due` keeps. The loop has nineteen
> phase rows, not twenty. `&md` keywords that
> nothing reads are refused by name (`schema/config.py::RETIRE_DYNAMICS_KEYS`); the
> velocity-control family, `randvelo` and `addsoc` are implemented, and `dissipate` delays the
> thermostat until the trajectory has been on the ground state for `thermodelay` steps.
> `RuntimeState` carries two new run-level values, `dissipate_step` and `random_state`, and the
> latter is what lets a resumed segment continue its surface-hopping stream. Details:
> `dev/methods/MD_DEVELOPMENT.md` §23, §28 and `dev/methods/ENSEMBLE_DEVELOPMENT.md` §3.4.


Last updated: 2026-11-12 19:20 (documentation sync for #489: lazy skill export, per-skill read_support, parser missing-block contract, optimizer/fix-mask declarations, qualification truthfulness. Earlier: 2026-11-12 15:50 (planning/workflow isolation is one-way; composite registry references are marked retired; `schema/reference.py` is pure data again. Earlier: 2026-11-12 12:45 (the five tool manuals moved out of `dev/tools/` into `tools/`, beside the scripts they document, and `dev/methods/adaptive_nn_draft.md` was removed; the tools tree lists the manuals and the docs count follows the disk from 51 to 45, checked against the `tools/` and `dev/` listings. The rest was not re-audited.) Earlier: 2026-11-12 00:55 (the tools and docs trees gained the migrated training-data, trajectory, multiscale and manual-update tools plus `dev/tools/`; the docs count follows the disk from 46 to 51; checked against the `tools/` and `dev/tools/` listings. The rest was not re-audited.) Earlier: 2026-11-09 16:40 (data/core.py split into data/state.py + data/build.py; task assembly moved to planning/assembly.py; persisted task fields collected in data/task.py; framework `pyrai3md/manifest.yaml` added to the package tree; the workflow→planning and planning→skills dependency exceptions recorded; the classic constraint writer now lives in qualification. The rest was not re-audited. Earlier: 2026-11-06 13:20 (only the planning bullet of section 3.4 was re-checked, and only against what this round ran: the label plan it describes is what `planning/task.py` and `workflow/multiscale/composite.py` actually did on the three local `examples/adaptive_*.yaml` carriers and on remote round `tmp_20261106_102745` - both solution spellings reported `source: layer` from the same task over the same six atoms, and their labels came out bit-identical. The rest of the document was not re-audited. Earlier: 2026-11-04 11:10 (#451: the Calculator bullet now names `validate_plan` as the thing that actually guards a composite queue - per scheme and structurally, with the qmmm duty rules - and says what `COMPOSITE_PATTERNS` still holds and who calls it (no production path). The rest of the file was not re-read. Earlier: 2026-10-24 09:40 (the utilities refactor is registered: 17 non-init modules, `structure_formatter.py` and `opt_formatter.py` in, `coordinates.py`, `image_flags.py` and `opt_status.py` out, and `geodesic.py` added; the docs tree replaces `utilities_refactor.md` with `UTILITIES_DEVELOPMENT.md` and the count stays 45. Earlier 08:45: the documentation tree listed `utilities_refactor.md`; the count followed the disk from 44 to 45. Earlier 2026-10-22 22:00 (the `optimize_cell` paragraph now points at `CELL_OPT_DEVELOPMENT.md`, where the cell and periodic design material moved from `OPTIMIZE_DEVELOPMENT.md`; the dispatch diagram calls `load_state()`, and the docs-tree descriptions of NUM_HESS and SEARCH no longer say placeholder; the rest was not re-read. Before it, 2026-10-13 14:22: the `data/` tree, the `integrator/operations.py` row and the constraint-delegation table were checked against the code for the `data/geometry.py` move; the rest of the file was not re-audited)

---

> **Multi-center unified base (refactor R0-R6, 2026-09-08).** `center` is the
> single partition axis of the computation data model: every calculation carries
> a length-C center plan (`data/center.py`), and the ordinary single-center run
> is C=1 (CHANGELOG #198-#204, `dev/MULTICENTER_DEVELOPMENT.md`). Layers: R1 the
> canonical center schema; R2 the unified static plan (`planning/task.py`
> real-first ONIOM expansion, per-task identity + electronic binding); R3 the
> term execution layer (`dispatch/terms.py` serial/thread + `term_worker.py`
> remote per-term, controlled submission + one failure protocol); R4 the center
> data model on `PyRAIState` (partition_plan / electronic[c] / active_surface /
> continuation, structured HistoryFrame, versioned serializer); R5 the electronic
> dispatch layer (`integrator/electronic.py` ElectronicStepper runs the UNCHANGED
> fssh/gsh/lzsh cores on per-center CenterView proxies, C=1 transparent); R6 the
> MD-loop functional extension (verlet reads the assembled active gradient,
> center-aware target states, center-protocol output/trajectory/log). Invariant:
> one global coordinate/velocity/mass set; centers never own a nested PyRAIState
> or an independent nuclear trajectory; C=1 and C>1 differ only in data length
> and physical strategy, never in data source or code path.

# 1. Design Philosophy

PyRAI3MD is designed as a modular scientific computing framework rather than a monolithic molecular dynamics program.

The architecture follows five fundamental principles:

1. Core owns protocols, not scientific knowledge.
2. Skills own domain knowledge and execution.
3. Maestro is the only mechanism for skill coordination and capability discovery.
4. Workflow composes capabilities rather than software.
5. Internal execution uses Python objects, external interaction uses schemas.

The architecture must remain backend-independent.

The Core must never know ORCA, Gaussian, CP2K, PySCF, NequIP, MACE, etc.

---

# 2. High-Level Architecture

```
PyRAI3MD/                              (git repo root)

├── setup.py                  # Package installer (name='pyrai3md')
│
├── pyrai3md/                 # Python package (import pyrai3md)
│   ├── __init__.py
│   ├── __main__.py           #   python -m entry
│   ├── pmd.py                #   Traditional CLI entry + orchestration
│   ├── agent.py              #   Agent entry (parallel)
│   ├── manifest.yaml         #   Framework discovery: input schema + code entry points
│   │                         #   (guarded by verification/test_manifest.py)
│   ├── skill_guide.md        #   Framework usage guide: CLI, input, outputs, contracts
│   │
│   ├── interface/            #   Input conversion + structural validation
│   │   ├── __init__.py       #     Exports parser, loader
│   │   ├── parser.py         #     Text/JSON input → raw dict + coordinate loading + structure validation
│   │   └── loader.py         #     Agent JSON config → raw dict
│   │
│   ├── schema/               #   Configuration building and validation
│   │   ├── __init__.py
│   │   ├── config.py         #     Pydantic models (Control ~10 fields, Adaptive ~30 fields)
│   │   │                      #   + the system constraint records (SystemConstraintConfig & friends)
│   │   ├── adapter.py        #     Raw dict → typed Config + build_skill_config
│   │   │                      #   + the classic &constraint text reader
│   │   │                      #     (writer: qualification/certification/constraint_writer.py)
│   │   │                      #   + the 19 legacy flat constraint keys → a constraint mapping
│   │   ├── normalize.py      #     Derived quantity computation + the constraint plan build
│   │   ├── validate.py       #     Consistency validation + Task + the constraint refusals
│   │   └── reference.py      #     Periodic table and constants
│   │
│   ├── planning/             #   Planning layer (capability + task resolution)
│   │   ├── __init__.py       #     Exports resolve_capabilities, CAPABILITY_NAMES,
│   │   │                          #   resolve_task_queue, resolve_composite_type
│   │   ├── resolver.py       #     Token → capability name strings + multi-backend constraints + dynamics derivation
│   │   ├── task.py           #     Composite detection + Task Queue expansion (ONIOM/fragment/qmmm)
│   │   └── assembly.py       #     Queue assembly: skill binding merge, task configs, driver/curvature gates
│   │
│   ├── workflow/             #   Workflow orchestrators + Calculator
│   │   │                     #   One module per workflow class at this level; the
│   │   │                     #   execution components live in the named subpackages
│   │   │                     #   (multiscale/, records/, dispatch/, integrator/,
│   │   │                     #   optimizer/, constraints/)
│   │   ├── __init__.py       #     Exports Workflow classes, Calculator, resolve_workflow,
│   │   │                          #   execute, WORKFLOW_REGISTRY, JOBTYPE_MAP
│   │   ├── calculator.py     #     Calculator(Dispatcher): initialize/run/sync,
│   │   │                          #   state ownership (get_state/update),
│   │   │                          #   backend role grouping + switching,
│   │   │                          #   result write-back
│   │   ├── base.py           #     Workflow(Dispatcher) base class (the only base;
│   │   │                          #   loops/batch algorithms live in concrete modules)
│   │   ├── optimize.py       #     Geometry optimization (iterative)
│   │   ├── optimizer/        #     Optimizer algorithms
│   │   │   ├── base.py       #       Optimizer ABC (run/convergence/_compute_step)
│   │   │   ├── minima.py     #       BFGS + RFO + Trust Radius
│   │   │   ├── gdiis.py      #       GDIIS extrapolation
│   │   │   ├── ts.py         #       Transition state (partitioned RFO, mode following)
│   │   │   ├── meci.py       #       Minimum energy conical intersection
│   │   │   ├── mecp.py       #       Minimum energy crossing point
│   │   │   ├── mep.py        #       Minimum energy path
│   │   │   └── neb.py        #       Nudged elastic band
│   │   ├── md.py             #     Ab initio MD, class AIMD (iterative)
│   │   ├── constraints/      #     System constraint subsystem (dev/methods/
│   │   │   │                   #   CONSTRAINT_DEVELOPMENT.md); default closed: every
│   │   │   │                   #   entry returns unchanged on an empty plan
│   │   │   ├── controller.py #       SystemConstraintController: the staged public API
│   │   │   ├── runtime.py    #     The two framework call sites (finalizer, apply_constrain)
│   │   │   ├── state_adapter.py #  PyRAIState -> SystemView + EvaluationContext
│   │   │   ├── bias.py       #       Harmonic bias, region selection, periodic difference
│   │   │   ├── wall.py       #       Walls: two laws over four geometries, grouped targets
│   │   │   ├── projection.py #       Fix mask, declared mask, anchors, condition rank, free DOF
│   │   │   ├── shake.py      #       Bond position solve + opt-in velocity projection
│   │   │   ├── metadynamics.py #     Gaussian and aligned-RMSD history potentials
│   │   │   └── schedule.py   #       Parameter ramps keyed on the accepted step
│   │   ├── integrator/       #     MD integrator and operations
│   │   │   ├── verlet.py     #       velocity-Verlet (verlet_i / verlet_ii)
│   │   │   ├── surface_hopping.py #  Surface hopping dispatch (nosh/gsh/fssh)
│   │   │   └── operations.py #       MD operations (constrain, phase, ensemble, kinetic);
│   │   │                    #       the periodic wrap is data/geometry.py
│   │   ├── records/          #     Run records of the MD chain (components, no
│   │   │   │                  #   Workflow classes, not in WORKFLOW_REGISTRY)
│   │   │   ├── trajectory.py #       TrajectoryResult / TrajectorySet: the universal
│   │   │   │                  #       return of an iterative run (idx-ordered)
│   │   │   ├── deviation.py  #       NN deviation reception (DEVIATION_KEYS,
│   │   │   │                  #       fetch/merge/judge; tolerant 0.0 fill)
│   │   │   ├── ml.py         #       ML outcome channel: does a train /
│   │   │   │                  #       prediction call carry the record its
│   │   │   │                  #       jobtype promises (no threshold, no
│   │   │   │                  #       arithmetic)?
│   │   │   ├── output.py     #       MdOutputWriter: the title.md.* / title.sh.*
│   │   │   │                  #       records + record_due (the step cadence)
│   │   │   └── diagnostics.py#       Run-health checks: cell/ensemble gates,
│   │   │                          #       the recorded-step sentinels (kinetic
│   │   │                          #       identity, image integrity, region
│   │   │                          #       reconstruction), summary,
│   │   │                          #       RuntimeState.diagnostics
│   │   ├── multiscale/       #     Multiscale execution components (multi-center is
│   │   │   │                  #   one case; later schemes land here)
│   │   │   ├── composite.py  #       pure aggregation (aggregate_results) + multi-center assembly
│   │   │   │                  #       (aggregate_results) and projection (project);
│   │   │   │                  #       multi-center strict assembly (validate_plan,
│   │   │   │                  #       assemble_results, TaskOutcome, normalize_task_result)
│   │   │   ├── regions.py    #       Pure geometric region operations (validate_region,
│   │   │   │                  #       materialize_region, project_gradient, slice_field)
│   │   │   └── charges.py    #       Embedding-charge sources, per-task embedding
│   │   │                          #       and the outcome-driven charge update
│   │   ├── frequency.py      #     Vibrational frequency analysis
│   │   ├── scan.py           #     PES scan workflow
│   │   ├── hessian.py        #     Numerical Hessian (batch)
│   │   ├── adaptive.py       #     Adaptive sampling (round-based loop)
│   │   ├── search.py         #     GridSearch: hyperparameter grid scan
│   │   │                        #   (batch of train calls; the structural
│   │   │                        #   placeholder it replaced is reserved for
│   │   │                        #   its own token -- dev/methods/
│   │   │                        #   SEARCH_DEVELOPMENT.md 1.1)
│   │   ├── grid/             #     Scan components (no Workflow class, not in
│   │   │                        #   WORKFLOW_REGISTRY): plan and read-back of
│   │   │                        #   a hyperparameter grid, TF-free
│   │   │   ├── plan.py       #       declaration surface (skill query), point
│   │   │   │                  #       expansion, key rendering, per-unit stamp
│   │   │   └── record.py     #       record read, ranking, tables, CSV
│   │   ├── single.py         #     Single-shot workflow: skill-driven jobs
│   │   │                          #   (driver != 'main') and the train/prediction
│   │   │                          #   jobtypes, whose procedure the skill owns
│   │   └── dispatch/         #     Dispatcher subsystem
│   │       ├── dispatcher.py #       Dispatcher base (unified run pipeline,
│   │       │                      #     unit materialization, universal rules)
│   │       ├── launcher.py   #       Launcher: remotetask -> 'local' or slurm_run
│   │       ├── scheduler.py  #       Scheduler: multitask -> multiprocessing Pool
│   │       ├── terms.py      #       Term axis: serial/thread executor, the
│   │       │                      #     skill-owned thread check + the
│   │       │                      #     work_item/outcome contracts
│   │       ├── term_worker.py #      Remote per-term worker execution
│   │       ├── planner.py    #       ExecutionGraphBuilder + tri-state aggregation
│   │       ├── slurm_run.py  #       Slurm single-task runner
│   │       └── runtime.py    #       RuntimeDirectoryManager + helpers
│   │
│   ├── data/                 #   Data core (runtime state)
│   │   ├── __init__.py
│   │   ├── constraint.py     #     Plan / state / result records (plain data, no algorithms)
│   │   ├── geometry.py       #     Structural API every workflow shares (periodic wrap +
│   │   │                    #     crossing ledger, cell spelling rule, continuous view,
│   │   │                    #     Kabsch scalar); no module-level workflow import
│   │   ├── topology.py       #     GroupTable numeric record (explicit member lists)
│   │   ├── state.py          #     PyRAIState (task queues, history, center carriers, read-only API)
│   │   ├── build.py          #     build_state + charge-store binding (composition root)
│   │   ├── task.py           #     TASK_FIELDS: the persisted field set of one task descriptor
│   │   ├── center.py         #     Multi-center partition plan (build_center_plan)
│   │   ├── charges.py        #     ChargeStore (ONIOM electrostatic embedding)
│   │   └── runtime/          #     RuntimeState, Result, serializer
│   │
│   ├── training_data/        #   Training-data layer (no state or skill import)
│   │   ├── __init__.py
│   │   ├── dataset.py        #     TrainingData: canonical JSON load/save, append,
│   │   │                    #     inline payload; refuses shape/atom-order mismatches
│   │   └── permutation.py    #     Legacy 1-based permute-map read/validate/expand
│   │
│   ├── kernel/               #   Maestro + runtime context
│   │   ├── __init__.py
│   │   ├── environment.py    #     Configuration loading and merging
│   │   ├── maestro.py        #     Skill coordination: discovery, resolution, config store
│   │   ├── cache.py          #     Computation cache
│   │   ├── exceptions.py     #     Exception hierarchy
│   │   └── logger.py         #     Logging configuration
│   │
│   ├── qualification/        # Software qualification infrastructure (independent)
│   │   ├── __init__.py
│   │   ├── manager.py            #   Aggregate the four test categories + shared script writers
│   │   ├── runner.py             #   Qualification execution + result reporting
│   │   ├── pmd_env.yaml          #   Environment configuration (YAML)
│   │   ├── verification/         #   Main framework tests + cross-skill contract
│   │   │                         #   tests (skill-specific tests live in each
│   │   │                         #   skill's own tests/)
│   │   │   ├── planner.py        #     Main framework test planning (plan_main_tests)
│   │   │   ├── __init__.py
│   │   │   ├── _parallel_helper.py #    Parallel execution helper
│   │   │   ├── test_pmd.py       #     CLI + parser + schema pipeline tests
│   │   │   ├── test_agent.py     #     Agent + loader tests
│   │   │   ├── test_calculator.py #    Calculator + Composite + PyRAIState data source tests
│   │   │   ├── test_task.py      #     Task Queue expansion tests (planning/task.py)
│   │   │   ├── test_workflow.py  #     Workflow layer tests
│   │   │   ├── test_property.py  #     Planning layer tests (resolver)
│   │   │   ├── test_kernel.py    #     Kernel + maestro tests
│   │   │   ├── test_dispatch.py  #     Dispatch subsystem tests
│   │   │   ├── test_flatten.py   #     Config flatten / merge tests
│   │   │   ├── test_integrator.py #    Integrator tests
│   │   │   ├── test_ensemble_stability.py # Variable-cell NPT stability + frame transport (analytic stress, no engine)
│   │   │   ├── test_npt_stability.py #  The qualification NPT cell-stability verdict (synthetic products)
│   │   │   ├── test_core.py      #     Data core (PyRAIState/driver) tests
│   │   │   ├── test_deploy_log.py #    Deploy + logging tests
│   │   │   ├── test_manager.py   #     Qualification manager tests
│   │   │   ├── test_optimizer.py #     Optimizer tests
│   │   │   ├── test_skill_input.py #   Skill test input generation tests
│   │   │   ├── test_benchmark.py #     Benchmark planner tests
│   │   │   ├── test_skill_packaging.py # Skill packaging tests
│   │   │   ├── test_strorb.py    #     StrOrb handling tests
│   │   │   └── test_qualification_slurm.py # Skill Slurm self-submission qualification tests
│   │   ├── certification/        #   Skill certification tests
│   │   │   ├── planner.py        #     Skill test planning (plan_skill_tests)
│   │   │   ├── handler.py        #     Skill test file discovery + skill section builders
│   │   │   └── converter.py      #     Test YAML → flat .json, flat .json → .input/.config converter
│   │   └── benchmark/            #   Benchmark calibration tests (flat layout)
│   │       ├── planner.py        #     Benchmark planning (plan_benchmark_tests)
│   │       ├── fssh.py           #     FSSH benchmark task module
│   │       ├── gsh.py, lzsh.py   #     GSH and LZSH task modules
│   │       ├── ktdc.py           #     Kinetic TDWC task module
│   │       ├── perf.py           #     Wall-clock perf task module
│   │       ├── timing.py, plotting.py  # Infrastructure only (never task modules)
│   │       └── <task>.yaml + <task>_lz[1-3].yaml  # Unified calibration cases;
│   │                             #     `method` selects the Tully/LZ model
│   │
│   ├── skills/               #   Skill plugins
│   │   ├── __init__.py
│   │   ├── query.py          #     Dynamic skill discovery querying
│   │   ├── wrapper.py        #     pyrai3md_skill command entry
│   │   │
│   │   ├── pyscf_skill/      #     PySCF skill
│   │   │   ├── __init__.py
│   │   │   ├── agent.py      #       CLI entry (python -m)
│   │   │   ├── backend.py    #       Public API (validate, build_state, calculate, local_run)
│   │   │   ├── query.py      #       Skill query module
│   │   │   ├── schema/       #       config, method, normalize, validate, adapter, reference
│   │   │   ├── service/      #       energy, gradient, hessian, optimize, analysis, spectra
│   │   │   │   ├── builder/  #         Workflow graph → dispatch
│   │   │   │   ├── workflow/ #         Profile → WorkflowGraph compiler
│   │   │   │   ├── runner/   #         Local and SLURM execution
│   │   │   │   ├── parser/   #         Output → standardized results
│   │   │   │   └── templates/#         Run script templates
│   │   │   └── tests/        #       *.yaml test cases + test_*.py runtime tests
│   │   │                     #       (discovered by qualification/certification/handler.py)
│   │   │
│   │   ├── molcas_skill/     #     OpenMolcas skill (same structure)
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── backend.py
│   │   │   ├── query.py
│   │   │   ├── schema/
│   │   │   ├── service/
│   │   │   └── tests/
│   │   │
│   │   ├── bagel_skill/      #     BAGEL skill (same structure)
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── backend.py
│   │   │   ├── query.py
│   │   │   ├── schema/
│   │   │   ├── service/
│   │   │   └── tests/
│   │   │
│   │   ├── orca_skill/       #     ORCA skill (dft/tddft, 14 profiles)
│   │   ├── xtb_skill/        #     xTB skill (gfn2/gfn1/gfn0/gfn_ff)
│   │   ├── openqp_skill/     #     OpenQP skill (hf/tddft/mrsf)
│   │   ├── lammps_skill/     #     LAMMPS skill (classical MM, ff_* + qmmm_field/qmmm_energy)
│   │   ├── cp2k_skill/       #     CP2K skill (Quickstep DFT/TDDFPT, 9 profiles)
│   │   ├── qdm_skill/        #     QDM model skill: analytic Tully / Landau-Zener engines
│   │   │                 #     and the wavepacket reference the NAMD calibration legs compare against
│   │   └── nn_pes_skill/     #     Neural network potential (label nn; mlp_eg/mlp_nac2/eg_nac,
│   │       ├── __init__.py   #       PEP 562 lazy export of NNPESBackend: no TensorFlow at import)
│   │       ├── agent.py      #       CLI entry (python -m)
│   │       ├── backend.py    #       load once, then per-step compiled serving; train dispatch)
│   │       ├── query.py      #       Skill query module (manifest, profiles, defaults, requires)
│   │       ├── schema/       #       config, method, hyper, normalize, validate, adapter, reference
│   │       ├── service/      #       calculator, trainer, trainer_data, train_runner
│   │       │   ├── runner/   #         env_check (lazy TF import), local, slurm arms
│   │       │   └── nn_pes/   #         vendored container: models, layers, scaler, datasets,
│   │       │                 #         utils, nn_pes_src{training, hypers, fit, device}
│   │       └── tests/        #       *.yaml cases + test_*.py runtime tests
│   │
│   └── utilities/            #   Shared utilities
│       ├── __init__.py
│       ├── read_tools.py     #     ReadVal, ReadIndex
│       ├── logo.py           #     Startup logo and version
│       ├── bonds.py          #     Bond perception, bond graph, components/offsets, unfolding,
│       │                       #     initial image counts; bond_lib reads
│       │                       #     schema/reference.BOND_LENGTHS
│       ├── structure_formatter.py #  Structure-table text for internal data communication
│       │                       #     (atom/vector rows, charges, matrix, cell, pbc, TurboMole)
│       ├── filesystem.py     #     Path and directory helpers
│       ├── geometric.py      #     The one bond/angle/torsion implementation (BND/AGL/DHD + the atan2-invariant torsion); mass-weighted group centers; radians, degenerate geometry raises
│       ├── selection.py      #     Explicit atom selection: indices, ranges, elements, named groups (no topology)
│       ├── cv.py             #     Collective variables and their coordinate Jacobians (bond, angle, dihedral, cell, strain); evaluate_on_images
│       ├── rmsd.py           #     Aligned RMSD as a collective variable: weighted Kabsch, D^2 and its derivative; the only superposition SVD in the package
│       ├── io.py             #     Input parsing only (coordinates, charges, initcond, xyz frames)
│       ├── lattice.py        #     Periodic-cell math (6-param<->matrix, wrap + image counts, unwrap, certified min-image, half_box_bound); pure functions, no cell state
│       ├── print_results.py  #     Step result formatting for the run log, plus the
│       │                       #     opt_status.txt record, reader and write guard;
│       │                       #     section titles are the constants
│       │                       #     SECTION_ENERGIES / SECTION_KINETIC /
│       │                       #     SECTION_VELOCITIES / SECTION_CONTRIBUTIONS
│       ├── opt_formatter.py  #     The opt_status.txt header/row format (columns, widths)
│       ├── geodesic.py       #     Riemannian interpolation over Morse-scaled internal distances
│       ├── serialization.py  #     json_safe (recursive over dict/list/tuple and
│       │                       #     numpy scalars) + atomic write_json (temp name,
│       │                       #     fsync, os.replace): the result-document writer
│       ├── timer.py          #     Performance timer + the PhaseProfiler per-step wall clock (pyrai3md_step.perf)
│       ├── units.py          #     Unit conversion
│       └── (no tests/ directory: these modules are covered from
│                           qualification/verification/, e.g. test_lattice,
│                           test_bonds, test_geometric, test_constraint_cv,
│                           test_structure_formatter, test_print_results,
│                           test_opt_status_record, test_timer)
│
├── tools/                    # Development utilities
│   ├── deploy.py               #   Remote deployment tool (SSH + rsync; the
│   │                           #   round's own log header is its only progress
│   │                           #   denominator; its poll classifies a
│   │                           #   round PROGRESS / QUIET / TRANSPORT_FAILURE /
│   │                           #   FINISHED and only consecutive QUIET ages the
│   │                           #   idle window - an unreachable host is not
│   │                           #   evidence about the run, until the outage
│   │                           #   outlasts its own budget, when the tool hands
│   │                           #   over the manual rsync route)
│   ├── verify.py               #   Post-round verifier: A records vs results.json,
│   │                           #   B step log vs frame record, C engine output vs
│   │                           #   recorded value (one reader per engine, and a
│   │                           #   directory nothing claims is reported with its
│   │                           #   file inventory), C2 the same for an optimize
│   │                           #   leg's own `opt_status.txt`, C3 the optimizer's
│   │                           #   term columns against its energy column, D the
│   │                           #   variable-cell verdict across its products -
│   │                           #   status rows, recorded metrics, log sentence,
│   │                           #   plus the trajectory's `Lattice=` rows as the
│   │                           #   second source for the shape reading - E carrier
│   │                           #   parity, F the path family's held conditions, G
│   │                           #   the molcas legs (three carriers parsed by the
│   │                           #   skill's own reader and the shape/provenance
│   │                           #   contract), K the three skill carriers' engine
│   │                           #   deck and final energy (files only, never a
│   │                           #   calculation), with --selftest as a planted-error
│   │                           #   negative control
│   ├── refresh_water_fixture.py  #  Single-purpose fixture refresh tool
│   ├── analyze_npt_frames.py     #  Recomputes a run's cell history from md.xyz + md.ensemble
│   ├── benchmark.py              #  Benchmark driver
│   ├── verify_seed_reproducibility.py  #  Cross-build bit-identity check
│   ├── training_data.py          #  Extract labels from finished calculations into the
│   │                             #  canonical training-data JSON; shuffle/split/merge/select
│   ├── TRAINING_DATA.md          #  Training-data tool manual
│   ├── traj_analyzer.py          #  Trajectory completeness, extraction, population and
│   │                             #  geometry analysis (PyRAI3MD / NewtonX / SHARC / fromage)
│   ├── TRAJ_ANALYZER.md          #  Trajectory analyzer manual
│   ├── traj_generator.py         #  Sample initial conditions and generate trajectory jobs
│   │                             #  (Molcas / BAGEL / PyRAI3MD / ORCA / OpenQP / LAMMPS / CP2K)
│   ├── TRAJ_GENERATOR.md         #  Trajectory generator manual
│   ├── multiscale_generator.py   #  Solvent/aggregate environment and MM preparation
│   ├── MULTISCALE_GENERATOR.md   #  Multiscale environment generator manual
│   ├── manual_update.py          #  List files that differ between the repo and the install
│   └── MANUAL_UPDATE.md          #  Manual update helper manual
│
└── dev/                      # Design documentation: 45 files on disk, listed
    #                         # in full below (the index and the tree are one
    #                         # list, so a doc missing here is visible at once)
    ├── ARCHITECTURE.md                # PyRAI3MD Architecture Specification
    ├── CALCULATOR_DEVELOPMENT.md      # PyRAI3MD Calculator Development Guide
    ├── COMPOSITE_DEVELOPMENT.md       # PyRAI3MD Composite Development Guide
    ├── DATA_CORE_DEVELOPMENT.md       # PyRAI3MD Data Core Development Guide
    ├── EXAMPLE_DEVELOPMENT.md         # PyRAI3MD Examples Framework Design
    ├── INPUT_SYSTEM_DEVELOPMENT.md    # PyRAI3MD Input System Development
    ├── INTERFACE_DEVELOPMENT.md       # PyRAI3MD Interface and Entry Architecture Design
    ├── KNOWLEDGE_BASE.md              # PyRAI3MD Knowledge Base
    ├── LOG_DEVELOPMENT.md             # PyRAI3MD Log Design Specification
    ├── MULTICENTER_DEVELOPMENT.md     # PyRAI3MD Multi-Center Unified Computation Foundation
    ├── PLANNING_DEVELOPMENT.md        # PyRAI3MD Planning Development Guide
    ├── QUALIFICATION_DEVELOPMENT.md   # PyRAI3MD Qualification Framework Design
    ├── SKILL_APP_DEVELOPMENT.md       # Skill Application Development Guide
    ├── SKILL_TEST_DEVELOPMENT.md      # Skill Test Development Guide
    ├── TODO.md                        # PyRAI3MD TODO List
    ├── WORKFLOW_DEVELOPMENT.md        # PyRAI3MD Workflow Development Guide
    ├── UTILITIES_DEVELOPMENT.md       # PyRAI3MD Utilities Development Guide
    ├── methods/                # Method-specific development guides
    │   ├── ADAPTIVE_DEVELOPMENT.md    # PyRAI3MD Adaptive Sampling Design
    │   ├── CELL_OPT_DEVELOPMENT.md    # PyRAI3MD Variable-Cell Geometry Optimization (Cell
    │   │                              #   Relaxation) Design
    │   ├── CONSTRAINT_DEVELOPMENT.md  # PyRAI3MD System Constraint, Frozen Atom and Bias Potential
    │   ├── ENSEMBLE_DEVELOPMENT.md    # PyRAI3MD Periodic Boundary, Ensemble and Thermostat/Barost
    │   ├── FSSH_DEVELOPMENT.md        # PyRAI3MD FSSH NumPy + Numba Optimization Design
    │   ├── GSH_DEVELOPMENT.md         # PyRAI3MD GSH (Zhu-Nakamura) Global Surface Hopping Develop
    │   ├── LZSH_DEVELOPMENT.md        # PyRAI3MD LZSH (Landau-Zener) Surface Hopping Development
    │   ├── MD_DEVELOPMENT.md          # PyRAI3MD Molecular Dynamics (NAMD) Design
    │   ├── NUM_HESS_DEVELOPMENT.md    # PyRAI3MD Numerical Hessian and Vibrational Analysis Design
    │   ├── ONIOM_DEVELOPMENT.md       # PyRAI3MD ONIOM2 / ONIOM3 Multilayer Design
    │   ├── OPTIMIZE_DEVELOPMENT.md    # PyRAI3MD Geometry Optimizer Design
    │   ├── PATH_DEVELOPMENT.md        # PyRAI3MD Path Family (MEP / IRC / NEB) Design
    │   ├── PBC_DEVELOPMENT.md         # Periodic Boundary Conditions (PBC) Design
    │   ├── QMMM_DEVELOPMENT.md        # PyRAI3MD Additive QM/MM (ESPF) Design, incl.
    │   │                              #   the multi-center charge iteration (#451)
    │   ├── QMMM_NN_drafts.md          # PyRAI3MD NN-QM/MM Design Draft: one
    │   │                              #   total model with an isolated-molecule
    │   │                              #   branch and an environment-response branch
    │   ├── SEARCH_DEVELOPMENT.md      # PyRAI3MD Hyperparameter Grid Scan Design
    │   └── TRAIN_DEVELOPMENT.md       # Neural Network Potential Training Design
    ├── skills/                 # Skill-specific implementation guides
        ├── BAGEL_SKILL_DEVELOPMENT.md # BAGEL Skill Development Guide
        ├── CP2K_SKILL_DEVELOPMENT.md  # CP2K Skill Development Guide
        ├── LAMMPS_SKILL_DEVELOPMENT.md # LAMMPS Skill Development Guide
        ├── MOLCAS_SKILL_DEVELOPMENT.md # Molcas Skill Development Guide
        ├── NN_PES_SKILL_DEVELOPMENT.md # NN_PES Skill Development Guide
        ├── OPENQP_SKILL_DEVELOPMENT.md # OpenQP Skill Development Guide
        ├── ORCA_SKILL_DEVELOPMENT.md  # ORCA Skill Development Guide
        ├── PYSCF_SKILL_DEVELOPMENT.md # PySCF Skill Development Guide
        ├── QDM_SKILL_DEVELOPMENT.md   # QDM Skill Development Guide
        ├── STRESS_DEVELOPMENT.md      # PyRAI3MD Periodic Stress Interface Design (CP2K, LAMMPS, x
        └── XTB_SKILL_DEVELOPMENT.md   # xTB Skill Development Guide
```

---

# 3. Responsibilities

## 3.1 Kernel

Kernel is the permanent core of PyRAI3MD.

Responsibilities:

- Maestro (skill coordination, discovery, resolution, config store)
- Plugin discovery
- Runtime management
- Workflow execution
- Logging
- Cache
- Exception handling
- Lifecycle management

Kernel must never contain software-specific knowledge.

Kernel must never import any specific Skill.

All capabilities must be discovered dynamically.

---

## 3.2 Schema

Schema defines the communication protocol of PyRAI3MD.

Schema does NOT define backend-specific parameters.

Schema consists of

- Configuration Models (Control, System, Dynamics, Adaptive — nested in Config)
- Adapter (raw dict → typed Config, type conversion, env defaults)
- Normalization (derived quantity computation: natom, total_electron, nalpha, nbeta, npair, etc.)
- Validation Rules (consistency checks: task, system, electronic, dynamics, adaptive, skill_name)
- Task Objects
- Result Objects
- Reference Data (periodic table, constants)

The framework Config is a nested Pydantic model with four submodels (control, system, dynamics, adaptive).
The control submodel carries the `driver` keyword (`Control.driver`, default `'main'`; may be
`'skill'` or a concrete skill name) which fixes the execution model: `'main'` lets the framework
resolve the workflow from jobtype, while any other driver hands the whole jobtype to the skill
(framework single shot, see Section 3.4.3 and WORKFLOW_DEVELOPMENT.md).

The framework PyRAIConfig keeps its own submodels (Control/System/Dynamics/Adaptive) for
workflow attribute access. Task configs are **not** extended PyRAIConfig objects: each task
carries a plain dict `{'skill_config': <skill flat dict>, 'results': {}}` built directly by
`build_task_queue` (planning/assembly.py). The former `build_pyrai_config()` helper is deprecated and
no longer used by the framework. The skill backend runs its own schema pipeline
(build_config, normalize, validate) on the flat dict.

Schema only defines

"What the program understands."

It never defines

"How a backend performs computation."

---

## 3.3 Skills

Skills are independent scientific capabilities.

Examples

- Electronic Structure
- Optimization
- Machine Learning
- Dynamics
- Analysis

Each Skill is implemented as a Plugin.

Each Skill owns

- supported capabilities
- backend knowledge
- default configuration
- validation
- execution strategy
- parser

Skills never modify Kernel.

Kernel discovers Skills automatically.

"Execution strategy" is deliberately broad: the framework's only contact is
`backend.calculate()`, and what that costs is the skill's business. The LAMMPS
skill shows two strategies for one engine — start the program per call, or hand
each step's coordinates and box to a **resident service process** that holds a
live LAMMPS instance and answers over a Unix socket (`python_enabled`;
skills/LAMMPS_SKILL_DEVELOPMENT.md §11). Both produce the same result
vocabulary, and the switch changes no framework code.

---

### 3.3.1 Pre-commit review contracts (2026-11-12)

- Each skill's package `__init__.py` exports its backend through PEP 562
  `__getattr__`; importing `<skill>.query` does not load `<skill>.backend` or
  its service modules. Discovery warns about a skill it cannot import instead
  of silently dropping it.
- Each skill owns `service/read_support.py`; there is no framework
  `pyrai3md.skills.read_support` module.
- A requested property whose block is absent raises: a zero array is reserved
  for a property nobody requested.
- Planning reads the pure optimizer declarations in
  `schema/optimizer_spec.py` and its fix mask from `data/constraint.py`; it has
  no import edge to `workflow/`.

## 3.4 Planning Layer

The Planning layer determines WHAT to compute. It resides in `planning/` and is responsible for:

- Parsing jobtype tokens into capability name strings
- Expanding advanced jobtypes (optimize → [energy, gradient], md → [energy, gradient, nac])
- Resolving dependency relationships between capabilities
- Distributing per-skill capabilities to each backend via Maestro
- Enforcing multi-backend constraints (see below)
- Deriving the adaptive label plan: the reference queue is expanded from
  its own method list and `molecule.highref/midref`, and the label
  source is chosen by atom-set equality with the theory electronic
  provider's region (layer term or composite total). The plan travels as
  `label_plan` on the reference tasks; assembly turns it into
  `label_channels` and the adaptive workflow only reads them.

Planning terminates at PyRAIState construction. Planning never:
- Executes calculations
- Imports a skill backend (it may read a skill's manifest through
  `skills/query.py`, e.g. `role_of`, which is the declaration channel)
- Instantiates backends
- Queries Maestro at runtime
- Builds CalculationRequest (this is a skill-internal concept)

Multi-backend constraints (when n_theory >= 2 or n_reference >= 2):
- Hessian/frequency is forbidden (produces incorrect results)
- MD advanced properties (nac, soc, overlap, etc.) apply only to the first theory + first reference pair; other backends are restricted to energy + gradient

See PLANNING_DEVELOPMENT.md Section 5.5 for details.

### 3.4.1 Capability Names

Planning resolves jobtype tokens into plain capability name strings. No intermediate objects are created:

| jobtype token | capability name | skill return format |
|---|---|---|
| energy | 'energy' | float (Hartree) |
| gradient | 'gradient' | ndarray (natom, 3) |
| nac | 'nac' | ndarray (nnac, natom, 3) |
| soc | 'soc' | ndarray (nsoc,) |
| frequency | 'hessian' | ndarray (3*natom, 3*natom) |
| absorption | 'transition_dipole' | ndarray (nstate, 3) |
| dipole | 'dipole' | ndarray (3,) |
| transition | 'transition_dipole' | ndarray (nstate, 3) |
| overlap | 'overlap' | ndarray (nstate, nstate) |

The full capability set is `CAPABILITY_NAMES` in `planning/resolver.py`
(energy, gradient, hessian, nac, soc, dipole, transition_dipole, overlap).
`overlap` is derived by the dynamics resolution (fssh + dcm), not from a jobtype token.

### 3.4.2 Calculator

Calculator resides in `workflow/calculator.py` and performs single-request computations.

It is a high-level functional object that owns state, backend access, and result processing.

- `Calculator(Dispatcher)` —
  `__init__(state=None, runtime_idx=None, dummy_backend=None, multitask=None, remotetask=None, local_executor=None, executor_spec=None)`;
  iterates the state task queue (`state.iter_tasks()`) of the active backend role
  (theory/reference), derives each
  task's geometry from the live `state.pmd.system` (through the state read-only
  pass-through properties) using pre-stored `region_indices`, calls
  `backend.calculate()` once per task, projects and aggregates composite results,
  and writes them back to `state.pmd.results` via `state.set_result()`
- `Calculator.sync()` — pushes `state.pmd.system` fields (`SYNC_SYSTEM_FIELDS`)
  to every task configuration; never touches calculation/method/control sections
- `aggregate_results(state, category=None)` (in `workflow/multiscale/composite.py`) — pure function
  aggregating task results by contribution (Σ contribution × property), optionally
  restricted to one backend role (`category='theory'`/`'reference'`);
  what a composite queue must look like is checked by `validate_plan`, per scheme and
  structurally - one classical term for qmmm, one probe per QM center, a charge duty
  that carries 0 like the field does. The fixed-length pattern table and
  `validate_contributions` were retired; `validate_plan` is the live guard
- `adapt_skill_result(skill_dict)` — converts skill dict to dict-like Result
- `_apply_skill_geometry` / `_apply_skill_velocities` / `_apply_skill_cell` —
  the three engine-ownership channels. When a skill's engine propagated the
  nuclei, moved the box or ran its own iteration, it returns `fin_coord`
  (Angstrom), `fin_velocity` (Bohr/au) or `fin_cell` (six cell parameters or
  nine flattened lattice components); each applier pops the key from the result
  and from the stored task config, treats an empty or absent value as "the
  engine does not own this", raises on a malformed one, skips a value equal to
  the current state, and otherwise is the single writer into
  `state.pmd.system` (`cell` also derives `lattice_vector` and `volume` in one
  update, and a `fin_cell` on a system with no `pbc` is refused as an input
  error). A composite of several sub-calculations rejects all three. See
  KNOWLEDGE_BASE.md Q43 and SKILL_APP_DEVELOPMENT.md §11.

The Calculator never inspects the workflow type (ONIOM/fragment/qmmm): it only executes
the Task Queue. Composite meaning is expressed by task `contribution` coefficients.

Calculator serves as both the default single-shot runner and the computation engine for
iterative workflows, following the loop contract `run()` → `state.update_system({...})`
→ `sync()` (see CALCULATOR_DEVELOPMENT.md). The `resolve_calculator()` factory function
is retained for convenience but is not used by the main pipeline.

The workflow layer never builds CalculationRequest. The Skill backend constructs it internally during `__init__(config)` by reading the task config dict.

### 3.4.3 Workflows

Workflow is a high-level scientific procedure.

Workflow never depends on specific software.

Workflow composes Capabilities.

Workflow receives a Calculator that provides access to PyRAIState via `calculator.get_state()` / `calculator.update()`. Workflow never:
- Performs planning (resolving input, deciding capabilities, parsing jobtype)
- Queries Maestro
- Imports the planning layer (one registered exception: the three
  task-descriptor selectors imported from `planning/task.py`, see
  WORKFLOW_DEVELOPMENT.md Section 13)

**One module per workflow class at the top level; execution components live
in named subpackages.** `workflow/records/` (what a run leaves behind:
`trajectory.py`, `deviation.py`, `ml.py`, `output.py`, `diagnostics.py`),
`workflow/grid/` (`plan.py`, `record.py`: the plan and the read-back of a
hyperparameter scan, reached only through a skill's query interface),
`workflow/multiscale/` (`regions.py`, `composite.py`, `charges.py`) and
`workflow/dispatch/` (the runtime axis `dispatcher.py`/`scheduler.py`/
`launcher.py` plus the term axis `terms.py`/`term_worker.py`) declare no
`Workflow` subclass and appear in neither `WORKFLOW_REGISTRY` nor
`JOBTYPE_MAP`. The two component families stay disjoint: the MD chain reads
`records/`, the Calculator and the composite layer read `multiscale/`, and
the term axis is what the Calculator drives per backend term.

**Machine learning: the skill computes, the framework receives and
reports.** A training or test-set evaluation is one call of the skill
backend under the single-shot path, so the fit, the serving over a test set,
every error statistic and the unit conversions of both live in the skill
(`skills/nn_pes_skill/service/trainer.py`, `predictor.py`,
`train_record.py`). What the framework owns is the *reception* of the record
(`workflow/records/ml.py`: is the record the jobtype promised actually
there?), the report of what the skill returned (`utilities/print_results.py`),
and the comparison of many records (`workflow/grid/`: rank the fit errors a
scan collected). Nothing in the framework computes a deviation, an error or a
convergence judgement of a model.

The framework's one participation in the *input* is data preparation, and it is
still not a computation: `&train permutation_map` is expanded by
`workflow/train.py::prepare_training_data`, invoked from the
`Workflow.prepare` lifecycle hook in `workflow/__init__.py::execute` before
`Calculator.initialize`, so the backend is constructed from frames and never
from a map. The container and the map reader live in `training_data/`
(`dataset.py`, `permutation.py`); the skill accepts the result as either an
inline `train_dataset` mapping or a `train_data` file
(`dev/methods/TRAIN_DEVELOPMENT.md` section 4.4).

**The execution-graph node name is declared, never derived.** Every workflow
class sets the class-level `Workflow.name`, and the planner's keyword tables
(`WORKFLOW_SECTIONS`, `WORKFLOW_NODE_DECLARATIONS`, `WORKFLOW_BASE_CHAINS`)
are keyed on that token; a name derived from the class spelling would miss
the tables and silently resolve no input keywords. The verification suite
guards both directions: no registry class without a declared name, and no
table key that no node can ever reach.

Examples

- AIMD (ab initio / nonadiabatic molecular dynamics)
- Geometry Optimization
- Active Learning
- Train / Prediction (one call each of the skill's training or serving
  profile, each its own node so its dispatch keywords come from `&train` /
  `&prediction`; `dev/methods/TRAIN_DEVELOPMENT.md` sections 2 and 8.1-8.2)
- Hyperparameter Grid Search (a planned batch of training calls, one child
  runtime per point, ranked afterwards; `dev/methods/SEARCH_DEVELOPMENT.md`)
- SingleShot (single-shot execution; `workflow/single.py`,
  `workflow_kind='single'`, node name `single`): returned by
  `resolve_workflow` whenever `control.driver != 'main'` (driver =
  `'skill'` or a skill name, validated by `_validate_driver` in
  `planning/assembly.py`). The jobtype reaches the skill as-is and the skill plans
  the whole job internally, while the framework executes exactly one
  `calculator.run()` and settles the outcome by `state.jobtype`. The adaptive loop does not evaluate in its own process
  (`needs_backends=False`): each round materializes its MD, reference and
  training stages as child runtimes. The training child resolves the
  `Train` node, so it keeps the ML record verdict a plain train run gets;
  the parent additionally checks `ml_verdict` before using the model.

Composite schemes (ONIOM2, ONIOM3, QMMM) are contribution patterns expanded in
`planning/task.py` and aggregated by the pure function `aggregate_results()`
(`workflow/multiscale/composite.py`); the Calculator writes the combined dict
back to `state.pmd.results`. The former `COMPOSITE_PATTERNS` registry and the
`project()` placeholder were retired.

**The AIMD loop owns the cell, and says so in one place.** A variable-cell NPT step
is not a special workflow: `workflow/md.py` calls `apply_ensemble()` around the split
velocity-Verlet (`stage='pre'` and `stage='post'`), and the barostat family in
`workflow/integrator/ensemble.py` decides what a step does to the box. Three
implementations exist and they differ in kind, not in degree: `_berendsen` is a
weak-coupling contraction with no cell momentum (and it does read `barostat_tau`),
`_scr` is stochastic cell rescaling in its **isotropic** form only, and
`_extended_cell` carries the cell momenta for Parrinello-Rahman and MTTK. When
`system.barostat_velocity` is non-zero, `verlet.py` scales coordinates and cell
together by `exp(v_eps * dt)` through `_scale_cell`, which writes `lattice_vector`,
`cell` and `volume` in a single `state.update_system()` and re-triangularizes on the
way (`_retriangularize`), because the representative that travels is the
positive-diagonal one with the sign carried by the tilt. Everything the loop
publishes about the cell goes through that one update: the `md.ensemble` product
columns (`geps_*` / `peps_*` / `veps_*` and friends) and, under
`control.debug: true`, one `Lattice=` per recorded frame.

Two boundaries belong to this design rather than to the engine skills. First, a cell
that wanders is stopped by the framework, not by the engine:
`workflow/records/diagnostics.py` splits "numerically impossible" (non-positive determinant,
non-finite lattice or extended degree of freedom) from "one frame out of bounds", and
the cumulative strain gate on `barostat_strain` raises `CellDivergenceError` by name
-- the reference is the run's starting box, so history-window trimming cannot move
the goalposts. Second, the framework barostat and an engine's own barostat never both
own one run: `md` with `control.driver: main` resolves to the per-step `ff_gradient`
profile, while the engine-integrated profiles (`ff_nve` / `ff_nvt` / `ff_npt`) are
reachable only through an explicit standalone profile, which is why a framework NPT
run reports no `fin_cell` while a LAMMPS-driven NPT run does. `optimize_cell` says
the same thing twice and both halves are one design: planning derives the `stress`
capability so the whole-system tensor reaches the skill, and the framework optimizer
carries the six movable lattice components as part of the optimized vector, driven by
the stress conjugate to them (`OptimizeConfig.optimize_cell`;
`dev/methods/CELL_OPT_DEVELOPMENT.md` section 5). An engine that relaxes the box
on its own does it through its own cell-relaxing profile and box keyword instead, and
the two routes are never asked for at once (`workflow/__init__.py::resolve_workflow`
refuses the pair by name).

A third boundary is the one a center's geometry crosses (#316): the framework, not the
engine, decides whether the region a center is cut from still holds together. Every recorded
frame of a periodic run with more than one center is measured against half the shortest cell
extent (`records/diagnostics.py::_check_region_reconstruction`), because the crossing counts
it is rebuilt from are kept in fractional units and a variable-cell run moves the ground under
them. What the framework cannot decide is which low layer is able to answer for a slice: a
skill whose topology comes from a fixed `<project>.data` template refuses a center smaller
than that template at backend initialization, measured rather than assumed, so a multi-center
periodic run needs a low layer that evaluates a cluster. Details and the two real-engine
examples: `dev/methods/PBC_DEVELOPMENT.md` section 4.2.

### 3.4.4 Result

Result is a dict-like container that stores all computation outputs uniformly.

- Dict-style access: `result['energy']`, `result['gradient']`
- Attribute-style access (backward compatible): `result.energy`, `result.gradient`
- Serialization: `to_dict()` / `from_dict()` / `copy()`
- Lists and tuples are automatically converted to numpy arrays

`energy` is state-resolved and every entry is an absolute total energy
on the same reference: index 0 is the ground state and index k the k-th
excited state, each comparable with `gradient[k]` and with the kinetic
energy the workflow records next to it. Skills whose engine only
reports one of those numbers have to reconstruct the rest, which is why
the ORCA skill resolves its reference from the engine's per-root
sections rather than from the single energy printed with the gradient
(see `skills/ORCA_SKILL_DEVELOPMENT.md` §5.2). A reference mismatch is
invisible in state energy differences, so it shows up only as a spurious
NVE total-energy drift.

### 3.4.5 Data Flow

The complete data flow from jobtype to result:

```
jobtype string
    ↓ build_config(raw_config) → Config
    ↓ normalize(config) → validate(config)
    ↓ build_state(config) → PyRAIState (task queues with API paths, no backend instances)
        maestro.resolve_skills(config) → skills
        resolve_task_queue(config, skills) → task descriptors (composite expansion)
        per task: plain dict task config {'skill_config': flat, 'results': {}}
                  + API path string + static_id → state.add_task()
state.theory/reference (task queues: one API path string + task config dict per task)
    ↓ execute(state) → creates Calculator(state)
    ↓ Calculator.initialize() → resolve_api() → instantiate backends
    ↓ returns Workflow instance (iterative) or calculator.run() (single-shot)
runner.run()
    ↓ Calculator.run() → for each task: derive region geometry → backend.calculate()
    ↓   → task results written to task config → composite aggregation → state.pmd.results
Result (dict-like container with all computed properties)
```

Three-layer isolation: planning (via Maestro), workflow/, and skill backend are completely independent. Communication happens through state.theory/reference (task queues + task config dicts).

---

## 3.5 Interfaces

Interfaces provide user entry points.

Supported interfaces

- CLI
- JSON
- YAML
- MCP
- REST
- Future GUI

Interfaces never perform scientific computation.

Interfaces only

Input

↓

Dictionary

↓

Schema

↓

Maestro

↓

Workflow / Skill

---

## 3.6 Utilities

Utilities contain only reusable generic functions.

Examples

- IO
- Filesystem
- Math
- Units
- Timer
- Profiler
- Parallel

Utilities must never contain backend-specific code.

Design semantics for this tree live in `dev/UTILITIES_DEVELOPMENT.md` (the authority); this
section records the tree and the edges. The one permitted package-level upward edge for data is
`utilities.* -> pyrai3md.schema.reference`: that module is pure data and must contain no import
statement. `bonds`, `io` and `geodesic` use it. `print_results` has two pre-existing lazy imports
into `data.center` and `workflow.records.deviation`; they are recorded as deferred extractions in
the utilities guide rather than counted as a second rule.

`schema/reference.py` stays import-free; the public unit constants it used to
re-export are imported by `schema/__init__.py` directly from
`utilities.units`.

---

## 3.7 Constraint Subsystem: Call and Dependency Map

How `workflow/constraints/` is actually reached from the framework. Design semantics live in
`dev/methods/CONSTRAINT_DEVELOPMENT.md` (the authority); this section only records the edges, so
a reader can find a function's upstream and its call frequency before changing it. Line numbers
are from the 2026-10-01 working tree (CHANGELOG #271), which is the round that dissolved
`schema/constraint.py` and `schema/legacy_adapter.py` into the four framework schema modules and
moved the classic `&constraint` text out of the interface layer.

Import direction is strictly downward; no framework module below the call site imports upward.

```
interface/parser.py ──> (nothing constraint-specific)              flat `&section` split only
schema/adapter.py ─────> _read_constraint_section
qualification/certification/constraint_writer.py ─────> write_constraint_section
                            (the classic &constraint grammar, both directions: it is the only
                             input spelling in text, YAML and JSON alike - one list element is
                             one object, and a record list at input is refused by name)
schema/adapter.py ─────> adapt_legacy_fields                      (19 flat keys, discouraged)
schema/config.py ──────> System.constraints: Optional[SystemConstraintConfig]
schema/normalize.py ───> build_plan ──> schema/validate.check_{periodic,ramp_shape,ramp_positive}
schema/normalize.py ──> data/constraint, data/topology, utilities/{selection,units,bonds}
workflow/constraints/runtime.py ──> controller ──> {schedule,bias,wall,shake,projection,
                                        metadynamics} ──> utilities/{cv,rmsd}
workflow/constraints/state_adapter.py ──> data/constraint.EvaluationContext, utilities.units
workflow/constraints/state_adapter.py ──> constraints.schedule.endpoint_variants  (ramped walls)
kernel/logger.py ──> state.constraint_plan() ──> &constraint plan block (ids + notes)
workflow/constraints/state_adapter.py ──> constraints/{images,wall}   (periodic guards)
workflow/constraints/images.py ──> utilities/{bonds,geometric,lattice}, utilities.units
```

Added by the periodic round (CHANGELOG #266), with the edges the map did not carry before:

| Module | Line | Entry point | Duty |
| --- | --- | --- | --- |
| `utilities/lattice.py` | :468 | `min_image_displacement` | nearest image by candidate search with an exhaustiveness certificate |
| `utilities/geometric.py` | :286, :379, :427 | `expand_components`, `evaluate_on_images`, `require_same_component` | unfold one connected component along its bond graph; evaluate a CV on the unfolded array |
| `utilities/selection.py` | :264 | `resolve_index_rows` | ordered 3- and 4-atom rows (an angle's vertex and a torsion's ends are not sortable) |
| `schema/validate.py` | :1122 | `check_periodic` | periodic refusal is per block kind, and the plan build calls this same function |
| `schema/normalize.py` | :1244 | `normalize_hard_constraints` | `kind: angle` / `dihedral` constraints enter the plan (the name is what the dissolved `schema/constraint.py` called `normalize_constraints`, since the pipeline step of that name already lived here) |
| `workflow/constraints/shake.py` | :537, :485 | `solve_conditions`, `condition_redundancy` | multi-condition sweep; dependent rows dropped by rank |
| `workflow/constraints/images.py` | :108, :185, :242, :258 | `read_expansion`, `require_half_box`, `group_center`, `fold_back` | the periodic seam: perceive, unfold, gate, and fold with the crossing counts |
| `workflow/constraints/state_adapter.py` | :212, :317, :435 | `require_usable`, `_validate_images`, `solve_geometry` | periodic guard asks per variable: the bond graph decides whether a cv has an image (#274); a wall's "member starts inside" rule is a statement about the starting geometry only (#275); one expansion read per evaluation |
| `workflow/constraints/wall.py` | :377 | `periodic_check` | the four input-time wall checks inside a cell |
| `workflow/constraints/controller.py` | :659, :664, :698 | `has_shape_constraints`, `_constraint_conditions`, `_bind_constraint_targets` | route a plan with an angle or torsion to the condition solver |
| `workflow/constraints/runtime.py` | :192, :202 | `solve_geometry`, `images.fold_back` | solve unfolded, write back wrapped **and** the image counts together |

Added by #267, the round that cleared the capabilities needing no machine:

| Module | Line | Entry point | Duty |
| --- | --- | --- | --- |
| `utilities/bonds.py` | `perceive_bonds_focus`, `bond_graph`, `bond_pairs_by_type` | one bond list per geometry for images, constraints and a `bond_type` selector alike; the non-periodic branch lives in `perceive_bonds` itself and the unknown-pair refusal in the topology service, because the two risks differ |
| `workflow/constraints/images.py` | :108 | `read_expansion(view, focus=)` | the focus set comes from the plan (`state_adapter._image_focus:248`), which is what took 85.6 ms/step at 300 atoms down to 4.1 ms |
| `schema/normalize.py` | :508, :533 | `_Normalizer.graph`, `bond_type_pairs` | lazy graph, built only when a record asks which atoms are bonded; the structure arrives through `build_plan(coordinates=, lattice=)` from `schema/normalize.py` and `data/state.py:PyRAIState.constraint_plan()` |
| `workflow/constraints/metadynamics.py` | :79, :105 | `_hill_terms`, `bias_value` | one Gaussian kernel feeding both the energy and the gradient, plus the accumulated bias a well-tempered height is scaled against |
| `workflow/constraints/controller.py` | :437 | `commit` (no-hill branch) | a flattened bias spends the step token and counts `zero_hills:<id>` instead of writing a hill `HillRecord` rightly refuses |
| `workflow/constraints/shake.py` | :863 | `solve_velocity_conditions` | the velocity half over a mixed bond/angle/torsion set, reached through `controller._tangent_conditions:600` |
| `utilities/geometric.py` | :213, :235, :279, :377 | `bond_hessian_core`, `angle_hessian_core`, `torsion_hessian_core`, `cv_hessian_block` | the second derivative of a variable, in the same atom order as the gradient it must be the derivative of; the torsion by slot insertion, so symmetry and translational invariance are structural |
| `workflow/constraints/bias.py` | :129 | `hessian` | `k (J J^T + chain e Q)` for one harmonic bias, dense (3N, 3N) |
| `workflow/constraints/wall.py` | :273, :311, :336 | `_curvature_terms`, `_face_powers`, `hessian` | per-face slope and curvature of the one-sided wall, chained through the group weights for a centroid target |
| `workflow/constraints/metadynamics.py` | :189 | `hessian` | the chain rule over the declared variables of a history potential: `sum_jk V_jk J_j J_k^T + sum_j V_j Q_j`, with `V_j`, `V_jk` from the dimensionless `z = (s-c)/sigma` |
| `workflow/constraints/controller.py` | :387 | `hessian_block` | the plan's additive curvature (bias, wall, history potential), and the named refusal of an aligned-RMSD history or any variable with no derived `Q`; a fix or a constraint contributes nothing, being a projection rather than a potential |
| `utilities/geometric.py` | `group_center` | the one mass-weighted center, now also behind `operations.velocity_correction` (D-04), with the division placed so the integrator's number stays bit-stable |

`constraints/images.py` is a leaf of the constraint package in the sense of rule 1 below: it reads a
state-shaped object by attribute (`view`), imports neither `data.state` nor `schema.config`, and is
imported only by `state_adapter` and `runtime`. A non-periodic view takes the identity path of
every function in it, which is what keeps these edges off the numbers of a molecule-in-a-box run.

The ten framework-side call sites:

| Caller | Line | Entry point | Duty |
| --- | --- | --- | --- |
| `workflow/calculator.py` | :62, :583, :649, :1342 | `runtime.finalize` | sum bias/wall/metadynamics into energy and gradient, write flat `energy_contributions` keys; **both** `_run_single` and `run_in_process` go through one helper |
| `workflow/integrator/operations.py` | :56, :187 | `runtime.constrain_state` | `apply_constrain` delegates: freeze mask plus SHAKE position solve |
| `workflow/md.py` | :49, :568 | `runtime.bind` | capture the reference geometry **before** the step loop |
| `workflow/md.py` | :636 | `runtime.commit` | deposit at the accepted step (metadynamics deposits once per step) |
| `workflow/md.py` | :851 | `_write_runtime_file` | write `<title>.rt` on the `md.checkpoint` cadence |
| `workflow/optimize.py` | :35, :165, :203 | `runtime.bind`, `runtime.constrain_state` | constrained optimization, plus one evaluation after the last projection |
| `workflow/optimizer/base.py` | :168, :255, :294, :297, `convergence` | `projection.build_fix_mask`, `projection.declared_mask` | project the frozen components out of the trial step, after the trust radius and before the step is recorded; the gradient half of the stopping criteria is asked only of the components no declaration owns (#282) |
| `workflow/integrator/ensemble.py` | :219, :243-246, :256 | `controller.statistics`, `state_adapter.read_state` | rank-based thermal degrees of freedom (see D-17: rebuilt per call) |
| `data/runtime/serializer.py` | :431, :492, :496, :554 | `_constraint_payload`, `_check_constraint_payload` | store version + fingerprint + history; refuse a restart whose input changed |
| `data/state.py` | `constraint_plan` | `schema.normalize.build_plan` | `PyRAIState.constraint_plan()` builds and caches the plan once, and hands it `md_step=` from `pmd.dynamics.step` - the value that closes a parameter ramp whose `stop` the input left open |

Per-jobtype order: **md** `bind` → loop {propagate → `constrain_state` → calculator (`finalize`) →
`_dof` → `commit` → ensemble post half step → `constrain_state` again when a barostat moved the cell
(`md.py` phase `constrain_after_cell`, #276) → advance the step counter → trajectory frame, run-log
block and runtime file, in that order and all three describing the step that has just finished
(#281)}. Step numbers are absolute: a resumed process starts at
`runtime.current_step + 1` and its `total_steps` is that number plus the steps it was asked to add,
and a step that writes a checkpoint also writes a frame, so a run split at its checkpoint cadence
produces the same frame list as the same run in one process. **optimize** `bind` → loop {calculate →
`apply_constraint` →
`constrain_state`} → one final evaluation; **single point** `finalize` only; **restart**
`StateSerializer.load` → `_check_constraint_payload` → the same `execute` path.

Default closure (DC-1) sits on every edge: each `runtime` entry reads `state.constraint_plan()` and
returns the caller's own object when the plan is empty - the velocity array by object identity, not
an equal copy - `finalize` writes no key, `_dof` uses the classic formula, and the payload block is
`{}`. Measured: a run with `checkpoint: 2` and no constraints has a byte-identical
`.md.energies` next to the same run without the keyword (§20 of the design document).

Three rules the shape depends on:

1. Only `constraints/runtime.py` and `constraints/state_adapter.py` may know a `PyRAIState` exists.
   `controller`, `shake`, `wall`, `bias`, `metadynamics`, `schedule` and `projection` take
   `(coords, state, context, masses)` and nothing else; `test_constraint_state_adapter.py` guards
   the negative side (no import of the state class, config models, parser or calculator).
2. One implementation per piece of mathematics. `utilities/geometric.py` holds
   `bond_core`/`angle_core`/`torsion_core` and `utilities/cv.py` delegates to it;
   `utilities/rmsd.py` is the only Kabsch and `operations.kabsch` is its scalar wrapper;
   `utilities/bonds.py::bond_lib` is the only bond-perception criterion, and its table is `schema/reference.py::BOND_LENGTHS`; its heavy-atom entries carry the SINGLE-bond distance: perception tests `distance < bond_lib * 1.2`, so a table keyed on the shortest member of each series (`C-C` as 1.20) puts every ordinary single bond out of reach and splits a molecule into components (D-28). Element radii live in `schema/reference.py` (`VDW_RADII`, and `COVALENT_RADII` only for the geodesic metric's rest lengths); neither is a perception criterion.
3. Unit conversion happens once, at binding. Input is Angstrom and degree, internals are Bohr and
   radian, and `state_adapter.py` is the single conversion entry point.

Two known structural items came out of this map and are registered as **D-16** (a dead
`runtime.restore` whose payload keys do not match the writer; deletion, not repair) and **D-17**
(`_dof` rebuilds the controller per call with no measured cost yet) in section 18 of the design
document.

---

# 4. Maestro

Maestro is the central skill coordination component of PyRAI3MD.

Every dynamic capability must be obtained through Maestro.

Maestro maintains

- Skill Cache (populated via query mechanism)
- Skill Config Store (skill runtime configs from pmd_env.yaml)
- Skill Class Registry (legacy backend class registration)

### 4.1 Skill Discovery via Query Mechanism

Maestro is populated through the query mechanism:

```
Environment.initialize()
    ↓
Maestro.populate()
    ↓
skills/query.py → discover_all()
    ↓
For each skill: import {skill}.query → get_metadata()
    ↓
Cache: {label: {skill_name, name, skill_path, backend_class, requires, ...}}
```

`requires` lists the third-party packages a skill needs beyond the core
install (only `nn_pes_skill` declares any: `tensorflow`). The qualification
planner turns a requirement that cannot be imported into a diagnostic skip
naming the package, so an optional stack that is absent is never mistaken
for a passing test, and never for an uninstalled skill.

`subsections` (also from `get_metadata()`) names any input blocks a skill
reads beside its own `&<label>` block — `nn_pes_skill` declares `eg`, `nac`
and `soc`, one hyperparameter block per ensemble it can fit. The framework folds each declared block into the skill config under
the same key (`schema/adapter.py:_merge_one_skill_config`) and renders it
back out of a yaml or agent config into its own block
(`qualification/certification/handler.py:build_subsection_blocks`); which
keywords live inside one, and what they mean, stays with the skill.

After populate(), all skill resolution goes through cached data.
No redundant query calls.

### 4.2 Skill Resolution

The framework resolves all skill APIs from config via `Maestro.resolve_skills(config)`:

```
config.control.theory = [molcas, xtb]
config.control.reference = [lammps]
    ↓ maestro.resolve_skills(config)
skills = {
    'theory': {
        'molcas_skill': {'role': 'qm',  'backend': MolcasBackend, 'capabilities': [...]},
        'xtb_skill':    {'role': 'qm2', 'backend': XTBBackend,    'capabilities': [...]},
    },
    'reference': {
        'lammps_skill':  {'role': 'ref1', 'backend': LammpsBackend, 'capabilities': [...]},
    },
}
```

- `resolve_skills()` returns a `skills` dict:
- `skills` contains theory and reference backends with roles and capabilities
- Per-skill capabilities are stored in each skill's dict

Role assignment rules:
- theory methods: ordered by position (1st=qm, 2nd=qm2, 3rd=qm3, ...)
- reference methods: ordered by position (same sequence; `ref{i+1}` past its end)
- one skill may hold two positions, and the second is keyed `skill@role`
  (`pyscf_skill` + `pyscf_skill@qm2`) while its `skill_name` field stays the bare
  name: a composite over one center can put the same program on two layers. The
  dict used to deduplicate by name, which made that unrepresentable; a consumer
  that needs the entry looks it up by (skill_name, role), and `_count_backends`
  counts bindings, not names
- Planning is integrated: resolve_capabilities() runs inside resolve_skills()

The skills dict carries all skill information through the pipeline:
- `build_state(config)` calls `resolve_skills` internally, then `resolve_task_queue()`
  expands composite workflows into task descriptors; each task gets its own plain dict
  task config (`{'skill_config': <skill flat dict>, 'results': {}}`)
  and API path string, registered to PyRAIState via `state.add_task()`
- Backend instances are **not** created in `build_state()`; they are instantiated later
  in `Calculator.initialize()` via `resolve_api()` (see [CALCULATOR_DEVELOPMENT.md](CALCULATOR_DEVELOPMENT.md) Section 3.6)
- All backends receive a task-specific config dict (the extended PyRAIConfig via
  `build_pyrai_config()` is deprecated and no longer used)
- Composite validation uses roles and contribution patterns (e.g. ONIOM2 requires
  qm+qm2 with contributions [+1, +1, -1])
- Tasks are registered in `state.theory/reference` as ordered task queues

### 4.3 Control Parameters

Skills declare their user-configurable parameters in `manifest.yaml`:

```yaml
control_parameters:
  method:
    type: str
    default: dft
  basis:
    type: str
    default: cc-pvdz
```

The framework extracts these via `get_metadata()` and caches them in Maestro.
Maestro provides `list_known_skill_names()` for validation purposes.

Control parameters define which keywords users can set in input files for a given skill.

#### 4.3.1 Declared channels

The framework reads the same manifest for the channels it must handle without
naming the skill. The readers live in `skills/query.py`
(`get_declaration`, `manifest_value`, `role_of`, `skill_stem`) and each key is a
statement the skill makes about itself:

| Key | What the framework does with it |
|---|---|
| `initial_guess` | Fills the named skill field with the input-resolved guess file when the input left it empty (e.g. Molcas `StrOrb`). |
| `receives_environment_charges` | Marks the skill as a consumer of the environment charge rows a multi-scale plan hands over. |
| `slurm_artifacts` | Names the products a self-submitting skill leaves for the runner to collect. |
| `result_contract` | Names the result fields and stubs that declare a finished run (training/prediction/scans). |
| `scan_role_aliases` | Maps a scan's declared role labels onto the flat skill fields. |
| `qmmm.results` | Names the native result blocks and their default units the QM/MM stitcher reads. |

A new channel is a manifest key plus its reader; a framework branch on a skill
name is not an alternative.

### 4.4 Skill Config Building

The `build_state(config)` function (in `data/build.py`) resolves skills, expands the
task queue, creates one plain dict task config per task, constructs API path strings,
and registers tasks:

```
build_state(config):
    env = Environment()
    maestro = env.get_maestro()
    state.pmd = config                       # sole runtime data source
    maestro.resolve_skills(config) → skills
    _build_task_queue(state, config, skills, maestro):
        _validate_driver(state)              # control.driver check for the whole queue
        resolve_task_queue(config, skills) → task descriptors
            (composite expansion: ONIOM/fragment/qmmm; contribution + region indices)
        _apply_standalone_embedding(queue, config)   # background charges/field rows
        _bind_center_plan(state, config)             # canonical plan + fingerprint
        _check_curvature_plan(state)                 # optimize curvature rules, once
        For each task in the queue:
            _merge_skill_binding(...) → skill flat dict (cached per skill;
              defaults filled and profile resolved via the skill query module)
            _assemble_task(...) → task config dict
              {'skill_config': <skill flat dict>, 'results': {}}
              → construct API path string (e.g. 'pyrai3md.skills.pyscf_skill.backend.PySCFBackend')
              → construct static_id (e.g. 'tmp_PYSCF')
              → state.add_task(category, task)  # register to state (no backend instance)
```

Backend instances are created later in `Calculator.initialize()` (see [CALCULATOR_DEVELOPMENT.md](CALCULATOR_DEVELOPMENT.md) Section 3.6).

No module should directly import a specific Skill.

Always use Maestro.

---

# 5. Skill Principles

Every Skill is a self-contained Plugin.

Each Skill own

- manifest.yaml (lightweight discovery metadata)
- query.py (sole exit point for skill information)
- capability definitions (profiles, methods)
- parameter definitions (skill Config)
- default values
- parameter validation
- execution planning
- job execution
- output parsing
- result construction

### 5.1 Skill Schema Design

Each skill maintains its own schema/ directory:

```
<skill>/schema/
├── config.py      # Skill Config (Pydantic models)
├── method.py      # AVAILABLE_METHOD, PROFILE_METHOD_MAP
├── reference.py   # AVAILABLE_PROFILES, AVAILABLE_BASIS, DEFAULT
├── adapter.py     # External task → Skill Config conversion
├── normalize.py   # Skill-specific derived quantities
└── validate.py    # Skill-specific consistency checks
```

The skill Config is independent from the framework Config.
Both share common fields (atoms, coordinates, charges, etc.)
but the skill Config adds skill-specific fields.

### 5.2 Query Mechanism

The query mechanism is the sole communication channel:

```
Framework (skills/query.py)           Skill (<skill>/query.py)
─────────────────────────           ─────────────────────────
discover_all()  ──────────────> get_metadata()
get_registry()  ──────────────> get_registry()
resolve_label() ──────────────> (via Maestro cached data)
```

Rules:
1. Skills never import framework modules
2. All paths in skill query returns are skill-relative
3. Framework prepends `pyrai3md.skills.<skill>` at runtime
4. A skill whose optional dependency is missing stays discoverable: its
   query layer imports nothing heavy, and the failure surfaces when the
   backend is instantiated (`nn_pes_skill` exports its backend class
   lazily for exactly this reason)

Scientific knowledge always belongs to Skills.

---

# 6. Execution Strategy

Execution logic belongs to Skills.

Execution includes

- Local execution
- Slurm execution
- PBS execution
- Remote execution
- Future execution backends

Kernel only provides Runtime Context.

Skills decide

How computation is actually executed.

---

# 7. Input Processing

Input parsing is NOT part of Kernel.

The complete pipeline in `pmd.py`:

```
Step 1: Interface — parse input + load coordinates + validate structure
    → parse_input() (internally: parse + _load_coordinate_file + _validate_structure)

Step 2: Adapter — build PyRAIConfig (includes Environment init)
    → build_config(raw_config)
    → config = Config

Step 3: Normalize — compute derived quantities
    → normalize(config)

Step 4: Validate — check consistency
    → validate(config)

Step 5: Build PyRAIState (task queues with API paths)
    → build_state(config)
    → state = PyRAIState (pmd attached as sole data source, task queues built, no backend instances)
    Each task: API path string + task config dict {'skill_config': flat, 'results': {}} + static_id
        (backend instantiation deferred to Calculator.initialize())

Step 6: Execute
    → execute(state)
    → creates Calculator(state)
    → Calculator.initialize() → resolve_api() → instantiate backends
    → resolve_workflow(state): control.driver != 'main' → SingleShot;
      otherwise jobtype token → JOBTYPE_MAP or None (single-shot)
    → returns Workflow instance (iterative) or calculator.run() (single-shot)
    → runner.run() → Calculator.run() → per task: backend.calculate() → composite aggregation → Result
```

Key design decisions:

1. **Interface completeness**: `parse_input()` handles parsing, coordinate loading, and structure
   validation in one step. The caller receives a fully validated raw dict.

2. **Normalize-before-validate**: normalize (Step 3) and validate (Step 4) operate on the
   Config object, decoupled from maestro.

3. **build_state combines task queue construction**: Step 5 creates PyRAIState and internally
   resolves skills via Maestro, expands composite workflows into task queues
   (`resolve_task_queue`), creates one plain dict task config per task, constructs API path strings,
   and registers to state. Backend instances are **not** created here; they are instantiated
   later in `Calculator.initialize()`. No separate `build_skill_configs()` step is needed.

4. **Task config dict as backend input**: Each backend receives a task-specific plain dict
   `{'skill_config': <skill flat dict>, 'results': {}}` instead of the full PyRAIState.
   The former extended PyRAIConfig via `build_pyrai_config()` is deprecated.

5. **Per-task capabilities**: Per-skill capabilities are resolved by planning and stored
   in each task's dict within `state.theory/reference`; non-primary composite tasks are
   restricted to energy + gradient.

6. **Multi-skill support**: `maestro.resolve_skills()` traverses all theory+reference methods,
   assigning ordered roles. Planning is integrated: resolve_capabilities() runs inside
   resolve_skills(), distributing per-skill capabilities directly.

7. **Task config delivery**: Per-task config is stored in `task['config']` as a plain dict
   (`{'skill_config': <skill flat dict>, 'results': {}}`).
   Configs never fix geometry: region slice indices stay in task metadata and the
   Calculator derives atoms/coordinates/charges from the live `state.pmd.system` at
   execution time. Skill results are written back to `task['config']['results']`; composite
   results are written to `state.pmd.results` via `state.set_result()`.

8. **Planning-Workflow isolation**: The planning (via Maestro) and workflow/ layers have zero imports
   between them. Communication happens through state.theory/reference (task queues + task config dicts)
   and state.jobtype. `execute(state)` is the unified entry point.

9. **Execution driver**: `control.driver` (default `'main'`) fixes the execution model at
   planning time. `resolve_workflow` checks the driver first: any driver != `'main'`
   (`'skill'` or a skill name) returns `SingleShot` regardless of jobtype, and the jobtype
   is handed to the skill as-is (validated by `_validate_driver` in `planning/assembly.py`).

Interface supports

- Legacy PyRAI2MD text input (interface.parser)
- JSON config (interface.loader)
- YAML (via skill test generators)

Adding a new input format must not modify Kernel.

---

# 8. Runtime Modes

PyRAI3MD supports two execution modes.

## Mode 1: External Runtime (CLI / Agent)

```
Step 1: Interface — parse_input() (parse + coord + validate)
    ↓
Step 2: Adapter — build_config(raw_config) → Config
    ↓
Step 3: Normalize — normalize(config)
    ↓
Step 4: Validate — validate(config)
    ↓
Step 5: Build PyRAIState — build_state(config)
    ↓   PyRAIState (pmd attached as sole data source, task queues with API paths)
    Each task: API path string + task config dict {'skill_config': flat, 'results': {}} + static_id
    (backend instantiation deferred to Calculator.initialize())
    ↓
Step 6: execute(state, runtime_idx=None)
    → ExecutionGraphBuilder.build() → freezes the execution graph into
      state.execution_graph (root builds once; workers only verify the
      parent-frozen graph, resolve_execution_graph=False)
    → creates Calculator(state, runtime_idx) with the calculator node
      flags from the graph
    → Calculator.initialize() → resolve_api() → instantiate backends
    → resolve_workflow(state): control.driver != 'main' → SingleShot;
      otherwise jobtype token → JOBTYPE_MAP or None
    → Workflow(calculator=...).run() or calculator.run()
    → Dispatcher.run(units, child_jobtype):
      materialize_unit → Scheduler(multitask)(Launcher(remotetask))
      → Calculator.run() → per task: backend.calculate()
      → composite aggregation → Result
```

Suitable for

- CLI
- Agent
- MCP
- REST

---

## Mode 2: Internal Runtime (Workflow Loop)

```
Workflow
    ↓
Maestro → Capability API → Skill
    ↓
Result
```

In internal runtime, backends are instantiated in `Calculator.initialize()` (Step 6).
The workflow loop calls lightweight `calculate()` on existing instances.

Suitable for

- AIMD (ab initio / nonadiabatic molecular dynamics)
- Active Learning
- Optimization

Internal Runtime never serializes objects.

Python objects remain in memory.

This minimizes overhead.

---

## Mode 3: Dispatched Child Runtimes

Batch workflows (numerical hessian, adaptive sampling, search) and
calculator dispatch materialize independent child units through the
Dispatcher when the frozen execution graph carries multitask/remotetask:

```
Dispatcher.run(units, child_jobtype)
    → materialize_unit: child states stamped worker=True
      (state.runtime.worker) + parent-frozen child execution graph
      (rule 2: worker graphs never carry remotetask=True)
    → Scheduler(multitask)(Launcher(remotetask))
    → launch tool (local in-process / subprocess / slurm)
    → child PyRAI3MD runtime (.rt) → load_state() → execute()
    → worker only verifies the parent-frozen graph (never rebuilds)
    → result.json collected back
```

The dispatch package never imports workflow; all runtime parameters
travel through `state.pmd.environment`. See `WORKFLOW_DEVELOPMENT.md`
Section 11.

**Who decides local versus Slurm, and only once.** Two separate
questions are answered by two separate keywords, and the framework never
states the skill's answer for it.

- `multitask` / `remotetask` / `multipool` are framework dispatch
  keywords: they choose the call shape (pool size, child runtime, thread
  terms) and are read by the planner from the molecule/workflow/skill
  sections into the frozen execution graph. They never reach a skill
  config -- every skill schema declares them as dispatch-layer keys it
  does not validate.
- `execution_mode` is a **skill** keyword: `skills/<skill>/service/runner`
  returns `run_slurm` only for `'slurm'`. It is written in the skill
  section (or injected there from `pmd_env.yaml` `skills.<name>` for a
  test), and `schema.config.RETIRE_ENVIRONMENT_KEYS` refuses it under the
  framework `&environment` section, where it used to be read by nothing.

When the framework's own dispatch says remote,
`planning.assembly.build_task_queue` writes `execution_mode=local` into that
task's skill config (warning when the input had declared `'slurm'`),
because a self-submitting skill inside an already-remote runtime would
queue the same job twice. The derived value travels with the serialized
state, so the child runtime needs no second decision (#300).

---

# 9. Domain Model

The software must support

- Molecular systems
- Periodic systems
- Surfaces
- Interfaces

Only one object is allowed

System

Boundary condition determines

- Cluster
- Periodic
- Surface
- Interface

Never create

Molecule

PeriodicSystem

CrystalSystem

as separate root objects.

---

# 10. Plugin Principles

Skills are Plugins.

Plugins may be installed independently.

Examples

pyrai3md-orca

pyrai3md-cp2k

pyrai3md-pyscf

pyrai3md-nequip

PyRAI3MD Core must work without modification after installing new Plugins.

---

# 11. Dependency Rules

Allowed

Kernel

↓

Maestro

↓

Skill

↓

Utilities

Workflow

↓

Maestro

↓

Skill

Forbidden

Kernel

→ ORCA

Kernel

→ CP2K

Workflow

→ ORCA

Workflow

→ CP2K

Schema

→ Backend

Utilities

→ Backend

---

# 12. Design Constraints

The following rules must never be violated.

Rule 1

Core owns protocols.

Skills own scientific knowledge.

Rule 2

Maestro is the only skill coordination and capability discovery mechanism.

Rule 3

Workflow depends only on capabilities.

Never on software.

Rule 4

Every backend must be replaceable without modifying Workflow.

Rule 5

Schema describes communication.

Skill describes computation.

Rule 6

Execution belongs to Skills.

Rule 7

Input parsing belongs to Interface.

Rule 8

Internal computation uses Python objects.

External communication uses Schemas.

Rule 9

Utilities contain no scientific logic.

Rule 10

Every new backend should be installable as an independent Plugin.

Rule 11

Planning terminates at PyRAIState.

PyRAIState is the only communication protocol between Framework and Skills.

The task config dict `{'skill_config': <skill flat dict>, 'results': {}}` is the
input for each Skill backend initialization (the extended PyRAIConfig via
`build_pyrai_config()` is deprecated).

Rule 12

Skill initialization owns Skill-specific preparation.

Workflow operates only on instantiated Backend objects.

Rule 13

planning (via Maestro) and workflow/ are import-isolated.

Communication happens through state.theory/reference (task queues + task config dicts) and state.jobtype.

---

# 13. Long-Term Goal

PyRAI3MD should evolve into an extensible scientific computing platform.

The architecture should remain stable while

- adding new quantum chemistry software
- adding new ML models
- adding new workflows
- adding new execution backends
- integrating AI Agents

without requiring modification of the Core architecture.
