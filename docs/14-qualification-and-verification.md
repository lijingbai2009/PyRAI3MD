---
title: "14. Qualification and verification"
nav_order: 14
---

This chapter describes the test suite that ships with the package and the two remote drivers. The suite runs through the same entry points a user uses, so a passing round is evidence that the installation and the examples agree.

## 14.1. The qualification round

pyrai3md test plans and runs a round from ~/.pyrai3md/pmd_env.yaml over the package template. It covers four categories: the framework unit tests and integration tests, the shipped-example carriers, the skill certification cases and the benchmark calibrations.

The framework category runs the modules under qualification/verification in one process and adds the cross-mode files under qualification/integration when those are opened. The example category renders every YAML under pyrai3md/examples into two user carriers, the YAML itself and the classic section input, so one example is two tasks. The skill category takes each skill tests directory: the pytest scripts first, then every case YAML expanded into three carriers, the classic input run by pyrai3md, the sectioned JSON run by pyrai3md_agent and the flat skill JSON run by pyrai3md_skill. The benchmark category runs the FSSH, GSH, LZSH, kTDC and performance calibrations against their analytic references.

The carriers are the point. An example or a skill case is written to disk and run through the command a user would type, so the parser, normalization, validation, planning, dispatch and logging of a real run are all exercised. A defect that appears only in the classic writer, or only in the agent schema, is a round failure.

Planning queries the skill registry, generates the inputs and opens the work directory tmp_YYYYMMDD_HHMMSS in the current directory. The round writes one record per task to results.json and the terminal state to status.json. A local round leaves one row fragment per task under jobs/; a grouped Slurm round leaves the job specs, the batch scripts and the fragments there and collects them in plan order.

Three levels of switches decide what runs, all read from the same file. The master _runall sits above everything: true opens every example, skill case and calibration, false closes them and leaves the tests that need no engine, and condition passes the decision down. The middle level is one switch per bucket or skill: test_integration, the six example family keys (optimize_test, path_test, md_lammps_test, constraint_md_test, multiscale_test and adaptive_test), skills.&lt;name&gt;.test, skills.&lt;name&gt;.runtime_tests and the benchmark entries. The bottom level is one key per case: examples.&lt;name&gt;, skills.&lt;name&gt;.tests.&lt;case&gt; and the enabled flag of a benchmark entry. Each takes true, false or condition, and a missing key reads as condition. The master is consulted above the families, so false there closes a family whose own key reads true; an optimize round is _runall: condition with the optimize family on.

```yaml
_runall: condition
test_integration: false
examples:
  optimize_test: true
  energy: false
skills:
  molcas_skill:
    test: condition
    tests:
      casscf_gradient: true
benchmark:
  fssh:
    - method: tully1
      enabled: false
```

A closed case stays in the plan as a skipped task whose reason names the key that closed it, and the report folds the closed rows of each bucket into one row carrying the count and the reasons.

## 14.2. Remote rounds

The deploy tool runs the same round on a configured host. It reads the deploy section, asks for anything incomplete, resolves the SSH route (direct, then the jump host) and the conda environment, syncs the checkout, installs it with pip, and starts pyrai3md test in the background so the run survives the starting session.

The sync excludes version-control files, byte-code caches, egg metadata, editor folders, earlier rounds, run logs and build products, and it deletes no excluded file on the host, so earlier artifacts survive. The install removes the build tree first, so a stale build cannot be reused.

The tool polls status.json, reports progress against the task count the round publishes, then fetches results.json and downloads the log as test_&lt;stamp&gt;.log. The generated carriers and calculation directories stay in the round directory on the host. The force form cancels the jobs a grouped Slurm round recorded and stops the orchestrator.

The deploy section holds the host, user, port, install path and optionally a jump host; environment.scheduler chooses a local or a grouped Slurm round. [Section 3.6](03-installation.html#36-installing-on-a-cluster) covers the client side.

## 14.3. Post-round verification

The verifier audits a finished round from its files alone.

```bash
python tools/verify.py tmp_20261112_090000
python tools/verify.py --selftest
```

It never starts an engine, a skill entry point or a calculation, and writes nothing into the round; where it needs a skill reader it calls that parser on a file. Its sections are lettered A to K.

| Section | What it cross-checks |
| :--- | :--- |
| A | Recomputes each recorded example metric from the frame records and flags a record older than the round. |
| B | Compares each logged step's electronic and kinetic energy with the same step's record row. |
| C | Requires the recorded potential energy to be a number a surviving engine file printed; C2 does the same for an optimize leg's status table, and C3 requires a composite row's term columns to add to the energy beside them. |
| D | Re-derives a variable-cell verdict from the status record, the cell metrics and the printed cell sentence, and re-reads each periodic path leg's status table by column name. |
| E | Compares one example's two carriers on the recorded metrics and on each record's term decomposition. |
| F | Re-derives the path family's claims: a held coordinate, frozen atoms, and the energy a composite leg and its single-layer counterpart record for one geometry. |
| G | Reads one OpenMolcas case's three carrier logs with the skill's own parser and requires the four arrays to agree. |
| H | Re-derives the periodic field probe's cross energy with a converged Ewald sum, which names a deck that lost its pinned k-space line. |
| I | Checks the QM/MM force and gradient files atom by atom against the assembled gradient on the QM rows and elsewhere. |
| J | Audits an adaptive run's two-round product tree: selections, label counts, a servable model folder, the alignment record, the structural counts of both carriers and the sidecar rule. |
| K | Compares every skill case's three carriers file-only, the engine deck and the final energy each printed, and runs no skill. |

The selftest plants a clean round, corrupts one number per section and requires every corruption to be named. The report also states its coverage: which skills have readers, which rows it refused to compare, and which channels have no artifact to check.
