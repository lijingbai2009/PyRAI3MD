---
title: "15. Extending PyRAI3MD"
nav_order: 15
---

This chapter describes the contract a new engine must satisfy. It is meant for readers who want to add a skill, not for users writing an input.

## 15.1. The skill contract

The framework and a skill meet at three declared places: the skill manifest, the query module and the skill configuration schema. The import tree is one-way. A skill may import pyrai3md.utilities and nothing else from the framework; it may not import the schema, planning, workflow or kernel layers. The framework never imports an engine: the backend import path is stored as a string and resolved at calculation start.

The query module is the skill's only exit point. The framework calls five public names from it: get_profiles and list_profiles describe the calculation profiles, resolve_profile maps a method, jobtype and driver onto one, get_registry names the backend module and class, and get_defaults returns the schema defaults used to build the skill configuration. Paths it returns are skill-relative; the framework prepends its package prefix at run time.

A calculation is one call to backend.calculate with the atoms, the coordinates, the target states and, for a periodic system, the cell. It returns a plain dictionary keyed by the property names the framework reads. The dictionary is not validated against a model: the contract is by name, and a key spelled differently is not read.

Four channels carry state rather than properties. fin_coord carries the final structure, one row per atom in Angstrom, after the skill ran its own loop; a length that disagrees with the atom count raises. fin_velocity carries final velocities in Bohr per atomic unit of time. fin_cell carries the box the returned coordinates live in, as six cell parameters or nine lattice components, and a value returned for a non-periodic system raises. opt_converge carries the optimization verdict when the driver is the skill. The framework consumes each channel and drops the key; a different spelling of the last one is not read.

Units at the boundary are Hartree, Hartree per Bohr and Hartree per cubic Bohr, with coordinates in Angstrom, velocities in Bohr per atomic unit of time and spin-orbit couplings in cm-1. A gradient is a derivative, never a force. A skill converts its native units and declares them in its manifest; the framework applies that declaration, and an unknown unit name raises.

## 15.2. Writing a skill

A skill is a directory under pyrai3md/skills/. The manifest declares what the framework may ask for: the profiles and the capability and method of each, the capability list, the backend module and class, the units of the returned arrays, the roles the skill can fill, whether it supports periodic systems, and its tunables, which are the control parameters a user may write, the resource keywords that size a run, the hyperparameters a grid scan may vary and the result channels a training or prediction call must return.

The schema package owns everything the framework communicates with: the configuration model, the supported methods and capabilities, the constants and defaults, the adapter that turns a framework task into a skill configuration, the normalization step and the validation rules. The service package owns the computation: the workflow compiler that turns a profile into a graph, the builders that write the engine input files, the runner, the output parsers and the templates.

```text
<skill>_skill/
  manifest.yaml
  query.py
  backend.py
  agent.py
  skill_guide.md
  schema/
    config.py
    method.py
    reference.py
    adapter.py
    normalize.py
    validate.py
  service/
    workflow/
    builder/
    runner/
    parser/
    templates/
  tests/
    <case>.yaml
    test_<module>.py
```

One rule has no diagnostic and no second chance: every configuration field must be registered in get_defaults. The framework generates the skill configuration from that table, so a field that is not registered is silently dropped between the input file and the backend, and the run proceeds without it. Registering a field also puts it in the vocabulary the query module reports.

The tests directory carries the offline pytest scripts and the case YAMLs the certification planner expands into three carriers ([Section 14.1](14-qualification-and-verification.html#141-the-qualification-round)). A skill is complete when pyrai3md_skill runs one of its JSON configs and one case appears in a round.

## 15.3. Where to read more

| Document | Covers |
| :--- | :--- |
| pyrai3md/skill_guide.md | The framework's own usage guide: entry points, input carriers, command reference, output layout and development rules. |

The keyword chapters are generated from the same configuration models, so a default quoted in [Section 5.1](05-01-control.html) is the value the program loads.
