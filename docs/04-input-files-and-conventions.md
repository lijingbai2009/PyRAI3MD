---
title: "4. Input files and conventions"
nav_order: 4
---

This chapter describes the files a user writes and the conventions every carrier shares. PyRAI3MD reads three user carriers - classic &SECTION text, sectioned YAML and agent JSON - and writes two serialized payloads of its own, .rt runtime files and .rtt term files. All five lead to one validated configuration or to the state built from it; the carrier decides how sections and values are spelled, never what a keyword means.

## 4.1. Input carriers

Classic text is the historical PyRAI2MD style, YAML is the form the shipped examples use, and agent JSON is the form the agent and the programmatic interface consume. The two runtime suffixes are written by the program and are not input formats.

| Carrier | Entry command | First line | Used by |
| :--- | :--- | :--- | :--- |
| classic text | pyrai3md water.input | &CONTROL | command line; generated example carriers |
| sectioned YAML | pyrai3md water.yaml | control: | command line; shipped examples |
| agent JSON | pyrai3md_agent water.json | { | agent command line and PyRAI3Agent |
| .rt runtime | pyrai3md water.rt | pickled PyRAIState | restart and remote dispatch |
| .rtt term payload | pyrai3md water.rtt | pickled term payload | one remote term worker |

The three user carriers enter one pipeline: parse, build the configuration, normalize, validate, build the runtime state and execute. Classic text and YAML enter through interface.parser, agent JSON through interface.loader; from the configuration model onward the route is identical. A keyword means the same thing and a mistake is refused at the same stage whichever file carried it. The same pipeline therefore serves the command line, the agent and child runtimes, which differ only in where the state comes from.

A .rt file is a serialized runtime state written for restart or remote dispatch; it re-enters execution at the restored state without parsing an input file. A .rtt file is a single-term payload for a remote worker; it executes one backend term and writes term_outcome.json. Neither suffix is a user input format.

## 4.2. Classic text input

A line whose first non-blank character is & opens a section and names it; the name is case-insensitive. A line &end closes the current section, and any key after it is ignored until the next section header. Inside a section each line is a keyword followed by one or more whitespace-separated values: the first token is the key, the rest of the line is its value. Keys are case-insensitive, blank lines are skipped, and a # starts a comment that runs to the end of the line.

Several values on one line are kept as one space-joined string, which the section reader splits according to the keyword type. A keyword repeated inside one section is different: all its values are kept, in order, as a list. Nothing is silently overwritten; a section reader that cannot use several values refuses the input and names the key. Booleans are true, false, 1 or 0. The aliases qm, abinit, temp and lattice are read as theory, reference, temperature and lattice_vector, and the log shows the modern names.

The input below is one complete water single point at the Hartree-Fock/STO-3G level. The comments mark the three parts every run needs: control, the molecular specification and one skill section. &MOLECULE is the historical spelling of the system section; &PYSCF is the skill section, and &PYSCF_SKILL is accepted as the same section. Unlike YAML and JSON, classic text has no inline structure, so the run reads water.xyz beside the input, or the path named by &MOLECULE coord.

```text
# water single point, classic text carrier
&CONTROL
title        water_energy     # names <title>.log and the scratch tree
jobtype      energy          # one evaluation; advanced jobs stand alone
theory       pyscf           # the method queue, high to low
driver       main            # the framework owns the loop
 
&MOLECULE
charges      0               # total charge
multiplicity 1               # singlet
ci           1               # number of states per multiplicity
spin         0               # total spin per state
 
&PYSCF
method       hf              # the skill reads this key in its own section
basis        sto3g
&end
```

Three separators carry structure inside a value. Whitespace separates the keyword from its values and separates several values from each other. A comma separates the two state indices of one pair. A semicolon separates centres in the multi-centre keywords. The table collects these conventions with the two line-level tokens.

| Separator | Meaning | Example |
| :--- | :--- | :--- |
| space | keyword from value; several values in one string | theory pyscf molcas |
| , (comma) | state indices inside one pair | coupling 1 2, 2 3 |
| ; (semicolon) | centres, in the multi-centre keywords | ci 2 2; 2 2 |
| # | comment to the end of the line | title water # run name |
| blank line | skipped; it has no meaning |  |
| &end | closes the current section | &end |

> **Note.** A repeated key is a list, not an overwrite. The section reader refuses a second value it cannot use instead of discarding one.

## 4.3. YAML input

A YAML input is one top-level mapping. Each framework section is a mapping named by its lower-case section name: control, molecule, md, environment, and so on. The parser lower-cases top-level keys, and native YAML types are preserved, so a scalar stays a scalar and a list stays a list. Comments use # and run to the end of the line.

The control mapping carries title, jobtype and theory; theory is a list because a method queue is a list even when it holds one entry. The structure arrays are top-level keys, atoms and coordinates, beside control and not inside molecule. The coordinate resolver returns early when a top-level atoms key exists, so an atoms list nested under molecule is never seen by that check; the file then needs a coord path or a same-basename .xyz, and otherwise reaches the no-coordinates error. [Section 4.5](#45-how-structure-and-guesses-are-resolved) states the full order.

A skill section is named after the skill: both pyscf: and pyscf_skill: resolve to the PySCF skill, and the short spelling is the one the shipped examples use. Lists keep their native shape, which matters for ci, spin and target_states. A pair field written as a string, for example coupling: "1 2", remains external and 1-based; a typed nested list is already the normalized 0-based entry.

The water single point in YAML keeps the same three parts as the classic file, with the structure written inline.

```yaml
# water single point, YAML carrier
control:
  title: water_energy
  jobtype: energy
  theory: [pyscf]
 
atoms: [O, H, H]
coordinates:
  - [0.000, 0.000, 0.000]
  - [0.000, 0.757, 0.587]
  - [0.000, -0.757, 0.587]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: [1]
  spin: [0]
 
pyscf:
  method: hf
  basis: sto3g
```

A dynamics input adds an md mapping. The example runs ten steps of FSSH on two singlet states under an NVT thermostat. ci and spin are lists; coupling is a string, so its pair is read as external 1-based indices; root is 1-based in every external spelling. The skill section selects TDA with PBE and asks for two states.

```yaml
# two-state FSSH dynamics, YAML carrier
control:
  title: water_namd
  jobtype: md
  theory: [pyscf]
 
atoms: [O, H, H]
coordinates:
  - [0.000, 0.000, 0.000]
  - [0.000, 0.757, 0.587]
  - [0.000, -0.757, 0.587]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: "2"
  spin: "0"
  coupling: "1 2"
 
md:
  ensemble: nvt
  thermostat: nose-hoover
  sfhp: fssh
  root: 1
  step: 10
  size: 20.67
 
pyscf:
  method: tda
  functional: pbe
  basis: sto-3g
  nstate: 2
  target_states: [0, 1]
```

## 4.4. Agent JSON

Agent JSON is a sectioned JSON document. Its top-level keys are the section names of the other carriers plus one section per skill, named by the skill package, for example pyscf_skill. The loader lower-cases top-level keys and does no type conversion; build, normalize, validate and state construction then follow the same pipeline. Because JSON has no comments, annotation has to live in the surrounding text.

The command line is pyrai3md_agent config.json. In Python, PyRAI3Agent.from_file(path) returns an agent that runs the same pipeline; initialize() is optional because run() initializes on first use. The programmatic run() writes no log, because a library caller owns its output. The command line calls run_cli(), which goes through the framework RunLogger and writes the same &lt;title&gt;.log and closing status line as pyrai3md.

```json
{
  "control": {
    "title": "water_energy",
    "jobtype": "energy",
    "theory": "pyscf"
  },
  "molecule": {
    "atoms": ["O", "H", "H"],
    "coordinates": [[0.000, 0.000, 0.000],
                    [0.000, 0.757, 0.587],
                    [0.000, -0.757, 0.587]],
    "charges": 0,
    "multiplicity": 1,
    "ci": [1],
    "spin": [0]
  },
  "pyscf_skill": {
    "method": "hf",
    "basis": "sto3g",
    "execution_mode": "local"
  }
}
```

> **Note.** from_file(...).run() is the library route; pyrai3md_agent config.json calls run_cli(). Only run_cli() writes the log.

## 4.5. How structure and guesses are resolved

Coordinates and velocities are resolved in the interface layer before the configuration is built. The routes are tried in a fixed order and the first that applies wins. The table states each route as the user writes it and as the program reads it.

| Route | What you write | What is read |
| :--- | :--- | :--- |
| 1 | top-level atoms and coordinates in YAML, JSON or agent input | used as they stand; no file is opened |
| 2 | molecule coord &lt;file.xyz&gt; | the first value that names an existing file |
| 3 | a same-basename .xyz beside the input | water.xyz for water.input, water.yaml or water.json |
| 4 | molecule velo &lt;file&gt;; otherwise &lt;coord base&gt;.velo or &lt;coord base&gt;.velocity.xyz | that velocity file; with none, zero velocities |
| 5 | the skill-declared initial guess {title}&lt;suffix&gt; or &lt;input stem&gt;&lt;suffix&gt; | the first candidate present, stored for that skill field |
| 6 | nothing that supplies a structure | a hard error naming the three coordinate routes |

Route 4 is separate from the coordinate choice: the velocity path is read from the molecule section if it names an existing file, otherwise the coordinate base supplies .velo or .velocity.xyz, and otherwise the run starts from rest and prints an informational line with the atom count.

Route 5 is driven by the skill manifest. A skill that starts from a converged file declares an initial_guess suffix and field; a skill that declares nothing is never consulted. The candidates are checked in the input file directory, title first and input stem second, and the resolved path is stored with its skill and field. The task queue fills that field only when the input left it empty, so an explicit path in the skill section always wins.

When no route supplies atoms or coordinates, the error is raised before any engine starts and names three ways forward: a molecule coord path, a same-basename .xyz file, or atoms and coordinates in the input itself. Route 1 tests a top-level atoms key; an atoms list nested under molecule does not satisfy it, so the file falls through to routes 2 and 3 and, if neither finds a file, reaches route 6.

## 4.6. Conventions

Six conventions apply to every carrier. They belong to the framework, not to an engine or a file format, so a number keeps its meaning after a switch from classic text to YAML.

Coordinates are in Angstrom, and so are cell lengths, while cell angles are in degrees; velocities travel in Bohr per atomic unit of time. Energies are in Hartree. Gradients are in Hartree/Bohr and are the derivative dU/dR, not a force: an engine that prints a force block, in its own energy-per-length unit and with the opposite sign, is converted by the parser before the values reach the framework.

The MD keyword size is a time step in atomic units of time, not in femtoseconds. One atomic unit is 0.02418884254 fs, so the common value 20.67 au is about 0.5 fs. The hopping thresholds gap and gapsoc are in eV and are converted to Hartree internally. Other keywords carry the unit named in their own table in [Section 5](05-keyword-sections.html).

Atom indices are 1-based in every carrier, including the region keywords, the coupling pairs and the constraint block. State indices are mixed: the initial state root is 1-based externally, while current_state, opt_state_i and opt_state_j are 0-based. A dynamics run started with root 1 therefore begins on the state that current_state 0 names. Section B tabulates every index-bearing keyword and its base.

At the framework boundary the units are Hartree, Hartree/Bohr and Hartree/Bohr^3. A skill receives and returns values in these units and converts its native units before returning: electronvolts, kcal/mol, Angstrom, femtoseconds and engine-specific force blocks are converted inside the skill or by the parser that reads its output. The framework never inspects an engine unit system.

## 4.7. A first calculation in three carriers

The same water single point can be written in each carrier. Save the three input files as water.input, water.yaml and water.json, with water.xyz beside water.input: O at the origin and two H at (0, 0.757, 0.587) and (0, -0.757, 0.587) Angstrom. Only the carrier changes; the title, jobtype, structure and PySCF keywords are the same.

```text
&CONTROL
title        water_energy
jobtype      energy
theory       pyscf
 
&MOLECULE
charges      0
multiplicity 1
ci           1
spin         0
 
&PYSCF
method       hf
basis        sto3g
&end
```

```yaml
control:
  title: water_energy
  jobtype: energy
  theory: [pyscf]
 
atoms: [O, H, H]
coordinates:
  - [0.000, 0.000, 0.000]
  - [0.000, 0.757, 0.587]
  - [0.000, -0.757, 0.587]
 
molecule:
  charges: 0
  multiplicity: 1
  ci: [1]
  spin: [0]
 
pyscf:
  method: hf
  basis: sto3g
```

```json
{
  "control": {"title": "water_energy", "jobtype": "energy",
              "theory": "pyscf"},
  "molecule": {
    "atoms": ["O", "H", "H"],
    "coordinates": [[0.000, 0.000, 0.000],
                    [0.000, 0.757, 0.587],
                    [0.000, -0.757, 0.587]],
    "charges": 0, "multiplicity": 1, "ci": [1], "spin": [0]
  },
  "pyscf_skill": {"method": "hf", "basis": "sto3g"}
}
```

Run the classic and YAML files with pyrai3md water.input and pyrai3md water.yaml; python -m pyrai3md water.yaml is equivalent. Run the JSON file with pyrai3md_agent water.json. Adding --info parses the file, builds and validates the configuration and prints a compact resolved summary without starting an engine:

```yaml
title:     water_energy
jobtype:   energy
theory:    pyscf
reference: -
driver:    main
natom:     3
tasks:     theory 1, reference 0
```

Without --info the run writes &lt;title&gt;.log in the launch directory, or under the scratch path when the environment section sets one. The log opens with the start time and logo, echoes the control, system and environment sections, then prints the result block. For this run the block reports the three coordinates and one electronic energy; the value below is the Hartree-Fock/STO-3G result at the geometry above, and the tail line carries the wall time and the final status.

```text
  &electronic energy in Hartree
---------------------------------------------------------------------------
  State   1        -74.9631499176452110
---------------------------------------------------------------------------
 
PyRAI3MD End: Nov 12 2026  17:08:31 Total: 00:00:00:02 Status: completed
```

The same title, jobtype and energy appear for all three carriers. A validation error stops the run before the log is opened, so a missing structure or an unknown keyword produces a message, not a half-written log. [Section 6](06-workflows.html) describes the workflows that replace jobtype energy, and [Section 12.2](12-output-and-analysis.html#122-result-and-trajectory-files) indexes every file a run can leave behind.
