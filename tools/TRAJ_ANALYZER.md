# Trajectory analyzer

`tools/traj_analyzer.py` reads a set of trajectory directories, checks
their completeness and energy conservation, extracts frames, and
computes populations and geometrical parameters. It is the legacy
Gen-FSSH / PyRAI2MD analyzer with a PyRAI3MD reader added; the NewtonX,
SHARC and fromage readers are unchanged.

```bash
python tools/traj_analyzer.py analysis.in
python tools/traj_analyzer.py            # print the keyword summary
```

`title` is required. Every output file is named after it. There are no
command-line options; every setting is a keyword in the input file.

## 1. Trajectory directories and readers

The analyzer can read trajectories anywhere: a directory named
`<title>_<i>` or `<title>-<i>` beside the input file, or any list of
absolute paths. The calculation folder and the project output file
should share the title, because every output of the run is named after
`title`.

`prog` selects the reader:

| `prog` | reads |
|---|---|
| `pyrai3md` (default) | PyRAI3MD `<project>.md.xyz`, `.md.energies`, `.md.pop`, `.sh.*` |
| `molcas` | legacy PyRAI2MD trajectories through the Molcas driver (log `Gnuplot` blocks, `.md.energies`, `.md.xyz`) |
| `newtonx` | NewtonX `RESULTS/` (`nx.log`, `dyn.out`, `sh.out`, `en.dat`) |
| `sharc` | SHARC output |
| `fromage` | fromage output |

With an integer index the analyzer uses `<title>_<i>` when that
directory exists and `<title>-<i>` otherwise. Indices accept single
values, ranges and mixed forms (`1 2-4 5`); duplicates are removed and
the list is sorted. A file of paths is the robust form:

```text
read_index /path/to/trajectories.txt
```

The file is split on whitespace, so use one absolute path per line and
no spaces inside a path.

For `prog pyrai3md` the reader discovers the project stem from the
`<project>.md.xyz` file inside each directory, so a directory named
`<project>_1` is read with project `<project>`. A directory with no
`.md.xyz`, or with more than one, stops the reader with the directory
name in the message.

## 2. Keywords

The input is a two-column text file, one `keyword value` pair per line,
in free order. Keywords are case-insensitive; extra columns after the
first value are ignored unless the keyword is list-valued.

| keyword | meaning | default |
|---|---|---|
| `title` | run title; required; names every output | none |
| `cpus` | worker processes; capped at the number of trajectories | 1 |
| `read_index` | index list or a file of trajectory paths | `1` |
| `prog` | reader, above | `pyrai3md` |
| `save_traj` | 1 = write the extracted `<title>.json` | 1 |
| `minstep` | minimum rows for a trajectory to count as complete | 1 |
| `maxstep` | read at most this many rows/frames; 0 = all | 0 |
| `maxdrift` | energy-conservation tolerance in hartree | 0.5 |
| `mode` | analysis mode, section 3 | 0 |
| `select` | trajectory whitelist (index list or file) | none |
| `read_init` | initial-condition file for mode 8 | none |
| `classify` | snapshots to classify: `all`, `init`, `final`, `hop` | `all` |
| `classify_state` | target state (0-based) for extraction and classification | 0 |
| `align` | 1 = Kabsch-align the snapshot files | 0 |
| `align_core` | 1-based atom indices defining the alignment core | all atoms |
| `output_atom` | 1-based atom indices kept in frames and snapshots | all atoms |
| `ref_geom` | reference xyz for RMSD; the first structure is used | none |
| `param` | geometrical parameters, or a file of them | none |
| `threshold` | one threshold per parameter | none |
| `prune_type` | stop a trajectory on a monitored distance: `dist` or `frag` | none |
| `prune_index` | distance pairs or fragment index groups, comma separated | none |
| `prune_thrhd` | one threshold per distance, or one for a fragment | none |
| `save_data` | write detailed data in mode 4: 0, 1 or 2 | 0 |

Use `threshold`. The help banner prints `thrhd`, but the parser reads
only `threshold`; a `thrhd` line is silently ignored and every
threshold stays zero. `classify` values other than the four listed fall
back to `all`.

When `output_atom` is used, frames are cut to those atoms before any
parameter is computed, so `align_core`, `param` and `prune_index`
indices then refer to positions in the reduced atom list, not to the
original atom numbers.

## 3. Modes

| mode | alias | action |
|---|---|---|
| 0 | | skip analysis (default) |
| 1 | `diag`, `diagnosis` | count rows per trajectory, write `complete` |
| 2 | `cons`, `conservation` | energy drift, write `conserved` and `energy_drift.txt` |
| 3 | `read`, `extract` | extract frames to JSON and snapshot xyz files |
| 4 | `pop`, `population` | average populations and hop energy gaps |
| 5 | `cat`, `classify` | classify snapshots by geometrical parameters |
| 6 | `plot`, `compute` | compute the parameters used for plots |
| 7 | `out`, `fetch` | fetch selected trajectories as xyz plus data |
| 8 | `pyrai3md` | snapshot reader for initial, final and hopping frames |

Mode 8 was called `pmd` in the legacy script. That spelling is no
longer an alias; use `8` or `pyrai3md`.

## 4. Step-by-step workflow

### 4.1 Check completeness (`mode 1`)

```text
title      file_name
mode       diag
read_index list.txt
minstep    50
```

Mode 1 prints a table of `.log`, `.md.energy` and `.md.xyz` row counts.
A trajectory is complete when the three counts are equal and at least
`minstep`. For `prog pyrai3md` the first column is the row count of
`<project>.md.pop`, or of `.md.energies` when a run wrote no population
record because it never hopped. The paths of complete trajectories are
written to `complete`, one per line.

### 4.2 Check energy conservation (`mode 2`)

```text
title      file_name
mode       cons
read_index complete
maxdrift   0.05
maxstep    200
```

Mode 2 computes per-step drift as `Etot(t) - Etot(0)`, where `Etot` is
the kinetic energy plus the potential energy of the active state. It
reports the mean absolute drift, the maximum absolute drift and the mean
drift. A trajectory is kept when the maximum absolute drift is below
`maxdrift`. Kept paths go to `conserved`; every per-step drift goes to
`energy_drift.txt`, one trajectory per line. After this step choose a
`maxstep` that covers the conserved part of the trajectories.

### 4.3 Extract trajectory data (`mode 3`)

```text
title      file_name
mode       read
read_index conserved
save_traj  1
maxstep    200
```

Mode 3 writes:

- `<title>.json` when `save_traj 1`, the arrays used by modes 4 to 7.
  With thousands of trajectories this file can be large; saving and
  loading it can be slower than reading the log or md files again, so
  `save_traj 0` is a legitimate choice. Later modes re-read the folders
  when the JSON is absent.
- `Geom-<title>.json`, the initial, final and hopping snapshots.
- `Ini.S<n>.xyz`, `Fin.S<n>.xyz` and `Hop.S<n>.xyz`, grouped by the
  final state `n` of each trajectory. `Ini` and `Fin` hold one frame per
  trajectory; `Hop` holds every hop frame of those trajectories. The
  JSON keeps the last hop per trajectory.

The uppercase `G` matters: `Geom-<title>.json` comes from mode 3, while
lowercase `geom-<title>.json` comes from mode 8. On a case-insensitive
filesystem they are the same name; mode 8 checks the schema and re-reads
the folders when the file is not a snapshot record.

Dissociation can produce artificial back hops at long distance. Prune
the trajectory before that point:

```text
prune_type  dist
prune_index 1 2, 3 4
prune_thrhd 3.0 4.0
```

This stops reading a trajectory when the 1-2 distance exceeds 3.0
Angstrom or the 3-4 distance exceeds 4.0 Angstrom. Section 7 describes
the full syntax.

### 4.4 Compute state populations (`mode 4`)

```text
title      file_name
mode       pop
read_index conserved
save_traj  1
maxstep    200
save_data  0
```

Mode 4 loads `<title>.json` when it exists, so `read_index`, `save_traj`
and `maxstep` have no effect in that case. Otherwise it reads the
folders first.

`select` restricts the average to a whitelist, an index list or a file.
Without it every trajectory is used. The analysis then removes failed
trajectories automatically:

- population contains NaN: skipped;
- population outside -0.01 to 1.01: skipped;
- early stopped while still in an excited state: skipped;
- early stopped in the ground state: repaired by repeating its last step
  to the longest trajectory length.

The run prints the four counts and the surviving index range, and
appends the same summary to `<title>.traj.log`.

Outputs:

- `average-<title>.dat`, the per-step `Ekin`, per-state energies,
  per-state total energies and per-state populations. The header row
  starts with the time step; data rows start with the 0-based step
  index. In this revision the `Epot <k>` columns are filled from the
  total-energy array too, so they repeat `Etot <k>`; use the population,
  `Ekin` and `Etot` columns.
- `hop-energy-<title>.dat`, one row per hop: trajectory index, state
  before the hop, state after the hop, and the absolute gap between
  those two state energies at the hop frame in eV.
- `state-pop-<title>.json` with `save_data 1` or larger: populations of
  every trajectory.
- `energy-profile-<title>.json` with `save_data 2`: labels, kinetic,
  potential and total energy arrays of every trajectory.

### 4.5 Classify trajectories (`mode 5`)

```text
title          file_name
mode           cat
read_index     conserved
save_traj      1
maxstep        200
classify       final
classify_state 1
align          1
align_core     1-5
output_atom    1-10
param          B 1 2 B 3 4
threshold      2.5 2.5
```

Mode 5 uses `Geom-<title>.json` when present and re-reads the folders
otherwise. `classify` chooses the snapshots (`init`, `final`, `hop` or
`all`), and `classify_state` chooses which final-state group is
classified. If that group has no snapshot the run stops with a message
asking for another `classify` or `classify_state`.

`param` is optional. Omitting it still aligns snapshots, which is the
way to produce aligned structures without computing parameters.
`align 1` writes `aligned-Ini.S<n>.xyz`, `aligned-Fin.S<n>.xyz` and
`aligned-Hop.S<n>.xyz`, using `align_core` as the alignment core.
`output_atom` limits the atoms written.

`threshold` supplies one value per parameter. The code labels each
parameter 1 when its value is greater than its threshold and 0
otherwise. When the counts of `param` and `threshold` differ, every
threshold is set to 0.

The parameter files are:

- `param-<title>.S<n>.ini` for initial snapshots;
- `param-<title>.S<n>.fin` for final snapshots;
- `param-<title>.S<n>.hop` for hop snapshots.

Each row starts with the trajectory index and the parameter values,
followed by the snapshot step and the potential energies of all states.
The 0/1 labels are used for the printed class summary; they are not
written into the parameter file. The run also prints a label table with
the number and percentage of trajectories in each class.

### 4.6 Compute plot data (`mode 6`)

```text
title      file_name
mode       plot
read_index conserved
save_traj  1
maxstep    200
param      B 1 2 B 3 4
select     file.txt
```

Mode 6 computes every parameter at every frame of the selected
trajectories, and writes `plot-<title>.json`. `param` is required here;
without a parameter there is nothing to compute. The JSON holds the
title, the parameter values per selected trajectory and frame, the total
energy of the active state per frame, the hop list and the frame labels.
`select` is optional; without it every trajectory is used.

### 4.7 Fetch a trajectory (`mode 7`)

```text
title      file_name
mode       out
read_index conserved
save_traj  1
maxstep    200
select     1
```

Mode 7 requires `select`; without it the run stops. For each selected
trajectory it writes `select-<title>-<i>.xyz` with all frames and
`select-<title>-<i>.dat` with columns step, active state, `Ekin`, the
potential energy of every state and the population of every state.

### 4.8 Special snapshot reader (`mode 8`)

Mode 8 reads the initial, final and hopping frames directly from the
PyRAI3MD trajectory directory and writes the lower-case snapshot record
`geom-<title>.json`. It was written for PyRAI2MD trajectories whose
`buffer` option could omit a hop frame from `.md.xyz` while `.sh.xyz`
still had it; for PyRAI3MD, `.sh.*` is written only when a hop happens,
so a run without hops has empty hop lists.

```text
title          file_name
mode           8
read_index     conserved
classify       all
classify_state 0
align          1
align_core     1-5
output_atom    1-10
read_init      file.init.xyz
param          B 1 2 B 3 4
select         file.txt
```

`classify` controls which snapshots are read: `init` and `final` read
`<project>.md.xyz`, `hop` reads `<project>.sh.xyz` and `all` reads both.
The snapshots are written as `ini.pmd.xyz`, `fin.pmd.xyz` and
`hop.pmd.xyz`; with `align 1` the aligned versions are
`aligned-ini.pmd.xyz`, `aligned-fin.pmd.xyz` and
`aligned-hop.pmd.xyz`.

`classify_state k` selects the hop from state `k+1` to state `k`, with
0-based states. `0` means a hop from S1 to S0. Only the last matching
hop of each trajectory is kept. It has no effect on initial or final
snapshots. Reverse hops from a lower to a higher state are not selected
by this key.

`read_init` reads the initial frames from a `.init` or `.init.xyz` file
instead of `md.xyz`. That file carries no energies, so the initial
potential energies are written as 0. It overwrites `ini.pmd.xyz`.

Parameter files use their own names: `param-<title>.pmd.ini`,
`param-<title>.pmd.fin` and `param-<title>.pmd.hop`. Rows have the same
layout as mode 5. With no `param` the run writes the snapshots and
alignment, then stops before computing parameters.

A common use of `classify_state 0` is a list of ground-state
trajectories: the last hop to S0 is then the hop of interest.

## 5. Geometrical parameters

`param` is a sequence of parameter codes and their 1-based atom
indices. A new parameter starts at each code in the list below. The same
sequence may be given in a file; `param file.txt` is read when the first
token is an existing file.

| code | parameter | indices | range |
|---|---|---|---|
| `B` | bond length | 2 | Angstrom |
| `A` | angle | 3 | 0 to pi |
| `D` | dihedral angle | 4 | 0 to pi |
| `D2` | dihedral angle | 4 | -pi to pi |
| `D3` | dihedral angle | 4 | 0 to 2pi |
| `DD` | double dihedral | 6 | 0 to pi |
| `DD2` | double dihedral | 6 | -pi to pi |
| `DD3` | double dihedral | 6 | 0 to 2pi |
| `O` | out-of-plane angle | 4 | 0 to pi |
| `P` | plane-plane angle | 6 | 0 to pi |
| `RMSD` | root-mean-square deviation | see section 6 | Angstrom |

Example: `param B 1 2 B 3 4 RMSD` defines one bond, a second bond and an
RMSD. `threshold` must then carry three values.

## 6. RMSD options

`RMSD` takes optional `name=value` tokens. There is no space around the
equal sign. Tokens are case-insensitive. Several `no=` or `on=` tokens
may appear.

| option | value | meaning |
|---|---|---|
| `no=` | element name | exclude an element; repeatable |
| `on=` | element name | include only this element; repeatable |
| `pick=` | `1,2,3` | 1-based atom indices, comma separated, no spaces |
| `align=` | `hung` or `no` | match atoms with the Hungarian algorithm before RMSD |
| `coord=` | `cart` or `sym` | Cartesian coordinates or symmetry functions |
| `cut=` | 1 or 2 | cutoff function: 1 cosine, 2 tanh cubed |
| `ver=` | 1, 2, 3 or 4 | symmetry function G1, G2, G3 or G4 |
| `rc=` | number | cutoff radius in Angstrom, default 6 |
| `rs=` | number | Gaussian center, default 0 |
| `eta=` | number | Gaussian exponent, default 1.2 |
| `zeta=` | number | angular exponent, default 1 |
| `lambda=` | 1 or -1 | angular factor; only the sign is used |

`ref_geom` is required for RMSD. When the reference file has several
structures the first one is used; the legacy convention is that the
first structure is the major product. Excluded or non-selected atoms
drop out of both structures before the deviation is computed.

## 7. Pruning

Pruning truncates a trajectory at the first frame where a monitored
distance exceeds its threshold. It applies whenever trajectories are
read: modes 2 to 8; diagnosis (mode 1) does not prune.

Distance mode (`prune_type dist`):

```text
prune_type  dist
prune_index 1 2, 3 4, 5 6
prune_thrhd 3.0 4.0
```

`prune_index` is a comma-separated list of atom pairs. `prune_thrhd`
gives one threshold per pair; a missing tail repeats the last threshold
and extra thresholds are ignored. The example monitors 1-2 at 3.0
Angstrom, and 3-4 and 5-6 at 4.0 Angstrom.

Fragment mode (`prune_type frag`):

```text
prune_type  frag
prune_index 1 2 3 4, 5 6 7 8
prune_thrhd 4.0
```

`prune_index` has two comma-separated index groups and the run stops
when the distance between the two group centers exceeds the single
threshold.

## 8. What the PyRAI3MD reader reads

- Geometry frames: `<project>.md.xyz`. The comment line is
  `<title> coord <step> state <k>[ to <m> CI]`; the reader takes the
  step and state from it and ignores a trailing `Lattice="..."` cell
  key.
- Energies: `<project>.md.energies`, columns
  `time / Epot / Ekin / Etot / per-state energies`. The header line is
  skipped.
- Populations: `<project>.md.pop`, columns
  `time / per-state populations / 0-1 active state`. The header line is
  skipped. A run without surface hopping writes no population record;
  populations are then zero-filled and the current state comes from the
  frame comment.
- Hops: a frame whose comment carries `to <m> CI` is a hop event.
  `<project>.sh.xyz`, `.sh.energies` and `.sh.pop` are the optional hop
  records used by mode 8.
- There is no log parsing for `prog pyrai3md`: populations never come
  from a `Gnuplot` block.

Multi-center partitioned output (`<project>.md.1.xyz`, comments with the
plural `states`) is refused by the reader; only the whole-system
`<project>.md.*` trajectory is analyzed.

The `molcas` reader still parses the legacy PyRAI2MD log and its
`.md.energies` / `.md.xyz` files, kept for archived trajectories.

## 9. Output files

| file | mode |
|---|---|
| `<title>.traj.log` | all; created at start, appended by later steps |
| `complete` | 1 |
| `conserved`, `energy_drift.txt` | 2 |
| `<title>.json`, `Geom-<title>.json`, `Ini/Fin/Hop.S<n>.xyz` | 3 |
| `average-<title>.dat`, `hop-energy-<title>.dat` | 4 |
| `state-pop-<title>.json` | 4, `save_data >= 1` |
| `energy-profile-<title>.json` | 4, `save_data >= 2` |
| `param-<title>.S<n>.ini/.fin/.hop`, `aligned-Ini/Fin/Hop.S<n>.xyz` | 5 |
| `plot-<title>.json` | 6 |
| `select-<title>-<i>.xyz`, `select-<title>-<i>.dat` | 7 |
| `geom-<title>.json`, `ini/fin/hop.pmd.xyz`, `aligned-ini/fin/hop.pmd.xyz` | 8 |
| `param-<title>.pmd.ini/.fin/.hop` | 8 |

## 10. Worked example

```text
title      demo
prog       pyrai3md
read_index 1 2-10
mode       1
minstep    50
```

Diagnose ten PyRAI3MD trajectories and write the ones that completed
with at least 50 recorded steps to `complete`. A second run with
`mode 2` and `read_index complete` removes the non-conserved ones, a
third run with `mode 3` extracts the frames, and a fourth run with
`mode 4` computes average populations. `mode 5` then classifies the
final structures, `mode 6` writes the plot data and `mode 7` fetches
individual trajectories.
