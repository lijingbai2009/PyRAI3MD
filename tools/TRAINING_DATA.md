# Training data tool

`tools/training_data.py` extracts labelled frames from finished
calculations and maintains the resulting dataset. Extraction is a
framework feature (`jobtype read`); the tool owns the filtering,
statistics and JSON file.

```bash
python tools/training_data.py extract  read.yaml [-o data.json]
python tools/training_data.py shuffle  data.json [-s 1234] [-g 200]
python tools/training_data.py split    data.json -p 200 [-r 0]
python tools/training_data.py merge    data.json data2.json [...]
python tools/training_data.py select   data.json --index 1-100,200
python tools/training_data.py describe data.json
```

## 1. Read-only extraction

A `jobtype read` run starts no engine and writes no data file. It walks
the paths named by the `read` section, hands each one to the parser of
the single theory skill, and returns the frames. `extract` calls that
run, keeps the requested label channels and saves the canonical
PyRAI3MD training-data JSON.

### 1.1 YAML input

```yaml
control:
  title: read_run
  jobtype: read
  theory: [molcas]

read:
  path: [run_1, run_2]      # directories, files, or a list file
  project: ''               # empty: discovered from the artifacts
  profile: ''               # empty: derived from the labels
  energy: true
  gradient: true
  nac: false
  soc: false

molecule:
  charges: 0
  multiplicity: 1
  ci: [1]
  spin: [0]

molcas:
  method: casscf
  basis: ano-rcc
```

### 1.2 Classic input

Sections are delimited by the next `&SECTION` header; there is no
`&end` terminator.

```text
&control
title read_run
jobtype read
theory molcas

&read
path dirs.txt
energy 1
gradient 1
nac 0
soc 0

&molecule
charges 0
multiplicity 1
ci 1
spin 0
```

### 1.3 Keywords

| keyword | meaning |
|---|---|
| `path` | one calculation directory, one output file, or a file listing one path per line |
| `project` | output file stem; empty means the directory basename, the file stem, or the unique artifact stem discovered in the directory |
| `profile` | skill profile used to parse iterative artifacts (MD/optimize); empty derives it from the requested labels |
| `energy` | ask for the state energies |
| `gradient` | ask for the state gradients |
| `nac` | ask for the nonadiabatic couplings |
| `soc` | ask for the spin-orbit couplings |

Rules:

- `read` binds exactly one theory skill, no reference, `driver: main`
  and no multiscale partition.
- At least one label must be enabled.
- `energy` is always requested when any label is: every other channel
  is a derivative or a coupling registered under an energy evaluation.
- A path may be a list file; relative entries are anchored to the list
  file's directory.
- A directory's project is discovered from its unique artifact stem, so
  a trajectory directory `<title>_1` holding `<title>.md.xyz` works
  without naming `project`.
- `read` never validates an engine installation and never creates a
  scratch directory.

### 1.4 Reading iterative artifacts

The parser branches on the skill profile. A directory written by an
optimization or an MD run needs the matching profile:

```yaml
read:
  path: [opt_run]
  profile: casscf_geometry_optimization
```

For a single-point directory the profile is derived from the labels
(e.g. `energy gradient`).

### 1.5 Supported skills

File-based readers exist for OpenMolcas, ORCA, BAGEL, xTB, OpenQP, CP2K
and LAMMPS. PySCF, QDM and the neural network potential compute in
process and write no directory a reader could reopen; a `read` run on
one of them reports that by name.

## 2. What `extract` writes

The dataset is the canonical `pyrai3md-training-data` JSON
(`pyrai3md/training_data/dataset.py`):

| field | shape |
|---|---|
| `xyz` | `(nmol, natom, 3)` Angstrom |
| `symbols` | `(nmol, natom)` |
| `energy` | `(nmol, nstate)` hartree |
| `grad` | `(nmol, nstate, natom, 3)` hartree/bohr |
| `nac` | `(nmol, nnac, natom, 3)` hartree/bohr |
| `soc` | `(nmol, nsoc)` cm-1 |
| `unit` | the unit block of the channels |

`nmol` is the number of frames. `extract`, the data operations and
`describe` print the shape of every present channel with these axis
names, e.g. `grad: (100, 3, 6, 3) (nmol, nstate, natom, 3)`.

A requested channel is written only when every selected frame carries
it. If only some frames carry it, the channel is omitted and the
number of frames that lacked it is printed; a partial channel would
break the container's uniform-frame contract.

The tool prints the frame count (`nmol`), `natom`, `nstate/nnac/nsoc/npair`,
the shape of every channel with its axis names and the unit block.

## 3. Data operations

All operations load the canonical JSON or a legacy `data.json` and save
the canonical format.

### shuffle

```bash
python tools/training_data.py shuffle data.json -s 1234
python tools/training_data.py shuffle data.json -s 1234 -g 200
```

Writes `data-shuffled.json`. `-s` seeds the permutation (default
1234). `-g N` shuffles the first `N` frames separately from the rest.

### split

```bash
python tools/training_data.py split data.json -p 200
python tools/training_data.py split data.json -p 200 -r 0
```

Writes `data-set-1.json` (the first `N` frames) and
`data-set-2.json` (the rest). The input is shuffled first by default;
`-r 0` keeps the original order.

### merge

```bash
python tools/training_data.py merge data.json data2.json data3.json
```

Writes `data-merged.json`. The first dataset is shuffled by default
(`-r 0` skips), then every other dataset is appended in order. The
container refuses an append whose molecule or channel layout differs.

### select

```bash
python tools/training_data.py select data.json --index 1-100,200
```

Writes `data-selected.json`. Indices are 1-based; ranges are inclusive.

### describe

```bash
python tools/training_data.py describe data.json
```

Prints `nmol`, `natom`, `nstate`, `nnac`, `nsoc`, `npair`, the shape of
every present channel with its axis names (for example
`grad: (100, 3, 6, 3) (nmol, nstate, natom, 3)`) and the unit block. It
does not write a file.

## 4. Legacy files

A legacy PyRAI2MD `data.json` carries `ncharge` and the per-frame
arrays. The container ignores `ncharge` and reads the rest, so the
operations accept it directly and write the new format. `soc` stays a
per-frame vector, which is the canonical `(nframe, nsoc)` layout.

## 5. Limits

- Multi-center partitioned MD output (`md.1.xyz`, `states` comment) is
  not read; the analyzer reports it rather than mixing center frames.
- No channel is invented: `cell`, `pbc` and `charge` are written only
  when a skill actually returns them.
- `extract` reads directories sequentially; it does not parallelize
  across paths.