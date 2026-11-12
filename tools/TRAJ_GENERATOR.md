# Trajectory generator

`tools/traj_generator.py` samples initial conditions from a frequency or
initial-condition file and writes a batch of trajectory calculation
directories, one per sample, each with a Slurm run script.

```bash
python tools/traj_generator.py sampling.in
python tools/traj_generator.py            # print the keyword summary
```

The generator runs in the directory that holds the frequency file,
the program templates and, for `prog pyrai3md`, the YAML template
`input.yaml`. Run it with no argument to print the keyword summary; the
legacy `for help` argument is treated as a file name and is not a help
route.

## 1. Units and normal modes

| quantity | conversion |
|---|---|
| length | 1 Bohr = 0.529177249 Angstrom |
| velocity | 1 au = 2.18769126364e6 m/s |
| time | 1 au = 2.41888432658e-17 s |
| temperature | 1 au = 315775.024804 K |
| mass | 1 au = 9.1093837015e-31 kg |
| wavenumber | 1 cm-1 = 4.55633518e-6 au |
| molar mass | 1 g/mol = 1822.8885 au (amu) |

Coordinates are written in Angstrom and velocities in Bohr per atomic
time unit. The sampled velocities are not mass-weighted.

Normal modes follow the program's own convention, and the readers
bring all four files to one internal convention, unnormalized and
unmass-weighted:

| file | program convention | reader |
|---|---|---|
| `.freq.molden` | Molcas: unnormalized, unmass-weighted | used as read |
| `.freq.g16` | Gaussian: normalized, unmass-weighted | converted to unnormalized |
| `.freq.orca` | ORCA: normalized, unmass-weighted | converted to unnormalized |
| `.freq.bagel` | BAGEL: normalized, mass-weighted | demass-weighted and used unnormalized |

Imaginary frequencies and their modes are dropped with a warning.
`temp` controls the sampling only; it does not change the trajectory
calculation, so make it match the temperature of the run you intend to
perform.

## 2. Input file names select the format

The project name is the part of the `input` file name before the first
dot. All templates and job directories are named after it, so use a
project name without dots.

| `input` name | meaning | action |
|---|---|---|
| `<name>.freq.molden` | Molcas frequency file from `&MCKINLEY` | sample |
| `<name>.freq.g16` | Gaussian log with frequencies and normal modes | sample |
| `<name>.freq.orca` | ORCA `.hess` file | sample |
| `<name>.freq.bagel` | BAGEL log from a `"title" : "hessian"` run | sample |
| `<name>.init.newtonx` | NewtonX `final_output` | read |
| `<name>.init.tar.xz` | Gen-FSSH initial-condition archive | read |
| `<name>.init.tar.xz2` | Gen-FSSH initial-condition archive, second layout | read |
| `<name>.xyz` | multi-frame XYZ of starting structures | read |
| anything else | not recognized | error |

Only the four frequency formats are sampled. The legacy Gaussian
`.freq.fchk` input (written with `Freq=SaveNormalModes`) is not read;
supply the `.freq.g16` log instead.

The XYZ path needs a note. An `input` line ending in `.init.xyz` is
accepted, but the XYZ reader always opens `<name>.xyz`, the stem before
the first dot, and ignores the `.init` token when locating the file. Put
the frames in `<name>.xyz`. Each frame contributes one starting
structure, velocities are set to zero, and the run stops when the file
holds fewer frames than `nodes` times `jobs`.

A frequency calculation needs the normal-mode output, not an optimized
geometry alone. Rename the program output to the names above:

| program | original | expected |
|---|---|---|
| Molcas | `.freq.molden` | `.freq.molden` |
| Gaussian | `.log` | `.freq.g16` |
| BAGEL | `.log` | `.freq.bagel` |
| ORCA | `.hess` | `.freq.orca` |

## 3. Program templates

Every generator reads templates from the current directory and copies
them into each trajectory directory. The checks below run before any
directory is written.

| `prog` | required | optional or notes |
|---|---|---|
| `molcas` | `<name>.inp`, and `<name>.StrOrb` or `<name>.JobIph` | `<name>.basis` supplies per-atom labels, `<name>.key` is copied for QM/MM |
| `bagel` | `<name>.bagel` (JSON), `<name>.archive` | `<name>.charge` |
| `nxbagel` | `control.dyn`, `bagelinput.basis.inp`, `bagelinput.part1.inp`, `part2.inp`, `part3.inp` | `jiri.inp`, `sh.inp`, `therm.inp`, `<name>.archive` |
| `pyrai3md` | `input.yaml` | the auxiliary files listed in section 7 |
| `fromage` | `fromage.in`, `mh.temp`, `rl.temp`, `ml.temp`, `xtb.input`, `xtb_charge.pc`, `<name>.StrOrb` | |
| `orca` | `<name>.orca` | coordinates are appended to the template |
| `openqp` | `<name>.openqp` | `<name>.json` orbital data, if present |
| `lammps` | `<name>.in`, `<name>.data` | `<name>.in.init` and `<name>.in.settings` are read into every job, so keep them present |
| `cp2k` | `<name>.cp2k` | |

The Molcas `.inp` is the molecular dynamics input and must request
velocities, for example `velocities=1`; the generator supplies
`<name>.xyz` and `<name>.velocity.xyz` beside it. The BAGEL `.bagel`
file is a JSON template, and the archive is the orbital file. The
NewtonX/BAGEL tree uses the four BAGEL input parts and `control.dyn`,
with `jiri.inp`, `sh.inp` and `therm.inp` copied when they exist.

The `fromage` row lists the intended inputs, but the current check
stops before generation even when `<name>.StrOrb` exists, because that
check resolves the wrong file name. Do not rely on `prog fromage` in
this revision; the other programs are unaffected.

The retired program names `pyrai2md`, `pyrai2mdnn`, `pyrai2mdmolcas`,
`pyrai2mdbagel`, `pyrai2mdorca`, `pyrai2mdhybrid` and `pyrai2mdoqp`
stop with a message pointing to `prog pyrai3md`.

## 4. Input file syntax and the `>shell` block

The sampling file has one `keyword value` pair per line, in free order.
Only the second column is read, so a trailing comment is harmless but a
value containing spaces is not supported. The keys are
case-insensitive.

A `>shell` block inserts shell commands into every generated Slurm
script:

```text
input         ethylene.freq.molden
prog          pyrai3md
cpus          8
seed          1
temp          300.0
method        wigner
nodes         1
cores         32
jobs          32
index         1
partition     normal
time          1-00:00:00
memory        3000
>shell
module load mkl
export PATH=/site/bin:$PATH
conda activate myenv
>shell
```

Exactly two `>shell` delimiters are allowed, or none at all. One
delimiter, or more than two, stops the run with a syntax error. The
text between the pair is copied verbatim into every runset and every
per-trajectory script, so this is the place for site-specific
`#SBATCH` lines, modules, environment variables and conda activation.

## 5. Keywords

| keyword | meaning | default |
|---|---|---|
| `input` | frequency or initial-condition file with its suffix | required |
| `cpus` | processes used to prepare the directories | 1 |
| `notraj` | 1 = sample and stop before writing job directories | 0 |
| `seed` | random seed; -1 draws an unseeded stream | 1 |
| `temp` | sampling temperature in K | 298.15 |
| `method` | `wigner` or `boltzmann` | `wigner` |
| `prog` | target program, section 6 | `molcas` |
| `partition` | Slurm partition | `normal` |
| `time` | Slurm time limit | `1-00:00:00` |
| `memory` | memory per trajectory in MB | 100 |
| `nodes` | number of runsets, one Slurm job each | 1 |
| `cores` | cores requested by one runset | 1 |
| `jobs` | trajectories in one runset | 1 |
| `index` | index of the first trajectory | 1 |
| `repeat` | 1 = reuse the first sample for every trajectory | 0 |
| `molcas` | Molcas installation path | site default |
| `newton` | NewtonX installation path | site default |
| `bagel` | BAGEL installation path | site default |
| `xtb` | xTB installation path | site default |
| `orca` | ORCA installation path | site default |
| `openqp` | OpenQP installation path | site default |
| `lammps` | LAMMPS installation path | site default |
| `cp2k` | CP2K installation path | site default |
| `lib_blas` | BLAS library path, used by BAGEL | empty |
| `lib_lapack` | LAPACK library path, used by BAGEL | empty |
| `lib_scalapack` | SCALAPACK library path, used by BAGEL | empty |
| `lib_boost` | Boost library path, used by BAGEL | empty |
| `mkl` | Intel oneAPI root; accepted, left commented in the script | empty |
| `mpi` | Intel MPI root; accepted, left commented in the script | empty |

The built-in program paths are the site defaults compiled into the
script:

| key | default |
|---|---|
| `molcas` | `/public/software/app/molcas` |
| `newton` | `/share/apps/NX-2.4-B06` |
| `bagel` | `/public/software/app/bagel` |
| `xtb` | `/public/software/app/xtb-6.7.1` |
| `orca` | `/public/software/app/orca_6_0_1_linux_x86-64_shared_openmpi416` |
| `openqp` | `/public/software/app/openqp` |
| `lammps` | `/public/software/app/lammps` |
| `cp2k` | `/public/software/app/cp2k-2025.2/` |

The legacy spellings `iseed` and `ncpus` are not read. A line starting
with either is ignored, and the run continues with the default `seed 1`
and `cpus 1`; use `seed` and `cpus`.

The number of trajectories is `nodes` times `jobs`. `index` names the
first one, so a later run with a higher `index` continues an existing
set without overwriting it. Each trajectory directory is
`<name>-<index>`.

## 6. Programs and generated jobs

| `prog` | per-trajectory output | parallel split |
|---|---|---|
| `molcas` | `<name>-<i>.inp`, `.xyz`, `.velocity.xyz`, `.sh`, plus the orbital file | one background process per trajectory, `MOLCAS_NPROCS=1` |
| `bagel` | BAGEL JSON and geometry/velocity inputs | threads per trajectory, `BAGELPAL` |
| `nxbagel` | NewtonX/BAGEL job tree with the four BAGEL parts and `control.dyn` | threads per trajectory, `BAGELPAL` |
| `pyrai3md` | YAML, `.init.xyz`, `run_pyrai3md.sh`, auxiliary files | `cpus-per-task` per trajectory |
| `fromage` | fromage inputs plus Molcas orbital file | one background process per trajectory |
| `orca` | `<name>-<i>.inp` with coordinates appended, `.velocity.xyz`, `.sh` | tasks per trajectory |
| `openqp` | OpenQP input, xyz and velocity files, `.sh` | `mpirun -np` tasks per trajectory |
| `lammps` | `.in`, `.data`, `.in.init`, `.in.settings`, `.sh` | `mpirun -np` tasks per trajectory |
| `cp2k` | CP2K input, xyz and velocity files, `.sh` | `mpirun -n` with the runset task count |

For every program the run writes one `runset-<n>.sh` per node and a
`runall.sh` that submits them:

```bash
sh runall.sh
```

Each runset requests one node, the `time` limit, the partition and
`jobs` trajectories, launches them in the background and waits. Every
trajectory also gets its own script, so one trajectory can be submitted
alone. The runset memory is `int(memory * cores * factor)`:

| programs | factor |
|---|---|
| `molcas`, `fromage` | 1.333 |
| `bagel`, `nxbagel`, `pyrai3md` | 1.1 |
| `orca`, `openqp`, `lammps`, `cp2k` | 1.0 |

Molcas also receives `MOLCAS_MEM=memory` and runs one rank per
trajectory, so set `cores` equal to `jobs`. BAGEL and NewtonX/BAGEL
split the cores per trajectory as `int(cores / jobs)` threads and do
not check the division; keep `cores` an integer multiple of `jobs`.
ORCA, OpenQP, LAMMPS and PyRAI3MD use `int(cores / jobs)` per
trajectory. The CP2K runset requests `cores` tasks and starts every
background trajectory with `mpirun -n $SLURM_NTASKS`, so keep `jobs 1`
there unless your site tolerates the oversubscription.

## 7. PyRAI3MD mode

`prog pyrai3md` requires a YAML template named `input.yaml` in the
working directory. For each trajectory it creates:

```text
<name>-<idx>/
  <name>-<idx>.yaml       # template with control.title rewritten,
                          # md.initcond: read, md.ninitcond: 1
  <name>-<idx>.init.xyz   # one starting frame (coordinates + velocities)
  run_pyrai3md.sh         # `pyrai3md <name>-<idx>.yaml`
```

The YAML is loaded and re-dumped, so formatting and comments are
normalized; every value is preserved. The sampler's molecule must match
the template's `atoms` exactly (same symbols, same order); a mismatch
stops generation before any directory is written.

Auxiliary files are copied under their own names when they exist:
`<name>.molcas`, `<name>.StrOrb`, `<name>.JobIph`, `<name>.bagel`,
`<name>.archive`, `<name>.orca`, `<name>.openqp`, `<name>.json`,
`<name>.xtb`, `<name>.charge`, `<name>.basis`, `<name>.in`,
`<name>.data`, `<name>.in.init`, `<name>.in.settings` and
`<name>.cp2k`. The template names them, so renaming would break its
references.

`runset-<n>.sh` runs the trajectories of one runset in the background;
`runall.sh` submits every runset:

```bash
sh runall.sh
```

## 8. Sampling, initial conditions and `repeat`

`method wigner` samples the quantized harmonic oscillator: a thermal
vibrational level is drawn per mode and the coordinate and momentum are
rejection-sampled against that level's Wigner function. `method
boltzmann` uses the classical Gaussian widths. Any other value stops
the sampler with an error.

Both methods write the sampled conditions beside the input:

- `wigner-<name>-<temp>.xyz` and `wigner-<name>-<temp>.velocity`;
- `boltzmann-<name>-<temp>.xyz` and `.velocity` for the other method.

`notraj 1` stops after writing those files, which is the way to inspect
or reuse the sample without creating job directories. With a
pre-sampled input format (`.init.newtonx`, `.init.tar.xz`, `.init.tar.xz2`,
`<name>.xyz`) there is nothing to sample, and `notraj 1` simply loads
and stops.

`repeat 1` draws or reads one condition and reuses it for every
trajectory, which is the control run for a single starting geometry.
Without it every trajectory gets its own sample.

## 9. Resource notes

`cpus` is the number of local worker processes that prepare
directories; it is not a Slurm setting. `cores`, `jobs`, `nodes`,
`time`, `memory` and `partition` are written into the Slurm scripts.
The generator does not validate the core and job relationship beyond
the integer division above, so check it yourself, especially for
Molcas (one rank per trajectory) and BAGEL (threads per trajectory).

The `>shell` block is inserted into the runsets and the per-trajectory
scripts, so it can carry site-specific `#SBATCH` constraints as well as
modules and environment setup.

## 10. Tips

- Run the generator in a directory containing the frequency file, the
  program templates and, for PyRAI3MD, `input.yaml`.
- Rename program outputs to the four `.freq.*` names before running.
- Keep the project name free of dots; everything is derived from the
  stem before the first dot.
- `nodes` times `jobs` is the number of trajectories.
- Use `index` to add a new batch after an existing one.
- Set Molcas `cores` equal to `jobs`; set BAGEL and NewtonX/BAGEL
  `cores` to an integer multiple of `jobs`.
- `repeat 1` is the control run from the first sample.
- The `>shell` block must use exactly two delimiters, or none.
