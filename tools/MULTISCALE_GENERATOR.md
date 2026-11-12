# Multiscale environment generator

`tools/multiscale_generator.py` builds a solvent or aggregate
environment, prepares classical MM files, and produces the combined
initial conditions for a multiscale trajectory.

```bash
python tools/multiscale_generator.py creation.in
python tools/multiscale_generator.py            # print the keyword summary
```

## 1. Creation file

One `keyword value` pair per line:

```text
title         waterbox
mode          create
env_type      solvent
solute        solute.xyz
solvent       solvent.xyz
nshell        2
density       1.0
solbox        60
mass          18.0
packmol       /path/to/packmol
cpus          8
```

| keyword | meaning |
|---|---|
| `title` | calculation title |
| `mode` | `create`, `merge`, `read`, `edit`, `rdf`, `den` or `mm` |
| `env_type` | `solvent` or `aggregate` |
| `solute`, `solvent` | molecule xyz files |
| `nshell` | number of solvent shells |
| `density`, `solbox`, `mass` | solvent packing parameters |
| `packmol` | packmol executable |
| `cell`, `nmol`, `center` | unit cell and the center molecule for an aggregate |
| `a`, `b`, `c`, `alpha`, `beta`, `gamma` | cell parameters |
| `na`, `nb`, `nc` | cell translations in each direction |
| `radius` | aggregate cutoff radius |
| `align` | target xyz for orientation alignment |
| `method`, `ninitcond`, `seed`, `format`, `temp` | initial-condition sampling |
| `read`, `skip`, `freq` | read equilibrated conditions from trajectories |
| `reorder`, `expand`, `c_atom`, `v_atom`, `nsol` | environment ordering and selection |
| `combine`, `read_init`, `init_to_xyz`, `scale`, `edit_atom` | merge and edit initial conditions |
| `append_init`, `remove_init` | append or remove frames of an initial-condition file |
| `file` | list of trajectory directories |
| `check_box`, `center_rdf`, `center_type` | box correction and RDF analysis |
| `cpus` | preparation processes |

## 2. Modes

| mode | action |
|---|---|
| `create` | build the environment and write the combined system |
| `merge` | merge a solute/solvent with sampled initial conditions |
| `read` | read the last frames of equilibrated trajectories |
| `edit` | edit initial conditions of selected atoms |
| `rdf` | compute radial distribution functions |
| `den` | compute radial densities |
| `mm` | prepare the classical MM input files |

## 3. Initial conditions

Sampling uses the PyRAI3MD sampler (`workflow/sampling.py`), the same
one `traj_generator.py` uses. `format` names the frequency file:
`molden`, `g16`, `orca`, `bagel`; `xyz` and the archive formats are
read as pre-sampled conditions. The tool builds the same
`(symbol, x, y, z, vx, vy, vz, mass, charge)` rows as the trajectory
generator, so a condition produced by either tool can be merged by the
other.

Atomic masses and van der Waals radii come from
`pyrai3md/schema/reference.py` (`ATOMIC_MASS`, `VDW_RADII`); the tool no
longer imports a PyRAI2MD `Atom` class.

## 4. Notes

- `packmol` must be installed and reachable; a `create` run invokes it.
- The environment files are written beside the creation file.
- The tool imports `traj_generator.py` from the same `tools/`
  directory for the shared sampler; keep the tools together.