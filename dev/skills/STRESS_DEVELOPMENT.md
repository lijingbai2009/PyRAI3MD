# PyRAI3MD Periodic Stress Interface Design (CP2K, LAMMPS, xTB/GFN-FF)

Last updated: 2026-11-12 19:20 (the requested-block rule is recorded in section 1; the zero fill is unrequested-only. Earlier: 2026-10-22 22:30 (re-read against the code: the CP2K `FILENAME`/parser pair, the LAMMPS `pyrai_s*` variables and `step_data_files`, the xTB `gradlatt` contraction with its volume guard, the `stress_method` field, `_build_skill_pairs` and the `supports_pbc` gate all still match what this document states; only the dated blockquotes are historical and were left as written. Nothing was re-measured. Before it, 2026-10-11 00:00)

---

This document summarizes the stress-tensor output, key-component extraction, unit and sign conversions for the three programs, and the division of responsibilities when MD/NPT is implemented by PyRAI3MD or another external driver. It is an interface design and implementation reference; it does not claim that existing calculators have already been modified accordingly. The scope is classical nuclei in a three-dimensional periodic cell; two-dimensional materials or systems with vacuum require a separate definition of the effective volume and the pressure-controlled directions.

> **Implementation status (2026-09-24, see CHANGELOG #214).** All three skills
> have been switched to the "write a dedicated file → parse from that file"
> pattern, consistent with the write-file→read-file pattern used for
> energies/gradients; the full output is no longer scanned:
> - **CP2K**: `&FORCE_EVAL &PRINT &STRESS_TENSOR ON` plus
>   `FILENAME ={project}_stress_tensor.dat` (the `=` pins the exact name);
>   `parse_stress_tensor` reads that file with format tolerance
>   (case-insensitive header, strips the `STRESS|` prefix, accepts lowercase
>   x/y/z, **excludes the eigenvalue block containing `eigen`**, takes the last
>   block when several are present), and falls back to the main log when the
>   file is missing. This lands the approach recommended in §2.1 (the file name
>   uses the project prefix to avoid calcdir collisions across multiple tasks).
> - **LAMMPS**: the single-point `run 0` keeps
>   `compute Pconf all pressure NULL virial`, and additionally uses
>   `variable pyrai_s* equal c_Pconf[k]` +
>   `print "STRESS ${...}" file {title}.stress` to write the six
>   components (xx yy zz xy xz yz); the parser reads that file and nothing
>   else, and a missing file is an error rather than a zero fill — as of
>   #234 the §4.1 log thermo fallback is gone, along with the print's
>   eight-significant-digit quantization it carried. `{title}` is
>   `output_basename(project, profile)`, the same stem as the input, script
>   and log, and `step_data_files()` names it for writer and parser alike.
>   This is the concrete implementation of the
>   §4.2 statement that the single point does not depend on ave/time (switched
>   to `print ... file`).
> - **xTB**: already reads from the dedicated data file `gradlatt`
>   (§3.2's `G.T@H/|det H|`) without scanning `.out`; a cell-volume consistency
>   guard was added (a relative deviation >1e-3 between `|det H|` and the
>   requested cell volume flags a stale file → zero-filled).
> - **Request propagation**: `stress` is now registered in the DEFAULT of
>   `schema/reference.py` for lammps/xtb (previously only cp2k had it);
>   otherwise, in framework mode
>   `qualification/certification/handler.py::_build_skill_pairs` does not route
>   that key into the skill section and the stress request is silently dropped.
>
> End-to-end validation (remote `tmp_20260924_214438`): for all three skills,
> the complete 3×3 components in the `&stress tensor` section of the log match
> the backend's raw values component-by-component after sign flip and unit
> conversion (CP2K 57260.34 bar, LAMMPS −4458.29 bar, xTB 25901.37 bar).

> **External strain stress in CP2K (2026-10-11, see CHANGELOG #322).** The
> cp2k_skill `stress_method: num` route is implemented (§2.4): the tensor is
> differentiated from strain single points inside the skill, it fills every
> state a TDDFPT run solved, and it is the only route that reaches an excited
> state, because the engine refuses a tensor for one. The validation numbers
> quoted above are for the analytical route; the numerical route has so far
> only been checked against a closed-form model with the engine replaced by a
> stand-in (deviation `1.093e-09` relative along the six directions, O(delta²)
> truncation `1.8333e-05` for the isotropic scope at `delta = 1e-3`). Its
> comparison against CP2K's own analytical tensor, and the excited-state
> accounting, are open until the two shipped cases run on a deployment.

## 1. Unified definitions and return interface

It is recommended that a calculator receive only the current elements, positions, and cell, and return:

```text
evaluate(species, positions, cell)
    -> potential_energy
    -> forces
    -> stress_potential
```

Unified conventions:

- **`stress_potential`** is a Cartesian 3×3 tensor, tensile-positive, excluding the atomic kinetic-energy term.
- **Sign convention**: the compressive-positive pressure tensor is denoted P; the tensile-positive total stress is denoted σ, satisfying σ = -P.
- **Potential stress**: defined as σ_pot = (1/V) ∂U/∂ε; the strain derivative is taken at fixed fractional coordinates, deforming the cell and the atomic positions simultaneously.
- **Units**: if internal units are eV and Å, the stress is in eV/Å³. If atomic units are retained, everything is in Hartree/Bohr³; mixing is forbidden.
- **Volume division**: the stress is already divided by the volume; a virial or an energy derivative with respect to strain is usually not divided by the volume — the variable name alone cannot be trusted.
- **Failure mode**: when unsupported or when extraction fails, return an explicit error/unavailable status; do not fake a valid result with an all-zero tensor.
- **Requested block**: a stress tensor the run requested but the completed output does not carry is a failure, not a zero (CP2K today). A zero tensor is reserved for a property the run did not request.

With no additional constraints, the driver assembles:

```text
P_kin[α,β] = sum_i m_i c_i[α] c_i[β] / V
P_total   = P_kin - stress_potential
σ_total   = stress_potential - P_kin
p_total   = trace(P_total) / 3
```

Here c is the thermal velocity consistent with the integrator. The kinetic pressure formula has no 1/2 factor; its trace satisfies `trace(P_kin)/3 = 2 E_kin/(3V)`.

## 2. CP2K

### 2.1 Enabling the calculation and printing

Merge the following settings into the existing FORCE_EVAL; do not create an unnecessary duplicate FORCE_EVAL block:

```text
&FORCE_EVAL
  ! keep the existing METHOD, DFT, SUBSYS, etc.
  STRESS_TENSOR ANALYTICAL
  &PRINT
    &STRESS_TENSOR ON
      FILENAME =stress_tensor.dat
    &END STRESS_TENSOR
  &END PRINT
&END FORCE_EVAL
```

- `STRESS_TENSOR ANALYTICAL` enables the calculation and the PRINT block enables the output; both are required.
- The method used must support analytical stress; check its NUMERICAL support when necessary (for CP2K the answer is no for an excited state -- measured -- which is why §2.4 exists, and the engine-side numerical route is not offered by the skill at all). A mode that computes only the diagonal elements cannot provide the full tensor.
- `FILENAME =stress_tensor.dat` sets the exact file name; removing FILENAME falls back to the default main output.
- The current trunk provides `STRESS_UNIT`, allowing `STRESS_UNIT bar` inside PRINT/STRESS_TENSOR; older versions may not support it. Read the actual unit label in the output; do not assume all versions use the same units.
- When used as an external calculator, run single-point evaluations, e.g. `RUN_TYPE ENERGY_FORCE`; do not start CP2K's own MD integration.

### 2.2 Extraction and conversion

1. Find the total stress matrix block corresponding to the current evaluation and read the nine values by x/y/z rows and columns.
2. Do not mistake the `1/3 Trace`, the determinant, the eigenvalues, or one entry of `COMPONENTS` for the tensor.
3. If the log contains multiple stress blocks, select the final valid block for the current call/configuration; guard against reading a file from a previous run.
4. CP2K's native stress uses the compressive-positive convention. Converting to this interface: `stress_potential = -stress_cp2k`, together with a unit conversion.
5. ASE's CP2K calculator has already flipped the sign; when using the wrapper, confirm the return convention first and do not negate again.

### 2.3 Difference from the MD stress output

When CP2K runs its own MD, the following can additionally be set:

```text
&MOTION
  &PRINT
    &STRESS ON
      FILENAME =md.stress
      &EACH
        MD 1
      &END EACH
    &END STRESS
  &END PRINT
&END MOTION
```

`FORCE_EVAL/PRINT/STRESS_TENSOR` does not include the atomic translational kinetic-energy term; the MD `MOTION/PRINT/STRESS` includes the kinetic energy and applicable constraint contributions. An external MD driver should read the former. The electronic kinetic-energy stress that appears in the electronic-structure output is part of the potential-surface stress, not the atomic kinetic pressure that the external driver must add, and must not be removed.

### 2.4 External strain finite differences (`stress_method: num`)

When the engine cannot supply a tensor -- which is the normal case for an
excited state, since CP2K aborts on `STRESS_TENSOR` under TDDFPT -- the same
`stress_potential` is obtained by differentiating the total energy over a
strained cell, with the cell and the Cartesian positions deformed together so
the fractional coordinates stay fixed (§1's potential-stress definition). The
row-vector conventions, the six directions with the shear slots at 1/2, the
central-volume denominator and the step-size guidance are in
[CP2K_SKILL_DEVELOPMENT.md](CP2K_SKILL_DEVELOPMENT.md) section 6.6; the
implementation is `skills/cp2k_skill/service/stress/`.

Interface points worth stating here, because each one is a place another skill
could get it wrong:

- **No engine keyword.** The route asks CP2K for ordinary single points, so
  `STRESS_TENSOR` is not emitted, the stress print block is not emitted, and
  the parser's `_pack_stress` is skipped: the driver owns the whole
  `(nstate, 3, 3)` array.
- **Sign.** The central difference `[E(+) - E(-)] / (2*delta*V0)` is already
  tensile-positive, so the negation that §2.2 step 4 requires for an engine
  print must NOT be applied to it. Applying both is a factor-of-minus-one bug
  that no shape check will catch.
- **Unit.** The energies are Hartree and the cell Å, so the raw quotient is
  Hartree/Å³; the skill converts with a factor derived from its own
  `eV/Å³`-to-GPa table entry and the Hartree-to-eV constant, and returns GPa
  so the framework's central GPa→Hartree/Bohr³ step stays the single
  conversion (§1's "mixing is forbidden" rule).
- **Isotropic scope.** With `B = I` the difference yields `tr(sigma)`, which
  the skill divides by 3 onto the diagonal rather than opening a scalar
  channel: the shape contract stays `(nstate, 3, 3)` for every consumer, and
  the shear components are then legitimately absent, not zero-measured.
- **Failure policy.** §1's "do not fake a valid result with an all-zero
  tensor" is enforced by raising: a failed point, a dropped TDDFPT root, or a
  root that moved past half the smallest central gap aborts the stress
  instead of entering the difference as a zero.

The points are evaluated as one batch by the skill itself (a spawned pool, or
one blocking submission each under the slurm arm), because how many engine
evaluations a stress method costs is a property of that method.

## 3. GFN-FF in xTB

This refers specifically to the original `xtb --gfnff`, not AMS GFNFF or the standalone gfnff library.

### 3.1 Output paths

The input must contain lattice and periodicity information, e.g. a periodic Turbomole coord file:

```bash
xtb coord --gfnff --grad > xtb.out
```

For a gradient calculation on a system correctly recognized as periodic, the output includes `energy`, `gradient`, and `gradlatt`. File names may be affected by namespace and similar settings; the interface should locate them according to the actual run configuration.

`gradient` is the energy gradient with respect to atomic coordinates — forces require a sign flip; `gradlatt` is the lattice gradient, not a stress already divided by the volume.

### 3.2 gradlatt layout and transformation

```text
$gradlatt
  cycle = ... SCF energy = ... |dE/dlatt| = ...
  ax ay az
  bx by bz
  cx cy cz
  dE/dax dE/day dE/daz
  dE/dbx dE/dby dE/dbz
  dE/dcx dE/dcy dE/dcz
$end
```

Reading one lattice vector per file line, denote the first three rows H and the last three rows G:

```text
V = abs(det(H))
sigma_raw = G.T @ H
stress_potential = sigma_raw / V
```

H is in Bohr; G is in Hartree/Bohr; `sigma_raw` is in Hartree and equals the energy derivative with respect to strain; the final stress is in Hartree/Bohr³, tensile-positive. If the internal representation switches to lattice vectors arranged as columns, the matrix formula must be adjusted accordingly — the transpose placement cannot be copied blindly.

Example of extracting the last record and saving it in GPa:

```python
from pathlib import Path
import numpy as np

lines = Path("gradlatt").read_text().splitlines()
records = [j for j, line in enumerate(lines) if "cycle" in line]
if not records:
    raise ValueError("No gradlatt record found")
i = records[-1]

def read3(rows):
    a = np.array([
        [float(x.replace("D", "E").replace("d", "e"))
         for x in line.split()]
        for line in rows
    ])
    if a.shape != (3, 3) or not np.isfinite(a).all():
        raise ValueError("Invalid 3x3 gradlatt block")
    return a

H = read3(lines[i + 1:i + 4])
G = read3(lines[i + 4:i + 7])
volume = abs(np.linalg.det(H))
if volume <= 0:
    raise ValueError("Invalid cell volume")
stress_au = G.T @ H / volume
stress_gpa = stress_au * 29421.0265
np.savetxt("stress_gpa.dat", stress_gpa, fmt="%20.10e",
           header="3x3 stress [GPa], tensile positive")
```

A real interface should also verify that the current cell matches the recorded cell, that the calculation finished normally, and that the volume is not pathological. Do not treat the mere existence of the file as proof that the evaluation succeeded.

`--grad` is the single-shot gradient output path; it does not imply that `--md` automatically saves the full stress per frame. The external driver must evaluate step by step and record the results itself. Periodic GFN-FF and this output path should be verified against the actually installed xTB version; the stress support documented for the old xtb-python cannot be taken directly as a capability statement for the command-line GFN-FF.

## 4. LAMMPS

### 4.1 Recommended extraction of the potential pressure tensor

Add after the system and potential definitions:

```lammps
compute Pconf all pressure NULL virial
thermo 1
thermo_style custom step pe vol c_Pconf[1] c_Pconf[2] c_Pconf[3] &
                               c_Pconf[4] c_Pconf[5] c_Pconf[6]
run 0
```

`virial` excludes the kinetic-energy term but still includes the applicable pair, bond, angle, dihedral, improper, kspace, and fix contributions. Check for extra fixes so their constraint/external-force virials do not duplicate what the driver already provides.

The six-component order is `xx yy zz xy xz yz`:

```python
pxx, pyy, pzz, pxy, pxz, pyz = values
Pconf = np.array([[pxx, pxy, pxz],
                  [pxy, pyy, pyz],
                  [pxz, pyz, pzz]])
stress_potential = -Pconf  # then convert according to units
```

The native `pxx pyy pzz pxy pxz pyz` include the kinetic-energy term by default, so a pure calculator must not use them in place of the Pconf above.

### 4.2 Separate trajectory file

When LAMMPS runs its own MD, the potential pressure can be written every 10 steps:

```lammps
fix saveP all ave/time 10 1 10 c_Pconf[1] c_Pconf[2] c_Pconf[3] &
                              c_Pconf[4] c_Pconf[5] c_Pconf[6] &
                              file pressure_tensor.dat
```

`Nrepeat=1` means no multi-sample averaging. This fix is for the actually propagated trajectory; a single-point `run 0` should read the thermo/compute results and not rely on ave/time to produce a single-point record. When the external driver handles MD, LAMMPS `fix npt` must not be enabled for a second integration.

### 4.3 Common extraction pitfalls

- `units metal` reports pressure in bar; `units real` in atm.
- `compute stress/atom` outputs a signed pressure×volume-type per-atom quantity, not a per-atom pressure. For the global interface, prefer compute pressure.
- `press` is a scalar and contains no full shear information; it cannot replace the six-component tensor.
- A general triclinic cell may be rotated internally by LAMMPS into a restricted triclinic cell. The extracted tensor, forces, and positions must be transformed back to the same external frame.
- `thermo_modify triclinic/general yes` applies to the corresponding thermo output; do not assume it automatically rotates the six components of a custom compute.

## 5. Component order, units, and sign checks

| Source | Native quantity | Unified tensile-positive potential stress |
| --- | --- | --- |
| CP2K FORCE_EVAL stress | compressive-positive, already divided by volume | negate and convert units |
| xTB gradlatt | lattice gradient G, no stress given directly | G.T @ H / abs(det(H)) |
| LAMMPS pressure NULL virial | compressive-positive, already divided by volume | rebuild the matrix, negate, and convert units |

Common conversions:

```text
1 bar                  = 1e-4 GPa
1 atm                  = 1.01325e-4 GPa
1 eV/Å³                ≈ 160.2176634 GPa
1 Hartree/Bohr³        ≈ 29421.0265 GPa
```

The LAMMPS six components are `[xx, yy, zz, xy, xz, yz]`; the ASE Voigt order is `[xx, yy, zz, yz, xz, xy]`. Stress shear components are not multiplied by 2; only between engineering shear strain and tensor shear strain is there such a factor distinction. Prefer storing the 3×3 matrix internally and converting only at the interface boundary.

## 6. Responsibilities of the external MD/NPT driver

### 6.1 Removing center-of-mass translation does not remove the kinetic stress

```text
V_COM = sum_i m_i v_i / sum_i m_i
c_i   = v_i - V_COM
sum_i m_i c_i = 0
```

This only eliminates the total momentum; `sum_i m_i c_i ⊗ c_i` is generally still nonzero. Two equal-mass atoms moving at +v and -v have zero total momentum, yet `P_kin,xx = 2 m v²/V > 0`.

```text
P_kin,after = P_kin,before - M/V * (V_COM ⊗ V_COM)
```

Therefore NPT still needs the kinetic pressure of the thermal motion. For a three-dimensional system with no other constraints and fixed total momentum, the thermal degrees of freedom number `3N-3`. With constraints, frozen atoms, or flow, compute the degrees of freedom and thermal velocities according to the actual dynamical model; do not indiscriminately remove driven flow step by step.

### 6.2 Pressure assembly and integration

Assuming masses, velocities, and stress are already in consistent units:

```python
kinetic_tensor = (masses[:, None] * c).T @ c
pressure_tensor = kinetic_tensor / volume - stress_potential
pressure_scalar = np.trace(pressure_tensor) / 3
```

- A kinetic energy obtained from amu and Å/fs cannot be added directly to eV; convert units first.
- Velocities, stress, and volume must be taken at the substep time prescribed by the integrator; do not arbitrarily mix half-step velocities with a configuration from another time.
- When SHAKE/RATTLE and similar are implemented by the driver, the constraint virial of the chosen algorithm must also be added. Subtracting thermal degrees of freedom is not a substitute for the constraint pressure.
- For systems without flow, velocities with the center-of-mass drift removed can be used; with flow/cell motion, distinguish thermal from flow velocities as the algorithm requires, and do not double-subtract affine motion already excluded by the integration variables.
- Pass the latest cell and atomic positions at every evaluation so periodic boundaries, neighbor lists, and long-range terms update correctly.
- Do not replace the periodic virial with a simple outer-product sum of wrapped coordinates and net forces.
- Isotropic pressure control uses trace(P)/3; anisotropic/flexible-cell control requires the full tensor and explicit cell degrees of freedom.
- Do not replace the instantaneous kinetic pressure with a fixed NkBT/V built from the target temperature, unless the chosen algorithm explicitly prescribes it.

### 6.3 Correct pressure is not a complete NPT algorithm

A barostat must be implemented as a complete algorithm, including cell/position/momentum updates, extended variables, thermostat coupling, and the required MTK corrections. In general, an elevated internal pressure should drive expansion (the correct physical direction).

Berendsen may be used for pre-equilibration but does not produce correct NPT volume fluctuations. For production sampling, choose a proper method — MTK, stochastic cell rescaling, etc. — and follow its complete equations. Monte Carlo volume moves can handle temperature effects through the acceptance probability and need not use the instantaneous kinetic pressure explicitly; the pressure-assembly rules of one barostat must not be mechanically applied to all NPT methods.

## 7. Post-implementation validation

1. **Energy–stress consistency**: at fixed fractional coordinates, apply small positive and negative strains to the cell while moving the atoms accordingly, and verify σ_pot by central differences. Test different step sizes, diagonal components, and shear components, and confirm the sign, the volume factor, and the transpose.
2. **Pressure-assembly consistency**: with the same simple LAMMPS potential and the same cell/positions/velocities, compare the external driver's total pressure against LAMMPS's native pressure tensor.
3. **Translation check**: verify that the kinetic pressure is generally nonzero after zeroing the total momentum, and check the analytic difference before and after removing translation.
4. **Frame-rotation check**: for triclinic cells, verify that positions, forces, and stress use the same coordinate frame.
5. **Integration check**: first verify NVE energy drift and step-size convergence, then the thermostat and barostat.
6. **Sampling check**: check average temperature, pressure, and volume and their fluctuations; averages near the target do not prove a correct NPT distribution.
7. **Backend numerical convergence**: for CP2K, check SCF, basis-set, and grid convergence of the stress; for xTB, check the actual version and results after cell updates; for LAMMPS, check consistency of cutoffs, long-range terms, and extra fixes.

## 8. Official documentation and source-code references

- [CP2K FORCE_EVAL: STRESS_TENSOR](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL.html)
- [CP2K potential-stress print settings](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PRINT/STRESS_TENSOR.html)
- [CP2K MD stress output settings](https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/PRINT/STRESS.html)
- [CP2K developer explanation of the difference between the two stress outputs](https://lists.cp2k.org/archives/cp2k-user/2022-August/017335.html)
- [Sign conversion in the ASE CP2K interface](https://docs.ase-lib.org/_modules/ase/calculators/cp2k.html)
- [xTB main program: gradient output invocation](https://github.com/grimme-lab/xtb/blob/main/src/prog/main.F90)
- [xTB Turbomole writer: gradlatt output](https://github.com/grimme-lab/xtb/blob/main/src/io/writer/turbomole.f90)
- [xTB GFN-FF calculator: strain derivatives](https://github.com/grimme-lab/xtb/blob/main/src/gfnff/calculator.f90)
- [LAMMPS compute pressure](https://docs.lammps.org/compute_pressure.html)
- [LAMMPS compute stress/atom](https://docs.lammps.org/compute_stress_atom.html)
- [LAMMPS thermo_style and triclinic-cell output](https://docs.lammps.org/thermo_style.html)
- [LAMMPS fix ave/time](https://docs.lammps.org/fix_ave_time.html)
- [LAMMPS units definitions](https://docs.lammps.org/units.html)
- [LAMMPS NPT/MTK integration notes](https://docs.lammps.org/fix_nh.html)
- [Original stochastic-cell-rescaling paper](https://arxiv.org/abs/2006.09250)

The trunk/main links in this document will drift with upstream updates; record the actual software versions during implementation and rely on those versions' outputs and validation results.
