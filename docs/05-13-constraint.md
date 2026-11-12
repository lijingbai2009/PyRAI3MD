---
title: "5.13. CONSTRAINT"
nav_order: 13
parent: "5. Keyword sections"
---

The constraint block adds potential terms, hard constraints and collective-variable bias to a dynamics or optimization run. In YAML it is written under molecule.constraints or as a top-level constraint section; in classic input it is the &CONSTRAINT section. Every atom number in this block is 1-based in all three carriers and is converted once when the state is built.

The block has eight subsections. units fixes the units of the lengths, angles and energies that the block reads. groups names atom selections that other subsections refer to by name. cvs defines collective variables. biases adds harmonic or flat-bottomed potentials on bonds, angles, dihedrals or a named collective variable. walls adds confining potentials. fixes holds atoms or directions. constraints adds hard bond, angle and dihedral constraints solved during integration. metadynamics adds history-dependent bias. notes is free text carried with the block.

A typical block defines a group and then uses it. The example below restrains two bonds and walls the whole solute into a sphere.

```yaml
molecule:
  constraints:
    units:
      length: angstrom
      energy: hartree
    groups:
      - name: solute
        atoms: 1-6
    biases:
      - id: b1
        atoms: [1, 2]
        target: 1.5
        force_constant: 0.1
    walls:
      - id: w1
        group: solute
        geometry: sphere
        center: [0.0, 0.0, 0.0]
        radius: 8.0
        law: soft
        force_constant: 1.0
```

## units

**&CONSTRAINT units**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| length | angstrom | Sets the unit of every length in the block: angstrom or bohr. |
| angle | degree | Sets the unit of every angle in the block: degree or radian. |
| energy | hartree | Sets the unit of every energy-like value in the block: hartree or kjmol. |

## groups

Each entry defines one named selection. The selector keys atoms, element and group are mutually exclusive: an entry names atoms by index or range, by element symbol, or by an already defined group.

**&CONSTRAINT groups**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| atoms | [] | Selects atoms by 1-based index or index range, for example 1-3 7. |
| element | [] | Selects atoms by element symbol. |
| group | '' | Selects the atoms of another named group. |
| name | '' | Sets the name other subsections use to refer to this group. |

## cvs

A collective variable is a named geometric coordinate used by biases and metadynamics. Three kinds are defined by the atoms list: bond (two atoms), angle (three atoms) and dihedral (four atoms).

**&CONSTRAINT cvs**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| name | '' | Sets the name that biases and metadynamics use for this variable. |
| kind | '' | Sets the variable kind: bond, angle or dihedral. |
| atoms | [] | Sets the 1-based atoms that define the variable; two, three or four atoms according to the kind. |

## biases

A bias adds a harmonic or flat-bottomed potential on a bond, angle, dihedral or named collective variable. Either atoms or cv must be given, not both. The region selects where the potential acts: all, inside a range, or outside a range; the inside and outside spellings are refused and the range is given by two bounds.

**&CONSTRAINT biases**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| atoms | [] | Sets the 1-based atoms of the biased bond, angle or dihedral. |
| id | '' | Sets an identifier used in logs and in ramp schedules. |
| cv | '' | Names a collective variable defined in cvs instead of giving atoms. |
| target | None | Sets the target value of the biased coordinate, in the length or angle unit of the block. |
| force_constant | None | Sets the harmonic force constant; it must be positive. |
| region | all | Selects where the bias acts: all, or a range of the coordinate. |

## walls

A wall confines atoms inside or outside a geometric region. The geometry key chooses the region and therefore which geometry keys are read: sphere reads radius, ellipsoid reads semiaxes, box reads lengths, and triclinic reads cell. A soft wall needs force constant and power and cannot be used in a periodic system; a hard wall reflects.

**&CONSTRAINT walls**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| atoms | [] | Selects the confined atoms by 1-based index or range. |
| element | [] | Selects the confined atoms by element symbol. |
| group | '' | Selects the confined atoms by the name of a defined group. |
| id | '' | Sets an identifier used in logs and in ramp schedules. |
| geometry | sphere | Sets the wall shape: sphere, ellipsoid, box or triclinic. |
| center | [] | Sets the centre of the wall, in the length unit of the block. |
| ref_atoms | [] | Selects atoms whose centre of mass is used instead of an explicit centre. |
| molecules | [] | Sets how many molecules each group member contains, for group-based walls on molecular systems. |
| law | hard | Sets the wall law: hard reflection or a soft potential. |
| radius | None | Sets the radius of a spherical wall. |
| semiaxes | [] | Sets the three semiaxes of an ellipsoidal wall. |
| lengths | [] | Sets the three box lengths of a box wall. |
| cell | [] | Sets the six cell parameters of a triclinic wall. |
| power | 2 | Sets the exponent of the soft wall potential; it must be at least 2. |
| force_constant | None | Sets the strength of the soft wall potential. |
| target | atom | Selects whether the wall acts on atoms or on the centre of a group: atom or center. |

## fixes

A fix removes selected coordinates from the dynamics or from an optimization step. The direction key uses any combination of x, y and z.

**&CONSTRAINT fixes**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| atoms | [] | Selects the fixed atoms by 1-based index or range. |
| element | [] | Selects the fixed atoms by element symbol. |
| group | '' | Selects the fixed atoms by the name of a defined group. |
| id | '' | Sets an identifier used in logs. |
| direction | xyz | Sets the frozen Cartesian directions, for example xz. |

## constraints

A hard constraint keeps bond lengths, angles or dihedrals at a target during integration. SHAKE is applied after each velocity update and can be paired with a velocity constraint, or the positions can be projected without touching velocities. A bond_type entry expands to every bond of that type found in the topology.

**&CONSTRAINT constraints**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| id | '' | Sets an identifier used in logs. |
| kind | bond | Sets the constrained coordinate: bond, angle or dihedral. |
| bonds | [] | Sets the 1-based atoms of each constrained coordinate. |
| bond_type | '' | Expands to every bond of the given element pair. |
| target | None | Sets the constrained value; when absent the starting value is kept. |
| tolerance | 1e-08 | Sets the convergence tolerance of the constraint solver. |
| max_iterations | 200 | Sets the maximum number of solver iterations per step. |
| velocity_constraint | none | Selects whether velocities are also constrained: none or rattle. |

## metadynamics

Metadynamics accumulates Gaussian bias along one or more collective variables. Each entry needs a height and a width per variable, or exactly one width for an RMSD variable. A well-tempered run sets bias_factor greater than one together with a positive temperature.

**&CONSTRAINT metadynamics**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| atoms | [] | Sets the 1-based reference atoms of an RMSD variable. |
| id | '' | Sets an identifier used in logs. |
| kind | gaussian | Sets the bias construction: gaussian or rmsd. |
| cvs | [] | Names the collective variables the bias is deposited along. |
| height | None | Sets the Gaussian height in the energy unit of the block. |
| width | [] | Sets the Gaussian width, one value per variable. |
| pace | 100 | Sets the number of steps between two Gaussian depositions. |
| reference | [] | Sets the reference structure used by an RMSD variable. |
| weight | uniform | Sets the weighting of atoms in an RMSD variable: uniform or mass. |
| bias_factor | 0 | Sets the well-tempered bias factor; 0 disables well-tempered metadynamics. |
| temperature | 0 | Sets the temperature used by the well-tempered update, in kelvin. |

## Ramp schedules

Any numeric parameter of the block can be ramped instead of held fixed. A schedule is written as a small mapping with a suffix on the parameter name: _scale multiplies the base value, _start and _end give the ramp endpoints, _delay postpones the start, and _stop ends the ramp. _scale cannot be combined with _start and _end; _delay and _stop require the endpoints. Endpoints take the shape of the base parameter, and a positive parameter may not ramp to a non-positive value.

```yaml
biases:
  - id: b1
    atoms: [1, 2]
    target: 3.0
    force_constant: 0.1
    force_constant_start: 0.1
    force_constant_end: 1.0
    force_constant_delay: 100
    force_constant_stop: 1000
```

## Legacy constraint spellings

The classic input also accepts the flat constraint tokens of PyRAI2MD. Each token maps onto one subsection entry. The table lists the tokens; the structured spelling above is the recommended one because it can name entries and carry schedules.

**&CONSTRAINT classic tokens**

| Keyword | Default | Description |
| :--- | :--- | :--- |
| group_name | → | groups: name |
| cv_bond | → | cvs: kind bond |
| cv_angle | → | cvs: kind angle |
| cv_dihedral | → | cvs: kind dihedral |
| shake_bond | → | constraints: kind bond |
| shake_bond_type | → | constraints: bond_type |
| shake_angle | → | constraints: kind angle |
| shake_dihedral | → | constraints: kind dihedral |
| bias_bond | → | biases: atoms (2) |
| bias_angle | → | biases: atoms (3) |
| bias_dihedral | → | biases: atoms (4) |
| bias_cv | → | biases: cv |
| wall_sphere_radius | → | walls: geometry sphere |
| wall_ellipsoid_semiaxes | → | walls: geometry ellipsoid |
| wall_box_lengths | → | walls: geometry box |
| wall_triclinic_cell | → | walls: geometry triclinic |
| fix_atoms | → | fixes: atoms |
| fix_element | → | fixes: element |
| fix_group | → | fixes: group |
| meta_cv | → | metadynamics: cvs |
| meta_rmsd | → | metadynamics: kind rmsd |

The retired structured spellings are refused with an error that names the replacement. [Appendix D](appendix-d-refusal-and-limitation-matrix.html) lists the refusal together with the constraint combinations that validation rejects, such as a soft wall in a periodic system, a dihedral bias with an inside or outside region, and a ramp that drives a positive parameter to zero.
