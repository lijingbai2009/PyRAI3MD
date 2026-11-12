---
title: "9. Constraints"
nav_order: 9
---

The constraint block adds potentials and geometric conditions to a run without touching the electronic structure calculation. It is closed by default: an empty block builds no plan and changes no number. The block is written under molecule.constraints in YAML, as a top-level constraint or system_constraint section, or as the classic &constraint section.

## 9.1. Constraint families

Five kinds of action are organized together and implemented separately. A bias adds a harmonic or flat-bottomed potential on a bond, an angle, a dihedral or a named collective variable; it never moves a geometry. A wall adds a confining potential around a sphere, an ellipsoid, a box or a triclinic cell, and acts on each atom or on a group centre. A fix removes selected Cartesian components from the motion and restores the anchors it captured. SHAKE projects the geometry onto a manifold of fixed bond lengths, angles or dihedrals and carries no force constant; its RATTLE option projects the velocities as well. Metadynamics accumulates Gaussian hills along a collective variable or an aligned RMSD, so the surface a trajectory sees depends on where it has been.

The families act at two points of a step, and that placement is physical, not an implementation detail. Hard projection and freeze happen between the position update and the electronic evaluation, so the reported energy belongs to the projected geometry. A bias, a wall and a metadynamics hill are additive, so they are applied in the calculator as the result is finalized: later would miss the velocity kicks.

A collective variable is a named coordinate, not a potential. Three kinds exist, bond, angle and dihedral. A CV may be referenced by several biases and by a metadynamics entry, and the same atoms may be named inline instead.

## 9.2. Writing the block

The key tables, their defaults and their one-line descriptions are in [Section 5.13](05-13-constraint.html). The block has eight subsections: units, groups, cvs, biases, walls, fixes, constraints, metadynamics. units applies to the whole block and is converted once when the plan is built: a force constant divides by the square of the length factor. Every atom number in the block is 1-based in all carriers. The example declares one group, one variable, one ramped bias, one wall, one fix, one RATTLE bond condition and one metadynamics entry.

```yaml
molecule:
  constraints:
    units:
      length: angstrom
      angle: degree
      energy: hartree
    groups:
      - name: solute
        atoms: 1-6
    cvs:
      - name: oh
        kind: bond
        atoms: [1, 2]
    biases:
      - id: hold_oh
        cv: oh
        target: 0.96
        force_constant: 0.02
        force_constant_start: 0.02
        force_constant_end: 0.2
        force_constant_delay: 100
    walls:
      - id: cage
        group: solute
        geometry: sphere
        center: [0.0, 0.0, 0.0]
        radius: 8.0
        force_constant: 0.1
    fixes:
      - id: anchor
        atoms: [1]
        direction: xyz
    constraints:
      - id: oh_bonds
        kind: bond
        bond_type: O-H
        tolerance: 1.0e-8
        velocity_constraint: rattle
    metadynamics:
      - id: explore
        cvs: [oh]
        height: 0.0005
        width: [0.05]
        pace: 100
```

The classic spelling carries the same actions as one flat list of keyword lines. Object lines declare what the action acts on, parameter lines follow in the same order, and a semicolon separates objects on one line while a comma separates fields of one object.

```text
&constraint
  unit_length angstrom
  unit_energy hartree
  group_name solute
  group_atoms 1-6
  cv_bond 1 2
  cv_name oh
  bias_cv oh
  bias_cv_ref 0.96
  bias_cv_fc 0.02
  wall_type sphere
  wall_sphere_radius 8.0
  wall_group solute
  wall_fc 0.1
  fix_atoms 1
  shake_bond 1 2
  shake_vel rattle
  meta_cv oh
  meta_height 0.0005
  meta_width 0.05
  meta_pace 100
```

## 9.3. Where constraints act

Support is a property of the workflow: a condition needs a step, and a schedule needs a clock. The table reads yes where the workflow honours the family and static where the potential is present but nothing propagates it.

| Jobtype | fix | wall | bias | metadynamics | SHAKE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| md, nve and nvt | yes | yes | yes | yes | yes |
| md, npt | yes | yes | yes | yes | yes |
| optimize | yes | static | static | not claimed | yes |
| adaptive | inherits | inherits | inherits | inherits | inherits |
| single point | no | static | static | no | no |
| hessian, search, read | no | no | no | no | no |

An optimization honours a fix as a mask on the step and a strict condition as a projected step, and judges convergence on the tangential gradient. Adaptive sampling runs its trajectories through the MD workflow, so they carry the MD set, while its labels come from the base electronic result and a wall or bias energy never enters a training set. MECI, MECP, saddle and band searches do not claim the block, and a constrained surface hop is not supported because a hop rescales velocities inside the feasible space.

## 9.4. Refused combinations

A declaration that cannot be served is refused at input rather than approximated at run time.

A soft wall in a periodic system is refused. Its law is a spring toward one point, so there is no shell to check against a cell face, and a wrapped coordinate jumps its offset by a whole lattice vector on every crossing. A bias region of the inside or outside kind is refused, and so is a one-sided upper or lower region on a dihedral: a one-sided region of a periodic coordinate needs a declared origin and direction. A collective variable whose atoms are not in one connected component is refused when the run is periodic, because there is no image rule to derive for it.

A ramp interpolates linearly between declared endpoints. _scale multiplies the declared value and cannot be combined with _start and _end. _delay and _stop bound a ramp and are refused when no endpoint exists, and endpoints must have the shape of the value they replace. A quantity that has to stay positive, such as a force constant or a radius, may not ramp to zero or below; the endpoint is refused at input and the same check runs at every step. A wall power below 2 is refused because the potential would not be differentiable at the wall. bond_type perceives the bond graph and expands bond pairs only, so it is refused for an angle or a dihedral condition.

The flat restraint keywords of the previous version, among them freeze, cbond, cavity, shape, factor and compress, are refused by name with the replacement spelling, and [Appendix D](appendix-d-refusal-and-limitation-matrix.html) collects these refusals beside their modern spellings.
