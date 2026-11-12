---
title: "7. Multiscale calculations"
nav_order: 7
---

A multiscale run takes its energy and gradient from more than one calculation. The system is split into regions, each is queued with its own method, and the results are assembled by the coefficients of the scheme before any workflow sees them. Both schemes are selected in the control section listed in [Section 5.1](05-01-control.html) and take their partition from the molecule section.

## 7.1. Additive QM/MM

Sets multiscale_type to qmmm with exactly two theory entries, the quantum method first and the classical provider last. molecule.highlevel names the quantum atoms as a 1-based index list, and the union of the regions must be a strict subset. Three entries always select ONIOM3, so qmmm with three entries and qmmm with a reference queue are both refused.

The scheme is additive: E = E_QM + E_classic. The classical provider returns one whole-system evaluation U_main, and the composite subtracts the in-cell point-charge sum U_gas and the QM-MM electrostatics qmmm_el = sum over quantum atoms of q_i V_i, so E_classic = U_main - U_gas - qmmm_el. The quantum rows of the gradient also carry the classical rows; every other row is classical.

With embedding true the plan is qmmm_field, then QM, then qmmm_energy, and the quantum calculation is embedded in the classical field at its nuclei, which the Molcas ESPF deck consumes. The field comes from an A/B Coulomb probe whose two legs must converge to the same reciprocal sum, and its values are converted to atomic units at the handoff. read_charge is not read for QM/MM: the classical stage supplies the field at every step, so both values are cleared without a warning. The classical provider owns the cell; a quantum region is cut out as a cluster and receives none.

q_qm holds the zero-order charges of the quantum atoms in highlevel order, one finite value per atom, and a list of any other length is refused; the key is refused under several centres. A backend that returns computed charges overrides q_qm at run time. qm_contains_field_gradient states who carries the field force: true, the default, means the quantum gradient already contains it; false means the composite adds the removed term back onto the quantum rows.

The point-charge interaction inside the cell is removed analytically, and the non-Coulomb part of the quantum-quantum interaction is removed by cloning the quantum LAMMPS types on the classical side, which leaves the QM-MM and MM-MM terms untouched. Supported jobtypes are energy, gradient, optimize including optimize_cell, md in the nve, nvt and npt ensembles, surface_hopping, adaptive sampling and the constraint block. A Hessian is numerical only, with hess_driver main or the num_hess and num_freq tokens; an analytic matrix is refused.

```yaml
control:
  title: qmmm_water_box
  jobtype: energy gradient
  theory: [molcas, lammps]
  multiscale_type: qmmm
 
molecule:
  charges: 0
  multiplicity: 1
  ci: "2 1"
  spin: "0 1"
  coupling: "1 2"
  highlevel: [1, 2, 3]
  q_qm: [-0.83, 0.415, 0.415]
  embedding: true
  pbc: [1, 1, 1]
  cell: [8.399091, 8.399091, 8.399091, 90.0, 90.0, 90.0]
 
molcas:
  method: caspt2
  basis: ano-l-vdzp
 
lammps:
  method: ff
  lammps_project: water_box
```

## 7.2. ONIOM2 and ONIOM3

Reads the theory queue as a stack, written high to low. Two entries select ONIOM2, spelled by multiscale_type oniom while an empty value keeps the legacy meaning, and three entries always select ONIOM3. highlevel and midlevel are disjoint 1-based lists, and the working regions are derived: region_h is highlevel, region_m is highlevel union midlevel, and region_l is the whole system.

The two-layer energy is E_L(real) plus, per centre, E_H(model) minus E_L(model), the pattern +1, +1, -1. The three-layer energy adds the middle rung E_M(I) minus E_L(I), giving +1, +1, -1, +1, -1. Each term scatters its gradient to its own atoms before the coefficient applies.

embedding true computes the environment charges. High-level charges come only from theory[0]; mid-level charges come only from theory[1] and only on the disjoint middle shell; the low level and the corrections never update charges. read_charge true loads {title}.charge instead, whose first line declares point angstrom external, point angstrom atom or field au, one kind per file.

A charged centre of a three-layer run keeps its net charge in the high region: the middle shell inherits the centre charge one to one, so it has to be neutral and supplies the field only. A static charge file under several ONIOM2 centres is refused, as is a two-layer run with a classical outer method and embedding true, which points to qmmm.

## 7.3. Multi-centre plans

Several centres are written as one region per centre, with a semicolon between centres in charges, multiplicity, ci, spin and coupling and a comma between coupling pairs. Each centre keeps its own state ladder, and centre_groups is refused with a partition. The composite section sets execution, not physics: executor is serial or thread and max_workers sizes the pool.

sc_charge true needs two centres and embedding, and repeats probe and energy-only charge rounds until the largest per-centre move is under sc_charge_threshold or sc_charge_maxiter rounds have run; sc_charge_damping weights the fresh fit.

A partitioned run writes the global md files and, beside them, md.1.xyz, md.2.xyz and the rest, with a 1-based centre index. The optimizer boundary is narrower than the electronic one: it reads the global state axis and has no multi-centre higher-derivative contract, so a multi-centre optimize with a Hessian is not supported, while the framework numerical batch still covers a multi-centre saddle.

## 7.4. Embedding charges and self-consistency

point_charges is a whole-system array of [q, x, y, z] rows, one per atom in framework order, with the charge in elementary units and the position in Angstrom. Row i is atom i, and a length that disagrees with the atom count is refused.

external_charges names a legacy charge file instead, and is a standalone form: no highlevel or midlevel region and no periodic cell, because the file cannot be replicated across a boundary. It excludes read_charge. The two differ in time: read_charge loads a static field once, while embedding recomputes the charges at every step.

Machine learning meets the same distinction at one rule. A model learns the map from a geometry to an energy, so the charge that enters a label must be fixed and declared. A static field meets the rule and the criterion judges a run trained under it; a computed field is read as not judged. The adaptive reference queue labels from the layer covering the neural-network surface.

## 7.5. Limitations

The table lists the refused and unimplemented combinations with the reason for each.

| Combination | Status | Why |
| :--- | :--- | :--- |
| qmmm with three theory entries | Refused | three entries always select ONIOM3 |
| qmmm with a reference queue | Refused | both entries are used at every geometry |
| q_qm under several centres | Refused | each centre is fed by its own fit |
| sc_charge with one centre or no embedding | Refused | the iteration exists for the other centres field |
| ONIOM2 with a classical outer method and embedding | Refused | use qmmm, whose field channel replaces it |
| Multi-centre ONIOM2 with read_charge | Refused | one static file has no per-centre binding |
| Analytic Hessian on a multiscale run | Refused | one geometry has one composite curvature |
| Per-term remote execution of QM/MM | Refused | the field handoff is local to the evaluation host |
| Multi-centre optimize with a Hessian | Not supported | no multi-centre higher-derivative contract |
| Link atoms, overlapping centres, mixed providers | Not implemented | a bond across a partition is not capped |
| Stress of a QM/MM box | Approximate | the cell response of qmmm_el and U_gas stays at point-charge level |
