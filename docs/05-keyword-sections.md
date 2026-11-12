---
title: "5. Keyword sections"
nav_order: 5
has_children: true
---

This chapter is the reference for every input keyword. Each section is introduced by a short paragraph, followed by a table of keywords with their defaults and one-line descriptions, and then by a paragraph for each keyword. The tables are generated from the configuration model that the program itself loads, so the default column shows exactly the value a run uses when the keyword is absent.

In the tables the keyword is printed in blue and its default in red. The classic text spelling of a section is given in the table header, for example &amp;CONTROL or &amp;MD. In YAML and agent JSON the same section is a mapping with the lower-case name, for example control: or md:. The section names SYSTEM and DYNAMICS are also accepted as &amp;MOLECULE and &amp;MD; MOLECULE is the historical name and remains the spelling used by most examples.

Four marks qualify a keyword. A dagger marks a keyword that is accepted but has no effect in v3.0.0: it is kept so that older input files still parse. A double dagger marks a keyword that accepts several values but honours only one of them in this version. A section sign marks an alias, and the description names the modern keyword. An asterisk marks a quantity that is derived rather than read; derived quantities are listed in [Appendix B](appendix-b-index-bases-and-derived-quantities.html). Retired keywords are not tabulated: they are refused at validation with a message that names their replacement, and [Appendix E](appendix-e-migrating-pyrai2md-inputs.html) lists them together with the migration from PyRAI2MD inputs.

The sections are read in any order. CONTROL and the molecular specification are the only parts every run needs; the workflow sections are read only by the workflow that the jobtype selects, and a skill section is read only when that skill is in the theory or reference queue. A key written in the wrong section is not silently moved: the owning model does not see it and validation reports the unknown name.

[Section 5.1](05-01-control.html) to [Section 5.12](05-12-environment-and-slurm.html) cover the framework. [Section 5.13](05-13-constraint.html) covers the structured constraint block, [Section 5.14](05-14-common-keywords-in-skill-sections.html) covers the keys that all skills share, and [Section 5.15](05-15-pyscf.html) to [Section 5.24](05-24-nn_pes.html) cover the ten calculation skills.

## Sections

- [5.1. CONTROL](05-01-control.html)
- [5.2. SYSTEM (MOLECULE)](05-02-system-molecule.html)
- [5.3. DYNAMICS](05-03-dynamics.html)
- [5.4. OPTIMIZE](05-04-optimize.html)
- [5.5. HESSIAN](05-05-hessian.html)
- [5.6. SEARCH](05-06-search.html)
- [5.7. ADAPTIVE](05-07-adaptive.html)
- [5.8. TRAIN](05-08-train.html)
- [5.9. PREDICTION](05-09-prediction.html)
- [5.10. READ](05-10-read.html)
- [5.11. COMPOSITE](05-11-composite.html)
- [5.12. ENVIRONMENT and SLURM](05-12-environment-and-slurm.html)
- [5.13. CONSTRAINT](05-13-constraint.html)
- [5.14. Common keywords in skill sections](05-14-common-keywords-in-skill-sections.html)
- [5.15. PYSCF](05-15-pyscf.html)
- [5.16. MOLCAS](05-16-molcas.html)
- [5.17. BAGEL](05-17-bagel.html)
- [5.18. ORCA](05-18-orca.html)
- [5.19. OPENQP](05-19-openqp.html)
- [5.20. XTB](05-20-xtb.html)
- [5.21. CP2K](05-21-cp2k.html)
- [5.22. LAMMPS](05-22-lammps.html)
- [5.23. QDM](05-23-qdm.html)
- [5.24. NN_PES](05-24-nn_pes.html)
