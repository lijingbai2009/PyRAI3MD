---
title: "1. What is PyRAI3MD"
nav_order: 1
---

PyRAI3MD is a framework for molecular and periodic simulations that mixes quantum chemistry, classical force fields, machine-learning potentials and nonadiabatic dynamics in one input file. It computes energies, gradients, couplings and spectra with ten interchangeable engines, propagates trajectories with surface hopping, optimizes minima, transition states and conical intersections, searches reaction paths, trains neural-network potentials and improves them by adaptive sampling.

## 1.1. Framework and skills

The program is split into a framework and a set of skills. The framework owns everything that is not physics: the input schema, the planning layer that turns a request into a list of tasks, the workflow loops, the logging, the parallel dispatch and the protocols every engine implements. A skill owns one scientific engine: its configuration schema, its input builder, its runner, its output parser and the conversion of its native units into the framework units.

The split is enforced by a one-way import rule. The framework never imports an engine module, and a skill never imports the framework planning or workflow layers; the two meet at the skill manifest, the query module and the SkillConfig schema. This is what lets a calculation run with PySCF today and with OpenMolcas tomorrow by editing one word of the method queue, and it is what allows an engine to be added without touching the workflow code.

The same tree serves three entry points: the classic or YAML command pyrai3md, the agent command pyrai3md_agent for sectioned JSON, and the single-skill command pyrai3md_skill. A child runtime that the framework writes for a parallel or remote task goes through the same pipeline, so a calculation behaves the same whether it runs on the login node or inside a scheduler job.

## 1.2. Relation to PyRAI2MD

PyRAI3MD is the successor of PyRAI2MD. The scientific core is the same: the neural-network potentials, the trajectory surface hopping algorithms and the adaptive-sampling idea were developed and validated in the PyRAI2MD papers listed in [Section 1.4](#14-citation-and-license). What changed is the architecture and the reach of the program.

The engines became skills, so the set of methods is open rather than fixed. The input gained two structured carriers beside the classic text format, and the configuration is layered over a home file. Planning became an explicit step that resolves capabilities, units and partition before a single calculation starts, which is why an unsupported request is reported as a refusal instead of failing halfway through a trajectory. The workflow set grew to optimization, path search, Hessians, thermochemistry, grid search and label extraction. Multiscale work is expressed as a method queue, and periodic systems, constraints and scheduler dispatch are first-class rather than bolt-ons.

Input files written for PyRAI2MD are not expected to run unchanged. [Appendix E](appendix-e-migrating-pyrai2md-inputs.html) is the migration table: it pairs each retired spelling with its replacement and marks the few constructs, such as the hop jobtype, that were removed rather than renamed.

## 1.3. How to read this manual

Chapter 2 lists the features and the engines. Chapter 3 installs the package and sets up engines. Chapter 4 explains the three ways to write an input and the conventions behind them. Chapter 5 is the keyword reference, one section per input section and one per skill, with the default of every keyword in red. Chapters 6 to 12 explain what each workflow does and how the pieces fit together. Chapter 13 walks through complete examples, Chapters 14 and 15 cover testing and extension, and the appendices hold the unit table, the index bases, the keyword status index and the refusal matrix.

In the keyword tables a blue keyword is followed by its default in red and a one-line description. The marks used in those tables and in the prose are: a dagger for a keyword that is accepted but has no effect in v3.0.0, a double dagger for a keyword that honours only one of its values, a section sign for an alias, and an asterisk for a derived quantity that is not read from the input. Defaults are quoted from the configuration model the program loads, so they describe version 3.0.0 and not an older release.

Code appears in grey boxes. Coordinates are in Angstrom, energies in Hartree and gradients in Hartree per Bohr unless a box says otherwise; [Section 4.6](04-input-files-and-conventions.html#46-conventions) states the six conventions a reader needs and [Appendix A](appendix-a-units-and-sign-conventions.html) gives the full table.

## 1.4. Citation and license

PyRAI3MD is distributed under the PyRAI3MD Research License 1.0; the LICENSE file in the repository is the authoritative text. The methods it implements are described in the following papers. Please cite the ones that cover the techniques you used.

Jingbai Li, Patrick Reiser, Benjamin R. Boswell, Andre Eberhard, Noah Z. Burns, Pascal Friederich, and Steven A. Lopez, "Automatic discovery of photoisomerization mechanisms with nanosecond machine learning photodynamics simulations", Chem. Sci. 2021, 12, 5302-5314. DOI:10.1039/D0SC05610C

Jingbai Li, Rachel Stein, Daniel Adrion, Steven A. Lopez, "Machine-learning photodynamics simulations uncover the role of substituent effects on the photochemical formation of cubanes", J. Am. Chem. Soc. 2021, 143, 48, 20166-20175. DOI:10.1021/jacs.1c07725

Jingbai Li, Steven A. Lopez, "Excited-state distortions promote the reactivities and regioselectivities of photochemical 4pi-electrocyclizations of fluorobenzenes", Chem. Eur. J. 2022, 28, e202200651. DOI:10.1002/chem.202200651

Jingbai Li, Steven A. Lopez, "A Look Inside the Black Box of Machine Learning Photodynamics Simulations", Acc. Chem. Res. 2022, 55, 1972-1984. DOI:10.1021/acs.accounts.2c00288

L. Wang, C. Salguero, S. A. Lopez, J. Li, "Machine learning photodynamics uncover blocked non-radiative mechanisms in aggregation-induced emission", Chem 2024, 10, 2295-2310. DOI:10.1016/j.chempr.2024.04.017

Z. Li, F. J. Hernandez, C. Salguero, R. Crespo-Otero, S. A. Lopez, J. Li, "Machine learning photodynamics decode multiple singlet fission channels in pentacene crystal", Nat. Commun. 2025, 16, 1194. DOI:10.1038/s41467-025-56480-y

Z. Li, H. Fu, S. A. Lopez, J. Li, "Machine learning photodynamics reveal intersystem-crossing-driven ladderdiene ring opening", Chem. Sci. 2025, 16, 13031-13041. DOI:10.1039/D4SC07395A

L. Wang, J. Li, "Modeling the Nonradiative Decay of Cyclooctatetrathiophene in Solution and Crystal Phases Using Machine Learning", J. Chem. Theory Comput. 2025, 21, 19, 9300-9308. DOI:10.1021/acs.jctc.5c01136

The programs that PyRAI3MD drives must be cited according to their own instructions, and the force fields and functionals used through CP2K, LAMMPS, xTB or the quantum-chemistry engines carry their own recommended references.
