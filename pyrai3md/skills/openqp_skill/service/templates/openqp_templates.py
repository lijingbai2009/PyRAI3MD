"""
======================================================================
This is a standalone OpenQP calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the OpenQP INI input section templates.
Placeholders are filled by the input builder based on the config
and the compiled workflow.

The export mechanism ([properties]export=true with an empty title)
makes OpenQP dump results into fixed-name text files
(energies, grad_<i>, nacme, dcme, nac_<i>_<j>, socme), which the output
parser reads directly.  A run that solves both spin ladders publishes
each manifold under its own label instead (energies_s, energies_t,
grad_s_<i>, grad_t_<i>, nacme_s, dcme_t, ...), and leaves the
unlabelled names out entirely - they are the channel between a
numerical differentiation parent and its displaced workers.

The orbital continuation mechanism relies on fixed file names:
save_mol=true always stores the orbital json next to the log file
(${OPENQP_PROJECT}.json), and [guess]file2 points to the same file
for nacme (nac_type=dcm, runtype=prop) calculations.

"""

# [input] section: system points to the xyz file written by the skill.
# The functional line is appended only for tddft/mrsf methods.
# input_extra_line carries the spin-orbit switches (ispher=false and
# soc_2e), empty on every other run so an unchanged deck stays unchanged.
INPUT_TEMPLATE = """[input]
system={system}
charge={charge}
method={method}
runtype={runtype}
basis={basis}{functional_line}{input_extra_line}"""

# [guess] section: save_mol is always on so that every calculation
# stores the orbital json as ${OPENQP_PROJECT}.json (fixed name).
# guess file enables automatic orbital continuation with type=auto:
# the json is read when present, otherwise OpenQP falls back to the
# huckel guess. file2 (previous-step orbitals for MO overlap) is only
# appended for nacme calculations (nac_type=dcm, runtype=prop).
GUESS_TEMPLATE = """[guess]
type={guess_type}
file={guess_file}
save_mol=true{file2_line}"""

# [scf] section. multiplicity is forced to 3 for the mrsf method
# (triplet ROHF reference); otherwise the system multiplicity is used.
# scf_extra_line carries the optional converger_type / scal_rel /
# escalation settings, each written only when it differs from OpenQP's
# own default, so a deck that sets none of them is byte-identical.
SCF_TEMPLATE = """[scf]
type={scf_type}
multiplicity={multiplicity}
maxit={scf_maxit}
conv={scf_converge}{scf_extra_line}"""

# [tdhf] section for tddft/mrsf methods. The multiplicity line is
# appended only for mrsf (target singlet states from a triplet
# reference, [tdhf]multiplicity=1).
# ladder_line names each spin ladder's root count for a two-manifold run;
# OpenQP sizes a ladder from nstate_s / nstate_t and falls back on [tdhf]
# nstate when one is unset, so nstate stays the configured ladder's count
# rather than the total across both.
TDHF_TEMPLATE = """[tdhf]
type={td_type}
nstate={nstate}
maxit={td_maxit}
conv={td_converge}{multiplicity_line}{ladder_line}"""

# [properties] section. export=true with an empty title gives fixed
# export file names that the parser reads. selector_line lists the states
# to compute gradients for: `grad` takes the 1-based excited-state indices
# (hf ground state = 0) of a single-block run, while a two-manifold run must
# use `grad_s` / `grad_t`, one list per spin ladder, because a bare index
# cannot say which ladder it counts. The soc line turns on spin-orbit
# coupling; the nac line selects nacme (dcm) or nac (vector) on the
# runtype=prop and runtype=data paths, and runtype=nac ignores it.
PROPERTIES_TEMPLATE = """[properties]
export=true
title=
{selector_lines}"""

# [nac] section: NAC vector calculation, numerical differentiation with time
# step dt and displacement dx; bp=true additionally computes the
# branching-plane gradients. `states` carries 1-based state pairs: one pair
# on runtype=nac, where OpenQP answers a single pair, and every declared pair
# (`1 2,2 3`) on the runtype=data vector path, which writes one export file
# per pair.
#
# Neither this section nor [hess] carries `nproc`: how many displaced
# structures run at once is decided by how the job was launched (the MPI
# ranks under mpirun, the OpenMP thread count per worker), not by the deck.
# The skill therefore exposes one knob per question: `openqp_nproc` for
# the rank count, `thread` / `openqp_thread` for OMP_NUM_THREADS. Neither
# reaches this file.
NAC_TEMPLATE = """[nac]
dt={dt}
dx={dx}
states={states}
bp={bp}"""

# [optimize] section. istate follows the grad index convention:
# hf ground state = 0, otherwise target state k -> k+1.
OPTIMIZE_TEMPLATE = """[optimize]
optimizer={optimizer}
maxit={maxit}
rmsd_grad={rmsd_grad}
max_grad={max_grad}
istate={istate}"""

# [hess] section. state follows the same index convention as istate.
HESS_TEMPLATE = """[hess]
state={state}
dx={dx}"""
