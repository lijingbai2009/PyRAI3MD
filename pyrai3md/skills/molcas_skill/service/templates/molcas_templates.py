"""
======================================================================
This is a standalone OpenMolcas calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Jul 29 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module defines the Molcas input file templates for each module.
All templates follow the rules recorded in
dev/skills/MOLCAS_SKILL_DEVELOPMENT.md section 5.8 (the example-input note
this skill was aligned against).

"""


# GATEWAY module template: defines molecular geometry and basis set.
# Coordinates are referenced via $MOLCAS_PROJECT.xyz variable and
# must appear on the first line right after &GATEWAY, otherwise
# Molcas cannot locate the coordinate file.
# RICD and Group=c1 are required for all calculations (per
# dev/skills/MOLCAS_SKILL_DEVELOPMENT.md section 5.8).
GATEWAY_TEMPLATE = """&GATEWAY
 coord=$MOLCAS_PROJECT.xyz
 Title
  {title}
 basis={basis}
 Group=c1
 RICD
"""

# GATEWAY SOC template: adds AMFI and Angmom keywords for relativistic effects.
GATEWAY_SOC_TEMPLATE = """&GATEWAY
 coord=$MOLCAS_PROJECT.xyz
 Title
  {title}
 basis={basis}
 Group=c1
 RICD
 AMFI
 Angmom
  0 0 0
"""

# XField block for embedding background point charges
XFIELD_TEMPLATE = """XField
{ncharges} Angstrom
{charge_block}
"""

# SEWARD module template: integral evaluation with analytic flag.
# doanalytic is required for all calculations (per
# dev/skills/MOLCAS_SKILL_DEVELOPMENT.md section 5.8).
SEWARD_TEMPLATE = """&SEWARD
doanalytic
"""

# SEWARD SOC template: adds relativistic keywords on top of doanalytic.
SEWARD_SOC_TEMPLATE = """&SEWARD
rela
R02O
relint
expert
doanalytic
"""

# RASSCF module template: multiconfigurational SCF.
# Format follows dev/skills/MOLCAS_SKILL_DEVELOPMENT.md section 5.8:
# Spin, NactEl, Inactive, Ras2, ITERATIONS, CIRoot.
# CIRoot uses three-number format: n n 1 (equal state averaging, no symmetry).
# ITERATIONS={maxit_scf},100: first number is the max CI iterations
# (maxit_scf, default 200), second is the super-CI orbital
# optimization limit (fixed 100).
RASSCF_TEMPLATE = """&RASSCF
 Spin={spin}
 NactEl={nact_el} {ras1} {ras3}
 Inactive={inactive}
 Ras2={ras2}
 ITERATIONS={maxit_scf},100
 CIRoot={ci_root} {ci_root} 1
"""

# RASSCF with fileorb: for normal usage where an external orbital
# guess is provided. The user guess is locked as title.StrOrb and
# copied to $WorkDir/$Project.RasOrb at runtime (dynamics COPY line)
# or staged in calcdir (manage_orbitals); RASSCF always reads the
# RasOrb-suffixed file.
RASSCF_FILEORB_TEMPLATE = """&RASSCF
 Spin={spin}
 NactEl={nact_el} {ras1} {ras3}
 Inactive={inactive}
 Ras2={ras2}
 ITERATIONS={maxit_scf},100
 CIRoot={ci_root} {ci_root} 1
 fileorb=$MOLCAS_PROJECT.RasOrb
"""

# RASSCF ALTEr block: interchanges MO pairs in INPORB/JOBOLD so the
# user-selected orbitals land in the Ras2 window (MOLCAS manual:
# ALTEr requires the number of pairs, then for each pair the symmetry
# species and the indices of the two permuting MOs). The molcas skill
# does not use symmetry, so the species is always 1. Appended to the
# RASSCF section when the user selects active orbitals.
RASSCF_ALTER_LINE = " ALTEr\n {n_alter}\n{alter_block}\n"

# Rlxroot keyword line: selects the CI root relaxed by SLAPAF in
# geometry optimization (1-based within the RASSCF job).
RASSCF_RLXROOT_LINE = "Rlxroot={rlxroot}\n"

# Orbital continuation line: RASSCF reads the RasOrb written by the
# previous FOREACH iteration inside the dynamics loop, or the user
# guess copied to RasOrb before the first iteration. All orbital
# files in the input carry the RasOrb suffix; the StrOrb suffix is
# reserved for locking the user-provided guess file (title.StrOrb).
RASSCF_FILEORB_LINE = "fileorb=$MOLCAS_PROJECT.RasOrb\n"

# MDRLXR keyword line: CI root (1-based) whose gradient drives the
# molecular dynamics propagation (DYNAMIX).
RASSCF_MDRLXR_LINE = "MDRLXR={mdrlxr}\n"

# SCF module template: Hartree-Fock calculation
SCF_TEMPLATE = """&SCF
charge={charge}
spin={spin}
"""

# ALASKA module template: analytical gradients (ground state)
ALASKA_TEMPLATE = """&ALASKA
"""

# ALASKA with a specific root for excited state gradient (1-based ROOT)
ALASKA_ROOT_TEMPLATE = """&ALASKA
ROOT={root}
"""

# ALASKA NAC template: nonadiabatic coupling between specific states (1-based)
ALASKA_NAC_TEMPLATE = """&ALASKA
NAC={state_i} {state_j}
"""

# COPY template: expose the CASPT2 JobMix as JOB001 so the RASSI
# Nrof JobIph block reads the multistate CASPT2 states instead of
# the underlying RASSCF JobIph (CASPT2 absorption workflow).
COPY_JOBMIX_TEMPLATE = ">>COPY $Project.JobMix JOB001\n"

# RASSI SOC computation template: with Nrof JobIph format
RASSI_SOC_TEMPLATE = """&RASSI
Nrof JobIph={njobs} {states_line}
SpinOrbit
EJob
SOCOupling=0
"""

# RASSI absorption template (CASSCF-level): transition properties with
# dipole print disabled. MEES prints the spin-free multipole matrices
# used to extract transition dipole moments.
RASSI_ABSORPTION_TEMPLATE = """&RASSI
DIPRint=0
MEES
"""

# RASSI absorption template (CASPT2-level): reads all {nstates} states
# back into RASSI (Nrof JobIph with one job) after multistate CASPT2,
# then computes transition properties with dipole print disabled.
# MEES prints the spin-free multipole matrices used to extract
# transition dipole moments.
RASSI_CASPT2_ABSORPTION_TEMPLATE = """&RASSI
Nrof JobIph=1 {nstates};{state_list}
EJob
DIPRint=0
MEES
"""

# CASPT2 module template: standard single-state CASPT2
# prop enables CASPT2-level property (dipole) output.
CASPT2_TEMPLATE = """&CASPT2
prop
shift
  {imag_shift}
"""

# CASPT2 multistate template: xmultistate all (xms switch, default)
# prop enables CASPT2-level property (dipole) output.
CASPT2_MULTISTATE_TEMPLATE = """&CASPT2
xmultistate
all
prop
shift
  {imag_shift}
"""

# SLAPAF module template: geometry optimization.
# Iterations is the max number of relaxation steps; when the input
# exports MOLCAS_MAXITER, SLAPAF uses it as its own default limit.
SLAPAF_TEMPLATE = """&SLAPAF
Iterations={maxit}
thrshld=0.0 5.0D-4
"""

# Optimization loop wrappers: MOLCAS_MAXITER (opt_maxit) bounds the
# Do while macro-loop that re-runs SEWARD/RASSCF/SLAPAF until SLAPAF
# converges; GATEWAY stays outside so the updated coordinates written
# by SLAPAF are re-read on every iteration.
EXPORT_MAXITER_TEMPLATE = ">> EXPORT MOLCAS_MAXITER={maxit}\n"
DO_WHILE_TEMPLATE = ">> Do while\n"
END_DO_TEMPLATE = ">> End Do\n"

# GATEWAY constraint block for MECI/MECP optimization. {states} is
# ' n m' (1-based CI roots) for a same-spin crossing (MECI) or empty
# for a cross-spin crossing (MECP), where the crossing pair is fixed
# by the Rlxroot keywords of the two RASSCF jobs.
GATEWAY_CONSTRAINT_TEMPLATE = """Constraints
   a = Ediff{states}
  Value
   a = 0.000
End of Constraints
"""

# DYNAMIX module template: molecular dynamics. velver propagates
# with velocity Verlet; dt is the timestep in au (dynamics.size);
# velo=1 reads the initial velocities from $Project.velocity.xyz;
# ther selects the thermostat (0 = NVE, 2 = Nose-Hoover NVT).
DYNAMIX_TEMPLATE = """&Dynamix
velver
dt = {size}
velo = 1
ther = {ther}
temp = {temperature}
"""

# SURFACEHOP module template: surface hopping dynamics. The
# decoherence value comes from dynamics.deco.
SURFACEHOP_TEMPLATE = """&Surfacehop
tully
decoherence = {deco}
psub
"""

# Dynamics loop wrappers: the initial coordinates and velocities are
# copied into the Molcas working directory, then MOLCAS_MAXITER
# bounds the total substitution steps (must exceed the MD step
# count). The FOREACH loop re-runs SEWARD..DYNAMIX once per MD step.
COPY_XYZ_TEMPLATE = (
    ">> COPY $MOLCAS_SUBMIT_DIR/$Project.xyz $WorkDir/$Project.xyz\n")
COPY_VELOCITY_TEMPLATE = (
    ">> COPY $MOLCAS_SUBMIT_DIR/$Project.velocity.xyz "
    "$WorkDir/$Project.velocity.xyz\n")
# User orbital guess: the locked title.StrOrb is copied into the
# working directory under the RasOrb suffix so RASSCF can read it
# via fileorb=$MOLCAS_PROJECT.RasOrb on the first iteration.
COPY_STRORB_AS_RASORB_TEMPLATE = (
    ">> COPY $MOLCAS_SUBMIT_DIR/$Project.StrOrb "
    "$WorkDir/$Project.RasOrb\n")
FOREACH_TEMPLATE = ">> FOREACH ITER in (1 .. {nstep})\n"
END_FOREACH_TEMPLATE = ">> End do\n"

# MCKINLEY module template: vibrational frequency analysis
MCKINLEY_TEMPLATE = """&MCKINLEY
"""

# MCLR module template: Lagrange multipliers for CASPT2 NAC.
MCLR_TEMPLATE = """&MCLR
"""

# ESPF grid step. OpenMolcas stores DELTAR in ESPF.DATA with a 9-decimal
# format and a later module (espf_analysis, called by SCF/RASSCF/CASPT2)
# rebuilds the grid from that value with int(range/DeltaR). More decimals
# in the default (1/Angstrom) therefore change the bucket count between
# the two calls and MkGrid aborts. This value has exactly 9 decimals, so
# the round trip is lossless and both calls divide by the same double;
# it is the 9-decimal floor of 1/Angstrom, which keeps the nominal 1 A
# spacing. Same two lines serve the charge and the potential blocks.
ESPF_PNT_GRID = """GRID
PNT 4 1.889726124
"""

# ESPF charge embedding. External = n Angstrom, then x y z q and three
# zero dipole components (OpenMolcas reads seven values per line).
ESPF_CHARGE_TEMPLATE = """&ESPF
MultipoleOrder = 0
External = {ncharges} Angstrom
{charge_block}
{grid_block}"""

# ESPF potential embedding. External = 0, then atom V Fx Fy Fz and six
# zero field-gradient components.
ESPF_POTENTIAL_TEMPLATE = """&ESPF
MultipoleOrder = 0
External = 0
{potential_block}
{grid_block}"""

# CASPT2 SOC through RASSI. HEff reads the CASPT2-modified JobIph;
# EJob is the single-state approximation (diagonal energies + overlap
# off-diagonals).
RASSI_SOC_CASPT2_HEFF_TEMPLATE = """&RASSI
Nrof JobIphs
 {njobs} {states_line}
{jobiph_list}IphNames
{name_list}
Spin Orbit
HEff
"""

RASSI_SOC_CASPT2_EJOB_TEMPLATE = """&RASSI
Nrof JobIphs
 {njobs} {states_line}
{jobiph_list}IphNames
{name_list}
Spin Orbit
EJob
"""

# Copy the CASPT2-modified JobIph before the next spin block or the
# second (NAC) pass overwrites it.
COPY_JOBIPH_S_TEMPLATE = ">>COPY $Project.JobIph JOBIPH_S\n"
COPY_JOBIPH_T_TEMPLATE = ">>COPY $Project.JobIph JOBIPH_T\n"

# Remove the ESPF communication files between the ESPF and NAC passes.
ESPF_CLEANUP_TEMPLATE = (
    ">> RM -FORCE ESPF.DATA\n"
    ">> RM -FORCE ESPF.EXTPOT\n")

# COPY template: for SOC dual RASSCF orbital file management
COPY_TEMPLATE = ">>COPY   $WorkDir/$Project.JobIph   $WorkDir/JOB{job_id:03d}\n"

# MECP optimization: lock the singlet RunFile before the triplet
# RASSCF job so SLAPAF sees both spin jobs inside the Do while loop
COPY_RUNFILE2_TEMPLATE = \
    ">> COPY  $WorkDir/$Project.RunFile  $WorkDir/RUNFILE2\n"
