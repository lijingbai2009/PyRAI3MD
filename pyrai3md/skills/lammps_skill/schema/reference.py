"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 11 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module provides reference data for the LAMMPS skill,
including the periodic table, available profiles, and default parameters.

"""

# Periodic table: element symbols with their atomic numbers (electrons)
PERIODIC_TABLE = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
    21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30,
    31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40,
    41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50,
    51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60,
    61: 61, 62: 62, 63: 63, 64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70,
    71: 71, 72: 72, 73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80,
    81: 81, 82: 82, 83: 83, 84: 84, 85: 85, 86: 86,
}

# Available calculation profiles
# LAMMPS is a classical force field program serving the mm role only.
# It provides ground-state energy and gradient (gradient = -force),
# internal geometry optimization (minimize) and standalone classical
# dynamics in the nve/nvt/npt ensembles (the final geometry is
# returned through the fin_coord channel).
AVAILABLE_PROFILES = [
    'ff_energy',
    'ff_gradient',
    'ff_optimize',
    # The same engine-owned minimization with the box among the relaxed degrees
    # of freedom, which is `fix box/relax` under `minimize`. The engine owns the
    # box for the whole relaxation, so this profile is not a per-step route and
    # never runs in the resident service (see the note in validate.py).
    'ff_optimize_cell',
    'ff_nve',
    'ff_nvt',
    'ff_npt',
    # The two QM/MM tasks are standalone profiles: qmmm_field probes the
    # MM electrostatic potential/field, qmmm_energy evaluates the full
    # force field with the QM charges. They are deliberately not
    # modifiers of ff_gradient.
    'qmmm_field',
    'qmmm_energy',
]

# Default calculation parameters
DEFAULT = {
    'execution_mode': 'local',
    'shell': '',
    'project': 'lammps_calc',
    'profile': 'ff_energy',
    'method': 'ff',
    'charges': 0,
    'multiplicity': 1,
    'nstate': 1,
    # State information (from framework)
    'ci': [1],
    'spin': [0],
    'coupling': [],
    'verbose': 4,
    'solvent': 'gas',
    'temperature': 298.15,
    # LAMMPS-specific runtime
    'lammps_path': '',
    'lammps_project': 'LAMMPS',
    'lammps_nproc': 1,
    'lammps_units': 'real',
    # The accuracy token this skill writes into every `kspace_style` line it
    # issues, taking the style name from the user's own template. It is not a
    # cosmetic dial: the periodic QMMM field probe reads the MM potential as
    # the difference of two `run 0` evaluations whose total squared charge
    # differs by orders of magnitude, and LAMMPS sizes the reciprocal sum from
    # that charge set at every init. Left unconverged, the two halves sum over
    # different k vectors and the delivered V is short by exactly that
    # truncation - measured 5.3e-4 relative on the 1581-atom box at 1e-6,
    # 2.1e-6 at 1e-8, with 1e-10 and 1e-12 no longer moving the readings. A
    # classical trajectory pays 1.3-2x for it, which is why the shipped
    # non-probe cases spell 1e-4 out instead of inheriting this value.
    'ewald_accuracy': '1e-8',
    # Resident execution switch. 'python_enabled' asks for the LAMMPS
    # Python module held by a service process that is loaded once and
    # fed per-step geometry over a socket; it takes effect only when
    # execution_mode is not 'slurm' (there the skill blocks on one
    # sbatch job per step, so nothing can stay resident). 'python_path'
    # is the interpreter of that service process, i.e. the one that can
    # import lammps/mpi4py; 'lammps_path' keeps meaning the lmp binary
    # or installation directory of the subprocess path.
    'python_enabled': False,
    'python_path': '',
    # Resident-path artifact switch. Off, a resident step moves its
    # geometry in and its numbers out over the socket and writes nothing
    # but its own log. On, every step also issues the write commands the
    # subprocess path issues, so the dumps land beside the input under the
    # same names and the two paths can be compared file for file. It costs
    # the per-step file I/O the resident path exists to avoid, so it is a
    # debugging aid rather than a mode to run production in.
    'debug': False,
    # Execution driver, copied in by the framework's skill-config merge
    # (schema/adapter.py Layer 0) and never set by a skill section: it says who
    # owns the geometry, which decides whether the shake keywords below mean
    # anything at all (D-27 ownership rule).
    # 'skill', not 'main', because the absence of a statement is not a
    # statement: a &lammps section built on its own has no framework in it, so
    # the engine integrates and its own constraint keywords apply, exactly as
    # they did before this field existed. Only a run the framework drives
    # arrives here with driver='main', and Layer 0 is what says so.
    'driver': 'skill',
    # Shake constraint switch (linked to the framework MD shake
    # constraint implementation; see skill_guide.md).
    # Default off keeps the legacy behavior where the TIP3P water
    # shake fix defined in the input template is disabled via unfix.
    'lammps_shake': 0,
    'shake_fix_id': 'fShakeTIP3P',
    'shake_keywords': '',
    # Geometry optimization parameters (minimize command,
    # aligned with the reference 'min_style cg' +
    # 'minimize 0.0001 1.0e-8 1000 1000000' input)
    # Engine-side box relaxation (`ff_optimize_cell`). `box_relax` is which
    # lattice degrees of freedom the engine may move - 'off' keeps the box the
    # framework handed over, 'iso' scales it, 'aniso' lets each axis answer for
    # itself - and `box_relax_pressure` is the target pressure in bar, the unit
    # every other pressure keyword in this package uses, converted to the
    # engine's own unit at the point of writing.
    'box_relax': 'off',
    'box_relax_pressure': 0.0,
    'min_style': 'cg',
    'min_etol': 1e-4,
    'min_ftol': 1e-8,
    'min_maxiter': 1000,
    'min_maxeval': 1000000,
    # Standalone MD parameters (aligned with the reference
    # nve/nvt/npt inputs)
    'md_nstep': 100,
    'md_timestep': 1.0,
    'md_seed': 12345,
    'dump_freq': 100,
    'thermo_damp': 100.0,
    'pressure': 1.0,
    'pressure_damp': 1000.0,
    # How the barostat couples to the cell: 'isotropic' scales the volume
    # as one scalar, 'anisotropic' lets the three axes move on their own.
    # Same word list as the framework's md.barostat_mode, which this mirrors
    # so a run driven by LAMMPS deforms its cell the way the same input
    # would when the framework integrator drives it. Declared here so the
    # keyword survives the framework-to-skill merge instead of being
    # silently dropped.
    'barostat_mode': 'isotropic',
    # Whole-system stress tensor request. The framework planning route
    # (md+ensemble=npt or optimize+optimize_cell) or a standalone test
    # yaml sets it; declaring it here routes the keyword into the skill
    # section so it reaches Calculation.stress in framework modes.
    'stress': False,
    # Topology rows this task computes (schema.config.System.atom_selection).
    # The framework writes it per region task from the plan's region indices;
    # [] is the whole system. Declared here so the framework-to-skill merge
    # routes the keyword instead of dropping it.
    'atom_selection': [],
    # QM/MM region treatment (schema.config.System.qmmm / qm_atoms).
    # `qm_atoms` is 0-based in the framework atom order. Declared here so
    # the framework planner can inject the partition into a task config
    # (a keyword the skill never registers is dropped by the merge) and so
    # the qualification converter can emit it from a test yaml.
    'qm_atoms': [],
    # The center a qmmm_field probe measures (0-based framework atom
    # indices, a subset of qm_atoms). Empty means the whole QM set, which is
    # every single-center run: the probe then sees only the MM charges, as
    # it always did. Under several centers the atoms it does not measure are
    # environment charges, on the same side of the A/B difference as the MM
    # subsystem, so each center has to name itself.
    'qm_probe': [],
    # How the QM set is tiled into centers (0-based framework atom indices,
    # one group per center). `qmmm_energy` uses it to remove the in-cell Q-Q
    # pairs inside each center only, which is what leaves the inter-center
    # pair in the classical surface where both centers feel its force. Empty
    # means one center, the single-center behavior.
    'qm_centers': [],
    # QM 0-order charges (qm_atoms order, e) and an optional precomputed
    # ESPF V/E file for the qmmm_energy task. Runtime code may also pass
    # them through calculate(qmmm_inputs=...); the static fields serve
    # cold standalone runs and test yaml files.
    'q_qm': [],
    'qmmm_field_file': '',
    'keep_tmp': 0,
    # External potential (from framework)
    'shape': 'ellipsoid',
    'factor': 40,
    'scale': 1.0,
    # Periodic settings (from framework)
    'pbc': [0, 0, 0],
    'cell': [99.0, 99.0, 99.0, 90.0, 90.0, 90.0],
    'lattice_vector': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    'kpoint': [1, 1, 1],
    # Slurm settings
    'slurm_time': '01:00:00',
    # Empty means "no --partition directive": the cluster's default
    # partition answers. A named partition is a site choice, so it comes
    # from the input or the deployment's skill section.
    'slurm_partition': '',
    'slurm_ntasks': 1,
    'slurm_memory': 4000,
    'slurm_command': '',
    # Compute resources, in two spellings each (#306): the bare name is
    # what a skill reads and what a standalone input writes, the
    # skill-prefixed name is what the framework forwards from &lammps.
    # The larger of a pair wins (schema/config.py Environment).
    'thread': 1,
    'lammps_thread': 1,
    'nproc': 1,
}
