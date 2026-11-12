"""
======================================================================
This module provides the MD trajectory output writer. It appends the
per-step trajectory records (structure, velocities, energies, state
populations) and the surface-hopping event records to disk, following
the legacy PyRAI2MD aimd file formats.

Files (all in the log directory, named after the run title):

    title.md.xyz        structure of every recorded step (Angstrom); the
                        frame is the wrapped main image and a periodic run
                        states its cell in the comment line as an
                        extended-XYZ Lattice="..." key, which is what lets
                        a reader apply the minimum-image convention to a
                        molecule that straddles a cell face
    title.md.velo       velocities of every recorded step (au)
    title.md.energies   time / Epot / Ekin / Etot / per-state energies
    title.md.pop        time / per-state populations / active state
    title.md.stress     time / active-state 3x3 stress / pressure (bar)
    title.md.ensemble   time / per-step thermostat & barostat record
    title.sh.xyz        structure at hopping steps only
    title.sh.velo       velocities at hopping steps only (post-hop)
    title.sh.energies   energies at hopping steps only
    title.sh.pop        populations at hopping steps only
    title.sh.stress     stress / pressure at hopping steps only
    title.sh.ensemble   ensemble record at hopping steps only

The md.stress/sh.stress files are written only when a backend supplied
a non-zero whole-system stress tensor (an NPT run with the cp2k / xtb /
lammps stress capability); the stress row carries the time, the nine
row-major components of the active-state 3x3 tensor in Hartree/Bohr^3
and the derived isotropic pressure P = -Tr(stress)/3 in bar. Runs
without stress stay byte-identical and create no stress file.

The md.ensemble/sh.ensemble files are written only for an extended
ensemble (dynamics.ensemble nvt/npt); an nve run creates no file and
stays byte-identical. Each row carries the time plus the per-step
thermostat/barostat record: instantaneous and target temperature,
N_df, Ekin, volume, the velocity scale factors, the Nose-Hoover chain
masses/positions/momenta, the CSVR rescaling parameters, the kinetic /
virial / internal / target pressures (bar) and the barostat matrices
(mu, g_eps, eps, p_eps, v_eps) as nine row-major components in atomic
units. The column set is planned once from the dynamics keywords and
headed lazily on first write, mirroring the stress-file contract. The
per-step intermediates come from the runtime.ensemble diagnostics
recorded by workflow.integrator.ensemble; persisted DOF (chain state,
strain, momenta) are read from the snapshot system fields.

The md/sh pop records carry two column blocks: the electronic state
populations (the density-matrix diagonal propagated by the surface
hopping integrator) followed by the active state population, a 0/1
one-hot vector marking the post-hop active state of the step.

The md.* records obey the &md direct/buffer output control: the first
`direct` steps are written every step; afterwards one record is
written every `buffer` steps and the final step is forced. The sh.*
records are written immediately whenever a hop happened, independent
of the direct/buffer control.

Aug 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np

from ...utilities.structure_formatter import print_charges, print_coord
from ...utilities.lattice import cell6_to_matrix
from ...utilities.units import (amu_to_au, au_pressure_to_bar,
                                KELVIN_TO_HARTREE)
from ...schema.validate import THERMOSTAT_ALIASES, BAROSTAT_ALIASES
from ...kernel.logger import get_logger

_LOGGER = get_logger(__name__)


def _print_coord(atoms, values):
    """Format a trajectory row without the run-log indentation."""
    return print_coord(atoms, values, indent='')


def lattice_comment(system):
    """Extended-XYZ ``Lattice="..."`` for a periodic frame, else empty.

    The geometry trajectory is written in the main image, which is what
    lets a reader apply the minimum-image convention to find bonds and
    contacts; it also means a molecule sitting on two sides of a cell face
    looks broken to a reader that does not know the cell. The box is
    exactly the missing fact, so it travels in the frame's comment line
    under the extended XYZ convention standard readers already parse. A
    non-periodic frame, or one whose cell is missing, keeps the legacy
    comment line byte-for-byte.

    Parameters
    ----------
    system : dict
        The snapshot system block of the frame being written.

    Returns
    -------
    str
        A leading-space key, or an empty string.
    """
    pbc = system.get('pbc') or []
    if not any(int(flag) for flag in pbc):
        return ''
    lattice = np.asarray(system.get('lattice_vector') or [],
                         dtype=np.float64)
    if lattice.shape != (3, 3) or not np.any(lattice):
        cell6 = np.asarray(system.get('cell') or [],
                           dtype=np.float64).reshape(-1)
        if cell6.size != 6 or not np.any(cell6):
            return ''
        lattice = cell6_to_matrix(cell6)
    return ' Lattice="%s"' % ' '.join('%.6f' % value
                                      for value in lattice.reshape(-1))


def record_due(step, direct, buffer, final=False):
    """Whether this step is one the run asked to have recorded.

    Steps 1..direct are written directly; afterwards every `buffer`-th step
    is written, counted from the end of the direct window, and the final
    step is always written. The predicate is a pure function of the step
    number, so a step that produces no record (a missing snapshot) cannot
    shift the phase of the following ones.

    This is the cadence of the md.* trajectory files, and it is also the
    cadence the run-health monitor emits on: a run that records 21 frames of
    2000 steps says about its box 21 times, not two thousand, and no second
    knob has to be tuned to get that. `MdOutputWriter.should_output` delegates
    here so the two cannot drift apart.
    """
    if step <= direct:
        return True
    return bool(final) or (step - direct) % max(int(buffer), 1) == 0


class MdOutputWriter:
    """Appends per-step and hopping-event MD trajectory records.

    Data source policy (identical to the checkpoint design): the
    recorded step data (coordinates, velocities, energies, kinetic
    energy) comes from the latest history snapshot, i.e. the values at
    computation time before surface hopping; the live state only
    supplies the post-hop active state, the post-hop velocities for
    the sh.* records and the propagated state populations.

    Parameters
    ----------
    title : str
        Run title; every file is named `<title>.<suffix>`.
    log_dir : str or None
        Output directory (same resolution as the Logger; defaults to
        the current working directory).
    direct : int
        Number of leading steps written every step.
    buffer : int
        Writing period once `direct` is exceeded.
    dt : float
        MD timestep in atomic units (used for the time column).
    total_steps : int
        Planned number of steps; the last step is always written.
    center_regions : list of list of int or None
        Per-center region atom indices (0-based), one list per center,
        or None for a plain single-center run. When at least one
        center carries a non-empty region (an environment partition,
        docs R6 r6c) the writer additionally emits the per-center
        md.#.*/sh.#.* files and expands the global energies/pop
        columns per center; a plain run stays byte-identical.
    charge_output : bool
        Whether ONIOM charge management is active. When True the
        writer additionally emits md.charges/sh.charges (and, for a
        partitioned run, the per-center md.#.charges/sh.#.charges)
        carrying the managed-region point charges [q, x, y, z] of each
        recorded frame; when False the file matrix is byte-identical to
        a run without charge management.
    """

    def __init__(self, title, log_dir=None, direct=2000, buffer=500,
                 dt=20.67, total_steps=0, center_regions=None,
                 charge_output=False):
        self.title = str(title)
        self.log_dir = log_dir or os.getcwd()
        self.direct = int(direct)
        self.buffer = max(int(buffer), 1)
        self.dt = float(dt)
        self.total_steps = int(total_steps)
        self.charge_output = bool(charge_output)
        # Lazily planned md.ensemble column set (stable per run).
        self._ensemble_col_cache = None
        # Columns whose read failed; each is reported once, not per step.
        self._ensemble_read_warnings = set()
        # Per-center region atom indices (one list per center, 0-based),
        # or None for a plain single-center run. A run is "partitioned"
        # (an environment partition exists, n_layer >= 2) when at least
        # one center carries a non-empty region; only then are the
        # per-center md.#.*/sh.#.* files written. C=1 without a region
        # and test doubles stay on the byte-identical legacy path.
        self.center_regions = ([list(region) for region in center_regions]
                               if center_regions else None)
        self.partitioned = bool(
            self.center_regions
            and any(len(region) > 0 for region in self.center_regions))

    # ---- Paths and headers ----

    def _path(self, suffix):
        return os.path.join(self.log_dir,
                            '%s.%s' % (self.title, suffix))

    def _file_suffixes(self):
        """Every trajectory file suffix this run writes.

        The global md.*/sh.* always; a partitioned run adds the
        per-center md.#.*/sh.#.* (# = 1-based center index, docs R6
        r6c). Used by reset() and write_headers() so the whole file
        matrix is cleared and headed consistently.

        Returns
        -------
        list of str
            File suffixes (appended to '<title>.').
        """
        suffixes = ['%s.%s' % (prefix, ext)
                    for prefix in ('md', 'sh')
                    for ext in ('xyz', 'velo', 'energies', 'pop', 'stress',
                                'ensemble')]
        if self.charge_output:
            suffixes.extend('%s.charges' % prefix
                            for prefix in ('md', 'sh'))
        if self.partitioned:
            for center in range(len(self.center_regions)):
                suffixes.extend(
                    '%s.%d.%s' % (prefix, center + 1, ext)
                    for prefix in ('md', 'sh')
                    for ext in ('xyz', 'velo', 'energies', 'pop'))
                if self.charge_output:
                    suffixes.extend(
                        '%s.%d.charges' % (prefix, center + 1)
                        for prefix in ('md', 'sh'))
        return suffixes

    def reset(self):
        """Remove any existing trajectory output files of this run.

        The md/sh records are append-mode files, so a fresh start
        must wipe the leftovers of a previous run in the same
        directory; otherwise repeated executions keep stacking
        frames and headers on top of each other. Called once at md
        workflow startup before the headers are written. A
        partitioned run also wipes the per-center md.#.*/sh.#.*
        files (docs R6, r6c).
        """
        for suffix in self._file_suffixes():
            path = self._path(suffix)
            if os.path.exists(path):
                os.remove(path)

    def write_headers(self):
        """Write the column headers of the energies and pop files.

        The xyz/velo files are self-describing (natom + comment line
        per frame) and get no header. Called once at the start of a
        fresh run; append mode keeps restart runs intact. A
        partitioned run also heads the per-center energies/pop files.
        """
        energy_head = '%20s%28s%28s%28s%28s\n' % (
            'time', 'Epot', 'Ekin', 'Etot', 'Epot1,2,3...')
        pop_head = '%20s%28s%28s\n' % ('time', 'Pop1,2,3...',
                                       'Act1,2,3...')
        for suffix in self._file_suffixes():
            if suffix.endswith('.energies'):
                with open(self._path(suffix), 'a') as log:
                    log.write(energy_head)
            elif suffix.endswith('.pop'):
                with open(self._path(suffix), 'a') as log:
                    log.write(pop_head)

    # ---- Direct/buffer output control ----

    def should_output(self, step, final=False):
        """Decide whether the current step is written to the md.* files.

        Delegates to :func:`record_due`, the same predicate the run-health
        monitor emits on, so "what this run asked to see" is decided in one
        place by one pair of numbers (`md.direct`, `md.buffer`).

        Parameters
        ----------
        step : int
            Current step number (1-based).
        final : bool
            Whether this is the last step of the trajectory.

        Returns
        -------
        bool
            True when the step record must be written.
        """
        return record_due(step, self.direct, self.buffer, final)

    # ---- Step recording ----

    def record(self, step, snapshot, state, final=False):
        """Append the trajectory records of one completed step.

        Parameters
        ----------
        step : int
            Current step number (1-based).
        snapshot : dict or None
            Latest history snapshot (pmd model dump) holding the
            pre-hop coordinates, velocities, energies and active
            state of the step.
        state : PyRAIState or None
            Live state after surface hopping; supplies the post-hop
            active state, the post-hop velocities and the propagated
            state populations.
        final : bool
            Whether this is the last step of the trajectory.

        Returns
        -------
        bool or None
            Whether the step was written to the md.* files (the
            direct/buffer gate), so the caller can apply the same
            control to the run-log step block. None means the record
            was skipped for missing data (no gate decision taken).
        """
        if not snapshot:
            return None
        if self.partitioned:
            return self._record_partitioned(step, snapshot, state, final)
        write_md = self.should_output(step, final=final)

        system = snapshot.get('system') or {}
        results = snapshot.get('results') or {}
        atoms = list(system.get('atoms') or [])
        coords = np.asarray(system.get('coordinates') or [],
                            dtype=np.float64)
        energies = np.asarray(results.get('energy') or [],
                              dtype=np.float64).reshape(-1)
        if not atoms or coords.size == 0:
            return None
        natom = len(atoms)
        coords = coords.reshape(natom, 3)
        snap_velo = np.asarray(system.get('velocities') or [],
                               dtype=np.float64).reshape(-1)

        pre_state = int(system.get('current_state', 0))
        live_state = int(state.current_state) if state is not None \
            else pre_state
        hoped = live_state != pre_state

        # Legacy comment line; a successful hop annotates both the
        # md.* and sh.* frames of the step.
        cmmt = '%s coord %d state %d' % (self.title, step,
                                         pre_state + 1)
        if hoped:
            cmmt += ' to %d CI' % (live_state + 1)

        ekin = self._kinetic_energy(system, snap_velo)
        epot = float(energies[pre_state]) if energies.size > pre_state \
            else 0.0
        population = self._live_population(state)
        # Active state population: a 0/1 one-hot vector of the
        # post-hop active state, same length as the populations.
        active = [1.0 if j == live_state else 0.0
                  for j in range(len(population))]

        stress = self._active_stress_tensor(results, pre_state)
        ensemble_row = self._ensemble_row(snapshot, state, system, ekin)

        if write_md:
            self._write_frame('md.xyz', natom,
                              cmmt + lattice_comment(system),
                              _print_coord(atoms, coords))
            self._write_frame(
                'md.velo', natom, cmmt,
                _print_coord(atoms, self._velo_matrix(snap_velo, natom)))
            self._write_row('md.energies', step, epot, ekin, energies)
            self._write_row('md.pop', step, None, None,
                            population + active)
            self._write_charges('md.charges', system, cmmt)
            if stress is not None:
                self._write_stress('md.stress', step, stress)
            if ensemble_row is not None:
                self._write_ensemble('md.ensemble', step, ensemble_row)

        if hoped:
            live_velo = np.asarray(state.velocities,
                                   dtype=np.float64).reshape(-1) \
                if state is not None else snap_velo
            self._write_frame('sh.xyz', natom, cmmt,
                              _print_coord(atoms, coords))
            self._write_frame(
                'sh.velo', natom, cmmt,
                _print_coord(atoms, self._velo_matrix(live_velo, natom)))
            self._write_row('sh.energies', step, epot, ekin, energies)
            self._write_row('sh.pop', step, None, None,
                            population + active)
            self._write_charges('sh.charges', system, cmmt)
            if stress is not None:
                self._write_stress('sh.stress', step, stress)
            if ensemble_row is not None:
                self._write_ensemble('sh.ensemble', step, ensemble_row)

        return write_md

    # ---- Partitioned (environment-partition) output: docs R6 r6c ----

    def _record_partitioned(self, step, snapshot, state, final):
        """Append the global + per-center records of one step.

        An environment partition exists (n_layer >= 2). The global
        md.*/sh.* carry the whole-system coordinates/velocities and
        the assembled energies/populations expanded per center; each
        center additionally writes md.#.*/sh.#.* scoped to its region
        atoms (# = 1-based center index). C=1 with a region degrades
        to a single center (the global files hold the single-center
        result and md.1.*/sh.1.* hold the region).
        """
        write_md = self.should_output(step, final=final)
        records = self._center_records(snapshot, state)
        system = snapshot.get('system') or {}
        atoms = list(system.get('atoms') or [])
        if not atoms or not records:
            return None
        natom = len(atoms)
        coords = np.asarray(system.get('coordinates') or [],
                            dtype=np.float64).reshape(natom, 3)
        velo_pre = np.asarray(system.get('velocities') or [],
                              dtype=np.float64).reshape(-1)
        ekin = self._kinetic_energy(system, velo_pre)
        epot = self._partitioned_epot(snapshot, records)

        pre_config = [record['active_pre'] for record in records]
        post_config = [record['active_post'] for record in records]
        hoped = post_config != pre_config

        # Per-center expanded columns: energies and populations are
        # concatenated center by center; the active one-hot follows
        # the same per-center layout (post-hop active local state).
        energy_cols = []
        pop_cols = []
        onehot_cols = []
        for record in records:
            energy_cols.extend(float(e) for e in record['energies'])
            pop_cols.extend(float(p) for p in record['population'])
            nstate_c = len(record['energies'])
            onehot_cols.extend(
                1.0 if j == record['active_post'] else 0.0
                for j in range(nstate_c))
        md_active = [float(record['active_pre'] + 1) for record in records]
        sh_active = [float(record['active_post'] + 1) for record in records]

        cmmt = '%s coord %d states %s' % (
            self.title, step,
            ' '.join(str(record['active_pre'] + 1) for record in records))
        if hoped:
            cmmt += ' to %s CI' % ' '.join(
                str(record['active_post'] + 1) for record in records)

        # Whole-system stress is a single structural-property provider
        # and is not partitioned per center; project the reference block
        # (the lone center's active state for C=1, index 0 for C>1) and
        # write it to the global md.stress/sh.stress only.
        stress_state = pre_config[0] if len(pre_config) == 1 else 0
        stress = self._active_stress_tensor(snapshot.get('results') or {},
                                            stress_state)
        # The ensemble record is a whole-system quantity (one global
        # thermostat / barostat over the assembled dynamics), so it is
        # written to the global md.ensemble/sh.ensemble only, exactly
        # like the stress files.
        ensemble_row = self._ensemble_row(snapshot, state, system, ekin)

        if write_md:
            self._write_frame('md.xyz', natom,
                              cmmt + lattice_comment(system),
                              _print_coord(atoms, coords))
            self._write_frame(
                'md.velo', natom, cmmt,
                _print_coord(atoms, self._velo_matrix(velo_pre, natom)))
            self._write_row('md.energies', step, epot, ekin,
                            md_active + energy_cols)
            self._write_row('md.pop', step, None, None,
                            pop_cols + onehot_cols)
            self._write_charges('md.charges', system, cmmt)
            if stress is not None:
                self._write_stress('md.stress', step, stress)
            if ensemble_row is not None:
                self._write_ensemble('md.ensemble', step, ensemble_row)
            for center, record in enumerate(records):
                self._write_center_files(
                    'md', center + 1, step, record, atoms, coords,
                    velo_pre, system, record['active_pre'])

        if hoped:
            live_velo = (np.asarray(state.velocities, dtype=np.float64)
                         .reshape(-1) if state is not None else velo_pre)
            self._write_frame('sh.xyz', natom, cmmt,
                              _print_coord(atoms, coords))
            self._write_frame(
                'sh.velo', natom, cmmt,
                _print_coord(atoms, self._velo_matrix(live_velo, natom)))
            self._write_row('sh.energies', step, epot, ekin,
                            sh_active + energy_cols)
            self._write_row('sh.pop', step, None, None,
                            pop_cols + onehot_cols)
            self._write_charges('sh.charges', system, cmmt)
            if stress is not None:
                self._write_stress('sh.stress', step, stress)
            if ensemble_row is not None:
                self._write_ensemble('sh.ensemble', step, ensemble_row)
            for center, record in enumerate(records):
                self._write_center_files(
                    'sh', center + 1, step, record, atoms, coords,
                    live_velo, system, record['active_post'])

        return write_md

    def _center_records(self, snapshot, state):
        """Per-center electronic records of the step (docs R6, r6c).

        One dict per center: the per-state energies of the step, the
        pre/post-hop active local states, the post-hop population
        (density diagonal) and the region atoms. C>1 reads the
        per-center electronic outcome and the active configuration;
        a single center with a region degrades to the legacy global
        state-resolved result and current_state.
        """
        n_center = len(self.center_regions)
        records = []
        if n_center > 1:
            outcome = snapshot.get('electronic_outcome') or {}
            nuclear = snapshot.get('nuclear_sample') or {}
            pre_config = tuple(nuclear.get('active_config_pre') or ())
            post_config = (tuple(state.get_active_configuration())
                           if state is not None else pre_config)
            for center in range(n_center):
                carrier = (outcome.get(center)
                           or outcome.get(str(center)) or {})
                energy = carrier.get('energy')
                energies = (np.asarray(energy, dtype=np.float64).reshape(-1)
                            if energy is not None
                            else np.zeros(0, dtype=np.float64))
                density = (state.center_electronic(center).get('density')
                           if state is not None else None)
                population = self._density_population(density, energies.size)
                records.append({
                    'energies': energies,
                    'active_pre': (int(pre_config[center])
                                   if center < len(pre_config) else 0),
                    'active_post': (int(post_config[center])
                                    if center < len(post_config) else 0),
                    'population': population,
                    'region': list(self.center_regions[center]),
                })
            return records

        # C=1 with a region: the single center reads the legacy global
        # state-resolved result and current_state (the R5 transparent
        # binding tracks the active state through pmd.system).
        system = snapshot.get('system') or {}
        results = snapshot.get('results') or {}
        energy = results.get('energy')
        energies = (np.asarray(energy, dtype=np.float64).reshape(-1)
                    if energy is not None
                    else np.zeros(0, dtype=np.float64))
        active_pre = int(system.get('current_state', 0))
        active_post = (int(state.current_state)
                       if state is not None else active_pre)
        population = list(self._live_population(state))
        if not population:
            population = [0.0] * energies.size
        records.append({
            'energies': energies,
            'active_pre': active_pre,
            'active_post': active_post,
            'population': population,
            'region': list(self.center_regions[0]),
        })
        return records

    @staticmethod
    def _density_population(density, nstate):
        """Real diagonal of a center's density matrix (population).

        The persisted density is an [re, im] nested list of shape
        (S, S, 2); the population is the real diagonal. Missing or
        malformed data yields a zero vector of the state length.
        """
        if density is None:
            return [0.0] * int(nstate)
        array = np.asarray(density, dtype=np.float64)
        if array.ndim == 3 and array.shape[-1] == 2:
            array = array[..., 0]
        if array.ndim == 2 and array.shape[0] == array.shape[1]:
            return [float(value) for value in np.diag(array)]
        return [0.0] * int(nstate)

    def _partitioned_epot(self, snapshot, records):
        """Global assembled active-combination potential energy.

        C>1 carries the committed active_energy; a single center with
        a region falls back to the active state's row of the legacy
        state-resolved result energy.
        """
        results = snapshot.get('results') or {}
        active_energy = results.get('active_energy')
        if active_energy is not None:
            return float(active_energy)
        energy = results.get('energy')
        energies = (np.asarray(energy, dtype=np.float64).reshape(-1)
                    if energy is not None
                    else np.zeros(0, dtype=np.float64))
        active_pre = records[0]['active_pre'] if records else 0
        if energies.size > active_pre:
            return float(energies[active_pre])
        return 0.0

    def _write_center_files(self, prefix, center_num, step, record,
                            atoms, coords, velo_flat, system, active_state):
        """Write one center's md.#/sh.# step record (region-scoped).

        The coordinates/velocities are sliced to the center's region
        atoms; the energies/population are the center's own. The
        format mirrors the global md.*/sh.* files (# = 1-based center
        index, docs R6 r6c).
        """
        region = [i for i in record['region'] if 0 <= i < len(atoms)]
        if not region:
            return
        region_atoms = [atoms[i] for i in region]
        region_coords = coords[region]
        velo = velo_flat.reshape(-1, 3)
        region_velo = velo[region] if velo.shape[0] == len(atoms) \
            else np.zeros((len(region), 3))
        energies = np.asarray(record['energies'], dtype=np.float64)
        nstate_c = energies.size

        masses = np.asarray(system.get('mass') or [], dtype=np.float64)
        if masses.size == len(atoms):
            mass_au = amu_to_au(masses[region])
            ekin_c = float(0.5 * np.sum(mass_au[:, None] * region_velo ** 2))
        else:
            ekin_c = 0.0
        epot_c = (float(energies[active_state])
                  if nstate_c > active_state else 0.0)

        suffix = '%s.%d' % (prefix, center_num)
        cmmt = '%s coord %d center %d state %d' % (
            self.title, step, center_num, active_state + 1)
        self._write_frame('%s.xyz' % suffix, len(region_atoms), cmmt,
                          _print_coord(region_atoms, region_coords))
        self._write_frame('%s.velo' % suffix, len(region_atoms), cmmt,
                          _print_coord(region_atoms, region_velo))
        self._write_row('%s.energies' % suffix, step, epot_c, ekin_c,
                        [float(active_state + 1)]
                        + [float(e) for e in energies])
        onehot = [1.0 if j == record['active_post'] else 0.0
                  for j in range(nstate_c)]
        self._write_row('%s.pop' % suffix, step, None, None,
                        [float(p) for p in record['population']] + onehot)
        if self.charge_output:
            region_charges = self._region_charges(
                self._point_charges(system), region)
            if region_charges:
                self._write_frame('%s.charges' % suffix,
                                  len(region_charges), cmmt,
                                  print_charges(region_charges))

    # ---- Formatting helpers ----

    def _kinetic_energy(self, system, velocities):
        """Kinetic energy (Hartree) of the recorded snapshot."""
        masses = np.asarray(system.get('mass') or [],
                            dtype=np.float64).reshape(-1)
        natom = len(system.get('atoms') or [])
        if masses.size != natom or velocities.size != natom * 3:
            return 0.0
        mass_au = amu_to_au(masses)
        return float(0.5 * np.sum(mass_au[:, None]
                                  * velocities.reshape(natom, 3) ** 2))

    @staticmethod
    def _live_population(state):
        """Propagated state populations from the live state.

        FSSH writes the density-matrix diagonal back through
        update_dynamics after every step; nosh runs may carry none.
        """
        if state is None or state.pmd is None:
            return []
        return [float(p) for p in
                (state.pmd.dynamics.state_population or [])]

    def _write_frame(self, suffix, natom, comment, body):
        """Append one xyz-style frame (natom + comment + rows)."""
        with open(self._path(suffix), 'a') as log:
            log.write('%d\n%s\n%s' % (natom, comment, body))

    @staticmethod
    def _velo_matrix(velo_flat, natom):
        """(natom, 3) velocity matrix, zero-filled when absent.

        A run whose input carried no velocities (an agent .config
        without the structure key, initcond '0' from a bare geometry)
        still records the velocity file, so the md.* matrix stays
        complete across the three input modes and every recorded step
        owns one velo frame.
        """
        array = np.asarray(velo_flat if velo_flat is not None else [],
                           dtype=np.float64).reshape(-1)
        if array.size == natom * 3:
            return array.reshape(natom, 3)
        return np.zeros((natom, 3), dtype=np.float64)

    def _write_row(self, suffix, step, epot, ekin, values):
        """Append one tabular row (energies or populations).

        The energies rows carry time / Epot / Ekin / Etot followed by
        the per-state energies; the population rows carry time
        followed by the per-state populations and the 0/1 active
        state population columns.
        """
        values = [] if values is None else list(values)
        if suffix.endswith('.pop') and not values:
            return
        text = '%20.2f' % (max(step - 1, 0) * self.dt)
        if epot is not None:
            text += '%28.16f%28.16f%28.16f' % (epot, ekin,
                                               epot + ekin)
        text += ''.join('%28.16f' % v for v in values)
        with open(self._path(suffix), 'a') as log:
            log.write(text + '\n')

    # ---- Stress-trajectory helpers (NPT whole-system virial) ----

    _STRESS_COMPONENTS = ('sxx', 'sxy', 'sxz', 'syx', 'syy', 'syz',
                          'szx', 'szy', 'szz')

    @staticmethod
    def _active_stress_tensor(results, pre_state):
        """Project the active-state 3x3 stress from a results dict.

        Mirrors PyRAIState.active_stress(): reads the (nstate, 3, 3)
        whole-system stress record and returns the reference block at
        pre_state (index 0 for a multi-center run or an out-of-range
        state). Returns None when no backend filled the stress for this
        step or the block is identically zero, so callers skip writing
        and non-NPT runs stay byte-identical.
        """
        stress = (results or {}).get('stress_tensor')
        if stress is None:
            return None
        array = np.asarray(stress, dtype=np.float64)
        if array.size == 0:
            return None
        if array.ndim == 2:
            array = array.reshape((1,) + array.shape)
        if array.ndim != 3 or array.shape[1:] != (3, 3):
            return None
        idx = int(pre_state or 0)
        if idx < 0 or idx >= array.shape[0]:
            idx = 0
        tensor = array[idx]
        if not np.any(tensor):
            return None
        return tensor

    def _write_stress(self, suffix, step, tensor):
        """Append one stress-trajectory row (time / 3x3 / pressure).

        The nine row-major tensor components are written in
        Hartree/Bohr^3 followed by the derived isotropic pressure
        P = -Tr(stress)/3 converted to bar. The column header is
        written lazily on first creation so a run that never records a
        stress leaves no file behind.
        """
        tensor = np.asarray(tensor, dtype=np.float64).reshape(3, 3)
        pressure_au = -float(np.trace(tensor)) / 3.0
        path = self._path(suffix)
        new_file = not os.path.exists(path)
        text = '%20.2f' % (max(step - 1, 0) * self.dt)
        text += ''.join(self._ens_cell(float(v))
                        for v in tensor.reshape(-1))
        text += '%20.4f' % au_pressure_to_bar(pressure_au)
        with open(path, 'a') as log:
            if new_file:
                head = '%20s' % 'time'
                head += ''.join('%20s' % c for c in self._STRESS_COMPONENTS)
                head += '%20s\n' % 'P(bar)'
                log.write(head)
            log.write(text + '\n')

    # ---- Ensemble-trajectory helpers (per-step thermostat/barostat) ----

    _MATRIX_COMPONENTS = ('xx', 'xy', 'xz', 'yx', 'yy', 'yz',
                          'zx', 'zy', 'zz')

    @staticmethod
    def _canon_token(value, aliases, default):
        """Normalize a dynamics token (lower-case, alias-resolved)."""
        token = str(value if value is not None else default).strip().lower()
        return aliases.get(token, token) or default

    @staticmethod
    def _ens_cell(value, width=20, precision=12):
        """Render one trajectory value in `width` characters, separator kept.

        The record is a whitespace-separated table and the ONLY separator
        two adjacent columns get is the padding of the right-hand one, so
        the fit test has to be on the digits themselves, not on the padded
        field: a value that fills all 20 characters leaves no space before
        its neighbour and the pair reads back as one 40-character token
        (`-10.996128489922` next to `9807535.803130459040` did exactly that
        in the 07:13 products, and every column after it shifted by one).
        Fixed-point stays the default so an ordinary row is byte-identical
        to before this existed; the exponent form is the first fallback and
        a forced leading space the last, so no magnitude can merge.
        """
        text = '%.*f' % (precision, value)
        if len(text) >= width:
            text = '%.*e' % (precision, value)
        if len(text) >= width:
            return ' ' + text
        return text.rjust(width)

    @staticmethod
    def _ens_num(value):
        """Coerce a diagnostic value to float (NaN when unusable)."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float('nan')

    @classmethod
    def _slot(cls, source, key, idx):
        """Column getter: component idx of a flat/nested list entry.

        Reads ctx[source][key] (a scalar list or a 3x3 nested list),
        flattens it row-major and returns the idx-th component, or NaN
        when the entry is missing or too short.
        """
        def getter(ctx):
            data = (ctx.get(source) or {}).get(key)
            flat = np.asarray(data if data is not None else [],
                              dtype=np.float64).reshape(-1)
            return float(flat[idx]) if idx < flat.size else float('nan')
        return getter

    def _ensemble_columns(self, dynamics):
        """Plan the ordered md.ensemble columns for this run's setup.

        The layout is derived once from the dynamics keywords (the
        column set is stable for the whole run): a fixed core block
        (temperatures, N_df, Ekin, volume), a thermostat block
        (Nose-Hoover chain scales/masses/DOF or the CSVR rescaling
        parameters) and a barostat block (Berendsen mu or the
        extended-cell pressures, force, strain, momenta and velocities
        as nine row-major components in atomic units; MTTK adds the
        cell-momentum bath). Missing per-step diagnostics render as
        NaN so the columns stay aligned.

        Parameters
        ----------
        dynamics : dict
            The snapshot dynamics section.

        Returns
        -------
        list of tuple
            (column name, getter(ctx)) in write order.
        """
        if self._ensemble_col_cache is not None:
            return self._ensemble_col_cache

        thermostat = self._canon_token(dynamics.get('thermostat'),
                                       THERMOSTAT_ALIASES, 'off')
        barostat = self._canon_token(dynamics.get('barostat'),
                                     BAROSTAT_ALIASES, 'off')
        try:
            chain = max(int(dynamics.get('thermostat_chain', 1) or 1), 1)
        except (TypeError, ValueError):
            chain = 1

        num = self._ens_num

        def diag_get(key):
            return lambda ctx: num((ctx.get('diag') or {}).get(key))

        def t_inst(ctx):
            ndof = num((ctx.get('diag') or {}).get('ndof'))
            ekin = num(ctx.get('ekin'))
            if not np.isfinite(ndof) or ndof <= 0 or not np.isfinite(ekin):
                return float('nan')
            return 2.0 * ekin / (ndof * KELVIN_TO_HARTREE)

        columns = [
            ('T_inst(K)', t_inst),
            ('T_target(K)',
             lambda ctx: num((ctx.get('dynamics') or {}).get('temperature'))),
            ('N_df', diag_get('ndof')),
            ('Ekin(Ha)', lambda ctx: num(ctx.get('ekin'))),
            ('V(A^3)', lambda ctx: num((ctx.get('system') or {}).get('volume'))),
        ]

        if thermostat in ('nose-hoover', 'nose-hoover-chain'):
            columns.append(('scale_pre', diag_get('scale_pre')))
            columns.append(('scale_post', diag_get('scale_post')))
            for k in range(chain):
                columns.append(('Q%d' % (k + 1), self._slot('diag', 'q', k)))
            for k in range(chain):
                columns.append(('xi%d' % (k + 1),
                                self._slot('system', 'thermostat_dof', k)))
            for k in range(chain):
                columns.append(('peta%d' % (k + 1),
                                self._slot('system', 'thermostat_dof',
                                           chain + k)))
        elif thermostat == 'csvr':
            columns.extend([
                ('alpha', diag_get('csvr_alpha')),
                ('scale_post', diag_get('scale_post')),
                ('ratio', diag_get('csvr_ratio')),
                ('c', diag_get('csvr_c')),
                ('r1', diag_get('csvr_r1')),
                ('s', diag_get('csvr_s')),
            ])

        def pressure_bar(key):
            def getter(ctx):
                value = (ctx.get('diag') or {}).get(key)
                if value is None:
                    return float('nan')
                flat = np.asarray(value, dtype=np.float64).reshape(-1)
                if flat.size < 9:
                    return float('nan')
                iso = -float(flat[:9].reshape(3, 3).trace()) / 3.0 \
                    if key == 'p_vir' \
                    else float(flat[:9].reshape(3, 3).trace()) / 3.0
                return au_pressure_to_bar(iso)
            return getter

        if barostat == 'berendsen':
            columns.extend([
                ('Pkin_bar', pressure_bar('p_kin')),
                ('Pvir_bar', pressure_bar('p_vir')),
                ('Pint_bar', pressure_bar('p_int')),
                ('Pext_bar',
                 lambda ctx: num((ctx.get('dynamics') or {}).get('pressure'))),
            ])
            for i, comp in enumerate(self._MATRIX_COMPONENTS):
                columns.append(('mu_%s' % comp,
                                self._slot('diag', 'mu', i)))
        elif barostat == 'scr':
            # The stochastic barostat's record is the rescaling matrix it
            # applied plus the amplitude of its noise: the correctness
            # claim is a variance, so both halves have to be readable
            # from the run directory, and an energy drift is not defined
            # for this integrator.
            columns.extend([
                ('Pkin_bar', pressure_bar('p_kin')),
                ('Pvir_bar', pressure_bar('p_vir')),
                ('Pint_bar', pressure_bar('p_int')),
                ('Pext_bar',
                 lambda ctx: num((ctx.get('dynamics') or {}).get('pressure'))),
                ('scr_sigma', diag_get('scr_sigma')),
                ('scr_det', diag_get('scr_mu_det')),
            ])
            for i, comp in enumerate(self._MATRIX_COMPONENTS):
                columns.append(('mu_%s' % comp,
                                self._slot('diag', 'mu', i)))
        elif barostat in ('mttk', 'parrinello-rahman'):
            columns.extend([
                ('W', diag_get('w')),
                ('Pkin_bar', pressure_bar('p_kin')),
                ('Pvir_bar', pressure_bar('p_vir')),
                ('Pint_bar', pressure_bar('p_int')),
                ('Pext_bar',
                 lambda ctx: num((ctx.get('dynamics') or {}).get('pressure'))),
            ])
            for i, comp in enumerate(self._MATRIX_COMPONENTS):
                columns.append(('geps_%s' % comp,
                                self._slot('diag', 'g_eps', i)))
            for i, comp in enumerate(self._MATRIX_COMPONENTS):
                columns.append(('eps_%s' % comp,
                                self._slot('system', 'barostat_strain', i)))
            for i, comp in enumerate(self._MATRIX_COMPONENTS):
                columns.append(('peps_%s' % comp,
                                self._slot('system', 'barostat_momentum', i)))
            for i, comp in enumerate(self._MATRIX_COMPONENTS):
                columns.append(('veps_%s' % comp,
                                self._slot('system', 'barostat_velocity', i)))
            columns.append(('scale_cell_pre',
                            diag_get('scale_particle_pre')))
            columns.append(('scale_cell_post',
                            diag_get('scale_particle_post')))
            if barostat == 'mttk':
                columns.append(('ke_bar', diag_get('ke_bar')))
                columns.append(('scale_bar', diag_get('scale_bar')))
                for k in range(chain):
                    columns.append(('Qbar%d' % (k + 1),
                                    self._slot('diag', 'q_bar', k)))
                for k in range(chain):
                    columns.append((
                        'xibar%d' % (k + 1),
                        self._slot('system', 'barostat_thermostat_dof', k)))
                for k in range(chain):
                    columns.append((
                        'petabar%d' % (k + 1),
                        self._slot('system', 'barostat_thermostat_dof',
                                   chain + k)))

        self._ensemble_col_cache = columns
        return columns

    def _ensemble_row(self, snapshot, state, system, ekin):
        """Assemble one md.ensemble row, or None for a non-ensemble run.

        The per-step diagnostics come from the live runtime record
        (state.runtime.ensemble, reset at the pre stage of every step)
        with the snapshot runtime projection as fallback; the persisted
        extended DOF (chain state, strain, momenta, velocities) are
        read from the snapshot system fields so the file and the
        checkpoint can never diverge.
        """
        dynamics = snapshot.get('dynamics') or {}
        ensemble = str(dynamics.get('ensemble') or 'nve').strip().lower()
        if ensemble not in ('nvt', 'npt'):
            return None

        diag = {}
        live = getattr(getattr(state, 'runtime', None), 'ensemble', None)
        if isinstance(live, dict) and live:
            diag = live
        else:
            frame_diag = (snapshot.get('runtime') or {}).get('ensemble')
            if isinstance(frame_diag, dict):
                diag = frame_diag

        ctx = {'diag': diag, 'system': system or {},
               'dynamics': dynamics, 'ekin': ekin}
        columns = self._ensemble_columns(dynamics)
        values = []
        for name, getter in columns:
            try:
                value = float(getter(ctx))
            except (TypeError, ValueError) as error:
                # Every column getter already answers a missing or
                # short diagnostic with NaN, so reaching here means the
                # read itself is broken. Report each column once.
                if name not in self._ensemble_read_warnings:
                    self._ensemble_read_warnings.add(name)
                    _LOGGER.warning(
                        'md.ensemble column %s could not be read (%s); '
                        'the column is written as NaN', name, error)
                value = float('nan')
            values.append(value)
        return [name for name, _ in columns], values

    def _write_ensemble(self, suffix, step, row):
        """Append one ensemble-trajectory row (time / planned columns).

        Values render as '%20.12f' (NaN for a missing diagnostic); the
        column header is written lazily on first creation, mirroring
        _write_stress, so an nve run leaves no file behind.
        """
        names, values = row
        path = self._path(suffix)
        new_file = not os.path.exists(path)
        text = '%20.2f' % (max(step - 1, 0) * self.dt)
        text += ''.join(self._ens_cell(v) for v in values)
        with open(path, 'a') as log:
            if new_file:
                head = '%20s' % 'time'
                head += ''.join('%20s' % n for n in names)
                log.write(head + '\n')
            log.write(text + '\n')

    # ---- Charge-frame helpers (ONIOM charge management) ----

    @staticmethod
    def _point_charges(system):
        """Whole-system [q, x, y, z] rows from a system dict.

        Reads the point charges the calculator synced into
        pmd.system.point_charges (row i is atom i); malformed short
        rows are dropped.
        """
        rows = []
        for row in (system.get('point_charges') or []):
            if len(row) >= 4:
                rows.append([float(row[0]), float(row[1]),
                             float(row[2]), float(row[3])])
        return rows

    def _write_charges(self, suffix, system, cmmt):
        """Append one global charge frame when charge management is on.

        The frame mirrors the legacy '{ncharge}\\n{cmmt}\\n{q x y z}'
        charge file, reusing the xyz-style frame writer. A run without
        charge management (or with no point charges) writes nothing, so
        its file matrix stays byte-identical.
        """
        if not self.charge_output:
            return
        charges = self._point_charges(system)
        if not charges:
            return
        self._write_frame(suffix, len(charges), cmmt,
                          print_charges(charges))

    @staticmethod
    def _region_charges(point_charges, region):
        """Point-charge rows whose atom index lies in one center region.

        ``point_charges`` is the whole-system array, so the region's
        atom indices select the rows directly (no coordinate match).
        Used for the per-center md.#.charges files, which scope to the
        center's high region like the per-center xyz files.
        """
        if not point_charges or not len(region):
            return []
        selected = []
        for index in region:
            if 0 <= int(index) < len(point_charges):
                selected.append(point_charges[int(index)])
        return selected
