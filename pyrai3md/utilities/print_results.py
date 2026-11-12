"""
======================================================================
This module formats calculation results into run-log text blocks for
PyRAI3MD. It reads plain dictionaries (the pmd model dump or a
history snapshot) and produces the per-step / final result section:
coordinates, background charges, electronic state, velocities,
electronic energy (with constraint and multiscale contributions),
kinetic energy, gradient (by state), NAC (by nac pair), SOC (by soc
pair), NN prediction error (received deviations judged against the
input thresholds), dipole moment, transition dipole moment (by
transition pair), thermodynamic properties (G, H, S) and the
whole-system stress tensor with its derived isotropic pressure.

Properties missing from the result (None or empty) are printed with
their section title and a 'Not computed' marker when the run
verbosity allows it (control verbose >= 3); lower verbosity levels
skip the missing-property sections entirely.

Content levels (cumulative): 0 prints the progress/error blocks (step
header, energies, constraint decomposition, contributions, prediction
error); 1 adds structure (coordinates, `&cell`, charges, state); 2 adds
the property arrays (population, velocities, gradient, NAC/SOC, moments,
thermodynamics, frequencies, normal modes, stress, ensemble); 3 adds the
hessian and every missing marker. The input header echo is not gated.

This module also owns the `opt_status.txt` lifecycle: the parsed record,
the reader, the writer classification and the refusal to truncate a
record another writer owns. The header/row format itself is
`utilities/opt_formatter.py`; the artifact name does not change.

Oct 24 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os
from collections import namedtuple

import numpy as np

from .opt_formatter import STATUS_ATOM_COLUMNS, STATUS_CELL_COLUMNS
from .structure_formatter import print_coord, print_charges
from .units import au_pressure_to_bar, KELVIN_TO_HARTREE

_LINE = '-' * 79
_NOT_COMPUTED = '  Not computed\n'

# Section titles a post-run audit has to find in <title>.log. They live here
# rather than being retyped by whoever reads the log back, because a drift in
# one of these strings silently turns such a reader into one that reports
# "nothing to compare" over a log that is in fact full of the values.
SECTION_ENERGIES = '&electronic energy in Hartree'
SECTION_KINETIC = '&kinetic energy in Hartree'
SECTION_VELOCITIES = '&velocities in Bohr/au'
SECTION_CONTRIBUTIONS = '&energy contributions (constraint/multiscale)'
# The force-constant family. These are printed only when the run produced them,
# never as `Not computed`: the result block is also the per-step block of a
# dynamics run, and a section that appears on every one of those steps to say
# nothing would change the log of every unrelated run that comes through here.
SECTION_FREQUENCIES = '&vibrational frequencies in cm-1'
SECTION_NORMAL_MODES = '&normal modes in Bohr (normalized to sum m x^2 = 1)'
SECTION_HESSIAN = '&hessian in Hartree/Bohr^2'
SECTION_THERMO_CORRECTIONS = '&thermodynamic corrections'
SECTION_CONSTRAINT_GEOMETRY = '&constraint geometry in Bohr and radian'


def _is_empty(value):
    """Check whether a result value is missing or empty.

    Parameters
    ----------
    value : any
        Result value to inspect.

    Returns
    -------
    bool
        True for None, empty sequences and zero-size arrays.
    """
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    try:
        return np.asarray(value).size == 0
    except (TypeError, ValueError):
        return False


def _section(title, body):
    """Wrap a body text into a titled log section.

    Parameters
    ----------
    title : str
        Section title line.
    body : str
        Section content.

    Returns
    -------
    str
        Formatted section string.
    """
    return '\n  %s\n%s\n%s%s\n' % (title, _LINE, body, _LINE)


def _not_computed(title):
    """Build a section whose property was not computed.

    Parameters
    ----------
    title : str
        Section title line.

    Returns
    -------
    str
        Section string with the 'Not computed' marker.
    """
    return _section(title, _NOT_COMPUTED)


def _valid_pairs(pairs):
    """Filter a pair map down to valid [s1, s2] entries.

    Schema defaults such as [[]] are dropped; the returned indices
    stay 0-based as stored in the system model.

    Parameters
    ----------
    pairs : list
        Pair map from the system schema.

    Returns
    -------
    list
        Valid (s1, s2) pairs.
    """
    valid = []
    for pair in pairs or []:
        if len(pair) == 2:
            valid.append(pair)
    return valid


def format_ml_report(results, system=None):
    """Build the report of a training or prediction record, or ''.

    The caller (RunLogger) decides this from the result keys, so the
    generic property formatter does not branch on the jobtype.

    Parameters
    ----------
    results : dict
        Result dictionary written by the skill.
    system : dict or None
        System section, for the counts the report heads with.

    Returns
    -------
    str
        The training/prediction report, or '' for a property run.
    """
    results = results or {}
    if not _is_empty(results.get('train_errors')):
        return _format_nn_training(results, system)
    if not _is_empty(results.get('predict_errors')):
        return _format_nn_prediction(results, system)
    return ''


def format_result_block(system, results, step=None, runtime=None,
                        thresholds=None, verbose=2, centers=None,
                        dynamics=None, include_system=True,
                        record_report=None):
    """Format one calculation result block for the run log.

    Parameters
    ----------
    system : dict
        System section (pmd.system model dump or snapshot entry).
    results : dict
        Result dictionary written by the skills.
    step : int or None
        Iteration step number; the step header is added when given.
    runtime : dict or None
        Runtime section used as the kinetic-energy fallback and
        carrying the per-step ensemble diagnostics.
    thresholds : dict or None
        Deviation thresholds keyed by the err_* result keys; entries
        carrying a threshold add the exceeded TRUE/FALSE judgment to
        the prediction error section.
    verbose : int
        Control verbosity of the run; the 'Not computed' sections
        of missing properties are included from verbose 3 upward
        and skipped on lower levels.
    centers : list of dict or None
        Per-center active state, population and region charge / spin
        multiplicity (see snapshot_centers); when given, the
        electronic-state, population and region charge & spin sections
        are rendered per center instead of from the global
        system/results (a multi-center run has no single global active
        state).
    dynamics : dict or None
        Dynamics section (pmd.dynamics model dump or snapshot entry);
        carries the ensemble/thermostat/barostat keywords for the
        '&ensemble' section.
    include_system : bool
        Whether to print the system and property sections.
    record_report : str or None
        A ready-made training/prediction report. When given it stands in
        place of the property sections, which a record-only run took none
        of; the caller decides that from the result keys. Adaptive
        sampling evaluates many structures, so the single parent-state
        geometry is not a product of the run; its result block carries
        only the `&adaptive sampling summary`.

    Returns
    -------
    str
        Formatted result block.
    """
    system = system or {}
    results = results or {}
    runtime = runtime or {}
    atoms = system.get('atoms', [])
    level = int(verbose)
    show_missing = level >= 3

    block = ''
    if step is not None:
        block += '\n*%s*\n*  Step %8d\n*%s*\n' % ('-' * 77, step, '-' * 77)

    if not include_system:
        return block + _format_adaptive(results, dynamics)

    if level >= 1:
        view = _multiscale_view(system, results)
        if view is None:
            block += _format_coordinates(atoms, system.get('coordinates'),
                                         show_missing)
            block += _format_cell(system, show_missing)
            block += _format_point_charges(
                atoms, system.get('point_charges'), show_missing=show_missing)
        else:
            block += _format_coordinates(
                atoms, system.get('coordinates'), show_missing,
                indices=view['coordinates'], label=view['label'])
            block += _format_cell(system, show_missing)
            if view['scheme'] == 'qmmm':
                if view['embedding']:
                    store = system.get('point_charges')
                    if centers:
                        for index, center in enumerate(centers):
                            qm = [int(value) for value in
                                  (center.get('high_indices') or [])]
                            block += _format_point_charges(
                                atoms, store, indices=qm,
                                title='&qmmm 0-order multipole charges in e '
                                      '(atom, q): center %d' % (index + 1))
                    else:
                        block += _format_point_charges(
                            atoms, store,
                            indices=view['charges'],
                            title='&qmmm 0-order multipole charges in e '
                                  '(atom, q)')
                    block += _format_qmmm_external_potential(
                        results.get('qmmm_components'))
            else:
                store = system.get('point_charges')
                outer = results.get('oniom_outer_charges')
                if centers:
                    for index, center in enumerate(centers):
                        high = [int(value) for value in
                                (center.get('high_indices') or [])]
                        block += _format_point_charges(
                            atoms, store, indices=high,
                            title='&center %d highlevel point charges in e '
                                  '(atom, q)' % (index + 1))
                        mid = [int(value) for value in
                               (center.get('mid_indices') or [])]
                        if mid:
                            block += _format_point_charges(
                                atoms, store, indices=mid,
                                title='&center %d midlevel point charges in e '
                                      '(atom, q)' % (index + 1))
                        elif outer:
                            environment = [
                                atom for atom in range(len(atoms))
                                if atom not in set(high)]
                            block += _format_point_charges(
                                atoms, outer, indices=environment,
                                title='&center %d embedding point charges in '
                                      'e (atom, q)' % (index + 1))
                else:
                    block += _format_point_charges(
                        atoms, store, indices=view['charges'],
                        title='&highlevel point charges in e (atom, q)')
                    if view['mid'] is not None:
                        block += _format_point_charges(
                            atoms, store, indices=view['mid'],
                            title='&midlevel point charges in e (atom, q)')
                    elif outer:
                        environment = [
                            atom for atom in range(len(atoms))
                            if atom not in set(view['charges'])]
                        block += _format_point_charges(
                            atoms, outer, indices=environment,
                            title='&embedding point charges in e (atom, q)')
        # A background embedding is one kind, never both: the file declares
        # which, so the log prints whichever arrived. If a result somehow
        # carries both, neither is printed and the conflict is named - two
        # blocks would read as a calculation embedded in two things at once.
        charges = results.get('external_charges')
        field = results.get('external_potential')
        if charges and field:
            block += _section(
                '&external embedding',
                '  both an external charge set and an external field are '
                'present; neither is printed\n')
        elif charges:
            block += _format_external_charges(charges, show_missing=False)
        elif field:
            block += _format_external_potential(field, show_missing=False)

    # A record-only run (training or test-set serving) evaluated no
    # geometry, so it has no property sections to print. The caller passes
    # the report it built; this function never guesses the jobtype.
    if record_report:
        return block + record_report

    if centers:
        if level >= 1:
            block += _format_state_centers(centers, show_missing)
            block += _format_center_regions(centers)
        if level >= 2:
            block += _format_population_centers(centers)
    else:
        if level >= 1:
            block += _format_state(system.get('current_state'), show_missing)
        if level >= 2:
            block += _format_population(results.get('population'),
                                        results.get('final_population'))
    # Energy and the numbers that explain it are the run's progress, not a
    # property listing: they stay at level 0.
    block += _format_energy(results.get('energy'),
                            results.get('energy_contributions'),
                            show_missing)
    block += _format_constraint_energy(results.get('energy'),
                                       results.get('energy_origin'),
                                       results.get('energy_contributions'))
    block += _format_composite(results.get('composite_terms'),
                               results.get('composite_total'))
    block += _format_qmmm_decomposition(results.get('qmmm_components'))
    block += _format_qmmm_charge_iteration(
        results.get('qmmm_charge_iteration'))
    block += _format_kinetic(results.get('kinetic_energy'),
                             runtime.get('kinetic_energy'),
                             show_missing)
    block += _format_prediction_error(results, thresholds)
    if level >= 2:
        block += _format_velocities(atoms, system.get('velocities'),
                                    show_missing)
        block += _format_gradient(atoms, results.get('gradient'),
                                  show_missing)
        center_couplings = bool(centers) and any(
            center.get('nac') is not None or center.get('nac_pair_map')
            or center.get('soc') is not None or center.get('soc_pair_map')
            for center in centers)
        if center_couplings:
            for index, center in enumerate(centers):
                high = [int(value) for value in
                        (center.get('high_indices') or [])]
                local_atoms = [atoms[atom] for atom in high
                               if atom < len(atoms)]
                prefix = 'center %d ' % (index + 1)
                block += _format_nac(
                    local_atoms, center.get('nac'),
                    center.get('nac_pair_map'), show_missing,
                    title_prefix=prefix)
                block += _format_soc(
                    center.get('soc'), center.get('soc_pair_map'),
                    show_missing, title_prefix=prefix)
        else:
            block += _format_nac(atoms, results.get('nac'),
                                 system.get('nac_pair_map'), show_missing)
            block += _format_soc(results.get('soc'),
                                 system.get('soc_pair_map'), show_missing)
        block += _format_dipole(results.get('dipole'), show_missing)
        block += _format_transition_dipole(
            results.get('transition_dipole'),
            system.get('transition_pair'), show_missing)
        block += _format_thermodynamics(results.get('free_energy'),
                                        results.get('enthalpy'),
                                        results.get('entropy'),
                                        show_missing)
        if (not _is_empty(results.get('free_energy'))
                and _is_empty(results.get('energy'))):
            # A frequency job that did not evaluate the reference surface
            # still reports thermal corrections; saying that here keeps a
            # reader from taking G/H/S as absolute.
            block += _section(
                '&thermodynamic reference',
                '  The reference geometry carries no electronic energy in '
                'this result; the G/H/S rows above are the thermal '
                'corrections alone.\n')
        block += _format_frequencies(results.get('frequencies'),
                                     results.get('hessian'),
                                     results.get('hessian_atoms'),
                                     results.get('reduced_mass'))
        block += _format_normal_modes(results.get('norm_mode'), atoms, level)
        block += _format_thermo_corrections(
            results.get('zero_point_energy'),
            results.get('internal_energy'),
            results.get('heat_capacity_cv'),
            results.get('heat_capacity_cp'))
        block += _format_stress(results.get('stress_tensor'),
                                system.get('current_state'), show_missing)
        block += _format_ensemble(dynamics, system, runtime, show_missing)
        block += _format_geometry_track(results.get('geometry_track'))
    if level >= 3:
        block += _format_hessian(results.get('hessian'), level)

    block += _format_adaptive(results, dynamics)
    return block


def _format_coordinates(atoms, coordinates, show_missing=True,
                        indices=None, label=''):
    """Format the coordinates section.

    ``indices`` selects a region (ONIOM highlevel, QMMM QM region); None
    keeps the whole system. ``label`` names the region in the title.
    """
    title = '&coordinates in Angstrom' + (
        ' (%s)' % label if label else '')
    if not atoms or _is_empty(coordinates) or len(atoms) != len(coordinates):
        return _not_computed(title) if show_missing else ''
    if indices is not None:
        atoms = [atoms[int(i)] for i in indices if 0 <= int(i) < len(atoms)]
        coordinates = [coordinates[int(i)] for i in indices
                       if 0 <= int(i) < len(coordinates)]
    if not atoms:
        return _not_computed(title) if show_missing else ''
    return _section(title, print_coord(atoms, coordinates))


def _format_cell(system, show_missing=True):
    """Format the periodic cell: lattice rows, cell parameters and volume.

    The box is structure, so it is printed from level 1 up. A non-periodic
    run has no lattice and gets no section (or the 'Not computed' marker at
    level 3), which is why the MD and optimization blocks of an isolated
    molecule stay exactly as they were.
    """
    title = '&cell in Angstrom'
    system = system or {}
    try:
        lattice = np.asarray(system.get('lattice_vector') or [],
                             dtype=float).reshape(-1, 3)
    except (TypeError, ValueError):
        lattice = np.zeros((0, 3))
    has_lattice = lattice.shape == (3, 3) and bool(np.any(lattice))
    try:
        cell6 = [float(value) for value in
                 np.asarray(system.get('cell') or []).reshape(-1)]
    except (TypeError, ValueError):
        cell6 = []
    if not has_lattice and len(cell6) != 6:
        return _not_computed(title) if show_missing else ''

    body = ''
    if has_lattice:
        for label, row in zip(('a:', 'b:', 'c:'), lattice):
            body += '  %-4s' % label + ''.join(
                '%20.12f' % float(value) for value in row) + '\n'
    if len(cell6) == 6:
        body += '  %-22s' % 'a b c (A):' + ''.join(
            '%16.8f' % float(value) for value in cell6[:3]) + '\n'
        body += '  %-22s' % 'alpha beta gamma (deg):' + ''.join(
            '%16.8f' % float(value) for value in cell6[3:]) + '\n'
    volume = system.get('volume')
    if volume is None and has_lattice:
        volume = abs(float(np.linalg.det(lattice)))
    if volume is not None:
        body += '  %-22s %20.8f\n' % ('volume (A^3):', float(volume))
    pbc = system.get('pbc') or []
    if pbc:
        body += '  %-22s %s\n' % (
            'pbc:', ' '.join(str(int(value)) for value in pbc))
    return _section(title, body)


def _flatten_indices(value):
    """Flatten a flat or per-center region list into sorted 0-based ints."""
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        return sorted({int(index) for group in value for index in group})
    return sorted({int(index) for index in value})


def _multiscale_view(system, results):
    """What a multiscale log may show, or None for a plain run.

    ONIOM: highlevel coordinates and charges always, midlevel charges
    only when embedding. QMMM: QM-region coordinates always, and the
    0-order multipole charges plus V/E only when embedding. MM
    coordinates/charges are never returned.
    """
    scheme = results.get('composite_scheme')
    if not scheme and isinstance(results.get('qmmm_components'), dict):
        scheme = 'qmmm'
    if scheme == 'qmmm':
        components = results.get('qmmm_components') or {}
        qm_atoms = [int(index) for index in
                    (components.get('qm_atoms') or [])]
        if not qm_atoms:
            return None
        embedding = str(components.get('embedding') or 'none') != 'none'
        return {'scheme': 'qmmm', 'coordinates': qm_atoms,
                'charges': qm_atoms if embedding else None,
                'mid': None, 'embedding': embedding,
                'label': 'QM region'}
    if scheme == 'oniom':
        highlevel = _flatten_indices(system.get('highlevel'))
        if not highlevel:
            return None
        midlevel = _flatten_indices(system.get('midlevel'))
        embedding = bool(system.get('embedding', False))
        return {'scheme': 'oniom', 'coordinates': highlevel,
                'charges': highlevel,
                'mid': midlevel if (embedding and midlevel) else None,
                'embedding': embedding, 'label': 'highlevel'}
    return None


def _format_point_charges(atoms, point_charges, indices=None,
                          title='&point charges in e (atom, q)',
                          show_missing=True):
    """Format per-atom charges as ``atom symbol q`` rows.

    ``point_charges`` is the whole-system array (row i is atom i);
    ``indices`` selects the region to print (ONIOM highlevel/midlevel,
    QMMM QM region). No placeholder is emitted when the array is empty.
    """
    if _is_empty(point_charges):
        return _not_completed_or_empty(title, show_missing)
    rows = []
    selected = (range(len(point_charges)) if indices is None else
                [int(index) for index in indices])
    for index in selected:
        if index < 0 or index >= len(point_charges):
            continue
        row = point_charges[index]
        if len(row) < 4:
            continue
        symbol = atoms[index] if index < len(atoms) else '?'
        rows.append('  %6d %-4s %18.10f\n'
                    % (index + 1, symbol, float(row[0])))
    if not rows:
        return _not_completed_or_empty(title, show_missing)
    return _section(title, ''.join(rows))


def _not_completed_or_empty(title, show_missing):
    """A charge block with no rows is omitted below level 3."""
    return _not_computed(title) if show_missing else ''


def _format_external_charges(rows,
                             title='&external charges in e (q x y z)',
                             show_missing=True):
    """Format arbitrary external charges read from a '.charge' file."""
    if _is_empty(rows):
        return _not_completed_or_empty(title, show_missing)
    return _section(title, print_charges(rows))


def _format_external_potential(rows,
                               title='&external field in au (atom, V Fx Fy Fz)',
                               show_missing=True):
    """Format an external electrostatic field read from a '.charge' file.

    The rows are the V/E block the calculation was embedded in, one per QM
    atom, in atomic units - the same numbers a QMMM field stage would supply,
    which is what makes a standalone run comparable with the composite one.
    """
    if _is_empty(rows):
        return _not_completed_or_empty(title, show_missing)
    body = ''
    for index, row in enumerate(rows, start=1):
        body += '  %4d %18.10f %18.10f %18.10f %18.10f\n' % (
            index, float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    return _section(title, body)


def _format_qmmm_decomposition(components, show_missing=True):
    """Format the QMMM energy decomposition (atomic units).

    Reads ``results['qmmm_components']``: QM energy per state, the
    classical term, its U_main/U_gas/qmmm_el split, and the total.
    """
    title = '&qmmm energy decomposition in Hartree'
    if not isinstance(components, dict):
        return ''
    qm_energy = components.get('qm_energy') or []
    if not qm_energy:
        return ''
    body = ''
    for state, value in enumerate(qm_energy):
        body += '  %-24s %18.10f\n' % (
            'E_QM state %d' % (state + 1), float(value))
    e_classic = float(components.get('e_classic', 0.0))
    body += '  %-24s %18.10f\n' % ('E_classic', e_classic)
    body += '  %-24s %18.10f\n' % ('U_main', float(
        components.get('u_main', 0.0)))
    body += '  %-24s %18.10f\n' % ('U_gas', float(
        components.get('u_gas', 0.0)))
    body += '  %-24s %18.10f\n' % ('qmmm_el', float(
        components.get('qmmm_el', 0.0)))
    for state, value in enumerate(qm_energy):
        body += '  %-24s %18.10f\n' % (
            'total state %d' % (state + 1), float(value) + e_classic)
    if int(components.get('n_center') or 1) > 1:
        # One center's numbers cannot be recovered from the sums: which
        # center contributed which energy, and which one still moved when
        # the iteration stopped, are only answerable per center.
        for center in components.get('centers') or []:
            label = 'center %d' % (int(center.get('center_index', 0)) + 1)
            for state, value in enumerate(center.get('qm_energy') or []):
                body += '  %-24s %18.10f\n' % (
                    '%s E_QM state %d' % (label, state + 1), float(value))
            body += '  %-24s %18.10f\n' % (
                '%s U_gas' % label, float(center.get('u_gas', 0.0) or 0.0))
    body += '  %-24s %s\n' % (
        'embedding', str(components.get('embedding') or 'none'))
    owns_field = components.get('qm_contains_field_gradient')
    if owns_field is not None and str(
            components.get('embedding') or 'none') != 'none':
        # An isolated QM run has no field term to assign. The classical
        # gradient always has dqmmm_el removed from its QM rows; this row
        # says who ends up carrying it.
        body += '  %-24s %s\n' % (
            'field gradient',
            'qm provider' if owns_field else 'classical side')
    return _section(title, body)


def _format_qmmm_external_potential(components):
    """Format the V/E rows handed to the QM stage (atomic units).

    One center prints one block over the whole QM set. With several
    centers the union array would hide which probe measured which row,
    and a reader checking a center's embedding needs its own rows, so
    each center is printed under its index.
    """
    if not isinstance(components, dict):
        return ''
    title = ('&qmmm external potential/field in a.u. '
             '(atom, V, Ex, Ey, Ez)')
    centers = components.get('centers') or []
    if len(centers) > 1:
        blocks = ''
        for center in centers:
            rows = center.get('external_potential')
            if _is_empty(rows):
                continue
            atoms = center.get('qm_atoms') or []
            body = ''
            for offset, row in enumerate(rows):
                if len(row) < 4:
                    continue
                atom = (int(atoms[offset]) + 1
                        if offset < len(atoms) else offset + 1)
                body += ('  %6d %18.10f %18.10f %18.10f %18.10f\n'
                         % (atom, float(row[0]), float(row[1]),
                            float(row[2]), float(row[3])))
            if body:
                blocks += _section(
                    '%s: center %d'
                    % (title, int(center.get('center_index', 0)) + 1), body)
        return blocks
    rows = components.get('external_potential')
    if _is_empty(rows):
        return ''
    atoms = components.get('qm_atoms') or []
    body = ''
    for offset, row in enumerate(rows):
        if len(row) < 4:
            continue
        atom = int(atoms[offset]) + 1 if offset < len(atoms) else offset + 1
        body += ('  %6d %18.10f %18.10f %18.10f %18.10f\n'
                 % (atom, float(row[0]), float(row[1]),
                    float(row[2]), float(row[3])))
    return _section(title, body) if body else ''


def _format_qmmm_charge_iteration(iteration):
    """Format the 0-order charge self-consistency report of one evaluation.

    The round count is a trajectory variable: two frames of the same run
    can differ in how many charge rounds they solved, and a log that only
    says "converged" cannot tell a self-consistent run from the
    single-round non-self-consistent one.
    """
    if not isinstance(iteration, dict):
        return ''
    title = '&qmmm charge iteration'
    body = '  %-24s %s\n' % (
        'sc_charge', 'on' if iteration.get('sc_charge') else 'off')
    body += '  %-24s %d\n' % (
        'rounds run', int(iteration.get('rounds_run') or 0))
    body += '  %-24s %d\n' % ('maxiter', int(iteration.get('maxiter') or 0))
    # Only a run that damps says so: an undamped fixed point keeps the log it
    # always had, and a damped one says by how much (the residual line above is
    # still the raw move, not the damped one).
    if iteration.get('damping') is not None:
        body += '  %-24s %.4f\n' % ('damping',
                                     float(iteration.get('damping')))
    body += '  %-24s %.6e\n' % (
        'tolerance', float(iteration.get('tolerance') or 0.0))
    residuals = iteration.get('residuals') or []
    for round_index, value in enumerate(residuals, start=1):
        body += '  %-24s %.6e\n' % (
            'max |dq| round %d' % round_index, float(value))
    for center in iteration.get('centers') or []:
        label = 'center %d' % (int(center.get('center_index') or 0) + 1)
        values = center.get('residuals') or []
        if values:
            body += '  %-24s %s\n' % (
                label, ', '.join('%.6e' % float(value)
                                 for value in values))
    body += '  %-24s %s\n' % (
        'stopped by', str(iteration.get('stopped_by') or 'unknown'))
    return _section(title, body)


def _format_state(current_state, show_missing=True):
    """Format the active electronic state line."""
    title = '&electronic state'
    if current_state is None:
        return _not_computed(title) if show_missing else ''
    return _section(title, '  Current state: %d\n' % (int(current_state) + 1))


def _format_population(population, final_population):
    """Format the state population section.

    Reads the 'population' result key, falling back to the
    'final_population' key reported by dynamics skills (e.g. the
    qdm reference solver). The section is omitted entirely when no
    population data exists (single-point jobs).
    """
    title = '&state population'
    value = population if not _is_empty(population) else final_population
    if _is_empty(value):
        return ''

    body = ''
    for n, pop in enumerate(
            np.asarray(value, dtype=float).reshape(-1)):
        body += '  State %3d%28.16f\n' % (n + 1, pop)
    return _section(title, body)


def _format_state_centers(centers, show_missing=True):
    """Format the per-center active electronic state lines (R6 r6c).

    A multi-center run has no single global active state; each
    center's active local state is listed by its 1-based center index.
    """
    title = '&electronic state'
    if not centers:
        return _not_computed(title) if show_missing else ''
    body = ''
    for index, center in enumerate(centers):
        body += '  Center %3d  Current state: %d\n' % (
            index + 1, int(center.get('active', 0)) + 1)
    return _section(title, body)


def _format_population_centers(centers):
    """Format the per-center state population section (R6 r6c).

    One row per center: the 1-based center index followed by that
    center's per-state populations (the multi-center run-log layout).
    Omitted entirely when no center carries populations.
    """
    title = '&state population'
    rows = ''
    for index, center in enumerate(centers):
        population = center.get('population') or []
        if len(population) == 0:
            continue
        values = ''.join('%28.16f' % float(pop) for pop in population)
        rows += '  Center %3d%s\n' % (index + 1, values)
    if not rows:
        return ''
    return _section(title, rows)


def _format_center_regions(centers):
    """Format the multi-scale region charge / spin multiplicity section.

    A multi-center run splits the system into per-center regions; each
    region carries its own electronic-structure charge and spin
    multiplicity (the static CenterSpec definition), while the
    full-system low-level baseline L(real) uses the aggregate: the sum
    of the region charges and the paired-off total multiplicity
    (``(sum(mult_c - 1) % 2) + 1``, e.g. triplet + triplet -> singlet).
    Because a charged / open-shell aggregate may redistribute charge in
    a physically ambiguous way, this block is printed so the user can
    verify the region and full-system charge / spin assignment.

    One row per center (1-based) plus one L(real) aggregate row.
    Omitted entirely for a single-center frame.
    """
    if not centers:
        return ''
    from ..data.center import aggregate_system_multiplicity
    title = '&region charge & spin multiplicity'
    charges = [int(center.get('charge', 0)) for center in centers]
    multiplicities = [int(center.get('multiplicity', 1))
                      for center in centers]
    rows = ''
    for index, center in enumerate(centers):
        rows += '  Center %3d  charge %4d  multiplicity %4d\n' % (
            index + 1, int(center.get('charge', 0)),
            int(center.get('multiplicity', 1)))
    rows += '  L(real)    charge %4d  multiplicity %4d\n' % (
        sum(charges), aggregate_system_multiplicity(multiplicities))
    return _section(title, rows)


def snapshot_centers(snapshot):
    """Per-center active state + population for the run log (R6 r6c).

    Returns None for a single-center frame (a plain run, or C=1 with a
    region, keeps the legacy log block); for a multi-center frame it
    returns one entry per center with its pre-hop active local state,
    its population (the frame density real diagonal) and its region
    charge / spin multiplicity (the static CenterSpec electronic
    definition), so the log lists each center's state, populations and
    region charge/spin by center index.

    Parameters
    ----------
    snapshot : dict or None
        History frame (pmd model dump + structured center fields).

    Returns
    -------
    list of dict or None
        [{'active': int, 'population': [float, ...],
          'charge': int, 'multiplicity': int}, ...] for a
        multi-center frame; None otherwise.
    """
    snapshot = snapshot or {}
    centers_spec = snapshot.get('centers') or []
    if len(centers_spec) <= 1:
        return None
    outcome = snapshot.get('electronic_outcome') or {}
    nuclear = snapshot.get('nuclear_sample') or {}
    pre_config = tuple(nuclear.get('active_config_pre') or ())
    records = []
    for index in range(len(centers_spec)):
        carrier = outcome.get(index) or outcome.get(str(index)) or {}
        spec = centers_spec[index] or {}
        records.append({
            'active': (int(pre_config[index])
                       if index < len(pre_config) else 0),
            'population': _density_diagonal(carrier.get('density')),
            'charge': int(spec.get('charge', 0)),
            'multiplicity': int(spec.get('multiplicity', 1)),
            # Region indices travel with the center so a multi-center
            # ONIOM log can scope coordinates/charges per center.
            'high_indices': [int(value) for value in
                             (spec.get('high_indices') or [])],
            'mid_indices': [int(value) for value in
                            (spec.get('mid_indices') or [])],
            'nac': carrier.get('nac'),
            'soc': carrier.get('soc'),
            'nac_pair_map': list(
                (spec.get('electronic_spec') or {}).get('nac_pair_map')
                or []),
            'soc_pair_map': list(
                (spec.get('electronic_spec') or {}).get('soc_pair_map')
                or []),
        })
    return records


def _density_diagonal(density):
    """Real diagonal of a persisted [re, im] density matrix (S, S, 2)."""
    if density is None:
        return []
    array = np.asarray(density, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 2:
        array = array[..., 0]
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        return [float(value) for value in np.diag(array)]
    return []


def _format_velocities(atoms, velocities, show_missing=True):
    """Format the velocities section."""
    title = SECTION_VELOCITIES
    if not atoms or _is_empty(velocities) or len(atoms) != len(velocities):
        return _not_computed(title) if show_missing else ''
    return _section(title, print_coord(atoms, velocities))


def _format_energy(energy, contributions, show_missing=True):
    """Format the electronic energy section with optional contributions."""
    block = ''
    title = SECTION_ENERGIES
    if _is_empty(energy):
        if show_missing:
            block += _not_computed(title)
    else:
        body = ''
        for n, value in enumerate(np.asarray(energy, dtype=float).reshape(-1)):
            body += '  State %3d%28.16f\n' % (n + 1, value)
        block += _section(title, body)

    # Constraint and multiscale contributions (optional skill key)
    if isinstance(contributions, dict) and contributions:
        body = ''
        for name, value in contributions.items():
            body += '  %-20s %24.16f\n' % (name, float(value))
        block += _section(SECTION_CONTRIBUTIONS, body)

    return block


def _constraint_energy_lines(label, value):
    """One label, one line per electronic state, aligned with the rows above."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return '  %-18s %24.16f\n' % (label, float(array[0]))
    return ''.join('  %-18s State %3d %13.16f\n' % (label, n + 1, float(value))
                   for n, value in enumerate(array))


def _format_constraint_energy(energy, origin, contributions):
    """Format the `&constraint` energy decomposition of one evaluation.

    Three kinds of row, in the order a reader adds them up: the energy the
    engine returned (`origin pot`), one row per declared constraint named by its
    own id as the input wrote it (bias1, wall2, ...), and the potential the
    integrator actually evolves, which is their sum and the number the
    trajectory's `Epot` column carries (D-18).

    Rendered only for a run whose plan was evaluated: an unconstrained
    calculation has no origin row to show and gets no section, while a run held
    by geometry alone has one, and its answer is that the two numbers are equal.
    """
    if origin is None:
        return ''
    rows = [(str(name).split('.')[-1], value)
            for name, value in (contributions or {}).items()
            if str(name).startswith('constraint.')]
    body = _constraint_energy_lines('origin pot', origin)
    for name, value in rows:
        body += '  %-18s %24.16f\n' % (name, float(value))
    body += _constraint_energy_lines('Epot', energy)
    return _section('&constraint energies in Hartree', body)


def format_constraint_geometry(split):
    """Format the `&constraint` geometry record of one optimization step.

    The energy decomposition above answers what a constraint cost; this answers
    what it held. One row per declared condition, in the order the input wrote
    them: the target it was bound to, the value at the geometry this step's
    gradient was computed on, the residual between them, and the multiplier,
    which is the force the condition absorbs. A constrained stationary point is
    one where the residual is the solve's and the multiplier is whatever the
    surface demands, so the two numbers a threshold can never reach are exactly
    the two printed here, and the criteria are judged on what is left over.

    Lengths are Bohr and angles radian, the units the constraint core works in;
    the title says so, the way the energy section says Hartree. The trailing
    counters are the position solve's own reading of the step.

    Parameters
    ----------
    split : dict or None
        The record from `workflow.constraints.runtime.constraint_split`. None,
        or a run that declares no condition, writes nothing.

    Returns
    -------
    str
        The section text, or an empty string.
    """
    if not split:
        return ''
    body = ('  %-14s %-9s %-11s %20s %20s %20s %20s\n'
            % ('', 'kind', 'atoms', 'target', 'value', 'residual',
               'multiplier'))
    for row in split.get('conditions') or []:
        body += '  %-14s %-9s %-11s %20.10f %20.10f %20.10e %20.10e\n' % (
            str(row.get('id', ''))[:14],
            str(row.get('kind', ''))[:9],
            '-'.join(str(int(atom)) for atom in row.get('atoms', []))[:11],
            float(row.get('target', 0.0)),
            float(row.get('value', 0.0)),
            float(row.get('residual', 0.0)),
            float(row.get('multiplier', 0.0)))
    projection = split.get('projection') or {}
    counters = [('condition rank', split.get('rank', 0)),
                ('frozen components', split.get('frozen_components', 0)),
                ('restored components', projection.get('restored_components',
                                                        0)),
                ('solve iterations', projection.get('iterations', 0)),
                ('max residual', projection.get('max_residual', 0.0))]
    body += '\n'
    for label, value in counters:
        body += '  %-22s %24.16f\n' % (label, float(value))
    return _section(SECTION_CONSTRAINT_GEOMETRY, body)


def _state_rows(value):
    """A result array as (nstate, n) rows, whatever shape it arrived in."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _format_frequencies(frequencies, hessian, mobile=None,
                        reduced_mass=None):
    """Format the vibrational frequencies of one force-constant analysis.

    A negative value is an imaginary mode and is marked as such: the sign is
    how every backend in this package reports the difference, and a reader who
    sees only the number cannot tell a soft vibration from a transition state.

    The matrix's dimensions are printed because they are the one fact that tells
    a partial Hessian apart from a whole one at a glance; the mobile atoms, when
    the run recorded them, are numbered the way the input numbers them.
    """
    if _is_empty(frequencies):
        return ''
    body = ''
    rows = _state_rows(frequencies)
    masses = None if _is_empty(reduced_mass) else _state_rows(reduced_mass)
    for state, row in enumerate(rows):
        for index, value in enumerate(row):
            label = 'imaginary' if value < 0.0 else ' ' * 9
            mass = ('' if masses is None or state >= masses.shape[0]
                    or index >= masses.shape[1]
                    else '   mu=%8.4f' % masses[state, index])
            body += ('  State %3d mode %4d  %14.4f cm-1  %s%s\n'
                     % (state + 1, index + 1, value, label, mass))
    if not _is_empty(hessian):
        matrix = np.asarray(hessian, dtype=float)
        shape = matrix.shape[1:] if matrix.ndim == 3 else matrix.shape
        body += '  force constants %dx%d' % (shape[0], shape[1])
        if mobile is not None and len(mobile):
            body += ' over mobile atoms %s' % ', '.join(
                str(int(atom) + 1) for atom in mobile)
        body += '\n'
    return _section(SECTION_FREQUENCIES, body)


def _format_normal_modes(norm_mode, atoms, verbose=2):
    """Format the normal modes, one block per mode, in the sampling units.

    Only above the default verbosity: three numbers per atom per mode is the
    largest block a small molecule can print, and the record a reader wants is
    the file, not the log.
    """
    if _is_empty(norm_mode) or int(verbose) < 2:
        return ''
    array = np.asarray(norm_mode, dtype=float)
    if array.ndim == 3:
        array = array.reshape(1, *array.shape)
    if array.ndim != 4:
        # A block that cannot be read as (state, mode, atom, xyz) is stated,
        # not raised: this function runs on the way out of a finished
        # calculation, and 2-D input -- the layout OpenQP's hess.json writes
        # before `openqp_skill` folds it, measured as (12, 18) -- used to
        # abort the whole result print with an IndexError inside the loop
        # below (iteration tmp_20261011_085800, openqp `*_frequency_input`).
        return _section(SECTION_NORMAL_MODES, '  unreadable shape %s; '
                        'expected (state, mode, atom, xyz)\n'
                        % (array.shape,))
    body = ''
    for state, block in enumerate(array):
        for index, mode in enumerate(block):
            body += '  State %3d mode %4d\n' % (state + 1, index + 1)
            body += print_coord(list(atoms or [])[:mode.shape[0]], mode)
    return _section(SECTION_NORMAL_MODES, body)


def _format_hessian(hessian, verbose=2):
    """Format the force-constant matrix, at the highest verbosity only."""
    if _is_empty(hessian) or int(verbose) < 3:
        return ''
    array = np.asarray(hessian, dtype=float)
    if array.ndim == 3:
        array = array[0]
    body = ''
    for row in array:
        body += ''.join('%16.8f' % value for value in row) + '\n'
    return _section(SECTION_HESSIAN, body)


def _format_thermo_corrections(zpe, internal_energy, cv, cp):
    """Format the corrections the G/H/S section does not carry.

    The existing `&thermodynamic properties` section prints G, H and S; these
    are the parts of the same sum a reader needs in order to check it: the zero
    point energy, the internal energy it was added to, and the two heat
    capacities, whose difference is the gas constant and can be verified on the
    spot.
    """
    parts = [('zero point energy', zpe, 'Hartree'),
             ('internal energy', internal_energy, 'Hartree'),
             ('heat capacity Cv', cv, 'Hartree/K'),
             ('heat capacity Cp', cp, 'Hartree/K')]
    present = [row for row in parts if not _is_empty(row[1])]
    if not present:
        return ''
    body = ''
    for label, value, unit in present:
        rows = np.asarray(value, dtype=float).reshape(-1)
        for state, item in enumerate(rows):
            body += '  %-22s State %2d %18.10f %s\n' % (
                label, state + 1, item, unit)
    return _section(SECTION_THERMO_CORRECTIONS, body)


def _format_composite(composite_terms, composite_total):
    """Format the &composite energy-decomposition section.

    Rendered only for a composite (multi-task) run, which carries the
    composite_terms breakdown; a single-task run has none and the
    section is omitted entirely (never a 'Not computed' placeholder).
    Each term occupies one line listing that term's per-state raw
    energies in order (a multi-state high-level provider prints every
    state on its line; a single-state correction prints one value),
    followed by a 'total' line listing the per-state ONIOM total energy
    (the active-combination U for a multi-center run). The terms are
    already ordered high->mid->low, per center, by the composite layer.

    Parameters
    ----------
    composite_terms : dict or None
        {'labels': [str], 'energies': [list of float],
        'contributions': [int]} from the composite assembly.
    composite_total : sequence of float or None
        Per-state ONIOM total energy (single value for C>1).

    Returns
    -------
    str
        Formatted section, or '' when there is no breakdown.
    """
    if not isinstance(composite_terms, dict):
        return ''
    labels = composite_terms.get('labels') or []
    energies = composite_terms.get('energies') or []
    if not labels:
        return ''
    title = '&composite energy decomposition in Hartree'
    width = max([len(str(label)) for label in labels] + [len('total')])
    body = ''
    for label, values in zip(labels, energies):
        row = ' '.join('%18.10f' % float(v) for v in (values or []))
        body += '  %-*s : %s\n' % (width, str(label), row)
    if not _is_empty(composite_total):
        total_values = np.asarray(composite_total, dtype=float).reshape(-1)
        row = ' '.join('%18.10f' % float(v) for v in total_values)
        body += '  %-*s : %s\n' % (width, 'total', row)
    return _section(title, body)


def _format_kinetic(kinetic_energy, runtime_kinetic, show_missing=True):
    """Format the kinetic energy section."""
    title = SECTION_KINETIC
    value = kinetic_energy
    if _is_empty(value):
        value = runtime_kinetic
    if _is_empty(value):
        return _not_computed(title) if show_missing else ''
    return _section(title, '  %28.16f\n' % float(value))


def _format_gradient(atoms, gradient, show_missing=True):
    """Format the gradient section sorted by electronic state."""
    if _is_empty(gradient):
        if not show_missing:
            return ''
        return _not_computed('&gradient in Hartree/Bohr')

    array = np.asarray(gradient, dtype=float)
    if array.ndim == 2:
        array = array.reshape((1,) + array.shape)

    block = ''
    for n, state_grad in enumerate(array):
        title = '&gradient state %3d in Hartree/Bohr' % (n + 1)
        if (not atoms or state_grad.shape != (len(atoms), 3)):
            if show_missing:
                block += _not_computed(title)
        else:
            block += _section(title, print_coord(atoms, state_grad))

    return block


def _format_nac(atoms, nac, nac_pair_map, show_missing=True,
                title_prefix=''):
    """Format the NAC section ordered by the nac pair map."""
    pairs = _valid_pairs(nac_pair_map)
    if not pairs:
        if _is_empty(nac):
            if not show_missing:
                return ''
            return _not_computed('&%snonadiabatic coupling in '
                                 'Hartree/Bohr' % title_prefix)
        return ''

    array = None if _is_empty(nac) else np.asarray(nac, dtype=float)

    block = ''
    for n, (s1, s2) in enumerate(pairs):
        title = '&%snonadiabatic coupling %3d - %3d in Hartree/Bohr' % (
            title_prefix, s1 + 1, s2 + 1)
        coupling = None
        if array is not None and n < array.shape[0]:
            coupling = array[n]
        if (coupling is None or not atoms
                or np.asarray(coupling).shape != (len(atoms), 3)):
            if show_missing:
                block += _not_computed(title)
        else:
            block += _section(title, print_coord(atoms, coupling))

    return block


def _format_soc(soc, soc_pair_map, show_missing=True, title_prefix=''):
    """Format the SOC section ordered by the soc pair map."""
    title = '&%sspin-orbit coupling in cm-1' % title_prefix
    pairs = _valid_pairs(soc_pair_map)
    if not pairs:
        if _is_empty(soc):
            return _not_computed(title) if show_missing else ''
        return ''

    values = None if _is_empty(soc) else np.asarray(soc, dtype=float).reshape(-1)

    body = ''
    for n, (s1, s2) in enumerate(pairs):
        if values is not None and n < values.size:
            body += '  <H>=%10.4f            %3d - %3d\n' % (
                values[n], s1 + 1, s2 + 1)
        elif show_missing:
            body += '  Not computed              %3d - %3d\n' % (
                s1 + 1, s2 + 1)

    if not body and not show_missing:
        return ''
    return _section(title, body)


def format_path_summary(status_rows, profile_points):
    """Format the final band/walker summary the path files carry.

    A NEB run writes `path_status.txt` and `path_profile.txt` but its main
    log used to stop at the optimization setup; a reader had to open the
    two side files to learn the final barrier or the image energies. The
    rows this takes are the optimizer's own `report()` and `path_points()`
    records, so the log and the files cannot disagree.
    """
    lines = []
    for row in status_rows or []:
        parts = []
        if row.get('step') is not None:
            parts.append('Step %d' % int(row['step']))
        if row.get('energy') is not None:
            parts.append('Energy %.10f' % float(row['energy']))
        if row.get('barrier') is not None:
            parts.append('Barrier %.10f' % float(row['barrier']))
        if parts:
            lines.append('  ' + '  '.join(parts))
    points = list(profile_points or [])
    if points:
        lines.append('  Profile:')
        for point in points:
            if point.get('energy') is None:
                continue
            role = str(point.get('role') or 'image')
            index = point.get('index')
            label = ('%d' % (int(index) + 1)) if index is not None else '-'
            lines.append('    %-3s %-10s %18.10f' % (
                label, role, float(point['energy'])))
    if not lines:
        return ''
    return _section('&path summary', '\n'.join(lines) + '\n')


_ADAPTIVE_CHANNELS = (
    ('err_energy', 'Energy'),
    ('err_grad', 'Gradient'),
    ('err_nac', 'NAC'),
    ('err_soc', 'SOC'),
)


def _check_status_summary(per_trajectory, key):
    """Collapse one per-trajectory data-check status for the log.

    A status the run switched off says nothing and is dropped; the rest
    collapse to the single shared value or to a count per value.
    """
    statuses = [result.get(key) for result in per_trajectory
                if result.get(key) not in (None, 'off')]
    if not statuses:
        return None
    if len(set(statuses)) == 1:
        return statuses[0]
    counts = {}
    for value in statuses:
        counts[value] = counts.get(value, 0) + 1
    return ', '.join('%s(%d)' % (value, counts[value])
                     for value in sorted(counts))


def format_adaptive_iteration(record, screen=None, thresholds=None,
                              training=None, system=None, gate=None):
    """Format one adaptive iteration for the run log.

    The block follows the iteration's actual order: the structures the
    previous iteration handed over and how many the reference verified,
    then this iteration's fit in the same `&nn training` grammar an
    independent training run uses, then the training-data quality gate
    when one is armed, then the trajectory screening and the sampling
    verdict. Iteration 1 has no handed-over data and no fit, so those
    two parts are absent; a run the gate stopped has no screening either.

    Parameters
    ----------
    record : dict
        Compact iteration record (iteration, received, verified, ntraj,
        completed, triggered, selected, reference_*, new_frames, ...).
    screen : dict or None
        Iteration screening record from records/adaptive.screen_iteration().
    thresholds : dict or None
        Deviation key -> maximum threshold of the iteration.
    training : dict or None
        The fit's result record; rendered through `_format_nn_training`.
    system : dict or None
        System counts the training report heads with.
    gate : dict or None
        Quality-gate record; rendered through `_format_validation_gate`.

    Returns
    -------
    str
        Titled iteration block.
    """
    record = record or {}
    if not screen:
        block = ''
        if int(record.get('iteration') or 1) > 1:
            block += _section('&data',
                              '  Received data: %d\n  Verified data: %d\n'
                              % (int(record.get('received') or 0),
                                 int(record.get('verified') or 0)))
        if training:
            block += _format_nn_training(training, system or {})
        if gate:
            block += _format_validation_gate(gate)
        body = ''
        body += '  Iteration:                  %s\n' % record.get(
            'iteration')
        body += '  External start:             %s\n' % (
            'yes' if record.get('external') else 'no')
        body += '  Reference ran:              %s\n' % bool(
            record.get('reference_ran'))
        body += '  Reference failed:           %s\n' % record.get(
            'reference_failed')
        body += '  New training frames:        %s (kept %s, pruned %s)\n' % (
            record.get('new_frames'), record.get('kept_frames', '--'),
            record.get('pruned_frames', 0))
        body += '  Status:                     %s\n' % (
            record.get('status') or 'not passed')
        return block + _section('&adaptive sampling', body)

    thresholds = thresholds or {}
    keys = [key for key, _label in _ADAPTIVE_CHANNELS
            if key in thresholds]
    if not keys:
        keys = [key for key, _label in _ADAPTIVE_CHANNELS
                if key in ('err_energy', 'err_grad')]
    labels = dict(_ADAPTIVE_CHANNELS)

    body = ''
    iteration = int(record.get('iteration') or 1)
    if iteration > 1:
        body += _section('&data',
                         '  Received data: %d\n  Verified data: %d\n'
                         % (int(record.get('received') or 0),
                            int(record.get('verified') or 0)))
        if training:
            body += _format_nn_training(training, system or {})
        if gate:
            body += _format_validation_gate(gate)
        body += '\n'
    body += '  &adaptive sampling\n'
    body += '%s\n' % ('-' * 55)
    body += '  Iteration:                  %s\n' % record.get('iteration')
    per_trajectory = screen.get('per_trajectory') or []
    for result in per_trajectory:
        errors = result.get('max_errors') or {}
        values = ' '.join(
            '%s: %8.4f' % (labels[key], float(errors.get(key, 0.0)))
            for key in keys)
        last_step = result.get('last_step')
        last_state = result.get('last_state')
        body += (
            '  Traj(ID: %6d Step: %8s State: %2s) => '
            'Geom(Sampled: %3d Collect: %3d Discard: %3d (Bad %3d) '
            'Refine: %3d) => MaxStd(%s)%s\n' % (
                int(result.get('traj_id', 0)) + 1,
                '-' if last_step is None else int(last_step),
                '-' if last_state is None else int(last_state),
                int(result.get('n_sampled', 0)),
                int(result.get('n_selected', 0)),
                int(result.get('n_discarded', 0)),
                int(result.get('n_bad', 0)),
                0,
                values,
                '' if result.get('triggered') else ' *'))
    checks = []
    for key, label in (('population_status', 'population'),
                       ('order_status', 'order')):
        summary = _check_status_summary(per_trajectory, key)
        if summary is not None:
            checks.append('%s=%s' % (label, summary))
    if checks:
        # The data checks run whether or not an analysis switch is on, so
        # their verdict belongs to the sampling block: a nosh run reports
        # population=not_applicable here even when no analysis block exists.
        body += '  Data checks:                %s\n' % ', '.join(checks)
    body += '\n'

    lengths = [int(result['last_step']) for result in per_trajectory
               if result.get('last_step') is not None]
    average = (sum(lengths) / float(len(lengths))) if lengths else 0.0
    production = [result for result in per_trajectory
                  if not result.get('exploration')]
    ground = sum(1 for result in production
                 if result.get('last_state') == 0)
    triggered = sum(1 for result in per_trajectory
                    if result.get('triggered'))
    body += '  Number of trajectories:     %-10s\n' % len(per_trajectory)
    body += '  Average length:             %-10.1f\n' % average
    body += '  Ground state:               %-10s\n' % ground
    if len(production) != len(per_trajectory):
        body += '  Exploration:                %-10s\n' % (
            len(per_trajectory) - len(production))
    body += '  Passed:                     %-10s\n' % int(
        screen.get('n_passed', 0))
    body += '  Completed:                  %-10s\n' % int(
        screen.get('n_completed', 0))
    body += '  Sampled trajectories:       %-10s\n' % triggered
    body += '  Sampled geometries:         %-10s\n' % int(
        screen.get('n_sampled', 0))
    body += '  Collected:                  %-10s\n' % int(
        screen.get('n_selected', 0))
    body += '  Discarded:                  %-10s\n' % (
        '%d (Bad %d)' % (int(screen.get('n_discarded', 0)),
                         int(screen.get('n_bad', 0))))
    body += '  Refinement:                 %-10s\n' % 0
    body += '\n'
    body += '  %-24s %10s %10s %8s\n' % (
        'Metrics', 'MaxStd', 'Threshold', 'Pass')
    for key in keys:
        value = float((screen.get('max_errors') or {}).get(key, 0.0))
        threshold = float(thresholds.get(key, 0.0))
        body += '  %-24s %10.4f %10.4f %8s\n' % (
            labels[key] + ':', value, threshold,
            'True' if value <= threshold else 'False')
    body += '%s\n' % ('-' * 55)
    body += '  Sampling status:            %s\n' % (
        'NOT PASSED' if screen.get('triggered') else 'PASS')
    return '\n' + body + '\n'


def _format_adaptive(results, dynamics=None):
    """Format the accumulated adaptive iterations of a finished run.

    Parameters
    ----------
    results : dict
        Final result dictionary; reads its 'adaptive' block.
    dynamics : dict or None
        Dynamics section; its 'step' is the configured trajectory
        length the per-iteration average is measured against.

    Returns
    -------
    str
        Summary block, or '' when the run was not adaptive.
    """
    block = (results or {}).get('adaptive')
    if not isinstance(block, dict):
        return ''
    iterations = block.get('iterations') or []
    body = ''
    body += '  Elapsed iteration:  %d\n' % int(
        block.get('n_iterations') or 0)
    body += '  Final convergence:  %s\n' % bool(block.get('converged'))
    body += '  Stop reason:        %s\n' % (block.get('reason') or '')
    if iterations:
        step = 0
        try:
            step = int((dynamics or {}).get('step') or 0)
        except (TypeError, ValueError):
            step = 0
        body += '  %-4s %6s %9s %8s\n' % (
            'Iter', 'New', 'Ground %', 'Length %')
        for row in iterations:
            average = float(row.get('average_length') or 0.0)
            percent = ('%.1f' % (100.0 * average / step)) if step else '-'
            ground_value = row.get('ground_percent')
            ground = ('%.1f' % float(ground_value)
                      if ground_value is not None else '--')
            # 'New' is what entered the dataset: a quality-gate round keeps
            # fewer frames than the reference verified.
            added = row.get('kept_frames', row.get('new_frames'))
            body += '  %-4s %6s %9s %8s\n' % (
                row.get('iteration'), added, ground, percent)
    return _section('&adaptive sampling summary', body)


def _format_prediction_error(results, thresholds):
    """Format the NN prediction error section.

    Lists every received err_* deviation of the step together with
    the threshold declared in the input system and the TRUE/FALSE
    judgment of the exceeded criterion (error > threshold). Keys
    without a threshold are listed with their value only; the whole
    section is omitted when the result carries no deviation keys
    (single-shot calculations).

    Parameters
    ----------
    results : dict
        Result dictionary of the recorded step.
    thresholds : dict or None
        Deviation thresholds keyed by the err_* result keys.

    Returns
    -------
    str
        Formatted section, or '' when no deviation key exists.
    """
    # Lazy import: the utilities package must stay importable
    # without the workflow package (no circular dependency).
    from ..workflow.records.deviation import DEVIATION_KEYS

    results = results or {}
    thresholds = thresholds or {}
    present = [key for key in DEVIATION_KEYS if key in results]
    if not present:
        return ''

    title = '&prediction error'
    body = ''
    for key in present:
        error = results.get(key)
        try:
            error_text = '%16.8f' % float(error)
        except (TypeError, ValueError):
            error_text = '%16s' % str(error)
        threshold = thresholds.get(key)
        try:
            exceeded = float(error) > float(threshold)
            body += '  %-18s%s  threshold %12.6f  Exceeded: %s\n' % (
                key, error_text, float(threshold),
                'TRUE' if exceeded else 'FALSE')
        except (TypeError, ValueError):
            body += '  %-18s%s\n' % (key, error_text)

    return _section(title, body)


# Channels the training report has columns for, in the order the original
# project printed them.
TRAINING_CHANNELS = ('energy', 'gradient', 'nac', 'soc')

# Ensemble folders in report order. A folder is a role: the energy family,
# the coupling family and the spin-orbit family train independently and
# each reports its own per-member error.
_TRAINING_ROLES = ('energy_grad', 'nac', 'soc')


def _format_nn_training(results, system=None):
    """Format the two sections of a model-training result.

    The numbers come from the skill's training record: the fit error of
    every ensemble member, in the unit the skill says it is in. A channel
    a folder does not produce gets a blank column rather than a zero --
    an untrained channel and a channel trained to nothing are different
    statements, and only the second one is a zero.

    Parameters
    ----------
    results : dict
        Result dictionary carrying the training record.
    system : dict or None
        System section, for the counts the original report headed with.

    Returns
    -------
    str
        The '&nn training' section followed by the error table.
    """
    results = results or {}
    system = system or {}
    errors = results.get('train_errors') or {}
    units = results.get('train_metric_units') or {}

    natom = int(system.get('natom') or len(system.get('atoms') or []))
    nstate = int(system.get('nstate') or 0)
    # The report describes what was fitted, not what the input declared an
    # interface for: an E+G fit has no coupling network even when the
    # molecule has states that could be coupled.
    trained = set(errors)
    nnac = len(system.get('nac_pair_map') or []) if 'nac' in trained else 0
    nsoc = len(system.get('soc_pair_map') or []) if 'soc' in trained else 0

    body = '  Number of atoms:  %-10dNumber of state:  %d\n' % (natom, nstate)
    body += '  Number of NAC:    %-10dNumber of SOC:    %d\n' % (nnac, nsoc)
    try:
        wall = '%.1f s' % float(results.get('train_wall_seconds'))
    except (TypeError, ValueError):
        wall = str(results.get('train_wall_seconds', ''))
    body += '  Training arm:     %-10sWall clock:       %s\n' % (
        str(results.get('train_arm', '')), wall)
    data = str(results.get('train_data') or '')
    if data:
        body += '  Training data:    %s\n' % data
    modeldir = str(results.get('train_modeldir') or '')
    if modeldir:
        body += '  Model folder:     %s\n' % modeldir
    fitlogs = results.get('train_fitlogs')
    for path in (fitlogs if fitlogs is not None else []):
        body += '  Fit log:          %s\n' % path
    block = _section('&nn training', body)

    header = '  %-20s' % ''
    unit_line = '  %-20s' % ''
    for channel in TRAINING_CHANNELS:
        header += '%14s' % channel
        unit_line += '%14s' % str(units.get(channel, ''))

    table = header + '\n' + unit_line + '\n'
    roles = [role for role in _TRAINING_ROLES if errors.get(role)]
    roles += sorted(set(errors) - set(roles))
    for role in roles:
        channels = errors.get(role) or {}
        count = max((len(np.atleast_1d(np.asarray(values)))
                     for values in channels.values() if values), default=0)
        for index in range(count):
            row = '  %-20s' % ('%s_v%d' % (role, index))
            for channel in TRAINING_CHANNELS:
                values = channels.get(channel)
                if values is None:
                    row += '%14s' % '--'
                    continue
                entries = np.atleast_1d(np.asarray(values, dtype=np.float64))
                if index >= entries.size:
                    row += '%14s' % '--'
                    continue
                row += '%14.8f' % float(entries[index])
            table += row + '\n'

    block += _section('&nn validation mean absolute error', table)
    block += _format_prediction_std(results)
    return block


def _format_prediction_std(results):
    """Format the post-fit ensemble spread on the training frames."""
    report = (results or {}).get('train_prediction_std')
    if not isinstance(report, dict) or not report:
        return ''
    channels = TRAINING_CHANNELS
    units = ('Hartree', 'Hartree/Bohr', 'Hartree/Bohr', 'cm-1')
    body = '  %-8s' % 'ndata'
    body += ''.join('%14s' % channel for channel in channels) + '\n'
    body += '  %-8s' % ''
    body += ''.join('%14s' % unit for unit in units) + '\n'
    body += '  %-8d' % int(report.get('ndata') or 0)
    for channel in channels:
        value = report.get(channel)
        body += '%14s' % ('--' if value is None else '%.6f' % float(value))
    return _section('&nn prediction max std', body + '\n')


def _format_validation_gate(gate):
    """Format the training-data quality gate of one adaptive iteration.

    The gate is printed whenever it was evaluated, including the round
    that passed on the first fit. A trial line names the frame that was
    removed, its robust z, the folder the retrained network landed in,
    the new MAE and whether the removal stayed.
    """
    if not isinstance(gate, dict) or gate.get('status') == 'off':
        return ''
    target = gate.get('target')
    mae = gate.get('mae')
    passed = (mae is not None and target is not None
              and float(mae) <= float(target))
    body = '  %-30s%s\n' % (
        'target energy MAE (eV):', _metric_text(target))
    body += '  %-30s%s%s\n' % (
        'validation energy MAE (eV):', _metric_text(mae),
        '' if passed else '   NOT PASSED')
    body += '  %-30s%d (kept %d, pruned %d)\n' % (
        'new frames:', int(gate.get('new_frames') or 0),
        int(gate.get('kept_frames') or 0), len(gate.get('pruned') or []))
    for attempt in gate.get('attempts') or []:
        if not int(attempt.get('round') or 0):
            continue
        # A check that removed nothing has no folder and no retrained MAE;
        # its outcome is the verdict and the reason lines.
        if not str(attempt.get('modeldir') or ''):
            continue
        decision = str(attempt.get('decision') or '')
        if decision == 'rejected':
            decision = 'rejected; frame stays'
        body += ('  round %-2d remove frame %-5s z=%-8s -> %s  MAE %s eV  '
                 '%s\n' % (
                     int(attempt.get('round') or 0),
                     attempt.get('pruned'),
                     '--' if attempt.get('pruned_z') is None
                     else '%.2f' % float(attempt['pruned_z']),
                     os.path.basename(str(attempt.get('modeldir') or '--')),
                     _metric_text(attempt.get('mae')), decision))
    body += '  %-30s%s\n' % ('verdict:', str(gate.get('status') or ''))
    reason = str(gate.get('reason') or '')
    if reason:
        body += '  %-30s%s\n' % ('reason:', reason)
    return _section('&nn validation gate', body)


def _metric_text(value):
    """Print a metric value, or '--' when it was not reported."""
    if value is None:
        return '--'
    try:
        return '%.6f' % float(value)
    except (TypeError, ValueError):
        return str(value)


# Statistics the test-set report prints, in the order a reader compares
# them: the average, the tail-sensitive average, the single worst
# component, and the ensemble's own disagreement.
_PREDICTION_STATISTICS = (
    ('mae', 'mean absolute error'),
    ('rmse', 'root mean square error'),
    ('max', 'maximum absolute error'),
    ('spread_max', 'ensemble spread'),
)


def _format_nn_prediction(results, system=None):
    """Format the sections of a model-prediction result.

    The numbers come from the skill's prediction record: how far the
    trained surface was from the reference values of the test file, one
    statistic at a time and in the unit the skill says it is in. A
    channel the profile does not serve prints no column, because a
    channel that was not computed and a channel computed as zero are
    different statements.

    The per-frame detail stays out of this report and in the raw files
    the record names: a table of a thousand frames is a file, not a log
    section, and the numbers printed here are derived from exactly those
    rows.

    Parameters
    ----------
    results : dict
        Result dictionary carrying the prediction record.
    system : dict or None
        System section, for the counts the report heads with.

    Returns
    -------
    str
        The analysis section followed by one table per statistic.
    """
    results = results or {}
    system = system or {}
    errors = results.get('predict_errors') or {}
    units = results.get('predict_metric_units') or {}

    natom = int(system.get('natom') or len(system.get('atoms') or []))
    nstate = int(system.get('nstate') or 0)
    try:
        wall = '%.1f s' % float(results.get('predict_wall_seconds'))
    except (TypeError, ValueError):
        wall = str(results.get('predict_wall_seconds', ''))

    body = '  Number of atoms:  %-10dNumber of state:  %d\n' % (natom, nstate)
    body += '  Test frames:    %-10dWall clock:       %s\n' % (
        int(results.get('predict_frames') or 0), wall)
    raw_files = results.get('predict_raw_files') or {}
    for channel in sorted(raw_files):
        body += '  Raw comparison: %-10s%s\n' % (channel, raw_files[channel])
    summary = results.get('predict_summary')
    if summary:
        body += '  Frame table:    %s\n' % summary
    block = _section('&nn prediction error analysis', body)

    channels = [channel for channel in TRAINING_CHANNELS
                if channel in errors]
    channels += sorted(set(errors) - set(channels))
    for statistic, title in _PREDICTION_STATISTICS:
        table = ('  %-20s' % '') + ''.join(
            '%16s' % channel for channel in channels) + '\n'
        table += ('  %-20s' % '') + ''.join(
            '%16s' % units.get(channel, '') for channel in channels) + '\n'
        row = '  %-20s' % 'value'
        for channel in channels:
            value = (errors.get(channel) or {}).get(statistic)
            try:
                row += '%16.8f' % float(value)
            except (TypeError, ValueError):
                row += '%16s' % '--'
        table += row + '\n'
        block += _section('&nn prediction %s' % title, table)

    return block


def _format_dipole(dipole, show_missing=True):
    """Format the dipole moment section."""
    title = '&dipole moment in Debye'
    if _is_empty(dipole):
        return _not_computed(title) if show_missing else ''

    array = np.asarray(dipole, dtype=float)
    if array.ndim == 1:
        array = array.reshape((1,) + array.shape)

    body = ''
    for n, moment in enumerate(array):
        if moment.size != 3:
            if show_missing:
                body += '  State %3d  Not computed\n' % (n + 1)
        else:
            x, y, z = moment
            body += '  State %3d%16.8f%16.8f%16.8f\n' % (n + 1, x, y, z)

    if not body and not show_missing:
        return ''
    return _section(title, body)


def _format_transition_dipole(transition_dipole, transition_pair,
                              show_missing=True):
    """Format the transition dipole section ordered by transition pairs."""
    pairs = _valid_pairs(transition_pair)
    if not pairs:
        if _is_empty(transition_dipole):
            if not show_missing:
                return ''
            return _not_computed('&transition dipole moment in Debye')
        return ''

    array = (None if _is_empty(transition_dipole)
             else np.asarray(transition_dipole, dtype=float))

    body = ''
    for n, (s1, s2) in enumerate(pairs):
        moment = None
        if array is not None and n < array.shape[0]:
            moment = array[n]
        if moment is None or np.asarray(moment).size != 3:
            if show_missing:
                body += '  Not computed              %3d - %3d\n' % (
                    s1 + 1, s2 + 1)
        else:
            x, y, z = np.asarray(moment, dtype=float).reshape(-1)
            body += '  <M>=%10.4f %10.4f %10.4f   %3d - %3d\n' % (
                x, y, z, s1 + 1, s2 + 1)

    if not body and not show_missing:
        return ''
    return _section('&transition dipole moment in Debye', body)


def _format_thermodynamics(free_energy, enthalpy, entropy,
                           show_missing=True):
    """Format the thermodynamic properties section (G, H, S)."""
    title = '&thermodynamic properties'
    if (_is_empty(free_energy) and _is_empty(enthalpy)
            and _is_empty(entropy)):
        return _not_computed(title) if show_missing else ''

    def _as_list(value):
        if _is_empty(value):
            return None
        return list(np.asarray(value, dtype=float).reshape(-1))

    g_list = _as_list(free_energy)
    h_list = _as_list(enthalpy)
    s_list = _as_list(entropy)
    nstate = max(len(x) for x in (g_list, h_list, s_list) if x is not None)

    body = '  %-8s%24s%24s%24s\n' % ('State', 'G (Hartree)',
                                      'H (Hartree)', 'S (Hartree/K)')
    for n in range(nstate):
        if show_missing:
            g = '%24.16f' % g_list[n] if g_list and n < len(g_list) else ' ' * 12 + 'Not computed'
            h = '%24.16f' % h_list[n] if h_list and n < len(h_list) else ' ' * 12 + 'Not computed'
            s = '%24.16f' % s_list[n] if s_list and n < len(s_list) else ' ' * 12 + 'Not computed'
        else:
            g = '%24.16f' % g_list[n] if g_list and n < len(g_list) else ' ' * 24
            h = '%24.16f' % h_list[n] if h_list and n < len(h_list) else ' ' * 24
            s = '%24.16f' % s_list[n] if s_list and n < len(s_list) else ' ' * 24
        body += '  %-8d%s%s%s\n' % (n + 1, g, h, s)

    return _section(title, body)


def _format_stress(stress_tensor, current_state, show_missing=True):
    """Format the whole-system stress tensor section (active state).

    Reads the (nstate, 3, 3) stress record the calculator writes back
    through the structural-property track and projects the active
    state's 3x3 block, mirroring PyRAIState.active_stress(). The
    tensor is stored in Hartree/Bohr^3 with the tensile-positive sign
    convention; the isotropic pressure P = -Tr(stress)/3 is derived and
    also reported in bar for readability. A backend that never filled
    stress_tensor yields the 'Not computed' marker (verbose >= 3) so
    non-NPT runs stay unaffected.

    Parameters
    ----------
    stress_tensor : list or None
        Whole-system stress record, shape (nstate, 3, 3) or (3, 3).
    current_state : int or None
        0-based active electronic state; None (or a multi-center run)
        falls back to the reference block at index 0.
    show_missing : bool
        Whether to emit the 'Not computed' marker when stress is absent.

    Returns
    -------
    str
        Formatted stress section string.
    """
    title = '&stress tensor in Hartree/Bohr^3'
    if _is_empty(stress_tensor):
        return _not_computed(title) if show_missing else ''

    array = np.asarray(stress_tensor, dtype=float)
    if array.ndim == 2:
        array = array.reshape((1,) + array.shape)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        return _not_computed(title) if show_missing else ''

    idx = int(current_state or 0)
    if idx < 0 or idx >= array.shape[0]:
        idx = 0
    tensor = array[idx]
    # An identically-zero block is the unfilled default, not a physical
    # result; treat it as unavailable so non-NPT runs stay unaffected.
    if not np.any(tensor):
        return _not_computed(title) if show_missing else ''

    body = ''
    for row in tensor:
        body += '  ' + ''.join('%20.12f' % float(v) for v in row) + '\n'
    pressure_au = -float(np.trace(tensor)) / 3.0
    body += '  %-28s%20.12f Hartree/Bohr^3\n' % (
        'Pressure (P = -Tr/3):', pressure_au)
    body += '  %-28s%20.4f bar\n' % ('', au_pressure_to_bar(pressure_au))
    return _section(title, body)

def _ens_float(value):
    """Coerce a diagnostic value to float (NaN when unusable)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _ens_iso_bar(matrix, sign=1.0):
    """Isotropic pressure (bar) of a 3x3 tensor in atomic units.

    sign=-1 converts the tensile-positive stress convention to the
    compressive-positive pressure convention used by md.stress.
    """
    arr = np.asarray(matrix if matrix is not None else [], dtype=float)
    if arr.size < 9:
        return float('nan')
    return au_pressure_to_bar(sign * float(np.trace(arr.reshape(3, 3))) / 3.0)


def _ens_matrix_rows(label, matrix):
    """Render a 3x3 atomic-unit matrix as three indented rows."""
    arr = np.asarray(matrix if matrix is not None else [], dtype=float)
    if arr.size != 9:
        return ''
    arr = arr.reshape(3, 3)
    text = '  %-28s\n' % label if label else ''
    for row in arr:
        text += '  ' + ''.join('%20.12f' % float(v) for v in row) + '\n'
    return text


def _format_geometry_track(rows):
    """Format the `&geometric parameter monitor` section (legacy GeomTracker).

    Rows come from `utilities/tracker.py::evaluate` through the md step's
    results, so the reading and the recorded frame describe one geometry.
    The section is written only when the run declared a tracked parameter; a
    run without one carries no rows and gains no section.

    Parameters
    ----------
    rows : list of dict or None
        One row per tracked object: kind, the atoms it names, the distance
        in Angstrom, its threshold and the `distance > threshold` judgment.

    Returns
    -------
    str
        Formatted section, or '' when no parameter was tracked.
    """
    rows = [row for row in (rows or []) if isinstance(row, dict)]
    if not rows:
        return ''
    body = '  %-6s %-24s %20s %20s %8s\n' % (
        'kind', 'atoms', 'distance (Angstrom)', 'threshold (Angstrom)',
        'exceeded')
    for row in rows:
        kind = str(row.get('kind', ''))
        if kind == 'frag':
            atoms = ' / '.join(_track_span(group)
                               for group in row.get('groups') or [])
        else:
            atoms = _track_span(row.get('atoms') or [])
        body += '  %-6s %-24s %20.4f %20.4f %8s\n' % (
            kind, atoms, float(row.get('distance', 0.0)),
            float(row.get('threshold', 0.0)),
            'True' if row.get('exceeded') else 'False')
    return _section('&geometric parameter monitor', body)


def _track_span(atoms):
    """A compact 1-based range for the monitor: [2, 5] prints as '2-5'."""
    values = [int(atom) for atom in atoms]
    if not values:
        return '-'
    if len(values) == 1:
        return str(values[0])
    return '%d-%d' % (values[0], values[-1])


def _format_ensemble(dynamics, system, runtime, show_missing=True):
    """Format the per-step ensemble (thermostat / barostat) section.

    Data sources mirror the md.ensemble trajectory file: the dynamics
    keywords (ensemble/thermostat/barostat and their targets), the
    per-step diagnostics recorded by workflow.integrator.ensemble on
    the runtime (scale factors, masses, pressure tensors, CSVR
    parameters), and the persisted extended DOF on the system fields
    (Nose-Hoover chain state, barostat strain / momentum / velocity).
    The section renders for nvt/npt runs only; an nve run yields the
    'Not computed' marker (verbose >= 3) so classical trajectories
    stay unaffected.

    Parameters
    ----------
    dynamics : dict or None
        Dynamics section of the snapshot (ensemble keywords).
    system : dict or None
        System section carrying the extended-ensemble DOF fields.
    runtime : dict or None
        Runtime projection carrying the 'ensemble' diagnostics of the
        step (and the kinetic-energy fallback).
    show_missing : bool
        Whether to emit the 'Not computed' marker for a non-ensemble
        run.

    Returns
    -------
    str
        Formatted ensemble section string.
    """
    title = '&ensemble'
    dynamics = dynamics or {}
    system = system or {}
    runtime = runtime or {}
    diag = runtime.get('ensemble') or {}
    ensemble = str(dynamics.get('ensemble') or 'nve').strip().lower()
    if ensemble not in ('nvt', 'npt'):
        return _not_computed(title) if show_missing else ''
    if not isinstance(diag, dict):
        diag = {}

    thermostat = str(dynamics.get('thermostat') or 'off').strip().lower()
    barostat = str(dynamics.get('barostat') or 'off').strip().lower()
    try:
        chain = max(int(dynamics.get('thermostat_chain') or 1), 1)
    except (TypeError, ValueError):
        chain = 1

    body = '  %-28s%s\n' % ('Ensemble:', ensemble)
    body += '  %-28s%s (M=%d, tau %.1f fs)\n' % (
        'Thermostat:', thermostat, chain,
        _ens_float(dynamics.get('thermostat_tau')))
    if ensemble == 'npt':
        body += '  %-28s%s (tau %.1f fs, %s)\n' % (
            'Barostat:', barostat,
            _ens_float(dynamics.get('barostat_tau')),
            str(dynamics.get('barostat_mode') or 'anisotropic'))
    body += '  %-28s%20.4f K\n' % (
        'T_target:', _ens_float(dynamics.get('temperature')))
    if ensemble == 'npt':
        body += '  %-28s%20.4f bar\n' % (
            'P_target:', _ens_float(dynamics.get('pressure')))

    ndof = diag.get('ndof')
    if ndof is not None:
        body += '  %-28s%20d\n' % ('N_df:', int(ndof))
    ke = diag.get('ke')
    if ke is None:
        ke = runtime.get('kinetic_energy')
    if ke is not None:
        body += '  %-28s%20.12f Hartree\n' % ('Ekin:', _ens_float(ke))
        if ndof is not None and _ens_float(ndof) > 0.0:
            t_inst = (2.0 * _ens_float(ke)
                      / (_ens_float(ndof) * KELVIN_TO_HARTREE))
            body += '  %-28s%20.4f K\n' % ('T_inst:', t_inst)
    if 'scale_pre' in diag or 'scale_post' in diag:
        body += '  %-28s%20.12f /%14.12f\n' % (
            'Velocity scale (pre/post):',
            _ens_float(diag.get('scale_pre')),
            _ens_float(diag.get('scale_post')))

    if thermostat in ('nose-hoover', 'nose-hoover-chain'):
        dof = np.asarray(system.get('thermostat_dof') or [],
                         dtype=float).reshape(-1)
        q = np.asarray(diag.get('q') or [], dtype=float).reshape(-1)
        if dof.size or q.size:
            body += '  Nose-Hoover chain (atomic units):\n'
            body += '  %-4s%24s%24s%24s\n' % ('k', 'Q', 'xi', 'p_eta')
            for k in range(chain):
                q_v = q[k] if k < q.size else float('nan')
                xi_v = dof[k] if k < dof.size else float('nan')
                p_v = (dof[chain + k] if chain + k < dof.size
                       else float('nan'))
                body += '  %-4d%24.12f%24.12f%24.12f\n' % (
                    k + 1, q_v, xi_v, p_v)
    elif thermostat == 'csvr':
        if any(key in diag for key in ('csvr_alpha', 'csvr_ratio')):
            body += '  %-28s%20.12f\n' % (
                "CSVR alpha (K'/K):", _ens_float(diag.get('csvr_alpha')))
            body += '  %-28s%20.12f\n' % (
                'CSVR ratio (K0/K):', _ens_float(diag.get('csvr_ratio')))
            body += '  %-28s%20.12f\n' % (
                'CSVR c (dt/tau):', _ens_float(diag.get('csvr_c')))

    if barostat == 'berendsen':
        if 'p_kin' in diag or 'mu' in diag:
            body += '  %-28s%20.4f bar\n' % (
                'P_kin:', _ens_iso_bar(diag.get('p_kin')))
            body += '  %-28s%20.4f bar\n' % (
                'P_vir:', _ens_iso_bar(diag.get('p_vir'), sign=-1.0))
            body += '  %-28s%20.4f bar\n' % (
                'P_int:', _ens_iso_bar(diag.get('p_int')))
        body += _ens_matrix_rows('Berendsen mu (atomic units):',
                                 diag.get('mu'))
    elif barostat == 'scr':
        # The record of a stochastic step is the matrix it applied and the
        # amplitude it drew from; there is no cell momentum to print, and
        # no energy drift to judge it by.
        if 'p_kin' in diag or 'mu' in diag:
            body += '  %-28s%20.4f bar\n' % (
                'P_kin:', _ens_iso_bar(diag.get('p_kin')))
            body += '  %-28s%20.4f bar\n' % (
                'P_vir:', _ens_iso_bar(diag.get('p_vir'), sign=-1.0))
            body += '  %-28s%20.4f bar\n' % (
                'P_int:', _ens_iso_bar(diag.get('p_int')))
        if 'scr_sigma' in diag:
            body += '  %-28s%20.12f\n' % (
                'SCR noise sigma:', _ens_float(diag.get('scr_sigma')))
        if 'scr_mu_det' in diag:
            body += '  %-28s%20.12f\n' % (
                'SCR mu determinant:',
                _ens_float(diag.get('scr_mu_det')))
        body += _ens_matrix_rows('SCR mu (atomic units):', diag.get('mu'))
    elif barostat in ('mttk', 'parrinello-rahman'):
        if 'w' in diag:
            body += '  %-28s%20.12f atomic units\n' % (
                'W (barostat mass):', _ens_float(diag.get('w')))
        if 'p_kin' in diag:
            body += '  %-28s%20.4f bar\n' % (
                'P_kin:', _ens_iso_bar(diag.get('p_kin')))
            body += '  %-28s%20.4f bar\n' % (
                'P_vir:', _ens_iso_bar(diag.get('p_vir'), sign=-1.0))
            p_int = diag.get('p_int')
            if p_int is None and diag.get('p_kin') is not None \
                    and diag.get('p_vir') is not None:
                p_int = (np.asarray(diag['p_kin'], dtype=float)
                         - np.asarray(diag['p_vir'], dtype=float))
            body += '  %-28s%20.4f bar\n' % ('P_int:', _ens_iso_bar(p_int))
        body += _ens_matrix_rows('G_eps (atomic units):',
                                 diag.get('g_eps'))
        body += _ens_matrix_rows('Strain eps:',
                                 system.get('barostat_strain'))
        body += _ens_matrix_rows('p_eps (atomic units):',
                                 system.get('barostat_momentum'))
        body += _ens_matrix_rows('v_eps (atomic units):',
                                 system.get('barostat_velocity'))
        if 'scale_particle_pre' in diag or 'scale_particle_post' in diag:
            body += '  %-28s%20.12f /%14.12f\n' % (
                'Cell scale (pre/post):',
                _ens_float(diag.get('scale_particle_pre')),
                _ens_float(diag.get('scale_particle_post')))
        if barostat == 'mttk':
            if 'ke_bar' in diag:
                body += '  %-28s%20.12f Hartree\n' % (
                    'Ekin (cell bath):', _ens_float(diag.get('ke_bar')))
                body += '  %-28s%20.12f\n' % (
                    'Cell bath scale:', _ens_float(diag.get('scale_bar')))
            bar_dof = np.asarray(
                system.get('barostat_thermostat_dof') or [],
                dtype=float).reshape(-1)
            q_bar = np.asarray(diag.get('q_bar') or [],
                               dtype=float).reshape(-1)
            if bar_dof.size or q_bar.size:
                body += '  Cell-momentum NHC chain (atomic units):\n'
                body += '  %-4s%24s%24s%24s\n' % ('k', 'Q', 'xi', 'p_eta')
                for k in range(chain):
                    q_v = q_bar[k] if k < q_bar.size else float('nan')
                    xi_v = bar_dof[k] if k < bar_dof.size else float('nan')
                    p_v = (bar_dof[chain + k]
                           if chain + k < bar_dof.size else float('nan'))
                    body += '  %-4d%24.12f%24.12f%24.12f\n' % (
                        k + 1, q_v, xi_v, p_v)

    volume = system.get('volume')
    if volume:
        body += '  %-28s%20.4f A^3\n' % ('Volume:', _ens_float(volume))
    cell = np.asarray(system.get('cell') or [], dtype=float).reshape(-1)
    if cell.size == 6 and np.any(cell):
        body += '  %-28s%s\n' % (
            'Cell (A, deg):',
            ' '.join('%12.6f' % v for v in cell))

    skip_labels = (('thermostat_skipped', 'thermostat step skipped'),
                   ('barostat_skipped', 'barostat step skipped'),
                   ('barostat_skipped_pre', 'barostat pre kick skipped'))
    for key, label in skip_labels:
        reason = diag.get(key)
        if reason:
            body += '  Note: %s (%s)\n' % (label, reason)

    return _section(title, body)

# ---- opt_status.txt read side ----
#
# The write format is `utilities/opt_formatter.py`; this half owns the
# persisted record: which writer produced it, the rows a reader can trust,
# and the guard that refuses to truncate a foreign table.

# Which writer produced a record.
WRITER_ATOM = 'fixed_cell'
WRITER_CELL = 'variable_cell'
WRITER_FOREIGN = 'engine'

OptStatusRecord = namedtuple(
    'OptStatusRecord',
    ['writer', 'columns', 'rows', 'separators', 'malformed', 'path'])
"""A parsed `opt_status.txt`.

writer
    `WRITER_ATOM`, `WRITER_CELL` or `WRITER_FOREIGN`.
columns
    The header tokens as written.
rows
    One dict per row whose token count matches the header, keyed by column
    name; values are floats where the field parses as one.
separators
    Lines that are only a run of dashes (an engine's rule under its title).
malformed
    One dict per non-empty line that is neither a matching row nor a
    separator, with its 1-based line number and token count.
path
    The file this record was read from.
"""


def read_status(path):
    """Read an `opt_status.txt` and report which table it is.

    The header/row shape is `utilities/opt_formatter.py`'s; this function
    reads it back and reports the writer.

    Parameters
    ----------
    path : str
        Path to the record.

    Returns
    -------
    OptStatusRecord
        Empty `rows` and `WRITER_FOREIGN` when the file holds fewer than
        two non-empty lines, so an unreadable record is reported as one
        rather than judged as a relaxation that took no steps.

    Raises
    ------
    OSError
        Propagated when the file cannot be opened; a missing record is the
        caller's finding, not this reader's.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
        lines = handler.read().splitlines()
    body = [(index, line) for index, line in enumerate(lines, start=1)
            if line.split()]
    columns = body[0][1].split() if body else []
    writer = classify_header(columns)
    if len(body) < 2:
        # A header with no steps is still this writer's table: a run that
        # opened its record and died before step one reads as zero steps,
        # not as a file the framework never wrote.
        return OptStatusRecord(writer, columns, [], 0,
                               [{'line': index, 'tokens': len(line.split())}
                                for index, line in body[1:]], path)
    rows = []
    separators = 0
    malformed = []
    for index, line in body[1:]:
        parts = line.split()
        if set(''.join(parts)) == {'-'}:
            separators += 1
            continue
        if len(parts) != len(columns):
            malformed.append({'line': index, 'tokens': len(parts)})
            continue
        rows.append({name: _as_number(value)
                     for name, value in zip(columns, parts)})
    return OptStatusRecord(writer, columns, rows, separators, malformed, path)


def classify_header(columns):
    """Which writer's table these header tokens name.

    Parameters
    ----------
    columns : list of str
        The header tokens, as read.

    Returns
    -------
    str
        `WRITER_ATOM`, `WRITER_CELL` or `WRITER_FOREIGN`.
    """
    # The prefix decides, not the whole line. A composite run appends one column
    # per energy term after the fixed ones, and those names belong to the
    # assembly rather than to this module, so a header with an intact fixed
    # prefix is this writer's table whatever follows it: the reader keys rows by
    # column name, and a row whose token count disagrees with the header is
    # reported as malformed instead of being dropped on a classification miss.
    atom = list(STATUS_ATOM_COLUMNS)
    cell = list(STATUS_CELL_COLUMNS)
    if list(columns)[:len(atom)] != atom:
        return WRITER_FOREIGN
    rest = list(columns)[len(atom):]
    if not rest or rest[:len(cell)] != cell:
        return WRITER_ATOM
    return WRITER_CELL


def assert_writable(path):
    """Refuse to truncate a record the framework's writer does not own.

    One directory has one owner for the plain name `opt_status.txt`. An engine
    writes its own convergence table under that name inside a calculation
    directory it keeps (openqp does), and a framework run truncating it would
    destroy the reading that engine's parser goes back for while leaving a
    header its own rows would not match.

    Parameters
    ----------
    path : str
        The record the caller is about to open for writing.

    Raises
    ------
    ValueError
        When the file exists and its header is not the framework table. A
        missing file, or one already written by this writer, is writable.
    """
    if not os.path.exists(path):
        return
    existing = read_status(path)
    if existing.writer == WRITER_FOREIGN:
        raise ValueError(
            'refusing to write %s: its header (%s) is not the framework '
            'table. A run that relaxes its own copy of this name has to own '
            'the directory; an engine that writes a convergence record here '
            '(openqp does) owns it until that run is archived elsewhere'
            % (path, ' '.join(existing.columns)))


def format_malformed(record):
    """Say what this reader refused to take, in as many words as there are.

    Parameters
    ----------
    record : OptStatusRecord
        A parsed record.

    Returns
    -------
    str
        Empty when every line matched the header; otherwise a clause naming
        the count, the header width, and the widths that were seen.
    """
    if not record.malformed:
        return ''
    seen = sorted({item['tokens'] for item in record.malformed})
    return (' %d line(s) do not match the %d-column header (widths seen: %s) '
            'and were not read as steps'
            % (len(record.malformed), len(record.columns),
               ', '.join(str(width) for width in seen)))


def _as_number(value):
    """Float when the field is a number, the text when it is not."""
    try:
        return float(value)
    except ValueError:
        return value


def _analysis_status(item):
    status = str(item.get('status', ''))
    reason = item.get('reason') or ''
    return status + ((' (%s)' % reason) if reason else '')


def _format_gap_item(item):
    if item.get('status') != 'ok':
        return _analysis_status(item)
    crossing = item.get('first_below_threshold')
    text = 'min %.4f Ha pair %s step %s' % (
        float(item.get('min_gap', 0.0)), item.get('min_pair'),
        item.get('min_step'))
    if crossing:
        text += '; first below at step %s' % crossing.get('step')
    else:
        text += '; none below %.4g Ha' % float(item.get('threshold', 0.0))
    text += '; %.2f below' % float(item.get('fraction_below_threshold', 0.0))
    return text


def _format_trace_item(item):
    if item.get('status') != 'ok':
        return _analysis_status(item)
    parts = []
    for key, label, spec in (
            ('min_gap', 'min_gap', '%.4f'),
            ('max_force', 'max_force', '%.4g'),
            ('max_energy_drift', 'max_drift', '%.3g'),
            ('min_distance', 'min_dist', '%.4f'),
            ('min_population', 'min_pop', '%.3f')):
        if item.get(key) is not None:
            parts.append(('%s ' + spec) % (label, float(item[key])))
    if item.get('min_distance_status'):
        parts.append(str(item['min_distance_status']))
    return ' '.join(parts) or 'ok'


def _format_pes_item(item):
    if item.get('status') != 'ok':
        return _analysis_status(item)
    candidates = item.get('candidates') or []
    residuals = [float(candidate['reconstruction']['rF'])
                 for candidate in candidates
                 if isinstance(candidate.get('reconstruction'), dict)
                 and candidate['reconstruction'].get('rF') is not None]
    curvatures = [float(candidate['curvature'])
                  for candidate in candidates
                  if candidate.get('curvature') is not None]
    energy_residual = [abs(float(candidate['energy_residual']))
                       for candidate in candidates
                       if candidate.get('energy_residual') is not None]
    parts = ['%d candidate(s)' % len(candidates)]
    if residuals:
        parts.append('rF %.2e..%.2e' % (min(residuals), max(residuals)))
    if curvatures:
        parts.append('kappa %.3f..%.3f'
                     % (min(curvatures), max(curvatures)))
    if energy_residual:
        parts.append('|rE|max %.2e' % max(energy_residual))
    return ', '.join(parts)


def format_analysis(report):
    """Render one round's adaptive analysis report.

    Parameters
    ----------
    report : dict or None
        The `analysis_iter-<k>.json` payload.

    Returns
    -------
    str
        The log block; empty when there is nothing to report.
    """
    if not report:
        return ''
    plan = report.get('plan') or {}
    switches = 'gap=%s, trace=%d, pes=%d, coverage=%d, predict=%d' % (
        plan.get('gap', 'off'), int(bool(plan.get('trace'))),
        int(bool(plan.get('pes'))), int(bool(plan.get('coverage'))),
        int(plan.get('predict_mode', 0) or 0))
    explore = [name for name, key in (('udd', 'explore_udd'),
                                      ('gap', 'explore_gap'))
               if plan.get(key)]
    if explore:
        switches += ', explore=%s' % '+'.join(explore)
    text = '\n  &adaptive analysis\n%s\n' % ('-' * 79)
    text += '  %-24s %s\n' % ('version', report.get('version'))
    text += '  %-24s %s\n' % ('switches', switches)
    for entry in report.get('trajectories') or []:
        traj = entry.get('traj_id')
        checks = []
        for key in ('population_status', 'order_status'):
            value = entry.get(key)
            if value not in (None, 'off'):
                checks.append('%s=%s' % (key.replace('_status', ''), value))
        if entry.get('exploration'):
            checks.append('exploration=True')
        if checks:
            text += '  %-24s traj %s  %s\n' % (
                'checks', traj, ', '.join(checks))
        if isinstance(entry.get('gap'), dict):
            text += '  %-24s traj %s  %s\n' % (
                'gap', traj, _format_gap_item(entry['gap']))
        if isinstance(entry.get('trace'), dict):
            text += '  %-24s traj %s  %s\n' % (
                'trace', traj, _format_trace_item(entry['trace']))
        if isinstance(entry.get('pes'), dict):
            text += '  %-24s traj %s  %s\n' % (
                'pes', traj, _format_pes_item(entry['pes']))
        if isinstance(entry.get('predictor'), dict):
            item = entry['predictor']
            if item.get('status') == 'ok':
                detail = 'model_ids %s' % (item.get('model_ids'),)
            else:
                detail = _analysis_status(item)
            text += '  %-24s traj %s  %s\n' % (
                'predictor', traj, detail)
        if entry.get('bias_status'):
            detail = ''
            for key in ('udd_energy', 'gap_bias_energy', 'gap_excess',
                        'gap_velocity_scale', 'bias_work',
                        'udd_force_ratio', 'udd_state_switches'):
                if entry.get(key) is not None:
                    detail += ' %s=%.6g' % (key, float(entry[key]))
            if entry.get('gap_force_switched') is not None:
                # A boolean, so it is not part of the numeric loop above;
                # it is the only sign that the paper gap method actually
                # replaced the propagation gradient.
                detail += ' gap_force_switched=%s' % bool(
                    entry['gap_force_switched'])
            text += '  %-24s traj %s  %s%s\n' % (
                'bias', traj, entry['bias_status'], detail)
    summary = report.get('summary') or {}
    coverage = summary.get('coverage')
    if isinstance(coverage, dict):
        fraction = coverage.get('covered_fraction')
        if fraction is not None:
            nearest = coverage.get('nearest_distance') or {}
            text += '  %-24s %.3f of %s candidates, radius %.4g, nearest median %.4g\n' % (
                'coverage', float(fraction),
                coverage.get('n_candidates'),
                float(coverage.get('coverage_radius') or 0.0),
                float(nearest.get('median') or 0.0))
        else:
            text += '  %-24s %s\n' % (
                'coverage', _analysis_status(coverage))
    pes = summary.get('pes')
    if isinstance(pes, dict):
        text += '  %-24s %s\n' % ('pes', _format_pes_item(pes))
    sat = summary.get('saturation')
    if isinstance(sat, dict):
        if sat.get('status') == 'ok':
            text += '  %-24s MAE %.6f  improvement %.6f  local %.4g  global %.4g  %s\n' % (
                'saturation', float(sat.get('moving_average', 0.0)),
                float(sat.get('improvement', 0.0)),
                float(sat.get('local_tolerance') or 0.0),
                float(sat.get('global_tolerance') or 0.0),
                'SATURATED' if sat.get('saturated') else 'improving')
        else:
            text += '  %-24s %s\n' % (
                'saturation', _analysis_status(sat))
    progress = summary.get('progress')
    if isinstance(progress, dict):
        text += '  %-24s %s new labels, %s round(s) since improvement\n' % (
            'progress', progress.get('cumulative_new_frames'),
            progress.get('rounds_since_improvement'))
    return text
