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

Input generation for the qmmm_field (Flow 1) task.

The task is a pure Coulomb probe: q_Q=+1, A = full system, B = the same
system with the MM charges zeroed. The two runs share the same box,
coefficients, cutoffs, special_bonds and kspace, so A-B is exactly the
MM contribution to each QM atom's potential and field. The script is
kept separate from the generic ff_* tail; the two QM/MM tasks do not
add branches to ff_energy/ff_gradient.
"""

import hashlib
import os

from ..utils import output_basename
from .input_builder import (_is_periodic, _reissue_kspace,
                            build_script_head, parse_template_box_origin,
                            resolve_live_box, write_input_package)

# Ewald splitting pinned into the periodic probe deck. LAMMPS re-derives
# alpha and the extent of the reciprocal sum from qsqsum at every init, so
# the A run (MM charges present) and the B run (MM charges zeroed) sum over
# different sets of k vectors - measured on the 1581-atom box at 1e-6, 9^3
# against 7^3 at the same pinned alpha - and V, being a difference of the
# two, is short by exactly that truncation while the group/group cross term
# is not.
#
# Only alpha is pinned, as the split parameter. The extent is not settable:
# `kspace_modify mesh` is accepted and silently ignored by ewald - two decks
# differing only by that line return bit-identical readings and the second
# probe keeps its own estimate - so the probe carries a tight accuracy from
# the `ewald_accuracy` setting instead and lets both runs converge to the
# same Ewald sum. That key, its numbers and the routes ruled out on the
# deployed engine are recorded in dev/skills/LAMMPS_SKILL_DEVELOPMENT.md;
# the short version is that 1e-6 left a 5.3e-4 relative residual there while
# 1e-8 leaves 2.1e-6, and that a scheme which only made the two grids
# identical would have passed the check with a 1.0e-4 error still in B.
_PIN_ALPHA_FACTOR = 3.2


def field_kspace_setup(box_lengths, coul_cutoff, style: str,
                       accuracy: str) -> str:
    """The lines that fix the probe's Ewald split.

    ``alpha = 3.2/rc`` keeps the real-space truncation at ``erfc(3.2)``;
    the style line carries the probe's accuracy so that the reciprocal
    sum each run estimates converges to the same value.
    """
    lengths = [float(value) for value in box_lengths or []]
    if len(lengths) != 3 or min(lengths) <= 0:
        return ''
    cutoff = float(coul_cutoff) if coul_cutoff else 0.5 * min(lengths)
    if cutoff <= 0:
        return ''
    lines = []
    name = str(style or '').strip()
    if name:
        lines.append('kspace_style %s %s' % (name, accuracy))
    lines.append('kspace_modify gewald %.6f' % (_PIN_ALPHA_FACTOR / cutoff))
    return '\n'.join(lines)


def _coul_cutoff(model):
    """The smallest real-space Coulomb cutoff declared by the pair styles.

    In every long-range style the Coulomb cutoff is the last numeric
    argument (``lj/charmm/coul/long inner cut_lj cut_coul``,
    ``lj/cut/coul/long cut_lj cut_coul``, ``coul/long cut_coul``).

    The model segments a *hybrid* declaration only, so a plain
    ``pair_style`` is read off the init line its entry points at. Both
    spellings have to answer: with no cutoff found the caller falls back
    to half the shortest box axis, which for a water box pins an alpha
    whose real-space truncation is ``erfc(alpha*rc)`` rather than the
    ``erfc(3.2)`` this pin is specified by - a silent accuracy loss on
    exactly the input form a non-hybrid force field writes.
    """
    init = (model or {}).get('init') or {}
    pair = (init.get('styles') or {}).get('pair') or {}
    declarations = [segment[1] if len(segment) > 1 else []
                    for segment in pair.get('segments') or []]
    if not declarations and not pair.get('hybrid'):
        lines = init.get('lines') or []
        index = pair.get('index')
        if isinstance(index, int) and 0 <= index < len(lines):
            tokens = lines[index].split('#')[0].split()
            if len(tokens) > 2:
                declarations.append(tokens[2:])
    cuts = []
    for tokens in declarations:
        numbers = []
        for token in tokens or []:
            try:
                numbers.append(float(token))
            except (TypeError, ValueError):
                continue
        if numbers:
            cuts.append(numbers[-1])
    return min(cuts) if cuts else None


def _kspace_style(kspace_line: str) -> str:
    tokens = str(kspace_line or '').split()
    return tokens[1] if len(tokens) > 1 else ''


def _field_kspace_setup(config, model, project: str, calcdir: str) -> str:
    """The pin for this deck, or '' when the probe is not periodic."""
    if not _is_periodic(config):
        return ''
    # Asked for the template line's *style name* only, so no accuracy is
    # passed here; the accuracy the probe carries comes from the config and
    # is written by field_kspace_setup below.
    kspace = _reissue_kspace(model, project)
    if not kspace:
        return ''
    path = os.path.join(calcdir, f'{project}.data')
    try:
        with open(path) as handle:
            data_lines = handle.read().splitlines()
    except OSError:
        return ''
    origin, _ = parse_template_box_origin(data_lines, f'{project}.data')
    box = resolve_live_box(config, origin)
    if box is None:
        return ''
    lengths = [box['xhi'] - box['xlo'], box['yhi'] - box['ylo'],
               box['zhi'] - box['zlo']]
    return field_kspace_setup(lengths, _coul_cutoff(model),
                              _kspace_style(kspace), config.ewald_accuracy)


def _groups(partition: dict) -> str:
    """Declare the probe and environment groups used by the probe.

    ``QMMM_QM`` is the set whose potential is measured, which under several
    centers is one center, not the whole QM set. ``QMMM_MM`` is everything
    else, and "everything else" is what makes the multi-center case work:
    the centers this probe does not measure fall on the environment side,
    so A carries their current charges and B -- which zeroes the whole
    environment -- removes them again, leaving their contribution in the
    difference exactly like the MM one.
    """
    ids = partition.get('probe_ids') or partition['qm_ids']
    return (f'group QMMM_QM id ' + ' '.join(str(value) for value in ids) +
            '\n'
            'group QMMM_MM subtract all QMMM_QM\n')


def _restore_groups(partition: dict):
    """Declare one MM charge group per distinct charge and return its map.

    A resident step has to put the MM charges back after the B run. Issuing
    one ``set atom`` per MM atom would make the per-step command count grow
    with the system; grouping by charge keeps it proportional to the number
    of distinct charges instead.
    """
    groups = {}
    for atom_id, charge in zip(partition['mm_ids'],
                               partition['mm_charges']):
        key = f'{float(charge):.12g}'
        groups.setdefault(key, []).append(int(atom_id))
    lines = []
    descriptor = []
    for index, (charge, ids) in enumerate(sorted(groups.items())):
        name = f'QMMM_MMC{index}'
        lines.append(f'group {name} id '
                     + ' '.join(str(value) for value in ids))
        descriptor.append({'name': name, 'charge': float(charge)})
    return '\n'.join(lines) + ('\n' if lines else ''), descriptor


def _probe_computes(periodic: bool) -> str:
    """Declare the per-atom energy and cross-group tallies."""
    pair = 'pair kspace' if periodic else 'pair'
    kspace = ' pair yes kspace yes' if periodic else ' pair yes'
    return (
        f'compute QMMM_PEQ QMMM_QM pe/atom {pair}\n'
        f'compute QMMM_GG QMMM_QM group/group QMMM_MM{kspace}\n'
        'thermo_style custom step pe c_QMMM_GG c_QMMM_GG[1] '
        'c_QMMM_GG[2] c_QMMM_GG[3]\n'
        'thermo 1\n'
    )


def _probe_run(label: str, basename: str, calcdir: str) -> str:
    """One A/B run plus its two dumps and the group-tally print.

    The dumps carry 17 significant digits: the cold path reads V from
    ``probe_{label}.pe``, and LAMMPS's default ``%g`` keeps six, which
    quantises V at ~5e-4 and hides everything the A/B guard could see.
    """
    pe = os.path.join(calcdir, f'{basename}.probe_{label}.pe')
    force = os.path.join(calcdir, f'{basename}.probe_{label}.force')
    return (
        f'dump QMMM_PE_{label} QMMM_QM custom 1 {pe} id c_QMMM_PEQ\n'
        f'dump_modify QMMM_PE_{label} sort id format float %20.17g\n'
        f'dump QMMM_F_{label} QMMM_QM custom 1 {force} id fx fy fz\n'
        f'dump_modify QMMM_F_{label} sort id format float %20.17g\n'
        'run 0\n'
        'variable pyrai_gg equal c_QMMM_GG\n'
        'variable pyrai_fx equal c_QMMM_GG[1]\n'
        'variable pyrai_fy equal c_QMMM_GG[2]\n'
        'variable pyrai_fz equal c_QMMM_GG[3]\n'
        f'print "QMMM_FIELD {label} ${{pyrai_gg}} ${{pyrai_fx}} '
        f'${{pyrai_fy}} ${{pyrai_fz}}" '
        f'file {os.path.join(calcdir, basename + ".field_" + label + ".groups")}\n'
        f'undump QMMM_PE_{label}\n'
        f'undump QMMM_F_{label}\n'
    )


def write_field_static_input_files(config, workflow, calcdir: str,
                                   model) -> dict:
    """Write the resident qmmm_field package.

    The loaded script declares the groups and computes but contains no
    A/B run: the service issues the two runs per step, after applying the
    step's box and coordinates, and then extracts the per-atom compute.

    Something has to consume that compute for LAMMPS to make it current,
    and this package used to do that with a `dump` of `c_QMMM_PEQ` whose
    file nobody ever read. A declared dump is also what forbids changing
    the box type, so the first tilted step of a variable-cell search died
    with `Cannot change box ortho/triclinic with dumps defined
    (src/change_box.cpp:308)` and took the resident session with it -
    measured on the deployed build with this deck, on the 1581-atom box.
    The consumer is therefore a `reduce` over an atom-style alias of the
    same compute: it holds no file, so the conversion goes through, and it
    was checked to refresh the per-atom values the service extracts (before
    a displacement the two agree at -224.236591155, after one at
    -218.176596704, the sum over the QM group and the reduce agreeing to
    the last printed digit).
    """
    project = config.lammps_project
    basename = output_basename(project, workflow.profile)
    os.makedirs(calcdir, exist_ok=True)
    write_input_package(config, calcdir, model)
    partition = model['qmmm']
    restore_lines, restore_groups = _restore_groups(partition)
    script = build_script_head(config, workflow, calcdir, model)
    script += _groups(partition) + restore_lines
    script += _probe_computes(_is_periodic(config))
    setup = _field_kspace_setup(config, model, project, calcdir)
    if setup:
        script += setup + '\n'
    script += ('variable QMMM_PEQV atom c_QMMM_PEQ\n'
               'compute QMMM_PEQS all reduce sum v_QMMM_PEQV\n')
    # The reduce has to be consumed by something, or LAMMPS never invokes
    # `compute pe/atom` and the library read hands back whatever the array
    # held last - in a resident session that is the previous evaluation's
    # value. The cold deck gets its invocation from the A/B dumps; this
    # package carries no dump (one would forbid the box change a tilted step
    # needs), so the thermo line is the consumer.
    script += ('thermo_style custom step pe c_QMMM_GG c_QMMM_GG[1] '
               'c_QMMM_GG[2] c_QMMM_GG[3] c_QMMM_PEQS\n'
               'thermo 1\n')
    with open(os.path.join(calcdir, f'{basename}.in'), 'w') as handle:
        handle.write(script)
    step_tail = ['run 0']
    with open(os.path.join(calcdir, f'{project}.data')) as handle:
        data_lines = handle.read().splitlines()
    origin, _ = parse_template_box_origin(
        data_lines, f'{project}.data')
    names = (f'{project}.data', f'{project}.in.init',
             f'{project}.in.settings', f'{basename}.in')
    files = {}
    for name in sorted(names):
        path = os.path.join(calcdir, name)
        if os.path.isfile(path):
            with open(path, 'rb') as handle:
                files[name] = hashlib.sha256(handle.read()).hexdigest()
    return {
        'script': basename,
        # A tilted step re-issues kspace_style; without the pin the next
        # init would fall back to LAMMPS's own alpha and split A from B.
        # The fallback carries the configured accuracy too, so a probe that
        # loses the alpha pin - no resolvable box, unreadable data - does not
        # silently lose the convergence requirement with it.
        'kspace': setup or _reissue_kspace(model, project,
                                           config.ewald_accuracy),
        'step_tail': step_tail,
        'files': files,
        'order': list(range(1, int(config.natom) + 1)),
        'origin': list(origin),
        'box': resolve_live_box(config, origin),
        'natom': int(config.natom),
        'restore_groups': restore_groups,
    }


def write_field_input_file(config, workflow, calcdir: str, model=None,
                           q_qm=None) -> str:
    """Write the cold qmmm_field input script.

    Returns the script basename. The data/init/settings package is the
    qmmm_energy model with every non-Coulomb pair parameter zeroed. The
    probed center is set to +1 at run time, so its own charges never enter
    the difference. Under several centers the *unprobed* QM atoms are
    environment charges and must carry their current 0-order values in the
    data file, because the A leg reads them; ``q_qm`` is that whole QM set
    and is written by atom id, exactly as the energy task writes its own
    charges. With one center the probed set is the whole QM set and nothing
    is written: the data file keeps the topology charges only, as before.
    """
    project = config.lammps_project
    basename = output_basename(project, workflow.profile)
    write_input_package(config, calcdir, model)
    partition = model['qmmm']
    if q_qm is not None and partition.get('env_ids'):
        from .qmmm_energy import _rewrite_qm_charges
        _rewrite_qm_charges(calcdir, project, model,
                            partition['qm_ids'], q_qm)
    head = build_script_head(config, workflow, calcdir, model)
    periodic = _is_periodic(config)
    script = head + _groups(partition)
    script += _probe_computes(periodic)
    setup = _field_kspace_setup(config, model, project, calcdir)
    if setup:
        script += setup + '\n'
    script += 'set group QMMM_QM charge 1.0\n'
    script += _probe_run('A', basename, calcdir)
    script += 'set group QMMM_MM charge 0.0\n'
    script += _probe_run('B', basename, calcdir)
    path = os.path.join(calcdir, f'{basename}.in')
    with open(path, 'w') as handle:
        handle.write(script)
    return basename
