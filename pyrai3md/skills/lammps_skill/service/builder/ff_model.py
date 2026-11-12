"""
======================================================================
This is a standalone LAMMPS calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 26 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module compresses the user-provided LAMMPS topology and force
field into a single JSON model that is the authoritative description of
what the calculation actually evaluates.

A shipped force-field template declares every type of a large library
(the reference water box declares 111 atom / 952 bond / 5355 angle /
1004 dihedral / 38 improper types) while a 60-atom TIP3P box only
references 2 atom, 1 bond and 1 angle type. LAMMPS materialises its
pair table from the *declared* type count (that template generates
6102 of 6105 mixed pair_coeff terms per call) and echoes the 8600-line
settings file into its log on every invocation, so a framework MD loop
pays that fixed cost once per step. The model keeps only the referenced
types, renumbers them onto 1..N, and renders the three input files from
them.

The model also owns the constraint state. A constraint has to be applied
by whatever integrates the atoms, and the framework integrator has no
constraint engine, so the shake fix is emitted only for the profiles
where LAMMPS integrates. Keeping that decision in one place removes the
historical split where the ``fix`` lived in the settings template while
the matching ``unfix`` was appended by the input script builder.

Duplicate coefficient keys are resolved last-write-wins: a template that
appends re-fitted parameters (the reference file carries the Price &
Brooks 2004 PPPM-optimised TIP3P values over generated defaults)
depends on exactly that order, and the model records every superseded
line so the override becomes explicit.

Anything this module cannot prove is rejected with
:class:`FFModelUnsupported` and the caller falls back to the verbatim
templates; compression is never allowed to guess.

The model also owns the ``{project}.in.init`` style declarations. A
``*_style hybrid`` wrapper requires every one of its sub-styles to be
assigned to at least one type pair (LAMMPS otherwise aborts with "Pair
hybrid sub-style is not used"), so trimming the coefficient rows of
unused types silently starves a sub-style the declaration still lists:
the reference template declares ``buck/coul/long``, ``fourier`` and
``cvff`` while the 60-atom box uses none of them. Keeping the
declaration honest means rendering it from the surviving rows, which is
why the init lines live in the model instead of being copied.

"""

import hashlib
import json
import os
import tempfile

from ...schema.method import ENGINE_INTEGRATED_PROFILES

# Model format 3 adds the optional 'xy xz yz' tilt line of a triclinic
# data box to model['box']; format 2 caches carried an orthorhombic box
# only, so load_cached_model must reject them rather than hand back a
# model that silently drops the tilt.
MODEL_FORMAT = 3

# Commands whose lines may carry a force-field type number. The census
# over the reference templates shows this set is complete, which is what
# makes renumbering provably safe; an unknown command disables the model
# rather than being copied through.
COEFF_COMMANDS = ('pair_coeff', 'bond_coeff', 'angle_coeff',
                  'dihedral_coeff', 'improper_coeff')
ALLOWED_SETTINGS_COMMANDS = COEFF_COMMANDS + ('group', 'fix')

# Style declarations of the init template. The family name doubles as
# the coefficient command and data-section prefix, and 'pair' is the one
# family that can never be switched off (charges need a long-range
# pair style).
STYLE_FAMILIES = ('pair', 'bond', 'angle', 'dihedral', 'improper')

# Data-file sections whose type column is renumbered.
TYPE_SECTIONS = (('Atoms', 'atom', 2), ('Bonds', 'bond', 1),
                 ('Angles', 'angle', 1), ('Dihedrals', 'dihedral', 1),
                 ('Impropers', 'improper', 1))

from .exceptions import FFModelUnsupported


def build_ff_model(config, profile: str) -> dict:
    """Compress the LAMMPS templates of one project into a model.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    profile : str
        Compiled workflow profile, used to decide who integrates.
    Returns
    -------
    dict
        The force-field model (see :func:`render_settings`). Carries
        ``selection`` only when the input asked for a subsystem, so the
        cache file of a whole-system run stays byte-identical to the
        generation before the key existed.

    Raises
    ------
    FFModelUnsupported
        When any template shape, section or command falls outside what
        this module can prove equivalent.
    """
    project = config.lammps_project
    sources = read_templates(project)
    model = {
        'format': MODEL_FORMAT,
        'project': project,
        'source_hash': _hash_sources(sources),
        'config_hash': _hash_config(config, profile),
        'used_types': {}, 'type_map': {}, 'counts': {}, 'masses': [],
        'box': [], 'sections': {}, 'coeffs': {}, 'init': {},
        'constraints': {}, 'audit': {},
    }

    model['sections'] = _parse_data(sources['data'])
    model['coeffs'], model['constraints'], model['audit'] = _parse_settings(
        sources['in_settings'], config, profile)
    # The selection is cut after the settings parse because it writes its
    # torn-entity record into model['audit'], which _parse_settings owns. It
    # stays before _build_type_maps for the load-bearing reason: the type
    # census, the declared-count check and the Masses rebuild all have to
    # see the surviving rows, not the template's.
    _apply_selection(model, getattr(config, 'atom_selection', None) or [])
    profile_lower = str(profile).lower()
    if profile_lower in ('qmmm_field', 'qmmm_energy'):
        # Imported here rather than at module scope: qmmm needs this
        # module's exception class, and a top-level import in both
        # directions would leave one of the two halves uninitialized
        # depending on which was imported first.
        from .qmmm import (apply_qmmm, plan_partition,
                           validate_partition_charges)
        qm_atoms = getattr(config, 'qm_atoms', None) or []
        if profile_lower == 'qmmm_energy':
            model['qmmm'] = apply_qmmm(
                model, qm_atoms,
                getattr(config, 'qm_centers', None) or [])
            # Split each QM atom type into a cloned type so the Q-Q pair
            # rows are distinct from Q-M/M-M. Without the split a
            # type-based zero would also remove the Q-M and M-M terms,
            # because QM and MM atoms normally share LAMMPS types.
            new_types = split_qm_types(
                model, model['qmmm']['qm_ids'])
            model['qmmm']['new_types'] = sorted(set(new_types.values()))
        else:
            partition = plan_partition(
                model, qm_atoms, getattr(config, 'qm_probe', None) or [])
            # No charges exist yet at model-build time: the probe's
            # environment is verified here only over the MM subsystem, and
            # re-verified per step with the charges that actually arrive.
            validate_partition_charges(partition, None, 'field')
            model['qmmm'] = dict(partition, removed={})
    # Read the style declarations against the full coefficient table: a
    # hybrid wrapper names its sub-styles in the same tokens the
    # coefficient rows use, and only the untrimmed rows list them all.
    model['init'] = _parse_init(sources['in_init'], model['coeffs'])
    if profile_lower == 'qmmm_field':
        # The pair style name of a non-hybrid row comes from the init
        # declaration, so this surgery has to follow _parse_init.
        zero_non_coulomb_in_model(model)
    elif profile_lower == 'qmmm_energy':
        zero_qm_new_pairs(model,
                          model['qmmm'].get('new_types') or [])
    if is_subsystem(model):
        _degrade_coulomb(model)
    _build_type_maps(model)
    _prune_styles(model)
    _validate(model, config)
    return model


def read_templates(project: str) -> dict:
    """Read the three user templates of one project from the launch dir.

    Parameters
    ----------
    project : str
        Template base name.

    Returns
    -------
    dict
        ``{'data', 'in_init', 'in_settings'}`` mapped to file text.

    Raises
    ------
    FFModelUnsupported
        When the topology template is missing (nothing to compress).
    """
    paths = {'data': f'{project}.data',
             'in_init': f'{project}.in.init',
             'in_settings': f'{project}.in.settings'}
    sources = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            if key == 'data':
                raise FFModelUnsupported(f'topology template {path} is absent')
            sources[key] = ''
            continue
        with open(path, 'r') as f:
            sources[key] = f.read()
    return sources


def model_path(project: str, calcdir: str = '') -> str:
    """Return the model cache path, inside the calculation directory.

    The cache is a convenience, not a keeper of state: the backend holds the
    resolved model in memory for its whole life (``LammpsBackend.ff_model``),
    so a multi-step run never reads this file back, and every step of the
    subprocess path rebuilds its input from that in-memory model anyway. What
    the file is for is a run that keeps its directory: with ``keep_tmp`` set,
    ``{project}.ff.json`` sits beside the inputs it was compressed from and
    shows what the engine was actually given. With ``keep_tmp`` clear it goes
    away with everything else, which is the point of writing it there.

    Parameters
    ----------
    project : str
        Template base name.
    calcdir : str
        Backend calculation directory (empty means the launch directory,
        for a standalone caller that has not created one).

    Returns
    -------
    str
        Path of ``{project}.ff.json``.
    """
    return os.path.join(calcdir or '.', f'{project}.ff.json')


def load_cached_model(path: str, sources: dict, config, profile: str):
    """Load a cached model when it still describes this calculation.

    Parameters
    ----------
    path : str
        Model cache path.
    sources : dict
        Current template texts, keyed as in :func:`read_templates`.
    config : SkillConfig
        Runtime configuration container.
    profile : str
        Compiled workflow profile.

    Returns
    -------
    dict or None
        The cached model, or None when it is absent, unreadable, of a
        different format, built from different template content, or
        built under a different constraint configuration.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            model = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(model, dict) or model.get('format') != MODEL_FORMAT:
        return None
    if model.get('source_hash') != _hash_sources(sources):
        return None
    if model.get('config_hash') != _hash_config(config, profile):
        return None
    # The hashes already separate two selections, so reaching here with a
    # mismatched record means a cache was written by a model that did not
    # carry the key at all. Refuse it rather than compute with rows that
    # belong to a different subsystem.
    if [int(value) for value in (model.get('selection') or [])] != (
            _selection_of(config)):
        return None
    return model


def save_model(path: str, model: dict) -> bool:
    """Write the model cache atomically, tolerating an unwritable directory.

    A plain ``open(path, 'w')`` truncates the file before writing, and a
    concurrent reader then hits the ``except`` of
    :func:`load_cached_model` and silently degrades to the verbatim
    templates; write to a temporary file in the same directory and
    rename, which is atomic within a directory.

    Parameters
    ----------
    path : str
        Model cache path.
    model : dict
        Model to persist.

    Returns
    -------
    bool
        True when the cache was written.
    """
    directory = os.path.dirname(path) or '.'
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + '.', dir=directory)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(model, f, indent=1)
            os.replace(tmp_path, path)
        except OSError:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError:
        return False
    return True


def render_settings(model: dict) -> str:
    """Render the compressed '{project}.in.settings' content.

    Parameters
    ----------
    model : dict
        Force-field model.

    Returns
    -------
    str
        Settings lines: one per surviving coefficient (renumbered,
        with the source line kept as a comment) plus the constraint
        declarations when they are enabled.
    """
    lines = [f'# PyRAI3MD compressed force field (model format '
             f'{MODEL_FORMAT})']
    provenance = model['audit'].get('provenance', {})
    for kind in ('pair', 'bond', 'angle', 'dihedral', 'improper'):
        for row in model['coeffs'].get(kind, []):
            src = provenance.get(kind, {}).get(_coeff_key(row))
            suffix = f'   # src L{src}' if src else ''
            lines.append(f'{kind}_coeff '
                         + ' '.join(str(token) for token in row) + suffix)
    for line in model['constraints'].get('lines', []):
        lines.append(line)
    return '\n'.join(lines) + '\n'


def render_init(model: dict) -> str:
    """Render the '{project}.in.init' content from the model.

    Parameters
    ----------
    model : dict
        Force-field model.

    Returns
    -------
    str
        The init lines as shipped, except the style declarations
        :func:`_prune_styles` had to trim; '' when the project ships no
        init template.
    """
    lines = model['init'].get('lines') or []
    if not lines:
        return ''
    return '\n'.join(lines) + '\n'


def render_data(model: dict) -> str:
    """Render the compressed '{project}.data' template content.

    The rendered file carries the template coordinates; the per-step
    writer still applies the live box, coordinates and velocities
    through the unchanged data-file path.

    Parameters
    ----------
    model : dict
        Force-field model.

    Returns
    -------
    str
        Data file text with renumbered types and pruned counts.
    """
    counts = model['counts']
    lines = [model['sections'].get('title', 'LAMMPS Description'), '']
    lines.extend(f'{counts[key]}  {label}' for label, key in
                 (('atoms', 'natom'), ('bonds', 'nbond'),
                  ('angles', 'nangle'), ('dihedrals', 'ndihedral'),
                  ('impropers', 'nimproper')))
    lines.append('')
    lines.extend(f'{counts[key]}  {label}' for label, key in
                 (('atom types', 'natom_types'),
                  ('bond types', 'nbond_types'),
                  ('angle types', 'nangle_types'),
                  ('dihedral types', 'ndihedral_types'),
                  ('improper types', 'nimproper_types')))
    lines.append('')
    lines.extend(model['box'])
    lines.append('')

    sections = model['sections']
    for name in ('Masses', 'Atoms', 'Bonds', 'Angles',
                 'Dihedrals', 'Impropers'):
        rows = sections.get(name)
        if not rows:
            continue
        lines.append(sections.get(f'{name}_header', name))
        lines.append('')
        lines.extend(rows)
        lines.append('')
    return '\n'.join(lines).rstrip('\n') + '\n'


def _hash_sources(sources: dict) -> dict:
    """Hash each template so a cache can be invalidated by content.

    Parameters
    ----------
    sources : dict
        Template texts keyed as in :func:`read_templates`.

    Returns
    -------
    dict
        Mapping of the same keys to sha256 hex digests.
    """
    return {key: hashlib.sha256(text.encode('utf-8')).hexdigest()
            for key, text in sources.items()}


def _selection_of(config) -> list:
    """The topology rows this configuration asks for, as integers.

    An absent key and an empty list are the same request (the whole system),
    which is why every consumer of this normalizes before comparing.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.

    Returns
    -------
    list of int
        1-based template atom ids in the handed order.
    """
    return [int(value) for value in (getattr(config, 'atom_selection', None)
                                     or [])]


def _hash_config(config, profile: str) -> str:
    """Hash the configuration the model depends on.

    The constraint state is decided from the profile and the shake
    keywords, so a cache keyed on the templates alone would hand an
    ff_gradient model to an ff_nvt run sharing the scratch directory.
    The atom selection joins that identity for the same reason: a
    multi-center run gives every LAMMPS task its own subsystem, and a
    cache that did not see the selection could answer one task's
    request with another task's rows. The QM/MM partition joins it for
    the same reason - a cached whole-system model would otherwise answer
    a QM/MM task whose bonded rows were never removed - and so does the
    reciprocal accuracy, which rewrites a token of the `kspace_style` line
    the deck carries: a cache blind to it would answer a converged probe
    with a deck sized for 1e-4, and the two runs would disagree only in
    their numbers.
    Each is appended only when it is set, so an ordinary whole-system
    run keeps the digest it had before the key existed and keeps
    reusing the cache file it wrote.

    Parameters
    ----------
    config : SkillConfig
        Runtime configuration container.
    profile : str
        Compiled workflow profile.
    Returns
    -------
    str
        sha256 hex digest of the model-relevant settings.
    """
    selected = ','.join(str(value) for value in _selection_of(config))
    # Ordered, because the row order is part of what the selection promises:
    # [3, 1, 2] and [1, 2, 3] are different subsystems.
    parts = [profile, str(getattr(config, 'lammps_shake', 0)),
             getattr(config, 'shake_fix_id', ''),
             getattr(config, 'shake_keywords', '')]
    if selected:
        parts.append(selected)
    accuracy = str(getattr(config, 'ewald_accuracy', '') or '')
    if accuracy:
        parts.append('ewald:' + accuracy)
    qm_atoms = [int(value) for value in
                (getattr(config, 'qm_atoms', None) or [])]
    if str(profile).lower() in ('qmmm_field', 'qmmm_energy') and qm_atoms:
        parts.append('qmmm:' + ','.join(str(value) for value in qm_atoms))
        # Which center the probe measures is part of the partition identity:
        # two centers share the QM set but load different groups, and a
        # cached model built for one of them must never answer for the other.
        probe = [int(value) for value in
                 (getattr(config, 'qm_probe', None) or [])]
        if probe:
            parts.append('probe:' + ','.join(str(value) for value in probe))
        # The per-center tiling decides which Q-Q pairs the in-cell removal
        # takes out of the classical surface, so it is part of the model
        # identity and not a runtime argument.
        centers = (getattr(config, 'qm_centers', None) or [])
        if centers:
            parts.append('centers:' + ';'.join(
                ','.join(str(int(value)) for value in group)
                for group in centers))
        q_qm = [float(value) for value in
                (getattr(config, 'q_qm', None) or [])]
        if q_qm:
            parts.append('q_qm:' + ','.join(f'{value:.12g}'
                                            for value in q_qm))
    return hashlib.sha256('|'.join(str(part) for part in parts)
                          .encode('utf-8')).hexdigest()


def _parse_data(text: str) -> dict:
    """Split a data file into counts, box and per-section rows.

    Parameters
    ----------
    text : str
        Data file content.

    Returns
    -------
    dict
        ``counts`` (declared integers by label), ``box`` (verbatim bound
        lines, plus the 'xy xz yz' tilt line when the box is triclinic)
        and one entry per section holding its verbatim rows plus the
        section header text.

    Raises
    ------
    FFModelUnsupported
        When a count line, the box or a section row does not match the
        expected shape.
    """
    lines = text.splitlines()
    # The first line of a LAMMPS data file is its title, not data.
    title = lines[0].rstrip() if lines else ''
    counts = {}
    box = []
    sections = {'title': title}
    current = None
    # The tilt line is not optional in the format, only in the box: it
    # follows 'zlo zhi' immediately when the box is triclinic.
    awaiting_tilt = False

    count_labels = ('atoms', 'bonds', 'angles', 'dihedrals', 'impropers',
                    'atom types', 'bond types', 'angle types',
                    'dihedral types', 'improper types')

    for raw in lines[1:]:
        stripped = raw.split('#')[0].strip()
        if not stripped:
            continue
        if stripped.startswith('$'):
            continue

        matched = False
        for label in count_labels:
            if stripped.endswith(' ' + label):
                head = stripped[:-(len(label) + 1)].strip()
                if head.isdigit():
                    counts[label] = int(head)
                    matched = True
                break
        if matched:
            continue

        if stripped.endswith(('xlo xhi', 'ylo yhi', 'zlo zhi')):
            box.append(raw.rstrip())
            awaiting_tilt = stripped.endswith('zlo zhi')
            continue

        if awaiting_tilt:
            awaiting_tilt = False
            tokens = stripped.split()
            labelled = (len(tokens) == 6
                        and tokens[3:] == ['xy', 'xz', 'yz'])
            if (labelled or len(tokens) == 3) and all(
                    _looks_like_number(token) for token in tokens[:3]):
                box.append(raw.rstrip())
                continue

        if raw.split('#')[0].strip() and raw.split()[0] in (
                'Masses', 'Atoms', 'Bonds', 'Angles', 'Velocities',
                'Dihedrals', 'Impropers'):
            current = raw.split()[0]
            sections[f'{current}_header'] = raw.rstrip()
            sections[current] = []
            continue

        if current is None:
            raise FFModelUnsupported(
                f'unrecognised line before any data section: {raw!r}')
        if not _looks_like_int(stripped.split()[0]):
            # A second section header would otherwise be absorbed as a
            # row of the previous one, silently corrupting the model.
            raise FFModelUnsupported(
                f'unrecognised section or line inside {current}: {raw!r}')
        sections[current].append(raw.rstrip())

    if len(box) not in (3, 4):
        raise FFModelUnsupported(
            f'expected 3 box bound lines (plus an optional tilt line), '
            f'found {len(box)}')
    if len(box) == 4:
        _validate_tilt_box(box)
    if 'Atoms' not in sections:
        raise FFModelUnsupported('data file has no Atoms section')
    if 'Velocities' in sections:
        # The per-step writer owns velocities; carrying a template
        # section through the model would give two sources of truth.
        raise FFModelUnsupported('data file carries a Velocities section')
    for label in count_labels:
        if label not in counts:
            raise FFModelUnsupported(f'data file does not declare "{label}"')
    return {'counts': counts, 'box': box, **sections}


def _apply_selection(model: dict, selection: list) -> None:
    """Cut the topology template down to the rows one task computes.

    ``selection`` holds 1-based atom ids of the user's ``{project}.data``, in
    the order the framework hands over the sliced geometry. That order is the
    whole contract: the per-step writer puts coordinate row ``i`` on the row
    whose id is ``i + 1``, so the new ids are the *positions* in the selection,
    never a re-sorted copy of its values. (The framework canonicalises a
    center's indices to ascending at input time, so what arrives here is
    ascending in practice; the skill takes no part in that and must stay
    faithful to whatever row order it was given - which is what
    ``test_the_new_numbering_follows_the_handed_order`` pins.)

    An entity section keeps a row only when every member atom survived; a
    dropped row is recorded in the audit rather than lost. That is the physics
    the multi-center difference wants: an interaction that crosses the region
    boundary belongs to ``L(real)`` and would be subtracted twice if the
    subsystem also carried it.

    Empty selection returns with the model untouched, so every existing path
    renders byte-identical input.

    Parameters
    ----------
    model : dict
        Model whose sections were just parsed (modified in place).
    selection : list of int
        1-based template atom ids, in the handed order.

    Raises
    ------
    FFModelUnsupported
        When a requested id is not a row of the template, is asked for twice,
        or an entity row does not carry integer atom ids.
    """
    if not selection:
        return
    sections = model['sections']
    rows = {}
    for raw in sections.get('Atoms') or []:
        tokens = raw.partition('#')[0].split()
        if not tokens:
            continue
        if not _looks_like_int(tokens[0]):
            raise FFModelUnsupported(
                f'Atoms row does not start with an atom id: {raw!r}')
        if int(tokens[0]) in rows:
            raise FFModelUnsupported(
                f'data file has two Atoms rows with id {tokens[0]}')
        rows[int(tokens[0])] = raw

    wanted = []
    for value in selection:
        atom_id = int(value)
        if atom_id not in rows:
            raise FFModelUnsupported(
                f'atom {atom_id} is selected but the topology template has no '
                f'row with that id')
        if atom_id in wanted:
            raise FFModelUnsupported(
                f'atom {atom_id} is selected twice')
        wanted.append(atom_id)
    new_id = {old: position + 1 for position, old in enumerate(wanted)}

    def _rewrite(raw, replacements):
        content, sep, comment = raw.partition('#')
        tokens = content.split()
        for column, value in replacements.items():
            tokens[column] = str(value)
        return _join_with_comment(tokens, sep, comment)

    sections['Atoms'] = [_rewrite(rows[old], {0: new_id[old]})
                         for old in wanted]

    counts = sections['counts']
    torn = []
    entity_labels = {'Bonds': 'bonds', 'Angles': 'angles',
                     'Dihedrals': 'dihedrals', 'Impropers': 'impropers'}
    for name, _kind, type_column in TYPE_SECTIONS:
        if name not in entity_labels:
            continue
        kept = []
        for raw in sections.get(name) or []:
            content = raw.partition('#')[0].split()
            members = content[type_column + 1:]
            outside = [token for token in members
                       if not _looks_like_int(token) or int(token) not in new_id]
            if outside:
                torn.append({'section': name, 'row': raw,
                             'outside': outside})
                continue
            replacements = {0: len(kept) + 1}
            replacements.update(
                {column: new_id[int(token)]
                 for column, token in enumerate(members,
                                                start=type_column + 1)})
            kept.append(_rewrite(raw, replacements))
        sections[name] = kept
        counts[entity_labels[name]] = len(kept)

    counts['atoms'] = len(wanted)
    model['selection'] = wanted
    model['audit']['torn_entities'] = torn


def is_subsystem(model: dict) -> bool:
    """Whether this model describes a cut-out subsystem rather than a system.

    One fact, read from one key: a model that carries a selection is a
    subsystem, and a subsystem is evaluated on its own, non-periodically
    (dev/methods/PBC_DEVELOPMENT.md and dev/skills/LAMMPS_SKILL_DEVELOPMENT.md
    on why the difference term may not carry a box).
    """
    return bool(model.get('selection'))


# The real-space siblings of the Ewald-dependent pair styles, measured on the
# LAMMPS build this skill drives (22Jul2025). Each entry gives the name the
# style is replaced by and how many of the source's style-level arguments the
# sibling takes; the numbers themselves are copied verbatim, which is what
# keeps the template's own declared cutoff in force.
#
#   lj/charmm/coul/long rmin cutoff [inner]  ->  lj/cut/coul/cut rmin cutoff
#   buck/coul/long      cutoff               ->  buck/coul/cut    cutoff
#   coul/long           cutoff               ->  coul/cut         cutoff
#
# Two of these were measured rather than read off a manual, because the manual
# and the binary disagree here and the energy is the arbiter: with
# `lj/cut/coul/cut` the *interaction range* follows the LAST style argument
# given (at a pair distance of 9.5 A, `6.0 12.0` reads the bare Coulomb and
# `12.0 6.0` reads exactly zero, both under hybrid and without it), so taking
# charmm's first two numbers puts the template's cutoff in the position that
# governs. A `buck/coul/cut` coefficient row keeps the same three per-pair
# numbers as its long-range twin (a two-number row is refused by the engine),
# and a `coul/cut` row takes none, which is why the per-pair rows below are
# renamed but never re-shaped.
#
# `lj/charmm/coul/long` has NO same-family real-space sibling in this build
# (`lj/charmm/coul`, `lj/charmmfsw`, `lj/charmmfsh` are all absent), so the
# swap changes the functional form as well as the range: the CHARMM style
# always mixes Rmin arithmetically and epsilon geometrically, while
# `lj/cut/coul/cut` honours the template's `pair_modify mix`. For the shipped
# water template that is inert to the last digit - every LJ epsilon it declares
# is 0.0, so `coul/cut`, `lj/cut/coul/cut` and `buck/coul/cut` all read the same
# isolated energy (measured: 1.2124000038669500 kcal/mol for one water). It is
# not inert in general, which is why the change is recorded in the audit and
# named in dev/skills/LAMMPS_SKILL_DEVELOPMENT.md.
COULOMB_REAL_SPACE = {
    'lj/charmm/coul/long': ('lj/cut/coul/cut', 2),
    'buck/coul/long': ('buck/coul/cut', 1),
    'coul/long': ('coul/cut', 1),
}


def _needs_real_space(name: str) -> bool:
    """Whether one pair sub-style is the Ewald-dependent form of a pair style.

    LAMMPS spells the long-range variant of a family with a trailing
    `/coul/long` (or, for the pure Coulomb family, `coul/long` alone), and that
    form requires a kspace style, which a subsystem does not have.
    """
    return name == 'coul/long' or name.endswith('/coul/long')


def _degrade_coulomb(model: dict) -> None:
    """Rewrite a subsystem's pair styles so they need no kspace command.

    The three parts travel together: the declaration in the init lines, the
    sub-style token each hybrid coefficient row names, and the kspace line
    itself, which is dropped because leaving it would put the Ewald sum back
    where the difference term must not have one.

    Parameters
    ----------
    model : dict
        Model with a selection and a parsed init section (modified in place).

    Raises
    ------
    FFModelUnsupported
        When a long-range style has no measured real-space sibling, or when it
        declares fewer style arguments than its sibling needs; the caller then
        keeps the verbatim templates rather than guessing an energy form.
    """
    entry = model['init']['styles'].get('pair')
    if entry is None:
        return
    lines = model['init']['lines']
    content, sep, comment = lines[entry['index']].partition('#')
    tokens = content.split()

    if entry['hybrid']:
        if entry['unobserved']:
            # No coefficient rows to segment the declaration with, so the
            # argument boundaries are unknown and a long-range name cannot be
            # handed a sibling's argument list. Refused, not guessed.
            listed = [token for token in tokens[2:]
                      if _needs_real_space(token)]
            if listed:
                raise FFModelUnsupported(
                    f'pair style {listed[0]!r} needs a kspace command, and '
                    f'the hybrid declaration has no coefficient rows to '
                    f'segment it, so its arguments cannot be placed in a '
                    f'real-space sibling')
            return
        groups = [(segment[0], list(segment[1]))
                  for segment in entry['segments']]
    else:
        groups = [(tokens[1], tokens[2:])]

    renamed = {}
    kept_groups = []
    for name, args in groups:
        if not _needs_real_space(name):
            kept_groups.append((name, args))
            continue
        target = COULOMB_REAL_SPACE.get(name)
        if target is None:
            raise FFModelUnsupported(
                f'pair style {name!r} needs a kspace command and this build '
                f'has no measured real-space sibling for it; refusing to '
                f'change the energy form (known: '
                f'{sorted(COULOMB_REAL_SPACE)})')
        sibling, keep = target
        if len(args) < keep:
            raise FFModelUnsupported(
                f'pair style {name!r} declares {args}, fewer than the {keep} '
                f'numbers its real-space sibling {sibling!r} takes')
        renamed[name] = sibling
        kept_groups.append((sibling, args[:keep]))
        model['audit'].setdefault('coulomb_degradation', []).append({
            'from': name, 'to': sibling, 'kept': args[:keep],
            'dropped': args[keep:]})

    rendered = ['pair_style'] if not entry['hybrid'] else ['pair_style',
                                                           tokens[1]]
    for name, args in kept_groups:
        rendered.append(name)
        rendered.extend(args)
    lines[entry['index']] = ' '.join(rendered) + (
        ' ' + sep + comment if sep else '')
    if entry['hybrid']:
        # The segments are the declaration's own structure, and _prune_styles
        # reads them against the coefficient rows it is about to rename, so
        # both halves move together.
        entry['segments'] = [[name, args] for name, args in kept_groups]

    for row in model['coeffs'].get('pair', []):
        style = _row_style(True, row)
        if style in renamed:
            row[_substyle_slot('pair')] = renamed[style]

    # The Ewald command goes with the box, not with the atoms. Blanking the
    # line in place keeps every other line's index, which the style entries
    # reference.
    for index, raw in enumerate(lines):
        if raw.split('#')[0].strip().startswith('kspace_style'):
            lines[index] = ''
            model['audit'].setdefault(
                'kspace_dropped', []).append({'line': index + 1,
                                              'command': raw.strip()})


def _validate_tilt_box(box: list) -> None:
    """Reject a triclinic bound set LAMMPS itself would abort on.

    A data file states the *bounding* planes plus the tilts, and
    LAMMPS turns them into the true box by pulling each plane back by
    the shear it has to carry::

        xlo = xlo_bound - min(0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0, yz)   yhi = yhi_bound - max(0, yz)

    A box whose bounding planes are ordered but whose shear is wider
    than that span has no positive length once LAMMPS applies the
    inverse, which is the "Triclinic box is invalid" abort. Catching it
    here names the line instead of surfacing a LAMMPS error later.

    Parameters
    ----------
    box : list of str
        Three bound lines plus the tilt line.

    Raises
    ------
    FFModelUnsupported
        When a resulting true box length is not positive.
    """
    bounds = []
    for line in box[:3]:
        tokens = line.split('#')[0].split()[:2]
        bounds.append((float(tokens[0]), float(tokens[1])))
    tilt_tokens = box[3].split('#')[0].split()[:3]
    xy, xz, yz = (float(token) for token in tilt_tokens)

    x_shifts = (0.0, xy, xz, xy + xz)
    spans = [
        (bounds[0][1] - bounds[0][0]) - (max(x_shifts) - min(x_shifts)),
        (bounds[1][1] - bounds[1][0]) - (max(0.0, yz) - min(0.0, yz)),
        bounds[2][1] - bounds[2][0],
    ]
    for axis, length in zip('xyz', spans):
        if length <= 0.0:
            raise FFModelUnsupported(
                f'triclinic box leaves the {axis} axis with no positive '
                f'length (got {length!r} after applying the tilt line '
                f'{box[3]!r}); check the bound lines of the data file')


def _parse_settings(text: str, config, profile: str):
    """Collect coefficients, constraints and the dedup audit.

    Parameters
    ----------
    text : str
        Settings file content.
    config : SkillConfig
        Runtime configuration container.
    profile : str
        Compiled workflow profile.

    Returns
    -------
    tuple of (dict, dict, dict)
        (coeffs, constraints, audit).

    Raises
    ------
    FFModelUnsupported
        When a command outside the census appears, a type number is a
        wildcard or range, or a coefficient row is malformed.
    """
    coeffs = {kind: {} for kind in
              ('pair', 'bond', 'angle', 'dihedral', 'improper')}
    order = {kind: [] for kind in coeffs}
    duplicates = []
    groups = {}
    fixes = []

    for n, raw in enumerate(text.splitlines(), start=1):
        content = raw.split('#')[0].strip()
        if not content:
            continue
        tokens = content.split()
        command = tokens[0]

        if command not in ALLOWED_SETTINGS_COMMANDS:
            raise FFModelUnsupported(
                f'settings command {command!r} is outside the model '
                f'census; refusing to compress')

        if command in COEFF_COMMANDS:
            kind = command[:-len('_coeff')]
            key, row = _parse_coeff_row(tokens, command, n)
            if key in coeffs[kind]:
                duplicates.append({'key': list(key), 'kind': kind,
                                   'superseded_line': coeffs[kind][key][0],
                                   'kept_line': n})
            coeffs[kind][key] = (n, row)
            if key not in order[kind]:
                order[kind].append(key)
        elif command == 'group':
            groups[tokens[1]] = (n, tokens)
        elif command == 'fix':
            fixes.append((n, tokens))

    rendered = {kind: [list(coeffs[kind][key][1]) for key in order[kind]]
                for kind in coeffs}
    provenance = {
        kind: {_coeff_key(coeffs[kind][key][1]): coeffs[kind][key][0]
               for key in order[kind]}
        for kind in coeffs
    }
    audit = {'dropped_duplicates': duplicates,
             'provenance': provenance,
             'dropped_constraints': []}

    constraints = _resolve_constraints(
        fixes, groups, config, profile, audit)
    return rendered, constraints, audit


def _parse_coeff_row(tokens: list, command: str, line_no: int):
    """Split one coefficient line into a key and its rendered row.

    The hybrid styles name the substyle explicitly between the type
    numbers and the values; a single-style declaration omits it.

    Parameters
    ----------
    tokens : list of str
        Command tokens.
    command : str
        Coefficient command name.
    line_no : int
        Source line number (audit only).

    Returns
    -------
    tuple of (tuple, list)
        (dedup key, rendered row) with the ids kept as strings.

    Raises
    ------
    FFModelUnsupported
        When an id is a wildcard, a range, or the row is too short.
    """
    kind = command[:-len('_coeff')]
    body = tokens[1:]
    for token in body:
        if '*' in token or ':' in token:
            raise FFModelUnsupported(
                f'{command} on L{line_no} uses a wildcard or range '
                f'({token!r}); renumbering would change its meaning')

    if kind == 'pair':
        if len(body) < 2:
            raise FFModelUnsupported(f'{command} on L{line_no} is too short')
        ids = sorted((int(body[0]), int(body[1])))
        key = tuple(ids)
        row = [str(ids[0]), str(ids[1])] + body[2:]
    else:
        if not body:
            raise FFModelUnsupported(f'{command} on L{line_no} has no type')
        key = (int(body[0]),)
        row = [body[0]] + body[1:]

    style = _row_style(kind == 'pair', row)
    return key + ((style,) if style else ()), row


def _row_style(pair: bool, row: list):
    """Return the sub-style one hybrid row assigns to, if it names one.

    A ``*_style hybrid`` declaration puts the sub-style name between the
    type ids and the values, so a row of the same type pair under two
    sub-styles carries two distinct parameter sets.

    Parameters
    ----------
    pair : bool
        Whether the row belongs to the pair family (two type ids).
    row : list
        Rendered row tokens (type ids first).

    Returns
    -------
    str or None
        The sub-style name, or None for a single-style row.
    """
    slot = 2 if pair else 1
    if len(row) > slot and not _looks_like_number(str(row[slot])):
        return str(row[slot])
    return None


def _substyle_slot(family: str) -> int:
    """Return the token position a hybrid row names its sub-style at."""
    return 2 if family == 'pair' else 1


def _observed_substyles(coeffs: dict, family: str) -> set:
    """Collect the sub-style names the coefficient rows of a family use.

    Parameters
    ----------
    coeffs : dict
        Coefficient rows by family.
    family : str
        One of :data:`STYLE_FAMILIES`.

    Returns
    -------
    set
        Sub-style names appearing in the rows of that family.
    """
    slot = _substyle_slot(family)
    return {str(row[slot]) for row in coeffs.get(family, [])
            if len(row) > slot and not _looks_like_number(row[slot])}


def _parse_init(text: str, coeffs: dict) -> dict:
    """Read the init template into lines plus its style declarations.

    A ``*_style hybrid`` line lists its sub-styles each followed by that
    sub-style's own arguments, so the line cannot be segmented from its
    own tokens. The coefficient rows name the sub-style they belong to,
    and the untrimmed table lists every declared one, which is what
    makes the split decidable.

    Parameters
    ----------
    text : str
        Init template content ('' when the file is absent).
    coeffs : dict
        Coefficient rows by family, before any trimming.

    Returns
    -------
    dict
        ``{'lines': [...], 'styles': {family: entry}}`` where an entry
        carries its line index, whether it is a hybrid wrapper and its
        ``[sub-style, args]`` segments.

    Raises
    ------
    FFModelUnsupported
        When a hybrid wrapper lists a sub-style no row uses (its
        arguments then cannot be told apart) or lists none at all.
    """
    lines = text.splitlines()
    styles = {}
    for index, raw in enumerate(lines):
        content = raw.split('#')[0].strip()
        tokens = content.split()
        if len(tokens) < 2 or not tokens[0].endswith('_style'):
            continue
        family = tokens[0][:-len('_style')]
        if family not in STYLE_FAMILIES:
            continue
        entry = {'index': index, 'hybrid': False, 'wrapper': '',
                 'segments': [], 'unobserved': False}
        if tokens[1].split('/')[0] == 'hybrid':
            # The wrapper keeps its exact spelling: hybrid/overlay and
            # hybrid scaled differ from plain hybrid in both grammar and
            # physics, and the model has no business choosing between
            # them.
            entry['hybrid'] = True
            entry['wrapper'] = tokens[1]
            names = _observed_substyles(coeffs, family)
            if not names:
                # No coefficient of this family exists at all, so every
                # sub-style it lists is unused; the declaration then
                # needs no segmentation to be answered, and attempting
                # one would only guess where a sub-style's arguments
                # end.
                entry['unobserved'] = True
            else:
                entry['segments'] = _style_segments(
                    tokens[0], tokens[2:], index + 1, names)
        styles[family] = entry
    return {'lines': list(lines), 'styles': styles}


def _style_segments(command: str, tokens: list, line_no: int,
                    known: set) -> list:
    """Split a hybrid argument list into ``[sub-style, args]`` segments.

    Parameters
    ----------
    command : str
        Style command name (audit only).
    tokens : list
        Everything after the hybrid keyword.
    line_no : int
        Source line number (audit only).
    known : set
        Sub-style names the coefficient rows of this family use.

    Returns
    -------
    list
        One ``[name, args]`` pair per declared sub-style.

    Raises
    ------
    FFModelUnsupported
        When a sub-style or its arguments cannot be recognised, which
        would let a declaration be re-rendered with a token attached to
        the wrong sub-style.
    """
    segments = []
    for token in tokens:
        if token in known:
            segments.append([token, []])
        elif not segments:
            raise FFModelUnsupported(
                f'{command} on L{line_no} opens with {token!r}, which no '
                f'coefficient row uses; the hybrid declaration cannot be '
                f'segmented')
        elif not _looks_like_number(token):
            raise FFModelUnsupported(
                f'{command} on L{line_no} passes {token!r} to '
                f'{segments[-1][0]!r}, which is not a number; a sub-style '
                f'no coefficient row names cannot be told from an '
                f'argument')
        else:
            segments[-1][1].append(token)
    if not segments:
        raise FFModelUnsupported(
            f'{command} on L{line_no} declares a hybrid wrapper with no '
            f'sub-style')
    return segments


def _prune_styles(model: dict) -> None:
    """Drop the hybrid sub-styles the trimmed coefficient table starves.

    LAMMPS insists that every sub-style a hybrid wrapper lists is
    assigned to at least one type pair and aborts otherwise, so dropping
    the rows of unused types leaves a declaration that cannot run. A
    sub-style with no surviving row contributes to no pair of the
    surviving types, which makes the trimmed declaration exactly
    equivalent. A family whose last sub-style goes while its types are
    still there cannot be argued away and is rejected instead.

    Parameters
    ----------
    model : dict
        Model with renumbered coefficients (modified in place).

    Raises
    ------
    FFModelUnsupported
        When a hybrid wrapper would keep no sub-style for a family with
        types left to evaluate.
    """
    for family in STYLE_FAMILIES:
        entry = model['init']['styles'].get(family)
        if entry is None or not entry['hybrid']:
            continue
        declared = [segment[0] for segment in entry['segments']]
        note = ''
        if entry['unobserved']:
            # No coefficient row of this family exists at all, so every
            # sub-style the wrapper lists is unused. Whether that is an
            # answer depends on the type census below, and the
            # declaration is never segmented for this case.
            survivors, kept, note = [], [], 'no coefficient row to segment'
        else:
            live = _observed_substyles(model['coeffs'], family)
            survivors = [segment for segment in entry['segments']
                         if segment[0] in live]
            kept = [segment[0] for segment in survivors]
            if len(kept) == len(declared):
                continue

        ntypes = _ntypes(model, family)
        if survivors:
            rendered = [f'{family}_style', entry['wrapper']]
            for name, args in survivors:
                rendered.append(name)
                rendered.extend(args)
        elif family != 'pair' and not ntypes:
            # Nothing of this family is left to evaluate, and 'none' is
            # the style LAMMPS defaults to for it anyway.
            rendered = [f'{family}_style', 'none']
        else:
            raise FFModelUnsupported(
                f'{family}_style hybrid keeps no sub-style while '
                f'{ntypes} {family} types remain to be evaluated')

        _write_style_line(model, family, entry, rendered,
                          dropped=[name for name in declared
                                   if name not in kept],
                          kept=kept or ['none'], note=note)
        entry['segments'] = survivors


def _write_style_line(model: dict, family: str, entry: dict, rendered: list,
                      dropped: list, kept: list, note: str = '') -> None:
    """Record one trimmed style declaration in the model's init lines.

    Parameters
    ----------
    model : dict
        Model being built (modified in place).
    family : str
        Style family the declaration belongs to.
    entry : dict
        The parsed style entry, carrying the line index.
    rendered : list
        Tokens of the trimmed command.
    dropped : list
        Sub-style names removed from the declaration.
    kept : list
        Sub-style names the trimmed declaration keeps.
    note : str
        Why the declaration was never segmented, when that is the case.
    """
    lines = model['init']['lines']
    index = entry['index']
    audit = {'family': family, 'line': index + 1, 'dropped': dropped,
             'kept': kept}
    if note:
        audit['note'] = note
    model['audit'].setdefault('pruned_substyles', []).append(audit)

    content, sep, comment = lines[index].partition('#')
    line = ' '.join(rendered)
    if comment:
        line += (' ' if content.endswith(' ') else '') + sep + comment
    lines[index] = line


def _ntypes(model: dict, family: str) -> int:
    """Return how many types of one style family the system declares.

    The pair family is counted by atom types, which is what its data
    column and its coefficient rows share.
    """
    key = 'natom_types' if family == 'pair' else f'n{family}_types'
    return model['counts'][key]


def _coeff_key(row) -> str:
    """Return a JSON-stable key for one rendered coefficient row.

    Mirrors the dedup key of :func:`_parse_coeff_row`, which counts the
    sub-style a hybrid row assigns to as part of the identity.
    """
    pair = _is_pair(row)
    parts = list(row[:2 if pair else 1])
    style = _row_style(pair, row)
    if style is not None:
        parts.append(style)
    return ' '.join(str(token) for token in parts)


def _is_pair(row) -> bool:
    """Tell a pair row (two type ids) from a bonded row (one)."""
    return len(row) > 2 and _looks_like_int(row[0]) and _looks_like_int(row[1])


def _looks_like_int(token) -> bool:
    """True when a token is a plain integer."""
    try:
        int(token)
    except (TypeError, ValueError):
        return False
    return True


def _looks_like_number(token) -> bool:
    """True when a token is an integer or a decimal cutoff value."""
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


def _join_with_comment(tokens: list, sep: str, comment: str) -> str:
    """Rejoin a data row with its trailing comment.

    The separator gets its own space back: ``1 1.008# hw`` would make
    LAMMPS read the comment as part of the mass token.

    Parameters
    ----------
    tokens : list of str
        Row tokens with the comment stripped.
    sep : str
        Partition separator ('#' when a comment was present).
    comment : str
        Comment text including its leading space.

    Returns
    -------
    str
        The reassembled row.
    """
    if not sep:
        return ' '.join(tokens)
    return ' '.join(tokens) + ' ' + sep + comment


def _resolve_constraints(fixes, groups, config, profile, audit) -> dict:
    """Decide the constraint state and renumber its declarations.

    Parameters
    ----------
    fixes : list
        ``(line_no, tokens)`` of every fix line.
    groups : dict
        Group name to ``(line_no, tokens)``.
    config : SkillConfig
        Runtime configuration container.
    profile : str
        Compiled workflow profile.
    audit : dict
        Audit block, extended in place.

    Returns
    -------
    dict
        ``{'enabled', 'engine', 'fix_id', 'decl', 'group', 'lines'}``
        where ``lines`` are the renumbered declarations to emit (empty
        when disabled).

    Raises
    ------
    FFModelUnsupported
        When a fix or group references a type in a shape this module
        cannot renumber.
    """
    shake_fix_id = str(getattr(config, 'shake_fix_id', '') or '')
    engine = ('lammps' if profile in ENGINE_INTEGRATED_PROFILES
              else 'framework')
    requested = bool(int(getattr(config, 'lammps_shake', 0) or 0))
    enabled = requested and engine == 'lammps'

    shake_fix = None
    for line_no, tokens in fixes:
        style = tokens[2] if len(tokens) > 2 else ''
        if (shake_fix_id and tokens[1] == shake_fix_id) or 'shake' in (
                style, tokens[1:]):
            shake_fix = (line_no, tokens)
            break

    lines = []
    decl = ''
    group_name = ''
    if shake_fix is not None:
        line_no, tokens = shake_fix
        if enabled:
            decl = ' '.join(tokens[1:])
            # A 'fix <id> <group> shake ...' line names its group third.
            group_name = tokens[2] if len(tokens) > 2 else ''
            lines.append(_renumber_fix(tokens))
            for name, (_g_line, g_tokens) in groups.items():
                if name == group_name:
                    lines.insert(0, _renumber_group(g_tokens))
        else:
            if engine == 'lammps':
                reason = ('lammps_shake=0 asks for the constraint off, so '
                          'the template fix is dropped rather than issued')
            else:
                reason = (f'{engine} integrates the atoms, so the '
                          f'constraint is its side to declare (the '
                          f'framework constraint block); a fix here would '
                          f'constrain a configuration nothing propagates')
            audit['dropped_constraints'].append(
                {'line': line_no, 'reason': reason})

    return {'enabled': enabled, 'engine': engine,
            'fix_id': shake_fix_id, 'decl': decl, 'group': group_name,
            'lines': lines}


def _renumber_group(tokens: list) -> str:
    """Renumber the type ids of a 'group <name> type ...' line."""
    if tokens[2] != 'type':
        raise FFModelUnsupported(
            f'group {tokens[1]!r} selects by {tokens[2]!r}, not by type')
    return ' '.join(tokens[:3] + ['{atom:%s}' % token for token in tokens[3:]])


def _renumber_fix(tokens: list) -> str:
    """Renumber the 'b <bond type>' and 'a <angle type>' markers."""
    markers = {'b': 'bond', 'a': 'angle', 't': 'dihedral', 'r': 'improper'}
    out = []
    n = 0
    while n < len(tokens):
        token = tokens[n]
        if token in markers and n + 1 < len(tokens):
            following = tokens[n + 1]
            if not _looks_like_int(following):
                raise FFModelUnsupported(
                    f'fix {tokens[1]!r} marker {token!r} is followed by '
                    f'{following!r}, which is not a single type id')
            out.append(token)
            out.append('{%s:%s}' % (markers[token], following))
            n += 2
            continue
        out.append(token)
        n += 1
    return ' '.join(out)


def _build_type_maps(model: dict) -> None:
    """Derive the used-type sets and the renumbering maps in place.

    Parameters
    ----------
    model : dict
        Model being built (modified in place).

    Raises
    ------
    FFModelUnsupported
        When a referenced type has no coefficient, has no mass, or when
        a declared type count disagrees with the section rows.
    """
    sections = model['sections']
    declared = sections['counts']
    used = {'atom': [], 'bond': [], 'angle': [], 'dihedral': [],
            'improper': []}

    for name, kind, column in TYPE_SECTIONS:
        rows = sections.get(name) or []
        for raw in rows:
            tokens = raw.split('#')[0].split()
            if len(tokens) <= column:
                raise FFModelUnsupported(
                    f'{name} row is too short to carry a type: {raw!r}')
            value = int(tokens[column])
            if value not in used[kind]:
                used[kind].append(value)

    used['atom'] = sorted(used['atom'])
    for kind in ('bond', 'angle', 'dihedral', 'improper'):
        used[kind] = sorted(used[kind])

    masses = {}
    for raw in sections.get('Masses') or []:
        tokens = raw.split('#')[0].split()
        if len(tokens) < 2:
            continue
        masses[int(tokens[0])] = raw
    missing = [t for t in used['atom'] if t not in masses]
    if missing:
        raise FFModelUnsupported(f'atom types {missing} have no Masses entry')

    maps = {'atom': {t: i + 1 for i, t in enumerate(used['atom'])},
            'bond': {t: i + 1 for i, t in enumerate(used['bond'])},
            'angle': {t: i + 1 for i, t in enumerate(used['angle'])},
            'dihedral': {t: i + 1 for i, t in enumerate(used['dihedral'])},
            'improper': {t: i + 1 for i, t in enumerate(used['improper'])}}

    counts = {
        'natom': len(sections.get('Atoms') or []),
        'nbond': len(sections.get('Bonds') or []),
        'nangle': len(sections.get('Angles') or []),
        'ndihedral': len(sections.get('Dihedrals') or []),
        'nimproper': len(sections.get('Impropers') or []),
        'natom_types': len(maps['atom']),
        'nbond_types': len(maps['bond']),
        'nangle_types': len(maps['angle']),
        'ndihedral_types': len(maps['dihedral']),
        'nimproper_types': len(maps['improper']),
    }
    for label, key in (('atoms', 'natom'), ('bonds', 'nbond'),
                       ('angles', 'nangle'), ('dihedrals', 'ndihedral'),
                       ('impropers', 'nimproper')):
        if declared[label] != counts[key]:
            raise FFModelUnsupported(
                f'data declares {declared[label]} {label} but the section '
                f'holds {counts[key]} rows')

    model['used_types'] = used
    model['type_map'] = {kind: {str(k): v for k, v in mapping.items()}
                         for kind, mapping in maps.items()}
    model['counts'] = counts
    raw_masses = [masses[t] for t in used['atom']]
    model['masses'] = [[maps['atom'][t], float(masses[t].split()[1])]
                       for t in used['atom']]
    model['box'] = sections['box']

    for name, kind, column in TYPE_SECTIONS:
        rows = sections.get(name) or []
        rendered = []
        for raw in rows:
            content, sep, comment = raw.partition('#')
            tokens = content.split()
            value = int(tokens[column])
            if value not in maps[kind]:
                raise FFModelUnsupported(
                    f'{name} references {kind} type {value}, which has no '
                    f'mapping')
            tokens[column] = str(maps[kind][value])
            rendered.append(_join_with_comment(tokens, sep, comment))
        sections[name] = rendered

    mass_rows = []
    for new_id, raw in zip((row[0] for row in model['masses']), raw_masses):
        content, sep, comment = raw.partition('#')
        tokens = content.split()
        tokens[0] = str(new_id)
        mass_rows.append(_join_with_comment(tokens, sep, comment))
    sections['Masses'] = mass_rows

    model['coeffs'], model['audit']['provenance'], dropped = (
        _renumber_coeffs(model['coeffs'], maps, model['audit']['provenance']))
    model['audit']['dropped_unused_coeffs'] = dropped
    model['constraints']['lines'] = [
        _apply_map(line, maps) for line in model['constraints']['lines']]


def _renumber_coeffs(coeffs: dict, maps: dict, provenance: dict):
    """Renumber the type ids of every coefficient row.

    Parameters
    ----------
    coeffs : dict
        Rendered rows by kind.
    maps : dict
        Type maps by kind.
    provenance : dict
        Source line number by kind and original row key.

    Returns
    -------
    tuple of (dict, dict, int)
        (renumbered rows, provenance keyed by the renumbered rows,
        number of rows dropped because they reference an unused type).
    """
    out = {}
    out_provenance = {}
    dropped = 0
    for kind, rows in coeffs.items():
        rendered = []
        new_provenance = {}
        mapping = maps['atom'] if kind == 'pair' else maps.get(kind, {})
        ids = (0, 1) if kind == 'pair' else (0,)
        for row in rows:
            tokens = list(row)
            if any(int(tokens[p]) not in mapping for p in ids):
                # Coefficients of types no atom/bond/angle references are
                # dropped: the declared type count, not the row count, is
                # what drives LAMMPS pair mixing.
                dropped += 1
                continue
            for position in ids:
                tokens[position] = str(mapping[int(tokens[position])])
            rendered.append(tokens)
            line = provenance.get(kind, {}).get(_coeff_key(row))
            if line is not None:
                new_provenance[_coeff_key(tokens)] = line
        out[kind] = rendered
        out_provenance[kind] = new_provenance
    return out, out_provenance, dropped


def _apply_map(line: str, maps: dict) -> str:
    """Substitute the '{kind:id}' placeholders of a constraint line."""
    rendered = line
    for kind, mapping in maps.items():
        for old, new in mapping.items():
            rendered = rendered.replace('{%s:%s}' % (kind, old), str(new))
    if '{' in rendered:
        raise FFModelUnsupported(
            f'constraint line keeps an unresolved placeholder: {line!r}')
    return rendered


def _pair_mix_rule(model: dict):
    """Return the `pair_modify mix` rule the init template asks for.

    A box that uses two atom types needs a coefficient for that combination.
    LAMMPS derives it from the two diagonal rows when the template asks for
    mixing (`pair_modify mix arithmetic|geometric`), which is how the shipped
    ethanol template runs on the verbatim path listing only same-type
    `pair_coeff` rows. With no mixing rule a missing cross row is a real hole,
    so the coverage check below relaxes only when this returns a rule name.

    Returns
    -------
    str or None
        The rule name, or None for no statement or `mix none`.
    """
    for raw in (model['init'].get('lines') or []):
        tokens = raw.split('#')[0].split()
        if len(tokens) >= 3 and tokens[0] == 'pair_modify' \
                and tokens[1] == 'mix':
            rule = tokens[2].lower()
            return None if rule == 'none' else rule
    return None


def _validate(model: dict, config) -> None:
    """Check that the compressed model can describe the system.

    Parameters
    ----------
    model : dict
        Built model.
    config : SkillConfig
        Runtime configuration container.

    Raises
    ------
    FFModelUnsupported
        When a used type has no coefficient, the maps are not
        contiguous, or a required coefficient kind is empty.
    """
    used = model['used_types']
    coeffs = model['coeffs']
    # The JSON-safe type_map carries string keys; restore the integers.
    maps = {kind: {int(old): new for old, new in mapping.items()}
            for kind, mapping in model['type_map'].items()}

    for kind in ('atom', 'bond', 'angle'):
        if not used[kind]:
            continue
        expected = sorted(range(1, len(maps[kind]) + 1))
        if sorted(int(v) for v in maps[kind].values()) != expected:
            raise FFModelUnsupported(f'{kind} type map is not contiguous')

    pairs = {tuple(int(token) for token in row[:2]) for row in coeffs['pair']}
    # The diagonal rows a mixing rule can combine, tagged with the sub-style
    # each names. A type that carries two different sub-styles has no single
    # cross term to derive, so it is dropped here and its pairs stay required.
    mix = _pair_mix_rule(model)
    diagonal = {}
    for row in coeffs['pair']:
        tokens = [int(token) for token in row[:2]]
        if tokens[0] != tokens[1] or len(row) < 3:
            continue
        for index in tokens:
            if diagonal.get(index, row[2]) != row[2]:
                diagonal.pop(index, None)
            else:
                diagonal[index] = row[2]

    for i in used['atom']:
        for j in used['atom']:
            key = tuple(sorted((maps['atom'][i], maps['atom'][j])))
            if key in pairs:
                continue
            derivable = (mix is not None and key[0] != key[1]
                         and key[0] in diagonal
                         and diagonal[key[0]] == diagonal.get(key[1]))
            if not derivable:
                raise FFModelUnsupported(
                    f'atom types {i}/{j} have no pair_coeff entry')

    for kind, section in (('bond', 'bond'), ('angle', 'angle')):
        if used[section] and not coeffs[kind]:
            raise FFModelUnsupported(
                f'{section} types are used but no {kind}_coeff survived')

    # A hybrid wrapper may not survive a sub-style the trimmed table
    # starves: LAMMPS aborts at run time with "Pair hybrid sub-style is
    # not used". _prune_styles rewrites the declarations to keep this
    # true, so reaching here with a starved name means a declaration and
    # its coefficients came from different renderings, which is a
    # physics change no writer may make silently.
    for family, entry in model['init']['styles'].items():
        if not entry['hybrid'] or entry.get('unobserved'):
            continue
        starved = ({segment[0] for segment in entry['segments']}
                   - _observed_substyles(coeffs, family))
        if starved:
            raise FFModelUnsupported(
                f"{family}_style {entry['wrapper']} still declares "
                f"{sorted(starved)}, for which no {family}_coeff row "
                f"survives trimming")


# --------------------------------------------------------------------
# QM/MM model surgery
#
# qmmm_field needs the exact Coulomb surface of the qmmm_energy model
# with every non-Coulomb pair term removed. Rewriting the pair style to
# a single coul/long would change the per-sub-style cutoffs and special
# bonds, so the model keeps its declarations and zeroes the parameters
# instead. qmmm_energy keeps the Q-M and M-M non-Coulomb terms but must
# remove the Q-Q ones, because the QM calculation owns the QM region's
# internal non-bonded interaction.
# --------------------------------------------------------------------

# Pair styles whose first coefficient is the LJ epsilon.
_LJ_PAIR_STYLES = (
    'lj/cut', 'lj/cut/coul/cut', 'lj/cut/coul/long',
    'lj/charmm/coul/cut', 'lj/charmm/coul/charmm',
    'lj/charmm/coul/long', 'lj/expand/coul/long',
    'lj/smooth/linear', 'lj/smooth/coul/long',
)

# Buck styles carry A, B and C; all three are non-Coulomb.
_BUCK_PAIR_STYLES = ('buck', 'buck/coul/cut', 'buck/coul/long',
                     'buck/long/coul/long')

# Coulomb-only and null styles have nothing to zero.
_NO_NONCOULOMB_STYLES = (
    'coul/cut', 'coul/long', 'coul/dsf', 'coul/wolf',
    'coul/wolf/cut', 'zero', 'none',
)


def _pair_style_of_row(model: dict, row: list) -> str:
    """Return the pair sub-style a coefficient row belongs to."""
    style = _row_style(True, row)
    if style:
        return style
    entry = (model.get('init', {}).get('styles', {}) or {}).get('pair')
    if entry is not None:
        tokens = (model['init']['lines'][entry['index']]
                  .split('#')[0].split())
        if len(tokens) >= 2:
            return tokens[1]
    raise FFModelUnsupported(
        'cannot identify the pair style of a coefficient row; the QM/MM '
        'model surgery would guess which parameters to zero')


def _zero_pair_row(model: dict, row: list) -> None:
    """Zero one pair coefficient row's non-Coulomb parameters in place."""
    style = _pair_style_of_row(model, row)
    if style in _NO_NONCOULOMB_STYLES:
        return
    offset = 3 if _row_style(True, row) else 2
    if style in _LJ_PAIR_STYLES:
        if len(row) <= offset:
            raise FFModelUnsupported(
                f'pair style {style!r} has no epsilon parameter to zero')
        row[offset] = '0.0'
        return
    if style in _BUCK_PAIR_STYLES:
        if len(row) < offset + 3:
            raise FFModelUnsupported(
                f'pair style {style!r} has fewer than three Buck '
                f'parameters')
        for index in range(offset, offset + 3):
            row[index] = '0.0'
        return
    raise FFModelUnsupported(
        f'pair style {style!r} is not handled by the QM/MM model '
        f'surgery; refusing to guess which parameters are non-Coulomb')


def zero_non_coulomb_in_model(model: dict) -> None:
    """Zero every non-Coulomb pair parameter in a qmmm_field model.

    The pair style names, hybrid sub-style assignment, cutoffs,
    special_bonds and kspace declarations stay untouched; only the
    dispersion/repulsion parameters are replaced by zero.
    """
    rows = (model.get('coeffs', {}) or {}).get('pair', [])
    for row in rows:
        _zero_pair_row(model, row)
    model.setdefault('audit', {})['qmmm_field_non_coulomb_zeroed'] = True


def _qm_type_set(model: dict, qm_ids) -> set:
    """Return the LAMMPS atom types used by the QM atom ids."""
    wanted = {int(value) for value in qm_ids}
    types = set()
    for raw in (model.get('sections', {}) or {}).get('atoms', []) or []:
        tokens = raw.split('#')[0].split()
        if len(tokens) < 3 or not tokens[0].isdigit():
            continue
        atom_id = int(tokens[0])
        if atom_id in wanted:
            try:
                types.add(int(tokens[2]))
            except (IndexError, ValueError):
                raise FFModelUnsupported(
                    f'cannot read the atom type of QM atom {atom_id}')
    if len(types) == 0 and wanted:
        raise FFModelUnsupported(
            'no Atoms row matches the QM atom ids; the Q-Q non-bonded '
            'term cannot be identified')
    return types



def split_qm_types(model: dict, qm_ids) -> dict:
    """Clone every QM atom's LAMMPS type into a new, QM-only type.

    LAMMPS pair coefficients are type-based. QM and MM atoms normally
    share types, so zeroing a Q-Q type pair would also zero Q-M and M-M.
    Cloning the QM types gives the Q-Q pairs their own rows; the Q-M rows
    are copied from the original Q-type/M-type rows (same pair style, LJ
    and Coulomb parameters), so only the Q-Q non-Coulomb parameters are
    removed later by :func:`zero_qm_new_pairs`.

    The data sections are updated in place: the QM Atoms rows get the new
    type, one Masses row is appended per new type, and the pair-coefficient
    rows for the new types are copied. The function refuses rather than
    invents a missing original row, because a guessed Q-M LJ parameter
    would silently change the MM force field.

    Returns
    -------
    dict
        ``{old_type: new_type}``.
    """
    from .qmmm import _charge_index

    sections = model.get('sections') or {}
    atom_rows = sections.get('Atoms') or []
    if not atom_rows:
        raise FFModelUnsupported('cannot split QM types: no Atoms rows')
    wanted = {int(value) for value in qm_ids}
    old_types = set()
    parsed = []
    for index, raw in enumerate(atom_rows):
        content, sep, comment = raw.partition('#')
        tokens = content.split()
        if not tokens or not tokens[0].isdigit():
            continue
        atom_id = int(tokens[0])
        type_column = _charge_index(model, tokens) - 1
        if type_column < 0 or type_column >= len(tokens):
            raise FFModelUnsupported(
                f'cannot locate the type of QM atom {atom_id}')
        atom_type = int(tokens[type_column])
        parsed.append((index, atom_id, type_column, atom_type, sep, comment))
        if atom_id in wanted:
            old_types.add(atom_type)
    if not old_types:
        raise FFModelUnsupported(
            'cannot split QM types: no QM atom found in the Atoms section')

    counts = model.setdefault('counts', {})
    max_type = int(counts.get('atom types', 0) or 0)
    for row in sections.get('Masses') or []:
        tokens = row.split('#')[0].split()
        if tokens and tokens[0].isdigit():
            max_type = max(max_type, int(tokens[0]))
    mapping = {old: max_type + offset + 1
               for offset, old in enumerate(sorted(old_types))}

    for index, atom_id, type_column, atom_type, sep, comment in parsed:
        if atom_id not in wanted:
            continue
        content = sections['Atoms'][index].split('#')[0]
        tokens = content.split()
        tokens[type_column] = str(mapping[atom_type])
        sections['Atoms'][index] = ' '.join(tokens) + (
            ' ' + sep + comment if sep else '')

    mass_of = {}
    mass_rows = sections.setdefault('Masses', [])
    for row in mass_rows:
        tokens = row.split('#')[0].split()
        if tokens and tokens[0].isdigit() and len(tokens) >= 2:
            mass_of[int(tokens[0])] = tokens[1]
    for old, new in sorted(mapping.items()):
        if old not in mass_of:
            raise FFModelUnsupported(
                f'cannot clone type {old}: its Masses row is missing')
        mass_rows.append(f'{new} {mass_of[old]}   # cloned from type {old}')

    pair_rows = (model.get('coeffs', {}) or {}).get('pair', [])
    existing = set()
    for row in pair_rows:
        if len(row) >= 2 and str(row[0]).isdigit() and str(row[1]).isdigit():
            style = _row_style(True, row)
            existing.add((min(int(row[0]), int(row[1])),
                          max(int(row[0]), int(row[1])), style))
    additions = []
    pending = set()
    for row in list(pair_rows):
        if len(row) < 2:
            continue
        try:
            first, second = int(row[0]), int(row[1])
        except (TypeError, ValueError):
            continue
        if first not in old_types and second not in old_types:
            continue
        # A type row does not say whether its partners are QM or MM:
        # the old type is shared by both. Copy the row into every
        # combination the split creates:
        #   newQ-old  (Q-M, kept)
        #   old-newQ  (M-Q, the same symmetric row)
        #   newQ-newQ (Q-Q, zeroed later)
        candidates = []
        if first in old_types:
            candidates.append((mapping[first], second))
            if second in old_types:
                candidates.append((mapping[first], mapping[second]))
        if second in old_types:
            candidates.append((first, mapping[second]))
        for a, b in candidates:
            key = (min(a, b), max(a, b), _row_style(True, row))
            # `pending` is the key set of `additions`, maintained as rows
            # are added: rebuilding it inside this loop made the pass
            # quadratic in the number of copied rows for no new answer.
            if key in existing or key in pending:
                continue
            copied = list(row)
            copied[0], copied[1] = str(key[0]), str(key[1])
            additions.append(copied)
            pending.add(key)
    if not additions:
        raise FFModelUnsupported(
            'cannot split QM types: no pair-coefficient row involves a QM '
            'type, so the Q-Q and Q-M rows cannot be separated')
    pair_rows.extend(additions)
    counts['atom types'] = max(mapping.values())
    model.setdefault('audit', {})['qmmm_type_split'] = {
        str(old): new for old, new in sorted(mapping.items())}
    return mapping


def zero_qm_new_pairs(model: dict, new_types) -> None:
    """Zero non-Coulomb parameters on the cloned Q-Q pair rows only."""
    wanted = {int(value) for value in new_types}
    if not wanted:
        raise FFModelUnsupported(
            'qmmm_energy has no cloned QM types; the Q-Q non-Coulomb term '
            'cannot be isolated')
    zeroed = 0
    for row in (model.get('coeffs', {}) or {}).get('pair', []):
        if len(row) < 2:
            continue
        try:
            first, second = int(row[0]), int(row[1])
        except (TypeError, ValueError):
            continue
        if first in wanted and second in wanted:
            _zero_pair_row(model, row)
            zeroed += 1
    if zeroed == 0:
        raise FFModelUnsupported(
            'no cloned Q-Q pair row was found after the type split')
    model.setdefault('audit', {})['qmmm_energy_qm_nonbonded_zeroed'] = zeroed
