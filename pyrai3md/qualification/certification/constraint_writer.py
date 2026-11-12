"""
======================================================================
Classic `&constraint` writer used by the qualification carriers.

This is the inverse of `schema.adapter._read_constraint_section`. It is
not part of the runtime: a run reads the structured constraint block and
never writes one for execution. It lives under qualification so the
conversion tests that round-trip a YAML/JSON constraint block can use it
without keeping a 400-line writer in the schema package.

Nov 09 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

from ...schema.adapter import (
    CONSTRAINT_FAMILIES,
    CONSTRAINT_NULLABLE_KEYS,
    CONSTRAINT_OBJECT_KEYS,
    CONSTRAINT_PARAM_KEYS,
    CONSTRAINT_RAMP_DELAY_DEFAULT,
    CONSTRAINT_RAMP_KEYS,
    CONSTRAINT_RAMP_SUFFIXES,
    CONSTRAINT_UNIT_KEYS,
)


CONSTRAINT_WRITABLE_SECTIONS = ('groups', 'cvs', 'constraints', 'biases',
                                'walls', 'fixes', 'metadynamics')


# The record model one section's structured spelling is validated against, so
# the writer can ask that model what a missing field means.
_CONSTRAINT_WRITE_MODELS = {
    'groups': 'GroupInput',
    'cvs': 'CVInput',
    'constraints': 'HardConstraintInput',
    'biases': 'BiasInput',
    'walls': 'WallInput',
    'fixes': 'FixInput',
    'metadynamics': 'MetadynamicsInput',
}


def _constraint_write_object_key(section_key, record):
    """Which object line one structured record is written by."""
    if section_key == 'groups':
        return 'group_name'
    if section_key == 'cvs':
        _require_field(record, 'atoms', 'cv_bond')
        return {'bond': 'cv_bond', 'angle': 'cv_angle',
                'dihedral': 'cv_dihedral'}[str(record.get('kind') or '')]
    if section_key == 'constraints':
        if record.get('bond_type'):
            return 'shake_bond_type'
        kind = str(record.get('kind') or 'bond')
        return {'bond': 'shake_bond', 'angle': 'shake_angle',
                'dihedral': 'shake_dihedral'}[kind]
    if section_key == 'biases':
        if record.get('cv'):
            return 'bias_cv'
        _require_field(record, 'atoms', 'bias_bond')
        return {2: 'bias_bond', 3: 'bias_angle',
                4: 'bias_dihedral'}[len(record['atoms'])]
    if section_key == 'walls':
        return {'sphere': 'wall_sphere_radius',
                'ellipsoid': 'wall_ellipsoid_semiaxes',
                'box': 'wall_box_lengths',
                'triclinic': 'wall_triclinic_cell'}[
                    str(record.get('geometry') or 'sphere')]
    if section_key == 'fixes':
        for key, name in (('atoms', 'fix_atoms'), ('element', 'fix_element'),
                          ('group', 'fix_group')):
            if record.get(key):
                return name
        raise ValueError(
            "a fix with no atom selection cannot be written as &constraint "
            'text; fix_atoms, fix_element or fix_group names one')
    if section_key == 'metadynamics':
        return 'meta_rmsd' if str(record.get('kind')) == 'rmsd' else 'meta_cv'
    raise ValueError("no &constraint object line writes a '%s' record"
                     % section_key)


def _require_field(record, field, keyword):
    """The record has to carry the field its object line is written from."""
    if not record.get(field):
        raise ValueError(
            "a %s record carries no '%s', which is what the '%s' line writes; "
            'the flat grammar has no other way to name this object'
            % (record.get('id') or 'constraint', field, keyword))


def _constraint_write_object_value(key, record):
    """The one segment of one object line, read from the record's own field.

    The object table says which field a keyword fills, so the writer needs no
    second map of it: a grammar that kept its own list of what to print would
    drift from the list of what it reads.
    """
    _family, field, _kind, _extra = CONSTRAINT_OBJECT_KEYS[key]
    value = record.get(field)
    if value is None or value == '' or value == []:
        raise ValueError(
            "a %s record carries no '%s', which is what the '%s' line writes; "
            'the flat grammar has no other way to name this object'
            % (record.get('id') or key, field, key))
    if field == 'bonds':
        # One constraint object may hold several rows (one named block of
        # bonds), and the segment states them in order: `shake_bond 1 2 1 3` is
        # the two bonds of one block, while `shake_bond 1 2; 1 3` is two
        # objects. Writing only the first row would silently drop a bond.
        rows = value if isinstance(value[0], (list, tuple)) else [value]
        return _constraint_join([atom for row in rows for atom in row])
    if field == 'cvs':
        return ', '.join(str(item) for item in value)
    if field in ('semiaxes', 'lengths', 'cell'):
        numbers = value
        if isinstance(numbers[0], (list, tuple)):
            numbers = [item for row in numbers for item in row]
        return ' '.join(_constraint_number(item) for item in numbers)
    if field in ('atoms', 'element'):
        return _constraint_join(value)
    if isinstance(value, float):
        return _constraint_number(value)
    return str(value)


def _constraint_join(values):
    """Atom numbers of one record as tokens, ranges left as they are."""
    return ' '.join(str(int(item)) if not isinstance(item, str) else item
                    for item in (values or []))


def _constraint_write_params(section_key, records, keys):
    """One keyword per parameter, its segments in the order of its group."""
    pairs = []
    for keyword, spec in CONSTRAINT_PARAM_KEYS.items():
        family, field, _kind, group = spec
        if family != section_key:
            continue
        positions = [index for index, key in enumerate(keys)
                     if group is None or key in group]
        if not positions:
            continue
        values = []
        declared = []
        for index in positions:
            value = records[index].get(field)
            if not _constraint_is_default(section_key, field, value):
                values.append(_constraint_write_value(field, value))
                declared.append(True)
                continue
            # A record that leaves a parameter out still needs a segment,
            # because the line is positional. The model's own default is the
            # value it means, and writing it out says so in the file rather
            # than letting the padding rule decide.
            default = _constraint_field_default(section_key, field)
            if default is None and keyword in CONSTRAINT_NULLABLE_KEYS:
                # This object declares nothing while another does, and the gap
                # has to be stated rather than invented: `none` reads back as
                # "no value here", which is what the record holds.
                values.append('none')
                declared.append(False)
                continue
            values.append(None if default is None else
                          _constraint_write_value(field, default))
            declared.append(False)
        if not any(declared):
            # Nothing in the input named this parameter: the records all mean
            # the model's default, and a line that prints defaults the run
            # would apply anyway is noise between the reader and the file.
            continue
        if any(value is None for value in values):
            hint = ('; name them all or none, since an unnamed id is minted '
                    'by position' if field == 'id' else '')
            raise ValueError(
                "'%s' is given for some of the %s objects and left unset for "
                'the rest, which no positional line can state%s'
                % (keyword, section_key, hint))
        pairs.append((keyword, '; '.join(values)))
    return pairs


def _constraint_field_default(section_key, field):
    """The model default of one record field, or None when it holds nothing.

    `get_default` is asked with the factory called, because a list field
    declares a `default_factory` and its `default` is pydantic's placeholder
    for "none": reading the attribute directly would compare a record's atoms
    against that placeholder and call every empty list a declaration.
    """
    from pyrai3md.schema import config as models
    model = _CONSTRAINT_WRITE_MODELS[section_key]
    info = getattr(models, model).model_fields[field]
    try:
        default = info.get_default(call_default_factory=True)
    except (TypeError, ValueError):
        return None
    if default is None or default == '' or default == [] or default is False:
        return None
    return default


def _constraint_number(value):
    """One number as the shortest text that reads back as the same double.

    `%g` stops at six significant digits, and a wall center or an rmsd
    reference block is stored at full precision: an input file that rendered
    4.199546 as 4.19955 would be a different run by five parts per million,
    which is exactly the drift a round trip is supposed to exclude.
    """
    return repr(float(value))


def _constraint_write_value(field, value):
    """One parameter value as the text its keyword reads."""
    if field == 'atoms':
        return _constraint_join(value)
    if field == 'element':
        return ' '.join(str(item) for item in value)
    if field == 'molecules':
        return _constraint_molecules_text(value)
    if isinstance(value, (list, tuple)):
        return ' '.join(_constraint_number(item) if isinstance(item, float)
                        else str(item) for item in value)
    if isinstance(value, float):
        return _constraint_number(value)
    return str(value)


def _constraint_molecules_text(value):
    """A molecule partition as one `wall_groups` segment.

    The reader cuts a list by the size it carries at the end, so the writer can
    only state a partition whose molecules are all the same length. Unequal
    molecules have no text form, and inventing one (a per-molecule trailing
    number, a second separator) would be a new grammar rather than the inverse of
    the one that reads.
    """
    rows = [list(item) if isinstance(item, (list, tuple)) else [item]
            for item in value]
    sizes = set(len(row) for row in rows)
    if len(sizes) > 1:
        raise ValueError(
            "'wall_groups' gives molecules of %s atoms; the flat grammar states "
            'one size per segment as its trailing number, so a partition of '
            'unequal molecules cannot be written as text'
            % ' and '.join(str(size) for size in sorted(sizes)))
    flat = [item for row in rows for item in row]
    return _constraint_join(flat + [sorted(sizes)[0]])


def _constraint_write_ramps(records, keys, section_key):
    """The `_start/_end/_scale/_delay/_stop` families of one family.

    A ramp is written on the key it ramps, as exact endpoints. Where the reader
    composed a vector from component keys the writer emits the whole vector:
    both spellings read back as the same two endpoints, and the vector form is
    the one that cannot put a component of a ramp in the wrong slot.
    """
    pairs = []
    for base, spec in CONSTRAINT_RAMP_KEYS.items():
        family, parameter, _kind, group = spec
        if family != section_key:
            continue
        positions = [index for index, key in enumerate(keys)
                     if group is None or key in group]
        rows = []
        for index in positions:
            found = [item for item in (records[index].get('schedules') or ())
                     if str(item.get('parameter')) == parameter]
            if len(found) > 1:
                raise ValueError(
                    "'%s' ramps '%s' %d times, which one keyword cannot state; "
                    'the grammar reads one schedule per parameter and object'
                    % (records[index].get('id') or base, parameter,
                       len(found)))
            rows.append(found[0] if found else None)
        if not any(rows):
            continue
        if any(row is None for row in rows):
            raise ValueError(
                "'%s' ramps some of the %s objects and not the others: a ramp "
                'keyword gives one value per object in order, so a gap would '
                'be read as the next object\'s endpoint'
                % (base, section_key))
        columns = dict((suffix, []) for suffix in CONSTRAINT_RAMP_SUFFIXES)
        for row in rows:
            _append_ramp_row(base, row, columns)
        for suffix in CONSTRAINT_RAMP_SUFFIXES:
            values = columns[suffix]
            if not any(value is not None for value in values):
                continue
            if any(value is None for value in values):
                raise ValueError(
                    "'%s%s' is given for some objects and not the others; the "
                    'ramp keywords are positional' % (base, suffix))
            pairs.append((base + suffix, '; '.join(values)))
    return pairs


def _append_ramp_row(base, row, columns):
    """Fold one inline schedule back into its five keyword columns."""
    scale = row.get('scale')
    value = row.get('value')
    if scale is not None and value is not None:
        raise ValueError(
            "'%s' ramp gives both factors and endpoints, which no keyword "
            'pair reads back' % base)
    if scale is not None:
        columns['_scale'].append(' '.join(_constraint_number(item)
                                          for item in scale))
        columns['_start'].append(None)
        columns['_end'].append(None)
    elif value is not None:
        columns['_scale'].append(None)
        columns['_start'].append(_constraint_endpoint_text(value[0]))
        columns['_end'].append(_constraint_endpoint_text(value[1]))
    else:
        raise ValueError("'%s' ramp carries neither endpoints nor factors, so "
                         'no keyword can write it' % base)
    columns['_delay'].append('%d' % int(row.get('start_step',
                                                CONSTRAINT_RAMP_DELAY_DEFAULT)))
    end_step = row.get('end_step')
    columns['_stop'].append(None if end_step is None else '%d' % int(end_step))


def _constraint_endpoint_text(item):
    """One ramp endpoint as text: a number, or the vector it ramps."""
    if isinstance(item, (list, tuple)):
        if len(item) == 3 and all(isinstance(row, (list, tuple))
                                  for row in item):
            return ' '.join(_constraint_endpoint_text(row) for row in item)
        return ' '.join(_constraint_number(value) for value in item)
    return _constraint_number(item)


def _constraint_is_default(section_key, field, value):
    """True when a record does not carry the field, in any of its spellings."""
    if value is None or value == '' or value == [] or value is False:
        return True
    from pyrai3md.schema import config as models
    model = getattr(models, _CONSTRAINT_WRITE_MODELS[section_key])
    if field not in model.model_fields:
        # A key the record model does not have is not a field that happens to
        # be set: it is a spelling that was retired, and a writer that guessed
        # here would emit a line the reader cannot give back.
        raise ValueError(
            "'%s' is not a field of a %s record; the flat grammar has no key "
            'that writes it either, so the declaration has to go (or move to '
            'the keyword that states the same thing)'
            % (field, section_key))
    default = model.model_fields[field].default
    return value == default


def _constraint_field_writable(section_key, name, value, key=None):
    """Whether one record field has a keyword that reads it back.

    `key` is the object line the record is written by, which carries two kinds
    of field itself: the one its segments fill (`atoms`, `radius`) and the ones
    it implies (`kind`, `geometry`, the rmsd history's `kind`). Neither needs a
    parameter line, so both count as written.
    """
    if name == 'schedules':
        # The inline ramps, which the five keyword families write back.
        return True
    if name == 'id':
        return True
    if key is not None:
        _family, field, _kind, extra = CONSTRAINT_OBJECT_KEYS[key]
        if name == field or name in dict(extra):
            return True
    for keyword, spec in CONSTRAINT_PARAM_KEYS.items():
        if spec[0] == section_key and spec[1] == name:
            return True
    return _constraint_is_default(section_key, name, value)


def write_constraint_section(block):
    """The structured constraint mapping as `&constraint` keyword pairs.

    The inverse of `_read_constraint_section`, and it refuses rather than
    approximates: a declaration the flat grammar cannot state (one value for
    some objects of a family and none for the rest, two ramps of one parameter,
    a partition of unequal molecules, a shared schedule list) has no faithful
    text form, and an
    input file that reads back as a different run is worse than no input file.

    Parameters
    ----------
    block : dict or None
        The `constraints` mapping of a test config.

    Returns
    -------
    list of tuple
        `(keyword, value)` pairs in the order the grammar reads them, empty when
        the block declares nothing.
    """
    values = dict(block or {})
    carried = set(CONSTRAINT_WRITABLE_SECTIONS) | {'units', 'notes'}
    # An empty field is not a declaration. A config that came out of
    # `model_dump` carries an empty inline ramp list and one empty wall field or
    # other on every run, and refusing those would send a run's own dumped config
    # back as an input error while it declares neither.
    extra = sorted(key for key in set(values) - carried
                   if values[key] not in (None, '', [], {}, 0, False))
    if extra:
        raise ValueError(
            "'%s' is not a field of the constraint block; the writer takes the "
            'structured mapping the reader emits (%s), and the flat keyword '
            'lines are what it emits, not what it reads'
            % (extra[0], ', '.join(sorted(CONSTRAINT_WRITABLE_SECTIONS))))
    if not any(values.get(key) for key in CONSTRAINT_WRITABLE_SECTIONS):
        return []
    pairs = []
    units = values.get('units') or {}
    for key, field in CONSTRAINT_UNIT_KEYS.items():
        if units.get(field):
            pairs.append((key, str(units[field])))
    for section_key, _object_keys in CONSTRAINT_FAMILIES:
        records = [item if isinstance(item, dict) else item.model_dump()
                   for item in (values.get(section_key) or ())]
        if not records:
            continue
        keys = [_constraint_write_object_key(section_key, record)
                for record in records]
        unknown = sorted({
            name
            for record, key in zip(records, keys)
            for name in record
            if not _constraint_field_writable(section_key, name, record[name],
                                              key)})
        if unknown:
            raise ValueError(
                "the &constraint text cannot write the field%s %s of a %s "
                'record; the structured form is the only spelling of it'
                % ('s' if len(unknown) > 1 else '',
                   ', '.join("'%s'" % name for name in unknown), section_key))
        if section_key == 'walls':
            order = []
            for record in records:
                name = str(record.get('geometry') or 'sphere')
                if name not in order:
                    order.append(name)
            pairs.append(('wall_type', ' '.join(order)))
        objects = {}
        for index, key in enumerate(keys):
            objects.setdefault(key, []).append(
                _constraint_write_object_value(key, records[index]))
        for key, segments in objects.items():
            pairs.append((key, '; '.join(segments)))
        pairs.extend(_constraint_write_params(section_key, records, keys))
        pairs.extend(_constraint_write_ramps(records, keys, section_key))
    return pairs


