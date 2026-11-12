"""
======================================================================
This is a standalone neural network potential energy surface skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Sep 30 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module owns the hyperparameter vocabulary of the nn_pes skill.

A role section ('eg', 'nac') carries the hyperparameters of one network
family, and each of its keywords may be written per ensemble member. The
member axis is always the outer one: one ';' segment in the classic text,
one element of the outer list in YAML and JSON. Inside a segment the value
keeps its own shape, which is why a keyword whose value is itself a list
has to be nested to differ between members.

The keyword names are those of the original project, so an old input file
translates without renaming. Each entry says where the value goes inside
the container's hyperparameter dictionary and what shape it must have:

    ('model'|'training'|'retraining'|'predict', dotted path), kind

Kinds:
    'scalar'  a single value; a list of them is read per member
    'list'    a fixed-length value; flat means one group to broadcast,
              nested means one group per member
    'pairs'   a list of integer groups (feature index tables)
    'bool'    same rule as 'scalar', with boolean checking
    'reg'     regularization selector 'none'/'l1'/'l2'/'l1_l2' expanded
              with reg_l1 / reg_l2 into the container's dict form

"""

import copy

# kind, target: (hyper section, dotted key inside it)
HYPER_KEYWORDS = {
    # ---- network shape (model section; fixed once built) --------------
    'nn_size':        ('scalar', ('model', 'nn_size')),
    'depth':          ('scalar', ('model', 'depth')),
    'use_dropout':    ('bool',   ('model', 'use_dropout')),
    'dropout':        ('scalar', ('model', 'dropout')),
    'invd_index':     ('pairs',  ('model', 'invd_index')),
    'angle_index':    ('pairs',  ('model', 'angle_index')),
    'dihed_index':    ('pairs',  ('model', 'dihed_index')),
    'use_reg_activ':  ('reg',    ('model', 'use_reg_activ')),
    'use_reg_weight': ('reg',    ('model', 'use_reg_weight')),
    'use_reg_bias':   ('reg',    ('model', 'use_reg_bias')),
    'reg_l1':         ('scalar', ('_reg', 'l1')),
    'reg_l2':         ('scalar', ('_reg', 'l2')),
    # ---- training schedule -------------------------------------------
    'epo':               ('scalar', ('training', 'epo')),
    'batch_size':        ('scalar', ('training', 'batch_size')),
    'epostep':           ('scalar', ('training', 'epostep')),
    'val_split':         ('scalar', ('training', 'val_split')),
    'val_disjoint':      ('bool',   ('training', 'val_disjoint')),
    'learning_rate':     ('scalar', ('training', 'learning_rate')),
    'normalization_mode': ('scalar', ('training', 'normalization_mode')),
    'initialize_weights': ('bool',  ('training', 'initialize_weights')),
    'loss_weights':      ('list',   ('training', 'loss_weights')),
    'energy_only':       ('bool',   ('training', 'energy_only')),
    'pre_epo':           ('scalar', ('training', 'pre_epo')),
    'phase_less_loss':   ('bool',   ('training', 'phase_less_loss')),
    # Feature and target scaling. The container spells the target half
    # 'energy', the caller half 'y', so both spellings resolve to it.
    'scale_x_mean':      ('bool',   ('training', 'auto_scaling.x_mean')),
    'scale_x_std':       ('bool',   ('training', 'auto_scaling.x_std')),
    'scale_y_mean':      ('bool',   ('training', 'auto_scaling.energy_mean')),
    'scale_y_std':       ('bool',   ('training', 'auto_scaling.energy_std')),
    # ---- callbacks ---------------------------------------------------
    'use_step_callback':    ('bool',   ('training', 'step_callback.use')),
    'epoch_step_reduction': ('list',   ('training', 'step_callback.epoch_step_reduction')),
    'learning_rate_step':   ('list',   ('training', 'step_callback.learning_rate_step')),
    'use_linear_callback':  ('bool',   ('training', 'linear_callback.use')),
    'learning_rate_start':  ('scalar', ('training', 'linear_callback.learning_rate_start')),
    'learning_rate_stop':   ('scalar', ('training', 'linear_callback.learning_rate_stop')),
    'epomin':               ('scalar', ('training', 'linear_callback.epomin')),
    'use_early_callback':   ('bool',   ('training', 'early_callback.use')),
    'patience':             ('scalar', ('training', 'early_callback.patience')),
    'max_time':             ('scalar', ('training', 'early_callback.max_time')),
    'delta_loss':           ('scalar', ('training', 'early_callback.delta_loss')),
    'loss_monitor':         ('text',   ('training', 'early_callback.loss_monitor')),
    'factor_lr':            ('scalar', ('training', 'early_callback.factor_lr')),
    'use_exp_callback':     ('bool',   ('training', 'exp_callback.use')),
    # ---- serving -----------------------------------------------------
    # The batch the compiled forward pass runs. It is a property of the
    # trained network (it is written into the folder's hyper file), which is
    # why it sits here rather than in the shared block.
    'batch_size_predict': ('scalar', ('predict', 'batch_size_predict')),
    # ---- activation: name plus its alpha, assembled into the dict form
    'activ':               ('text',   ('_activation', 'name')),
    'activ_alpha':         ('scalar', ('_activation', 'alpha')),
}

# The activation is spelled as a name plus its alpha, and the container
# wants the serialized dict form; both are assembled here.
ACTIVATION_DEFAULT = 'leaky_softplus'
ACTIVATION_ALPHA_DEFAULT = 0.03

# 'reg_l1' and 'reg_l2' are shared by the three regularizer switches, so
# they are resolved before any per-member value is written.
_REG_SECTION = '_reg'

# Values accepted for the regularization switches
REG_STYLES = ('none', 'l1', 'l2', 'l1_l2')


def member_groups(raw, kind):
    """Split a keyword's raw value into its per-member groups.

    The member axis is always the outer one. In the classic text it is the
    ';' separator; in YAML and JSON it is the outer list. A keyword whose
    own value is a list ('list', 'pairs') is therefore nested once per
    level of structure, and a flat value means one group to broadcast.

    Parameters
    ----------
    raw : object
        The value as it arrived from any of the three carriers.
    kind : str
        Declared kind of the keyword.

    Returns
    -------
    list
        One entry per member group, still unparsed.
    """
    if isinstance(raw, str):
        return [chunk.strip() for chunk in raw.split(';')]
    if not isinstance(raw, (list, tuple)):
        return [raw]
    if kind in ('scalar', 'bool'):
        return list(raw)
    if not raw:
        return [[]]
    first = raw[0]
    nested = isinstance(first, (list, tuple))
    if kind == 'list':
        return [list(group) for group in raw] if nested else [list(raw)]
    # 'pairs': one member's table is itself a list of groups, so the
    # member axis only starts one level deeper
    if nested and any(isinstance(item, (list, tuple)) for item in first):
        return [group for group in raw]
    return [list(raw)]


def parse_value(raw, kind):
    """Convert one member group into the container's value."""
    if kind == 'list':
        items = raw if isinstance(raw, (list, tuple)) \
            else str(raw).replace(',', ' ').split()
        return [_number(item, kind) for item in items]
    if kind == 'pairs':
        return _as_groups(raw)
    if kind == 'text':
        return str(raw).strip()
    if kind == 'bool':
        if isinstance(raw, bool):
            return raw
        token = str(raw).strip().lower()
        if token in ('true', '1', 'yes', 'y', 't'):
            return True
        if token in ('false', '0', 'no', 'n', 'f'):
            return False
        raise ValueError("expected a boolean, got %r" % raw)
    if kind == 'reg':
        token = str(raw).strip().lower()
        if token in ('', 'none', 'off', 'false', 'no'):
            return None
        if token not in REG_STYLES:
            raise ValueError(
                "regularizer must be one of %s, got %r" % (REG_STYLES, raw))
        return token
    return _number(raw, kind)


def _as_groups(raw):
    """Read one member's index table into a list of integer rows.

    Inside a member group a comma ends one row and a space separates the
    indices in it, so the classic text '1 0, 2 0, 2 1' and the typed
    spelling [[1, 0], [2, 0], [2, 1]] mean the same table. A row keeps its
    own length: an inverse-distance table is pairs, an angle table is
    triples, and neither is judged by the other's shape.
    """
    if isinstance(raw, str):
        return [[int(token) for token in chunk.split()]
                for chunk in raw.split(',') if chunk.strip()]
    if any(isinstance(item, (list, tuple)) for item in raw):
        return [[int(value) for value in row] for row in raw]
    return [[int(value) for value in raw]]


def _number(raw, kind):
    """Read one number, keeping integers integral."""
    if isinstance(raw, bool):
        raise ValueError("expected a number, got a boolean")
    if isinstance(raw, (int, float)):
        return raw
    token = str(raw).strip()
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        raise ValueError(
            "%s expects a number or a list of numbers, got %r" % (kind, raw))


def check_value(keyword, value):
    """Reject a parsed value the container can do nothing with.

    A training run dies in a child process, where a bad number arrives
    as a stack trace from Keras. The handful of bounds below are the ones
    that decision cannot be recovered from, so they are reported while
    the config is still being read.

    Parameters
    ----------
    keyword : str
        Hyperparameter name.
    value : object
        Its parsed value for one member.

    Raises
    ------
    ValueError
        If the value is outside the range the keyword admits.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    if keyword in ('depth', 'nn_size', 'epo', 'batch_size',
                   'batch_size_predict') and value < 1:
        raise ValueError(
            "%s must be at least 1, got %s" % (keyword, value))
    if keyword == 'val_split' and not 0.0 < value < 1.0:
        raise ValueError(
            "val_split is a fraction of the data held out, so it must lie "
            "between 0 and 1, got %s" % value)


def broadcast(groups, members, keyword):
    """Fit member groups to the ensemble size.

    One group serves every member. More groups than members keep the
    leading ones, because mult_nn decides how many members are read.

    Parameters
    ----------
    groups : list
        Parsed groups in member order.
    members : int
        Ensemble size.
    keyword : str
        Keyword name, for the message.

    Returns
    -------
    list
        Exactly `members` values.

    Raises
    ------
    ValueError
        If nothing was given, or several groups cover fewer than all
        members.
    """
    if members < 1:
        raise ValueError("mult_nn must be at least 1, got %d" % members)
    if not groups:
        raise ValueError("keyword %r carries no value" % keyword)
    if len(groups) == 1:
        return groups * members
    if len(groups) < members:
        raise ValueError(
            "keyword %r gives %d member groups but the ensemble has %d "
            "members; write one group to broadcast it, or one per member"
            % (keyword, len(groups), members))
    return list(groups[:members])


def member_list(raw, members, keyword):
    """Read a per-member value list written with the same rule as a hyper.

    Used by the shared section, where the member axis matters for the
    device each training subprocess gets. A single value serves every
    member; the ';' groups are read in member order.

    Parameters
    ----------
    raw : object
        Classic text with ';' segments, a list, or a single value.
    members : int
        Ensemble size.
    keyword : str
        Keyword name, for the message.

    Returns
    -------
    list
        `members` values.
    """
    groups = member_groups(raw, 'scalar')
    return [parse_value(group, 'scalar') for group in broadcast(
        groups, members, keyword)]


def build_member_hyper(role, defaults, members, values):
    """Expand one role section into one hyper dict per ensemble member.

    Parameters
    ----------
    role : str
        Role name for messages ('eg' or 'nac').
    defaults : dict
        The container's default hyper dict of the model type.
    members : int
        Ensemble size.
    values : dict
        Raw role section: keyword to classic text or structured value.

    Returns
    -------
    list of dict
        `members` hyperparameter dictionaries.

    Raises
    ------
    ValueError
        For an unknown keyword or a value its kind rejects.
    """
    unknown = sorted(set(values) - set(HYPER_KEYWORDS))
    if unknown:
        raise ValueError(
            "unknown hyperparameter in &%s: %s; available: %s"
            % (role, ', '.join(unknown), ', '.join(sorted(HYPER_KEYWORDS))))

    columns = {}
    for keyword, raw in values.items():
        kind = HYPER_KEYWORDS[keyword][0]
        groups = [parse_value(group, kind)
                  for group in member_groups(raw, kind)]
        for value in groups:
            check_value(keyword, value)
        columns[keyword] = broadcast(groups, members, keyword)

    built = []
    for index in range(members):
        member = copy.deepcopy(defaults)
        regular = {
            'l1': columns['reg_l1'][index] if 'reg_l1' in columns else 1e-5,
            'l2': columns['reg_l2'][index] if 'reg_l2' in columns else 1e-5,
        }
        name = columns['activ'][index] if 'activ' in columns else ACTIVATION_DEFAULT
        alpha = (columns['activ_alpha'][index] if 'activ_alpha' in columns
                 else ACTIVATION_ALPHA_DEFAULT)
        member['model']['activ'] = {'class_name': str(name),
                                    'config': {'alpha': float(alpha)}}
        mirror = []
        for keyword, values_per_member in columns.items():
            kind, (section, dotted) = HYPER_KEYWORDS[keyword]
            if section in (_REG_SECTION, '_activation'):
                continue
            value = values_per_member[index]
            if kind == 'reg':
                member['model'][dotted] = _regularizer(value, regular)
                continue
            _write(member, section, dotted, value)
            mirror.append(dotted.split('.')[0])
        _mirror_retraining(member, sorted(set(mirror)))
        built.append(member)
    return built


def _regularizer(style, reg_values):
    """Expand a regularization selector into the container's dict form."""
    if not style:
        return None
    if style == 'l1':
        return {'class_name': 'L1', 'config': {'l1': reg_values['l1']}}
    if style == 'l2':
        return {'class_name': 'L2', 'config': {'l2': reg_values['l2']}}
    return {'class_name': 'L1_L2',
            'config': {'l1': reg_values['l1'], 'l2': reg_values['l2']}}


def _write(hyper, section, dotted, value):
    """Write a parsed value into its hyper slot, creating dicts as needed."""
    if section in (_REG_SECTION, '_activation'):
        return
    target = hyper.setdefault(section, {})
    keys = dotted.split('.')
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _mirror_retraining(member, written):
    """Carry what the user wrote into the retraining section as well.

    The container reads 'training' or 'retraining' depending on the fit
    mode, so spelling 'epo' once must not mean it applies to only one of
    the two. A keyword whose top-level slot the role's retraining section
    does not carry (g2 has no loss_weights there) is skipped rather than
    invented.
    """
    training = member.get('training', {})
    retraining = member.setdefault('retraining', {})
    for key in written:
        if key in retraining:
            retraining[key] = copy.deepcopy(training[key])
