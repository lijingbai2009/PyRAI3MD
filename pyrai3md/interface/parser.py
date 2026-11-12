"""
======================================================================
This module provides the traditional input file parser for PyRAI3MD.
It reads PyRAI3MD input files (text or JSON format), splits them by
&SECTION markers, and extracts key-value pairs as raw strings.

No type conversion, no validation, no default filling is performed.
All values remain as raw strings for downstream Schema and Skill layers.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import json
import os

import yaml


def parse_input(input_file):
    """Parse a PyRAI3MD input file into a complete raw configuration.

    This is the single entry point for the Interface layer. It performs:
    1. Parse input file to raw section dict
    2. Load coordinate and velocity files
    3. Validate that coordinates are resolved

    Parameters
    ----------
    input_file : str
        Path to the input file.

    Returns
    -------
    dict
        Complete raw configuration with atoms, coordinates, and
        optionally velocities resolved. All values remain as raw
        strings or string lists (except coordinate arrays).

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file format is not recognized or coordinates missing.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError("Input file not found: %s" % input_file)

    # Detect format by extension or content
    _, ext = os.path.splitext(input_file)
    if ext.lower() == '.json':
        raw_config = _parse_json(input_file)
    elif ext.lower() in ('.yaml', '.yml'):
        raw_config = _parse_yaml(input_file)
    else:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        raw_config = _parse_text(lines)

    # Load coordinate and velocity files
    raw_config = _load_coordinate_file(raw_config, input_file)

    # Resolve a skill-declared initial-guess file next to the input
    raw_config = _detect_initial_guess(raw_config, input_file)

    # Validate that coordinates are resolved
    _validate_structure(raw_config)

    return raw_config


def _parse_text(lines):
    """Parse text-format input into raw section dict.

    Rules (following classic format):
    - Lines starting with # are comments, stripped
    - Lines starting with & start a new section
    - Each subsequent line: first token = key, rest = value(s)
    - A key repeated in one section keeps both values (as a list), so the
      section's own reader can use or refuse them instead of losing one
    - Lines with < 2 tokens are skipped
    - All values stored as raw strings (single value) or
      raw string lists (multiple values on one line)

    Parameters
    ----------
    lines : list of str
        Lines from the input file.

    Returns
    -------
    dict
        Nested dict of sections and their key-value pairs.
    """
    sections = {}
    current_section = None
    index = 0
    total = len(lines)

    while index < total:
        # Strip whitespace
        line = lines[index].strip()

        # Strip inline comments (everything after #)
        # Matches classic behavior: line.split('#')[0]
        line = line.split('#')[0].strip()

        # Skip empty lines
        if not line:
            index += 1
            continue

        # New section marker
        if line.startswith('&'):
            section_name = line[1:].strip().lower()
            if section_name == 'end':
                # The classic format closes a section with `&end`. Filing it as
                # a section named 'end' was the framework's behaviour and it
                # swallowed every key that followed, which is the trap the
                # design document records in section 3.2; the loop simply stops
                # owning keys here. Every section, the constraint block
                # included, is flat `keyword value` lines, so nothing here needs
                # to know what a section means - the schema adapter reads it.
                current_section = None
                index += 1
                continue
            current_section = section_name
            if current_section not in sections:
                sections[current_section] = {}
            index += 1
            continue

        # Key-value pair within a section
        if current_section is not None:
            tokens = line.split()
            if len(tokens) < 2:
                # Skip lines without a value. The increment is written out
                # because this loop advances an index rather than iterating a
                # generator: a bare `continue` here would spin forever.
                index += 1
                continue

            key = tokens[0].lower()
            # Single value -> string, multiple values -> space-joined string
            # Keep as raw string for downstream type conversion
            if len(tokens) == 2:
                value = tokens[1]
            else:
                value = ' '.join(tokens[1:])

            # A keyword written twice in one section carries both values, and
            # they are kept in the order they appeared. Keeping only the last
            # was a silent discard: `grid depth 1 2` followed by
            # `grid nn_size 8 16` planned a scan over one axis, and nothing
            # downstream could tell that the other line had ever existed. A
            # section reader that cannot use several values for one keyword
            # now says so, and this loop still knows nothing about what any key
            # means.
            filed = sections[current_section]
            if key not in filed:
                filed[key] = value
            elif isinstance(filed[key], list):
                filed[key].append(value)
            else:
                filed[key] = [filed[key], value]

        index += 1

    return sections


def _parse_json(filepath):
    """Parse JSON-format input file.

    Parameters
    ----------
    filepath : str
        Path to the JSON input file.

    Returns
    -------
    dict
        Parsed JSON as a dictionary with lowercase section keys.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Normalize section names to lowercase
    normalized = {}
    for key, value in data.items():
        normalized[key.lower()] = value

    return normalized


def _parse_yaml(filepath):
    """Parse YAML-format input file.

    YAML inputs use the nested section layout of the official
    workflow examples (control / molecule / md / environment /
    <skill short name>). Section names are normalized to lowercase,
    matching the JSON handling. Native YAML types are preserved
    (no raw-string conversion).

    Parameters
    ----------
    filepath : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML as a dictionary with lowercase top-level keys.

    Raises
    ------
    ValueError
        If the YAML file does not contain a mapping.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "YAML input must be a mapping of sections: %s" % filepath
        )

    # Normalize section names to lowercase
    normalized = {}
    for key, value in data.items():
        normalized[str(key).lower()] = value

    return normalized


def _section_path(value):
    """Reduce a `&molecule coord`/`velo` keyword to the file path it names.

    These keywords name one file each. A section that carried the same
    keyword on several lines now keeps every value rather than only the last
    one (see `_parse_text`), so a list arrives here: the entries of the
    classic row spelling are not paths, none of them was ever read as
    coordinates by this framework, and the same-basename `.xyz` fallback
    below is what has always supplied that geometry. The first value that is
    a path therefore wins and the rest are passed over.
    """
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and os.path.isfile(item):
                return item
        return ''
    return value if isinstance(value, str) else ''


def _load_coordinate_file(raw_config, input_file):
    """Resolve coordinate and velocity files for the calculation.

    Resolution order:
    1. If atoms/coordinates already in raw_config (config.json, customized
       pmd_env.yaml, or inline) -- use them directly.
    2. If &molecule section has 'coord' keyword pointing to an existing
       file -- read from that file.
    3. Fallback: look for same-basename .xyz file in the input file's
       directory (e.g., input.inp -> input.xyz).

    Velocity resolution:
    - If 'velo' keyword in &molecule points to an existing file -- read it.
    - Fallback: look for same-basename .velo or .velocity.xyz.
    - If no velocity file found -- fill with zeros.

    Parameters
    ----------
    raw_config : dict
        Parsed input configuration.
    input_file : str
        Path to the input file.

    Returns
    -------
    dict
        Updated raw_config with atoms, coordinates, and optionally velocities.
    """
    from ..utilities.io import read_coord

    # Skip if atoms already specified (config.json, customized pmd_env.yaml)
    if raw_config.get('atoms'):
        return raw_config

    # Extract coord/velo paths from &molecule section
    molecule = raw_config.get('molecule', {})
    if not isinstance(molecule, dict):
        molecule = {}
    coord_path = _section_path(molecule.get('coord', ''))
    velo_path = _section_path(molecule.get('velo', ''))

    # Get basename for fallback: input.inp -> input
    base = os.path.splitext(input_file)[0]

    # --- Resolve coordinate file ---
    coord_base = None  # basename for read_coord (without extension)
    if coord_path and os.path.isfile(coord_path):
        # Explicit coord keyword pointing to existing file
        coord_base = os.path.splitext(coord_path)[0]
    elif os.path.isfile(base + '.xyz'):
        # Fallback: same basename as input
        coord_base = base

    if coord_base is None:
        # No coordinate file found -- return as-is, downstream will handle
        return raw_config

    # Read coordinates (and velocities if available)
    try:
        atoms, coord, velo = read_coord(coord_base)
    except Exception as error:
        # A file that exists but cannot be read is an input error, not the
        # "no coordinates supplied" case handled above; name the file.
        raise ValueError(
            "could not read coordinates from %r: %s" % (coord_base, error)
        ) from error

    raw_config['atoms'] = atoms.flatten().tolist()
    raw_config['coordinates'] = coord.tolist()

    # --- Resolve velocity file ---
    velo_found = False
    if velo_path and os.path.isfile(velo_path):
        # Explicit velo keyword pointing to existing file
        from ..utilities.io import read_float_from_text
        import numpy as np
        velo = np.array(read_float_from_text(velo_path))
        velo_found = True
    else:
        # read_coord already checked .velo / .velocity.xyz
        natom = len(atoms.flatten())
        if os.path.exists(f'{coord_base}.velo') or os.path.exists(f'{coord_base}.velocity.xyz'):
            velo_found = True

    # Always set velocities (read_coord fills zeros if no .velo/.velocity.xyz)
    if velo is not None:
        raw_config['velocities'] = velo.tolist()
        if not velo_found:
            print(f'  [Info] No velocity file found for {input_file}. '
                  f'Initial velocities set to zero ({natom} atoms).')

    return raw_config


def _detect_initial_guess(raw_config, input_file):
    """Resolve a skill-declared initial-guess file next to the input.

    A skill that starts from a converged file states the channel in its
    manifest:

        initial_guess:
          suffix: .StrOrb
          field: guess_orbital

    The candidates are `{title}<suffix>` and `<input stem><suffix>` in
    the input's directory, title first (the same `title.*` convention as
    title.coord / title.velo). The resolved path is stored under
    'initial_guess' with the skill and field it belongs to; the task
    queue fills that field when the input left it empty. A skill that
    declares nothing is never consulted, so the framework names none of
    them.

    Parameters
    ----------
    raw_config : dict
        Parsed raw configuration.
    input_file : str
        Path to the PyRAI3MD input file.

    Returns
    -------
    dict
        Updated raw_config, with 'initial_guess' set when resolved.
    """
    control = raw_config.get('control', {})
    if not isinstance(control, dict):
        control = {}
    theory = control.get('theory') or []
    if not isinstance(theory, list):
        theory = [theory]
    title = str(control.get('title', '') or '')
    stem = os.path.splitext(input_file)[0]

    from ..skills.query import discover_all, get_declaration

    for label in theory:
        declared = get_declaration(label, 'initial_guess')
        suffix = str(declared.get('suffix') or '')
        field = str(declared.get('field') or '')
        if not suffix or not field:
            continue
        candidates = []
        if title:
            candidates.append(
                os.path.join(os.path.dirname(input_file), title + suffix))
        candidates.append(stem + suffix)
        for path in candidates:
            if os.path.isfile(path):
                meta = discover_all().get(str(label)) or {}
                raw_config['initial_guess'] = {
                    'skill': meta.get('_dir_name') or str(label),
                    'field': field,
                    'path': os.path.abspath(path),
                }
                return raw_config
    return raw_config


def _validate_structure(raw_config):
    """Validate that coordinates are resolved before downstream processing.

    Parameters
    ----------
    raw_config : dict
        Raw configuration with potential coordinate data.

    Raises
    ------
    ValueError
        If atoms or coordinates are missing.
    """
    atoms = raw_config.get('atoms', [])
    coordinates = raw_config.get('coordinates', [])

    if not atoms:
        msg = (
            "\n  ERROR: No atomic coordinates found.\n"
            "  PyRAI3MD requires atomic coordinates to proceed.\n"
            "  Provide coordinates via one of:\n"
            "    - &molecule coord <file.xyz> in input file\n"
            "    - Same-basename .xyz file (e.g., input.xyz for input.inp)\n"
            "    - atoms/coordinates in config.json or customized pmd_env.yaml\n"
            "  Aborting."
        )
        raise ValueError(msg)

    if not coordinates:
        msg = (
            "\n  ERROR: Atoms are specified but coordinates are empty.\n"
            "  Check coordinate file format.\n"
            "  Aborting."
        )
        raise ValueError(msg)
