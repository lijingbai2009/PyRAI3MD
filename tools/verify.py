#!/usr/bin/env python3
"""
======================================================================
This module is the post-round verifier for PyRAI3MD qualification runs.

It reads a finished qualification work directory (`tmp_<stamp>`) and
re-derives, from the files the run left on disk, the numbers the round
reported. It never starts an engine or a calculation; section G reads a
molcas log through the skill's own parser, which is file work. Sections A
to K:

A  Records vs the round document. Every example verdict that stores a
   metric (`etot_drift`, `cell_frames`, ...) is recomputed from the
   frame records the runner claims to have judged, and compared at full
   precision. A record whose mtime predates the round is reported too:
   that would mean a verdict read a leftover from an earlier round.

B  Step log vs frame record. Each MD step writes two documents about the
   same state, the run's `<title>.log` step block and the
   `<title>.md.energies` row. Nothing in the suite compares them, so a
   divergence between the two writers passes silently. This walks every
   step of every paired run and reports potential and kinetic energy
   disagreements separately.

C  Engine output vs the recorded value. The framework parses an engine
   file into `results['energy']` and records it; an audit that only
   re-reads the record never tests that parse. Where a calculation
   directory survived the run, the engine's own number is read back and
   compared against the last recorded row.

C2 Engine output vs the optimizer's own record. Section C reads MD frame
   records, of which an optimize round has none, so a round that relaxes a
   geometry or a box leaves its `opt_status.txt` energy column compared against
   nothing but the documents written downstream of the same parse. C2 asks the
   last row for the engine's own number, after the same unit conversions. It also
   counts the legs whose own log closed on an optimization budget stop
   (`budget_stopped`): the verdict is `pass` either way, so that count is what
   separates a converged pass from a capped one - 32 legs of round
   `tmp_20261019_200652` are the second kind (#399).

C3 Optimizer term columns vs the energy column. A composite row spells one
   signed contribution per assembled term, so its columns have to add up to the
   energy beside them. A leg that is not a crossing must close, and a deviation
   of either sign is named; a crossing subtracts a gap term that reaches the
   energy column and no term column, so its offset is reported as a reading
   instead. The leg's own `opt_method` decides which question is asked.

D  Variable-cell verdict. One run's box is reported in three places: the
   `opt_status.txt` rows the optimizer wrote, the `cell_metrics` the judge
   stored in `results.json`, and the `[CELL]` sentence the round log
   printed. D re-derives all three from the file and names any pair that
   disagrees, reports a record that is an engine's table rather than the
   framework's, and counts the rows that do not match their header instead
   of dropping them. The one metric the status table has no column for -
   `shape_ratio_change`, what `[COUPLING]` decides on - gets a second
   source instead of a skip: the trajectory's own `Lattice=` rows, through
   the same ratio function the judge used.

E  Carrier parity. Every example runs twice, from its YAML and from the
   generated classic input, and the two stop in sibling directories. E compares
   the numbers each carrier records - the round document's `cell_metrics`, and
   each record's own term decomposition - because a mode that renders or reads
   one of them differently is exactly how the last two defects in this area hid.

G  Molcas legs. A skill round runs each molcas case on three carriers, and
   since the per-leg scratch fix each keeps its own calculation directory, so
   three engine logs now exist for one case and sections A to F compare none of
   them. G parses all three with the skill's own parser and requires the energy,
   gradient, NAC and SOC arrays to agree; it then asserts the shape and
   provenance contract of each property, which is what names a zero-filled
   array of the right length.

I  QM/MM gradient audit. One `qmmm_energy` step leaves four files beside it:
   the engine's own `force` (the UNREDUCED U_main, F = -dU/dR), the two QM-row
   auxiliary force files, and `grad` - the reduced classical surface in dU/dR
   sign. The reduction itself is exact by construction, so no residual inside
   the writer can report a wrong index or a dropped term; only reading the four
   files back can. I checks `grad = -force + u_gas.force + qmmm_el.force` on
   the QM rows and `grad = -force` elsewhere. The tolerance follows the file
   that sets it: a cold `write_dump` carries six significant digits and cannot
   be widened, the resident writer and the skill's own writers carry sixteen.

K  Skill carrier parity. Every skill case runs through the skill entry, the
   framework agent and the classic input, each with its own calculation
   directory. C and G judge a leg against its own record; K compares the legs
   with each other: the engine deck each carrier wrote (per-carrier paths
   masked) and the final energy each engine printed, where the engine writes
   one. Files only - it never runs a skill. Round tmp_20261110_093321 is why
   this section exists: OpenQP `tddft_gradient` asked the skill carrier for
   `grad=2,3` and both framework carriers for `grad=1`, and every leg passed
   its own status.

Coverage is reported, never assumed: skills whose reader is not
implemented, and channels a given round did not exercise, are counted
and printed as not covered.

What is NOT independent of the judged code (#5.12): section D's cell
verdict and the LAMMPS/ORCA energy readers call the same functions the
runner and the skill use (`runner.check_cell_optimization`,
`row_length_ratios`, `optimization_budget_stop`, and the skills' own
parsers). Those sections can catch a value that was recorded wrong or
read from the wrong file, but they cannot catch an error in the criterion
itself -- a shared threshold that is wrong is wrong on both sides.

This tool is located in tools/ and is run directly:
    python tools/verify.py <work_dir>
    python tools/verify.py --selftest

The audits read the round and never write into it, and never run a skill or
engine: every section is post-processing of files the round already left on
disk (section G uses the molcas skill's parser to read a log; no skill entry
point is invoked). It does not participate in test planning or execution logic.

Oct 06 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import glob
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time

# Import the pyrai3md that sits beside this tool, so the audit reads the
# records with the same code that wrote them.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrai3md.qualification.runner import (cell_frame_metrics,
                                           closing_status,
                                           energy_conservation_metrics,
                                           is_center_record,
                                           read_md_energies,
                                           read_md_ensemble,
                                           read_md_lattice)
from pyrai3md.qualification.runner import (
    check_cell_optimization,
    optimization_budget_stop,
    row_length_ratios,
    read_optimize_tail,
)
from pyrai3md.utilities.opt_formatter import (STATUS_ATOM_COLUMNS,
                                              STATUS_CELL_COLUMNS)
from pyrai3md.utilities.print_results import (SECTION_CONTRIBUTIONS,
                                              SECTION_ENERGIES,
                                              SECTION_KINETIC, WRITER_FOREIGN,
                                              read_status)
from pyrai3md.utilities.lattice import min_image_displacement
from pyrai3md.utilities.units import (ANGSTROM_TO_BOHR, EV_TO_HARTREE,
                                      KCAL_TO_HARTREE)

# A frame record is `<title>.md.<kind>` or `<title>.md.<layer>.<kind>` for a
# multicentre run. `<title>.sh.<kind>` hop records are sparse by construction
# and are not frames of this trajectory, so they stay out of every comparison.
FRAME = re.compile(r'\.md(\.\d+)?\.(energies|xyz)$')
STATE_LINE = re.compile(r'^\s*State\s+(\d+)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*$')
KINETIC_LINE = re.compile(r'^\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*$')
STEP_MARKER = re.compile(r'^\*\s+Step\s+(\d+)\b')
CURRENT_STATE = re.compile(r'^\s*Current state:\s*(\d+)\s*$')
SECTION_LINE = re.compile(r'^\s*&(.+?)\s*$')
SEPARATOR = '-' * 79

# Absolute tolerance for the two writers of one step. The log prints 16
# decimals, so this only absorbs the difference between rounding and a
# genuinely different value.
STEP_TOL = 5e-11
METRIC_TOL = 1e-12

# The optimizer's own table writes fifteen decimals, so a term decomposition
# that closes does so to the printed precision, while a deviation above this is
# a real offset: a crossing subtracts its own gap term, which no column carries.
# A POSITIVE deviation is a different thing again and is judged below.
TERM_TOL = 1e-12

# One example's two carriers run the same computation through different
# plumbing, so their recorded metrics may differ in the last digits. Measured on
# round tmp_20261019_200652 over 24 pairs: every metric was equal except
# `optimize_cell_gap_cp2k`'s `cell_drms`/`cell_dmax`, which differ by 3.4e-10
# relative - so this is stated as a reading, not a convenience, and it still
# names a metric that moved.
PARITY_TOL = 1e-8

def _float_lines(path):
    """Every bare float of a file that holds one number per line."""
    values = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
        for line in handler:
            try:
                values.append(float(line))
            except ValueError:
                continue
    return values



def _engine_openqp(calcdir):
    """OpenQP per-state energies from its `energies` export.

    Line 0 is the internal triplet reference for an MRSF run, which the
    skill's own parser offsets past; that line simply never matches a
    recorded value and is counted as engine_extra.
    
    Read locally on purpose (#3.7): the skill's parser
    (`openqp_skill/service/parser/output_parser.py:parse_export_files`)
    needs the run's config (state ladder and profile) to know which
    rows belong to which state, and a fetched product tree does not
    carry one; a config-free audit reads the exported numbers as they
    stand and compares them to the record.
"""
    values = []
    # A two-manifold SOC run publishes one energy per ladder under
    # `energies_s` / `energies_t` and writes no plain `energies`; a campaign
    # that reads only the latter calls every such directory unread, which is
    # how `md_openqp_fssh_soc`'s `tmp_OPENQP` sat outside section C.
    for name in ('energies', 'energies_s', 'energies_t'):
        path = os.path.join(calcdir, name)
        if os.path.isfile(path):
            values.extend(_float_lines(path))
    return values or None


def _engine_xtb(calcdir):
    """xTB total energy from the plain `energy` file, else the engrad block.

    The engrad export comments its own header, so the number sits two lines
    under the line that names it: `#`, `# The current total energy in Eh`,
    `#`, then the value. Reading only the immediately following line returned
    nothing, and an engine that reports nothing is reported as unread rather
    than compared - which is how round tmp_20261017_094721's every xtb leg
    came to sit in `dirs_unread` while its own number was on disk.
    
    Read locally on purpose (#3.7): the skill's parser
    (`xtb_skill/service/parser/output_parser.py:parse_output`) needs
    the run's config and workflow, which a fetched product tree does
    not carry. The engrad fallback here follows the layout the
    deployment writes (a comment bar between the header and the
    number).
"""
    path = os.path.join(calcdir, 'energy')
    if os.path.isfile(path):
        values = _float_lines(path)
        if values:
            return values[:1]
    for path in sorted(glob.glob(os.path.join(calcdir, '*.engrad'))):
        with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
            lines = handler.read().splitlines()
        for index, line in enumerate(lines):
            if 'total energy' not in line.lower():
                continue
            for follow in lines[index + 1:index + 5]:
                candidate = follow.strip()
                if not candidate or candidate.startswith('#'):
                    continue
                try:
                    return [float(candidate.split()[0])]
                except (ValueError, IndexError):
                    continue
    return None


def _engine_bagel(calcdir):
    """BAGEL state energies from the `ENERGY.out` export.
    Read locally on purpose (#3.7): the skill's parser
    (`bagel_skill/service/parser/log_parser.py:_parse_energy`) needs
    the run's method and state count to pick the energy block; a
    config-free audit takes the export as it stands.
"""
    path = os.path.join(calcdir, 'ENERGY.out')
    if not os.path.isfile(path):
        return None
    return _float_lines(path) or None


def _engine_lammps(calcdir):
    """LAMMPS single-point energies, through the skill's own reader.

    `lammps_skill/service/parser/output_parser.py:read_energy_file` is
    the reader the backend itself uses for `{title}.energy` (it takes the
    last number and quotes the log tail when there is none), so the audit
    and the run cannot disagree about which value a step was scored on.
    """
    from pyrai3md.skills.lammps_skill.service.parser.output_parser import (
        read_energy_file)

    values = []
    for path in sorted(glob.glob(os.path.join(calcdir, '*.energy'))):
        try:
            values.append(read_energy_file(path))
        except RuntimeError:
            continue
    return values or None


def _engine_orca(calcdir):
    """ORCA total energy, through the skill's own .engrad parser.

    `orca_skill/service/parser/engrad_parser.py:parse_engrad` reads the
    line ORCA actually writes ("The current total energy in Eh") and
    zero-fills when it finds none, so the marker is checked here first:
    an unmarked file is unread, never a zero. The previous reader looked
    for a `$ENERGY` block no ORCA run writes, and on such a block took
    the print-level integer as the energy.
    """
    from pyrai3md.skills.orca_skill.service.parser.engrad_parser import (
        parse_engrad)

    for path in sorted(glob.glob(os.path.join(calcdir, '*.engrad'))):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
                text = handler.read()
        except OSError:
            continue
        if 'The current total energy in Eh' not in text:
            continue
        energy, _gradient = parse_engrad(path, 0)
        return [float(energy[0])]
    return None


def _engine_cp2k(calcdir):
    """CP2K total energy from the LAST `Total energy:` of its run log.

    Two readings make this a rule rather than a grep. CP2K prints the line once
    per converged SCF and again after a restart, and the skill's own parser keeps
    the last value (`cp2k_skill/service/parser/log_parser.py:46, 86`), so a
    reader that returned the first would compare a recorded row against a number
    the run overwrote. And the file has to be *claimed*: `attribute_engine` asks
    its readers in sorted order and `CP2K` sorts before `LAMMPS`, so a reader
    that took any `*.log` it found would answer for a LAMMPS log, and a LAMMPS
    number read as CP2K loses the kcal/mol variant that lets it be compared at
    all. The banner the engine always prints (` CP2K| version string:`) is the
    evidence, and it lives in the same file the energy is read from.

    The energy is Hartree as it stands, so `ENGINE_VARIANTS` gains no entry.
    
    Read locally on purpose (#3.7): the skill's parser
    (`cp2k_skill/service/parser/log_parser.py:parse_log`) takes the
    run's config to map excitation rows onto the state manifold, and
    a fetched product tree carries no config; the "last Total energy"
    rule below is the same rule that parser applies.
"""
    for path in sorted(glob.glob(os.path.join(calcdir, '*.log'))):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
                lines = handler.read().splitlines()
        except OSError:
            continue
        if not any(line.startswith(' CP2K|') for line in lines):
            continue
        values = []
        for line in lines:
            if 'Total energy:' not in line:
                continue
            tail = line.split('Total energy:', 1)[1].split()
            if not tail:
                continue
            try:
                values.append(float(tail[0]))
            except ValueError:
                continue
        if values:
            return [values[-1]]
    return None


def _engine_pyscf(calcdir):
    """PySCF's total energy, from the LAST `converged SCF energy` it printed.

    Written against the directory a run left on this host, not against a guess:
    `tmp_high_qm_PYSCF/` holds one log and one `.chk`, and the log carries
    `converged SCF energy = -78.4753762463297` followed by a `**** SCF Summaries
    ****` block whose `Total Energy =` row repeats the same number - twice for a
    gradient job, once per state evaluation, because each state runs its own
    ground-state SCF before it solves the response problem. The LAST reading is
    the one the finished step was scored on, which is the rule the CP2K reader
    above already follows for the same reason.

    The claim threshold is the token `converged SCF energy`, which no other
    engine in `ENGINE_READERS` writes: `attribute_engine` asks its readers in
    sorted order and PYSCF comes before XTB, so a reader that took any `*.log`
    would answer for another engine's file and hand back a Hartree number for a
    row that was never one.

    What the reader returns is the ground-state total. An excited-state leg
    records the state energy - that total plus the `Excitation energies (eV)`
    row of the same block - so those rows stay under `NOT_COVERED`'s
    `state_scoped` entry rather than being matched here against a number the
    engine printed for a different quantity. The energy is Hartree as it stands,
    so `ENGINE_VARIANTS` gains no entry.
    
    Read locally on purpose (#3.7): the skill's parser
    (`pyscf_skill/service/parser/output_parser.py:parse_output`) takes
    a runtime record (project, calcdir and the driver) that the audit
    does not rebuild; the token rule below is pinned instead.
"""
    for path in sorted(glob.glob(os.path.join(calcdir, '*.log'))):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
                lines = handler.read().splitlines()
        except OSError:
            continue
        values = []
        for line in lines:
            if 'converged SCF energy' not in line:
                continue
            tail = line.split('converged SCF energy', 1)[1].lstrip(' =\t')
            if not tail:
                continue
            try:
                values.append(float(tail.split()[0]))
            except (IndexError, ValueError):
                continue
        if values:
            return [values[-1]]
    return None


def _engine_molcas(calcdir):
    """MOLCAS ground-state SCF total, from the LAST one it printed.

    The two tokens are the skill's own parser's, which was written against a
    deployment log (`molcas_skill/service/parser/log_parser.py:96, 279`):
    `Happy landing` is the completion marker every finished run prints, and
    `Total SCF energy` is the line whose last token that parser reads as the
    total. The LAST reading is taken for the reason the CP2K and PySCF readers
    take the last one: a restart, or a later module, prints the line again and
    the row on record belongs to the evaluation the run finished on.

    The marker is load-bearing. `attribute_engine` asks its readers in sorted
    order and PYSCF, which also reads `*.log`, sorts after MOLCAS, so a reader
    that took any log it found would answer for a PySCF directory and hand
    back a number no PySCF row was ever compared against.

    No captured MOLCAS log is shipped in this repository, so what the criterion
    pins is the token contract, not an observed line layout. A CASSCF-only run
    prints no SCF total: its log carries `RASSCF root number N Total energy:`
    lines instead, and the reader answers those (the last occurrence of each
    root, in root order) - the shape `examples/md_molcas_fssh.yaml` leaves, whose
    state-resolved record carries the same root energies. The dual
    singlet/triplet block layout is still unpinned by any captured log. The
    energy is Hartree as it stands, so `ENGINE_VARIANTS` gains no entry.
    
    Read locally on purpose (#3.7): the skill's parser
    (`molcas_skill/service/parser/log_parser.py:parse_log`) takes the
    run's nstate/natom/ci/spin and its coupling request, none of which
    a fetched product tree carries; the two tokens below are that
    parser's own.
"""
    for path in sorted(glob.glob(os.path.join(calcdir, '*.log'))):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
                lines = handler.read().splitlines()
        except OSError:
            continue
        if not any('Happy landing' in line for line in lines):
            continue
        values = []
        roots = {}
        for line in lines:
            if 'Total SCF energy' in line:
                tail = line.split()
                if tail:
                    try:
                        values.append(float(tail[-1]))
                    except ValueError:
                        pass
                continue
            match = re.search(
                r'RASSCF root number\s+(\d+)\s+Total energy:\s*'
                r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            if match:
                # A CASSCF-only run prints no SCF total: its energies are the
                # RASSCF roots, and the state-resolved record (an FSSH leg)
                # carries them as its per-state columns. The last occurrence of
                # each root is kept, the same last-value rule the SCF branch
                # follows; the directory holds one evaluation, so a restart
                # would otherwise leave an older block winning.
                roots[int(match.group(1))] = float(match.group(2))
        if values:
            return [values[-1]]
        if roots:
            return [roots[number] for number in sorted(roots)]
    return None


# Skills whose calculation directory can be read back for section C.
# Anything else is reported as not covered rather than quietly skipped.
ENGINE_READERS = {'OPENQP': _engine_openqp, 'XTB': _engine_xtb,
                  'BAGEL': _engine_bagel, 'LAMMPS': _engine_lammps,
                  'ORCA': _engine_orca, 'CP2K': _engine_cp2k,
                  'PYSCF': _engine_pyscf, 'MOLCAS': _engine_molcas}


# An engine value may be recorded after a documented unit conversion. LAMMPS
# writes its energy in the unit style the skill was configured with (real:
# kcal/mol, metal: eV), and that style is a config value rather than a fact
# readable from the file, so every documented conversion counts as the same
# number here; matching none of them is a real disagreement.
ENGINE_VARIANTS = {
    'LAMMPS': lambda value: [value,
                             value * KCAL_TO_HARTREE,
                             value * EV_TO_HARTREE],
}


def attribute_engine(calcdir):
    """Which registered engine's own files this directory holds, or None.

    A calculation directory is named after the run's `project`, not after its
    backend (`CHANGELOG.md` #354), so `tmp_water25` is a LAMMPS directory while
    `tmp_CP2K_num_xyp` is a CP2K one. Reading a name token for the engine is
    therefore not a reading at all: it mislabels every project-named case, and
    the mislabel used to be load-bearing, because the unit-variant table below
    is keyed by engine and a LAMMPS directory that did not call itself LAMMPS
    lost its kcal/mol-to-Hartree conversions and reported its own recorded
    energy as a disagreement. What a directory *contains* is the evidence, and
    each reader already returns nothing when the file it reads is absent, so
    the attribution asks the readers in turn and takes the first that answers.
    """
    for tag in sorted(ENGINE_READERS):
        values = ENGINE_READERS[tag](calcdir)
        if values:
            return tag, list(values)
    return None, []


def round_start(work_dir):
    """Instant the round created its directory, from `tmp_<stamp>`."""
    stamp = re.search(r'tmp_(\d{8})_(\d{6})', os.path.basename(work_dir))
    if not stamp:
        return 0.0
    return time.mktime(time.strptime(''.join(stamp.groups()),
                                     '%Y%m%d%H%M%S'))


def frame_files(run_dir):
    """Frame records in the run directory and one level below it."""
    found = []
    for path in glob.glob(os.path.join(run_dir, '*')):
        if os.path.isfile(path) and FRAME.search(path):
            found.append(path)
    for sub in glob.glob(os.path.join(run_dir, '*/')):
        for path in glob.glob(os.path.join(sub, '*')):
            if os.path.isfile(path) and FRAME.search(path):
                found.append(path)
    return sorted(found)


def titles_of(files, kind):
    """Run titles named by the aggregate records `<title>.md.<kind>`."""
    suffix = '.md.%s' % kind
    return sorted({os.path.basename(path)[:-len(suffix)] for path in files
                   if path.endswith(suffix)})


def scoped(files, title, kind):
    """The records belonging to one title, the rule the runner judges by."""
    prefix = title + '.md'
    suffix = '.%s' % kind
    keep = []
    for path in files:
        name = os.path.basename(path)
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        if re.match(r'^\.sh(\.\d+)?$', name[len(prefix):-len(suffix)]):
            continue
        keep.append(path)
    return sorted(keep)


def merge_metrics(paths, compute):
    """Worst value of each metric over the given records."""
    merged = {}
    for path in paths:
        for name, value in compute(path).items():
            merged[name] = max(merged.get(name, value), value)
    return merged


def energy_metrics(paths):
    """The energy metrics the runner would report for these records.

    The trajectory total decides and the widest per-center band is carried
    as `etot_center_drift`, through the runner's own predicate
    (`runner.is_center_record`): a center's own file is not a constant of
    the motion, and merging the two kinds here would have this verifier
    disagree with a round that had judged the composite.
    """
    def compute(path):
        return energy_conservation_metrics(read_md_energies(path))

    disk = merge_metrics([path for path in paths
                          if not is_center_record(path)], compute)
    centers = [compute(path).get('etot_drift', 0.0)
               for path in paths if is_center_record(path)]
    if centers:
        disk['etot_center_drift'] = max(centers)
    return disk


def example_case_and_mode(name):
    """Map `example_<case>_<mode>` to its case and mode directory."""
    stem = name[len('example_'):] if name.startswith('example_') else name
    for mode in ('yaml', 'input'):
        if stem.endswith('_' + mode):
            return stem[:-len(mode) - 1], mode
    return None, None


def _is_adaptive_run_dir(run_dir):
    """Whether this run directory belongs to an adaptive example.

    Adaptive sampling has its own section J: its products are summaries
    and iteration directories, not the per-step records sections A-I
    compare. The generic audits skip it rather than reporting every
    missing `.md.energies` as a defect.
    """
    return bool(glob.glob(os.path.join(run_dir, '*.adaptive')))


def audit_records(work_dir, examples_dir=''):
    """Section A: recorded metrics vs the frame records on disk."""
    problems = []
    counts = {'entries': 0, 'compared': 0, 'disagreeing': 0,
              'not_recomputable': 0, 'stale': 0,
              'missing_dir': 0}
    path = os.path.join(work_dir, 'results.json')
    if not os.path.isfile(path):
        return [('NO RESULTS: %s is missing' % path)], counts
    start = round_start(work_dir)
    with open(path, 'r', encoding='utf-8') as handler:
        entries = json.load(handler).get('example_results', [])
    for entry in entries:
        name = entry.get('name', '')
        energy = (entry.get('metrics') or {}).get('metrics') or {}
        box = (entry.get('box_metrics') or {}).get('metrics') or {}
        if not energy and not box:
            continue
        counts['entries'] += 1
        case, mode = example_case_and_mode(name)
        run_dir = os.path.join(work_dir, 'examples', case, mode) if case else ''
        if not run_dir or not os.path.isdir(run_dir):
            counts['missing_dir'] += 1
            problems.append('A NO DIR: %s -> %s' % (name, run_dir))
            continue
        files = frame_files(run_dir)
        if files and max(os.path.getmtime(p) for p in files) < start:
            counts['stale'] += 1
            problems.append('A STALE: %s predates the round' % name)

        for recorded, kind, compute, tolerance in (
                (energy, 'energies', None, METRIC_TOL),
                (box, 'xyz',
                 _box_metrics_for(work_dir, examples_dir, case, run_dir),
                 1e-9)):
            if not recorded:
                continue
            titles = titles_of([p for p in files if p.endswith('.' + kind)],
                               kind)
            disk = {}
            for title in titles:
                found = scoped(files, title, kind)
                merged = (energy_metrics(found) if compute is None
                          else merge_metrics(found, compute))
                for key, value in merged.items():
                    disk[key] = max(disk.get(key, value), value)
            for key, value in sorted(recorded.items()):
                got = disk.get(key)
                if got is None:
                    # Nothing on disk carries this metric, so there is no
                    # second reading to compare against -- which is not the
                    # same finding as a value that disagrees. Reporting it as
                    # a mismatch is how a missing `--examples` produced 29
                    # fake `temperature_drift` alarms: without the source YAML
                    # the target temperature is unknown, and
                    # `cell_frame_metrics` then never emits the key at all.
                    counts['not_recomputable'] += 1
                    continue
                counts['compared'] += 1
                if abs(float(got) - float(value)) > tolerance:
                    counts['disagreeing'] += 1
                    problems.append('A MISMATCH: %s.%s recorded=%.9g '
                                    'disk=%s' % (name, key, value, got))
    return problems, counts


def _box_metrics_for(work_dir, examples_dir, case, run_dir):
    """Build the cell metric reader, which needs the run's temperature."""
    def compute(path):
        temperature = 0.0
        if examples_dir:
            spec_path = os.path.join(examples_dir, '%s.yaml' % case)
            if os.path.isfile(spec_path):
                import yaml
                with open(spec_path, 'r',
                          encoding='utf-8') as handler:
                    spec = yaml.safe_load(handler) or {}
                temperature = float((spec.get('md') or {}).get(
                    'temperature', 0.0) or 0.0)
        return cell_frame_metrics(
            read_md_lattice(path),
            read_md_ensemble(path.replace('.xyz', '.ensemble'),
                             ('T_inst(K)',)),
            temperature)

    return compute


def step_blocks(log_path):
    """Per-step (epot, ekin, current_state) triples read from a step log.

    The step boundaries are the log's own `*  Step  N` markers; inside one
    block the framework writes the state cursor, the per-state energies and
    the kinetic energy as separate sections. The titles come from
    `print_results`, the module that writes them, so a renamed section shows
    up here as nothing-to-compare instead of a silent empty pass.
    """
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as handler:
        lines = handler.read().splitlines()
    marks = [index for index, line in enumerate(lines)
             if STEP_MARKER.match(line)]
    steps = []
    for position, start in enumerate(marks):
        end = (marks[position + 1] if position + 1 < len(marks)
               else len(lines))
        block = lines[start:end]
        section = None
        states, current, kinetic = [], None, None
        for line in block:
            if line.strip() == SEPARATOR:
                continue
            match = SECTION_LINE.match(line)
            if match:
                title = match.group(1).strip().lower()
                if title == SECTION_ENERGIES.lstrip('&').lower():
                    section = 'e'
                elif title == SECTION_KINETIC.lstrip('&').lower():
                    section = 'k'
                elif title.startswith('electronic state'):
                    section = 's'
                else:
                    section = None
                continue
            if section == 'e':
                state = STATE_LINE.match(line)
                if state:
                    states.append(float(state.group(2)))
            elif section == 's':
                cursor = CURRENT_STATE.match(line)
                if cursor:
                    # The writer prints the 1-based state; the record indexes
                    # the energy array from zero.
                    current = int(cursor.group(1)) - 1
            elif section == 'k':
                value = KINETIC_LINE.match(line)
                if value:
                    kinetic = float(value.group(1))
                    section = None
        if states:
            steps.append({'states': states, 'current': current,
                          'kinetic': kinetic})
    return steps


def audit_log(work_dir):
    """Section B: each logged step against the same step's frame record."""
    problems = []
    counts = {'pairs': 0, 'steps': 0, 'epot_diverged': 0, 'ekin_diverged': 0,
              'unaligned': 0, 'no_record': 0, 'multi_state': 0}
    for run_dir in sorted(glob.glob(os.path.join(work_dir, 'examples',
                                                 '*', '*'))):
        if not os.path.isdir(run_dir) or _is_adaptive_run_dir(run_dir):
            continue
        scopes = [run_dir] + sorted(
            path for path in glob.glob(os.path.join(run_dir, '*'))
            if os.path.isdir(path))
        for scope in scopes:
            for log_path in sorted(glob.glob(os.path.join(scope, '*.log'))):
                stem = os.path.basename(log_path)[:-len('.log')]
                record = os.path.join(scope, '%s.md.energies' % stem)
                if not os.path.isfile(record):
                    continue
                rows = read_md_energies(record)
                steps = step_blocks(log_path)
                if not steps or not rows:
                    counts['no_record'] += 1
                    problems.append('B NOTHING TO COMPARE: %s (log=%d '
                                    'record=%d)' % (
                                        os.path.relpath(log_path, work_dir),
                                        len(steps), len(rows)))
                    continue
                counts['pairs'] += 1
                compared = min(len(steps), len(rows))
                if compared != len(steps) or compared != len(rows):
                    counts['unaligned'] += 1
                    problems.append('B STEP COUNT: %s log=%d record=%d' % (
                        os.path.relpath(log_path, work_dir), len(steps),
                        len(rows)))
                for index in range(compared):
                    step, row = steps[index], rows[index]
                    counts['steps'] += 1
                    if len(step['states']) > 1:
                        # A multi-state log line list cannot be summed: the
                        # record stores the active state's energy, which is
                        # what `current` selects when the log carried one.
                        counts['multi_state'] += 1
                        chosen = (step['states'][step['current']]
                                  if step['current'] is not None
                                  and step['current'] < len(step['states'])
                                  else None)
                        if chosen is None:
                            continue
                    else:
                        chosen = step['states'][0]
                    if abs(chosen - row[1]) > STEP_TOL:
                        counts['epot_diverged'] += 1
                        problems.append('B EPOT: %s step %d log=%.16f '
                                        'record=%.16f' % (
                                            os.path.relpath(log_path, work_dir),
                                            index, chosen, row[1]))
                    if step['kinetic'] is not None:
                        if abs(step['kinetic'] - row[2]) > STEP_TOL:
                            counts['ekin_diverged'] += 1
                            problems.append('B EKIN: %s step %d log=%.16f '
                                            'record=%.16f' % (
                                                os.path.relpath(log_path,
                                                                work_dir),
                                                index, step['kinetic'],
                                                row[2]))
    return problems, counts


def engine_dirs(run_dir):
    """Surviving calculation directories: `tmp_*` names, as the backend writes them.

    The name says which `project` the run used, not which engine ran, so this
    only collects candidates and the attribution happens in `attribute_engine`.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(run_dir, '*/'))):
        name = os.path.basename(path.rstrip('/'))
        if not name.startswith('tmp_'):
            continue
        found.append((name, path))
    return found


def record_columns(path):
    """All numeric columns of the record's last frame.

    `read_md_energies` keeps the four documented columns; a multicentre or
    NAMD run carries the per-state energies beyond them, and those are the
    numbers an engine export prints.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
        lines = [line for line in handler.read().splitlines()[1:]
                 if line.split()]
    if not lines:
        return []
    try:
        return [float(value) for value in lines[-1].split()]
    except ValueError:
        return []


def layer_of(path):
    """Layer number of a `<title>.md.<n>.energies` record, or None."""
    match = re.search(r'\.md\.(\d+)\.energies$', os.path.basename(path))
    return int(match.group(1)) if match else None


def audit_engine(work_dir):
    """Section C: every recorded potential energy against the engine numbers.

    The check runs in the safe direction: a number the framework recorded has
    to be a number a surviving engine file actually printed. Asking it the
    other way round would need a per-task attribution that the round does not
    record, and an engine export legitimately holds lines the framework does
    not keep (OpenQP's `energies` starts with the internal triplet reference
    for MRSF, which its own parser offsets past), so extra engine values are
    counted and reported, never failed.

    An aggregate record of a layered run stores the sum over centres rather
    than any single engine value, so those rows are classified and skipped,
    not compared.
    """
    problems = []
    counts = {'runs': 0, 'checked': 0, 'matched': 0, 'unmatched': 0,
              'composite': 0, 'constraint_adjusted': 0, 'no_engine': 0,
              'empty_record': 0, 'layer_unattributed': 0,
              'reader_missing': 0, 'engine_extra': 0, 'term_outcomes': 0}
    tags = set()
    unread = set()
    for run_dir in sorted(glob.glob(os.path.join(work_dir, 'examples',
                                                 '*', '*'))):
        if not os.path.isdir(run_dir) or _is_adaptive_run_dir(run_dir):
            continue
        records = [p for p in frame_files(run_dir) if p.endswith('.energies')]
        # A field-only step (the QM/MM probes, the layer that only rendered an
        # `.esfp`) leaves calculation directories but no frame record to
        # compare. Its readers still count as covered, but its directories are
        # not "unread": counting them made `reader_missing` mostly noise (104
        # of 148 in tmp_20261110_093321 belonged to runs with no `.energies`).
        if records:
            counts['runs'] += 1
        # A layered leg records an assembled total (for example
        # `high[qm] + high[qm2] - low[qm2]`), so no single engine file
        # contains it. A layer directory that kept only orbitals is expected
        # there, not a reader this tool lacks.
        leg_composite = _leg_declares_a_region(run_dir)
        run_term_outcomes = len(
            glob.glob(os.path.join(run_dir, '**', 'term_outcome.json'),
                      recursive=True))
        counts['term_outcomes'] += run_term_outcomes
        engines = {}
        for name, calcdir in engine_dirs(run_dir):
            tag, values = attribute_engine(calcdir)
            if tag is None:
                # No registered reader recognises what is in there. Report the
                # directory, never a skill inferred from its name - but report
                # what the directory holds, so the name answers why: a reader
                # that lost its token and a directory the tool has no reader for
                # at all used to print the same string, and only `NOT_COVERED`
                # said which case this was.
                if records and not leg_composite:
                    counts['reader_missing'] += 1
                    try:
                        inventory = sorted(entry for entry in os.listdir(calcdir)
                                           if not entry.endswith('.chk'))
                    except OSError:
                        inventory = ['<unreadable>']
                    unread.add('%s (%s)' % (name, ', '.join(inventory)[:160]
                                            or 'empty'))
                continue
            tags.add(tag)
            if values:
                engines[calcdir] = (tag, values)
        if not records:
            continue
        if not engines:
            counts['composite' if leg_composite else 'no_engine'] += 1
            continue
        # A run that reports energy contributions folds them into the
        # recorded Epot (the constraint bias is part of what the framework
        # stores), so the recorded number is deliberately not the raw engine
        # number. The framework prints a section for exactly this, so its own
        # marker, not a guess, decides whether a row is comparable.
        adjusted = False
        for log_path in glob.glob(os.path.join(run_dir, '*.log')):
            with open(log_path, 'r', encoding='utf-8',
                      errors='ignore') as handler:
                if SECTION_CONTRIBUTIONS.lstrip('&').lower() in \
                        handler.read().lower():
                    adjusted = True
                    break
        pool = []
        for calcdir, (tag, values) in engines.items():
            variants = ENGINE_VARIANTS.get(tag)
            for value in values:
                pool.append(value)
                if variants:
                    pool.extend(variants(value))
        layers = [p for p in records if layer_of(p) is not None]
        # A per-centre row belongs to the engine that computed that centre,
        # and this tool can only tell which one is which when the round wrote
        # per-term attribution (`term_outcome.json`, present only when the
        # Calculator dispatches terms remotely -- 0 in these rounds) or when a
        # single engine directory is in play. With several calcdirs and a
        # centre no engine directory covers at all (a deleted scratch, or an
        # artifact type no reader answers for), the
        # centre's Hartree row was being
        # compared against the low layer's kcal/mol exports and reported as a
        # problem: 8 false alarms over the multicentre examples in rounds
        # tmp_20261011_184226 and tmp_20261011_224520. Unattributed, not
        # wrong -- the same direction the engine-extra rule already takes.
        attributable = len(engines) == 1 or run_term_outcomes > 0
        for record in records:
            if leg_composite:
                counts['composite'] += 1
                continue
            if layer_of(record) is None and layers:
                counts['composite'] += 1
                continue
            if layer_of(record) is not None and not attributable:
                counts['layer_unattributed'] += 1
                continue
            if adjusted:
                counts['constraint_adjusted'] += 1
                continue
            columns = record_columns(record)
            if len(columns) < 2:
                counts['empty_record'] += 1
                continue
            counts['checked'] += 1
            recorded = columns[1]
            if any(abs(recorded - value) <= 1e-8 for value in pool):
                counts['matched'] += 1
            else:
                counts['unmatched'] += 1
                problems.append(
                    'C RECORDED ENERGY NOT IN ENGINE FILES: %s Epot=%.10f '
                    '(engine values: %s)' % (
                        os.path.relpath(record, work_dir), recorded,
                        ' '.join('%.6f' % value for value in pool[:8])))
        # An engine number that reached the record through a documented unit
        # conversion is recorded, not extra: the comparison has to expand the
        # same variants the pool above does, or every LAMMPS run - whose energy
        # file is written in kcal/mol while the record is in Hartree - reports
        # its own matched number a second time as an unrecorded engine value.
        columns = [column for record in records
                   for column in record_columns(record)]
        for tag, values in engines.values():
            variants = ENGINE_VARIANTS.get(tag)
            for value in values:
                candidates = variants(value) if variants else [value]
                if not any(abs(candidate - column) <= 1e-8
                           for candidate in candidates
                           for column in columns):
                    counts['engine_extra'] += 1
    counts['engines_matched'] = sorted(tags)
    counts['dirs_unread'] = sorted(unread)
    counts['readers_registered'] = sorted(ENGINE_READERS)
    return problems, counts

def _leg_declares_a_region(run_dir):
    """Whether this leg assembles its energy from more than one layer.

    Read from the flat config the plan laid down beside the run, the same file
    the run was launched from: a declared managed region (`highlevel`, or
    `midlevel`) is what makes the recorded energy an ONIOM assembly rather than
    one engine's number. An unreadable or absent config answers False, which
    leaves the leg compared - the direction that reports rather than hides.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as handler:
                config = json.load(handler)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        for key in ('highlevel', 'midlevel'):
            value = config.get(key)
            if isinstance(value, (list, tuple)) and value:
                return True
    return False


def _leg_declares_a_pressure(run_dir):
    """Whether this leg records an enthalpy rather than one engine's energy.

    A run under `opt_pressure` steps on `E + pV` and records it: the objective
    carries the same `pV` its lattice gradient already does
    (`optimizer/base.py::_enthalpy`), so the energy column of `opt_status.txt` is
    a number no engine file printed - the same shape a composite leg has, for a
    different reason. The declaration is read off the leg's own flat config, the
    same source the region test above uses.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as handler:
                config = json.load(handler)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        try:
            if float(config.get('opt_pressure') or 0.0):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _leg_drives_an_excited_state(run_dir):
    """Whether this leg records a state energy rather than the SCF total.

    An excited-state leg records `S0 + excitation`. Round
    `tmp_20261018_141444` measured the two sides of that gap on one geometry:
    the record's last row said -13.5101372537, the central CP2K directory's last
    `Total energy:` said -13.56505323574282, and the engine's own
    `TDDFPT| 1 1.49462 eV` line is the difference. Adding those two here would
    make this tool a second implementation of the skill's state arithmetic, which
    is the drift this file has been fixed for elsewhere. The declaration is read
    off the leg's own flat config (`opt_state_i`), the same source the two scope
    tests around it use.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as handler:
                config = json.load(handler)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        try:
            if int(config.get('opt_state_i') or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _leg_stopped_on_its_budget(run_dir):
    """Whether the run's own log closed on an optimization budget stop.

    The verdict a round records is `pass` either way - a truncated test
    optimization is expected to stop unconverged - so the only place the two
    kinds of pass differ is the run's own closing line. Until #399 the row
    carried no marker on the returncode-0 route, and all 32 variable-cell legs of
    round `tmp_20261019_200652` read as plain passes; counting the closure here
    is what makes the distinction auditable from the products instead of only
    from the round log, and it reuses the runner's own two definitions
    (`closing_status`, `optimization_budget_stop`) rather than re-deriving
    what a budget stop is.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, '*.log'))):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
                lines = handler.read().splitlines()
        except OSError:
            continue
        status = closing_status(lines)
        if status and optimization_budget_stop(status):
            return True
    return False


def audit_optimize_engine(work_dir):
    """Section C2: the last energy a variable-cell or relaxation record wrote.

    Section C compares MD frame records, and an optimize round has none - which
    is why rounds of the optimize stage report `runs 0` there and nothing in the
    suite re-reads an optimizer's energy from the engine that printed it. This
    closes that half for the record the optimizer itself keeps
    (`utilities/opt_formatter.py`'s table): the energy of its **last row** has to be
    a number a surviving engine file printed, after the same documented unit
    conversions section C allows.

    Only the last row is comparable. A backend's calculation directory is named
    per task and rewritten at every evaluation, so the round leaves one engine
    number on disk and it belongs to one step - asking more than that of it would
    be asking the tool to invent the missing frames.

    What this can catch and nothing else does: the parse and the unit conversion
    between an engine's energy export and the row the convergence criteria were
    judged on. Section D checks that row against the verdict and the round log,
    which are all written downstream of the same parse.

    What it deliberately does not ask: a leg that declares a managed region
    assembles `L(real) + H(model) - L(model)`, and that sum is in no engine file
    - the files hold its three terms. Such a leg is counted under `composite`
    and left unjudged, which is a scope statement about the assembly, not a
    waiver: a single-engine leg with a row that no engine printed still reports.
    A leg driven on an excited state (`opt_state_i` > 0) is the same case in the
    other direction - its row is the total plus an excitation no engine line
    prints - and is counted under `state_scoped`.
    """
    problems = []
    counts = {'legs': 0, 'checked': 0, 'matched': 0, 'unmatched': 0,
              'no_engine': 0, 'unreadable': 0, 'foreign': 0, 'composite': 0,
              'pressurized': 0, 'state_scoped': 0, 'budget_stopped': 0}
    for run_dir in sorted(glob.glob(os.path.join(work_dir, 'examples',
                                                 '*', '*'))):
        status_path = os.path.join(run_dir, 'opt_status.txt')
        if not os.path.isdir(run_dir) or not os.path.isfile(status_path):
            continue
        counts['legs'] += 1
        record = read_status(status_path)
        if _leg_stopped_on_its_budget(run_dir):
            # Counted before the scope exemptions below, because the closure is a
            # fact about the run's own log and not about the engine comparison: a
            # pressure or composite leg stops on its budget like any other.
            counts['budget_stopped'] += 1
        if record.writer is WRITER_FOREIGN:
            counts['foreign'] += 1
            continue
        if not record.rows:
            counts['unreadable'] += 1
            continue
        if _leg_declares_a_pressure(run_dir):
            # `E + pV` is in no engine file either: the files hold the energy,
            # and the volume the run multiplied is the framework's own reading of
            # the box it moved. Reported as an objective, never as a defect.
            counts['pressurized'] += 1
            continue
        if _leg_drives_an_excited_state(run_dir):
            # The row is the state energy and the engine files hold the ground
            # state's total. Left unjudged on the same footing as the two scope
            # cases below, so an excited round does not read as four defects.
            counts['state_scoped'] += 1
            continue
        if _leg_declares_a_region(run_dir):
            # A single-center composite records the assembled objective
            # `L(real) + H(model) - L(model)`, so no engine file in the leg
            # prints its energy column - the numbers on disk are the three
            # terms, not the sum. Comparing the row against one of them would
            # report a correct run as a defect, which is the false alarm this
            # tool has been fixed for elsewhere. The region declaration is read
            # off the leg's own flat config because it is what defines the
            # assembly: `optimize_ts_multiscale` composites on ONE theory, so
            # the length of `theory` would not find it.
            counts['composite'] += 1
            continue
        pool = []
        for name, calcdir in engine_dirs(run_dir):
            tag, values = attribute_engine(calcdir)
            if not values:
                continue
            variants = ENGINE_VARIANTS.get(tag)
            for value in values:
                pool.append(value)
                if variants:
                    pool.extend(variants(value))
        if not pool:
            counts['no_engine'] += 1
            continue
        recorded = record.rows[-1].get('energy')
        if recorded is None:
            counts['unreadable'] += 1
            continue
        counts['checked'] += 1
        if any(abs(recorded - value) <= 1e-8 for value in pool):
            counts['matched'] += 1
        else:
            counts['unmatched'] += 1
            problems.append(
                'C2 RECORDED ENERGY NOT IN ENGINE FILES: %s row %d '
                'energy=%.10f (engine values: %s)' % (
                    os.path.relpath(status_path, work_dir),
                    int(record.rows[-1].get('step', 0)), recorded,
                    ' '.join('%.6f' % value for value in pool[:8])))
    return problems, counts


def term_columns(record):
    """The composite term columns of one optimizer table.

    `status_header` writes the seven atomic columns, the five cell columns when
    the run is variable-cell, then one column per assembled term
    (`composite.build_composite_terms`, labelled by `term_label`), so what
    follows those twelve - or those seven - is the decomposition. Read from the
    header rather than counted from the end of the row, because the term count
    follows the layer and center plan and varies by example.
    """
    columns = list(record.columns)
    if columns[:len(STATUS_ATOM_COLUMNS)] != list(STATUS_ATOM_COLUMNS):
        return []
    body = columns[len(STATUS_ATOM_COLUMNS):]
    if body[:len(STATUS_CELL_COLUMNS)] == list(STATUS_CELL_COLUMNS):
        body = body[len(STATUS_CELL_COLUMNS):]
    # A term is labelled `<region>[<role>]`. The columns after the atomic
    # block are not all terms: an ordinary optimizer appends `neg_modes` and
    # `mode_eig`, and a cell optimization appends its cell diagnostics. The
    # bracket is what the composite writer owns, so it is the discriminator.
    return [name for name in body if '[' in name and name.endswith(']')]


def term_deviations(record):
    """`(step, sum(term columns) - energy)` for every row that carries them all.

    A leg with no term columns returns `[]`: a single-engine relaxation has one
    writer for its energy column and nothing beside it to add up.
    """
    terms = term_columns(record)
    if not terms:
        return []
    found = []
    for row in record.rows:
        values = [row.get(name) for name in terms]
        energy = row.get('energy')
        if energy is None or any(value is None for value in values):
            continue
        found.append((int(row.get('step', 0)),
                      float(sum(float(value) for value in values))
                      - float(energy)))
    return found


def _leg_is_a_crossing(run_dir):
    """Whether this leg's objective carries a crossing's own gap term.

    A crossing optimizes the average of two surfaces and adds its gap term
    (`optimizer/meci.py::_cell_gap_conjugate`), and that term reaches the `energy`
    column without reaching the per-term columns - which is the whole reason the
    decomposition below does not close for one of these. Read off the leg's own
    flat config, the same source the three scope helpers above use:
    `optimize.method` is written there as `opt_method`.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as handler:
                config = json.load(handler)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        if str(config.get('opt_method') or '').lower() in ('meci', 'mecp'):
            return True
    return False


def audit_term_decomposition(work_dir):
    """Section C3: the optimizer's table, added up.

    The writer spells one signed contribution per assembled term
    (`composite.build_composite_terms`, #384), precisely so that a row's columns
    add up to the `energy` column beside them - that sum is the machine-checkable
    identity the table was built to carry. One assembly escapes it, and the
    escape is a property of the leg's own declaration rather than of the numbers:
    a crossing subtracts a gap term
    (`optimizer/meci.py::_cell_gap_conjugate`) which no engine prints and no
    column carries, so its rows sum to the energy MINUS that term. Measured on
    round `tmp_20261019_200652`: `optimize_ts_multiscale` (a `ts` leg, and a
    composite one) closes to exactly 0.0 on all 32 rows of each carrier, while
    the four `meci`/`mecp` legs sum below by 1.16e-02 to 7.74e-02 with the offset
    shrinking as the search walks toward the intersection.

    So the leg's declaration decides which question is being asked. A leg that is
    not a crossing must close, and a deviation of EITHER sign is named - the
    columns dropped, doubled, or read from the wrong end of the row all land
    there. A crossing's offset is a reading, printed with the leg it belongs to
    (`offsets`), because that number is what a round is asked to keep even where
    nothing is claimed from it. A row that does not carry every term column its
    header declares is the table broken, and is named: the sections downstream
    read those columns by name.
    """
    problems = []
    counts = {'legs': 0, 'rows': 0, 'exact': 0, 'offset_readings': 0,
              'ragged': 0, 'offsets': []}
    for run_dir in sorted(glob.glob(os.path.join(work_dir, 'examples', '*',
                                                 '*'))):
        status_path = os.path.join(run_dir, 'opt_status.txt')
        if not os.path.isdir(run_dir) or not os.path.isfile(status_path):
            continue
        record = read_status(status_path)
        if record.writer is WRITER_FOREIGN or not record.rows:
            continue
        terms = term_columns(record)
        if not terms:
            continue
        counts['legs'] += 1
        deviations = term_deviations(record)
        counts['rows'] += len(deviations)
        if len(deviations) != len(record.rows):
            counts['ragged'] += 1
            problems.append(
                'C3 RAGGED TERM ROW: %s declares %d term column(s) (%s) and %d '
                'of its %d rows do not carry them'
                % (os.path.relpath(status_path, work_dir), len(terms),
                   ' '.join(terms), len(record.rows) - len(deviations),
                   len(record.rows)))
        if not deviations:
            continue
        step, worst = max(deviations, key=lambda item: abs(item[1]))
        counts['offsets'].append('%s %+.6e'
                                 % (os.path.relpath(status_path, work_dir),
                                    worst))
        if abs(worst) <= TERM_TOL:
            counts['exact'] += 1
            continue
        if _leg_is_a_crossing(run_dir):
            counts['offset_readings'] += 1
            continue
        problems.append(
            'C3 TERM COLUMNS DO NOT CLOSE: %s row %d: sum(term columns) - '
            'energy = %+.6e, and this leg declares no crossing to account for '
            'the difference' % (os.path.relpath(status_path, work_dir), step,
                                worst))
    counts['offsets'] = sorted(counts['offsets'])
    return problems, counts


def audit_carrier_parity(work_dir):
    """Section E: one example's two carriers have to record the same numbers.

    A round runs every example twice - from its YAML and from the generated
    classic input - and the two stop in sibling directories. Both of the last two
    defects in this area were found by comparing them: a flat config the agent
    entry refused, and three dispatch keywords the writer never emitted. The
    comparison is therefore a section rather than an operator's habit. It
    compares the two numbers the round itself reports: `cell_metrics` from
    `results.json`, and each record's own term decomposition (its identity status
    through the worst deviation).

    Measured on round `tmp_20261019_200652`: 24 pairs, every metric equal except
    `optimize_cell_gap_cp2k`'s `cell_drms`/`cell_dmax`, which differ in the tenth
    significant digit (3.4e-10 relative) - inside `PARITY_TOL`, which is why that
    tolerance is stated as a reading there.
    """
    problems = []
    counts = {'pairs': 0, 'metrics': 0, 'term_pairs': 0, 'disagreeing': 0}
    path = os.path.join(work_dir, 'results.json')
    if not os.path.isfile(path):
        return [('NO RESULTS: %s is missing' % path)], counts
    with open(path, 'r', encoding='utf-8') as handler:
        entries = json.load(handler).get('example_results', [])
    carriers = {}
    for entry in entries:
        case, mode = example_case_and_mode(entry.get('name', ''))
        if case and mode:
            carriers.setdefault(case, {})[mode] = entry
    for case, modes_of in sorted(carriers.items()):
        if len(modes_of) < 2:
            continue
        counts['pairs'] += 1
        modes = sorted(modes_of)
        for index, left in enumerate(modes):
            for right in modes[index + 1:]:
                recorded_left = modes_of[left].get('cell_metrics') or {}
                recorded_right = modes_of[right].get('cell_metrics') or {}
                for key in sorted(set(recorded_left) & set(recorded_right)):
                    try:
                        one = float(recorded_left[key])
                        other = float(recorded_right[key])
                    except (TypeError, ValueError):
                        continue
                    counts['metrics'] += 1
                    limit = max(PARITY_TOL,
                                PARITY_TOL * max(abs(one), abs(other)))
                    if abs(one - other) > limit:
                        counts['disagreeing'] += 1
                        problems.append(
                            'E CARRIER DIVERGENCE: %s %s vs %s: %s = %.12g '
                            'against %.12g' % (case, left, right, key, one,
                                               other))
        worst = {}
        for mode in modes:
            status_path = os.path.join(work_dir, 'examples', case, mode,
                                       'opt_status.txt')
            if not os.path.isfile(status_path):
                continue
            deviations = term_deviations(read_status(status_path))
            if deviations:
                worst[mode] = max(deviations,
                                  key=lambda item: abs(item[1]))[1]
        if len(worst) < 2:
            continue
        counts['term_pairs'] += 1
        items = sorted(worst.items())
        for index, (left, one) in enumerate(items):
            for right, other in items[index + 1:]:
                limit = max(TERM_TOL, PARITY_TOL * max(abs(one), abs(other)))
                if abs(one - other) > limit:
                    counts['disagreeing'] += 1
                    problems.append(
                        'E TERM DIVERGENCE: %s %s vs %s: sum(term columns) - '
                        'energy = %+.6e against %+.6e'
                        % (case, left, right, one, other))
    return problems, counts


def cell_verdict(record):
    """The `[CELL]` sentence a variable-cell run's verdict prints.

    Rendered the way the runner prints it (`qualification/runner.py`), so the
    three sources of one verdict are compared as the same sentence rather than
    through a tolerance invented here.

    Parameters
    ----------
    record : OptStatusRecord
        A parsed `opt_status.txt`.

    Returns
    -------
    str
        Empty when the record holds fewer than two steps, or when it has no cell
        columns: a fixed-cell table cannot render a `[CELL]` sentence, and asking
        it for one used to raise instead of answering.
    """
    if len(record.rows) < 2:
        return ''
    volumes = [row.get('volume') for row in record.rows]
    if any(value is None for value in volumes):
        return ''
    drift = abs(float(volumes[-1]) - float(volumes[0]))
    last = record.rows[-1]
    return ('[CELL] cell columns present; volume moved %.6f over %d steps, '
            'final cell_gmax %.3e cell_grms %.3e'
            % (drift, len(record.rows), float(last['cell_gmax']),
               float(last['cell_grms'])))


def round_log_path(work_dir):
    """The downloaded round log of this work directory, when one is beside it.

    `tools/deploy.py` names it `test_<stamp>.log` after the work directory
    `tmp_<stamp>`, and writes it into the repository root. Searching the work
    directory, its parent and that root means one `--` free call works on a
    round as it was collected.
    """
    stamp = os.path.basename(os.path.normpath(work_dir))
    if stamp.startswith('tmp_'):
        stamp = stamp[4:]
    name = 'test_%s.log' % stamp
    folders = (work_dir, os.path.dirname(os.path.normpath(work_dir)),
               os.path.dirname(os.path.abspath(__file__)))
    for folder in folders:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return ''


def round_log_verdicts(path):
    """Map `task name -> [CELL] sentence` for one downloaded round log."""
    verdicts = {}
    if not path:
        return verdicts
    current = ''
    marker = re.compile(r'^\s*\[\s*\d+/\d+\]\s*[+~]\s+(\S+)')
    with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
        for line in handler:
            found = marker.match(line)
            if found:
                current = found.group(1)
                continue
            if '[CELL] cell columns present' in line and current:
                verdicts[current] = line.strip()
    return verdicts


def shape_change_from_geometry(path):
    """The `shape_ratio_change` of one trajectory, from its own lattice rows.

    Section D's job is to read a recorded number off a second artifact, and
    `shape_ratio_change` had no second source: the status record carries no shape
    column, because the shape lives in `opt_geom.xyz`'s `Lattice=` comments. So
    this re-runs the runner's own ratios (`runner.row_length_ratios`) on the
    first and last frame and takes the larger relative change - the same
    expression `[COUPLING]` is judged with, not a restatement of it.

    Returns
    -------
    tuple of (float or None, int)
        The change, and the number of frames that carried a lattice row; a None
        change means the file cannot answer, which the caller reports rather
        than passes over.
    """
    lattices = read_md_lattice(path)
    if len(lattices) < 2:
        return None, len(lattices)
    first = row_length_ratios(lattices[0])
    last = row_length_ratios(lattices[-1])
    if not first or not last:
        return None, len(lattices)
    return max(abs(last[0] / first[0] - 1.0),
               abs(last[1] / first[1] - 1.0)), len(lattices)


def path_cell_legs(work_dir):
    """The variable-cell path legs of a round, read off their own configs.

    A path leg declares the cell switch like any other search (`optimize_cell`),
    and its method sits beside it (`opt_method`), so the two keys together name
    the legs section D cannot re-read with its own reader. An unreadable config
    answers nothing, which leaves a leg out of the count rather than inventing
    one.
    """
    found = []
    for run_dir in sorted(glob.glob(os.path.join(work_dir, 'examples',
                                                 'path_*', '*'))):
        if not os.path.isdir(run_dir):
            continue
        for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
            try:
                with open(path, 'r', encoding='utf-8') as handler:
                    config = json.load(handler)
            except (OSError, ValueError):
                continue
            if not isinstance(config, dict):
                continue
            if not config.get('optimize_cell'):
                continue
            if str(config.get('opt_method') or '') in ('mep', 'irc', 'neb'):
                found.append(run_dir)
                break
    return found


def _determinant3(flat):
    """|det| of a row-major nine-element cell, in whatever unit it is written."""
    a, b, c, d, e, f, g, h, i = [float(value) for value in flat[:9]]
    return abs(a * (e * i - f * h) - b * (d * i - f * g)
               + c * (d * h - e * g))


def _path_cell_leg(run_dir):
    """One path leg's variable-cell record, read the way the run wrote it.

    `path_status.txt` is not `opt_status.txt`: its header is
    `step direction energy ...` (a band's is `step energy ... image ...`), so
    `utilities/print_results.py::classify_header` answers FOREIGN and `read_status`
    must not be used. The five cell columns
    `utilities/opt_formatter.py::STATUS_CELL_COLUMNS` are looked up by name, and the
    volume has to move between the first and the last row.

    The boxes are then matched to the record, not through the last row: the
    record has one row per **evaluation** (a walker's rejected probes included)
    or per **round** (a band's row is its reading *before* the FIRE step), so
    the two tables describe the same box at different moments. What each kind
    can answer:

    - a walker writes `Point N Direction <d>` for a point it accepted, and every
      accepted point was an evaluation: every frame's box has to appear among
      the rows of the same direction (measured: a rejected probe is why
      `path_cell_irc`'s last row is not the last frame's box);
    - a band writes `Round N Image I Role <role>` *after* the step, while the
      row of round N reports the geometry that round was **evaluated on** -
      which is the geometry round N-1 committed. So a row of step N >= 2 has to
      match one of round N-1's frame boxes (measured on `path_cell_cineb`: round
      2's row reads 1000.1076, the box round 1 left image 2 at).

    The tolerance is set by the print and by the unit round trip, not by the
    physics: the volume column has ten decimals, the `Lattice=` comment nine,
    and the joint vector carries the box through `coord_to_bohr`, whose value is
    rounded against `disp_to_angstrom` (their product is 1 - 7.2e-8, cubed to
    2.2e-7 relative). Measured: the two readings of one box agree to 1.5e-4
    cubic Angstrom at 1000. The limit is 1e-4 absolute or 1e-6 relative,
    whichever is larger - while the defect this check found (a band writing the
    parent box into every frame) was off by 0.638.

    Returns
    -------
    tuple of (list of str, dict)
        The problems this leg has, and its `rows`/`moved`/`matched` readings.
    """
    name = '%s/%s' % (os.path.basename(os.path.dirname(run_dir)),
                      os.path.basename(run_dir))
    readings = {'rows': 0, 'moved': 0, 'matched': 0}
    record = os.path.join(run_dir, 'path_status.txt')
    if not os.path.isfile(record):
        return (['D PATH RECORD: %s carries the cell switch but no '
                 'path_status.txt' % name], readings)
    try:
        with open(record, 'r', encoding='utf-8') as handler:
            lines = [line.split() for line in handler.read().splitlines()
                     if line.strip()]
    except OSError as exc:
        return (['D PATH RECORD: %s could not be read: %s' % (name, exc)],
                readings)
    if not lines:
        return ['D PATH RECORD: %s is empty' % name], readings
    header = lines[0]
    missing = [column for column in STATUS_CELL_COLUMNS
               if column not in header]
    if missing:
        return (['D PATH CELL COLUMNS: %s has no %s (header: %s): they are '
                 'printed only when the lattice is a variable, so their '
                 'absence says the switch was dropped between the input and '
                 'the walk' % (name, missing, ' '.join(header))], readings)
    volume_at = header.index('volume')
    step_at = header.index('step')
    direction_at = header.index('direction') if 'direction' in header else None
    rows = []
    for fields in lines[1:]:
        if len(fields) != len(header):
            continue
        try:
            rows.append({'step': int(float(fields[step_at])),
                         'volume': float(fields[volume_at]),
                         'direction': (fields[direction_at]
                                       if direction_at is not None else '')})
        except (ValueError, IndexError):
            continue
    readings['rows'] = len(rows)
    if len(rows) < 2 or rows[0]['volume'] == rows[-1]['volume']:
        return (['D PATH CELL STATIC: %s reads %s for the volume across %d '
                 'row(s): a box that did not move was not relaxed'
                 % (name, [row['volume'] for row in rows], len(rows))],
                readings)
    readings['moved'] = 1
    geometry = os.path.join(run_dir, 'path_geom.xyz')
    boxes = [_determinant3(flat) for flat in read_md_lattice(geometry)]
    frames = _xyz_frames(geometry)
    if not boxes or len(boxes) != len(frames):
        return (['D PATH CELL NO BOX: %s records %d row(s) and its trajectory '
                 'carries %d frame(s) with %d Lattice= row(s)'
                 % (name, len(rows), len(frames), len(boxes))], readings)

    problems = []
    matched = 0

    def close(one, other):
        return abs(one - other) <= max(1e-4, 1e-6 * abs(other))

    if any(re.search(r'Round\s+(\d+)', str(comment))
           for comment, _coordinates in frames):
        by_round = {}
        for (comment, _coordinates), box in zip(frames, boxes):
            round_at = int(re.search(r'Round\s+(\d+)', str(comment)).group(1))
            by_round.setdefault(round_at, []).append(box)
        for row in rows:
            if row['step'] - 1 not in by_round:
                continue
            matched += 1
            if not any(close(box, row['volume'])
                       for box in by_round[row['step'] - 1]):
                problems.append(
                    'D PATH CELL MISMATCH: %s: step %d was evaluated on a box '
                    'of %.6f cubic Angstrom and no image round %d left has one '
                    '(frames: %s)'
                    % (name, row['step'], row['volume'], row['step'] - 1,
                       ['%.6f' % box for box in by_round[row['step'] - 1]]))
    else:
        for (comment, _coordinates), box in zip(frames, boxes):
            direction = re.search(r'Direction\s+(\w+)', str(comment))
            wanted = direction.group(1) if direction else ''
            candidates = [row['volume'] for row in rows
                          if row['direction'] == wanted]
            if any(close(box, volume) for volume in candidates):
                matched += 1
            else:
                problems.append(
                    'D PATH CELL MISMATCH: %s: a frame box of %.6f cubic '
                    'Angstrom appears in no row of direction %r (last rows: '
                    '%s)' % (name, box, wanted,
                             ['%.6f' % volume for volume in candidates[-3:]]))
    readings['matched'] = matched
    return problems, readings


def audit_cell(work_dir):
    """Section D: the three sources of one variable-cell verdict."""
    problems = []
    counts = {'entries': 0, 'legs': 0, 'missing_records': 0,
              'foreign_records': 0, 'unjudged': 0, 'malformed_rows': 0,
              'metric_disagreements': 0, 'text_disagreements': 0,
              'log_tasks': 0, 'log_disagreements': 0,
              'pressure_legs': 0, 'pressure_disagreements': 0,
              'pressure_missing_rows': 0,
              'shape_legs': 0, 'shape_source_disagreements': 0,
              'shape_unreadable': 0, 'path_cell_legs': 0,
              'path_cell_rows': 0, 'path_cell_moved': 0,
              'path_cell_matched': 0}
    path = os.path.join(work_dir, 'results.json')
    if not os.path.isfile(path):
        return [('NO RESULTS: %s is missing' % path)], counts
    with open(path, 'r', encoding='utf-8') as handler:
        entries = json.load(handler).get('example_results', [])
    verdicts = round_log_verdicts(round_log_path(work_dir))
    counts['log_tasks'] = len(verdicts)
    for entry in entries:
        recorded = entry.get('cell_metrics') or {}
        if not recorded:
            continue
        name = entry.get('name', '')
        counts['entries'] += 1
        case, mode = example_case_and_mode(name)
        run_dir = os.path.join(work_dir, 'examples', case, mode) if case else ''
        status_path = os.path.join(run_dir, 'opt_status.txt') if run_dir else ''
        if not status_path or not os.path.isfile(status_path):
            counts['missing_records'] += 1
            problems.append('D NO RECORD: %s -> %s' % (name, status_path))
            continue
        counts['legs'] += 1
        record = read_status(status_path)
        counts['malformed_rows'] += len(record.malformed)
        if record.writer == WRITER_FOREIGN:
            counts['foreign_records'] += 1
            problems.append('D ENGINE RECORD: %s header is not the framework '
                            'table (%s)' % (name, ' '.join(record.columns)))
            continue
        for key, value in sorted(recorded.items()):
            disk = _cell_metric_from_record(record, key)
            if disk is None:
                continue
            if abs(float(disk) - float(value)) > METRIC_TOL:
                counts['metric_disagreements'] += 1
                problems.append('D MISMATCH: %s.%s recorded=%.9g file=%.9g'
                                % (name, key, value, disk))
        if 'shape_ratio_change' in recorded:
            # The `[COUPLING]` reading, checked against the trajectory rather
            # than against the record that does not carry it. The key was
            # invisible here before: `_cell_metric_from_record` answers None for
            # any metric the status table has no column for, section D skipped
            # such keys in silence, and the one number that decides an
            # anisotropic-versus-isotropic verdict was never reconciled with the
            # file it was computed from.
            counts['shape_legs'] += 1
            change, frames = shape_change_from_geometry(
                os.path.join(run_dir, 'opt_geom.xyz'))
            stated = float(recorded['shape_ratio_change'])
            if change is None:
                counts['shape_unreadable'] += 1
                problems.append('D SHAPE SOURCE: %s records shape_ratio_change '
                                '%.9g, but its trajectory carries %d frame(s) '
                                'with a lattice row, and a shape needs two'
                                % (name, stated, frames))
            elif abs(change - stated) > METRIC_TOL:
                counts['shape_source_disagreements'] += 1
                problems.append('D SHAPE SOURCE: %s.shape_ratio_change records '
                                '%.9g while the first and last Lattice= rows '
                                'of its own trajectory give %.9g'
                                % (name, stated, change))
        if 'pressure_target_bar' in recorded:
            # The pressure clause's three numbers come off the run log's last
            # optimization-progress block, not off the status record: the record
            # carries the cell force in Hartree per Bohr while the judgment is
            # made on the residual of `sigma + p I` in bar. Re-reading the rows
            # here is what catches a transcription between the log and the
            # document, which no other section of this tool looks at.
            counts['pressure_legs'] += 1
            tail = {}
            for log_path in sorted(glob.glob(os.path.join(run_dir, '*.log'))):
                tail = read_optimize_tail(log_path)
                if tail:
                    break
            for key, row in (('pressure_target_bar', 'External Pressure'),
                             ('pressure_residual_rms_bar', 'Cell Stress RMS'),
                             ('pressure_residual_max_bar', 'Cell Stress Max')):
                if key not in recorded or row not in tail:
                    counts['pressure_missing_rows'] += 1
                    continue
                if abs(float(recorded[key]) - float(tail[row])) > METRIC_TOL:
                    counts['pressure_disagreements'] += 1
                    problems.append(
                        'D PRESSURE ROW: %s.%s reads %.6f in results.json and '
                        '%.6f in the log block'
                        % (name, key, float(recorded[key]), float(tail[row])))
        sentence = cell_verdict(record)
        if not sentence:
            counts['unjudged'] += 1
            continue
        details = entry.get('details') or ''
        if sentence not in details:
            counts['text_disagreements'] += 1
            problems.append('D VERDICT TEXT: %s prints %r, the record gives %r'
                            % (name, [line for line in details.splitlines()
                                      if line.startswith('[CELL]')],
                               sentence))
        printed = verdicts.get(name)
        if printed and printed != sentence:
            counts['log_disagreements'] += 1
            problems.append('D LOG LINE: %s log reads %r, the record gives %r'
                            % (name, printed, sentence))
    if not counts['entries']:
        # A path round runs its variable-cell legs too, and they record into
        # `path_status.txt`. That table is not this section's (its header starts
        # with a direction column), so the legs are re-read by
        # `_path_cell_leg`: the five cell columns by name, a volume that moved,
        # and that volume against the leg's own `Lattice=` frames. A round with
        # neither kind of cell leg has nothing for this section to answer.
        legs = path_cell_legs(work_dir)
        counts['path_cell_legs'] = len(legs)
        for run_dir in legs:
            problems_here, readings = _path_cell_leg(run_dir)
            problems.extend(problems_here)
            counts['path_cell_rows'] += readings['rows']
            counts['path_cell_moved'] += readings['moved']
            counts['path_cell_matched'] += readings['matched']
        if not legs:
            # A round without variable-cell examples has no verdict to
            # audit. That is a coverage statement, not a defect: an
            # adaptive-only round would otherwise fail the whole tool on
            # a section it never asked for.
            counts['no_cell_examples'] = 1
    return problems, counts


def _cell_metric_from_record(record, key):
    """One recorded metric, recomputed from the status record.

    A record without cell columns answers None for a volume metric instead of
    raising: a fixed-cell leg's entry can still carry a `cell_metrics` block, and
    what the audit owes there is the same answer it gives for any metric the
    table has no column for - that there is no second reading to compare against.
    Reading the volumes before the key test made that case a KeyError, which
    takes the whole report down rather than naming one unrecomputable metric.
    """
    rows = record.rows
    if not rows:
        return None
    if key == 'steps':
        return float(len(rows))
    volumes = [row.get('volume') for row in rows]
    if any(value is None for value in volumes):
        return None
    if key == 'volume_first':
        return float(volumes[0])
    if key == 'volume_last':
        return float(volumes[-1])
    if key == 'volume_drift':
        return abs(float(volumes[-1]) - float(volumes[0]))
    value = rows[-1].get(key)
    return float(value) if isinstance(value, float) else None


# ---- F: the path family's own claims ----

# A path leg holds a strict condition through the framework's position solve
# (`constraints/runtime.constrain_state`), to the tolerance that leg declares,
# and the artifact the hold is read from is its `path_geom.xyz` - ten decimals
# of an Angstrom. Measured on round `tmp_20261021_195813` over the ten condition
# legs (five examples on two carriers), worst deviation from the first frame, in
# the units the tolerance is stated in: 1.001e-10 Bohr for the held C-N of
# `path_mep_bond`, 9.3e-11 Bohr for the held C-N of `path_irc_bond`, 4.4e-16
# Bohr for the held C-N of the band `path_neb_bond`, 5.5e-11 radian for the held
# H-O-H angle and 9.0e-11 radian for the held H-O-O-H dihedral. The first of
# those sits at the declared tolerance itself, so the rounded reading and not
# the solve is the floor; ten times the declared tolerance is the limit below,
# and it is still 1.5e8 times smaller than what a hold that was not held does -
# the walk moves by its own arc, 0.15 Angstrom, between points.
PATH_HELD_SCALE = 10.0

# A frozen atom is restored to its anchor exactly by the position solve
# (`constraints/projection.py::apply_fix_positions` assigns the anchor's
# components rather than stepping toward them), and the geometry record prints
# ten decimals of an Angstrom. Measured on `path_mep_freeze` and
# `path_neb_freeze` (five and twenty frames, HF/STO-3G): the deviation was
# exactly 0. So the tolerance only has to absorb the print, and a frozen atom
# that moved by one path step (0.15 Bohr = 0.079 Angstrom) is 7.9e7 times it.
FIX_TOL = 1e-9

# A composite leg has to reproduce the surface its single-layer counterpart
# evaluates on the geometry both declare. Measured on the same round: the three
# pairs agree to the print of `path_profile.txt` (ten decimals of a Hartree), so
# this is stated as a reading of that print.
PATH_PAIR_TOL = 1e-9


def _leg_conditions(run_dir):
    """The strict conditions a leg declares, in the framework's units.

    Read from the flat config the plan laid down beside the run - the same file
    the leg was launched from, and the same source the three scope tests of
    section C2 use. A leg that declares none answers empty, which leaves it out
    of section F rather than reported as a leg with no hold to read.

    Returns
    -------
    list of (str, list of int, float)
        Each condition's kind, the 0-based atom indices of its own member group,
        and its tolerance. A config with several member groups of one kind
        returns one entry per group, so a second held bond cannot hide behind
        the first.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as handler:
                config = json.load(handler)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        block = config.get('constraints')
        if not isinstance(block, dict):
            continue
        items = block.get('constraints')
        if not isinstance(items, list):
            continue
        found = []
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get('kind') or '')
            if kind not in ('bond', 'angle', 'dihedral'):
                continue
            # The member atoms live under `bonds` for every kind
            # (`schema/config.py::ConstraintInput`): a held angle and a held
            # dihedral are "a list of atom groups by nature" too, which is why
            # the model carries no `angles`/`dihedrals` field. Reading a
            # kind-named key here answered empty for the two angle legs of the
            # round and audited six of ten legs while reporting no problem - the
            # held-angle and held-dihedral walks were never read at all.
            groups = item.get('bonds') or []
            try:
                tolerance = float(item.get('tolerance') or 1.0e-8)
            except (TypeError, ValueError):
                tolerance = 1.0e-8
            for group in groups:
                try:
                    indices = [int(index) - 1 for index in group]
                except (TypeError, ValueError):
                    continue
                if indices:
                    found.append((kind, indices, tolerance))
        if found:
            return found
    return []


def _xyz_frames(path):
    """Every frame of an XYZ record, as `(comment, coordinates)`.

    `utilities/io.py::read_xyz_frame` reads one frame and refuses a record that
    carries more, because a structure file it is handed has to be one molecule;
    a path's geometry record is a trajectory by construction, so this walks the
    frames instead. A record that ends mid-frame is reported by its shorter
    length rather than patched here.
    """
    try:
        with open(path, 'r', encoding='utf-8') as handler:
            lines = [line for line in handler.read().splitlines() if line.strip()]
    except OSError:
        return []
    frames = []
    cursor = 0
    while cursor + 1 < len(lines):
        try:
            count = int(lines[cursor].split()[0])
        except (IndexError, ValueError):
            break
        comment = lines[cursor + 1]
        body = lines[cursor + 2:cursor + 2 + count]
        if len(body) != count:
            break
        coordinates = []
        for line in body:
            fields = line.split()
            if len(fields) < 4:
                coordinates = []
                break
            try:
                coordinates.append([float(value) for value in fields[1:4]])
            except ValueError:
                coordinates = []
                break
        if not coordinates:
            break
        frames.append((comment, coordinates))
        cursor += 2 + count
    return frames


def _held_reading(kind, coordinates, indices):
    """The held internal coordinate of one frame, in the solve's own units.

    A length is returned in Bohr and an angle or a dihedral in radian: those are
    the units the declared tolerance is stated in (`the numbers below are
    internal (bohr, radian, hartree)`, the run's own log), so the conversion
    happens here rather than being folded into the tolerance.
    """
    points = [coordinates[index] for index in indices]

    def sub(one, other):
        return [one[axis] - other[axis] for axis in range(3)]

    def dot(one, other):
        return sum(one[axis] * other[axis] for axis in range(3))

    def norm(one):
        return math.sqrt(dot(one, one))

    if kind == 'bond':
        return norm(sub(points[0], points[1])) * ANGSTROM_TO_BOHR
    if kind == 'angle':
        first = sub(points[0], points[1])
        second = sub(points[2], points[1])
        cosine = dot(first, second) / (norm(first) * norm(second))
        return math.acos(max(-1.0, min(1.0, cosine)))
    first = sub(points[0], points[1])
    second = sub(points[1], points[2])
    third = sub(points[2], points[3])

    def cross(one, other):
        return [one[1] * other[2] - one[2] * other[1],
                one[2] * other[0] - one[0] * other[2],
                one[0] * other[1] - one[1] * other[0]]

    first_normal = cross(first, second)
    second_normal = cross(second, third)
    axis = [value / norm(second) for value in second]
    reference = cross(first_normal, axis)
    return math.atan2(dot(reference, second_normal),
                      dot(first_normal, second_normal))


def _profile_rows(path):
    """The `path_profile.txt` table, one dict per row keyed by its header."""
    try:
        with open(path, 'r', encoding='utf-8') as handler:
            lines = [line.split() for line in handler.read().splitlines()
                     if line.strip()]
    except OSError:
        return []
    if len(lines) < 2:
        return []
    header = lines[0]
    rows = []
    for fields in lines[1:]:
        if len(fields) != len(header):
            continue
        row = {}
        for key, value in zip(header, fields):
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
        rows.append(row)
    return rows


def _leg_system(run_dir):
    """A leg's atoms and starting geometry, from its own flat config.

    Used to decide whether two legs are two views of **one** system. A name is
    not enough: `path_cell_mep_multiscale` and `path_cell_mep` differ by the
    suffix and are different systems (a 12-atom composite against a 6-atom
    single layer), and comparing their energies would be comparing two
    molecules.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as handler:
                config = json.load(handler)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        atoms = config.get('atoms')
        coordinates = config.get('coordinates')
        if not isinstance(atoms, list) or not isinstance(coordinates, list):
            continue
        try:
            shape = tuple(tuple(round(float(value), 6) for value in row)
                          for row in coordinates)
        except (TypeError, ValueError):
            continue
        return tuple(str(name) for name in atoms), shape
    return None


def path_pair_legs(work_dir):
    """Every composite path leg with a single-layer counterpart on disk.

    The counterpart is the same case without `_multiscale`, and it has to be the
    same system - the same atoms and the same starting geometry - or the two legs
    are different molecules and their energies are not two readings of one
    surface. Measured on round `tmp_20261022_134728`: `path_cell_mep_multiscale`
    shares only its suffix with `path_cell_mep` (-77.32 against -0.63 Hartree),
    so the pairing skips it; its two carriers are still compared by section E.
    """
    examples = os.path.join(work_dir, 'examples')
    pairs = []
    for run_dir in sorted(glob.glob(os.path.join(examples, 'path_*', '*'))):
        if not os.path.isdir(run_dir):
            continue
        if not _leg_declares_a_region(run_dir):
            continue
        case = os.path.basename(os.path.dirname(run_dir))
        if not case.endswith('_multiscale'):
            continue
        partner = case[:-len('_multiscale')]
        partner_dir = os.path.join(examples, partner,
                                   os.path.basename(run_dir))
        if not os.path.isdir(partner_dir):
            continue
        here = _leg_system(run_dir)
        there = _leg_system(partner_dir)
        if here is None or there is None or here != there:
            continue
        pairs.append((case, os.path.basename(run_dir), run_dir,
                      partner_dir))
    return pairs


def _leg_fixes(run_dir):
    """The atoms a leg freezes, 0-based, from its own flat config.

    `system_constraint: fix_atoms` lands in the leg's config as
    `constraints.fixes[*].atoms`, written 1-based (the same source the held
    conditions are read from). A leg that declares none answers empty and stays
    out of the frozen clause rather than being reported.
    """
    for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
        try:
            with open(path, 'r', encoding='utf-8') as handler:
                config = json.load(handler)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        block = config.get('constraints')
        if not isinstance(block, dict):
            continue
        fixes = block.get('fixes')
        if not isinstance(fixes, list) or not fixes:
            continue
        found = []
        for item in fixes:
            atoms = item.get('atoms') if isinstance(item, dict) else None
            if not isinstance(atoms, list):
                continue
            try:
                found.extend(int(atom) - 1 for atom in atoms)
            except (TypeError, ValueError):
                continue
        if found:
            return found
    return []


def _is_band_endpoint(comment):
    """Whether a frame's comment names a band endpoint.

    A band writes `Round N Image I Role <role>` and the two endpoints are its
    boundary conditions, never landed by `pathbase.py::_land_images`; a walker
    writes `Point N Direction ...` and has no role. So a comment with no role is
    a walker point and counts.
    """
    match = re.search(r'Role\s+(\w+)', str(comment))
    return bool(match) and match.group(1) in ('reactant', 'product')


def audit_path(work_dir):
    """Section F: the path family's own claims, re-derived from its files.

    The round's verdict reads none of these: the runner's post-check asks only
    whether a variable-cell path record carries the cell columns and whether its
    volume moved. Two claims the path examples make are recomputable from what
    the walk left:

    - a leg that declares a strict condition held it on every frame it
      committed. `path_geom.xyz` carries every accepted point (a walker) or every
      image of every round (a band), so the held internal coordinate is
      recomputed frame by frame and the worst deviation from the first frame is
      compared against the tolerance that leg's own config declares. This is the
      walker's held bond and the band's frozen components in one reading.
    - a leg that freezes atoms (`fix_atoms`) kept them where the run bound them.
      The atoms are read from the leg's own config and their coordinates are
      checked over the walkable frames: every frame of a walker, the interior
      images of a band - the two endpoints are the band's boundary conditions
      and `_land_images` never lands them, so a frozen atom that differs there
      is the input's own choice rather than a broken ride.
    - a composite leg reproduces the surface of its single-layer counterpart.
      Both declare the same starting geometry (a walker) or the same endpoint
      files (a band), so the energies their two profile tables record for it have
      to agree. A composite that assembles its layers wrongly shows up here and
      nowhere else in the round.

    What is deliberately not read: the reason a band's frozen atom may sit
    elsewhere in the endpoints (they are inputs), and the position solve's
    multipliers, which reach no file.
    """
    problems = []
    counts = {'condition_legs': 0, 'held_rows': 0, 'held_broken': 0,
              'unreadable_geometry': 0, 'pairs': 0, 'pair_rows': 0,
              'pair_disagreements': 0, 'pair_missing_profile': 0,
              'fix_legs': 0, 'fix_atoms': 0, 'fix_frames': 0,
              'fix_broken': 0, 'irc_legs': 0, 'irc_origin_broken': 0,
              'band_legs': 0, 'band_jumps': 0}
    examples = os.path.join(work_dir, 'examples')
    if not os.path.isdir(examples):
        return [('NO EXAMPLES: %s is missing' % examples)], counts

    for run_dir in sorted(glob.glob(os.path.join(examples, 'path_*', '*'))):
        if not os.path.isdir(run_dir):
            continue
        conditions = _leg_conditions(run_dir)
        if not conditions:
            continue
        counts['condition_legs'] += 1
        name = '%s/%s' % (os.path.basename(os.path.dirname(run_dir)),
                          os.path.basename(run_dir))
        frames = _xyz_frames(os.path.join(run_dir, 'path_geom.xyz'))
        if len(frames) < 2:
            counts['unreadable_geometry'] += 1
            problems.append('F NO GEOMETRY: %s carries %d frame(s), so a hold '
                            'cannot be read over the walk' % (name, len(frames)))
            continue
        for kind, indices, tolerance in conditions:
            values = []
            for _, coordinates in frames:
                if max(indices) >= len(coordinates):
                    values = []
                    break
                values.append(_held_reading(kind, coordinates, indices))
            if not values:
                counts['unreadable_geometry'] += 1
                problems.append('F NO ATOMS: %s declares a held %s on %s with '
                                '%d atom(s) per frame'
                                % (name, kind, [i + 1 for i in indices],
                                   len(frames[0][1])))
                continue
            counts['held_rows'] += len(values)
            worst = max(abs(value - values[0]) for value in values)
            limit = PATH_HELD_SCALE * tolerance
            if worst > limit:
                counts['held_broken'] += 1
                problems.append(
                    'F HOLD BROKEN: %s: the %s %s moved by %.3e from its first '
                    'frame over %d frame(s) (first %.10f, limit %.1e)'
                    % (name, kind, [i + 1 for i in indices], worst, len(values),
                       values[0], limit))

    for run_dir in sorted(glob.glob(os.path.join(examples, 'path_*', '*'))):
        if not os.path.isdir(run_dir):
            continue
        fixed = _leg_fixes(run_dir)
        if not fixed:
            continue
        counts['fix_legs'] += 1
        counts['fix_atoms'] += len(fixed)
        name = '%s/%s' % (os.path.basename(os.path.dirname(run_dir)),
                          os.path.basename(run_dir))
        frames = [(comment, coordinates) for comment, coordinates
                  in _xyz_frames(os.path.join(run_dir, 'path_geom.xyz'))
                  if not _is_band_endpoint(comment)]
        if len(frames) < 2:
            counts['unreadable_geometry'] += 1
            problems.append('F NO FROZEN FRAMES: %s declares %d frozen atom(s) '
                            'and carries %d walkable frame(s)'
                            % (name, len(fixed), len(frames)))
            continue
        counts['fix_frames'] += len(frames)
        anchor = frames[0][1]
        for index in fixed:
            if max(index, 0) >= len(anchor):
                counts['unreadable_geometry'] += 1
                problems.append('F NO FROZEN ATOM: %s freezes atom %d and its '
                                'frames carry %d atom(s)'
                                % (name, index + 1, len(anchor)))
                continue
            worst = max(float(max(abs(coordinate[index][axis]
                                      - anchor[index][axis])
                                  for axis in range(3)))
                        for _comment, coordinate in frames)
            if worst > FIX_TOL:
                counts['fix_broken'] += 1
                problems.append(
                    'F FROZEN ATOM MOVED: %s: atom %d moved by %.3e Angstrom '
                    'from its anchor over %d walkable frame(s) (anchor %s)'
                    % (name, index + 1, worst, len(frames),
                       ['%.10f' % value for value in anchor[index]]))

    for case, mode, run_dir, partner_dir in path_pair_legs(work_dir):
        counts['pairs'] += 1
        left = _profile_rows(os.path.join(run_dir, 'path_profile.txt'))
        right = _profile_rows(os.path.join(partner_dir, 'path_profile.txt'))
        if not left or not right:
            counts['pair_missing_profile'] += 1
            problems.append('F NO PROFILE: %s/%s has %d row(s) and its '
                            'single-layer counterpart %d'
                            % (case, mode, len(left), len(right)))
            continue
        method = ''
        for path in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
            try:
                with open(path, 'r', encoding='utf-8') as handler:
                    method = str(json.load(handler).get('opt_method') or '')
            except (OSError, ValueError):
                continue
            break
        readings = [('first', left[0], right[0])]
        if method == 'neb' and len(left) > 1 and len(right) > 1:
            readings.append(('last', left[-1], right[-1]))
        for label, one, other in readings:
            try:
                energy_one = float(one['energy'])
                energy_other = float(other['energy'])
            except (KeyError, TypeError, ValueError):
                continue
            counts['pair_rows'] += 1
            if abs(energy_one - energy_other) > PATH_PAIR_TOL:
                counts['pair_disagreements'] += 1
                problems.append(
                    'F COMPOSITE DIVERGENCE: %s/%s: the %s point is %.10f on '
                    'the composite and %.10f on %s'
                    % (case, mode, label, energy_one, energy_other,
                       case[:-len('_multiscale')]))
    # A periodic band's committed frames are one chain in one frame: no
    # consecutive pair inside a round may differ by more than the nearest image
    # allows, or an atom was carried through the wall. The interpolation now
    # brings the product into the reactant's image (`geodesic.consistent_pair`),
    # and this reads the result off the record.
    for run_dir in sorted(glob.glob(os.path.join(examples, 'path_*', '*'))):
        if not os.path.isdir(run_dir):
            continue
        geometry = os.path.join(run_dir, 'path_geom.xyz')
        frames = _xyz_frames(geometry)
        if not any('Round ' in str(comment) for comment, _c in frames):
            continue
        boxes = read_md_lattice(geometry)
        if not boxes or len(boxes) != len(frames):
            continue
        counts['band_legs'] += 1
        previous = None
        previous_round = None
        for (comment, coordinates), box in zip(frames, boxes):
            round_at = re.search(r'Round\s+(\d+)', str(comment))
            round_at = (int(round_at.group(1)) if round_at is not None
                        else None)
            if previous is not None and round_at == previous_round:
                delta = [[coordinates[atom][axis] - previous[atom][axis]
                          for axis in range(3)]
                         for atom in range(len(coordinates))]
                cell = [box[0:3], box[3:6], box[6:9]]
                reduced = min_image_displacement(delta, cell)
                worst = max(abs(float(delta[atom][axis])
                                - float(reduced[atom][axis]))
                            for atom in range(len(delta))
                            for axis in range(3))
                if worst > 1e-6:
                    counts['band_jumps'] += 1
                    problems.append(
                        'F BAND FRAME JUMP: %s/%s: round %s carries two '
                        'adjacent frames %.6f Angstrom apart in one atom, which '
                        'is not the nearest image - the chain crosses a cell '
                        'face between them'
                        % (os.path.basename(os.path.dirname(run_dir)),
                           os.path.basename(run_dir), round_at, worst))
            previous = coordinates
            previous_round = round_at

    # The two branches of an IRC start from one saddle: a walker is driven on
    # the state in place, so a branch that inherits the previous one's endpoint
    # starts somewhere else. Measured on `path_irc` before the reset: the two
    # `saddle` rows read -91.5616686633 and -91.6662865772, and
    # `path_cell_irc`'s forward branch was an empty walk from the backward one's
    # last box.
    for run_dir in sorted(glob.glob(os.path.join(examples, 'path_*', '*'))):
        if not os.path.isdir(run_dir):
            continue
        rows = _profile_rows(os.path.join(run_dir, 'path_profile.txt'))
        saddles = [row for row in rows if str(row.get('role')) == 'saddle']
        directions = {str(row.get('direction')) for row in saddles}
        if len(saddles) < 2 or len(directions) < 2:
            continue
        counts['irc_legs'] += 1
        energies = []
        for row in saddles:
            try:
                energies.append(float(row['energy']))
            except (KeyError, TypeError, ValueError):
                continue
        if len(energies) < 2:
            continue
        if max(energies) - min(energies) > PATH_PAIR_TOL:
            counts['irc_origin_broken'] += 1
            problems.append(
                'F IRC BRANCHES: %s/%s: the two saddle rows read %.10f and '
                '%.10f, so one branch did not start from the saddle the other '
                'did'
                % (os.path.basename(os.path.dirname(run_dir)),
                   os.path.basename(run_dir), min(energies), max(energies)))

    return problems, counts


# The three carrier legs a skill round writes for one case, in the order the
# planner emits them.
MOLCAS_LEGS = ('skill_agent', 'pyrai3md_agent', 'pyrai3md_input')
MOLCAS_ARRAYS = ('energy', 'gradient', 'nac', 'soc')

# Cross-leg floor. Every number a leg reports is read out of that leg's own
# engine log, whose energy lines print eight decimals, so two legs that ran
# the same computation can only ever agree to that printed precision and no
# further. Measured on round `tmp_20261027_173607` over the 24-case property
# matrix: all four arrays were equal to exactly 0.0 across the three legs. A
# leg whose root energy was edited by 1.9e-05 in its own log was named, and
# the other 23 cases did not move, so the floor separates a planted leg from
# a printed one.
MOLCAS_LEG_TOL = 1e-8



def _molcas_cases(work_dir):
    """The molcas matrix cases a round reports as run and passed."""
    path = os.path.join(work_dir, 'results.json')
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as handler:
        entries = json.load(handler).get('skill_results', [])
    names = []
    for entry in entries:
        name = entry.get('name', '')
        if (name.startswith('molcas_skill_') and name.endswith('_skill')
                and entry.get('mode') == 'skill_agent'
                and entry.get('status') == 'pass'):
            names.append(name[len('molcas_skill_'):-len('_skill')])
    return sorted(set(names))


def _molcas_leg_dir(work_dir, case, leg):
    """One carrier leg's directory: per-case group first, flat legacy next.

    Grouped rounds place each leg under its case group (`gNN`), so the
    config is `<skill>/<group>/<leg>/<case>.json`; earlier rounds kept it
    at `<skill>/<leg>/<case>.json`. Both are read, so an old work
    directory still audits.
    """
    matches = sorted(glob.glob(os.path.join(
        work_dir, 'molcas_skill', '*', leg, '%s.json' % case)))
    if matches:
        return os.path.dirname(matches[0])
    legacy = os.path.join(work_dir, 'molcas_skill', leg)
    if os.path.isfile(os.path.join(legacy, '%s.json' % case)):
        return legacy
    return None


def _molcas_parse_leg(work_dir, case, leg):
    """Parse one carrier leg's own engine log with the skill's parser.

    The config the leg ran on is read back from that leg's directory, so the
    parse is driven by the leg's own profile and state list rather than by
    what the audit assumes. The backend is never constructed: only its schema
    pipeline and workflow compiler are called, which keeps the environment
    check (and the engine it looks for) out of an audit that reads files.
    """
    from pyrai3md.skills.molcas_skill.schema.adapter import build_config
    from pyrai3md.skills.molcas_skill.schema.method import (
        has_transition_request)
    from pyrai3md.skills.molcas_skill.schema.normalize import normalize
    from pyrai3md.skills.molcas_skill.schema.validate import validate
    from pyrai3md.skills.molcas_skill.service.parser import (format_results,
                                                             parse_output)
    from pyrai3md.skills.molcas_skill.service.workflow import compile_workflow

    leg_dir = _molcas_leg_dir(work_dir, case, leg)
    if leg_dir is None:
        return None, ('config missing under %s'
                      % os.path.join(work_dir, 'molcas_skill', '*', leg))
    cfg_path = os.path.join(leg_dir, '%s.json' % case)
    calcdir = os.path.join(leg_dir, 'tmp', 'tmp_%s_test' % case)
    if not os.path.isfile(cfg_path):
        return None, 'config missing: %s' % cfg_path
    if not os.path.isdir(calcdir):
        return None, 'calculation directory missing: %s' % calcdir
    with open(cfg_path, 'r', encoding='utf-8') as handler:
        raw = json.load(handler)
    # The audit reads logs; it never starts the engine. The schema rejects a
    # `molcas_path` that is not a directory on this host, and the round was
    # written in the test environment, so the two environment fields are pointed at
    # directories that exist here. Nothing else in the parse consults them.
    raw['molcas_path'] = os.path.dirname(cfg_path)
    raw['scratch_path'] = tempfile.gettempdir()
    config = validate(normalize(build_config(raw)))
    workflow = compile_workflow(
        config.calculation.profile,
        transition=has_transition_request(config.calculation.jobtype),
        config=config)
    try:
        parsed = parse_output(config, calcdir, workflow)
    except (RuntimeError, OSError, ValueError) as error:
        return None, 'parse failed: %s' % error
    formatted = format_results(parsed, config)
    # The pairs the run asked for live on the normalized config, not in the
    # log, and the contract below counts couplings against them.
    formatted['target_nac_pair'] = [list(pair) for pair in
                                    (config.target_nac_pair or [])]
    formatted['target_soc_pair'] = [list(pair) for pair in
                                    (config.target_soc_pair or [])]
    return formatted, ''


def _molcas_max_diff(left, right):
    """Worst absolute difference between two nested result lists.

    A shape mismatch is reported as `None` rather than a number: a missing
    state and a different state are different defects, and a section that
    flattened both into one float could not name either.
    """
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return None
        worst = 0.0
        for one, other in zip(left, right):
            value = _molcas_max_diff(one, other)
            if value is None:
                return None
            worst = max(worst, value)
        return worst
    return None


def _molcas_leg_diffs(arms):
    """Per-array worst difference across the legs of one case."""
    reference = arms[MOLCAS_LEGS[0]]
    diffs = {}
    for key in MOLCAS_ARRAYS:
        value = reference.get(key)
        if value is None:
            continue
        worst = 0.0
        for leg in MOLCAS_LEGS[1:]:
            other = arms[leg].get(key)
            if other is None:
                continue
            delta = _molcas_max_diff(value, other)
            if delta is None:
                diffs['%s_shape' % key] = float('inf')
                worst = None
                break
            worst = delta if worst is None else max(worst, delta)
        if worst is not None:
            diffs[key] = worst
    return diffs


def _molcas_leg_verdict(case, diffs):
    """Problem lines for one case's cross-leg differences."""
    problems = []
    for key, value in diffs.items():
        limit = 0.0 if key.endswith('_shape') else MOLCAS_LEG_TOL
        if value > limit:
            problems.append(
                'G LEG DIVERGENCE: %s: %s differs across the carrier legs by '
                '%.3e (floor %g)' % (case, key, value, limit))
    return problems


def _molcas_shapes(value):
    shapes = []
    while isinstance(value, (list, tuple)):
        shapes.append(len(value))
        value = value[0] if value else None
    return shapes


def _molcas_contract(case, raw, arm, expected):
    """Lines a molcas result must satisfy beyond its numbers.

    A round that returns the right count of zeros passes every tolerance in
    section A, so the shape, the ESPF pass count and the source of the
    reported energy are asserted here instead: they are what names a parse
    that answered a different question than the case asked.
    """
    problems = []
    if not isinstance(arm, dict):
        return problems
    nstate = int(raw.get('nstate') or 1)
    natom = len(raw.get('atoms') or [])
    jobtype = str(raw.get('jobtype') or '')
    embedding = str(raw.get('embedding') or 'none')
    method = str(raw.get('method') or '')
    # The requested pairs live on the config and the parsed leg copies them
    # into its record; read whichever side carries them.
    nac_pairs = (arm.get('target_nac_pair')
                 or raw.get('target_nac_pair') or [])
    soc_pairs = (arm.get('target_soc_pair')
                 or raw.get('target_soc_pair') or [])
    if expected is None:
        return ['G EXPECTATION MISSING: %s: the test YAML was not found, so '
                'no property was asserted for it' % case]
    if 'gradient' in expected:
        want = [nstate, natom, 3]
        got = _molcas_shapes(arm.get('gradient'))
        if got != want:
            problems.append('G GRADIENT SHAPE: %s: %s instead of %s'
                            % (case, got, want))
    if 'nac' in expected:
        pairs = arm.get('target_nac_pair') or []
        want = [len(pairs), natom, 3]
        got = _molcas_shapes(arm.get('nac'))
        if got != want:
            problems.append('G NAC SHAPE: %s: %s instead of %s'
                            % (case, got, want))
        if arm.get('nac_csf_included'):
            problems.append('G NAC CSF TERM: %s: the coupling carries the '
                            'CSF derivative' % case)
    if 'soc' in expected:
        pairs = arm.get('target_soc_pair') or []
        want = [len(pairs)]
        got = _molcas_shapes(arm.get('soc'))
        if got != want:
            problems.append('G SOC SHAPE: %s: %s instead of %s'
                            % (case, got, want))
    # ESPF is on when the run names a file (whose first line declares the
    # representation) or asks for one through `embedding`; the compile path
    # asks the same question, so the two cannot disagree about the pass count.
    read_charge = str(raw.get('read_charge', 'none') or 'none').strip().lower()
    esfp = read_charge not in ('none', 'false', '0', '') or \
        embedding in ('charge', 'field')
    two_pass = ('nac' in jobtype.split() and esfp)
    if bool(arm.get('two_pass')) != two_pass:
        problems.append('G PASS COUNT: %s: two_pass=%s but jobtype %s with '
                        'embedding %s asks for %s'
                        % (case, arm.get('two_pass'), jobtype or 'energy',
                           embedding, two_pass))
    if two_pass and arm.get('nac_approximation') != 'two_pass_no_esfp':
        problems.append('G NAC APPROXIMATION: %s: %s'
                        % (case, arm.get('nac_approximation')))
    source = str(arm.get('energy_source') or '')
    if method == 'caspt2' and not source.startswith('caspt2_'):
        problems.append('G ENERGY SOURCE: %s: a CASPT2 run reported from %s'
                        % (case, source or 'nothing'))
    if method == 'casscf' and source != 'casscf':
        problems.append('G ENERGY SOURCE: %s: a CASSCF run reported from %s'
                        % (case, source or 'nothing'))
    return problems


def _field_split_inputs(work_dir):
    """Every cold qmmm_field directory: (deck, groups file)."""
    for groups in sorted(glob.glob(os.path.join(
            work_dir, '**', '*.field_A.groups'), recursive=True)):
        calcdir = os.path.dirname(groups)
        decks = sorted(glob.glob(os.path.join(calcdir, '*.in')))
        for deck in decks:
            with open(deck, 'r', encoding='utf-8', errors='ignore') as handle:
                text = handle.read()
            if '.probe_A.pe' in text and 'read_data' in text:
                yield deck, groups, text


def _field_data(deck, text):
    """(charges, positions, box lengths, qm ids) from the deck's data file."""
    match = re.search(r'read_data\s+"?([^"\s]+)"?', text)
    if not match:
        return None
    path = os.path.join(os.path.dirname(deck), match.group(1))
    if not os.path.isfile(path):
        return None
    charges, positions, box = [], [], {}
    section = ''
    with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            head = line.split('#', 1)[0].strip()
            if not head:
                continue
            tokens = head.split()
            if len(tokens) == 4 and tokens[2:] in (
                    ['xlo', 'xhi'], ['ylo', 'yhi'], ['zlo', 'zhi']):
                box[tokens[2][0]] = float(tokens[1]) - float(tokens[0])
                continue
            if head in ('Atoms', 'Masses', 'Bonds', 'Angles',
                        'Dihedrals', 'Impropers'):
                section = head
                continue
            if section != 'Atoms' or len(tokens) < 7:
                continue
            try:
                charges.append(float(tokens[3]))
                positions.append(tuple(float(v) for v in tokens[4:7]))
            except ValueError:
                continue
    if len(box) != 3 or not charges:
        return None
    ids = [int(v) for v in re.search(
        r'group\s+QMMM_QM\s+id\s+([\d\s]+)', text).group(1).split()]
    return charges, positions, box, ids


def _ewald_cross(charges, positions, qm_ids, lengths,
                 alpha=0.32, kmax=8.0, images=3):
    """Converged QM-MM Ewald cross energy, in LAMMPS real units.

    The A probe sets the whole QM group to +1, so the QM column is the
    probe charge and the MM column keeps the topology charges.
    """
    qscale = 332.06371
    qm = [index - 1 for index in qm_ids]
    qm_set = set(qm)
    probe = list(charges)
    for index in qm:
        probe[index] = 1.0
    volume = lengths[0] * lengths[1] * lengths[2]
    real = 0.0
    for i in qm:
        for j in range(len(probe)):
            if j in qm_set:
                continue
            for nx in range(-images, images + 1):
                for ny in range(-images, images + 1):
                    for nz in range(-images, images + 1):
                        d = [positions[j][axis] + (nx, ny, nz)[axis]
                             * lengths[axis] - positions[i][axis]
                             for axis in range(3)]
                        r = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
                        if r < 1e-8:
                            continue
                        real += (probe[i] * probe[j]
                                 * math.erfc(alpha * r) / r)
    nmax = int(kmax * max(lengths) / (2 * math.pi)) + 1
    reciprocal = 0.0
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                if nx == ny == nz == 0:
                    continue
                k = (2 * math.pi / lengths[0] * nx,
                     2 * math.pi / lengths[1] * ny,
                     2 * math.pi / lengths[2] * nz)
                k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
                if k2 > kmax * kmax:
                    continue

                def structure(indices):
                    total = 0.0
                    for index in indices:
                        phase = (k[0] * positions[index][0]
                                 + k[1] * positions[index][1]
                                 + k[2] * positions[index][2])
                        total += probe[index] * complex(math.cos(phase),
                                                        math.sin(phase))
                    return total
                # exp(i k r_QM) * conj(exp(i k r_MM)): the cross term
                # couples the two densities at the same wavevector.
                cross = structure(qm) * structure(
                    [j for j in range(len(probe)) if j not in qm_set]
                ).conjugate()
                reciprocal += (4 * math.pi / volume
                               * math.exp(-k2 / (4 * alpha ** 2)) / k2
                               * cross.real)
    return (real + reciprocal) * qscale


def _dump_column(path, column=1):
    values = {}
    header = False
    for line in open(path, 'r', encoding='utf-8', errors='ignore'):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 'ITEM:' and 'ATOMS' in line:
            header = True
            continue
        if not header or len(tokens) <= column:
            continue
        try:
            values[int(tokens[0])] = float(tokens[column])
        except ValueError:
            continue
    return values


def _direct_cross(charges, positions, qm_ids):
    """QM-MM Coulomb cross energy of a non-periodic cluster, real units.

    The A probe sets the whole QM group to +1; the MM column keeps the
    topology charges. No images and no reciprocal sum: the deck runs
    real-space Coulomb only.
    """
    qscale = 332.06371
    qm = [index - 1 for index in qm_ids]
    qm_set = set(qm)
    total = 0.0
    for i in qm:
        for j in range(len(charges)):
            if j in qm_set:
                continue
            d = [positions[j][axis] - positions[i][axis] for axis in range(3)]
            r = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if r < 1e-8:
                continue
            total += 1.0 * charges[j] / r
    return total * qscale


def audit_field_split(work_dir):
    """Section H: the periodic field probe's A/B split and its V.

    The probe computes V_i = 2(peA_i - peB_i) and checks it against the
    group/group Q-M energy. That identity only holds when both runs share
    one Ewald split: LAMMPS re-derives alpha and the FFT mesh from qsqsum
    at every init, and zeroing the MM charges changes qsqsum. This section
    re-derives the cross energy with a converged Ewald sum and compares
    both numbers against it, so a deck that lost the pinned
    ``kspace_modify gewald/mesh`` line is named rather than passing on a
    guard it never exercised.
    """
    problems = []
    counts = {'decks': 0, 'pinned': 0, 'checked': 0, 'skipped': 0}
    for deck, groups, text in _field_split_inputs(work_dir):
        counts['decks'] += 1
        pinned = 'kspace_modify' in text and 'gewald' in text
        counts['pinned'] += 1 if pinned else 0
        data = _field_data(deck, text)
        if data is None:
            counts['skipped'] += 1
            continue
        charges, positions, box, qm_ids = data
        base = deck[:-3]
        pe_a = _dump_column(base + '.probe_A.pe')
        pe_b = _dump_column(base + '.probe_B.pe')
        if not pe_a or not pe_b:
            counts['skipped'] += 1
            continue
        try:
            with open(groups, 'r', encoding='utf-8', errors='ignore') as handle:
                tokens = handle.read().split()
            u_gg = float(tokens[2])
        except (OSError, IndexError, ValueError):
            counts['skipped'] += 1
            continue
        sum_v = sum(2 * (pe_a[i] - pe_b[i]) for i in qm_ids)
        periodic_deck = ('kspace_style' in text
                         and 'kspace yes' in text)
        if periodic_deck:
            reference = _ewald_cross(charges, positions, qm_ids,
                                     (box['x'], box['y'], box['z']))
        else:
            reference = _direct_cross(charges, positions, qm_ids)
        counts['checked'] += 1
        scale = max(1.0, abs(sum_v), abs(u_gg))
        # The production guard (qmmm_output.assemble_qmmm_field) asks the
        # same question at 1e-4*scale; the unpinned A/B pair measured
        # 1.3e-1 on this fixture, 34x above this line.
        if abs(sum_v - u_gg) > 1e-4 * scale:
            problems.append(
                'H A/B SPLIT: %s: sum V %.9f vs group/group %.9f '
                '(residual %.3e); the A and B probes ran with different '
                'Ewald splits or the dumps keep too few digits'
                % (os.path.basename(deck), sum_v, u_gg, sum_v - u_gg))
        if abs(sum_v - reference) > 1e-3 * max(1.0, abs(reference)):
            problems.append(
                'H REFERENCE: %s: sum V %.9f vs %s %.9f (relative %.2e)'
                % (os.path.basename(deck), sum_v,
                   'converged Ewald' if periodic_deck else 'direct Coulomb',
                   reference,
                   abs(sum_v - reference) / max(1.0, abs(reference))))
    return problems, counts


def _vector_table(path):
    """Read one atom-indexed (x, y, z) table from a QM/MM audit file.

    Three shapes arrive here: a cold LAMMPS dump (an `ITEM: ATOMS` header and
    NO id column, whose box-bounds block before that header is not atom data),
    the resident dump (same header, with an id column), and the skill's own
    writers (comment lines then `id x y z`). A file without an id column is
    numbered by row because every writer sorts by id.

    Returns:
        (table, has_id) with table a {atom id: [x, y, z]} dict, or
        (None, False) when the file is missing or malformed.
    """
    table, dump, collecting, has_id, numbered = {}, False, False, False, 0
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None, False
    for line in lines:
        tokens = line.split()
        if not tokens or tokens[0].startswith('#'):
            continue
        if tokens[0] == 'ITEM:':
            dump = True
            # 'ITEM: NUMBER OF ATOMS' also carries the word.
            if len(tokens) > 1 and tokens[1] == 'ATOMS':
                columns = tokens[2:]
                has_id = bool(columns) and columns[0] == 'id'
                collecting = True
            continue
        if dump and not collecting:
            # Everything above ITEM: ATOMS is a box or a count, not a row.
            continue
        if len(tokens) != (4 if (has_id or not dump) else 3):
            return None, False
        try:
            if has_id or len(tokens) == 4:
                key = int(tokens[0])
                vector = [float(value) for value in tokens[1:4]]
                # The skill's own writers spell `id x y z` under a comment
                # header; they have no ITEM line to announce the id column.
                has_id = True
            else:
                numbered += 1
                key = numbered
                vector = [float(value) for value in tokens[0:3]]
        except ValueError:
            return None, False
        table[key] = vector
    return (table, has_id) if table else (None, False)


def audit_grad_identity(work_dir):
    """Section I: the four QM/MM force/gradient files agree with each other.

    The reduction that turns the engine's U_main force into `grad` is exact by
    construction, so no residual computed inside the writer can name a wrong
    QM-row index or a dropped term. This section reads the files back and
    checks, per atom:

        grad = -force + u_gas.force + qmmm_el.force   (QM rows)
        grad = -force                                 (every other row)

    The tolerance is set by the least precise file in the comparison: a cold
    `write_dump` of the forces keeps six significant digits and offers no way
    to widen it, while the resident path and the skill's own writers keep
    sixteen. A cold deck is recognised by the missing id column in its dump
    header, so the loose line is never applied to a high-precision file.
    """
    problems = []
    counts = {'grads': 0, 'checked': 0, 'skipped': 0}
    pattern = os.path.join(work_dir, '**', '*.grad')
    for path in sorted(glob.glob(pattern, recursive=True)):
        counts['grads'] += 1
        base = path[:-len('.grad')]
        grad, _ = _vector_table(path)
        force, force_ids = _vector_table(base + '.force')
        gas, _ = _vector_table(base + '.u_gas.force')
        el, _ = _vector_table(base + '.qmmm_el.force')
        if not grad or not force or not gas or not el:
            counts['skipped'] += 1
            continue
        # A table without an id column is numbered by row: every writer sorts
        # by id, so a cold dump's row number *is* its atom id and the four
        # files are still matchable. Requiring `has_id` here skipped every
        # cold-dump round (235 of 235 files in tmp_20261110_093321), which
        # left the section reporting clean because it checked nothing.
        counts['checked'] += 1
        # The least precise file sets the line: a cold `write_dump` keeps six
        # significant digits and no id column, the resident writer keeps both
        # sixteen and an id, and the skill's own writers are always sixteen.
        tolerance = 1e-8 if force_ids else 1e-5
        qm_ids = set(gas)
        if qm_ids != set(el):
            problems.append(
                'I GRAD ROWS: %s: u_gas.force ids %s and qmmm_el.force ids '
                '%s differ; the two QM-row auxiliaries must cover the same '
                'atoms' % (os.path.basename(base), sorted(qm_ids - set(el)),
                           sorted(set(el) - qm_ids)))
            continue
        if not qm_ids <= set(grad) or set(grad) != set(force):
            problems.append(
                'I GRAD ROWS: %s: grad ids %s do not cover the QM rows %s or '
                'do not match the force dump rows %s'
                % (os.path.basename(base), len(grad), len(qm_ids),
                   len(force)))
            continue
        for atom in sorted(grad):
            expected = [-value for value in force[atom]]
            if atom in qm_ids:
                for axis in range(3):
                    expected[axis] += gas[atom][axis] + el[atom][axis]
            limit = tolerance * max(1.0, max(abs(value)
                                             for value in force[atom]))
            for axis in range(3):
                residual = grad[atom][axis] - expected[axis]
                if abs(residual) > limit:
                    problems.append(
                        'I GRAD IDENTITY: %s atom %d axis %d: grad %.9e vs '
                        '%s %.9e (residual %.3e above %.3e)'
                        % (os.path.basename(base), atom, axis,
                           grad[atom][axis],
                           'reduced' if atom in qm_ids else '-force',
                           expected[axis], residual, limit))
    return problems, counts


def audit_molcas_legs(work_dir):
    """Section G: one molcas case, three carriers, one answer.

    A skill round runs each molcas case on three carriers - the skill's own
    entry, the framework agent, and the classic input - and each carrier keeps
    its own calculation directory since the per-leg scratch fix. Nothing in
    sections A to F compares them: a carrier that renders a keyword the engine
    ignores, or reads a block of the log the others do not, still passes,
    because each leg is judged only against its own record.

    G parses all three logs with the skill's own parser - the same code the
    backend returns from - and requires the four arrays to agree. It then
    asserts the shape and provenance contract of each property against the
    skill-agent leg, because a zero-filled array of the right length clears
    every tolerance above.

    Measured on round `tmp_20261027_173607` (deployment OpenMolcas 26.10, the
    24-case property matrix): 24 cases, 72 legs parsed, worst cross-leg
    difference 0.0e+00; 0 contract problems.
    """
    problems = []
    counts = {'cases': 0, 'legs parsed': 0, 'expected yamls': 0,
              'problems': 0, 'leg diff max': 0.0}
    leg_worst = 0.0
    leg_comparisons = 0
    cases = _molcas_cases(work_dir)
    if not cases:
        counts['not covered'] = ('no passed molcas_skill case in results.json')
        return [], counts

    from pyrai3md.skills.molcas_skill import __file__ as skill_init
    tests_dir = os.path.join(os.path.dirname(skill_init), 'tests')
    import yaml

    for case in cases:
        counts['cases'] += 1
        arms = {}
        for leg in MOLCAS_LEGS:
            parsed, reason = _molcas_parse_leg(work_dir, case, leg)
            if parsed is None:
                problems.append('G LEG UNREADABLE: %s %s: %s'
                                % (case, leg, reason))
                continue
            arms[leg] = parsed
            counts['legs parsed'] += 1
        if len(arms) == len(MOLCAS_LEGS):
            leg_comparisons += 1
            leg_diffs = _molcas_leg_diffs(arms)
            for key, value in leg_diffs.items():
                if not key.endswith('_shape'):
                    leg_worst = max(leg_worst, value)
            problems.extend(_molcas_leg_verdict(case, leg_diffs))

        raw_dir = _molcas_leg_dir(work_dir, case, 'skill_agent')
        if raw_dir is None:
            problems.append('G CONFIG UNREADABLE: %s: no skill_agent leg'
                            % case)
            continue
        raw_path = os.path.join(raw_dir, '%s.json' % case)
        with open(raw_path, 'r', encoding='utf-8') as handler:
            raw = json.load(handler)
        yaml_path = os.path.join(tests_dir, '%s.yaml' % case)
        expected = None
        if os.path.isfile(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as handler:
                expected = (yaml.safe_load(handler) or {}).get('expected') or {}
            counts['expected yamls'] += 1

        contract_arm = arms.get('skill_agent') or {}
        problems.extend(_molcas_contract(case, raw, contract_arm, expected))

    counts['leg diff max'] = ('%.1e' % leg_worst if leg_comparisons else
                              'no comparison')
    counts['leg comparisons'] = leg_comparisons
    counts['problems'] = len(problems)
    return problems, counts


NOT_COVERED = (
    'md.pop populations: no engine reports them; they are produced by the '
    'framework integrator',
    'per-step engine files: the calculation directory name is fixed per '
    'backend and overwritten every step, so only the last step survives',
    'qdm_skill and nn_pes_skill: keep_tmp is off, no per-step engine '
    'artifact remains',
    'aggregate rows of a layered run: the recorded Epot is the sum over '
    'centres, which no single engine file prints',
    'runs that report energy contributions: the constraint bias or multiscale '
    'term is folded into the recorded Epot, so it is deliberately not the raw '
    'engine number; such rows are counted as constraint_adjusted',
    'excited-state legs: a run driven on `opt_state_i` > 0 records the state '
    'energy (the SCF total plus the TDDFT excitation), while an engine file '
    'prints only the total; such legs are counted as state_scoped',
    'per-term dispatch: term_outcome.json is written only when the '
    'Calculator dispatches terms remotely; the count says whether a round '
    'exercised it',
    'skill final energies: K reads the five engines that write one number '
    'per run (ORCA, CP2K, xTB, OpenQP, LAMMPS); pyscf/qdm/nn_pes run in '
    'process and keep no engine file, molcas is section G\'s subject, and '
    'bagel prints no single final-energy line',
    'MOLCAS: the reader of section C answers the ground-state `Total SCF '
    'energy` when the log prints it, and otherwise the `RASSCF root number N '
    'Total energy:` lines of a CASSCF-only run (the shape md_molcas_fssh '
    'leaves). What stays unpinned there is the skill\'s dual singlet/triplet '
    'block layout, because no captured log of that layout is in this '
    'repository; section G reads those logs with the skill\'s own parser '
    'instead, so a round that carries the property matrix is covered by G, '
    'not by C',
)


def _selftest_flat_leg(directory, name, volumes, theory, pressure=0.0):
    """Plant one optimize leg with its cell volumes and its own config."""
    from pyrai3md.utilities.opt_formatter import status_header, status_line
    os.makedirs(directory)
    with open(os.path.join(directory, 'opt_status.txt'), 'w') as h:
        h.write(status_header(cell=True) + '\n')
        for number, volume in enumerate(volumes, start=1):
            h.write(status_line(number, (-8.0 - number, 0.0, 1.0e-3,
                                         1.0e-4, 1.0e-3, 1.0e-4,
                                         2.4e-2, 3.4e-2, 2.7e-3,
                                         6.3e-3, volume)) + '\n')
    with open(os.path.join(directory, '%s.json' % name), 'w') as h:
        json.dump({'optimize_cell': True, 'opt_method': 'minima',
                   'opt_pressure': pressure, 'theory': theory}, h)
    return directory


def _selftest_path_leg(directory, energies):
    os.makedirs(directory)
    with open(os.path.join(directory, 'demo.json'), 'w') as h:
        json.dump({'opt_method': 'neb', 'theory': ['xtb'],
                   'atoms': ['H', 'C', 'N'],
                   'coordinates': [[0.0, 0.0, 1.1],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, -1.1]],
                   'constraints': {'units': {'length': 'angstrom',
                                             'angle': 'degree'},
                                   'constraints': [{'kind': 'bond',
                                                    'bonds': [[2, 3]],
                                                    'tolerance': 1e-10,
                                                    'id': 'cn'}]}}, h)
    with open(os.path.join(directory, 'path_profile.txt'), 'w') as h:
        h.write(' index direction energy role\n')
        for index, (energy, role) in enumerate(energies, start=1):
            h.write('%6d forward %20.10f %s\n' % (index, energy, role))
    with open(os.path.join(directory, 'path_geom.xyz'), 'w') as h:
        for _ in range(2):
            h.write('3\n')
            h.write('Round 1 Image 0 Role reactant E = -91.0\n')
            h.write('H   0.0000000000  0.0000000000  1.1000000000\n')
            h.write('C   0.0000000000  0.0000000000  0.0000000000\n')
            h.write('N   0.0000000000  0.0000000000 -1.1000000000\n')


def _selftest_fix_geometry(fix_dir, moved_in='image'):
    roles = ['reactant', 'image', 'image', 'product']
    with open(os.path.join(fix_dir, 'path_geom.xyz'), 'w') as h:
        for index, role in enumerate(roles):
            h.write('3\n')
            h.write('Round 1 Image %d Role %s E = -1.0\n'
                    % (index, role))
            h.write('H   %.10f  0.0000000000  0.0000000000\n'
                    % (0.1 * index))
            # The same atom, moved only in the frame the caller names.
            moved = index == 2 and role == moved_in
            h.write('C   %s  0.0000000000  1.0000000000\n'
                    % ('0.5000000000' if moved else '0.0000000000'))
            h.write('N   0.0000000000  0.0000000000  2.0000000000\n')


def _selftest_path_cell(cell_leg, volumes):
    header = (' step direction energy shift max_d rms_d max_g rms_g '
              'radius residual cell_grms cell_gmax cell_drms cell_dmax '
              'volume')
    with open(os.path.join(cell_leg, 'path_status.txt'), 'w') as h:
        h.write(header + '\n')
        for index, volume in enumerate(volumes, start=1):
            h.write('%6d forward -91.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 '
                    '0.0 0.0 0.0 0.0 %.10f\n' % (index, volume))
    with open(os.path.join(cell_leg, 'path_geom.xyz'), 'w') as h:
        for volume in volumes:
            h.write('1\n')
            h.write('Point 1  Direction forward  E = -91.0 '
                    'Lattice="%.9f 0.000000000 0.000000000 '
                    '0.000000000 10.000000000 0.000000000 '
                    '0.000000000 0.000000000 %.9f"\n'
                    % (10.0, volume / 100.0))
            h.write('H   0.0000000000  0.0000000000  0.0000000000\n')


def _selftest_band_cell(cell_leg, second_frame):
    header = (' step energy shift max_d rms_d max_g rms_g image climb '
              'cell_grms cell_gmax cell_drms cell_dmax volume')
    with open(os.path.join(cell_leg, 'path_status.txt'), 'w') as h:
        h.write(header + '\n')
        for step, volume in ((1, 1000.0), (2, 1000.1)):
            h.write('%6d -91.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 '
                    '0.0 0.0 %.10f\n' % (step, volume))
    with open(os.path.join(cell_leg, 'path_geom.xyz'), 'w') as h:
        for round_at, volumes in ((1, (1000.0, second_frame)),
                                  (2, (1000.1, 1000.2))):
            for image, volume in enumerate(volumes):
                h.write('1\n')
                h.write('Round %d  Image %d  Role image  E = -91.0 '
                        'Lattice="%.9f 0.000000000 0.000000000 '
                        '0.000000000 10.000000000 0.000000000 '
                        '0.000000000 0.000000000 %.9f"\n'
                        % (round_at, image, 10.0, volume / 100.0))
                h.write('H   0.0000000000  0.0000000000  0.0000000000\n')


def _selftest_grad_rows(main_force, gas_force, el_force, drop_field=False, drop_gas=False):
    rows = []
    for index, vector in enumerate(main_force):
        row = [-value for value in vector]
        if index < len(gas_force):
            if not drop_gas:
                row = [row[axis] + gas_force[index][axis]
                       for axis in range(3)]
            if not drop_field:
                row = [row[axis] + el_force[index][axis]
                       for axis in range(3)]
        rows.append(row)
    return rows


def _plant_section_cell_wiring(root, work, caught):
    """Plant a round whose cell leg carries no metrics of its own.

    The wiring is what this section proves: a round with no
    `cell_metrics` reaches the path-cell clause, and the leg is counted
    as an objective rather than silently dropped.
    """
    # The wiring: a round with no `cell_metrics` reaches the clause, and one
    # with them does not (it is the single-structure reader's round).
    bare = os.path.join(root, 'tmp_20260901_130000')
    os.makedirs(bare)
    shutil.copytree(os.path.join(work, 'examples', 'path_demo_cell'),
                    os.path.join(bare, 'examples', 'path_demo_cell'))
    with open(os.path.join(bare, 'results.json'), 'w') as h:
        json.dump({'example_results': []}, h)
    problems, cell_counts = audit_cell(bare)
    if problems:
        caught.append('D path cell wiring reported: %s' % problems)
    if cell_counts['path_cell_legs'] != 1 \
            or cell_counts['path_cell_moved'] != 1 \
            or cell_counts['path_cell_matched'] != 2:
        caught.append('D did not reach the path cell clause: %s'
                      % cell_counts)


def _plant_section_molcas_leg_verdict(caught):
    """Plant three agreeing legs and the two ways one stops agreeing.

    The magnitude is the one the detector was checked with on round
    `tmp_20261027_173607`: editing one printed digit of
    `casscf_gradient`'s `pyrai3md_input` log by 1.9e-05 named that case
    and left the other 23 at 0.0.
    """
    # The magnitude planted here is the one the detector was checked with on
    # round `tmp_20261027_173607`: editing one printed digit of
    # `casscf_gradient`'s `pyrai3md_input` log by 1.9e-05 named that case and
    # left the other 23 at 0.0.
    legs = {leg: {'energy': [-77.70750863, -77.55634956],
                  'gradient': [[[0.0, 0.0, -0.016943]]]}
            for leg in MOLCAS_LEGS}
    if _molcas_leg_verdict('demo', _molcas_leg_diffs(legs)):
        caught.append('G reported clean legs: %s'
                      % _molcas_leg_verdict('demo',
                                            _molcas_leg_diffs(legs)))
    legs['pyrai3md_agent']['energy'][0] -= 1.9e-05
    planted = _molcas_leg_verdict('demo', _molcas_leg_diffs(legs))
    if not any('energy' in line for line in planted):
        caught.append('G missed the leg whose root energy moved: %s'
                      % planted)
    legs['pyrai3md_agent']['energy'].pop()
    planted = _molcas_leg_verdict('demo', _molcas_leg_diffs(legs))
    if not any('energy_shape' in line for line in planted):
        caught.append('G read a dropped state as agreement: %s' % planted)


def _plant_section_molcas_contract(caught):
    """Plant the ESPF two-pass contract and its five refusals."""
    # requested coupling, the ESPF two-pass answer, and an `expected` block
    # that asks for energy and NAC but no gradient.
    raw = {'nstate': 3, 'atoms': ['C'] * 6, 'jobtype': 'nac',
           'embedding': 'charge', 'method': 'casscf'}
    arm = {'gradient': [[[0.0, 0.0, 0.0]] * 6],
           'nac': [[[0.0, 0.0, 0.0]] * 6],
           'target_nac_pair': [[0, 1]],
           'two_pass': True,
           'nac_approximation': 'two_pass_no_esfp',
           'nac_csf_included': False,
           'energy_source': 'casscf'}
    expected = {'energy': {}, 'nac': {'shape': [1, 6, 3]}}
    if _molcas_contract('demo', raw, arm, expected):
        caught.append('G reported a clean contract: %s'
                      % _molcas_contract('demo', raw, arm, expected))
    asks_gradient = dict(expected, gradient={'shape': [3, 6, 3]})
    if _molcas_contract('demo', raw,
                        dict(arm, gradient=[[[0.0, 0.0, 0.0]] * 6] * 3),
                        asks_gradient):
        caught.append('G reported a clean three-state gradient')
    if not any('GRADIENT SHAPE' in line for line in
               _molcas_contract('demo', raw, arm, asks_gradient)):
        caught.append('G missed a gradient with the wrong state count')
    if not any('PASS COUNT' in line for line in
               _molcas_contract('demo', raw, dict(arm, two_pass=False),
                                expected)):
        caught.append('G missed the ESPF run that took one pass')
    if not any('ENERGY SOURCE' in line for line in
               _molcas_contract('demo', dict(raw, method='caspt2'), arm,
                                expected)):
        caught.append('G read a CASSCF energy for a CASPT2 case')
    if not any('EXPECTATION MISSING' in line for line in
               _molcas_contract('demo', raw, arm, None)):
        caught.append('G stayed silent with no test YAML to read')


def _plant_section_grad_identity(work, caught):
    """Plant one QM/MM step's four files and break the three rows.

    A detector that only sees the first break (a moved MM row) is not
    reading the auxiliaries at all.
    """
    # that moved, a QM row that lost both auxiliaries, and a QM row whose
    # field term was dropped. A detector that only sees the first is not
    # reading the auxiliaries at all.
    grad_dir = os.path.join(work, 'tmp_lammps_qmmm_energy_DEMO')
    os.makedirs(grad_dir, exist_ok=True)
    main_force = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0], [7.0, -8.0, 9.0],
                  [-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]]
    gas_force = [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.0, 0.0]]
    el_force = [[-0.75, 0.75, -0.75], [0.5, -0.5, 0.5], [0.0, 0.0, 0.25]]

    def _write_table(name, rows, header):
        with open(os.path.join(grad_dir, name), 'w') as h:
            h.write(header + '\n')
            for index, vector in enumerate(rows, start=1):
                h.write('%d ' % index + ' '.join(
                    '%.16e' % value for value in vector) + '\n')


    dump_header = 'ITEM: ATOMS id fx fy fz'
    for name, rows in (('demo.force', main_force),
                       ('demo.u_gas.force', gas_force),
                       ('demo.qmmm_el.force', el_force),
                       ('demo.grad', _selftest_grad_rows(main_force, gas_force, el_force, ))):
        _write_table(name, rows, dump_header)
    problems, counts = audit_grad_identity(work)
    if problems or counts['checked'] != 1:
        caught.append('I reported a clean grad set: checked=%d %s'
                      % (counts['checked'], problems))
    clean = _selftest_grad_rows(main_force, gas_force, el_force, )
    _write_table('demo.grad',
                 clean[:3] + [[0.0, 0.0, 0.0]] + clean[4:],
                 dump_header)
    problems, _ = audit_grad_identity(work)
    if not any('I GRAD IDENTITY' in line for line in problems):
        caught.append('I missed a moved MM row: %s' % problems)
    _write_table('demo.grad', _selftest_grad_rows(main_force, gas_force, el_force, drop_field=True), dump_header)
    problems, _ = audit_grad_identity(work)
    if not any('I GRAD IDENTITY' in line for line in problems):
        caught.append('I missed a QM row that dropped its field term: %s'
                      % problems)
    _write_table('demo.grad', _selftest_grad_rows(main_force, gas_force, el_force, drop_gas=True), dump_header)
    problems, _ = audit_grad_identity(work)
    if not any('I GRAD IDENTITY' in line for line in problems):
        caught.append('I missed a QM row that dropped U_gas: %s'
                      % problems)
    # A deck written back by the framework is still the clean one.
    _write_table('demo.grad', _selftest_grad_rows(main_force, gas_force, el_force, ), dump_header)

    # The skill's own writers spell `id x y z` under a comment header, and
    # the main force dump has no id column. Both shapes are the files a
    # skill round actually leaves behind; the dump-only tables above hid a
    # reader that called every comment+id table "no id" and skipped it.
    def _write_ids(name, rows, header):
        with open(os.path.join(grad_dir, name), 'w') as h:
            h.write(header + '\n')
            for index, vector in enumerate(rows, start=1):
                h.write('%d ' % index + ' '.join(
                    '%.16e' % value for value in vector) + '\n')

    def _write_cold_dump(name, rows):
        with open(os.path.join(grad_dir, name), 'w') as h:
            h.write('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n'
                    'ITEM: BOX BOUNDS ff ff ff\n' % len(rows))
            for _axis in range(3):
                h.write('-1.0 1.0\n')
            h.write('ITEM: ATOMS fx fy fz\n')
            for vector in rows:
                h.write(' '.join('%.16e' % value for value in vector) + '\n')

    _write_ids('demo.grad', _selftest_grad_rows(main_force, gas_force,
                                                el_force), '# id gx gy gz')
    _write_cold_dump('demo.force', main_force)
    _write_ids('demo.u_gas.force', gas_force, '# id fx fy fz')
    _write_ids('demo.qmmm_el.force', el_force, '# id fx fy fz')
    problems, counts = audit_grad_identity(work)
    if problems or counts['checked'] != 1:
        caught.append('I skipped the skill-writer table shapes: checked=%d %s'
                      % (counts['checked'], problems))


def _plant_section_skill_carriers(root, caught):
    """Plant one case on three carriers; break its deck, then its energy.

    The clean pass proves K reads the three directories; the two corruptions
    prove it compares the deck and the number rather than counting files.
    """
    work = os.path.join(root, 'tmp_skill_carriers')
    os.makedirs(work)
    entries = []
    dirs = {}
    for mode, suffix in (('skill_agent', '_skill'), ('pyrai3md_agent', '_agent'),
                         ('pyrai3md_input', '_input')):
        entries.append({'name': 'openqp_skill_demo' + suffix, 'mode': mode,
                        'status': 'pass'})
        base = os.path.join(work, 'openqp_skill', mode, 'tmp', 'tmp_demo_test')
        os.makedirs(base)
        dirs[mode] = base
        with open(os.path.join(base, 'demo_test.inp'), 'w') as handler:
            handler.write('grad=1\npath=/%s/tmp/tmp_demo_test\n' % mode)
        with open(os.path.join(base, 'energies'), 'w') as handler:
            handler.write('-77.0\n')
    with open(os.path.join(work, 'results.json'), 'w') as handler:
        json.dump({'skill_results': entries}, handler)

    problems, counts = audit_skill_carriers(work)
    if problems or counts['deck_files'] != 1 \
            or counts['energy_compared'] != 1:
        caught.append('clean K: deck_files=%d energy_compared=%d problems=%s'
                      % (counts['deck_files'], counts['energy_compared'],
                         problems))

    deck = os.path.join(dirs['pyrai3md_input'], 'demo_test.inp')
    with open(deck, 'w') as handler:
        handler.write('grad=2\npath=/pyrai3md_input/tmp/tmp_demo_test\n')
    problems, _ = audit_skill_carriers(work)
    if not any('K DECK DIVERGENCE' in line for line in problems):
        caught.append('K missed a changed engine deck: %s' % problems)

    with open(deck, 'w') as handler:
        handler.write('grad=1\npath=/pyrai3md_input/tmp/tmp_demo_test\n')
    energy = os.path.join(dirs['pyrai3md_input'], 'energies')
    with open(energy, 'w') as handler:
        handler.write('-77.5\n')
    problems, _ = audit_skill_carriers(work)
    if not any('K ENERGY DIVERGENCE' in line for line in problems):
        caught.append('K missed a changed final energy: %s' % problems)


def _plant_section_path(work, caught):
    """Plant the path family's composite, held-bond and cell arms.

    A composite path leg and its single-layer partner, whose profile
    tables agree on the geometry both declare, plus a held bond that
    never moves: the clean state each arm below is judged against. The
    geometry record is written the way the walk writes it (count,
    comment, atoms) because the hold is read frame by frame from that
    layout, and a band's held bond is read over its images.
    """
    # Section F. A composite path leg and its single-layer partner, whose
    # profile tables agree on the geometry both declare, plus a held bond
    # that never moves - the clean state each arm below is judged against.
    # The geometry record is written the way the walk writes it (count,
    # comment, atoms) because the hold is read frame by frame from that
    # layout, and a band's held bond is read over its images.

    pair_dir = os.path.join(work, 'examples', 'path_demo_multiscale', 'yaml')
    _selftest_path_leg(pair_dir, [(-91.5, 'reactant'), (-91.4, 'image'),
                               (-91.3, 'product')])
    single_dir = os.path.join(work, 'examples', 'path_demo', 'yaml')
    _selftest_path_leg(single_dir, [(-91.5, 'reactant'), (-91.2, 'image'),
                                 (-91.3, 'product')])
    # The composite leg's region declaration, which is what makes it a pair
    # member: read off its own flat config, like the scope tests of C2. The
    # single-layer half declares no region and no condition - it is the
    # partner and nothing else, so it must not be read as a condition leg.
    with open(os.path.join(pair_dir, 'demo.json'), 'r') as h:
        declared = json.load(h)
    declared['highlevel'] = [[1, 2, 3]]
    with open(os.path.join(pair_dir, 'demo.json'), 'w') as h:
        json.dump(declared, h)
    with open(os.path.join(single_dir, 'demo.json'), 'r') as h:
        plain = json.load(h)
    plain.pop('constraints', None)
    with open(os.path.join(single_dir, 'demo.json'), 'w') as h:
        json.dump(plain, h)

    # A second condition leg whose held quantities are an angle and a
    # dihedral. The member atoms of every kind live under `bonds`, so a
    # reader that looked for `angles`/`dihedrals` would answer empty here and
    # audit this leg as one with no hold at all - the shape a round's
    # held-angle and held-dihedral walks were lost to before the key was
    # read off the model. The geometry is bent, not collinear, because both
    # readings divide by a vector length.
    kinds_dir = os.path.join(work, 'examples', 'path_demo_kinds', 'yaml')
    os.makedirs(kinds_dir)
    with open(os.path.join(kinds_dir, 'demo.json'), 'w') as h:
        json.dump({'opt_method': 'mep', 'theory': ['xtb'],
                   'constraints': {'units': {'angle': 'degree'},
                                   'constraints': [
                                       {'kind': 'angle',
                                        'bonds': [[1, 2, 3]],
                                        'tolerance': 1e-10,
                                        'id': 'angle'},
                                       {'kind': 'dihedral',
                                        'bonds': [[1, 2, 3, 4]],
                                        'tolerance': 1e-10,
                                        'id': 'torsion'}]}}, h)
    with open(os.path.join(kinds_dir, 'path_geom.xyz'), 'w') as h:
        for _ in range(2):
            h.write('4\n')
            h.write('Point 1 Direction forward E = -91.0\n')
            h.write('H   0.0000000000  0.0000000000  1.1000000000\n')
            h.write('C   0.0000000000  0.0000000000  0.0000000000\n')
            h.write('N   1.1000000000  0.0000000000  0.0000000000\n')
            h.write('O   1.1000000000  1.1000000000  0.0000000000\n')

    problems, path_counts = audit_path(work)
    if problems:
        caught.append('clean F reported: %s' % problems)
    if (path_counts['condition_legs'] != 2 or path_counts['held_rows'] != 6
            or path_counts['held_broken']
            or path_counts['pairs'] != 1 or path_counts['pair_rows'] != 2
            or path_counts['pair_disagreements']):
        caught.append('clean F: legs=%d rows=%d broken=%d pairs=%d '
                      'pair_rows=%d disagreements=%d, expected 2, 6, 0, 1, '
                      '2 and 0 (%s)'
                      % (path_counts['condition_legs'],
                         path_counts['held_rows'], path_counts['held_broken'],
                         path_counts['pairs'], path_counts['pair_rows'],
                         path_counts['pair_disagreements'], problems))

    # Arm one: move one frame's C-N, and the hold has to be named. Only the
    # first of the two identical frames moves, so the reading is a genuine
    # deviation (1.0 Angstrom against a limit of 1.9e-9) rather than the
    # same motion on both, which would be a constant length again.
    geometry = os.path.join(pair_dir, 'path_geom.xyz')
    with open(geometry, 'r') as h:
        clean_geometry = h.read()
    with open(geometry, 'w') as h:
        h.write(clean_geometry.replace(
            'C   0.0000000000  0.0000000000  0.0000000000',
            'C   0.0000000000  0.0000000000  1.0000000000', 1))
    problems, path_counts = audit_path(work)
    if not any('F HOLD BROKEN' in line for line in problems) \
            or path_counts['held_broken'] != 1:
        caught.append('F missed the planted broken hold: broken=%d %s'
                      % (path_counts['held_broken'], problems))
    with open(geometry, 'w') as h:
        h.write(clean_geometry)
    _problems, _counts = audit_path(work)
    if _problems:
        caught.append('F reported the restored geometry: %s' % _problems)

    # Arm two: move the single-layer partner's product row away from the
    # composite's. The two tables share their first and last rows, so one
    # moved endpoint is exactly what the pair reading exists to catch - and
    # the partner alone has to be enough to name it.
    profile = os.path.join(single_dir, 'path_profile.txt')
    with open(profile, 'r') as h:
        clean_profile = h.read()
    with open(profile, 'w') as h:
        h.write(clean_profile.replace('-91.3000000000', '-91.2990000000'))
    problems, path_counts = audit_path(work)
    if not any('F COMPOSITE DIVERGENCE' in line for line in problems) \
            or path_counts['pair_disagreements'] != 1:
        caught.append('F missed the planted composite divergence: '
                      'disagreements=%d %s'
                      % (path_counts['pair_disagreements'], problems))
    with open(profile, 'w') as h:
        h.write(clean_profile)
    _problems, _counts = audit_path(work)
    if _problems:
        caught.append('F reported the restored profile: %s' % _problems)

    # Section F's frozen-atom clause. A leg that declares `fix_atoms` keeps
    # atom 2 at its anchor; the first frame is the reference and the second
    # moves atom 1 only, so the clean record is a constant atom 2.
    fix_dir = os.path.join(work, 'examples', 'path_demo_fix', 'yaml')
    os.makedirs(fix_dir)
    with open(os.path.join(fix_dir, 'demo.json'), 'w') as h:
        json.dump({'opt_method': 'mep', 'theory': ['xtb'],
                   'constraints': {'units': {'length': 'angstrom'},
                                   'constraints': [],
                                   'fixes': [{'atoms': [2],
                                              'id': 'hold_c'}]}}, h)


    _selftest_fix_geometry(fix_dir, 'none')
    problems, path_counts = audit_path(work)
    if problems:
        caught.append('clean F with the frozen leg reported: %s' % problems)
    if path_counts['fix_legs'] != 1 or path_counts['fix_atoms'] != 1:
        caught.append('clean F did not read the frozen leg: legs=%d '
                      'atoms=%d' % (path_counts['fix_legs'],
                                    path_counts['fix_atoms']))
    # Arm: the frozen atom moves in a walkable frame, and the ride is named.
    _selftest_fix_geometry(fix_dir, 'image')
    problems, path_counts = audit_path(work)
    if not any('F FROZEN ATOM MOVED' in line for line in problems) \
            or path_counts['fix_broken'] != 1:
        caught.append('F missed the planted frozen motion: broken=%d %s'
                      % (path_counts['fix_broken'], problems))
    # ... and the same motion in a `Role product` frame is the endpoint's
    # own choice, not a broken ride: the endpoint is never landed.
    _selftest_fix_geometry(fix_dir, 'product')
    problems, path_counts = audit_path(work)
    if any('F FROZEN ATOM MOVED' in line for line in problems):
        caught.append("F reported the endpoint's frozen atom: %s"
                      % problems)
    if not path_counts['fix_atoms']:
        caught.append('F stopped reading the frozen leg after the endpoint '
                      'arm: %s' % path_counts)

    # Section F's IRC branch clause: the two saddle rows of a `both` walk
    # have to be the one geometry. The arm is the defect measured on
    # `path_irc` before the reset (-91.5617 against -91.6663).
    irc_dir = os.path.join(work, 'examples', 'path_demo_irc', 'yaml')
    os.makedirs(irc_dir)
    with open(os.path.join(irc_dir, 'demo.json'), 'w') as h:
        json.dump({'opt_method': 'irc', 'theory': ['xtb']}, h)

    def _write_irc_profile(second):
        with open(os.path.join(irc_dir, 'path_profile.txt'), 'w') as h:
            h.write(' index direction energy de arc length perp_rms '
                    'perp_max radius role\n')
            h.write('%6d backward %20.10f 0.0 0.0 0.0 0.0 0.0 0.0 saddle\n'
                    % (1, -91.5616686633))
            h.write('%6d forward  %20.10f 0.0 0.0 0.0 0.0 0.0 0.0 saddle\n'
                    % (1, second))

    _write_irc_profile(-91.5616686633)
    problems, path_counts = audit_path(work)
    if any('F IRC BRANCHES' in line for line in problems):
        caught.append('F reported a clean IRC pair: %s' % problems)
    _write_irc_profile(-91.6662865772)
    problems, path_counts = audit_path(work)
    if not any('F IRC BRANCHES' in line for line in problems) \
            or path_counts['irc_origin_broken'] != 1:
        caught.append('F missed the IRC branch that started elsewhere: '
                      'broken=%d %s'
                      % (path_counts['irc_origin_broken'], problems))
    _write_irc_profile(-91.5616686633)

    # Section F's band-frame clause: two adjacent frames of one round may
    # not differ by a lattice translation. The arm is the shape the fix
    # closes (a product written on the far side of a face).
    band_dir = os.path.join(work, 'examples', 'path_demo_band', 'yaml')
    os.makedirs(band_dir)
    with open(os.path.join(band_dir, 'demo.json'), 'w') as h:
        json.dump({'opt_method': 'neb', 'theory': ['xtb']}, h)

    def _write_band_frames(second_x):
        with open(os.path.join(band_dir, 'path_geom.xyz'), 'w') as h:
            for image, x in ((0, 0.4), (1, second_x)):
                h.write('1\n')
                h.write('Round 1  Image %d  Role image  E = -1.0 '
                        'Lattice="10.000000000 0.000000000 0.000000000 '
                        '0.000000000 10.000000000 0.000000000 '
                        '0.000000000 0.000000000 10.000000000"\n' % image)
                h.write('H   %.10f  0.0000000000  0.0000000000\n' % x)

    _write_band_frames(0.5)
    problems, path_counts = audit_path(work)
    if any('F BAND FRAME JUMP' in line for line in problems):
        caught.append('F reported a clean band chain: %s' % problems)
    _write_band_frames(0.4 + 10.0)
    problems, path_counts = audit_path(work)
    if not any('F BAND FRAME JUMP' in line for line in problems) \
            or path_counts['band_jumps'] != 1:
        caught.append('F missed the frame one box away: jumps=%d %s'
                      % (path_counts['band_jumps'], problems))
    _write_band_frames(0.5)

    # Section D's path-cell clause. A path leg's table is not `read_status`'s
    # (its header starts with a direction), so it is read by name.
    cell_leg = os.path.join(work, 'examples', 'path_demo_cell', 'yaml')
    os.makedirs(cell_leg)
    with open(os.path.join(cell_leg, 'demo.json'), 'w') as h:
        json.dump({'opt_method': 'mep', 'optimize_cell': True,
                   'theory': ['xtb']}, h)


    _selftest_path_cell(cell_leg, [1000.0, 1000.2])
    problems, readings = _path_cell_leg(cell_leg)
    if problems:
        caught.append('clean D path cell reported: %s' % problems)
    if readings['moved'] != 1 or readings['matched'] != 2:
        caught.append('clean D path cell readings: %s' % readings)
    # Arm one: a box that did not move.
    _selftest_path_cell(cell_leg, [1000.0, 1000.0])
    problems, _readings = _path_cell_leg(cell_leg)
    if not any('D PATH CELL STATIC' in line for line in problems):
        caught.append('D missed the stationary path box: %s' % problems)
    # Arm two: a recorded volume no frame of the run carries.
    _selftest_path_cell(cell_leg, [1000.0, 1000.2])
    record = os.path.join(cell_leg, 'path_status.txt')
    with open(record, 'r') as h:
        clean_record = h.read()
    with open(record, 'w') as h:
        h.write(clean_record.replace('1000.2000000000', '1005.0000000000'))
    problems, _readings = _path_cell_leg(cell_leg)
    if not any('D PATH CELL MISMATCH' in line for line in problems):
        caught.append('D missed the volume no frame carries: %s' % problems)
    with open(record, 'w') as h:
        h.write(clean_record)
    # Arm three: the cell columns dropped between input and walk.
    with open(record, 'w') as h:
        h.write(' step direction energy shift max_d rms_d max_g rms_g '
                'radius residual\n')
        h.write('%6d forward -91.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n' % 1)
        h.write('%6d forward -91.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n' % 2)
    problems, _readings = _path_cell_leg(cell_leg)
    if not any('D PATH CELL COLUMNS' in line for line in problems):
        caught.append('D missed the dropped cell columns: %s' % problems)
    with open(record, 'w') as h:
        h.write(clean_record)

    # A band's row is its reading *before* the step, so step N has to match a
    # frame of round N-1. The arm plants the defect this check found: round 2
    # was evaluated on a box round 1 never committed (the parent box written
    # into every frame).

    _selftest_band_cell(cell_leg, 1000.1)
    problems, readings = _path_cell_leg(cell_leg)
    if problems:
        caught.append('clean D path band reported: %s' % problems)
    if readings['matched'] != 1:
        caught.append('clean D path band matched %d step(s), expected 1'
                      % readings['matched'])
    _selftest_band_cell(cell_leg, 1000.0)
    problems, _readings = _path_cell_leg(cell_leg)
    if not any('D PATH CELL MISMATCH' in line for line in problems):
        caught.append('D missed the band whose frames carry one box: %s'
                      % problems)
    with open(record, 'w') as h:
        h.write(clean_record)
    _selftest_path_cell(cell_leg, [1000.0, 1000.2])


def _selftest_cp2k_log_text(total, banner=True):
    """One engine log: the banner the claim rests on, two SCF totals."""
    head = (' CP2K| version string:                          '
            '       CP2K version 2024.3' + chr(10) if banner else
            ' some other program, which prints a total too:' + chr(10))
    return (head + '  Total energy:                     %.16f'
            % (total - 1.0) + chr(10)
            + '  Total energy:                     %.16f' % total
            + chr(10))


def _plant_section_cp2k(work, opt_dir, opt_values, rows, caught):
    """Plant the CP2K reader: the banner claim, its two totals, a decoy log."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    # A CP2K leg, for the reader this tool did not have. The engine writes its
    # own run log and the recorded row has to agree with the LAST
    # `Total energy:` in it, so the planted log prints a first total a whole
    # Hartree away: the clean audit below is itself the witness for that rule,
    # because a reader that took the first value would report the leg here.
    cp2k_dir = os.path.join(work, 'examples', 'demo_cp2k_opt', 'yaml')
    os.makedirs(cp2k_dir)
    with open(os.path.join(cp2k_dir, 'opt_status.txt'), 'w') as handler:
        handler.write(status_header() + chr(10))
        for number, values in enumerate(opt_values, start=1):
            handler.write(status_line(number, values) + chr(10))
    cp2k_calcdir = os.path.join(cp2k_dir, 'tmp_CP2K_demo')
    os.makedirs(cp2k_calcdir)
    cp2k_log = os.path.join(cp2k_calcdir, 'demo_cp2k.log')

    with open(cp2k_log, 'w') as handler:
        handler.write(_selftest_cp2k_log_text(rows[-1][1]))

    # The gate's control, planted where it can bite: a `*.log` of its own
    # inside the LAMMPS leg's directory, printing a number no engine agrees
    # with. A CP2K reader that skipped the banner test answers first
    # (`attribute_engine` asks in sorted order, and CP2K sorts before LAMMPS)
    # and hands back this decoy for a LAMMPS run, which no documented unit
    # variant converts into the recorded row. The clean pass below is what
    # the banner is buying.
    with open(os.path.join(opt_dir, 'tmp_demo25', 'decoy.log'),
              'w') as handler:
        handler.write('  Total energy:                     -99.0000000000'
                      '000000' + chr(10))

    problems, counts = audit_optimize_engine(work)
    if problems:
        caught.append('clean C2 with the cp2k leg reported: %s' % problems)
    # The leg count is four here and five on the way in, because the section
    # D arms just above remove one record on purpose (`D NO RECORD`); this is
    # the same fixture, walked later. `checked` and `matched` both rising by
    # one is the reading that matters: the new leg took part and agreed.
    if counts['legs'] != 7 or counts['checked'] != 3 \
            or counts['matched'] != 3 or counts['composite'] != 1:
        caught.append('clean C2 with the cp2k leg: legs=%d checked=%d '
                      'matched=%d composite=%d, expected 7, 3, 3 and 1 (%s)'
                      % (counts['legs'], counts['checked'],
                         counts['matched'], counts['composite'], problems))

    # Arm one, the engine side: move the last total the engine printed and
    # the row it was judged against has to be named.
    with open(cp2k_log, 'w') as handler:
        handler.write(_selftest_cp2k_log_text(rows[-1][1] + 1.0e-3))
    problems, _ = audit_optimize_engine(work)
    if not any('demo_cp2k_opt' in line for line in problems):
        caught.append('C2 missed the planted cp2k engine change: %s'
                      % problems)

    # Arm two, the other direction: take the banner away and the reader has
    # to decline the directory. The leg does not then fail - it leaves the
    # compared set, which is what these two counts say. A reader that claimed
    # a directory by its name would keep matching here, and that is exactly
    # the reading this arm refuses to allow.
    with open(cp2k_log, 'w') as handler:
        handler.write(_selftest_cp2k_log_text(rows[-1][1], banner=False))
    _problems, counts = audit_optimize_engine(work)
    if counts['checked'] != 2 or counts['no_engine'] != 4:
        caught.append('C2 with the banner gone: checked=%d no_engine=%d, '
                      'expected 2 and 4 (%s)'
                      % (counts['checked'], counts['no_engine'], _problems))
    with open(cp2k_log, 'w') as handler:
        handler.write(_selftest_cp2k_log_text(rows[-1][1]))
    _problems, _ = audit_optimize_engine(work)
    if _problems:
        caught.append('C2 reported the restored cp2k leg: %s' % _problems)


def _plant_section_molcas(work, opt_values, rows, caught):
    """Plant the MOLCAS reader: the completion marker and its SCF totals."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    # A MOLCAS leg, for the reader this tool did not have either. MOLCAS
    # prints neither `Total energy:` nor `converged SCF energy`: it writes
    # its own completion marker and its own SCF total, and the skill's own
    # parser answers for exactly those two tokens
    # (`molcas_skill/service/parser/log_parser.py:96, 279`). The planted log
    # carries two totals a whole Hartree apart, so the clean audit below is
    # itself the witness for the last-value rule - a reader that took the
    # first number would report the leg here - and it carries the marker,
    # because PySCF reads `*.log` too and sorts after MOLCAS.
    molcas_dir = os.path.join(work, 'examples', 'demo_molcas_opt', 'yaml')
    os.makedirs(molcas_dir)
    with open(os.path.join(molcas_dir, 'opt_status.txt'), 'w') as handler:
        handler.write(status_header() + chr(10))
        for number, values in enumerate(opt_values, start=1):
            handler.write(status_line(number, values) + chr(10))
    molcas_calcdir = os.path.join(molcas_dir, 'tmp_MOLCAS_demo')
    os.makedirs(molcas_calcdir)
    molcas_log = os.path.join(molcas_calcdir, 'demo_molcas.log')

    def molcas_log_text(total, marker=True):
        """One engine log: the completion marker, two SCF totals."""
        tail = ('Happy landing!' + chr(10)) if marker else ''
        first = (' Total SCF energy                              = '
                 '%.16f' % (total - 1.0) + chr(10))
        last = (' Total SCF energy                              = '
                '%.16f' % total + chr(10))
        return first + last + tail

    with open(molcas_log, 'w') as handler:
        handler.write(molcas_log_text(rows[-1][1]))

    problems, counts = audit_optimize_engine(work)
    if problems:
        caught.append('clean C2 with the molcas leg reported: %s' % problems)
    if counts['matched'] != 4 or counts['composite'] != 1:
        caught.append('clean C2 with the molcas leg: matched=%d '
                      'composite=%d, expected 4 and 1 (%s)'
                      % (counts['matched'], counts['composite'], problems))

    # Arm one, the engine side: move the last total the engine printed and
    # the row it was judged against has to be named.
    with open(molcas_log, 'w') as handler:
        handler.write(molcas_log_text(rows[-1][1] + 1.0e-3))
    problems, _ = audit_optimize_engine(work)
    if not any('demo_molcas_opt' in line for line in problems):
        caught.append('C2 missed the planted molcas engine change: %s'
                      % problems)

    # Arm two, the other direction: take the completion marker away and the
    # reader has to decline the directory. The leg then leaves the compared
    # set, which is what these two counts say; a reader that claimed a
    # directory by its name would keep matching here.
    with open(molcas_log, 'w') as handler:
        handler.write(molcas_log_text(rows[-1][1], marker=False))
    _problems, counts = audit_optimize_engine(work)
    if counts['checked'] != 3 or counts['no_engine'] != 4:
        caught.append('C2 with the molcas marker gone: checked=%d '
                      'no_engine=%d, expected 3 and 4 (%s)'
                      % (counts['checked'], counts['no_engine'], _problems))
    with open(molcas_log, 'w') as handler:
        handler.write(molcas_log_text(rows[-1][1]))
    _problems, _ = audit_optimize_engine(work)
    if _problems:
        caught.append('C2 reported the restored molcas leg: %s' % _problems)


def _plant_section_casscf(root, caught):
    """Plant the CASSCF branch: root energies, no SCF total, marker required."""
    # The CASSCF branch of the same reader: a log that prints no SCF total
    # but the RASSCF roots, the shape `md_molcas_fssh` leaves (round
    # tmp_20261020_111423's tmp_MOLCAS dirs are claimed by it, and the
    # state-resolved record carries the same root energies). The last
    # occurrence of each root wins, in root order; the marker is still
    # required.
    cas_dir = os.path.join(root, 'molcas_casscf')
    os.makedirs(cas_dir, exist_ok=True)
    cas_log = os.path.join(cas_dir, 'MOLCAS.log')

    def cas_text(marker=True):
        tail = ('Happy landing!' + chr(10)) if marker else ''
        return (('::    RASSCF root number  1 Total energy:    '
                 '-77.80294722' + chr(10))
                + ('::    RASSCF root number  2 Total energy:    '
                   '-77.53924403' + chr(10))
                + ('::    RASSCF root number  3 Total energy:    '
                   '-77.43586102' + chr(10)) + tail)

    with open(cas_log, 'w') as handler:
        handler.write(cas_text())
    got = _engine_molcas(cas_dir)
    if got != [-77.80294722, -77.53924403, -77.43586102]:
        caught.append('the CASSCF molcas branch returned %r' % (got,))
    with open(cas_log, 'w') as handler:
        handler.write(cas_text(marker=False))
    if _engine_molcas(cas_dir) is not None:
        caught.append('the CASSCF molcas branch claimed a log with no '
                      'completion marker')


def _plant_section_pressure_scope(work, opt_values, rows, caught):
    """Plant the pressure scope: E + pV is an objective, not a defect."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    # A leg under a pressure, for the other half of the same scope statement:
    # its energy column is `E + pV`, which no engine file printed, so it is
    # counted as an objective and left unjudged. The arm is two-sided on
    # purpose - the same directory, with only the declared pressure changed,
    # has to go back to being compared and matched. Otherwise the exemption
    # would pass a leg that never had a pressure at all, and a leg whose row
    # disagrees with its own engine file.
    press_dir = os.path.join(work, 'examples', 'demo_press', 'yaml')
    os.makedirs(press_dir)
    with open(os.path.join(press_dir, 'opt_status.txt'), 'w') as handler:
        handler.write(status_header() + chr(10))
        for number, values in enumerate(opt_values, start=1):
            handler.write(status_line(number, values) + chr(10))
    press_config = os.path.join(press_dir, 'demo_press.json')
    term = 1.0e-4
    with open(press_config, 'w') as handler:
        json.dump({'theory': ['xtb'], 'opt_pressure': 10000.0}, handler)
    press_calcdir = os.path.join(press_dir, 'tmp_press_xtb')
    os.makedirs(press_calcdir)
    # The engine's own reading: the row above carries the same number, while
    # the enthalpy the leg records is that number plus `pV` (simulated here by
    # writing the bare energy into the file and the raised one into the row).
    with open(os.path.join(press_calcdir, 'press.energy'), 'w') as handler:
        handler.write('%.16f' % (rows[-1][1] - term) + chr(10))
    _problems, counts = audit_optimize_engine(work)
    if counts['pressurized'] != 1 or _problems:
        caught.append('C2 with the pressure leg: pressurized=%d problems=%s'
                      % (counts['pressurized'], _problems))
    with open(press_config, 'w') as handler:
        json.dump({'theory': ['xtb'], 'opt_pressure': 0.0}, handler)
    _problems, counts = audit_optimize_engine(work)
    if counts['pressurized'] != 0 or counts['unmatched'] != 1:
        caught.append('C2 with the pressure declaration removed: '
                      'pressurized=%d unmatched=%d, expected 0 and 1 (%s)'
                      % (counts['pressurized'], counts['unmatched'],
                         _problems))
    # Put the declaration back before anything else walks the directory: the
    # arm above leaves the leg deliberately disagreed, and a later scope test
    # that re-audits the same fixture would read that planted row as a defect
    # of its own section.
    with open(press_config, 'w') as handler:
        json.dump({'theory': ['xtb'], 'opt_pressure': 10000.0}, handler)


def _plant_section_state_scope(work, opt_values, rows, caught):
    """Plant the excited-state scope: the row carries the SCF total plus an excitation."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    # A leg driven on an excited state, the third scope case: its row is the
    # SCF total plus an excitation, which no engine line prints (the reading
    # round 15 gave on one geometry: row -13.5101372537 against a last
    # `Total energy:` of -13.56505323574282, with `TDDFPT| 1 1.49462 eV`
    # between them). The arm is two-sided like the pressure one - the same
    # directory, the same numbers, with only the declared state back on the
    # ground state, has to go back into the compared set and be named. That
    # keys the exemption to the declaration rather than to this fixture.
    state_dir = os.path.join(work, 'examples', 'demo_state', 'yaml')
    os.makedirs(state_dir)
    excitation = 0.05
    state_values = [list(values) for values in opt_values]
    state_values[-1][0] = rows[-1][1] + excitation
    with open(os.path.join(state_dir, 'opt_status.txt'), 'w') as handler:
        handler.write(status_header() + chr(10))
        for number, values in enumerate(state_values, start=1):
            handler.write(status_line(number, values) + chr(10))
    state_calcdir = os.path.join(state_dir, 'tmp_CP2K_state')
    os.makedirs(state_calcdir)
    with open(os.path.join(state_calcdir, 'state.log'), 'w') as handler:
        handler.write(_selftest_cp2k_log_text(rows[-1][1]))
    state_config = os.path.join(state_dir, 'demo_state.json')

    def _declare_state(index):
        with open(state_config, 'w') as handler:
            json.dump({'theory': ['cp2k'], 'method': 'tddft',
                       'opt_state_i': index}, handler)

    _declare_state(1)
    _problems, counts = audit_optimize_engine(work)
    if counts['state_scoped'] != 1 or _problems:
        caught.append('C2 with the excited-state leg: state_scoped=%d '
                      'problems=%s' % (counts['state_scoped'], _problems))
    _declare_state(0)
    _problems, counts = audit_optimize_engine(work)
    if counts['state_scoped'] != 0 or not any('demo_state' in line
                                              for line in _problems):
        caught.append('C2 with the state back on the ground state: '
                      'state_scoped=%d problems=%s, expected 0 and that the '
                      'leg is named' % (counts['state_scoped'], _problems))


def _plant_section_budget(work, opt_dir, caught):
    """Plant the budget-stop count: a leg that stops on its step budget still reads pass."""
    # The budget-stop count, and the reason it is a count and not a verdict: a
    # leg that stops on its step budget records `pass` either way, so the
    # closure is the only place the two kinds of pass differ. That is exactly
    # why #399 had to mark the returncode-0 route, and why the count is read
    # from the leg's own log here. `demo_opt`'s record is the leg and the log
    # beside it is the closure; the arm below says a converged one counts as
    # nothing.
    budget_log = os.path.join(opt_dir, 'demo_opt.log')
    with open(budget_log, 'w') as handler:
        handler.write('PyRAI3MD End: Oct 19 2026 21:04:16 Total: '
                      '00:00:12:08 Status: failed: Maximum optimization '
                      'steps reached' + chr(10))
    _problems, counts = audit_optimize_engine(work)
    if counts['budget_stopped'] != 1:
        caught.append('C2 with the budget-stopped leg: budget_stopped=%d, '
                      'expected 1 (%s)'
                      % (counts['budget_stopped'], _problems))
    if any('demo_opt' in line for line in _problems):
        # A budget stop is a kind of pass, not a defect: the state arm above
        # leaves its own leg named on purpose, so this checks the new count
        # against the leg it is about.
        caught.append('C2 reported the budget-stopped leg as a defect: %s'
                      % _problems)
    with open(budget_log, 'w') as handler:
        handler.write('PyRAI3MD End: Oct 19 2026 21:19:33 Total: '
                      '00:00:00:18 Status: completed' + chr(10))
    _problems, counts = audit_optimize_engine(work)
    if counts['budget_stopped'] != 0:
        caught.append('C2 with the leg converged: budget_stopped=%d, '
                      'expected 0 (%s)'
                      % (counts['budget_stopped'], _problems))
    os.remove(budget_log)
    _problems, _counts = audit_optimize_engine(work)
    if any('demo_opt' in line for line in _problems):
        caught.append('C2 after the closure was removed: %s' % _problems)


def _plant_section_carriers(work, rows, caught):
    """Plant the C3/E carriers: two carriers of one composite example and its terms."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    # C3 and E, the two sections an operator's hand-checks became. The
    # fixture is planted here, after every count above, so the sections that
    # were already pinned read the leg set they were written against. Two
    # carriers of one composite example, each record adding up to its own
    # energy column, and a document entry per carrier carrying the same
    # numbers: the clean reading below is itself the witness for both rules,
    # because a reader that took the columns from the wrong end of the row,
    # or summed the cell columns into the terms, would report this fixture.
    for carrier in ('yaml', 'input'):
        directory = os.path.join(work, 'examples', 'demo_terms', carrier)
        os.makedirs(directory)
        with open(os.path.join(directory, 'opt_status.txt'), 'w') as h:
            h.write(status_header(terms=('high[qm]', 'low[qm2]')) + '\n')
            for number in (1, 2):
                energy = rows[-1][1] + number * 1.0e-4
                # Two halves add back to the energy column exactly: this leg
                # declares no crossing, so the identity has to close.
                h.write(status_line(number,
                                    [energy, 0.0, 1.5e-02, 4.0e-03,
                                     5.0e-02, 1.3e-02, energy / 2.0,
                                     energy / 2.0], extra=2) + '\n')
    with open(os.path.join(work, 'results.json'), encoding='utf-8') as h:
        carrier_document = json.load(h)
    for carrier in ('yaml', 'input'):
        carrier_document['example_results'].append({
            'name': 'example_demo_terms_' + carrier,
            'status': 'pass',
            'theory': 'xtb',
            'cell_metrics': {'steps': 2, 'volume_last': 1000.0},
        })
    with open(os.path.join(work, 'results.json'), 'w') as h:
        json.dump(carrier_document, h)

    # The two scope controls the section's judgement rests on, planted where
    # they are cheap: the same record declared as a crossing must be a
    # READING, because a gap term outside the columns is what a crossing is.
    terms_dir = os.path.join(work, 'examples', 'demo_terms', 'yaml')
    terms_config = os.path.join(terms_dir, 'demo_terms.json')

    def _declare_terms_method(method):
        with open(terms_config, 'w') as handler:
            json.dump({'theory': ['xtb'], 'opt_method': method}, handler)

    _declare_terms_method('ts')
    problems, counts = audit_term_decomposition(work)
    if problems:
        caught.append('clean C3 with the term legs reported: %s' % problems)
    if counts['legs'] != 2 or counts['rows'] != 4 or counts['exact'] != 2 \
            or counts['offset_readings'] or counts['ragged']:
        caught.append('clean C3 with the term legs: legs=%d rows=%d '
                      'exact=%d readings=%d ragged=%d, expected 2, 4, 2, 0 '
                      'and 0 (%s)'
                      % (counts['legs'], counts['rows'], counts['exact'],
                         counts['offset_readings'], counts['ragged'],
                         problems))
    problems, counts = audit_carrier_parity(work)
    if problems:
        caught.append('clean E with the term pair reported: %s' % problems)
    if counts['term_pairs'] != 1 or counts['disagreeing']:
        caught.append('clean E with the term pair: term_pairs=%d '
                      'disagreeing=%d, expected 1 and 0 (%s)'
                      % (counts['term_pairs'], counts['disagreeing'],
                         problems))

    # Section D reads the same fixture, and that is the criterion for the
    # two guards this plant needed: `demo_terms` records a cell metric for a
    # table with no cell columns, which used to raise a KeyError out of
    # `_cell_metric_from_record` and again out of `cell_verdict` - taking the
    # whole report down instead of answering that it cannot recompute that one
    # metric. Reaching the counts below without a crash is what pins both.
    problems, cell_counts = audit_cell(work)
    if any('demo_terms' in line for line in problems):
        caught.append('D reported the fixed-cell term leg: %s' % problems)
    if cell_counts['legs'] < 2:
        caught.append('D did not reach the fixed-cell term leg: legs=%d'
                      % cell_counts['legs'])

    # Arm one, the C3 side: double one term column of a leg that declares no
    # crossing, and it has to be named. The direction matters - this fixture's
    # terms are both positive halves, so the doubling sums ABOVE the energy
    # column, and a crossing's legitimate negative offset is on the other side
    # of zero and must stay a reading.
    terms_record = os.path.join(terms_dir, 'opt_status.txt')
    with open(terms_record, 'r') as handler:
        clean_terms = handler.read()
    doubled = []
    for line in clean_terms.splitlines()[1:]:
        values = line.split()
        values[-1] = '%.15e' % (float(values[-1]) * 2.0)
        doubled.append(line[:len(line) - len(line.lstrip())]
                       + '  '.join(values))
    with open(terms_record, 'w') as handler:
        handler.write(chr(10).join([clean_terms.splitlines()[0]]
                                   + doubled) + chr(10))
    problems, counts = audit_term_decomposition(work)
    if not any('C3 TERM COLUMNS DO NOT CLOSE' in line for line in problems):
        caught.append('C3 missed the planted double count: %s' % problems)

    # ... and the same deviation under a crossing declaration has to leave
    # the judged set: without this arm the exemption would pass a leg that
    # never declared a crossing at all.
    _declare_terms_method('mecp')
    problems, counts = audit_term_decomposition(work)
    if problems or counts['offset_readings'] != 1:
        caught.append('C3 did not read the same deviation as a crossing: '
                      'readings=%d %s' % (counts['offset_readings'],
                                          problems))
    _declare_terms_method('ts')
    with open(terms_record, 'w') as handler:
        handler.write(clean_terms)
    _problems, _counts = audit_term_decomposition(work)
    if _problems:
        caught.append('C3 reported the restored term leg: %s' % _problems)

    # Arm two, the E side: move one carrier's recorded metric. The two
    # carriers of this fixture share every number, so a difference here is
    # exactly what the section exists to catch - and one carrier alone has to
    # be enough to name it, which is why only the input half moves.
    for entry in carrier_document['example_results']:
        if entry.get('name') == 'example_demo_terms_input':
            entry['cell_metrics']['volume_last'] = 1001.0
    with open(os.path.join(work, 'results.json'), 'w') as h:
        json.dump(carrier_document, h)
    problems, counts = audit_carrier_parity(work)
    if not any('E CARRIER DIVERGENCE' in line for line in problems) \
            or counts['disagreeing'] != 1:
        caught.append('E missed the planted carrier divergence: '
                      'disagreeing=%d %s' % (counts['disagreeing'],
                                             problems))
    for entry in carrier_document['example_results']:
        if entry.get('name') == 'example_demo_terms_input':
            entry['cell_metrics']['volume_last'] = 1000.0
    with open(os.path.join(work, 'results.json'), 'w') as h:
        json.dump(carrier_document, h)
    _problems, _counts = audit_carrier_parity(work)
    if _problems:
        caught.append('E reported the restored carriers: %s' % _problems)


def _selftest_write_record(record, extras, entries):
    """Write the framework's frame record with the planted per-state columns."""
    with open(record, 'w') as handler:
        handler.write('%s\n' % ' '.join(
            ['time', 'Epot', 'Ekin', 'Etot', 'Epot1,2,3...']))
        for row in entries:
            handler.write('%15.2f%28.16f%28.16f%28.16f'
                          '%28.16f%28.16f\n'
                          % (row + extras))


def _plant_section_record_corruption(work, rows, record, extras, lammps_energy_file, caught):
    """Break the record and require A, B and C to name it."""
    # Corrupt the record's potential energy so A, B and C disagree with
    # the document, the log and the engine files respectively.
    broken = list(rows)
    broken[1] = (broken[1][0], broken[1][1] - 0.5, broken[1][2],
                 broken[1][3] - 0.5)
    _selftest_write_record(record, extras, broken)
    problems, _ = audit_records(work)
    if not any('A MISMATCH' in line for line in problems):
        caught.append('A missed the planted record change: %s' % problems)
    problems, _ = audit_log(work)
    if not any('B EPOT' in line for line in problems):
        caught.append('B missed the planted record change: %s' % problems)
    problems, _ = audit_engine(work)
    if not any('C RECORDED ENERGY NOT IN ENGINE FILES' in line
               for line in problems):
        caught.append('C missed the planted record change: %s' % problems)

    # A stale record (mtime before the round) must be named.
    os.utime(record, (time.mktime(time.strptime('20260831_120000',
                                                '%Y%m%d_%H%M%S')),
                      time.mktime(time.strptime('20260831_120000',
                                                '%Y%m%d_%H%M%S'))))
    problems, _ = audit_records(work)
    if not any('A STALE' in line for line in problems):
        caught.append('A missed the planted stale mtime')

    # C, on the engine side of the project-named directory: move LAMMPS's
    # own number and the disagreement must name that run. The arm fails
    # twice over if the attribution is lost - an unread directory reports
    # nothing at all, so a detector that only ever looks for a problem would
    # pass. The counts are read as well as the text.
    with open(lammps_energy_file, 'w') as handler:
        handler.write('%.16f\n' % (rows[-1][1] / KCAL_TO_HARTREE * 1.5))
    problems, counts = audit_engine(work)
    if not any('demo_lammps_example' in line for line in problems):
        caught.append('C missed the planted LAMMPS engine change: %s'
                      % problems)
    with open(lammps_energy_file, 'w') as handler:
        handler.write('%.16f\n' % (rows[-1][1] / KCAL_TO_HARTREE))
    _problems, counts = audit_engine(work)
    if 'LAMMPS' not in counts['engines_matched']:
        caught.append('C lost the LAMMPS attribution when the planted '
                      'change was undone: %s' % counts['engines_matched'])


def _plant_section_optimize_corruption(work, opt_values, opt_record, caught):
    """Move the optimizer's own last row and require C2 to name it."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    # C2: move the optimizer's own last row and the engine disagreement must
    # be named, with the step it was recorded at.
    moved_row = list(opt_values)
    moved_row[-1] = (moved_row[-1][0] - 0.5,) + moved_row[-1][1:]
    with open(opt_record, 'w') as handler:
        handler.write(status_header() + '\n')
        for number, values in enumerate(moved_row, start=1):
            handler.write(status_line(number, values) + '\n')
    problems, _ = audit_optimize_engine(work)
    if not any('C2 RECORDED ENERGY NOT IN ENGINE FILES' in line
               for line in problems):
        caught.append('C2 missed the planted optimizer row change: %s'
                      % problems)
    with open(opt_record, 'w') as handler:
        handler.write(status_header() + '\n')
        for number, values in enumerate(opt_values, start=1):
            handler.write(status_line(number, values) + '\n')
    problems, _ = audit_optimize_engine(work)
    if problems:
        caught.append('C2 reported the restored leg: %s' % problems)


def _plant_section_cell_corruption(work, cell_values, cell_record, caught):
    """Move the last recorded volume, substitute an engine table, then drop the record."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    # D: the last step's volume moves in the file only, so the recorded
    # metrics, the printed verdict and the round log all disagree with it.
    moved = list(cell_values)
    moved[1] = moved[1][:10] + (moved[1][10] - 0.5,)
    with open(cell_record, 'w') as handler:
        handler.write(status_header(cell=True) + '\n')
        for number, values in enumerate(moved, start=1):
            handler.write(status_line(number, values) + '\n')
    problems, _ = audit_cell(work)
    for prefix in ('D MISMATCH', 'D VERDICT TEXT', 'D LOG LINE'):
        if not any(prefix in line for line in problems):
            caught.append('D missed the planted volume move (%s): %s'
                          % (prefix, problems))

    # D: an engine's own convergence table under the same file name is not
    # a framework run's record, whatever its rows say.
    with open(cell_record, 'w') as handler:
        handler.write(' Step           Energy            Shift'
                      '      RMSD Step       Max Step      RMSD Grad'
                      '       Max Grad\n')
        handler.write('-' * 100 + '\n')
        handler.write('    1     -76.85494948     -76.85494948'
                      '       0.000000       0.000000       0.042160'
                      '       0.076263\n')
    problems, _ = audit_cell(work)
    if not any('D ENGINE RECORD' in line for line in problems):
        caught.append('D missed the planted engine header: %s' % problems)

    # D: a run that declares the switch and leaves no record must be named,
    # not counted as a leg that took zero steps.
    os.remove(cell_record)
    problems, _ = audit_cell(work)
    if not any('D NO RECORD' in line for line in problems):
        caught.append('D missed the planted missing record: %s' % problems)


def _plant_section_engine(work, run_dir, rows, extras, caught):
    """Plant the engine exports for the A/B/C clean pass and return the LAMMPS file."""
    calcdir = os.path.join(run_dir, 'tmp_low_qm_task1_XTB')
    os.makedirs(calcdir)
    with open(os.path.join(calcdir, 'energy'), 'w') as handler:
        handler.write('%.16f\n' % rows[-1][1])
    calcdir = os.path.join(run_dir, 'tmp_high_qm_task1_OPENQP')
    os.makedirs(calcdir)
    with open(os.path.join(calcdir, 'energies'), 'w') as handler:
        for value in (rows[-1][1],) + extras:
            handler.write('%.16f\n' % value)

    # A second run whose calculation directory is named after its
    # `project` and not after its backend - which is how the backend has
    # named them since #354 - holding LAMMPS's own energy file in the unit
    # style the skill was configured with. The attribution below has to
    # come from what is inside the directory: read from the name,
    # `tmp_water25` is no engine at all, its kcal/mol row never gets
    # converted, and this run's own recorded energy is reported as a
    # disagreement between the framework and an engine that agreed with it.
    lammps_dir = os.path.join(work, 'examples', 'demo_lammps_md', 'yaml')
    os.makedirs(lammps_dir)
    with open(os.path.join(lammps_dir,
                           'demo_lammps_example.md.energies'),
              'w') as handler:
        handler.write('%s\n' % ' '.join(
            ['time', 'Epot', 'Ekin', 'Etot']))
        handler.write('%15.2f%28.16f%28.16f%28.16f\n' % rows[-1])
    calcdir = os.path.join(lammps_dir, 'tmp_water25')
    os.makedirs(calcdir)
    lammps_energy_file = os.path.join(calcdir, 'water25_force.energy')
    with open(lammps_energy_file, 'w') as handler:
        handler.write('%.16f\n' % (rows[-1][1] / KCAL_TO_HARTREE))

    # A third run whose engine export is xTB's `*.engrad`, in the layout the
    # program actually writes: the line that names the energy is followed by
    # a comment bar, not by the number. The directory holds no plain `energy`
    # file for this leg, so a reader that stops at the line after the header
    # finds nothing, the directory lands in `dirs_unread`, and the clean
    # fixture below reports fewer than three checked rows rather than
    # silently agreeing.
    xtb_dir = os.path.join(work, 'examples', 'demo_xtb_md', 'yaml')
    os.makedirs(xtb_dir)
    with open(os.path.join(xtb_dir, 'demo_xtb_example.md.energies'),
              'w') as handler:
        handler.write('%s\n' % ' '.join(
            ['time', 'Epot', 'Ekin', 'Etot']))
        handler.write('%15.2f%28.16f%28.16f%28.16f\n' % rows[-1])
    calcdir = os.path.join(xtb_dir, 'tmp_XTB')
    os.makedirs(calcdir)
    xtb_engrad_file = os.path.join(calcdir, 'demo.engrad')
    with open(xtb_engrad_file, 'w') as handler:
        handler.write('#\n# Number of atoms\n#\n         6\n'
                      '#\n# The current total energy in Eh\n#\n'
                      '     %.16f\n' % rows[-1][1])

    # A two-manifold SOC run publishes `energies_s` / `energies_t` and no
    # plain `energies`; the reader has to know both names or the directory
    # lands in `dirs_unread` (round tmp_20261110_093321, md_openqp_fssh_soc).
    soc_dir = os.path.join(work, 'examples', 'demo_openqp_soc', 'yaml')
    os.makedirs(soc_dir)
    with open(os.path.join(soc_dir, 'demo_soc_example.md.energies'),
              'w') as handler:
        handler.write('%s\n' % ' '.join(['time', 'Epot', 'Ekin', 'Etot']))
        handler.write('%15.2f%28.16f%28.16f%28.16f\n' % rows[-1])
    calcdir = os.path.join(soc_dir, 'tmp_OPENQP')
    os.makedirs(calcdir)
    for name in ('energies_s', 'energies_t'):
        with open(os.path.join(calcdir, name), 'w') as handler:
            handler.write('%.16f\n' % rows[-1][1])

    # A field-only step keeps a calculation directory but no frame record;
    # its directory is neither unread nor a run C should count.
    field_dir = os.path.join(work, 'examples', 'demo_field', 'yaml')
    os.makedirs(field_dir)
    calcdir = os.path.join(field_dir, 'tmp_all_field_task1_h2o')
    os.makedirs(calcdir)
    with open(os.path.join(calcdir, 'h2o_qmmm_field.in'), 'w') as handler:
        handler.write('# field probe\n')

    # A layered run keeps only orbitals in its high-level directory; the
    # record is an assembly, so that directory is expected to have no
    # readable energy and must not be reported as a missing reader.
    composite_dir = os.path.join(work, 'examples', 'demo_region_layer', 'yaml')
    os.makedirs(composite_dir)
    with open(os.path.join(composite_dir, 'demo_region_layer.json'), 'w') as handler:
        json.dump({'highlevel': [0]}, handler)
    with open(os.path.join(composite_dir, 'demo_region_layer.md.energies'),
              'w') as handler:
        handler.write('%s\n' % ' '.join(['time', 'Epot', 'Ekin', 'Etot']))
        handler.write('%15.2f%28.16f%28.16f%28.16f\n' % rows[-1])
    calcdir = os.path.join(composite_dir, 'tmp_high_qm_OPENQP')
    os.makedirs(calcdir)
    with open(os.path.join(calcdir, 'OPENQP.RasOrb'), 'w') as handler:
        handler.write('orbital\n')

    problems, counts = audit_records(work)
    if problems:
        caught.append('clean A reported: %s' % problems)
    problems, counts = audit_log(work)
    if problems:
        caught.append('clean B reported: %s' % problems)
    problems, counts = audit_engine(work)
    if problems:
        caught.append('clean C reported: %s' % problems)
    if counts['checked'] != 4 or counts['matched'] != 4:
        caught.append('clean C checked=%d matched=%d, expected 4 and 4'
                      % (counts['checked'], counts['matched']))
    # The project-named directory has to be read as the engine it holds,
    # not as a name: `tmp_water25` is LAMMPS, and its kcal/mol row has to
    # reach the recorded Hartree number through the variant table.
    if 'LAMMPS' not in counts['engines_matched']:
        caught.append('clean C attributed the project-named directory to '
                      '%s, not to LAMMPS' % (counts['engines_matched'],))
    if counts['dirs_unread']:
        caught.append('clean C could not read planted directories: %s'
                      % counts['dirs_unread'])
    if counts['reader_missing']:
        caught.append('clean C counted %d field/layer directories as unread'
                      % counts['reader_missing'])
    if counts['composite'] < 1:
        caught.append('clean C did not classify the layered record as '
                      'composite: %s' % counts)
    if counts['engine_extra']:
        caught.append('clean C found %d unrecorded engine values'
                      % counts['engine_extra'])
    # The registry itself, next to the attribution rule that consults it: a
    # reader added for one engine is only worth having if the tool says it
    # has it, because `dirs_unread` is otherwise a list of names with no way
    # to tell "this engine has no reader" from "this reader missed".
    if 'PYSCF' not in counts['readers_registered']:
        caught.append('clean C registered readers=%s, expected PYSCF among '
                      'them' % counts['readers_registered'])

    # A per-centre row is attributed only when the run's own
    # `term_outcome.json` says which term computed it, or when one engine
    # directory answers for everything. The flag is per run: a round that
    # wrote one record for another case must not make this one comparable.
    layer_dir = os.path.join(work, 'examples', 'demo_layers', 'yaml')
    os.makedirs(layer_dir)
    with open(os.path.join(layer_dir, 'demo_layers.md.1.energies'),
              'w') as handler:
        handler.write('%s\n' % ' '.join(['time', 'Epot', 'Ekin', 'Etot']))
        handler.write('%15.2f%28.16f%28.16f%28.16f\n'
                      % (0.0, -70.123, 0.0, -70.123))
    for name, filename in (('tmp_A_OPENQP', 'energies'),
                           ('tmp_B_XTB', 'energy')):
        calcdir = os.path.join(layer_dir, name)
        os.makedirs(calcdir)
        with open(os.path.join(calcdir, filename), 'w') as handler:
            handler.write('-70.123\n')
    problems, counts = audit_engine(work)
    if problems or counts['layer_unattributed'] != 1:
        caught.append('C without term_outcome: unattributed=%d problems=%s'
                      % (counts['layer_unattributed'], problems))
    with open(os.path.join(layer_dir, 'term_outcome.json'), 'w') as handler:
        handler.write('{}\n')
    problems, counts = audit_engine(work)
    if problems or counts['layer_unattributed'] != 0 \
            or counts['checked'] != 5 or counts['matched'] != 5:
        caught.append('C with term_outcome: checked=%d matched=%d '
                      'unattributed=%d problems=%s'
                      % (counts['checked'], counts['matched'],
                         counts['layer_unattributed'], problems))

    problems, counts = audit_optimize_engine(work)
    # The variable-cell leg section D judges has no engine directory of its
    # own in this fixture, which is the same shape a real cp2k leg has: the
    # row is counted as a leg, and reported as having nothing to compare
    # against rather than silently dropped.
    if counts['legs'] != 4 or counts['checked'] != 0 \
            or counts['no_engine'] != 4:
        caught.append('clean C2 before the planted leg: legs=%d checked=%d '
                      'no_engine=%d, expected 4, 0 and 4 (%s)'
                      % (counts['legs'], counts['checked'],
                         counts['no_engine'], problems))
    return lammps_energy_file


def _plant_section_optimize(work, rows, caught):
    """Plant the C2 legs: the optimizer record, its engine export, and the composite."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    opt_dir = os.path.join(work, 'examples', 'demo_opt', 'yaml')
    os.makedirs(opt_dir)
    opt_record = os.path.join(opt_dir, 'opt_status.txt')
    opt_values = [(-77.06211556898168, 0.0, 0.015, 0.004, 0.05, 0.013),
                  (rows[-1][1], -0.00096567, 0.015, 0.0038, 0.049, 0.012)]
    with open(opt_record, 'w') as handler:
        handler.write(status_header() + '\n')
        for number, values in enumerate(opt_values, start=1):
            handler.write(status_line(number, values) + '\n')
    calcdir = os.path.join(opt_dir, 'tmp_demo25')
    os.makedirs(calcdir)
    opt_engine_file = os.path.join(calcdir, 'demo25_force.energy')
    with open(opt_engine_file, 'w') as handler:
        handler.write('%.16f\n' % (rows[-1][1] / KCAL_TO_HARTREE))

    pyscf_dir = os.path.join(work, 'examples', 'demo_pyscf', 'yaml')
    os.makedirs(pyscf_dir)
    pyscf_scf = rows[-1][1]
    pyscf_record = os.path.join(pyscf_dir, 'opt_status.txt')
    with open(pyscf_record, 'w') as handler:
        handler.write(status_header() + '\n')
        for number, values in enumerate(
                [(-77.06211556898168, 0.0, 0.015, 0.004, 0.05, 0.013),
                 (pyscf_scf, -0.00096567, 0.015, 0.0038, 0.049, 0.012)],
                start=1):
            handler.write(status_line(number, values) + '\n')
    pyscf_calcdir = os.path.join(pyscf_dir, 'tmp_PYSCF')
    os.makedirs(pyscf_calcdir)
    pyscf_log = os.path.join(pyscf_calcdir, 'PYSCF.log')

    def _pyscf_log(total, earlier=None):
        lines = []
        if earlier is not None:
            lines.append('converged SCF energy = %.16f' % earlier)
            lines.append('')
        lines.append('converged SCF energy = %.16f' % total)
        lines.append('')
        lines.append('**** SCF Summaries ****')
        lines.append('Total Energy =                         %.16f'
                     % total)
        lines.append('Nuclear Repulsion Energy =             32.7978892106')
        return '\n'.join(lines) + '\n'

    with open(pyscf_log, 'w') as handler:
        handler.write(_pyscf_log(pyscf_scf, pyscf_scf - 0.5))
    with open(os.path.join(pyscf_calcdir, 'PYSCF_tda_gradient.chk'),
              'w') as handler:
        handler.write('MAP\n')
    problems, counts = audit_optimize_engine(work)
    if problems:
        caught.append('clean C2 reported: %s' % problems)
    if counts['legs'] != 6 or counts['checked'] != 2 \
            or counts['matched'] != 2:
        caught.append('clean C2 legs=%d checked=%d matched=%d, expected 6, '
                      '2 and 2 (a section that reads nothing also reports '
                      'nothing)'
                      % (counts['legs'], counts['checked'],
                         counts['matched']))
    comp_dir = os.path.join(work, 'examples', 'demo_composite', 'yaml')
    os.makedirs(comp_dir)
    with open(os.path.join(comp_dir, 'demo_composite.json'), 'w') as handler:
        json.dump({'theory': ['xtb'], 'highlevel': [[1, 2]]}, handler)
    with open(os.path.join(comp_dir, 'opt_status.txt'), 'w') as handler:
        handler.write(status_header() + '\n')
        for number, values in enumerate(opt_values, start=1):
            handler.write(status_line(number, values) + '\n')
    for term, offset in (('real', 0.5), ('model', 1.5)):
        term_dir = os.path.join(comp_dir, 'tmp_demo%s' % term)
        os.makedirs(term_dir)
        with open(os.path.join(term_dir, 'demo%s_force.energy' % term),
                  'w') as handler:
            handler.write('%.16f\n'
                          % ((rows[-1][1] + offset) / KCAL_TO_HARTREE))
    problems, counts = audit_optimize_engine(work)
    if problems:
        caught.append('C2 reported the composite leg it should leave '
                      'unjudged: %s' % problems)
    if counts['legs'] != 7 or counts['composite'] != 1 \
            or counts['checked'] != 2 or counts['matched'] != 2:
        caught.append('C2 with the composite leg: legs=%d composite=%d '
                      'checked=%d matched=%d, expected 7, 1, 2 and 2'
                      % (counts['legs'], counts['composite'],
                         counts['checked'], counts['matched']))
    if not _leg_declares_a_region(comp_dir):
        caught.append('the region reader missed the leg it was written for')
    with open(pyscf_log, 'w') as handler:
        handler.write(_pyscf_log(pyscf_scf - 0.5, pyscf_scf))
    _problems, _counts = audit_optimize_engine(work)
    if not any('demo_pyscf' in line for line in _problems):
        caught.append('C2 read the first SCF total of the PySCF log: %s'
                      % _problems)
    with open(pyscf_log, 'w') as handler:
        handler.write(_pyscf_log(pyscf_scf, pyscf_scf - 0.5))
    _problems, _counts = audit_optimize_engine(work)
    if any('demo_pyscf' in line for line in _problems):
        caught.append('C2 reported the restored PySCF leg: %s' % _problems)
    return opt_dir, opt_record, opt_values


def _plant_section_cell(work, pressure_dir, pressure_log, pressure_spec,
                       quiet_volumes, caught):
    """Run the cell section over the planted legs and its three refusals."""
    from pyrai3md.utilities.opt_formatter import (
        status_header, status_line)
    problems, cell_counts = audit_cell(work)
    if problems:
        caught.append('clean D reported: %s' % problems)
    if cell_counts['pressure_legs'] != 1 \
            or cell_counts['pressure_disagreements'] \
            or cell_counts['pressure_missing_rows']:
        caught.append('clean D pressure: legs=%d disagreements=%d '
                      'missing=%d, expected 1, 0 and 0 (%s)'
                      % (cell_counts['pressure_legs'],
                         cell_counts['pressure_disagreements'],
                         cell_counts['pressure_missing_rows'], problems))
    if cell_counts['legs'] != 2 or cell_counts['log_tasks'] != 1:
        caught.append('clean D legs=%d log_tasks=%d, expected 1 and 1 '
                      '(a section that reads nothing also reports nothing)'
                      % (cell_counts['legs'], cell_counts['log_tasks']))
    if cell_counts['shape_legs'] != 2 \
            or cell_counts['shape_source_disagreements'] \
            or cell_counts['shape_unreadable']:
        caught.append('clean D shape: legs=%d disagreements=%d '
                      'unreadable=%d, expected 2, 0 and 0 (%s)'
                      % (cell_counts['shape_legs'],
                         cell_counts['shape_source_disagreements'],
                         cell_counts['shape_unreadable'], problems))
    with open(pressure_log, 'w') as h:
        h.write('  &optimization progress\n%s\n'
                % '-------------------------------------------------------------------------------')
        for key, value in (('External Pressure', 10000.0),
                           ('Cell Stress RMS', 9972.6),
                           ('Cell Stress Max', 9972.6),
                           ('Cell Strain RMS', 6e-05),
                           ('Cell Strain Max', 6e-05)):
            h.write('  %-18s %20.8f\n' % (key + ':', value))
        h.write('  %-18s %20s\n' % ('Converged:', 'True'))
    _problems, cell_counts = audit_cell(work)
    if not cell_counts['pressure_disagreements']:
        caught.append('D missed the planted residual move in the log: %s'
                      % _problems)

    with open(os.path.join(work, 'results.json'), encoding='utf-8') as h:
        document = json.load(h)
    for entry in document['example_results']:
        if entry['name'] == 'example_demo_cell_yaml':
            metrics = entry['cell_metrics']
            shape_key = 'shape_ratio_change'
            original_shape = metrics[shape_key]
            metrics[shape_key] = original_shape * 2.0 + 1.0e-4
    with open(os.path.join(work, 'results.json'), 'w',
              encoding='utf-8') as h:
        json.dump(document, h)
    _problems, cell_counts = audit_cell(work)
    if not cell_counts['shape_source_disagreements']:
        caught.append('D missed the planted shape record: %s' % _problems)
    metrics[shape_key] = original_shape
    with open(os.path.join(work, 'results.json'), 'w',
              encoding='utf-8') as h:
        json.dump(document, h)
    _problems, cell_counts = audit_cell(work)
    if cell_counts['shape_source_disagreements']:
        caught.append('D still reports the shape record it was handed back: '
                      '%s' % _problems)

    twin_status = os.path.join(work, 'examples', 'demo_quiet', 'yaml',
                               'opt_status.txt')
    with open(twin_status, 'w') as h:
        h.write(status_header(cell=True) + '\n')
        for number, volume in enumerate([15774.53498224576, 15800.0],
                                        start=1):
            h.write(status_line(
                number, (-8.0 - number, 0.0, 0.001, 0.0001, 0.001, 0.0001,
                         0.024, 0.034, 0.0027, 0.0063, volume)) + '\n')
    tampered = {'status': 'pass'}
    check_cell_optimization(tampered, pressure_spec, pressure_dir,
                            'demo_pressure')
    if "failed to oppose the box's own motion" not in (
            tampered.get('details') or ''):
        caught.append('the pairing missed a twin that drifted less than the '
                      'loaded leg: %r' % tampered.get('details'))
    with open(twin_status, 'w') as h:
        h.write(status_header(cell=True) + '\n')
        for number, volume in enumerate(quiet_volumes, start=1):
            h.write(status_line(
                number, (-8.0 - number, 0.0, 0.001, 0.0001, 0.001, 0.0001,
                         0.024, 0.034, 0.0027, 0.0063, volume)) + '\n')
    restored = {'status': 'pass'}
    check_cell_optimization(restored, pressure_spec, pressure_dir,
                            'demo_pressure')
    if 'paired against demo_quiet' not in (restored.get('details') or '') \
            or 'failed to oppose' in (restored.get('details') or ''):
        caught.append('the pairing stayed broken after the twin was '
                      'restored: %r' % restored.get('details'))


def selftest():
    """Negative control: the detectors must fire on deliberately wrong data.

    Builds a throwaway round directory whose record, log and results
    document agree, checks that all three sections pass it, then corrupts
    one number per section and checks that each corruption is caught. An
    audit that cannot see a planted error is not an audit.
    """
    root = tempfile.mkdtemp(prefix='verify_selftest_')
    work = os.path.join(root, 'tmp_20260901_120000')
    try:
        run_dir = os.path.join(work, 'examples', 'demo_md', 'yaml')
        os.makedirs(run_dir)
        title = 'demo_md_example'
        rows = [(0.00, -77.0621155689816817, 0.0064875623414381,
                 -77.0556280066402479),
                (20.67, -77.0612541112045051, 0.0056330479583983,
                 -77.0556210632461074)]
        # Two per-state columns beyond the documented four, the shape a
        # multicentre record writes and an engine export reports against.
        extras = (-70.1234567890123451, -70.9876543210987654)
        record = os.path.join(run_dir, '%s.md.energies' % title)

        _selftest_write_record(record, extras, rows)
        log = os.path.join(run_dir, '%s.log' % title)
        from pyrai3md.utilities import print_results
        with open(log, 'w') as handler:
            for number, row in enumerate(rows, start=1):
                handler.write('*  Step %10d\n' % number)
                handler.write(print_results._section(
                    '&electronic state', '  Current state: 1\n'))
                handler.write(print_results._section(
                    SECTION_ENERGIES, '  State %3d%28.16f\n' % (1, row[1])))
                handler.write(print_results._section(
                    SECTION_KINETIC, '  %28.16f\n' % row[2]))
        metrics = energy_conservation_metrics(read_md_energies(record))

        # A variable-cell leg, judged by the runner itself so the numbers this
        # section audits come from the code that produces them on a real
        # round, not from a sentence hand-written to match the detector.
        caught = []
        from pyrai3md.utilities.opt_formatter import status_header, status_line
        cell_dir = os.path.join(work, 'examples', 'demo_cell', 'yaml')
        os.makedirs(cell_dir)
        cell_values = [
            (-8.265022160551094, 0.0, 1.501118817262213e-02,
             3.835617010285512e-03, 5.003729390874045e-02,
             1.278539003428504e-02, 1.248755444438083e-01,
             1.904484168349469e-01, 3.746266333314250e-02,
             5.713452505048406e-02, 1.572157528502303e+04),
            (-8.491232162482934, -2.262100019318396e-01,
             1.647853489545187e-02, 7.272109642528405e-03,
             3.917252720234554e-02, 9.687255464370834e-03,
             7.707718959128584e-02, 1.165111185916476e-01,
             4.566013301001057e-03, 6.653772242961771e-03,
             1.577453498224576e+04)]
        cell_record = os.path.join(cell_dir, 'opt_status.txt')
        with open(cell_record, 'w') as handler:
            handler.write(status_header(cell=True) + '\n')
            for number, values in enumerate(cell_values, start=1):
                handler.write(status_line(number, values) + '\n')
        cell_spec = os.path.join(cell_dir, 'demo_cell.yaml')
        with open(cell_spec, 'w') as handler:
            handler.write('control:\n  title: demo_cell\n  jobtype: optimize\n'
                          '  theory: [xtb]\n'
                          'optimize:\n  method: minima\n'
                          '  optimize_cell: true\n')

        # The trajectory the shape reading comes from. `[COUPLING]` and section
        # D's second source both read `opt_geom.xyz`, not the status record -
        # the record has no shape column - so both are planted here, two frames
        # whose row-length product is the volume the record carries for the same
        # step. The first row length differs between the frames, which is what
        # an anisotropic box does and what the floor separates it from.
        def _geometry(directory, volumes, row_x=(25.0, 25.02)):
            with open(os.path.join(directory, 'opt_geom.xyz'), 'w') as h:
                for index, volume in enumerate(volumes):
                    length = row_x[min(index, len(row_x) - 1)]
                    h.write('1\n')
                    h.write('Lattice="%.9f 0.0 0.0 0.0 24.500000000 0.0 '
                            '0.0 0.0 %.9f" Properties=species:S:1:pos:R:3\n'
                            % (length, volume / (length * 24.5)))
                    h.write('H 0.0 0.0 0.0\n')

        _geometry(cell_dir, [values[-1] for values in cell_values])
        verdict = {'status': 'pass'}
        check_cell_optimization(verdict, cell_spec, cell_dir, 'demo_cell')
        cell_details = (verdict.get('details') or '').strip()
        if 'cell columns present' not in cell_details:
            caught.append('fixture D: the runner did not judge the planted '
                          'record: %r' % cell_details)
        with open(os.path.join(work, 'test_20260901_120000.log'), 'w') as h:
            h.write('  [100/190]  + example_demo_cell_yaml '
                    '[pyrai3md_yaml][lammps] -> PASS\n')
            h.write('            %s\n' % cell_details)

        # A leg under an applied pressure, for the clause `opt_pressure` now
        # carries: the same two products the runner reads (its status record and
        # its optimization-progress block), and the metrics document it wrote.
        # The progress block is the only product that holds the residual of
        # `sigma + p I` in bar, so the comparison below is results.json against
        # that block - what catches a number transcribed between the two.
        pressure_dir = os.path.join(work, 'examples', 'demo_pressure', 'yaml')
        os.makedirs(pressure_dir)
        with open(os.path.join(pressure_dir, 'opt_status.txt'), 'w') as h:
            h.write(status_header(cell=True) + '\n')
            for number, values in enumerate(reversed(cell_values), start=1):
                h.write(status_line(number, values) + '\n')
        pressure_log = os.path.join(pressure_dir, 'demo_pressure.log')
        with open(pressure_log, 'w') as h:
            h.write('  &optimization progress\n%s\n' % ('-' * 79))
            for key, value in (('External Pressure', 10000.0),
                               ('Cell Stress RMS', 12.0),
                               ('Cell Stress Max', 20.0),
                               ('Cell Strain RMS', 6.0e-5),
                               ('Cell Strain Max', 6.0e-5)):
                h.write('  %-18s %20.8f\n' % (key + ':', value))
            h.write('  %-18s %20s\n' % ('Converged:', 'True'))
        pressure_spec = os.path.join(pressure_dir, 'demo_pressure.yaml')
        with open(pressure_spec, 'w') as h:
            h.write('control:\n  title: demo_pressure\n  jobtype: optimize\n'
                    '  theory: [xtb]\n'
                    'optimize:\n  method: minima\n'
                    '  optimize_cell: true\n'
                    '  opt_pressure: 10000.0\n  opt_csrms: 40.0\n'
                    '  opt_csmax: 60.0\n')
        _geometry(pressure_dir,
                  [values[-1] for values in reversed(cell_values)])

        # The zero-pressure twin the direction clause pairs against, and a
        # decoy on another engine. The twin drifted 105.47 A^3 where the loaded
        # leg drifted 52.96, which is a load opposing the box; the decoy shares
        # optimizer, coupling and starting volume but is scored by cp2k, so it
        # must not be read as a second reference. Round 17's products paired the
        # anisotropic arm this way (`optimize_cell_aniso`, 17.087668 A^3 against
        # a loaded 15.730721) and could not pair the isotropic one at all.

        quiet_volumes = [15774.53498224576, 15880.0]
        _selftest_flat_leg(os.path.join(work, 'examples', 'demo_quiet', 'yaml'),
                  'demo_quiet', quiet_volumes, ['xtb'])
        # A different drift on purpose: an engine guard that stopped working
        # then leaves the clause with two references under two names, which is
        # the failure the plan wants visible rather than a silent merge.
        _selftest_flat_leg(os.path.join(work, 'examples', 'demo_quiet', 'slurm'),
                  'demo_quiet', [15774.53498224576, 15900.0], ['cp2k'])
        pressure_verdict = {'status': 'pass'}
        check_cell_optimization(pressure_verdict, pressure_spec,
                                pressure_dir, 'demo_pressure')
        pressure_details = (pressure_verdict.get('details') or '').strip()
        if 'pressure 10000.0 bar' not in pressure_details:
            caught.append('fixture D: the runner did not judge the planted '
                          'pressure leg: %r' % pressure_details)
        # The pairing the fixture exists to exercise: the loaded leg has to name
        # its engine twin as the reference, and the cp2k arm of the same box has
        # to be refused as one. Both halves are read, because a scan that simply
        # found no twin would also print "paired against nothing".
        if 'paired against demo_quiet' not in pressure_details:
            caught.append('fixture D: the pressure leg did not pair with its '
                          'zero-pressure twin: %r' % pressure_details)
        if (pressure_verdict['cell_metrics'].get('pressure_reference_example')
                != 'demo_quiet'):
            caught.append('fixture D: the recorded reference is %r, expected '
                          'demo_quiet'
                          % pressure_verdict['cell_metrics'].get(
                              'pressure_reference_example'))

        document = {
            'status': 'pass',
            'main_results': [],
            'skill_results': [],
            'benchmark_results': [],
            'example_results': [{
                'name': 'example_demo_pressure_yaml',
                'status': 'pass',
                'theory': 'xtb',
                'details': pressure_details,
                'cell_metrics': pressure_verdict['cell_metrics'],
            }, {
                'name': 'example_demo_md_yaml',
                'status': 'pass',
                'metrics': {'metrics': metrics, 'tolerances': {}}
            }, {
                'name': 'example_demo_cell_yaml',
                'status': 'pass',
                'theory': 'lammps',
                'details': cell_details,
                'cell_metrics': verdict['cell_metrics'],
            }]
        }
        with open(os.path.join(work, 'results.json'), 'w') as handler:
            json.dump(document, handler)

        # Surviving calculation directories holding the last step's numbers:
        # one single-value export and one per-state export.
        lammps_energy_file = _plant_section_engine(work, run_dir, rows, extras, caught)
        opt_dir, opt_record, opt_values = _plant_section_optimize(work, rows, caught)
        _plant_section_cell(work, pressure_dir, pressure_log, pressure_spec, quiet_volumes, caught)
        _plant_section_record_corruption(work, rows, record, extras, lammps_energy_file, caught)
        _plant_section_optimize_corruption(work, opt_values, opt_record, caught)
        _plant_section_cell_corruption(work, cell_values, cell_record, caught)
        _plant_section_cp2k(work, opt_dir, opt_values, rows, caught)
        _plant_section_molcas(work, opt_values, rows, caught)
        _plant_section_casscf(root, caught)
        _plant_section_pressure_scope(work, opt_values, rows, caught)
        _plant_section_state_scope(work, opt_values, rows, caught)
        _plant_section_budget(work, opt_dir, caught)
        _plant_section_carriers(work, rows, caught)
        _plant_section_path(work, caught)

        _plant_section_cell_wiring(root, work, caught)

        _plant_section_molcas_leg_verdict(caught)

        _plant_section_molcas_contract(caught)

        _plant_section_grad_identity(work, caught)

        _plant_section_skill_carriers(root, caught)

        return caught
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _adaptive_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as handler:
            data = json.load(handler)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _adaptive_frames(path):
    """Frame count of an adaptive `.xyz` product, or an `.init.xyz` file."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
            lines = handler.read().splitlines()
    except OSError:
        return None
    frames = 0
    index = 0
    while index < len(lines):
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            break
        try:
            natom = int(lines[index].strip())
        except ValueError:
            return None
        index += 1
        if index < len(lines):
            index += 1
        index += natom
        frames += 1
    return frames


def _adaptive_dataset_frames(path):
    data = _adaptive_json(path)
    if data is None:
        return None
    if isinstance(data.get('xyz'), list):
        return len(data['xyz'])
    try:
        return int(data.get('nframe'))
    except (TypeError, ValueError):
        return None


def _adaptive_sidecar_has(path, channel):
    """Whether every frame in the aux sidecar carries one channel."""
    import json

    try:
        with open(path, 'r', encoding='utf-8') as handle:
            frames = (json.load(handle) or {}).get('frames') or []
    except (OSError, ValueError):
        return False
    return bool(frames) and all(channel in frame for frame in frames)


def _adaptive_numeric_finite(path):
    """Whether a training-data JSON carries only finite energy/gradient values."""
    data = _adaptive_json(path)
    if data is None:
        return False
    for key in ('energy', 'grad'):
        value = data.get(key)
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, (int, float)):
                if not math.isfinite(float(item)):
                    return False
    return True


def _adaptive_model_complete(modeldir):
    base = os.path.join(modeldir, 'energy_grad')
    if not os.path.isdir(base):
        return False
    for pattern in ('hyper_v*.json', 'weights_v*.weights.h5',
                    'scaler_v*.json', 'fiterr_valid_v*.npy'):
        if not glob.glob(os.path.join(base, pattern)):
            return False
    return True


def _adaptive_declared(run_dir):
    """(maxiter, maxsample) the run's input declares, or (None, None).

    The adaptive examples do not all use the same shape: the analysis
    carriers run four rounds of two trajectories, the older carriers two
    rounds of five. The audit reads the declaration rather than assuming
    one example's numbers; an input it cannot read leaves the shape
    unchecked instead of inventing a failure.
    """
    import yaml as _yaml

    for path in sorted(glob.glob(os.path.join(run_dir, '*.yaml'))):
        try:
            document = _yaml.safe_load(open(path, encoding='utf-8'))
        except (OSError, _yaml.YAMLError):
            continue
        adaptive = (document or {}).get('adaptive')
        if not isinstance(adaptive, dict):
            continue
        md = (document or {}).get('md') or {}
        try:
            return (int(adaptive.get('maxiter')),
                    int(md.get('ninitcond')))
        except (TypeError, ValueError):
            continue
    return None, None


def audit_adaptive(work_dir):
    """Section J: adaptive sampling's two-round product tree.

    The runner judges one carrier by the summary and the second
    iteration. This audit adds the consistency readings the workflow
    itself does not state: five trajectories in both rounds, the label
    count equal to the new dataset frames, the model folder servable,
    the method-alignment record present, and the two carriers of one
    case agreeing on every structural count. It deliberately does not
    compare the reference energies between carriers: those are
    calculations, not a serialization, and the QM label may carry a
    different SCF path.
    """
    problems = []
    counts = {'cases': 0, 'carriers': 0, 'missing_summary': 0,
              'bad_rounds': 0, 'bad_ntraj': 0, 'no_reference': 0,
              'no_training': 0, 'no_data': 0, 'no_model': 0,
              'missing_alignment': 0, 'missing_log': 0,
              'missing_aux': 0, 'unexpected_aux': 0, 'not_servable': 0,
              'count_mismatch': 0,
              'carrier_mismatch': 0, 'reference_failed': 0,
              'unverified': 0, 'nonfinite': 0}
    examples = os.path.join(work_dir, 'examples')
    if not os.path.isdir(examples):
        return [('J NO EXAMPLES: %s is missing' % examples)], counts
    runs = []
    for case_dir in sorted(glob.glob(os.path.join(examples, 'adaptive_*',
                                                  '*'))):
        if not os.path.isdir(case_dir):
            continue
        for root in sorted(glob.glob(os.path.join(case_dir, '*.adaptive'))):
            title = os.path.basename(root)[:-len('.adaptive')]
            runs.append((case_dir, title))
    by_case = {}
    for run_dir, title in runs:
        case = os.path.basename(os.path.dirname(run_dir))
        mode = os.path.basename(run_dir)
        counts['carriers'] += 1
        root = os.path.join(run_dir, '%s.adaptive' % title)
        summary_path = os.path.join(root, '%s.adaptive.json' % title)
        summary = _adaptive_json(summary_path)
        if summary is None:
            counts['missing_summary'] += 1
            problems.append('J NO SUMMARY: %s/%s' % (case, mode))
            continue
        by_case.setdefault(case, {})[mode] = {
            'summary': summary, 'root': root, 'run_dir': run_dir}
        declared_rounds, declared_traj = _adaptive_declared(run_dir)
        n_iterations = int(summary.get('n_iterations') or 0)
        iter_dirs = [path for path in glob.glob(
            os.path.join(root, 'iter-*')) if os.path.isdir(path)]
        if (n_iterations <= 0 or n_iterations != len(iter_dirs)
                or (declared_rounds is not None
                    and n_iterations > declared_rounds)):
            counts['bad_rounds'] += 1
            problems.append(
                'J ROUNDS: %s/%s n_iterations=%s dirs=%d declared=%s'
                % (case, mode, summary.get('n_iterations'), len(iter_dirs),
                   declared_rounds))
            # Without a usable round count the per-round checks have no
            # shape to test; the carrier comparison still runs.
            n_iterations = 0
        statuses = {}
        for iteration in range(1, n_iterations + 1):
            status_path = os.path.join(
                root, 'iter-%d' % iteration,
                'status_iter-%d.json' % iteration)
            status = _adaptive_json(status_path)
            if status is None:
                counts['bad_rounds'] += 1
                problems.append('J NO STATUS: %s/%s iter-%d'
                                % (case, mode, iteration))
                continue
            statuses[iteration] = status
            ntraj = int(status.get('ntraj') or 0)
            if declared_traj is not None and ntraj != declared_traj:
                counts['bad_ntraj'] += 1
                problems.append(
                    'J NTRAJ: %s/%s iter-%d ntraj=%d declared=%d'
                    % (case, mode, iteration, ntraj, declared_traj))
            selected = int(status.get('selected') or 0)
            selected_path = os.path.join(
                root, 'iter-%d' % iteration,
                'selected_iter-%d.xyz' % iteration)
            frames = _adaptive_frames(selected_path)
            if frames is None or frames != selected:
                counts['count_mismatch'] += 1
                problems.append(
                    'J SELECTED: %s/%s iter-%d status=%d xyz=%s'
                    % (case, mode, iteration, selected, frames))
            population = _adaptive_json(os.path.join(
                root, 'iter-%d' % iteration,
                'population_iter-%d.json' % iteration))
            trajectories = (population or {}).get('trajectories')
            if (not isinstance(trajectories, list)
                    or len(trajectories) != int(status.get('ntraj') or 0)):
                counts['count_mismatch'] += 1
                problems.append('J POPULATION: %s/%s iter-%d'
                                % (case, mode, iteration))
        if not statuses:
            continue
        last = max(statuses)
        # A single-round run has no reference/training stage: the workflow
        # only labels after a completed production round. Such a case is
        # audited for its round shape and log, not for a stage it never had.
        if last <= 1:
            continue
        iter2 = statuses.get(last) or {}
        if not iter2.get('reference_ran'):
            counts['no_reference'] += 1
            problems.append('J REFERENCE: %s/%s did not run' % (case, mode))
        if int(iter2.get('reference_failed') or 0) != 0:
            counts['reference_failed'] += 1
            problems.append('J REFERENCE FAILED: %s/%s count=%s'
                            % (case, mode, iter2.get('reference_failed')))
        received = int(iter2.get('received') or 0)
        verified = int(iter2.get('verified') or 0)
        if not (received == verified == int(iter2.get('new_frames') or 0)):
            counts['unverified'] += 1
            problems.append('J VERIFIED: %s/%s received=%d verified=%d '
                            'new=%d' % (case, mode, received, verified,
                                        int(iter2.get('new_frames') or 0)))
        if not iter2.get('trained'):
            counts['no_training'] += 1
            problems.append('J TRAINING: %s/%s did not run' % (case, mode))
        new_frames = int(iter2.get('new_frames') or 0)
        data_files = sorted(glob.glob(os.path.join(
            root, 'iter-%d' % last, 'data_iter-%d_*.json' % last)))
        if not data_files:
            counts['no_data'] += 1
            problems.append('J DATA: %s/%s has no iter-%d snapshot'
                            % (case, mode, last))
            data_frames = None
        else:
            data_frames = _adaptive_dataset_frames(data_files[-1])
        samples = _adaptive_dataset_frames(os.path.join(
            root, 'iter-%d' % last, 'reference', 'new-samples.json'))
        if samples != new_frames:
            counts['count_mismatch'] += 1
            problems.append('J LABELS: %s/%s status=%d new-samples=%s'
                            % (case, mode, new_frames, samples))
        for path in data_files[-1:] + [
                os.path.join(root, 'iter-%d' % last, 'reference',
                             'new-samples.json')]:
            if os.path.isfile(path) and not _adaptive_numeric_finite(path):
                counts['nonfinite'] += 1
                problems.append('J NONFINITE: %s/%s %s'
                                % (case, mode, os.path.basename(path)))
        if data_frames is not None and samples is not None:
            initial = _adaptive_dataset_frames(os.path.join(
                run_dir, 'egs_ethylene_100.json'))
            cumulative = sum(int(status.get('new_frames') or 0)
                             for status in statuses.values())
            if (initial is not None
                    and data_frames != initial + cumulative):
                counts['count_mismatch'] += 1
                problems.append(
                    'J DATASET: %s/%s frames=%d initial=%d cumulative=%d'
                    % (case, mode, data_frames, initial, cumulative))
        from pyrai3md.skills.nn_pes_skill import query as nn_query
        model = os.path.join(
            root, 'iter-%d' % last, nn_query.model_dir_name(title, 'iter-%d' % last))
        if not os.path.isdir(model):
            counts['no_model'] += 1
            problems.append('J MODEL: %s/%s missing' % (case, mode))
        elif not _adaptive_model_complete(model):
            counts['not_servable'] += 1
            problems.append('J MODEL FILES: %s/%s incomplete'
                            % (case, mode))
        references = glob.glob(os.path.join(
            root, 'iter-%d' % last, 'reference', 'ref_*'))
        if len(references) != new_frames:
            counts['count_mismatch'] += 1
            problems.append('J REF DIRS: %s/%s dirs=%d labels=%d'
                            % (case, mode, len(references), new_frames))
        alignment = summary.get('method_alignment')
        if not isinstance(alignment, dict) or not alignment:
            counts['missing_alignment'] += 1
            problems.append('J ALIGNMENT: %s/%s missing' % (case, mode))
        else:
            # The plan's own aux list is the expectation; what an audit
            # can check is that the product agrees with it -- every
            # channel the plan named is in the sidecar, and a plan naming
            # none leaves no sidecar behind (#4.7).
            aux_channels = [str(value)
                            for value in (alignment.get('aux') or [])]
            sidecar = os.path.join(root, 'iter-%d' % last, 'reference',
                                   'aux-labels.json')
            has_sidecar = os.path.isfile(sidecar)
            if bool(aux_channels) != has_sidecar:
                counts['missing_aux'] += 1
                problems.append('J AUX: %s/%s aux=%s sidecar=%s'
                                % (case, mode, aux_channels, has_sidecar))
            elif has_sidecar:
                for channel in aux_channels:
                    if not _adaptive_sidecar_has(sidecar, channel):
                        counts['missing_aux'] += 1
                        problems.append('J AUX: %s/%s sidecar lacks %s'
                                        % (case, mode, channel))
                        break
        log_path = os.path.join(root, '%s.log' % title)
        try:
            with open(log_path, 'r', encoding='utf-8',
                      errors='ignore') as handler:
                log_text = handler.read()
        except OSError:
            log_text = ''
        for marker in ('&method alignment', '&adaptive sampling',
                       '&adaptive sampling summary'):
            if marker not in log_text:
                counts['missing_log'] += 1
                problems.append('J LOG: %s/%s missing %r'
                                % (case, mode, marker))
                break
        # The iteration directory has a known product set; a new artifact
        # is reported rather than silently accepted, but the analysis and
        # model files the newer examples add are part of that set.
        iteration_dir = os.path.join(root, 'iter-%d' % iteration)
        allowed = {
            'status_iter-%d.json' % iteration,
            'selected_iter-%d.xyz' % iteration,
            'population_iter-%d.json' % iteration,
            'analysis_iter-%d.json' % iteration,
        }
        for name in sorted(os.listdir(iteration_dir)):
            path = os.path.join(iteration_dir, name)
            if os.path.isdir(path):
                # `prune/` holds the robust-z gate's trial datasets and
                # their trial model directories (#486).
                if (name in ('md', 'reference', 'prune')
                        or name.startswith('NN-')):
                    continue
            elif (name in allowed or name.endswith('.log')
                  or (name.startswith('data_iter-%d_' % iteration)
                      and name.endswith('.json'))):
                continue
            counts['unexpected_aux'] += 1
            problems.append('J AUX FILE: %s/%s iter-%d %s'
                            % (case, mode, iteration, name))
    for case, modes in sorted(by_case.items()):
        counts['cases'] += 1
        if len(modes) < 2:
            continue
        keys = ('n_iterations',)
        left, right = modes.get('yaml'), modes.get('input')
        if left is None or right is None:
            continue
        for key in keys:
            if left['summary'].get(key) != right['summary'].get(key):
                counts['carrier_mismatch'] += 1
                problems.append('J CARRIER: %s %s yaml=%r input=%r'
                                % (case, key, left['summary'].get(key),
                                   right['summary'].get(key)))
        for side in (left, right):
            side['alignment'] = side['summary'].get('method_alignment')
        if left['alignment'] != right['alignment']:
            counts['carrier_mismatch'] += 1
            problems.append('J CARRIER ALIGNMENT: %s' % case)
    return problems, counts


# Section K reads one skill case's three carriers through files only: the
# engine deck each carrier wrote and the final energy each engine printed.
# No skill or engine is invoked. A separate result tree per case is why the
# comparison is by `<carrier>/tmp/tmp_<stem>_test` and not by title alone.
SKILL_CARRIER_MODES = ('skill_agent', 'pyrai3md_agent', 'pyrai3md_input')
SKILL_CARRIER_SUFFIX = {'skill_agent': '_skill', 'pyrai3md_agent': '_agent',
                        'pyrai3md_input': '_input'}
SKILL_ENGINE_DECKS = {
    'orca_skill': ('*.inp',),
    'cp2k_skill': ('*.inp',),
    'molcas_skill': ('*.inp',),
    'openqp_skill': ('*.inp',),
    'xtb_skill': ('*.inp',),
    'lammps_skill': ('*.in.init', '*.in.settings', '*.in'),
    'bagel_skill': ('*.json',),
}


def _skill_carrier_cases(work_dir):
    """The skill cases a round ran and passed on all three carriers."""
    path = os.path.join(work_dir, 'results.json')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handler:
            entries = json.load(handler).get('skill_results', [])
    except (OSError, ValueError):
        return {}
    groups = {}
    for entry in entries:
        mode = entry.get('mode')
        suffix = SKILL_CARRIER_SUFFIX.get(mode)
        name = str(entry.get('name') or '')
        if suffix is None or not name.endswith(suffix):
            continue
        case = name[:-len(suffix)]
        if '_skill_' not in case:
            continue
        skill = case.split('_skill_')[0] + '_skill'
        group = groups.setdefault((skill, case), {})
        group[mode] = entry
    return {key: modes for key, modes in groups.items()
            if all(mode in modes for mode in SKILL_CARRIER_MODES)
            and all(modes[mode].get('status') == 'pass'
                    for mode in SKILL_CARRIER_MODES)}


def _skill_carrier_bases(skill_dir, mode):
    """Every directory holding one carrier mode's legs.

    A grouped round puts each case group between the skill and the leg
    (`<skill>/<group>/<mode>`); earlier rounds used `<skill>/<mode>`.
    Both are enumerated so either work directory audits.
    """
    bases = sorted(path for path in glob.glob(
        os.path.join(skill_dir, '*', mode)) if os.path.isdir(path))
    legacy = os.path.join(skill_dir, mode)
    if os.path.isdir(legacy):
        bases.append(legacy)
    return bases


def _skill_carrier_dirs(base, stem):
    """The calculation directories of one carrier, keyed by directory name."""
    root = os.path.join(base, 'tmp')
    exact = os.path.join(root, 'tmp_%s_test' % stem)
    if os.path.isdir(exact):
        return {os.path.basename(exact): exact}
    return {os.path.basename(path): path
            for path in glob.glob(os.path.join(root, 'tmp_*'))
            if os.path.isdir(path) and stem in os.path.basename(path)}


def _normalized_deck(path):
    """Deck text with the per-carrier paths the planner writes masked out."""
    try:
        with open(path, 'rb') as handler:
            raw = handler.read()
    except OSError:
        return None
    if b'\x00' in raw:
        return None
    text = raw.decode('utf-8', errors='replace')
    for mode in SKILL_CARRIER_MODES:
        text = text.replace('/%s/' % mode, '/<CARRIER>/')
    return re.sub(r'tmp_[A-Za-z0-9_.-]+?_test', 'tmp_<CALCDIR>', text)


def _skill_final_energy(calcdir, skill):
    """The final energy one carrier's engine printed, or None.

    pyscf, qdm and nn_pes run in process and keep no engine file; molcas is
    section G's subject; bagel prints no single final-energy line. The other
    five engines each write one number per run, which is what this reads.
    """
    def _last_float(path, pattern=None):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
                text = handler.read()
        except OSError:
            return None
        if pattern is not None:
            found = pattern.findall(text)
            if not found:
                return None
            text = found[-1]
        for line in reversed(text.splitlines()):
            try:
                return float(line.split()[-1])
            except (ValueError, IndexError):
                continue
        return None

    def _first_float(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as handler:
                for line in handler:
                    try:
                        return float(line.split()[0])
                    except (ValueError, IndexError):
                        continue
        except OSError:
            return None
        return None

    if skill == 'openqp_skill':
        for name in ('energies', 'energies_s', 'energies_t'):
            value = _first_float(os.path.join(calcdir, name))
            if value is not None:
                return value
        return None
    if skill == 'xtb_skill':
        value = _first_float(os.path.join(calcdir, 'energy'))
        if value is not None:
            return value
        files = glob.glob(os.path.join(calcdir, '*.out'))
        pattern = re.compile(r'TOTAL ENERGY\s+(-?\d+\.\d+)')
        return _last_float(files[0], pattern) if files else None
    if skill == 'lammps_skill':
        files = glob.glob(os.path.join(calcdir, '*.energy'))
        return _last_float(files[0]) if files else None
    if skill == 'orca_skill':
        files = glob.glob(os.path.join(calcdir, '*.out'))
        pattern = re.compile(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)')
        return _last_float(files[0], pattern) if files else None
    if skill == 'cp2k_skill':
        files = glob.glob(os.path.join(calcdir, '*.log'))
        pattern = re.compile(
            r'ENERGY\| Total FORCE_EVAL \( QS \) energy \[hartree\]\s*'
            r'(-?\d+\.\d+)')
        return _last_float(files[0], pattern) if files else None
    return None


def audit_skill_carriers(work_dir):
    """Section K: one skill case, three carriers, one engine answer.

    A skill round runs every case through three entry points and keeps each
    leg's calculation directory. Sections C and G judge a leg against its own
    record; nothing compared the engine input and the final engine number
    across the three carriers. Round tmp_20261110_093321 is the measured
    reason: OpenQP `tddft_gradient` wrote `grad=2,3` on the skill leg and
    `grad=1` on both framework legs, every leg passed its own status, and
    the difference was visible only in the three decks.

    K reads files only. For every case that passed on all three carriers it
    compares the engine deck(s) the carrier wrote, with the per-carrier paths
    the planner injects masked out, and the final energy each engine printed
    (where the engine writes one). It never imports a skill or starts a
    calculation.
    """
    problems = []
    counts = {'cases': 0, 'dirs': 0, 'dirs_unpaired': 0,
              'deck_files': 0, 'deck_divergences': 0,
              'decks_unpaired': 0, 'energy_compared': 0,
              'energy_divergences': 0, 'energy_unpaired': 0,
              'no_engine': 0}
    cases = _skill_carrier_cases(work_dir)
    for (skill, case), _modes in sorted(cases.items()):
        counts['cases'] += 1
        stem = case[len(skill) + 1:]
        skill_dir = os.path.join(work_dir, skill)
        dirs = {}
        for mode in SKILL_CARRIER_MODES:
            found = {}
            for base in _skill_carrier_bases(skill_dir, mode):
                found.update(_skill_carrier_dirs(base, stem))
            dirs[mode] = found
        shared = set.intersection(*(set(dirs[mode]) for mode in
                                    SKILL_CARRIER_MODES))
        counts['dirs_unpaired'] += len(
            set.union(*(set(dirs[mode]) for mode in SKILL_CARRIER_MODES))
            - shared)
        for name in sorted(shared):
            counts['dirs'] += 1
            paths = {mode: dirs[mode][name] for mode in SKILL_CARRIER_MODES}

            patterns = SKILL_ENGINE_DECKS.get(skill)
            if patterns:
                found = set()
                for mode in SKILL_CARRIER_MODES:
                    for pattern in patterns:
                        found.update(
                            os.path.basename(path) for path in
                            glob.glob(os.path.join(paths[mode], pattern))
                            if os.path.isfile(path))
                for filename in sorted(found):
                    files = {mode: os.path.join(paths[mode], filename)
                             for mode in SKILL_CARRIER_MODES}
                    present = [mode for mode in SKILL_CARRIER_MODES
                               if os.path.isfile(files[mode])]
                    if len(present) < len(SKILL_CARRIER_MODES):
                        counts['decks_unpaired'] += 1
                        problems.append(
                            'K DECK UNPAIRED: %s %s: %s exists in %s only'
                            % (case, name, filename, ', '.join(present)))
                        continue
                    texts = {mode: _normalized_deck(files[mode])
                             for mode in SKILL_CARRIER_MODES}
                    if any(text is None for text in texts.values()):
                        continue
                    counts['deck_files'] += 1
                    if len(set(texts.values())) > 1:
                        counts['deck_divergences'] += 1
                        problems.append(
                            'K DECK DIVERGENCE: %s %s: %s differs across '
                            'carriers' % (case, name, filename))

            values = {}
            for mode in SKILL_CARRIER_MODES:
                value = _skill_final_energy(paths[mode], skill)
                if value is not None:
                    values[mode] = value
            if not values:
                counts['no_engine'] += 1
                continue
            if len(values) < len(SKILL_CARRIER_MODES):
                counts['energy_unpaired'] += 1
                missing = [mode for mode in SKILL_CARRIER_MODES
                           if mode not in values]
                problems.append(
                    'K ENERGY MISSING: %s %s: no final energy from %s'
                    % (case, name, ', '.join(missing)))
                continue
            counts['energy_compared'] += 1
            worst = max(values.values()) - min(values.values())
            limit = max(1e-6, 1e-9 * max(abs(value)
                                         for value in values.values()))
            if worst > limit:
                counts['energy_divergences'] += 1
                problems.append(
                    'K ENERGY DIVERGENCE: %s %s: %s'
                    % (case, name, ' '.join(
                        '%s=%.12g' % item for item in sorted(values.items()))))
    return problems, counts


def main(argv):
    """Run the requested sections over one work directory."""
    if '--selftest' in argv:
        caught = selftest()
        for line in caught:
            print('SELFTEST FAIL: %s' % line)
        print('selftest: %s' % ('PASSED' if not caught
                                else 'FAILED (%d)' % len(caught)))
        return 0 if not caught else 1

    if len(argv) < 2:
        print('usage: verify.py <work_dir> [--examples <dir>]')
        return 2
    work_dir = os.path.abspath(argv[1])
    examples_dir = ''
    if '--examples' in argv:
        examples_dir = os.path.abspath(argv[argv.index('--examples') + 1])
    if not os.path.isdir(work_dir):
        print('not a directory: %s' % work_dir)
        return 2

    failures = 0
    for label, function in (('A records vs results.json',
                             lambda: audit_records(work_dir, examples_dir)),
                            ('B step log vs record',
                             lambda: audit_log(work_dir)),
                            ('C engine output vs record',
                             lambda: audit_engine(work_dir)),
                            ('C2 engine output vs optimize record',
                             lambda: audit_optimize_engine(work_dir)),
                            ('C3 optimizer term columns vs the energy column',
                             lambda: audit_term_decomposition(work_dir)),
                            ('D cell record vs verdict vs round log',
                             lambda: audit_cell(work_dir)),
                            ('E carrier parity: one example, two carriers',
                             lambda: audit_carrier_parity(work_dir)),
                            ('F path: held conditions and the composite surface',
                             lambda: audit_path(work_dir)),
                            ('G molcas legs: three carriers, one answer',
                             lambda: audit_molcas_legs(work_dir)),
                            ('H qmmm_field A/B Ewald split and V',
                             lambda: audit_field_split(work_dir)),
                            ('I qmmm_energy grad against force and auxiliaries',
                             lambda: audit_grad_identity(work_dir)),
                            ('J adaptive: two rounds, five trajectories, '
                             'labels, model and carrier parity',
                             lambda: audit_adaptive(work_dir)),
                            ('K skill carriers: engine deck and final energy',
                             lambda: audit_skill_carriers(work_dir))):
        problems, counts = function()
        print('== %s' % label)
        for key in sorted(counts):
            print('   %-16s %s' % (key, counts[key]))
        for line in problems[:30]:
            print('   %s' % line)
        if len(problems) > 30:
            print('   (%d problem lines total)' % len(problems))
        failures += len(problems)

    print('== not covered by this tool')
    for line in NOT_COVERED:
        print('   - %s' % line)
    print('== verdict: %s' % ('clean' if not failures else
                              '%d problems' % failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
