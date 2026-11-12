"""
======================================================================
This module provides the vibrational initial-condition sampling of
the md workflow: it turns a frequency/normal-mode record into an
ensemble of starting points (coordinates + velocities) consumed by
the multi-trajectory fan-out (workflow/initcond.py).

Sampling distributions:
- 'wigner'      Wigner sampling of the quantized harmonic oscillator
                (vibrational level populations times the Wigner
                quasi-probability rejection step)
- 'boltzmann'   classical Boltzmann sampling of the harmonic
                oscillator (Gaussian position/momenta widths)

Frequency data readers (one per &md format value, all returning the
same freqdata dict):
- 'molden'      <title>.freq.molden   (Molcas; modes already
                                       unnormalized unmass-weighted)
- 'g16'         <title>.freq.g16[+.freq.fchk] (Gaussian; normalized
                 unmass-weighted modes converted in place)
- 'orca'        <title>.freq.orca     (ORCA .hess layout; normalized
                                       unmass-weighted modes)
- 'bagel'       <title>.freq.bagel    (BAGEL log; normalized
                                       mass-weighted modes)

freqdata dict contract:
    nfreq   number of vibrational degrees kept
    freqs   frequencies in cm-1, signed (a negative value is an
            imaginary mode), shape (nfreq, 1)
    natom   number of atoms
    atoms   element symbols, shape (natom,)
    xyz     equilibrium coordinates in Bohr, shape (natom, 3)
    vib     normal modes in Bohr, shape (nfreq, natom, 3),
            unnormalized and unmass-weighted
    amass   atomic masses in g/mol, shape (natom, 1)

The module is a pure-function service: it never writes files and
never touches the global random state; every random draw goes
through the np.random.Generator handed in by the caller (the md
workflow seeds it from the control gl_seed). Ported from the legacy
PyRAI2MD Utils/sampling.py; the unit-conversion chain now reads the
framework's single set of constants (utilities/units.py) instead of a
second rounded copy of it.

Aug 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import os

import numpy as np

from ..kernel.logger import get_logger
from ..schema.reference import ATOMIC_MASS

LOGGER = get_logger('pyrai3md.sampling')

# ---- Unit conversions ----
# The framework's one set of constants (utilities/units.py). This module
# used to keep a second, rounded set "verbatim" from the legacy chain:
# its Bohr factor 0.529177249 is 7.199e-08 relative away from
# 0.52917721090299996, its amu factor 1.802e-08 away, its cm-1 factor
# 1.580e-08 away, and its cm-1-per-Kelvin 0.69503 is 6.907e-06 away from
# the quotient of the two constants it is made of. Every sampled frame
# therefore differed from the one the same code would produce with the
# shared constants, in the last digits and, at a level boundary, in the
# sampled quantum number.
from ..utilities.units import (AMU_TO_AU, BOHR_TO_ANGSTROM,
                              BOLTZMANN_TO_HARTREE_PER_K, CM_TO_HARTREE)

CM_TO_KELVIN = BOLTZMANN_TO_HARTREE_PER_K / CM_TO_HARTREE

# Maximal vibrational quantum number sampled by the Wigner level
# population draw (legacy guard against pathological temperatures).
MAX_WIGNER_LEVEL = 150

# Cumulative level population cutoff of the Wigner state draw.
WIGNER_POPULATION_CUTOFF = 0.9999


# ---- Frequency file naming (legacy convention) ----
# The readers resolve '<title>.freq.<format>' in the current
# working directory, the same cwd contract as <title>.init.xyz.
FREQ_SUFFIX = {
    'molden': '.freq.molden',
    'g16': '.freq.g16',
    'orca': '.freq.orca',
    'bagel': '.freq.bagel',
}


def atomic_mass(symbol):
    """Return the atomic mass (g/mol) of an element symbol.

    Parameters
    ----------
    symbol : str
        Element symbol as printed by the frequency file (any case).

    Returns
    -------
    float
        Atomic mass in g/mol.

    Raises
    ------
    ValueError
        If the symbol is not in the lookup table.
    """
    key = str(symbol).strip().capitalize()
    if key not in ATOMIC_MASS:
        raise ValueError("Unknown element symbol: %r" % (symbol,))
    return ATOMIC_MASS[key]


def _require_file(path):
    """Raise FileNotFoundError with the sampling context."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Frequency file not found: %s" % path)
    return path


def read_molden(path):
    """Parse a `.freq.molden` file.

    Molcas prints unnormalized unmass-weighted normal modes, so the
    modes are used as read. Frequencies are returned **signed**, with a
    warning for each imaginary one.

    The sign is the contract, and it is what makes this reader usable by
    the IRC walk: `write_freq_molden` writes the imaginary mode with its
    negative frequency, and a reader that took the absolute value left the
    two halves of one file disagreeing about the same mode. Every consumer
    that wants a magnitude takes it itself; `sample_initial_conditions` is
    the only one, and it says so where it does.

    Parameters
    ----------
    path : str
        Path of the `.freq.molden` file.

    Returns
    -------
    dict
        The freqdata dict (see the module header contract).
    """
    path = _require_file(path)
    with open(path) as handler:
        lines = handler.read().splitlines()

    freqs = inten = rmass = None
    atoms = xyz = None
    natom = nmode = 0
    modes = []
    for index, line in enumerate(lines):
        if '[N_FREQ]' in line:
            nmode = int(lines[index + 1])
        elif '[FREQ]' in line:
            freqs = np.array([float(value) for value in
                              lines[index + 1: index + 1 + nmode]])
            for value in freqs:
                if value < 0:
                    LOGGER.warning(
                        'Imaginary frequency: %f', value)
        elif '[INT]' in line:
            inten = np.array([float(value) for value in
                              lines[index + 1: index + 1 + nmode]])
        elif '[NATOM]' in line:
            natom = int(lines[index + 1])
        elif '[FR-COORD]' in line:
            block = [line.split() for line in
                     lines[index + 1: index + 1 + natom]]
            atoms = np.array([row[0] for row in block])
            xyz = np.array([[float(v) for v in row[1:4]]
                            for row in block])
        elif ' vibration' in line:
            vib = np.array([[float(v) for v in row.split()] for row in
                            lines[index + 1: index + 1 + natom]])
            modes.append(vib)
        elif '[RMASS]' in line:
            rmass = np.array([float(value) for value in
                              lines[index + 1: index + 1 + nmode]])
    del inten, rmass

    if freqs is None or atoms is None or not modes:
        raise ValueError("Malformed molden frequency file: %s" % path)
    amass = np.array([atomic_mass(symbol) for symbol in atoms])
    return {
        'nfreq': freqs.shape[0],
        'freqs': freqs.reshape((-1, 1)),
        'natom': natom,
        'atoms': atoms,
        'xyz': xyz,
        'vib': np.array(modes).reshape((nmode, natom, 3)),
        'amass': amass.reshape((-1, 1)),
    }


def _g16_flat(lines):
    """Flatten whitespace-separated tokens into one float array."""
    values = []
    for line in lines:
        values.extend(float(token) for token in line.split())
    return np.array(values)


def _g16_mode_blocks(vect_lines, natom):
    """Collect Gaussian 'Atom AN' mode columns into mode arrays."""
    modes = []
    for block in vect_lines:
        rows = np.array([[float(v) for v in line.split()[2:]]
                         for line in block])
        ncolumn = int(rows.shape[1] / 3)
        for column in range(ncolumn):
            modes.append(rows[:, column: column + 3])
    return np.array(modes)


def _unmass_weight_normalized(modes, amass):
    """Convert normalized unmass-weighted modes to unnormalized.

    Gaussian/ORCA print modes normalized to one in the plain
    cartesian norm; the sampling formulas expect unnormalized
    unmass-weighted modes (the Molcas convention).

    Parameters
    ----------
    modes : np.ndarray
        Mode vectors, shape (nmode, natom, 3).
    amass : np.ndarray
        Atomic masses, shape (natom, 1).

    Returns
    -------
    np.ndarray
        Unnormalized unmass-weighted modes, same shape.
    """
    amass = amass.reshape((-1, 1))
    return np.array(
        [mode / np.linalg.norm(mode * amass ** 0.5)
         for mode in modes])


def read_g16(path):
    """Parse a Gaussian .freq.g16 log file.

    Gaussian prints normalized unmass-weighted normal modes; they
    are converted to the unnormalized unmass-weighted convention.

    Parameters
    ----------
    path : str
        Path of the .freq.g16 file.

    Returns
    -------
    dict
        The freqdata dict (see the module header contract).
    """
    path = _require_file(path)
    with open(path) as handler:
        log = handler.read().splitlines()

    natom = 0
    cart_lines = []
    freq_tokens = []
    vect_blocks = []
    for index, line in enumerate(log):
        if 'Stoichiometry' in line:
            formula = line.split()[-1]
            digits = ''
            for char in formula + 'X':
                if char.isdigit():
                    digits += char
                else:
                    if digits:
                        natom += int(digits)
                        digits = ''
        if 'Input orientation' in line:
            cart_lines = log[index + 5: index + 5 + natom]
        if 'Frequencies -- ' in line:
            freq_tokens.append(line.split(' -- ')[-1])
        if ' Atom  AN ' in line:
            vect_blocks.append(log[index + 1: index + 1 + natom])

    if not cart_lines or not freq_tokens:
        raise ValueError("Malformed Gaussian frequency file: %s"
                         % path)
    atoms = []
    xyz = []
    for line in cart_lines:
        _, number, _, x, y, z = line.split()
        atoms.append(int(number))
        xyz.append([float(x), float(y), float(z)])
    symbols = {number: symbol for symbol, number in
               [(s, i + 1) for i, s in enumerate(
                   'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar '
                   'K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se '
                   'Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn '
                   'Sb Te I Xe Cs Ba'.split())]}
    atoms = np.array([symbols[number] for number in atoms])
    xyz = np.array(xyz)
    freqs = _g16_flat(freq_tokens).reshape((-1, 1))
    modes = _g16_mode_blocks(vect_blocks, natom)
    amass = np.array([atomic_mass(symbol) for symbol in atoms])
    amass = amass.reshape((-1, 1))
    modes = _unmass_weight_normalized(modes, amass)
    return {
        'nfreq': freqs.shape[0],
        'freqs': freqs,
        'natom': natom,
        'atoms': atoms,
        'xyz': xyz,
        'vib': modes.reshape((freqs.shape[0], natom, 3)),
        'amass': amass,
    }


def read_orca(path):
    """Parse an ORCA .hess layout file (.freq.orca).

    ORCA prints normalized unmass-weighted normal modes; imaginary
    frequencies and their modes are dropped with a warning (legacy
    behavior).

    Parameters
    ----------
    path : str
        Path of the .freq.orca file.

    Returns
    -------
    dict
        The freqdata dict (see the module header contract).
    """
    path = _require_file(path)
    with open(path) as handler:
        hess = handler.read().splitlines()

    atoms = xyz = None
    freqs = inten = None
    modes = None
    for index, line in enumerate(hess):
        if '$vibrational_frequencies' in line:
            nrow = int(hess[index + 1])
            block = hess[index + 2: index + 2 + nrow]
            freqs = np.array([[float(v) for v in row.split()]
                              for row in block])[:, 1]
        elif '$ir_spectrum' in line:
            nrow = int(hess[index + 1])
            block = hess[index + 2: index + 2 + nrow]
            inten = np.array([[float(v) for v in row.split()]
                              for row in block])[:, 1]
        elif '$atoms' in line:
            natom = int(hess[index + 1])
            block = [row.split() for row in
                     hess[index + 2: index + 2 + natom]]
            atoms = np.array([row[0] for row in block])
            xyz = np.array([[float(v) for v in row[2:5]]
                            for row in block])
        elif '$normal_modes' in line:
            nmode = int(hess[index + 1].split()[0])
            nline = (nmode + 1) * (int(nmode / 5)
                                   + (nmode % 5 > 0))
            vects = hess[index + 2: index + 2 + nline]
            modes = [[] for _ in range(nmode)]
            for row_index, row in enumerate(vects):
                slot = row_index % (nmode + 1) - 1
                if slot >= 0:
                    modes[slot] += [float(v) for v in
                                    row.split()[1:]]
            # The hess layout stores the transpose of the mode
            # matrix (legacy 'Transpose array' convention).
            modes = np.array(modes).T.reshape(
                (nmode, int(nmode / 3), 3))
    del inten

    if freqs is None or atoms is None or modes is None:
        raise ValueError("Malformed ORCA frequency file: %s" % path)
    natom = len(atoms)
    # Drop imaginary frequencies and their modes.
    kept = []
    for position, value in enumerate(freqs):
        if value > 0:
            kept.append(position)
        elif value < 0:
            LOGGER.warning(
                'Imaginary frequency: %10f mode: %6d (Ignored)',
                value, position + 1)
    freqs = freqs[kept].reshape((-1, 1))
    modes = modes[kept]
    amass = np.array([atomic_mass(symbol) for symbol in atoms])
    amass = amass.reshape((-1, 1))
    modes = _unmass_weight_normalized(modes, amass)
    return {
        'nfreq': freqs.shape[0],
        'freqs': freqs,
        'natom': natom,
        'atoms': atoms,
        'xyz': xyz,
        'vib': modes,
        'amass': amass,
    }


def read_bagel(path):
    """Parse a BAGEL log file (.freq.bagel).

    BAGEL prints normalized mass-weighted normal modes; they are
    demass-weighted into the unnormalized unmass-weighted
    convention. Imaginary frequencies and their modes are dropped
    with a warning (legacy behavior).

    Parameters
    ----------
    path : str
        Path of the .freq.bagel file.

    Returns
    -------
    dict
        The freqdata dict (see the module header contract).
    """
    path = _require_file(path)
    with open(path) as handler:
        log = handler.read().splitlines()

    atoms = []
    xyz = []
    nmode = 0
    vect_lines = []
    freq_values = []
    for index, line in enumerate(log):
        if 'Freq (cm-1)' in line:
            freq_values += [float(v) for v in line.split()[2:]]
        if '"atom"' in line:
            natom = len(atoms) + 1
            nmode = 3 * natom
            tokens = line.replace(',', ' ').replace(
                '"', ' ').split()
            atoms.append(tokens[3])
            xyz.append([float(v) for v in tokens[7:10]])
        if '++ Mass Weighted Hessian Eigenvectors ++' in line:
            nline = (nmode + 2) * (int(nmode / 6)
                                   + (nmode % 6 > 0))
            vect_lines = log[index + 1: index + 1 + nline]

    if not freq_values or not atoms:
        raise ValueError("Malformed BAGEL frequency file: %s" % path)
    natom = len(atoms)
    modes = [[] for _ in range(nmode)]
    for row_index, line in enumerate(vect_lines):
        slot = row_index % (nmode + 2) - 2
        if row_index % (nmode + 2) > 1:
            modes[slot] += [float(v) for v in line.split()[1:]]
    # The BAGEL layout stores the transpose of the mode matrix
    # (legacy 'Transpose array' convention).
    modes = np.array(modes).T.reshape((nmode, natom, 3))

    atoms = np.array(atoms)
    xyz = np.array(xyz)
    freqs = np.array(freq_values)
    kept = []
    for position, value in enumerate(freqs):
        if value > 0:
            kept.append(position)
        elif value < 0:
            LOGGER.warning(
                'Imaginary frequency: %10f mode: %6d (Ignored)',
                value, position + 1)
    freqs = freqs[kept].reshape((-1, 1))
    modes = modes[kept]
    amass = np.array([atomic_mass(symbol) for symbol in atoms])
    amass = amass.reshape((-1, 1))
    # Demass-weight the mass-weighted normalized modes.
    modes = np.array([mode / amass ** 0.5 for mode in modes])
    return {
        'nfreq': freqs.shape[0],
        'freqs': freqs,
        'natom': natom,
        'atoms': atoms,
        'xyz': xyz,
        'vib': modes,
        'amass': amass,
    }


FREQ_READERS = {
    'molden': read_molden,
    'g16': read_g16,
    'orca': read_orca,
    'bagel': read_bagel,
}


def load_frequency_data(title, fmt):
    """Load the frequency record '<title>.freq.<format>' from cwd.

    Parameters
    ----------
    title : str
        Run title (control section).
    fmt : str
        Frequency file format: molden, g16, orca, bagel.

    Returns
    -------
    dict
        The freqdata dict (see the module header contract).

    Raises
    ------
    ValueError
        If the format is not recognized.
    FileNotFoundError
        If the frequency file does not exist in the cwd.
    """
    fmt = str(fmt).strip().lower()
    if fmt not in FREQ_READERS:
        raise ValueError(
            "Unrecognized frequency file format: %r (expected one "
            "of %s)" % (fmt, sorted(FREQ_READERS)))
    path = '%s%s' % (title, FREQ_SUFFIX[fmt])
    return FREQ_READERS[fmt](path)


# ---- Sampling kernels ----

def _laguerre(order, x):
    """Evaluate the Laguerre polynomial L_n(x) (n <= 150).

    L_n(x) = sum_m (-1)^m n(n-1)...(n-m+1)/(m!)^2 x^m; the factor
    recurrence avoids the factorial overflow of the direct form.
    """
    value = 0.0
    factor = 1.0
    for step in range(order + 1):
        if step > 0:
            factor *= float(order - step + 1) / step ** 2
        value += (-1) ** step * factor * x ** step
    return value


def _pick_level(cumulative, target):
    """Inverse-CDF walk over the cumulative level populations."""
    for level, edge in enumerate(cumulative):
        if target <= edge:
            if level > MAX_WIGNER_LEVEL:
                LOGGER.warning(
                    'Sampled vibrational state is higher than %d, '
                    'adjusted to %d', MAX_WIGNER_LEVEL,
                    MAX_WIGNER_LEVEL)
                level = MAX_WIGNER_LEVEL
            return level
    return len(cumulative) - 1


def wigner_kern(freqs, temp, rng):
    """Wigner-sample one (Q, P) pair per vibrational mode.

    For each mode a thermal vibrational level is drawn, then the
    dimensionless coordinate Q and momentum P are rejection-sampled
    against the Wigner quasi-probability of that level
    (W(Q,P) = (-1)^n L_n(2(Q^2+P^2)) exp(-(Q^2+P^2)), the SHARC
    wigner.py scheme).

    Unlike the legacy wignerfunc, the level is drawn once per
    sample: the legacy outer retry redrew the level on every
    rejection, biasing the level mixture by the per-level
    acceptance rate; drawing once keeps the Boltzmann weights
    exact.

    Parameters
    ----------
    freqs : np.ndarray
        Frequencies in cm-1, shape (nfreq,) or (nfreq, 1).
    temp : float
        Temperature in K.
    rng : np.random.Generator
        Random number generator owned by the caller.

    Returns
    -------
    np.ndarray
        (Q, P) pairs, shape (nfreq, 2).
    """
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    q_p = np.zeros((freqs.shape[0], 2))
    for row, freq_cm in enumerate(freqs):
        ex = freq_cm / (CM_TO_KELVIN * temp)
        cumulative = []
        population = 0.0
        level = -1
        while True:
            level += 1
            population += float(np.exp(-ex * level)
                                * (1 - np.exp(-ex)))
            cumulative.append(population)
            if population >= WIGNER_POPULATION_CUTOFF:
                break
        target = rng.uniform(0, population)
        level = _pick_level(cumulative, target)
        while True:
            q = rng.uniform(0, 1) * 10.0 - 5.0
            p = rng.uniform(0, 1) * 10.0 - 5.0
            rho2 = 2 * (q ** 2 + p ** 2)
            w = (-1) ** level * _laguerre(level, rho2) \
                * np.exp(-0.5 * rho2)
            if rng.uniform(0, 1) < w < 1:
                break
        q_p[row] = (q, p)
    return q_p


def boltzmann_kern(freqs, temp, rng):
    """Boltzmann-sample one (Q, P) pair per vibrational mode.

    Classical limit of the harmonic oscillator: both the
    dimensionless coordinate and momentum are standard-normal
    variates (the sigma scaling is applied by the assembly step).

    Parameters
    ----------
    freqs : np.ndarray
        Frequencies in cm-1, shape (nfreq,) or (nfreq, 1).
    temp : float
        Temperature in K (kept for interface symmetry; the
        classical widths are scaled in the assembly step).
    rng : np.random.Generator
        Random number generator owned by the caller.

    Returns
    -------
    np.ndarray
        (Q, P) pairs, shape (nfreq, 2).
    """
    del temp
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    return rng.standard_normal((freqs.shape[0], 2))


# ---- Assembly ----

def sample_initial_conditions(freqdata, nsample, method='wigner',
                              temp=298.15, seed=None):
    """Sample an ensemble of starting points from a frequency record.

    Parameters
    ----------
    freqdata : dict
        Frequency record from one of the readers (see the module
        header contract). Frequencies are signed, and an imaginary one
        is sampled as a vibration of the same magnitude with a warning:
        a thermal distribution has no imaginary mode.
    nsample : int
        Number of initial conditions to draw.
    method : str
        Sampling distribution: 'wigner' or 'boltzmann'.
    temp : float
        Temperature in K.
    seed : int or None
        Seed of the sampling generator (None = unseeded).

    Returns
    -------
    list of dict
        One {'coordinates', 'velocities'} descriptor per sample;
        coordinates in Angstrom, velocities in Bohr/au, each with
        shape (natom, 3).

    Raises
    ------
    ValueError
        If the method is unrecognized or nsample < 1.

    Notes
    -----
    Unit-conversion chain (legacy, kept verbatim):

    Wigner: dimensionless Q/P of the harmonic oscillator level n.
    Position: q = Q / sqrt(mu * mu_to_hartree * ma_to_amu) [Bohr],
        the mode sum R = R0 + sum(q * mode) is unmass-weighted by
        construction of the unnormalized unmass-weighted modes.
    Velocity: v = P * sqrt(mu * mu_to_hartree / ma_to_amu)
        [Bohr/au], summed over the modes.

    Boltzmann (classical): the same dimensionless Q and P, each drawn with the
        thermal variance kT/(hbar w), through the one conversion below.

    Both branches go through `_mode_scales`, which holds the mass conversion.
    The modes are normalized so sum(m x^2) = 1 (amu), so dividing the summed
    displacement by a per-atom sqrt(mass) as well -- what this used to do, on a
    chain inherited from mode lists that were not normalized -- scaled every
    atom's motion by 1/sqrt(mass) and left the classical branch's kinetic
    energy a factor of a few below kT/2, with the potential far below that.
    """
    method = str(method).strip().lower()
    if method not in ('wigner', 'boltzmann'):
        raise ValueError("Unrecognized sampling method: %r"
                         % (method,))
    nsample = int(nsample)
    if nsample < 1:
        raise ValueError(
            "nsample must be >= 1, got %d" % nsample)

    freqs = np.asarray(freqdata['freqs'],
                       dtype=np.float64).reshape(-1)
    nfreq = freqs.shape[0]
    xyz = np.asarray(freqdata['xyz'], dtype=np.float64)
    vib = np.asarray(freqdata['vib'], dtype=np.float64)
    amass = np.asarray(freqdata['amass'],
                       dtype=np.float64).reshape((-1, 1))
    if nfreq < 1:
        raise ValueError(
            "The frequency record carries no vibrational mode")

    imaginary = int(np.count_nonzero(freqs <= 0.0))
    if imaginary:
        # The magnitude is taken here, not in the readers: a thermal
        # distribution has no imaginary mode, and a record taken from a
        # transition state is sampled as if that mode were a vibration of the
        # same magnitude - which is what this sampler has always done, but the
        # sign now survives as far as the caller instead of being hidden by the
        # reader that parsed it (`read_molden`). The alternative - refusing the
        # record by name - would take away the only route to an initial
        # condition at a saddle geometry, so it is reported instead.
        LOGGER.warning(
            '%d imaginary mode(s) in this frequency record are sampled as '
            'vibrations of the same magnitude', imaginary)
        freqs = np.abs(freqs)

    rng = np.random.default_rng(seed)
    if method == 'wigner':
        kern = wigner_kern
    else:
        kern = boltzmann_kern

    position_scale, velocity_scale = _mode_scales(freqs)
    ensemble = []
    for _ in range(nsample):
        q_p = kern(freqs, temp, rng)
        q = q_p[:, 0].reshape((nfreq, 1))
        p = q_p[:, 1].reshape((nfreq, 1))
        if method != 'wigner':
            # Classical limit of the same dimensionless oscillator: both
            # quadratures carry the thermal variance kT / (hbar w), which is
            # what makes the drawn Q and P dimensionless in the same way the
            # Wigner ones are.
            sigma = np.sqrt(temp * BOLTZMANN_TO_HARTREE_PER_K
                            / (freqs.reshape((-1, 1)) * CM_TO_HARTREE))
            q = q * sigma
            p = p * sigma
        # One conversion for both branches. The classical branch used to
        # multiply by its own widths and then divide the summed result by
        # sqrt(mass) as well, a step inherited from a mode list that is not
        # normalized the way every reader here normalizes it (sum m x^2 = 1);
        # with that normalization it shrinks each atom's motion by the square
        # root of its own mass, and measured the kinetic energy of a single
        # 500 cm-1 mode at 298.15 K came out 6.7 times below kT/2 while the
        # potential fell three orders of magnitude short.
        qvib = np.sum(vib * (q * position_scale).reshape((nfreq, 1, 1)),
                      axis=0)
        newc = (xyz + qvib) * BOHR_TO_ANGSTROM
        velo = np.sum(vib * (p * velocity_scale).reshape((nfreq, 1, 1)),
                      axis=0)
        ensemble.append({'coordinates': newc,
                         'velocities': velo})
    return ensemble


def _mode_scales(freqs):
    """Bohr and Bohr/au per unit of one mode's dimensionless Q and P.

    The modes arrive normalized so that sum(m x^2) = 1 with the masses in amu,
    which puts the whole unit problem in these two factors: a mass-weighted
    coordinate of a mode of angular frequency w spreads over
    1 / sqrt(w_au * m_e->amu) in Bohr, and its momentum over the reciprocal.
    Both sampling branches draw that same dimensionless oscillator, so both
    have to land here; keeping the arithmetic in one place is the point.
    """
    omega = (np.asarray(freqs, dtype=np.float64).reshape(-1, 1)
             * CM_TO_HARTREE)
    return 1.0 / np.sqrt(omega * AMU_TO_AU), np.sqrt(omega / AMU_TO_AU)
