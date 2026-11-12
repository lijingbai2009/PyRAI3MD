"""
======================================================================
This is a standalone CP2K calculation skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Oct 10 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

This module holds the pure mathematics of the external strain
finite-difference stress route ('num'): the six independent strain
directions, the cell/coordinate deformation that keeps the fractional
coordinates unchanged, and the central-difference combination that
turns a +/- energy pair into a stress component.

No I/O and no CP2K dependency live here, so every convention below is
unit-testable on its own (dev/skills/CP2K_SKILL_DEVELOPMENT.md section
6.6, the external difference).

Conventions, stated once because each one is a factor-of-one-and-a-half
to two source of error:
- The lattice vectors are stored BY ROW: H = [a; b; c].
- Diagonal directions set B[a, a] = 1; shear directions set
  B[a, b] = B[b, a] = 1/2, which makes sigma : B equal sigma[a, b].
  Setting both shear slots to 1 would return 2 * sigma[a, b].
- The difference denominator uses the CENTRAL cell volume V0, never
  the perturbed volume.
- The external difference yields the tensile-positive potential stress
  directly; no sign flip is applied (CP2K's own printed tensor is
  compressive-positive and is negated on read instead).

"""

import numpy as np

from ...schema.reference import STRESS_SCOPES
from ..parser.output_parser import EV_TO_HARTREE, _STRESS_LABEL_TO_GPA

# Hartree/Angstrom^3 -> GPa, derived from the two constants the parser
# already owns (1 Hartree in eV and the eV/Angstrom^3 label factor),
# so the finite-difference result and the analytical tensor read from a
# CP2K print land in the same unit: the manifest-declared GPa.
HARTREE_A3_TO_GPA = _STRESS_LABEL_TO_GPA['ev/a^3'] / EV_TO_HARTREE

# The six independent components of a symmetric tensor, ordered
# (label, axis_a, axis_b) as documented in dev/skills/
# CP2K_SKILL_DEVELOPMENT.md section 6.6 (shear factor).
FULL_DIRECTIONS = (
    ('xx', 0, 0),
    ('yy', 1, 1),
    ('zz', 2, 2),
    ('xy', 0, 1),
    ('xz', 0, 2),
    ('yz', 1, 2),
)

# The strain scopes themselves are declared once in schema/reference.py
# STRESS_SCOPES (imported above), which is what `stress_scope` is
# validated against, so the mathematics and the schema cannot drift
# apart.

# (a, b) slot pairs in the evaluation order of direction_matrices()
# ('full' branch); kept next to FULL_DIRECTIONS so that reordering one
# list cannot silently pair a strain matrix with the wrong slots.
_FULL_AXIS_ORDER = tuple((a, b) for _, a, b in FULL_DIRECTIONS)


def direction_matrices(scope: str = 'full'):
    """Return the strain direction matrices of one scope.

    Parameters
    ----------
    scope : str
        'full' for the six independent components of the tensor,
        'iso' for the uniform dilation only (B = I).

    Returns
    -------
    list of tuple(str, numpy.ndarray)
        (label, B) pairs in evaluation order; 'iso' yields the single
        'iso' direction.

    Raises
    ------
    ValueError
        If the scope is not one of STRESS_SCOPES.
    """
    if scope not in STRESS_SCOPES:
        raise ValueError(
            f"Unknown stress_scope {scope!r}; expected one of "
            f"{list(STRESS_SCOPES)}.")
    if scope == 'iso':
        return [('iso', np.eye(3))]

    directions = []
    for label, a, b in FULL_DIRECTIONS:
        B = np.zeros((3, 3))
        if a == b:
            B[a, a] = 1.0
        else:
            # Shear convention: both slots carry 1/2 so that the
            # double contraction sigma : B is exactly sigma[a, b].
            B[a, b] = B[b, a] = 0.5
        directions.append((label, B))
    return directions


def cell_volume(H) -> float:
    """Return the volume of a row-vector lattice.

    Parameters
    ----------
    H : numpy.ndarray
        3x3 lattice with the vectors stored by row.

    Returns
    -------
    float
        abs(det(H)) in Angstrom^3.

    Raises
    ------
    ValueError
        If the lattice is not 3x3 or its volume is not a positive
        finite number (a singular cell has no stress definition).
    """
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"lattice_vector must be a 3x3 matrix, got shape {matrix.shape}.")
    volume = abs(float(np.linalg.det(matrix)))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(
            f"Cell volume must be a positive finite number, got {volume!r} "
            f"(lattice_vector {matrix.tolist()}).")
    return volume


def perturb(H, R, B, delta: float, sign: int):
    """Apply one signed strain to the cell and the Cartesian positions.

    Both the lattice and the atomic positions are transformed by the
    same factor, which leaves the fractional coordinates S unchanged
    (dev/skills/CP2K_SKILL_DEVELOPMENT.md section 6.6, the deformation).
    The perturbed points
    must never be re-optimized: the result is then not the
    instantaneous stress of the central configuration.

    Parameters
    ----------
    H : array-like
        Central 3x3 lattice, row vectors, Angstrom.
    R : array-like
        Central (natom, 3) Cartesian coordinates, Angstrom.
    B : array-like
        Strain direction matrix from direction_matrices().
    delta : float
        Dimensionless strain step (start at 1e-4 ~ 1e-3).
    sign : int
        +1 for the plus perturbation, -1 for the minus one.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        (H_signed, R_signed): deformed lattice and coordinates.

    Raises
    ------
    ValueError
        If delta is not positive or sign is not +/-1.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(f"strain step must be positive, got {delta!r}")
    if sign not in (1, -1):
        raise ValueError(f"strain sign must be +1 or -1, got {sign!r}")

    lattice = np.asarray(H, dtype=np.float64)
    positions = np.asarray(R, dtype=np.float64)
    factor = np.eye(3) + float(sign) * float(delta) * np.asarray(
        B, dtype=np.float64)
    # Row-vector convention: post-multiply by the transpose.
    return lattice @ factor.T, positions @ factor.T


def difference_component(energy_plus, energy_minus, delta: float,
                         volume0: float):
    """Central-difference one stress component per electronic state.

    Implements sigma : B = [E(+) - E(-)] / (2 * delta * V0) in the
    tensile-positive convention. Energies arrive in Hartree (the skill
    result contract), the cell in Angstrom^3, so the raw quotient is
    Hartree/Angstrom^3; it is returned in GPa.

    Parameters
    ----------
    energy_plus : array-like
        State-resolved total energies of the + perturbation (Hartree).
    energy_minus : array-like
        State-resolved total energies of the - perturbation (Hartree).
    delta : float
        Dimensionless strain step used for both sides.
    volume0 : float
        CENTRAL cell volume (Angstrom^3).

    Returns
    -------
    numpy.ndarray
        (nstate,) stress component sigma : B, in GPa.

    Raises
    ------
    ValueError
        If either side carries a non-finite energy, or the two sides
        cover a different number of states.
    """
    plus = np.asarray(energy_plus, dtype=np.float64)
    minus = np.asarray(energy_minus, dtype=np.float64)
    if plus.shape != minus.shape:
        raise ValueError(
            f"+/- strain energies cover different state counts "
            f"({plus.shape} vs {minus.shape}).")
    if not (np.all(np.isfinite(plus)) and np.all(np.isfinite(minus))):
        raise ValueError(
            f"Non-finite strain energy: plus={plus.tolist()} "
            f"minus={minus.tolist()}.")
    return (plus - minus) / (2.0 * float(delta) * float(volume0)) \
        * HARTREE_A3_TO_GPA


def assemble_stress(components, scope: str = 'full'):
    """Assemble the symmetric 3x3 tensor from the direction values.

    'full' fills the diagonal from xx/yy/zz and the symmetric shear
    slots from xy/xz/yz (the shear convention of direction_matrices()
    already divided out the factor of two). 'iso' contracts the trace
    direction onto the diagonal: the single component of B = I is
    tr(sigma), so the isotropic tension-positive diagonal is
    component / 3, which is the -p * I form of
    dev/skills/CP2K_SKILL_DEVELOPMENT.md section 6.6 (isotropic only),
    written without a second
    negation.

    Parameters
    ----------
    components : list of array-like
        One (nstate,) sigma : B value per direction, in the order
        returned by direction_matrices().
    scope : str
        'full' or 'iso'.

    Returns
    -------
    numpy.ndarray
        (nstate, 3, 3) stress tensor in GPa, tensile-positive.

    Raises
    ------
    ValueError
        If the number of components does not match the scope.
    """
    values = np.asarray(components, dtype=np.float64)
    expected = 1 if scope == 'iso' else len(FULL_DIRECTIONS)
    if values.ndim != 2 or values.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} strain direction(s) for scope "
            f"{scope!r}, got array of shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Non-finite stress component: {values.tolist()}.")

    nstate = values.shape[1]
    tensor = np.zeros((nstate, 3, 3))

    if scope == 'iso':
        # component = sigma : I = tr(sigma); the isotropic tensor keeps
        # the (nstate, 3, 3) shape so every consumer of the stress
        # channel sees the same contract.
        tensor[:, 0, 0] = tensor[:, 1, 1] = tensor[:, 2, 2] = values[0] / 3.0
        return tensor

    for index, (a, b) in enumerate(_FULL_AXIS_ORDER):
        value = values[index]
        tensor[:, a, b] = value
        tensor[:, b, a] = value
    return tensor
