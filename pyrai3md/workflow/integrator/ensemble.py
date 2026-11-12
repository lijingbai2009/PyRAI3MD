"""
======================================================================
This module provides the extended-ensemble operators (thermostat and
barostat) of the NAMD workflow. It is the implementation behind
operations.apply_ensemble(state, stage), which delegates here; the
ensemble control is NOT part of the Verlet time integration itself but
an independent operation split around the force kicks
(dev/methods/ENSEMBLE_DEVELOPMENT.md sections 2 and 6).

The main loop calls apply_ensemble twice per step (sections 2, 12):

    apply_ensemble(state, stage='pre')    NHC half step + barostat
                                          momentum half kick (uses the
                                          previous frame stress)
    verlet_i / pbc / force eval / verlet_ii / ...
    apply_ensemble(state, stage='post')   NHC half step + barostat
                                          momentum half kick (uses the
                                          new stress)

Implemented operators
---------------------
Thermostats (NVT / NPT):
    nose-hoover / nose-hoover-chain   Nose-Hoover Chain (Martyna,
        Klein, Tuckerman, J. Chem. Phys. 97, 2635 (1992)): a chain of
        M fictitious momenta thermalizing the particle kinetic energy,
        propagated with a Yoshida-Suzuki 4th-order composition and a
        Strang-split recursive substep; M=1 is the single Nose-Hoover
        oscillator. Applied at both pre and post (each over dt/2), so
        the net velocity scaling per step is exp(-p_xi1/Q1 dt/2).
    csvr   Stochastic velocity rescaling (Bussi, Donadio, Parrinello,
        J. Chem. Phys. 126, 014101 (2007)): one whole-step random
        rescale K' = alpha K sampling the canonical kinetic-energy
        distribution exactly. Applied once, at post.

Barostats (NPT):
    berendsen   First-order weak coupling (Berendsen et al.,
        J. Chem. Phys. 81, 3684 (1984)): mu = I - (dt/3 tau_p) beta
        (P0 - P_int) rescales coordinates and cell. Correct average
        pressure, wrong fluctuations (equilibration only). Applied at
        post.
    parrinello-rahman / mttk   Extended-cell full-tensor barostat in
        the strain (epsilon) formulation (Martyna, Tobias, Klein,
        J. Chem. Phys. 101, 4177 (1994); Parrinello, Rahman,
        J. Appl. Phys. 52, 7182 (1981)): epsilon, p_epsilon and
        v_epsilon = p_epsilon / W are 3x3, the drift exp(v_epsilon dt)
        is applied by verlet_i, and the barostat momentum gets a half
        kick at pre and post. MTTK (alpha = 1 + 3/N_df) additionally
        hangs a Nose-Hoover chain on the cell momentum; Parrinello-
        Rahman uses alpha = 1.
    scr   Stochastic cell rescaling (Bernetti & Bussi, J. Chem. Phys.
        153, 114107 (2020); full tensor: Del Tatto, arXiv:2111.06402).
        First-order like berendsen and applied once, at post, but with
        the Ito drift correction and a Gaussian random matrix in the
        rescaling, which is what makes its volume fluctuations the
        isobaric ones. It carries no cell momentum, so it has no W and
        no barostat_* tensor; tau_p here is a relaxation time. Because
        the cell is stochastic, it is the barostat that pairs with csvr
        (their thermostat pair is exactly the same construction), and
        it is the one whose correctness must be judged on <(dlnV)^2>
        rather than on an energy drift, which its Euler step does not
        define. Isotropic cell only: validate.py refuses it with
        barostat_mode 'anisotropic' for the reason recorded in docs
        section 8.13.

Internal pressure (docs section 6):

    P_int = P_kin - P_vir
    P_kin[a,b] = (1/V) sum_i m_i v_i[a] v_i[b]     (Hartree / Bohr^3)
    P_vir      = state.active_stress()             (Hartree / Bohr^3)

What v_i IS, because the whole barostat reads this one line: the stored
velocities are PECULAR momenta, the velocity relative to the affine cell
flow, not lab-frame velocities. Freeze every atom, drive only the cell,
and state.kinetic_energy() reports 0.0 while the coordinates still move
-- that is the audit, pinned as
qualification/verification/_ensemble_helpers.py
::TestVelocityConventionAudit. It matters because subtracting the
streaming velocity a second time (the change a lab-frame reading would
demand) halves P_kin, and a barostat whose kinetic pressure is halved
under-pressurizes every NPT run it touches, quietly. The cell-drift
coupling verlet.py applies to the velocities at exp(-alpha v_eps dt/2)
is the MTTK peculiar-momentum form, which is the same statement seen
from the other side.

The cp2k / xtb / lammps backends populate stress_tensor when the
planning layer derives the 'stress' capability (md + ensemble=npt, or
optimize + optimize_cell); the calculator converts it centrally to
Hartree/Bohr^3, so active_stress() already matches the P_int = P_kin -
stress convention (tensile-positive potential stress). A backend that
provides no stress leaves active_stress() None, and the barostats raise
instead of treating a missing stress as a physical zero (docs section 8).

Degrees of freedom (docs section 6): N_df = 3N - N_c - N_com, with N_c
the frozen atoms (system.freeze) and N_com = 3 (a periodic system
removes the center-of-mass drift but not the rotation). The SHAKE
constraint count is left for the second phase.

All reads go through the PyRAIState read-only properties and all writes
through state.update_system() (the extended DOF live in pmd.system and
are persisted with the checkpoint/history, never synced to a backend).

Unit convention (framework-wide internal units):
    coordinates  Angstrom   velocities   Bohr / au
    masses       amu (au via state.mass_au)   time  au
    cell/lattice Angstrom   volume (system field) Angstrom^3
    pressure     Hartree / Bohr^3   temperature -> kT via KELVIN_TO_HARTREE

Sep 12 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

import numpy as np

from ...utilities.units import (
    KELVIN_TO_HARTREE,
    ANGSTROM_TO_BOHR,
    FS_TO_AU_TIME,
    bar_to_au_pressure,
    compressibility_bar_to_au,
)
from ...data.geometry import lattice_matrix
from ...schema.validate import THERMOSTAT_ALIASES, BAROSTAT_ALIASES
from ..records.diagnostics import check_finite
from .verlet import _matrix_expm, _scale_cell

# Number of center-of-mass degrees of freedom removed for a periodic
# system (docs section 6): the drift is removed, the rotation is kept.
N_COM_DOF = 3

# Yoshida-Suzuki 4th-order composition weights (three substeps whose
# intervals sum to one). nys=1 falls back to the plain Strang split.
_YOSHIDA_CBRT2 = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1 = 1.0 / (2.0 - _YOSHIDA_CBRT2)
_YOSHIDA_W0 = -_YOSHIDA_CBRT2 / (2.0 - _YOSHIDA_CBRT2)

# Barostat-mode normalization (case-insensitive), the vocabulary is
# validated in schema.validate.
_BAROSTAT_MODE_ALIASES = {'iso': 'isotropic', 'aniso': 'anisotropic'}

MISSING_STRESS_MSG = (
    "The NPT barostat requires a whole-system stress tensor, but "
    "state.active_stress() returned None. The active theory backend did "
    "not provide Properties.stress_tensor for this step (only the "
    "cp2k/xtb/lammps skills populate it, and only when the run derives "
    "the 'stress' capability; dev/methods/ENSEMBLE_DEVELOPMENT.md section "
    "8); the barostat refuses to treat a missing stress as a physical "
    "zero.")


def _norm_token(value, aliases, default):
    """Normalize a control token (lower-case, alias-resolved).

    Parameters
    ----------
    value : str or None
        Raw keyword value.
    aliases : dict
        Alias -> canonical map (may be empty).
    default : str
        Fallback when value is empty.

    Returns
    -------
    str
        The canonical lower-case token.
    """
    token = str(value if value is not None else default).strip().lower()
    token = aliases.get(token, token)
    return token or default


def _diag(state, **entries):
    """Record per-step ensemble diagnostics on the runtime.

    The entries land in state.runtime.ensemble (a transient dict reset
    at the pre stage of every step) and are consumed by the trajectory
    writer (md.ensemble / sh.ensemble) and the run-log '&ensemble'
    section. Missing runtime or a missing dict is tolerated so mock
    states without the field never break the integrators. Values are
    stored as plain python floats / (nested) lists.

    Parameters
    ----------
    state : PyRAIState
        State carrying the runtime container.
    **entries
        Diagnostic key/value pairs to merge into the record.
    """
    record = getattr(getattr(state, 'runtime', None), 'ensemble', None)
    if not isinstance(record, dict):
        return
    record.update(entries)


def _iso_scale(matrix):
    """Isotropic equivalent of a 3x3 scaling matrix: det^(1/3).

    Parameters
    ----------
    matrix : np.ndarray
        (3, 3) scaling matrix.

    Returns
    -------
    float
        The cube root of the determinant (NaN for a non-finite or
        non-positive determinant).
    """
    with np.errstate(invalid='ignore', over='ignore'):
        det = float(np.linalg.det(np.asarray(matrix, dtype=np.float64)))
    if not np.isfinite(det) or det <= 0.0:
        return float('nan')
    return det ** (1.0 / 3.0)


def _dof(state):
    """Count the thermal degrees of freedom N_df = 3N - N_c - N_com.

    With a constraint plan, `N_c` is the rank of its condition matrix, so a
    frozen atom (a `fix`), a SHAKE bond and an angle each cost what the
    projection itself removes (docs sections 6 and 9, D-06). Without one, `N_c`
    is zero: every way to freeze an atom or hold a distance now lives in the
    plan, so an empty plan means nothing was taken away. N_com = 3 removes the
    center-of-mass drift of a periodic system on both paths.

    Parameters
    ----------
    state : PyRAIState
        State carrying pmd.system (natom) and the constraint plan.

    Returns
    -------
    int
        The number of thermal degrees of freedom (at least 1).
    """
    plan = state.constraint_plan()
    if not plan.is_empty:
        # D-06: the count comes from the rank of the condition matrix, so a
        # frozen atom, a SHAKE bond and the periodic center-of-mass removal are
        # subtracted once each, by the same arithmetic the projection itself
        # uses. The formula below can only count atoms, and a bond constraint
        # costs a degree of freedom without freezing an atom.
        from pyrai3md.workflow.constraints import controller as _controller
        from pyrai3md.workflow.constraints import state_adapter as _adapter

        controller = _controller.SystemConstraintController(plan)
        view, _context = _adapter.read_state(state, purpose='md')
        statistics = controller.statistics(
            _adapter.solve_geometry(view), view.velocities, view.masses,
            getattr(state, 'constraint_state', None))
        # The center-of-mass removal is the framework's own convention and the
        # classic path below applies it unconditionally, so the constrained path
        # subtracts the same three here: a thermostat must see one counting rule
        # whether or not constraints exist. What changes is that a zero is now
        # reported as a refusal instead of clamped up to one.
        ndof = int(statistics.free_dof) - N_COM_DOF
        if ndof <= 0:
            raise ValueError(
                'the declared constraints leave %d thermal degrees of freedom; '
                'a thermostat cannot control this system, and the count is not '
                'clamped to one, because that would report a temperature the '
                'run does not have' % ndof)
        return ndof
    system = getattr(state.pmd, 'system', None)
    natom = int(getattr(system, 'natom', 0) or 0)
    # Nothing is frozen on this path, and that is now a fact rather than a
    # reading: a frozen atom is a `fix` action, an action makes a plan, and a
    # plan took the branch above. The archive's flat `freeze` list cannot reach
    # here even if an input still writes it, because `schema.config.System`
    # refuses that keyword (#272), so counting it here would be arithmetic on a
    # number no run can hold.
    ndof = 3 * natom - N_COM_DOF
    # The `max(..., 1)` clamp stays on the unconstrained path on purpose: it is
    # what every existing NVT and NPT run reads today, and removing it changes
    # the ensemble line's own behaviour rather than the constraint subsystem's.
    # It is reached by one input only - a single atom in a periodic cell, where
    # 3 - 3 is zero. The constrained path above never clamps, because a system
    # left with no thermal degree of freedom is a refusal, not a temperature.
    return max(ndof, 1)


def _lattice_matrix(state):
    """Return the (3, 3) lattice matrix (Angstrom) or None.

    The rule that decides which spelling of the cell is authoritative lives in
    `data/geometry.py::lattice_matrix`, because a barostat, a periodic wrap and
    a geometry search must not prefer them in different orders. This is the
    state-level reading of it.

    Parameters
    ----------
    state : PyRAIState
        State carrying pmd.system (lattice_vector / cell).

    Returns
    -------
    np.ndarray or None
        Lattice matrix (3, 3), rows are cell vectors (Angstrom).
    """
    return lattice_matrix(getattr(state.pmd, 'system', None))


def _volume_bohr3(state):
    """Return the cell volume in Bohr^3 (0.0 when no cell is defined).

    Uses the persisted system.volume (Angstrom^3) when positive, else
    the determinant of the lattice matrix.

    Parameters
    ----------
    state : PyRAIState
        State carrying pmd.system (volume / lattice_vector / cell).

    Returns
    -------
    float
        Cell volume in Bohr^3.
    """
    system = getattr(state.pmd, 'system', None)
    vol = float(getattr(system, 'volume', 0.0) or 0.0)
    if vol <= 0.0:
        lattice = _lattice_matrix(state)
        if lattice is not None:
            vol = float(abs(np.linalg.det(lattice)))
    if vol <= 0.0:
        return 0.0
    return vol * ANGSTROM_TO_BOHR ** 3


def _kinetic_pressure_tensor(state, volume_bohr3):
    """Kinetic pressure tensor P_kin[a,b] = (1/V) sum_i m_i v_i[a] v_i[b].

    Velocities are Bohr/au and masses atomic units, so the result is in
    Hartree/Bohr^3 (the atomic pressure unit). Symmetrized defensively.

    Parameters
    ----------
    state : PyRAIState
        State carrying velocities (Bohr/au) and masses.
    volume_bohr3 : float
        Cell volume in Bohr^3.

    Returns
    -------
    np.ndarray or None
        The (3, 3) kinetic pressure tensor, or None when unavailable.
    """
    mass_au = state.mass_au
    velocities = np.asarray(state.velocities, dtype=np.float64)
    if (mass_au is None or velocities.size == 0
            or volume_bohr3 <= 0.0
            or mass_au.shape[0] != velocities.shape[0]):
        return None
    pk = np.einsum('i,ia,ib->ab', mass_au, velocities, velocities)
    pk = pk / volume_bohr3
    return 0.5 * (pk + pk.T)


def _previous_stress(state):
    """Read the previous frame's active-state stress (3, 3) or None.

    The pre-stage barostat half kick reuses the last recorded step's
    virial (history nuclear_sample.active_stress), captured when that
    step was recorded (docs section 4).

    Parameters
    ----------
    state : PyRAIState
        State with a history buffer.

    Returns
    -------
    np.ndarray or None
        The (3, 3) previous stress (Hartree/Bohr^3), or None.
    """
    snap = state.get_history_step(-1, copy=False)
    if not snap:
        return None
    sample = snap.get('nuclear_sample') or {}
    stress = sample.get('active_stress')
    if stress is None:
        return None
    arr = np.asarray(stress, dtype=np.float64)
    if arr.shape != (3, 3):
        return None
    return 0.5 * (arr + arr.T)


def _thermostat_masses(state, ndof, kt, chain, tau_au=None):
    """Build the Nose-Hoover chain masses Q (docs section 6).

    Q_1 = N_df kT tau^2 / (4 pi^2) and Q_k = kT tau^2 / (4 pi^2) for
    k >= 2, with tau the chain's own time constant (the plan convention
    Q = tau^2 T0 / 4 pi^2, T0 -> N_df kT for the head). An explicit
    thermostat_mass overrides Q_1.

    Which field supplies tau is not a detail: a chain coupled FASTER than
    the period of the motion it thermostats does not damp that motion, it
    pumps it. The particle bath takes thermostat_tau (tau_au=None); the
    cell momentum takes barostat_tau, because the cell it thermalizes
    oscillates with a period of order tau_p (dev/methods/
    ENSEMBLE_DEVELOPMENT.md section 8.4: tau_p/sqrt(3) for the isotropic
    mode).

    Measured on a six-atom box driven by a CONSERVATIVE stress (see
    qualification/verification/_ensemble_helpers.py::cell_potential),
    2000 steps, particle thermostat off, and with W held at tau_p = 1000 fs
    so that only this mass varies. max |dV/V| and the rms of the excursion,
    against an equipartition width of 1.4% for this box:

        cell-chain tau   max |dV|/V   rms dV/V   reading
        100 fs (tau_T)      0.241       0.115    ~11 sigma: pumped
        400 fs              0.108       0.047    still over-driven
        1000 fs (tau_p)     0.044       0.011    ~ the expected band
        4000 fs             0.0007      0.0003   frozen, not sampled
        Parrinello-Rahman   0.004       0.0024   reference, no cell chain

    The 100 fs row is what every MTTK example ran on before this parameter
    existed, and it is the mechanism behind the volume runaway reported in
    CHANGELOG #246 (docs section 8.6) -- there mis-attributed in magnitude
    because that measurement drove the cell with a non-conservative stress.

    Parameters
    ----------
    state : PyRAIState
        State carrying pmd.dynamics (thermostat_tau / thermostat_mass).
    ndof : int
        Thermal degrees of freedom of the bath being thermalized.
    kt : float
        Target thermal energy kT (Hartree).
    chain : int
        Chain length M (>= 1).
    tau_au : float, optional
        Chain time constant in atomic units; None reads thermostat_tau.

    Returns
    -------
    np.ndarray
        Chain masses Q, shape (M,).
    """
    dynamics = getattr(state.pmd, 'dynamics', None)
    explicit = getattr(dynamics, 'thermostat_mass', None)
    if tau_au is None:
        tau_au = float(getattr(dynamics, 'thermostat_tau', 100.0)
                       or 100.0) * FS_TO_AU_TIME
    denom = 4.0 * np.pi ** 2
    if explicit is not None:
        q1 = float(explicit)
        qk = q1 / max(ndof, 1)
    else:
        q1 = ndof * kt * tau_au ** 2 / denom
        qk = kt * tau_au ** 2 / denom
    q1 = max(q1, 1e-30)
    qk = max(qk, 1e-30)
    return np.array([q1] + [qk] * (max(chain, 1) - 1), dtype=np.float64)


def _barostat_mass(state, kt, volume_bohr3):
    """Build the extended-cell barostat mass W (docs section 6).

    Derived from the isothermal compressibility and the coupling time,
    W = (V / beta) tau_p^2 / (4 pi^2) (Hartree au^2), so the cell
    oscillation period tracks tau_p; an explicit barostat_mass
    overrides it. W is evaluated from the instantaneous volume.

    Parameters
    ----------
    state : PyRAIState
        State carrying pmd.dynamics (barostat_tau / compressibility /
        barostat_mass).
    kt : float
        Target thermal energy kT (Hartree).
    volume_bohr3 : float
        Cell volume in Bohr^3.

    Returns
    -------
    float
        The barostat mass W (atomic units).
    """
    dynamics = getattr(state.pmd, 'dynamics', None)
    explicit = getattr(dynamics, 'barostat_mass', None)
    if explicit is not None:
        return max(float(explicit), 1e-30)
    tau_au = float(getattr(dynamics, 'barostat_tau', 1000.0)
                   or 1000.0) * FS_TO_AU_TIME
    beta_au = compressibility_bar_to_au(
        float(getattr(dynamics, 'compressibility', 4.5e-5) or 4.5e-5))
    denom = 4.0 * np.pi ** 2
    if beta_au > 0.0 and volume_bohr3 > 0.0:
        w = (volume_bohr3 / beta_au) * tau_au ** 2 / denom
    else:
        w = kt * tau_au ** 2 / denom
    return max(float(w), 1e-30)


def _yoshida_weights(nys):
    """Yoshida-Suzuki composition weights summing to one.

    Parameters
    ----------
    nys : int
        Composition order (1 = plain Strang, 3 = 4th-order Yoshida).

    Returns
    -------
    list of float
        Substep interval weights.
    """
    if nys <= 1:
        return [1.0]
    if nys == 3:
        return [_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1]
    raise ValueError(
        "Unsupported Yoshida order %d (use 1 or 3)." % nys)


def _nhc_substep(p_eta, xi, Q, ke, ndof, kt, tau):
    """One Strang-split Nose-Hoover-chain substep over interval tau.

    Recursive symmetric propagation of the chain momenta p_eta (the
    momentum form of the legacy validated M=2 nose_hoover): the outer
    thermostat gets a quarter kick, the coupling cascades inward with
    exp(-p_k/Q_k tau/8) factors, the particle kinetic energy is scaled
    by exp(-p_eta1/Q_1 tau/2) at the center, then the cascade unwinds.
    The thermostat positions xi are advanced with the midpoint momenta
    (the same p_eta the scaling used) so the extended Hamiltonian
    H_ext = KE + PE + sum p_k^2/(2 Q_k) + N_df kT xi_1 + kT sum_{k>=2}
    xi_k stays conserved (xi is a spectator that only enters H_ext).

    Parameters
    ----------
    p_eta : np.ndarray
        Chain momenta, shape (M,), modified in place.
    xi : np.ndarray
        Chain positions, shape (M,), modified in place.
    Q : np.ndarray
        Chain masses, shape (M,).
    ke : float
        Kinetic energy of the bath being thermalized (2*ke = sum p^2/m
        for particles, or W*sum p_eps^2 for the cell momentum).
    ndof : int
        Degrees of freedom of the bath.
    kt : float
        Target thermal energy kT (Hartree).
    tau : float
        Substep interval (atomic units).

    Returns
    -------
    tuple of (float, float)
        (velocity scale factor, updated kinetic energy).
    """
    m = len(p_eta)

    def force(k):
        if k == 0:
            return 2.0 * ke - ndof * kt
        return p_eta[k - 1] ** 2 / Q[k - 1] - kt

    p_eta[m - 1] += force(m - 1) * tau / 4.0
    for k in range(m - 1, 0, -1):
        c = np.exp(-(p_eta[k] / Q[k]) * tau / 8.0)
        p_eta[k - 1] *= c
        p_eta[k - 1] += force(k - 1) * tau / 4.0
        p_eta[k - 1] *= c
    scale = float(np.exp(-(p_eta[0] / Q[0]) * tau / 2.0))
    ke = ke * scale ** 2
    # The substep advances the NHC over the interval tau/2 (the scaling
    # exp(-xi_dot tau/2) and the two G tau/4 momentum kicks), so the
    # thermostat positions advance by xi_dot (tau/2) with the midpoint
    # momenta -- matching the scaling exactly keeps H_ext conserved.
    for k in range(m):
        xi[k] += (p_eta[k] / Q[k]) * (tau / 2.0)
    for k in range(1, m):
        c = np.exp(-(p_eta[k] / Q[k]) * tau / 8.0)
        p_eta[k - 1] *= c
        p_eta[k - 1] += force(k - 1) * tau / 4.0
        p_eta[k - 1] *= c
    p_eta[m - 1] += force(m - 1) * tau / 4.0
    return scale, ke


def _nhc_propagate(state, ke, ndof, kt, tau, field, nys=3,
                   mass_tau_au=None):
    """Propagate a Nose-Hoover chain over tau and return the scaling.

    Reads/writes the chain state (positions xi and momenta p_eta) from
    the named pmd.system field ([xi_1..xi_M, p_1..p_M]), initializing
    it to zeros on the first call. The chain is composed of nys
    Yoshida substeps; the accumulated velocity scale factor is returned
    for the caller to apply to whatever it thermalizes.

    Parameters
    ----------
    state : PyRAIState
        State carrying pmd.system and pmd.dynamics.
    ke : float
        Kinetic energy of the thermalized bath.
    ndof : int
        Bath degrees of freedom.
    kt : float
        Target thermal energy kT (Hartree).
    tau : float
        Propagation interval (atomic units).
    field : str
        pmd.system field storing the chain DOF.
    nys : int
        Yoshida composition order (1 or 3).

    Returns
    -------
    tuple of (float, float)
        (accumulated velocity scale factor, final bath kinetic energy).
    """
    dynamics = getattr(state.pmd, 'dynamics', None)
    chain = int(getattr(dynamics, 'thermostat_chain', 1) or 1)
    chain = max(chain, 1)
    # The masses are built HERE, from this call's own chain length -- a
    # caller that computed Q elsewhere changed only its own copy, which is
    # how a cell-chain time constant passed for the diagnostic could leave
    # the chain that actually runs on thermostat_tau. mass_tau_au exists so
    # the caller can say which time constant the masses must use.
    Q = _thermostat_masses(state, ndof, kt, chain, tau_au=mass_tau_au)

    system = getattr(state.pmd, 'system', None)
    raw = getattr(system, field, None) if system else None
    dof = np.asarray(raw, dtype=np.float64).reshape(-1) if raw is not None \
        else np.array([])
    if dof.size != 2 * chain:
        dof = np.zeros(2 * chain, dtype=np.float64)
    xi = np.array(dof[:chain], dtype=np.float64)
    p_eta = np.array(dof[chain:], dtype=np.float64)

    scale_total = 1.0
    for w in _yoshida_weights(nys):
        dtt = w * tau
        scale, ke = _nhc_substep(p_eta, xi, Q, ke, ndof, kt, dtt)
        scale_total *= scale

    updates = {field: np.concatenate([xi, p_eta]).tolist()}
    check_finite(state, updates)
    state.update_system(updates)
    return scale_total, ke


def _nose_hoover_chain(state, stage, dt):
    """Nose-Hoover / NHC thermostat half step (docs section 6).

    Applies one exp(iL_NHC dt/2) Strang half of the chain: _nhc_substep
    carries the internal 1/2 factors, so passing the full dt makes each
    call a dt/2 chain propagation that rescales the velocities by
    exp(-p_xi1/Q1 dt/2). Applied at both pre and post, the two halves
    bracket the force kicks and the net per-step scaling is
    exp(-p_xi1/Q1 dt), the reversible measure-preserving NHC-NVT split.

    Parameters
    ----------
    state : PyRAIState
        State carrying velocities, masses and pmd.dynamics.
    stage : str
        'pre' or 'post' (both apply one exp(iL_NHC dt/2) half).
    dt : float
        Full timestep (atomic units).

    Returns
    -------
    PyRAIState
        State with rescaled velocities and updated chain DOF.
    """
    dynamics = getattr(state.pmd, 'dynamics', None)
    kt = float(getattr(dynamics, 'temperature', 300.0)
               or 0.0) * KELVIN_TO_HARTREE
    ndof = _dof(state)
    mass_au = state.mass_au
    velocities = np.asarray(state.velocities, dtype=np.float64)
    if mass_au is None or velocities.size == 0 or kt <= 0.0:
        _diag(state, thermostat_skipped='missing mass/velocities or T <= 0')
        return state
    if mass_au.shape[0] != velocities.shape[0]:
        _diag(state, thermostat_skipped='mass/velocity count mismatch')
        return state
    ke = 0.5 * float(np.sum(mass_au[:, None] * velocities ** 2))
    # _nhc_substep(tau) realizes exp(iL_NHC tau/2), so tau=dt gives the
    # dt/2 chain propagation each stage needs (see the docstring).
    scale, _ = _nhc_propagate(state, ke, ndof, kt, dt,
                              field='thermostat_dof')
    chain = max(int(getattr(dynamics, 'thermostat_chain', 1) or 1), 1)
    _diag(state, ndof=ndof, ke=ke, kt=kt,
          q=_thermostat_masses(state, ndof, kt, chain).tolist(),
          thermostat_skipped=None,
          **{'scale_%s' % stage: float(scale)})
    if abs(scale - 1.0) < 1e-16:
        return state
    state.update_system({'velocities': (velocities * scale).tolist()})
    return state


def _csvr(state, dt):
    """Canonical sampling through stochastic velocity rescaling (CSVR).

    Bussi-Donadio-Parrinello whole-step rescale K' = alpha K with alpha
    drawn so the canonical kinetic-energy distribution is sampled
    exactly (docs section 6):

        alpha = (1-c) + c (K0/K)(S + R1^2)/N_df
                + 2 R1 sqrt(c(1-c)(K0/K)/N_df),   c = dt/tau_T

    with R1 a standard normal and S a chi-square(N_df - 1) deviate,
    drawn from the trajectory-seeded NumPy stream (md._seed_trajectory).

    Parameters
    ----------
    state : PyRAIState
        State carrying velocities, masses and pmd.dynamics.
    dt : float
        Full timestep (atomic units).

    Returns
    -------
    PyRAIState
        State with rescaled velocities.
    """
    dynamics = getattr(state.pmd, 'dynamics', None)
    kt = float(getattr(dynamics, 'temperature', 300.0)
               or 0.0) * KELVIN_TO_HARTREE
    tau_au = float(getattr(dynamics, 'thermostat_tau', 100.0)
                   or 100.0) * FS_TO_AU_TIME
    ndof = _dof(state)
    mass_au = state.mass_au
    velocities = np.asarray(state.velocities, dtype=np.float64)
    if mass_au is None or velocities.size == 0 or kt <= 0.0:
        _diag(state, thermostat_skipped='missing mass/velocities or T <= 0')
        return state
    if mass_au.shape[0] != velocities.shape[0]:
        _diag(state, thermostat_skipped='mass/velocity count mismatch')
        return state
    ke = 0.5 * float(np.sum(mass_au[:, None] * velocities ** 2))
    if ke <= 0.0:
        _diag(state, ndof=ndof, ke=ke, kt=kt,
              thermostat_skipped='Ekin <= 0 (cold start, no rescale)')
        return state
    ke0 = 0.5 * ndof * kt
    c = dt / tau_au if tau_au > 0.0 else 1.0
    c = min(max(c, 0.0), 1.0)
    ratio = ke0 / ke
    r1 = float(np.random.normal())
    s = float(np.random.chisquare(max(ndof - 1, 1)))
    alpha = ((1.0 - c)
             + c * ratio * (s + r1 ** 2) / ndof
             + 2.0 * r1 * np.sqrt(max(c * (1.0 - c) * ratio / ndof, 0.0)))
    if alpha <= 0.0:
        _diag(state, ndof=ndof, ke=ke, ke0=ke0, kt=kt,
              csvr_c=float(c), csvr_ratio=float(ratio),
              csvr_r1=r1, csvr_s=s, csvr_alpha=float(alpha),
              thermostat_skipped='alpha <= 0 (no rescale)')
        return state
    _diag(state, ndof=ndof, ke=ke, ke0=ke0, kt=kt,
          csvr_c=float(c), csvr_ratio=float(ratio),
          csvr_r1=r1, csvr_s=s, csvr_alpha=float(alpha),
          scale_post=float(np.sqrt(alpha)), thermostat_skipped=None)
    state.update_system(
        {'velocities': (velocities * np.sqrt(alpha)).tolist()})
    return state


def _berendsen(state, dt):
    """Berendsen weak-coupling barostat step (docs section 6).

    First-order relaxation of the internal pressure toward the target:

        mu = I - (dt / 3 tau_p) beta (P0 - P_int)

    rescales the coordinates and the cell by mu (full tensor for
    barostat_mode='anisotropic', its isotropic trace for 'isotropic').
    Velocities are untouched. Applied at post with the current-frame
    internal pressure.

    Parameters
    ----------
    state : PyRAIState
        State carrying coordinates, cell/lattice and the current stress.
    dt : float
        Full timestep (atomic units).

    Returns
    -------
    PyRAIState
        State with rescaled coordinates and cell.

    Raises
    ------
    ValueError
        When the whole-system stress is unavailable
        (MISSING-PHASE1-STRESS).
    """
    dynamics = getattr(state.pmd, 'dynamics', None)
    volume = _volume_bohr3(state)
    p_kin = _kinetic_pressure_tensor(state, volume)
    stress = state.active_stress()
    if stress is None or p_kin is None or volume <= 0.0:
        if stress is None:
            raise ValueError(MISSING_STRESS_MSG)
        _diag(state, barostat_skipped='missing kinetic pressure or cell')
        return state
    p_int = p_kin - stress
    mode = _norm_token(getattr(dynamics, 'barostat_mode', 'anisotropic'),
                       _BAROSTAT_MODE_ALIASES, 'anisotropic')
    p_ext = bar_to_au_pressure(
        float(getattr(dynamics, 'pressure', 1.0) or 0.0))
    beta_au = compressibility_bar_to_au(
        float(getattr(dynamics, 'compressibility', 4.5e-5) or 0.0))
    tau_au = float(getattr(dynamics, 'barostat_tau', 1000.0)
                   or 1000.0) * FS_TO_AU_TIME

    diff = p_ext * np.eye(3) - p_int
    if mode == 'isotropic':
        diff = (np.trace(diff) / 3.0) * np.eye(3)
    mu = np.eye(3) - (dt / (3.0 * tau_au)) * beta_au * diff

    coordinates = np.asarray(state.coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.size == 0:
        _diag(state, barostat_skipped='missing coordinates')
        return state
    _diag(state, volume_bohr3=float(volume),
          p_kin=p_kin.tolist(), p_vir=np.asarray(stress).tolist(),
          p_int=p_int.tolist(), p_ext=float(p_ext), mu=mu.tolist(),
          barostat_skipped=None)
    scaled = coordinates @ mu
    # The cell commit may re-triangularize the anisotropically scaled
    # lattice and rotate the frame; the rotated coordinates (and the
    # frame-rotated velocities) then come back through the updates.
    updates = _scale_cell(state, mu, coordinates=scaled,
                          velocities=np.asarray(state.velocities,
                                                dtype=np.float64))
    if 'coordinates' not in updates:
        updates['coordinates'] = scaled.tolist()
    check_finite(state, updates)
    state.update_system(updates)
    return state


def _scr(state, dt):
    """Stochastic cell rescaling: first-order NPT with a noise term.

    Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020), eq 3. The log
    volume is propagated by an Ornstein-Uhlenbeck step,

        d ln V = -(beta_T / tau_p)(P0 - P_int) dt
                 + sqrt(2 kT beta_T dt / (V tau_p)) dW

    and the length scale is its cube root, so the cell breathes and nothing
    else. Two things about this equation are the whole method:

      * the noise amplitude is fixed by the same beta_T/tau_p that appears in
        the drift, and that pairing is what makes the stationary width the
        isobaric one, <(d ln V)^2> = kT beta_sys / <V>, independent of both
        barostat parameters. Judge it on that variance: an energy drift is
        not defined for this integrator at all, because the noise prefactor
        depends on the propagated variable (arXiv:2111.06402 section 4);
      * tau_p is a RELAXATION time here, not the oscillation period of a cell
        mass. The long barostat_tau the parrinello-rahman and mttk examples
        need (10000 fs, docs section 8.10) has no counterpart in this scheme,
        and 1 ps is the value the source uses.

    Momenta are not rescaled. Their eq 29b offers p^T <- p^T mu^-1, but the
    paper's own chapter-3 pseudocode parenthesises that line, section 2.1
    says rescaling momenta is "not in general a necessary feature", and all
    three of its implementations are written for the cell and the positions.
    With the positions-only formulation the kinetic part of the drive is the
    ensemble average N_df kT / 3V (their eq 12) rather than the instantaneous
    tensor, which is why `p_kin` in the record can and does disagree with
    what the barostat used.

    The full-tensor form of the same scheme (arXiv:2111.06402 eq 28: nine
    independent Gaussians, an Ito drift correction, mu acting on the box from
    the Cartesian side) was implemented here, measured, and removed. It is
    correct as written and it does not work on this framework's systems: a
    liquid has no shear modulus, so the shear it injects (~1.5e-3 of strain
    per 1 fs step) is dissipated into heat rather than stored, and the cell
    then collapses behind the temperature. Docs section 8.13 has the numbers.
    schema/validate.py refuses `scr` with `barostat_mode: anisotropic`, so
    nothing here should reintroduce it without that measurement being
    revisited; a tilted cell has parrinello-rahman and mttk.

    Parameters
    ----------
    state : PyRAIState
        State carrying coordinates, the lattice, the volume and the stress of
        this step. Velocities are read for the record and otherwise untouched.
    dt : float
        Full timestep (atomic units).

    Returns
    -------
    PyRAIState
        State with the cell and coordinates rescaled proportionally.

    Raises
    ------
    ValueError
        When the whole-system stress is unavailable
        (MISSING-PHASE1-STRESS).
    """
    dynamics = getattr(state.pmd, 'dynamics', None)
    volume = _volume_bohr3(state)
    p_kin = _kinetic_pressure_tensor(state, volume)
    stress = state.active_stress()
    if stress is None or p_kin is None or volume <= 0.0:
        if stress is None:
            raise ValueError(MISSING_STRESS_MSG)
        _diag(state, barostat_skipped='missing kinetic pressure or cell')
        return state
    coordinates = np.asarray(state.coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.size == 0:
        _diag(state, barostat_skipped='missing coordinates')
        return state
    kt = float(getattr(dynamics, 'temperature', 300.0)
               or 0.0) * KELVIN_TO_HARTREE
    beta_au = compressibility_bar_to_au(
        float(getattr(dynamics, 'compressibility', 4.5e-5) or 0.0))
    tau_au = float(getattr(dynamics, 'barostat_tau', 1000.0)
                   or 1000.0) * FS_TO_AU_TIME
    if kt <= 0.0 or tau_au <= 0.0 or beta_au <= 0.0:
        _diag(state, barostat_skipped='scr needs kT, beta_T and tau_p > 0')
        return state

    # eq 12, not eq 60: the drive reads the target temperature, and the
    # instantaneous kinetic tensor stays in the record for comparison.
    p_int = ((_dof(state) * kt / (3.0 * volume)) * np.eye(3)
             - np.asarray(stress, dtype=np.float64))
    p_ext = float(bar_to_au_pressure(
        float(getattr(dynamics, 'pressure', 1.0) or 0.0)))
    sigma = float(np.sqrt(2.0 * kt * beta_au * dt / (volume * tau_au)))
    drift = -(beta_au / tau_au) * (
        p_ext - float(np.trace(p_int) / 3.0)) * dt
    d_epsilon = drift + sigma * float(np.random.normal())
    length = float(np.exp(d_epsilon / 3.0))
    mu = length * np.eye(3)

    _diag(state, volume_bohr3=float(volume),
          p_kin=p_kin.tolist(), p_vir=np.asarray(stress).tolist(),
          p_int=p_int.tolist(), p_ext=p_ext,
          scr_sigma=sigma, scr_mu_det=length ** 3, mu=mu.tolist(),
          barostat_skipped=None)
    scaled = coordinates @ mu
    updates = _scale_cell(state, mu, coordinates=scaled,
                          velocities=np.asarray(state.velocities,
                                                dtype=np.float64))
    if 'coordinates' not in updates:
        updates['coordinates'] = scaled.tolist()
    check_finite(state, updates)
    state.update_system(updates)
    return state


def _extended_cell(state, stage, dt):
    """Full-tensor extended-cell barostat half step (MTTK / PR).

    Strain (epsilon) formulation (docs section 6): the cell momentum
    p_epsilon (3x3) gets a half kick from the barostat force

        G_eps = V (alpha P_kin - P_vir - P_ext I)

    with alpha = 1 + 3/N_df (MTTK) or 1 (Parrinello-Rahman) and
    P_vir = the active stress. MTTK additionally thermalizes p_epsilon
    with a Nose-Hoover chain (barostat_thermostat_dof). The particle
    velocities are coupled by exp(-alpha v_eps dt/2); the coordinate /
    cell drift exp(v_eps dt) is applied by verlet_i (which reads the
    barostat_velocity stored here). The pre stage kicks with the
    previous-frame stress, the post stage advances epsilon with the
    drifted cell and kicks with the new stress.

    Parameters
    ----------
    state : PyRAIState
        State carrying velocities, masses, cell/lattice, volume, the
        barostat DOF and the stress.
    stage : str
        'pre' (previous-frame stress) or 'post' (new stress).
    dt : float
        Full timestep (atomic units).

    Returns
    -------
    PyRAIState
        State with updated velocities, barostat momentum/velocity and
        (post) strain.

    Raises
    ------
    ValueError
        When the post-stage whole-system stress is unavailable
        (MISSING-PHASE1-STRESS).
    """
    dynamics = getattr(state.pmd, 'dynamics', None)
    barostat = _norm_token(getattr(dynamics, 'barostat', 'off'),
                           BAROSTAT_ALIASES, 'off')
    kt = float(getattr(dynamics, 'temperature', 300.0)
               or 0.0) * KELVIN_TO_HARTREE
    ndof = _dof(state)
    volume = _volume_bohr3(state)
    lattice = _lattice_matrix(state)
    if kt <= 0.0 or ndof <= 0 or volume <= 0.0 or lattice is None:
        _diag(state, barostat_skipped='missing cell/temperature')
        return state
    mode = _norm_token(getattr(dynamics, 'barostat_mode', 'anisotropic'),
                       _BAROSTAT_MODE_ALIASES, 'anisotropic')
    p_ext = bar_to_au_pressure(
        float(getattr(dynamics, 'pressure', 1.0) or 0.0))
    w = _barostat_mass(state, kt, volume)
    alpha = 1.0 + 3.0 / ndof if barostat == 'mttk' else 1.0
    tau = 0.5 * dt

    system = getattr(state.pmd, 'system', None)
    p_eps = np.asarray(getattr(system, 'barostat_momentum', None) or [],
                       dtype=np.float64)
    if p_eps.shape != (3, 3):
        p_eps = np.zeros((3, 3), dtype=np.float64)
    v_eps = np.asarray(getattr(system, 'barostat_velocity', None) or [],
                       dtype=np.float64)
    if v_eps.shape != (3, 3):
        v_eps = np.zeros((3, 3), dtype=np.float64)

    if stage == 'pre':
        stress = _previous_stress(state)
        if stress is None:
            # First step (no history stress yet): skip the pre kick; the
            # post stage raises if the backend never provides a stress.
            _diag(state, barostat_skipped_pre='no previous stress')
            return state
    else:
        stress = state.active_stress()
        if stress is None:
            raise ValueError(MISSING_STRESS_MSG)
        eps = np.asarray(getattr(system, 'barostat_strain', None) or [],
                         dtype=np.float64)
        if eps.shape != (3, 3):
            eps = np.zeros((3, 3), dtype=np.float64)
        # The drift exp(v_eps dt) was applied by verlet_i; advance the
        # strain consistently (epsilon is the log-deformation).
        eps = eps + v_eps * dt

    p_kin = _kinetic_pressure_tensor(state, volume)
    if p_kin is None:
        _diag(state, barostat_skipped='missing kinetic pressure')
        return state
    g_eps = volume * (alpha * p_kin - stress - p_ext * np.eye(3))
    if mode == 'isotropic':
        g_eps = (np.trace(g_eps) / 3.0) * np.eye(3)
    g_eps = 0.5 * (g_eps + g_eps.T)
    p_eps = p_eps + g_eps * tau

    diag = {
        'ndof': ndof, 'volume_bohr3': float(volume), 'w': float(w),
        'alpha_cell': float(alpha), 'p_ext': float(p_ext),
        'p_kin': p_kin.tolist(),
        'p_vir': np.asarray(stress).tolist(),
        'p_int': (p_kin - np.asarray(stress, dtype=np.float64)).tolist(),
        'g_eps': g_eps.tolist(),
        'barostat_skipped': None,
    }
    if barostat == 'mttk':
        ndof_bar = 3 if mode == 'isotropic' else 9
        # Cell-momentum kinetic energy of the extended Hamiltonian:
        # sum p_eps^2 / (2 W), consistent with v_eps = p_eps / W.
        ke_bar = 0.5 * float(np.sum(p_eps ** 2)) / w
        # The cell-momentum NHC uses the same exp(iL dt/2) half as the
        # particle chain (_nhc_substep carries the internal 1/2), so the
        # interval argument is the full dt (the momentum half kick above
        # keeps its own dt/2 via tau).
        q_bar = _thermostat_masses(state, ndof_bar, kt,
                                   max(int(getattr(
                                       dynamics, 'thermostat_chain', 1)
                                       or 1), 1),
                                   # The cell momentum is thermalized on the
                                   # BAROSTAT time, not the particle
                                   # thermostat's: see _thermostat_masses.
                                   tau_au=float(getattr(
                                       dynamics, 'barostat_tau', 1000.0)
                                       or 1000.0) * FS_TO_AU_TIME)
        # _nhc_propagate returns (scale_total, ke): the cell-momentum
        # NHC thermalizes p_eps with the accumulated scale factor (the
        # first return), not the propagated bath kinetic energy.
        scale_bar, _ = _nhc_propagate(
            state, ke_bar, ndof_bar, kt, dt,
            field='barostat_thermostat_dof',
            mass_tau_au=float(getattr(dynamics, 'barostat_tau', 1000.0)
                              or 1000.0) * FS_TO_AU_TIME)
        p_eps = p_eps * scale_bar
        diag.update(ke_bar=float(ke_bar), scale_bar=float(scale_bar),
                    ndof_bar=ndof_bar, q_bar=q_bar.tolist())
    p_eps = 0.5 * (p_eps + p_eps.T)
    v_eps = p_eps / w

    updates = {
        'barostat_momentum': p_eps.tolist(),
        'barostat_velocity': v_eps.tolist(),
    }
    if stage == 'post':
        updates['barostat_strain'] = eps.tolist()
        diag['eps'] = eps.tolist()

    velocities = np.asarray(state.velocities, dtype=np.float64)
    mass_au = state.mass_au
    if (mass_au is not None and velocities.size
            and mass_au.shape[0] == velocities.shape[0]):
        scale_particle = _matrix_expm(-alpha * v_eps * tau)
        updates['velocities'] = (velocities @ scale_particle).tolist()
        diag['scale_particle_%s' % stage] = _iso_scale(scale_particle)

    _diag(state, **diag)
    check_finite(state, updates)
    state.update_system(updates)
    return state


def _dissipate_gate_open(state, dynamics):
    """Whether the thermostat is allowed to act on this step.

    `&md dissipate 1` propagates an excited trajectory on NVE and hands it to
    the thermostat only once the run has been on the electronic ground state
    for `thermodelay` steps. The reason is the one the legacy `nve_nvt` and
    `mixednvt` ensembles had: a chain coupled during the hop-dominated part
    of the dynamics takes away the kinetic energy that the hopping
    probability is made of, so the run thermalizes before it has finished
    relaxing electronically, and the two effects cannot be told apart
    afterwards.

    Three things this gate deliberately does not do:

    - it does not touch the barostat. The user ruling is nve to nvt, and a
      cell that stopped breathing mid-run would be a different ensemble
      rather than a delayed one;
    - it does not half-act. Both the 'pre' and the 'post' half of the split
      Nose-Hoover propagation are gated by this one reading, because one
      half step without the other is not the same integrator;
    - it does not close again. `RuntimeState.dissipate_step` is set the first
      time the trajectory is found on the ground state and never moved back,
      so a hop up does not switch the thermostat off and a restarted segment
      continues the delay its predecessor had already run down.

    Returns
    -------
    bool
        True when the thermostat acts. A run that does not declare
        `dissipate` is always open, which is what keeps every existing
        example byte-for-byte unchanged.
    """
    if int(getattr(dynamics, 'dissipate', 0) or 0) != 1:
        return True
    runtime = getattr(state, 'runtime', None)
    landed = int(getattr(runtime, 'dissipate_step', 0) or 0)
    if landed <= 0:
        return False
    delay = int(getattr(dynamics, 'thermodelay', 0) or 0)
    # The step being integrated, not the counter: both ensemble stages run
    # before `advance_step`, so the counter still reads one less than the
    # `md.py` loop's absolute step number, which is what `dissipate_step`
    # is stamped with.
    current = int(getattr(runtime, 'current_step', 0) or 0) + 1
    return current - landed >= delay


def apply_ensemble(state, stage='post'):
    """Apply the ensemble control (thermostat / barostat) to one stage.

    Dispatcher behind operations.apply_ensemble (docs sections 2 and 6).
    NVE is a no-op (both stages empty, so the split Verlet reduces to
    the classic merged form). NVT runs the thermostat; NPT runs the
    thermostat plus the barostat. The extended-cell barostats (MTTK /
    Parrinello-Rahman) act at both stages; CSVR and Berendsen act once,
    at post.

    Parameters
    ----------
    state : PyRAIState
        State with completed velocities for this stage.
    stage : str
        'pre' (before verlet_i) or 'post' (after velocity_correction).

    Returns
    -------
    PyRAIState
        The ensemble-controlled state.
    """
    if state is None or getattr(state, 'pmd', None) is None:
        return state
    dynamics = getattr(state.pmd, 'dynamics', None)
    if dynamics is None:
        return state
    ensemble = _norm_token(getattr(dynamics, 'ensemble', 'nve'), {}, 'nve')
    stage = str(stage or 'post').strip().lower()
    if stage == 'pre':
        # Fresh per-step record; an nve run keeps it empty so the
        # trajectory writer creates no md.ensemble file.
        record = getattr(getattr(state, 'runtime', None), 'ensemble', None)
        if isinstance(record, dict):
            record.clear()
    if ensemble == 'nve':
        return state
    dt = float(state.timestep)
    thermostat = _norm_token(getattr(dynamics, 'thermostat', 'off'),
                             THERMOSTAT_ALIASES, 'off')
    barostat = _norm_token(getattr(dynamics, 'barostat', 'off'),
                           BAROSTAT_ALIASES, 'off')

    if ensemble in ('nvt', 'npt') and _dissipate_gate_open(state, dynamics):
        if thermostat in ('nose-hoover', 'nose-hoover-chain'):
            state = _nose_hoover_chain(state, stage, dt)
        elif thermostat == 'csvr' and stage == 'post':
            state = _csvr(state, dt)

    if ensemble == 'npt' and barostat != 'off':
        if barostat in ('berendsen', 'scr'):
            # Both are first-order in the cell and read this step's
            # stress, so both act once per step at post.
            if stage == 'post':
                state = (_scr(state, dt) if barostat == 'scr'
                         else _berendsen(state, dt))
        elif barostat in ('mttk', 'parrinello-rahman'):
            state = _extended_cell(state, stage, dt)

    return state
