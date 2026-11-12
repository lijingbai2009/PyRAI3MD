"""
======================================================================
This is a standalone quantum dynamics model skill
designed for AI agent orchestration,

It provides modular interfaces for workflow-driven AI system,
and supports for programmatic invocation,
capatible in subprocess and slurm execution environments

Aug 24, 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================

One-dimensional fully quantum reference solver.

Propagates a Gaussian wavepacket on the two-state analytic model
with the Strang split-operator scheme (atomic units, hbar = 1):

    U(dt) ~ exp(-i T dt/2) exp(-i V dt) exp(-i T dt/2)

The kinetic half-step is applied in momentum space through numpy
FFT; the potential step is applied grid-point wise by diagonalizing
the 2x2 diabatic Hamiltonian and exponentiating the adiabatic
energies. The propagated potential surfaces come from the same
model object used by the single-point calculator, so both share a
single physical source of truth.

Recorded populations are adiabatic state populations: at every
snapshot the wavepacket is projected onto the local adiabatic
basis, P_i = int |psi_ad_i(x)|^2 dx. This is the representation
reported by the framework FSSH integrator (md.pop state
populations), so the reference is directly comparable to surface
hopping ensembles; the saved density record stays in the diabatic
basis. The initial condition follows the same convention: the
current_state keyword indexes the ADIABATIC state populated at
x0 (the production md root semantics), realized through the local
adiabatic eigenvector as diabatic component weights.

"""

import os
import warnings

import numpy as np

from .calculator import model_coordinate


# Fixed default dynamics window (Bohr): every reference propagation
# without explicit ref_xmin/ref_xmax runs on this grid so results
# from different calibrations stay directly comparable. The window
# covers the tully benchmark momenta k0 <= 30 over T = 6000 au
# starting from x0 = -20 (packet center reaches +70 plus spreading).
REF_XMIN_DEFAULT = -30.0
REF_XMAX_DEFAULT = 80.0

# Edge probability mass above which the snapshot warns once that the
# packet touches the periodic FFT boundary.
EDGE_MASS_THRESHOLD = 1.0e-6


def build_grid(ngrid, xmin, xmax):
    """Build the evenly spaced coordinate grid.

    Parameters
    ----------
    ngrid : int
        Number of grid points.
    xmin, xmax : float
        Grid boundaries in Bohr.

    Returns
    -------
    numpy.ndarray
        Grid coordinates, shape (ngrid,), periodic spacing
        dx = (xmax - xmin) / ngrid.
    """
    return np.linspace(xmin, xmax, ngrid, endpoint=False)


def initial_wavepacket(grid, x0, sigma, k0, components):
    """Build the normalized Gaussian wavepacket.

    psi(x) = N exp(-(x - x0)^2 / (4 sigma^2) + i k0 (x - x0))
    distributed over the two diabatic components by the weight
    vector components. Populating one adiabatic state at x0
    (the initial_state semantics of run_reference) passes the
    local adiabatic eigenvector as components; a single diabatic
    component uses a unit vector.

    Parameters
    ----------
    grid : numpy.ndarray
        Coordinate grid, shape (ngrid,).
    x0 : float
        Packet center in Bohr.
    sigma : float
        Packet width in Bohr.
    k0 : float
        Initial momentum in 1/Bohr.
    components : sequence of float
        Diabatic component weights (2,), e.g. the adiabatic
        eigenvector of the populated state at x0.

    Returns
    -------
    numpy.ndarray
        Complex wavefunction, shape (ngrid, 2), norm 1.
    """
    dx = grid[1] - grid[0]
    norm = (2.0 * np.pi * sigma * sigma) ** (-0.25)
    envelope = norm * np.exp(-(grid - x0) ** 2 / (4.0 * sigma * sigma)
                             + 1j * k0 * (grid - x0))
    weights = np.asarray(components, dtype=np.complex128).reshape(-1)
    psi = envelope[:, None] * weights[None, :]
    # Renormalize on the discrete grid to keep the norm exact.
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi


def kinetic_half_propagator(grid, mass, dt):
    """Build the momentum-space kinetic half-step factor.

    exp(-i k^2 dt / (4 M)) applied to the FFT of the wavefunction
    realizes exp(-i T dt/2) with T = k^2 / (2 M).

    Parameters
    ----------
    grid : numpy.ndarray
        Coordinate grid, shape (ngrid,).
    mass : float
        Model mass in atomic units.
    dt : float
        Time step in atomic units.

    Returns
    -------
    numpy.ndarray
        Multiplicative factor in momentum space, shape (ngrid,).
    """
    dx = grid[1] - grid[0]
    k = 2.0 * np.pi * np.fft.fftfreq(grid.size, d=dx)
    return np.exp(-1j * k * k * dt / (4.0 * mass))


def adiabatic_states(model, grid):
    """Diagonalize the diabatic Hamiltonian on every grid point.

    Parameters
    ----------
    model : AnalyticModel
        Model providing hamiltonian(x).
    grid : numpy.ndarray
        Coordinate grid, shape (ngrid,).

    Returns
    -------
    energies : numpy.ndarray
        Adiabatic energies per grid point, shape (ngrid, 2),
        sorted ascending.
    eigvecs : numpy.ndarray
        Eigenvectors per grid point, shape (ngrid, 2, 2); column i
        is the adiabatic state i expressed in the diabatic basis.
    """
    hamiltonians = np.stack([model.hamiltonian(x) for x in grid])
    energies, eigvecs = np.linalg.eigh(hamiltonians)
    return energies, eigvecs


def potential_propagator(energies, eigvecs, dt):
    """Build the grid-point wise potential propagator.

    exp(-i V(x) dt) = U(x) diag(exp(-i E(x) dt)) U(x)^T from the
    adiabatic energies and eigenvectors at every grid point.

    Parameters
    ----------
    energies : numpy.ndarray
        Adiabatic energies per grid point, shape (ngrid, 2).
    eigvecs : numpy.ndarray
        Eigenvectors per grid point, shape (ngrid, 2, 2).
    dt : float
        Time step in atomic units.

    Returns
    -------
    numpy.ndarray
        Propagator matrices, shape (ngrid, 2, 2).
    """
    phase = np.exp(-1j * energies * dt)              # (ngrid, 2)
    # U diag(phase) U^T per grid point
    return np.einsum('gij,gj,gkj->gik',
                     eigvecs, phase, eigvecs)


def adiabatic_populations(psi, eigvecs, dx):
    """Adiabatic state populations of the wavepacket.

    Projects psi onto the local adiabatic basis at every grid
    point, psi_ad_i(x) = sum_j U_ij(x) psi_j(x), and integrates
    |psi_ad_i|^2 over the grid.

    Parameters
    ----------
    psi : numpy.ndarray
        Wavefunction in the diabatic basis, shape (ngrid, 2).
    eigvecs : numpy.ndarray
        Eigenvectors per grid point, shape (ngrid, 2, 2); column i
        is the adiabatic state i.
    dx : float
        Grid spacing for norm integration.

    Returns
    -------
    numpy.ndarray
        Adiabatic populations, shape (2,), sum to the norm.
    """
    psi_ad = np.einsum('gji,gj->gi', eigvecs, psi)
    return np.sum(np.abs(psi_ad) ** 2, axis=0) * dx


def propagate(psi, kprop, vprop, nsteps, dt,
              dx, save_every, eigvecs, record=None):
    """Run the Strang split-operator propagation loop.

    Parameters
    ----------
    psi : numpy.ndarray
        Wavefunction, shape (ngrid, 2); propagated in place.
    kprop : numpy.ndarray
        Kinetic half-step factor, shape (ngrid,).
    vprop : numpy.ndarray
        Potential propagator, shape (ngrid, 2, 2).
    nsteps : int
        Number of time steps.
    dt : float
        Time step in atomic units.
    dx : float
        Grid spacing for norm integration.
    save_every : int
        Snapshot stride for the density record.
    eigvecs : numpy.ndarray
        Adiabatic eigenvectors per grid point, shape (ngrid, 2,
        2); the snapshot populations are recorded in this basis.
    record : dict, optional
        Accumulator with 'population', 'norm', 'time', 'density'
        lists; created when omitted.

    Returns
    -------
    dict
        The populated record dictionary.
    """
    if record is None:
        record = {'population': [], 'norm': [],
                  'time': [], 'density': []}

    warned = {'edge': False}

    def snapshot(step):
        populations = adiabatic_populations(psi, eigvecs, dx)
        record['time'].append(step * dt)
        record['population'].append(populations.copy())
        record['norm'].append(float(np.sum(populations)))
        # Density stays in the diabatic basis (visualization aid).
        record['density'].append(np.abs(psi) ** 2)
        if not warned['edge']:
            # ~1% of the grid at each end; a healthy window keeps
            # this at the 1e-30 level.
            nedge = max(1, psi.shape[0] // 100)
            edge_mass = float((np.sum(np.abs(psi[:nedge]) ** 2)
                               + np.sum(np.abs(psi[-nedge:]) ** 2))
                              * dx)
            if edge_mass > EDGE_MASS_THRESHOLD:
                warned['edge'] = True
                warnings.warn(
                    'reference wavepacket reached the grid boundary '
                    '(edge probability mass %.3e): the periodic FFT '
                    'boundary wraps the packet around and corrupts '
                    'the populations; widen ref_xmin/ref_xmax or '
                    'shorten ref_nsteps' % edge_mass)

    snapshot(0)
    for step in range(1, nsteps + 1):
        # kprop is (ngrid,); broadcast across the two components
        psi = np.fft.ifft(kprop[:, None] * np.fft.fft(psi, axis=0),
                          axis=0)
        psi = np.einsum('gij,gj->gi', vprop, psi)
        psi = np.fft.ifft(kprop[:, None] * np.fft.fft(psi, axis=0),
                          axis=0)
        if step % save_every == 0 or step == nsteps:
            snapshot(step)
    return record


def write_reference_files(calcdir, project, grid, record):
    """Write the reference trajectory files into calcdir.

    Parameters
    ----------
    calcdir : str
        Calculation directory (created when missing).
    project : str
        Project name prefixing every output file.
    grid : numpy.ndarray
        Coordinate grid in Bohr.
    record : dict
        Accumulator produced by propagate().

    Returns
    -------
    dict
        Output file paths keyed by population_file, norm_file,
        grid_file, density_file.
    """
    os.makedirs(calcdir, exist_ok=True)
    time = np.asarray(record['time'])
    population = np.asarray(record['population'])
    norm = np.asarray(record['norm'])

    files = {}
    files['population_file'] = os.path.join(
        calcdir, '%s_population.dat' % project)
    np.savetxt(files['population_file'],
               np.column_stack([time, population]),
               header='t(au)  adiabatic population per state')

    files['norm_file'] = os.path.join(
        calcdir, '%s_norm.dat' % project)
    np.savetxt(files['norm_file'],
               np.column_stack([time, norm]),
               header='t(au)  total norm')

    files['grid_file'] = os.path.join(
        calcdir, '%s_grid.dat' % project)
    np.savetxt(files['grid_file'], grid,
               header='coordinate grid (Bohr)')

    files['density_file'] = os.path.join(
        calcdir, '%s_density.npy' % project)
    np.save(files['density_file'], np.asarray(record['density']))
    return files


def run_reference(model, coordinates, calcdir, project,
                  velocities=None, current_state=0, *,
                  model_mass, wavepacket_sigma, ref_ngrid, ref_dt,
                  ref_nsteps, ref_save_every, wavepacket_k0,
                  ref_xmin=None, ref_xmax=None):
    """Run the full split-operator reference dynamics.

    Parameters
    ----------
    model : AnalyticModel
        Model providing the propagated potential surfaces.
    coordinates : array-like
        Framework coordinates in Angstrom; the first atom x
        component seeds the packet center x0.
    calcdir : str
        Calculation directory receiving the output files.
    project : str
        Project name prefixing the output files.
    velocities : array-like, optional
        Framework velocities (Bohr/au); when present the initial
        momentum is k0 = model_mass * velocities[0][0], otherwise
        wavepacket_k0 is used.
    current_state : int
        Initially populated adiabatic state index (the packet
        carries the local adiabatic eigenvector at x0 as its
        diabatic component weights).
    model_mass, wavepacket_sigma, ref_ngrid, ref_dt, ref_nsteps,
    ref_save_every, wavepacket_k0 : number
        The skill config's typed fields, taken as they are: the schema
        validated their types, and re-coercing them here was a second
        conversion for no new answer.
    ref_xmin, ref_xmax : number, optional
        Grid window; None falls back on [REF_XMIN_DEFAULT,
        REF_XMAX_DEFAULT]. x0 outside the window raises an error.

    Returns
    -------
    dict
        Final summary (final_population, final_norm, grid and time
        parameters, k0/x0 actually used) plus the output file paths.
    """
    mass = model_mass
    sigma = wavepacket_sigma
    ngrid = ref_ngrid
    dt = ref_dt
    nsteps = ref_nsteps
    save_every = max(1, int(ref_save_every))

    x0 = model_coordinate(coordinates)
    if velocities is not None and len(velocities) > 0:
        k0 = mass * float(velocities[0][0])
    else:
        k0 = wavepacket_k0

    xmin = REF_XMIN_DEFAULT if ref_xmin is None else ref_xmin
    xmax = REF_XMAX_DEFAULT if ref_xmax is None else ref_xmax
    if not xmin < x0 < xmax:
        raise ValueError(
            'wavepacket center x0=%.6f Bohr lies outside the grid '
            'window [%.3f, %.3f] Bohr' % (x0, xmin, xmax))

    grid = build_grid(ngrid, xmin, xmax)
    dx = grid[1] - grid[0]
    # Grid Nyquist bound on the packet momentum: sampling
    # exp(i k0 x) with spacing dx aliases every |k0| above
    # pi/dx onto the mirrored band (a +77 au packet on dx =
    # 0.054 Bohr propagates backwards as -40), silently
    # inverting the reference dynamics. Refuse instead of
    # propagating a mirrored packet; widen ref_ngrid.
    k_nyquist = np.pi / dx
    if abs(k0) >= k_nyquist:
        raise ValueError(
            'wavepacket momentum |k0|=%.3f exceeds the grid '
            'Nyquist bound pi/dx=%.3f (ngrid=%d over [%.3f, '
            '%.3f]); increase ref_ngrid so the momentum is '
            'representable' % (abs(k0), k_nyquist, ngrid, xmin,
                               xmax))
    # current_state indexes the ADIABATIC state (the production
    # md root semantics): the packet carries the local adiabatic
    # eigenvector at x0 as its diabatic components. Populating a
    # diabatic component instead misinitializes the reference
    # whenever the diabatic/adiabatic characters disagree (tully3
    # at x0 = -20 places diabatic 0 on the excited adiabatic
    # state, silently swapping the initial populations).
    _e0, eigvec0 = np.linalg.eigh(model.hamiltonian(x0))
    psi = initial_wavepacket(grid, x0, sigma, k0,
                             eigvec0[:, int(current_state)])
    kprop = kinetic_half_propagator(grid, mass, dt)
    energies, eigvecs = adiabatic_states(model, grid)
    vprop = potential_propagator(energies, eigvecs, dt)

    record = propagate(psi, kprop, vprop, nsteps, dt,
                       dx, save_every, eigvecs)
    files = write_reference_files(calcdir, project, grid, record)

    results = {
        'final_population': record['population'][-1].tolist(),
        'final_norm': record['norm'][-1],
        'nsteps': nsteps,
        'dt': dt,
        'ngrid': ngrid,
        'xmin': float(grid[0]),
        'xmax': float(grid[-1]),
        'x0': x0,
        'k0': k0,
        'wavepacket_sigma': sigma,
        'initial_state': int(current_state),
    }
    results.update(files)
    return results
