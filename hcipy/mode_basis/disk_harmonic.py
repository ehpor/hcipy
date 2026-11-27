import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros

from .mode_basis import ModeBasis
from ..aperture import make_circular_aperture
from ..field import Field

def disk_harmonic(n, m, D=1, bc='dirichlet', grid=None):
    '''Create a disk harmonic.

    Parameters
    ----------
    n : int
        Radial order
    m : int
        Azimuthal order
    D : scalar
        The diameter of the pupil.
    bc : string
        The boundary conditions to use. This can be either 'dirichlet', or
        'neumann' for a Dirichlet or Neumann boundary condition respectively.
    grid : Grid
        The grid on which to evaluate the function.

    Returns
    -------
    Field
        The disk harmonic function evaluated on `grid`.

    Raises
    ------
    ValueError
        If the boundary condition is not recognized.

    Notes
    -----
    The mode is normalized to unit RMS over the disk, that is,
    :math:`\\frac{1}{\\pi (D/2)^2} \\int_{\\mathrm{disk}} z^2 \\, \\mathrm{d}A = 1`,
    where the integral is approximated using the grid area weights.
    '''
    polar_grid = grid.as_('polar')
    r = 2 * polar_grid.r / D
    theta = polar_grid.theta

    m_negative = m < 0
    m = abs(m)

    if bc == 'dirichlet':
        lambda_mn = jn_zeros(m, n)[-1]
    elif bc == 'neumann':
        if n == 0 and m == 0:
            # The constant mode is the only Neumann mode with zero eigenvalue.
            lambda_mn = 0
        else:
            lambda_mn = jnp_zeros(m, n)[-1]
    else:
        raise ValueError('Boundary condition not recognized.')

    if m_negative:
        z = jv(m, lambda_mn * r) * np.sin(m * theta)
    else:
        z = jv(m, lambda_mn * r) * np.cos(m * theta)

    mask = np.asarray(make_circular_aperture(D)(grid)) > 0.5
    z = z * mask

    # Normalize to unit RMS over the disk.
    area = np.pi * (D / 2)**2
    rms = np.sqrt(np.sum(z**2 * grid.weights) / area)

    return Field(z / rms, grid)

def disk_harmonic_energy(n, m, bc='dirichlet'):
    '''Get the energy of a disk harmonic function.

    This allows for functions to sort a disk harmonic mode basis on energy.

    Parameters
    ----------
    n : int
        Radial order
    m : int
        Azimuthal order
    bc : string
        The boundary conditions to use. This can be either 'dirichlet', or
        'neumann' for a Dirichlet or Neumann boundary condition respectively.

    Returns
    -------
    scalar
        The energy corresponding to the mode.

    Raises
    ------
    ValueError
        If the boundary condition is not recognized.
    '''
    m = abs(m)

    if bc == 'dirichlet':
        lambda_mn = jn_zeros(m, n)[-1]
    elif bc == 'neumann':
        if n == 0 and m == 0:
            lambda_mn = 0
        else:
            lambda_mn = jnp_zeros(m, n)[-1]
    else:
        raise ValueError('Boundary condition not recognized.')

    return lambda_mn**2

def get_disk_harmonic_orders_sorted(num_modes, bc='dirichlet'):
    '''Return the orders for the first `num_modes` modes.

    Parameters
    ----------
    num_modes : int
        The number of modes to return.
    bc : string
        The boundary conditions to use. This can be either 'dirichlet', or
        'neumann' for a Dirichlet or Neumann boundary condition respectively.

    Returns
    -------
    list of tuples
        A list of orders (n, m).
    '''
    if bc == 'dirichlet':
        # For Dirichlet boundary conditions the piston mode is identically zero
        # and therefore not part of the basis.
        orders = [(1, 0)]
    elif bc == 'neumann':
        # For Neumann boundary conditions the constant mode is a valid mode.
        orders = [(0, 0)]
    else:
        raise ValueError('Boundary condition not recognized.')

    energies = [disk_harmonic_energy(order[0], order[1], bc) for order in orders]

    results = []
    while len(results) < num_modes:
        k = np.argmin(energies)
        order = orders[k]

        if order[1] != 0:
            results.append((order[0], -order[1]))
        results.append(order)

        del orders[k]
        del energies[k]

        new_order = (order[0], order[1] + 1)
        # Radial order zero only exists for the constant Neumann mode.
        if new_order[0] > 0 and new_order not in results and new_order not in orders:
            orders.append(new_order)
            energies.append(disk_harmonic_energy(new_order[0], new_order[1], bc))
        new_order = (order[0] + 1, order[1])
        if new_order not in results and new_order not in orders:
            orders.append(new_order)
            energies.append(disk_harmonic_energy(new_order[0], new_order[1], bc))

    return results[:num_modes]

def make_disk_harmonic_basis(grid, num_modes, D=1, bc='dirichlet'):
    '''Create a disk harmonic mode basis.

    Parameters
    ----------
    grid : Grid
        The grid on which to evaluate the disk harmonic modes.
    num_modes : int
        The number of modes to create.
    D : scalar
        The diameter of the disk.
    bc : string
        The boundary conditions to use. This can be either 'dirichlet', or
        'neumann' for a Dirichlet or Neumann boundary condition respectively.

    Returns
    -------
    ModeBasis
        The evaluated disk harmonic modes.
    '''
    orders = get_disk_harmonic_orders_sorted(num_modes, bc)

    modes = [disk_harmonic(order[0], order[1], D, bc, grid) for order in orders]
    return ModeBasis(modes, grid)
