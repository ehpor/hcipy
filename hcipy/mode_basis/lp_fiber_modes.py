import numpy as np
from scipy.special import jv, kn
from scipy.optimize import fsolve

from ..mode_basis import ModeBasis

def eigenvalue_equation(u, m, V):
    '''Evaluates the eigenvalue equation for a circular step-index fiber.

    Parameters
    ----------
    u : scalar
        The normalized propagation constant.
    m : int
        The azimuthal order
    V : scalar
        The normalized frequency parameter of the fiber.

    Returns
    -------
    scalar
        The eigenvalue equation value
    '''
    w = np.sqrt(V**2 - u**2)
    return jv(m, u) / (u * jv(m + 1, u)) - kn(m, w) / (w * kn(m + 1, w))

def find_branch_cuts(m, V):
    '''Find all the solutions for the eigenvalue function.

    Parameters
    ----------
    m : int
        The azimuthal order
    V : scalar
        The normalized frequency parameter of the fiber.

    Returns
    -------
    Tuple
        A tuple containing the solutions of the eigenvalue function. If no solutions were found returns None.
    '''
    # Make an initial rough grid
    num_steps = 501
    theta = np.linspace(np.pi * 0.499, 0, num_steps, endpoint=False)
    u = V * np.cos(theta)

    # Find the position where the eigenvalue equation goes through zero
    diff = eigenvalue_equation(u, m, V)
    fu = np.diff(np.sign(diff)) < 0
    ind = np.where(abs(fu - 1) <= 0.01)[0]

    if len(ind) > 0:
        # Refine the zero with a rootfinding algorithm
        u0 = fsolve(eigenvalue_equation, u[ind], args=(m, V))
        w0 = np.sqrt(V**2 - u0**2)
        return u0, w0
    else:
        return None

def lp_radial(m, u, w, r):
    '''Evaluates the radial profile of the LP modes.

    Parameters
    ----------
    m : int
        The azimuthal order
    u : scalar
        The normalized inner propagation constant.
    w : scalar
        The normalized outer propagation constant.
    r : array_like
        The radial coordinates on which to evaluate the bessel modes.

    Returns
    -------
    array_like
        An array that contains the radial profile.
    '''
    # The scaling factor for the continuity condition
    scaling_factor = jv(m, u) / kn(m, w)

    # Find the grid inside and outside the core radius
    mask = r < 1

    # Evaluate the radial mode profile
    mode_field = np.zeros_like(r)
    mode_field[mask] = jv(m, u * r[mask])
    mode_field[~mask] = scaling_factor * kn(m, w * r[~mask])

    return mode_field

def lp_azimuthal(m, theta):
    '''Evaluates the azimuthal profile of the LP modes.

    Parameters
    ----------
    m : int
        The azimuthal order
    theta : array_like
        The azimuthal coordinates on which to evaluate the cosine and sine modes.

    Returns
    -------
    array_like
        An array that contains the azimuthal profile.
    '''
    if m >= 0:
        return np.cos(m * theta)
    else:
        return np.sin(m * theta)

def make_lp_modes(grid, V_number, core_radius, return_betas=False):
    '''Make a ModeBasis out of the guided modes that are supported by a step-index fiber.

    This function solves the eigenvalue equation of for a radial step-index profile fiber.
    It calculates the mode profiles and can also return the propagation constants.

    Parameters
    ----------
    grid : Grid
        The grid on which to calculate the mode basis.
    V_number : scalar
        The normalized frequency parameter of the fiber.
    core_radius : scalar
        The core radius of a step-index fiber.
    return_betas : bool
        If true this function also returns the propagation constants.
    Returns
    -------
    ModeBasis
        An ModeBasis containing all supported LP modes.
    array_like
        An array containing all propagation constants. This is only returned if return_betas is True.
    '''
    finding_new_modes = True
    m = 0
    num_modes = 0

    # scaled grid
    R, Theta = grid.scaled(1 / core_radius).as_('polar').coords
    modes = []
    betas = []
    while finding_new_modes:

        solutions = find_branch_cuts(m, V_number)
        if solutions is not None:
            for ui, wi in zip(solutions[0], solutions[1]):

                radial_prodile = lp_radial(m, ui, wi, R)
                beta = np.sqrt((V_number**2 + wi**2 - ui**2) / (2 * core_radius**2))

                ms = [m, -m] if m > 0 else [m]
                for mi in ms:
                    azimutal_profile = lp_azimuthal(mi, Theta)
                    mode_profile = radial_prodile * azimutal_profile

                    # Normalize the mode numerically as there is no analytical normalization
                    norm = np.sqrt(np.sum(mode_profile * mode_profile.conj() * grid.weights))
                    mode_profile /= norm
                    modes.append(mode_profile)
                    betas.append(beta)
                    num_modes += 1

            m += 1
        else:
            finding_new_modes = False

    # Sort the modes according to their propagation constant
    betas = np.array(betas)
    index_sorting = np.argsort(betas)[::-1]
    modes = np.array(modes)[index_sorting].T
    betas = betas[index_sorting]

    if return_betas:
        return ModeBasis(modes, grid), np.array(betas)
    else:
        return ModeBasis(modes, grid)

def make_LP_modes(*args, **kwargs):  # noqa: N802
    import warnings
    warnings.warn('Use the lower-case version "make_lp_modes" instead.', DeprecationWarning, stacklevel=2)

    return make_lp_modes(*args, **kwargs)
