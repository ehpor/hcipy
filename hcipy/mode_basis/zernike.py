import numpy as np
import scipy.linalg
from math import sqrt

from .mode_basis import ModeBasis
from ..aperture import make_hexagonal_aperture, make_wedge_aperture

def noll_to_zernike(i):
    '''Get the Zernike index from a Noll index.

    Parameters
    ----------
    i : int
        The Noll index.

    Returns
    -------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.
    '''
    n = int(sqrt(2 * i - 1) + 0.5) - 1
    if n % 2:
        m = 2 * int((2 * (i + 1) - n * (n + 1)) // 4) - 1
    else:
        m = 2 * int((2 * i + 1 - n * (n + 1)) // 4)
    return n, m * (-1)**(i % 2)

def ansi_to_zernike(i):
    '''Get the Zernike index from an ANSI index.

    Parameters
    ----------
    i : int
        The ANSI index.

    Returns
    -------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.
    '''
    n = int((sqrt(8 * i + 1) - 1) / 2)
    m = 2 * i - n * (n + 2)
    return (n, m)

def zernike_to_ansi(n, m):
    '''Get the ANSI index for a pair of Zernike indices.

    Parameters
    ----------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.

    Returns
    -------
    int
        The ANSI index.
    '''
    return (m + n * n) // 2 + n

def zernike_to_noll(n, m):
    '''Get the Noll index for a pair of Zernike indices.

    Parameters
    ----------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.

    Returns
    -------
    int
        The Noll index.
    '''
    i = int(((n + 0.5)**2 + 1) / 2) + 1
    Nn = (n + 1) * (n + 2) // 2 + 1

    # Brute force search
    for j in range(i, i + Nn):
        nn, mm = noll_to_zernike(j)
        if nn == n and mm == m:
            return j
    raise ValueError(f'Could not find noll index for n={n}, m={m}.')

def zernike_radial_indices(num_modes):
    """
    Return the first `num_modes` unique (n, m) pairs for
    radial Zernike polynomials.

    The ordering is:
        n = 0, m = 0
        n = 1, m = 1
        n = 2, m = 0, 2
        n = 3, m = 1, 3
        n = 4, m = 0, 2, 4
        ...

    Parameters
    ----------
    num_modes : int
        Number of unique radial Zernike polynomials.

    Returns
    -------
    list of tuple
        List of (n, m) pairs.
    """
    indices = []

    n = 0
    while len(indices) < num_modes:
        for m in range(n % 2, n + 1, 2):
            indices.append((n, m))

            if len(indices) == num_modes:
                break

        n += 1

    return indices

def zernike_radial_chong(n, m, r, cache=None):
    '''The radial component of a Zernike polynomial.

    We use the q-recursive method, which uses recurrence relations to calculate the radial
    Zernike polynomials without using factorials. A description of the method can be found
    in [Chong2003]_. Additionally, this function optionally caches results of previous calls.

    .. [Chong2003] Chong, C. W., Raveendran, P., & Mukundan, R. (2003). A comparative analysis of algorithms
        for fast computation of Zernike moments. Pattern Recognition, 36(3), 731-742.

    Parameters
    ----------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.
    r : array_like
        The (normalized) radial coordinates on which to calculate the polynomial.
    cache : dictionary or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    array_like
        The radial component of the evaluated Zernike polynomial.
    '''
    m = abs(m)

    if cache is not None:
        if ('rad', n, m) in cache:
            return cache[('rad', n, m)]

    if n == m:
        res = r**n
    elif (n - m) == 2:
        z1 = zernike_radial_chong(n, n, r, cache)
        z2 = zernike_radial_chong(n - 2, n - 2, r, cache)

        res = n * z1 - (n - 1) * z2
    else:
        p = n
        q = m + 4

        h3 = -4 * (q - 2) * (q - 3) / float((p + q - 2) * (p - q + 4))
        h2 = h3 * (p + q) * (p - q + 2) / float(4 * (q - 1)) + (q - 2)
        h1 = q * (q - 1) / 2.0 - q * h2 + h3 * (p + q + 2) * (p - q) / 8.0

        r2 = zernike_radial_chong(2, 2, r, cache)
        res = h1 * zernike_radial_chong(p, q, r, cache) + (h2 + h3 / r2) * zernike_radial_chong(n, q - 2, r, cache)

    if cache is not None:
        cache[('rad', n, m)] = res

    return res

def zernike_radial_andersen(n, m, r, cache=None):
    """
    The radial component of a Zernike polynomial.

    This implementation uses the Cartesian/polar-compatible recurrence
    described by Andersen [Andersen2018], rather than the q-recursive
    recurrence that contains a 1/r**2 term.

    In particular, it uses

        R_n^m = r * (R_{n-1}^{m-1} + R_{n-1}^{m+1}) - R_{n-2}^m

    with the appropriate boundary conditions.

    This recurrence contains no division by r and therefore remains
    well-defined when r == 0.

    .. [Andersen2018] Andersen, T. B. (2018). Efficient and robust recurrence relations
      for the Zernike circle polynomials and their derivatives in Cartesian coordinates. 
      Optics Express, 26(15), 18878-18896.

    Parameters
    ----------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.
    r : array_like
        The (normalized) radial coordinates on which to calculate the polynomial.
    cache : dict or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    array_like
        The radial component of the evaluated Zernike polynomial.
    """

    m = abs(m)

    # Invalid Zernike combinations.
    if m > n or (n - m) % 2 != 0:
        return np.zeros_like(r)

    if cache is not None:
        key = ('rad', n, m)

        if key in cache:
            return cache[key]

    # We calculate all radial polynomials up to n.
    # The Andersen recurrence couples radial orders n-1 and n-2,
    # so calculating the whole triangular set is considerably simpler
    # than trying to recursively evaluate individual terms.

    R = {}

    # Starting polynomials:
    # R_0^0 = 1
    # R_1^1 = r
    R[(0, 0)] = np.ones_like(r)

    if n >= 1:
        R[(1, 1)] = r

    for nn in range(2, n + 1):

        # Only m values with the same parity as n are valid.
        for mm in range(nn % 2, nn + 1, 2):

            if mm == 0:
                R[(nn, 0)] = 2 * r * R[(nn - 1, 1)] - R[(nn - 2, 0)]
            elif mm == nn:
                R[(nn, mm)] = r * R[(nn - 1, mm - 1)]
            else:
                R[(nn, mm)] = r * (R[(nn - 1, mm-1)] + R[(nn - 1, mm + 1)]) - R[(nn - 2, mm)]

    result = R[(n, m)]

    if cache is not None:
        cache[('rad', n, m)] = result

    return result

def zernike_radial(n, m, r, cache=None, use_andersen=True):
    """
    The radial component of a Zernike polynomial.

    Parameters
    ----------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.
    r : array_like
        The (normalized) radial coordinates on which to calculate the polynomial.
    cache : dict or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.
    use_andersen : boolean
        This function uses the Andersen recurrence relationship if True and otherwise the Chong one.

    Returns
    -------
    array_like
        The radial component of the evaluated Zernike polynomial.
    """
    if use_andersen:
        return zernike_radial_andersen(n, m, r, cache)
    else:
        return zernike_radial_chong(n, m, r, cache)

def zernike_azimuthal(m, theta, cache=None):
    '''The azimuthal component of a Zernike polynomial.

    This function optionally caches results of previous calls.

    Parameters
    ----------
    m : int
        The azimuthal Zernike order.
    theta : array_like
        The azimuthal coordinates on which to calculate the polynomial.
    cache : dictionary or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    array_like
        The azimuthal component of the evaluated Zernike polynomial.
    '''
    if cache is not None:
        if ('azim', m) in cache:
            return cache[('azim', m)]

    if m < 0:
        res = sqrt(2) * np.sin(-m * theta)
    elif m == 0:
        return 1
    else:
        res = sqrt(2) * np.cos(m * theta)

    if cache is not None:
        cache[('azim', m)] = res

    return res

def zernike(n, m, D=1, grid=None, radial_cutoff=True, cache=None):
    '''Evaluate the Zernike polynomial on a grid.

    Parameters
    ----------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.
    D : scalar
        The diameter of the Zernike polynomial.
    grid : Grid
        The grid on which to evaluate the Zernike polynomial. If this is None,
        a Field generator will be returned.
    radial_cutoff : boolean
        Whether to apply a circular aperture to cutoff the modes.
    cache : dictionary or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    Field or Field generator
        The evaluated Zernike polynomial. If `grid` is None, a Field generator is returned,
        which evaluates the Zernike polynomial on the supplied grid.
    '''
    from ..field import Field

    if grid is None:
        return lambda grid: zernike(n, m, D, grid, radial_cutoff, cache)

    if grid.is_separated and grid.is_('polar'):
        R, Theta = grid.separated_coords
        z_r = zernike_radial(n, m, 2 * R / D, cache)
        if radial_cutoff:
            z_r *= (2 * R) < D
        z = sqrt(n + 1) * np.outer(zernike_azimuthal(m, Theta, cache), z_r).flatten()
    else:
        r, theta = grid.as_('polar').coords
        z = sqrt(n + 1) * zernike_azimuthal(m, theta, cache) * zernike_radial(n, m, 2 * r / D, cache)
        if radial_cutoff:
            z *= (2 * r) < D

    return Field(z, grid)

def zernike_ansi(i, D=1, grid=None, radial_cutoff=True, cache=None):
    '''Evaluate the Zernike polynomial on a grid using an ANSI index.

    Parameters
    ----------
    i : int
        The ANSI index.
    D : scalar
        The diameter of the Zernike polynomial.
    grid : Grid or None
        The grid on which to evaluate the Zernike polynomial. If this is None,
        a Field generator will be returned.
    radial_cutoff : boolean
        Whether to apply a circular aperture to cutoff the modes.
    cache : dictionary or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    Field or Field generator
        The evaluated Zernike polynomial. If `grid` is None, a Field generator is returned,
        which evaluates the Zernike polynomial on the supplied grid.
    '''
    n, m = ansi_to_zernike(i)
    return zernike(n, m, D, grid, radial_cutoff, cache)

def zernike_noll(i, D=1, grid=None, radial_cutoff=True, cache=None):
    '''Evaluate the Zernike polynomial on a grid using a Noll index.

    Parameters
    ----------
    i : int
        The Noll index.
    D : scalar
        The diameter of the Zernike polynomial.
    grid : Grid or None
        The grid on which to evaluate the Zernike polynomial. If this is None,
        a Field generator will be returned.
    radial_cutoff : boolean
        Whether to apply a circular aperture to cutoff the modes.
    cache : dictionary or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    Field or Field generator
        The evaluated Zernike polynomial. If `grid` is None, a Field generator is returned,
        which evaluates the Zernike polynomial on the supplied grid.
    '''
    n, m = noll_to_zernike(i)
    return zernike(n, m, D, grid, radial_cutoff, cache)

def make_zernike_basis(num_modes, D, grid, starting_mode=1, ansi=False, radial_cutoff=True, use_cache=True, cache=None):
    '''Make a ModeBasis of Zernike polynomials.

    Parameters
    ----------
    num_modes : int
        The number of Zernike polynomials to generate.
    D : scalar
        The diameter of the Zernike polynomial.
    grid : Grid or None
        The grid on which to evaluate the Zernike polynomials. If this is None,
        a list of Field generators will be returned.
    starting_mode : int
        The first mode to evaluate.
    ansi : boolean
        If this is True, the modes will be indexed using ANSI indices. Otherwise, a Noll
        indexing scheme is used.
    radial_cutoff : boolean
        Whether to apply a circular aperture to cutoff the modes.
    use_cache : boolean
        Whether to use a cache while calculating the modes. A cache uses memory, so turn it
        off when you are limited on memory.

    Returns
    -------
    ModeBasis or list of Field generators
        The evaluated mode basis of Zernike polynomials, or a list of Field generators for
        each of the Zernike polynomials.
    '''
    from .mode_basis import ModeBasis
    f = zernike_ansi if ansi else zernike_noll

    if grid is None:
        polar_grid = None
    else:
        polar_grid = grid.as_('polar')

    if use_cache:
        if cache is None:
            cache = {}
    else:
        cache = None

    modes = [f(i, D, polar_grid, radial_cutoff, cache) for i in range(starting_mode, starting_mode + num_modes)]

    if grid is None:
        return modes
    else:
        return ModeBasis(modes, grid)

def hexike(n, m, circum_diameter, hexagon_angle=0, grid=None, cache=None):
    '''Evaluate a Hexike polynomial on a grid.

    Parameters
    ----------
    n : int
        The radial Hexike order.
    m : int
        The azimuthal Hexike order.
    circum_diameter : scalar
        The circum-diameter of the hexagonal aperture.
    hexagon_angle : float
        The angle of the hexagon. At an angle of 0, the point of the hexagon points up.
    grid : Grid or None
        The grid on which to evaluate the Hexike polynomial. If this is None, a Field
        generator will be returned.
    cache : dictionary or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    Field or Field generator
        The evaluated Hexike polynomial. If `grid` is None, a Field generator is returned,
        which evaluates the Hexike polynomial on the supplied grid.
    '''
    if grid is None:
        return lambda grid: hexike(n, m, circum_diameter, hexagon_angle, grid, cache)

    noll_index = zernike_to_noll(n, m)

    basis = make_hexike_basis(grid, noll_index, circum_diameter, hexagon_angle, cache)

    return basis[-1]

def hexike_ansi(i, circum_diameter, hexagon_angle=0, grid=None, cache=None):
    '''Evaluate a Hexike polynomial on a grid using an ANSI index.

    Parameters
    ----------
    i : int
        The ANSI index.
    circum_diameter : scalar
        The circum-diameter of the hexagonal aperture.
    hexagon_angle : float
        The angle of the hexagon. At an angle of 0, the point of the hexagon points up.
    grid : Grid or None
        The grid on which to evaluate the Hexike polynomial. If this is None, a Field
        generator will be returned.
    cache : dictionary or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    Field or Field generator
        The evaluated Hexike polynomial. If `grid` is None, a Field generator is returned,
        which evaluates the Hexike polynomial on the supplied grid.
    '''
    n, m = ansi_to_zernike(i)
    return hexike(n, m, circum_diameter, hexagon_angle, grid, cache)

def hexike_noll(i, circum_diameter, hexagon_angle=0, grid=None, cache=None):
    '''Evaluate a Hexike polynomial on a grid using a Noll index.

    Parameters
    ----------
    i : int
        The Noll index.
    circum_diameter : scalar
        The circum-diameter of the hexagonal aperture.
    hexagon_angle : float
        The angle of the hexagon. At an angle of 0, the point of the hexagon points up.
    grid : Grid or None
        The grid on which to evaluate the Hexike polynomial. If this is None, a Field
        generator will be returned.
    cache : dictionary or None
        A dictionary containing previously calculated Zernike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.

    Returns
    -------
    Field or Field generator
        The evaluated Hexike polynomial. If `grid` is None, a Field generator is returned,
        which evaluates the Hexike polynomial on the supplied grid.
    '''
    n, m = noll_to_zernike(i)
    return hexike(n, m, circum_diameter, hexagon_angle, grid, cache)

def make_hexike_basis(grid, num_modes, circum_diameter, hexagon_angle=0, cache=None):
    '''Make a hexike basis.

    This is based on [Mahajan2006]_. This function creates a Zernike mode basis, then
    numerically orthogonalizes it using Gram-Schmidt orthogonalization.

    .. [Mahajan2006] Virendra N. Mahajan and Guang-ming Dai,
        "Orthonormal polynomials for hexagonal pupils," Opt.
        Lett. 31, 2462-2464 (2006)

    Parameters
    ----------
    grid : hcipy.Grid
        The grid on which to compute the mode basis.
    num_modes : int
        The number of hexike modes to compute.
    circum_diameter : float
        The circum-diameter of the hexagonal aperture.
    hexagon_angle : float
        The angle of the hexagon. At an angle of 0, the point of the hexagon points up.
    cache : dictionary or None
        A dictionary containing previously calculated Hexike modes on the same grid.
        This function is for speedup only, and therefore the cache is expected to be
        valid. You can reuse the cache for future calculations on the same exact grid.
        The given dictionary is updated with the current calculation.
    '''
    if cache is None:
        cache = {}

    zernike_basis = make_zernike_basis(num_modes, circum_diameter, grid, cache=cache)
    hexagonal_aperture = make_hexagonal_aperture(circum_diameter, hexagon_angle)(grid)

    sqrt_weights = np.sqrt(grid.weights)

    if np.isscalar(sqrt_weights):
        sqrt_weights = np.array([sqrt_weights])

    # Un-normalize the zernike modes using the weights of the grid.
    zernike_basis.transformation_matrix *= np.array(hexagonal_aperture * sqrt_weights)[:, np.newaxis]

    # The QR decomposition is expensive. Let's get it from the cache if already computed.
    if 'R' in cache and cache['R'].shape[0] >= num_modes:
        R = cache['R'][:num_modes, :num_modes]

        Q = scipy.linalg.solve_triangular(R, zernike_basis.transformation_matrix.T, trans=1, lower=False, overwrite_b=True).T
    else:
        # Perform Gramm-Schmidt orthogonalization using a QR decomposition.
        Q, R = np.linalg.qr(zernike_basis.transformation_matrix)

        cache['R'] = R.copy()

    # Correct for negative sign of components of the Q matrix.
    hexike_basis = ModeBasis(Q / np.sign(np.diag(R)), grid)

    # Renormalize the resulting functions using the area of a hexagon and the grid weights.
    area_hexagon = 3 * np.sqrt(3) / 8 * circum_diameter**2
    hexike_basis.transformation_matrix *= np.array(np.sqrt(area_hexagon) / sqrt_weights)[:, np.newaxis]

    return hexike_basis

def keystone(n, m, inner_diameter, outer_diameter, angle_width, spider_width, keystone_angle=0, grid=None, cache=None):
    '''Evaluate a keystone mode on a grid.

    Parameters
    ----------
    n : int
        The radial keystone order.
    m : int
        The azimuthal keystone order.
    inner_diameter : scalar
        The inner diameter of the keystone aperture.
    outer_diameter : scalar
        The outer diameter of the keystone aperture.
    angle_width : scalar
        The angular width of the keystone aperture in degrees.
    spider_width : scalar
        The full width of the spiders bordering the keystone aperture.
    keystone_angle : float
        The angle of the keystone aperture in radians. At an angle of 0, the center
        of the keystone points to the right.
    grid : Grid or None
        The grid on which to evaluate the keystone mode. If this is None, a Field
        generator will be returned.
    cache : dictionary or None
        A dictionary of intermediate calculations for this exact grid, aperture
        geometry, and orientation. The dictionary is updated in place.

    Returns
    -------
    Field or Field generator
        The evaluated keystone mode. If grid is None, a Field generator is returned,
        which evaluates the keystone mode on the supplied grid.
    '''
    if grid is None:
        return lambda grid: keystone(n, m, inner_diameter, outer_diameter, angle_width, spider_width, keystone_angle, grid, cache)

    noll_index = zernike_to_noll(n, m)

    basis = make_keystone_basis(grid, noll_index, inner_diameter, outer_diameter, angle_width, spider_width, keystone_angle, cache)

    return basis[-1]

def keystone_ansi(i, inner_diameter, outer_diameter, angle_width, spider_width, keystone_angle=0, grid=None, cache=None):
    '''Evaluate a keystone mode on a grid using an ANSI index.

    Parameters
    ----------
    i : int
        The ANSI index.
    inner_diameter : scalar
        The inner diameter of the keystone aperture.
    outer_diameter : scalar
        The outer diameter of the keystone aperture.
    angle_width : scalar
        The angular width of the keystone aperture in degrees.
    spider_width : scalar
        The full width of the spiders bordering the keystone aperture.
    keystone_angle : float
        The angle of the keystone aperture in radians. At an angle of 0, the center
        of the keystone points to the right.
    grid : Grid or None
        The grid on which to evaluate the keystone mode. If this is None, a Field
        generator will be returned.
    cache : dictionary or None
        A dictionary of intermediate calculations for this exact grid, aperture
        geometry, and orientation. The dictionary is updated in place.

    Returns
    -------
    Field or Field generator
        The evaluated keystone mode. If grid is None, a Field generator is returned,
        which evaluates the keystone mode on the supplied grid.
    '''
    n, m = ansi_to_zernike(i)
    return keystone(n, m, inner_diameter, outer_diameter, angle_width, spider_width, keystone_angle, grid, cache)

def keystone_noll(i, inner_diameter, outer_diameter, angle_width, spider_width, keystone_angle=0, grid=None, cache=None):
    '''Evaluate a keystone mode on a grid using a Noll index.

    Parameters
    ----------
    i : int
        The Noll index.
    inner_diameter : scalar
        The inner diameter of the keystone aperture.
    outer_diameter : scalar
        The outer diameter of the keystone aperture.
    angle_width : scalar
        The angular width of the keystone aperture in degrees.
    spider_width : scalar
        The full width of the spiders bordering the keystone aperture.
    keystone_angle : float
        The angle of the keystone aperture in radians. At an angle of 0, the center
        of the keystone points to the right.
    grid : Grid or None
        The grid on which to evaluate the keystone mode. If this is None, a Field
        generator will be returned.
    cache : dictionary or None
        A dictionary of intermediate calculations for this exact grid, aperture
        geometry, and orientation. The dictionary is updated in place.

    Returns
    -------
    Field or Field generator
        The evaluated keystone mode. If grid is None, a Field generator is returned,
        which evaluates the keystone mode on the supplied grid.
    '''
    n, m = noll_to_zernike(i)
    return keystone(n, m, inner_diameter, outer_diameter, angle_width, spider_width, keystone_angle, grid, cache)

def _keystone_geometry(inner_diameter, outer_diameter, angle_width, spider_width):
    '''Calculate the area, radial centroid and Zernike diameter of a keystone aperture.'''
    inner_radius = inner_diameter / 2
    outer_radius = outer_diameter / 2
    half_angle = np.radians(angle_width) / 2
    half_spider_width = spider_width / 2

    if inner_radius < 0:
        raise ValueError('inner_diameter must be non-negative.')
    if outer_radius <= inner_radius:
        raise ValueError('outer_diameter must be larger than inner_diameter.')
    if angle_width <= 0 or angle_width > 360:
        raise ValueError('angle_width must be larger than zero and at most 360 degrees.')
    if spider_width < 0:
        raise ValueError('spider_width must be non-negative.')

    if half_spider_width == 0:
        area = half_angle * (outer_radius**2 - inner_radius**2)
        first_moment = 2 / 3 * np.sin(half_angle) * (outer_radius**3 - inner_radius**3)
    else:
        def area_term(radius):
            return radius**2 * np.arcsin(half_spider_width / radius) + half_spider_width * np.sqrt(radius**2 - half_spider_width**2)

        def first_moment_term(radius):
            return 2 / 3 * np.sin(half_angle) * (radius**2 - half_spider_width**2)**1.5 - half_spider_width * np.cos(half_angle) * radius**2

        area = 0
        first_moment = 0

        if half_angle > np.pi / 2 and inner_radius < half_spider_width:
            transition_radius = min(outer_radius, half_spider_width)
            area += (half_angle - np.pi / 2) * (transition_radius**2 - inner_radius**2)
            first_moment -= 2 / 3 * np.cos(half_angle) * (transition_radius**3 - inner_radius**3)
            integration_radius = half_spider_width
        elif half_angle <= np.pi / 2:
            integration_radius = max(inner_radius, half_spider_width / np.sin(half_angle))
        else:
            integration_radius = inner_radius

        if integration_radius < outer_radius:
            area += half_angle * (outer_radius**2 - integration_radius**2)
            area -= area_term(outer_radius) - area_term(integration_radius)
            first_moment += first_moment_term(outer_radius) - first_moment_term(integration_radius)

        if area <= 0:
            raise ValueError('spider_width is too large for the angle width.')

    centroid_radius = first_moment / area
    inner_extent = np.sqrt(inner_radius**2 + centroid_radius**2 - 2 * inner_radius * centroid_radius * np.cos(half_angle))
    outer_extent = np.sqrt(outer_radius**2 + centroid_radius**2 - 2 * outer_radius * centroid_radius * np.cos(half_angle))
    zernike_diameter = 2 * max(inner_extent, outer_extent)

    return area, centroid_radius, zernike_diameter

def make_keystone_basis(grid, num_modes, inner_diameter, outer_diameter, angle_width, spider_width, keystone_angle=0, cache=None):
    '''Make a keystone mode basis.

    This function orthogonalizes Noll-ordered Zernike polynomials over HCIPy's
    spider-trimmed annular-sector aperture using a weighted QR decomposition.
    The construction follows the general-aperture method of [Swantner1994]_
    and the annular-sector treatment of [Diaz2013]_.

    .. [Swantner1994] W. Swantner and Weng W. Chow,
        "Gram-Schmidt orthonormalization of Zernike polynomials for general
        aperture shapes," Appl. Opt. 33, 1832-1837 (1994).
    .. [Diaz2013] Jose Antonio Diaz and Virendra N. Mahajan,
        "Orthonormal aberration polynomials for optical systems with circular
        and annular sector pupils," Appl. Opt. 52, 1136-1147 (2013).

    Parameters
    ----------
    grid : hcipy.Grid
        The grid on which to compute the mode basis.
    num_modes : int
        The number of keystone modes to compute.
    inner_diameter : float
        The inner diameter of the keystone aperture.
    outer_diameter : float
        The outer diameter of the keystone aperture.
    angle_width : float
        The angular width of the keystone aperture in degrees.
    spider_width : float
        The full width of the spiders bordering the keystone aperture.
    keystone_angle : float
        The angle of the keystone aperture in radians. At an angle of 0, the center
        of the keystone points to the right.
    cache : dictionary or None
        A dictionary of intermediate calculations for this exact grid, aperture
        geometry, and orientation. The dictionary is updated in place.
    '''
    if cache is None:
        cache = {}

    area, centroid_radius, zernike_diameter = _keystone_geometry(inner_diameter, outer_diameter, angle_width, spider_width)

    keystone_grid = grid.rotated(-keystone_angle)
    zernike_grid = keystone_grid.shifted([-centroid_radius, 0])
    zernike_basis = make_zernike_basis(num_modes, zernike_diameter, zernike_grid, radial_cutoff=False, cache=cache)
    keystone_aperture = make_wedge_aperture(inner_diameter, outer_diameter, angle_width, spider_width)(keystone_grid)

    sqrt_weights = np.sqrt(grid.weights)

    if np.isscalar(sqrt_weights):
        sqrt_weights = np.array([sqrt_weights])

    # Apply the aperture mask and quadrature weights before orthogonalization.
    zernike_basis.transformation_matrix *= np.array(keystone_aperture * sqrt_weights)[:, np.newaxis]

    # Reuse the cached QR decomposition when available.
    if 'R' in cache and cache['R'].shape[0] >= num_modes:
        R = cache['R'][:num_modes, :num_modes]

        Q = scipy.linalg.solve_triangular(R, zernike_basis.transformation_matrix.T, trans=1, lower=False, overwrite_b=True).T
    else:
        # Orthogonalize the weighted modes using a QR decomposition.
        Q, R = np.linalg.qr(zernike_basis.transformation_matrix)

        if np.linalg.matrix_rank(R) < num_modes:
            raise ValueError('The requested number of keystone modes is not supported by this grid.')

        cache['R'] = R.copy()

    # Correct for numerical leakage outside the keystone aperture.
    Q *= np.array(keystone_aperture)[:, np.newaxis]

    # Fix the arbitrary signs of the QR modes.
    keystone_basis = ModeBasis(Q / np.sign(np.diag(R)), grid)

    # Normalize the modes to unit RMS over the keystone aperture.
    keystone_basis.transformation_matrix *= np.array(np.sqrt(area) / sqrt_weights)[:, np.newaxis]

    return keystone_basis
