import numpy as np
from ..field import Field
import warnings

def _fill_nans(arr, fill_value):
    '''Fill in NaNs with a fill value.

    Parameters
    ----------
    arr : array_like
        The array in which to replace NaNs.
    fill_value : scalar or {'min', 'max'} or None
        The value with which to replace NaNs. If this is None, no
        NaNs will be replaced and the original array is returned. If
        this is either 'min' or 'max', the NaNs will be replaced by
        the minimum or maximum of the array respectively.

    Returns
    -------
    array_like
        The input array with its NaNs replaced.
    '''
    if fill_value is None:
        return arr

    if fill_value == 'max':
        fill_value = np.nanmax(arr)
    elif fill_value == 'min':
        fill_value = np.nanmin(arr)

    arr[~np.isfinite(arr)] = fill_value

    return arr

def _compute_r2(grid):
    if grid.is_separated and grid.is_('cartesian'):
        x, y = grid.separated_coords

        return (x[np.newaxis, :]**2 + y[:, np.newaxis]**2).ravel()
    else:
        return grid.as_('polar').r**2

def spherical_surface_sag(radius_of_curvature, fill_value=None):
    '''Makes a Field generator for the surface sag of an even aspherical surface.

    Parameters
    ----------
    radius_of_curvature : scalar
        The radius of curvature of the surface.
    fill_value : scalar or {'min', 'max'} or None
        The value with which to replace NaNs. If this is None, no
        NaNs will be replaced and the original array is returned. If
        this is either 'min' or 'max', the NaNs will be replaced by
        the minimum or maximum of the array respectively.

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get the sag profile.
    '''
    return conical_surface_sag(radius_of_curvature, conic_constant=0, fill_value=fill_value)

def parabolic_surface_sag(radius_of_curvature, fill_value=None):
    '''Makes a Field generator for the surface sag of an even aspherical surface.

    Parameters
    ----------
    radius_of_curvature : scalar
        The radius of curvature of the surface.
    fill_value : scalar or {'min', 'max'} or None
        The value with which to replace NaNs. If this is None, no
        NaNs will be replaced and the original array is returned. If
        this is either 'min' or 'max', the NaNs will be replaced by
        the minimum or maximum of the array respectively.

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get the sag profile.
    '''
    return conical_surface_sag(radius_of_curvature, conic_constant=-1, fill_value=fill_value)

def conical_surface_sag(radius_of_curvature, conic_constant=0, fill_value=None):
    r'''Makes a Field generator for the surface sag of a conical surface.

    The surface profile is defined as:

    .. math:: z = \frac{cr^2}{1 + \sqrt{1-\left(1+k\right)c^2r^2}}

    with `z` the surface sag, `c` the curvature and `k` the conic constant.

    Parameters
    ----------
    radius_of_curvature : scalar
        The radius of curvature of the surface.
    conic_constant : scalar
        The conic constant of the surface.
    fill_value : scalar or {'min', 'max'} or None
        The value with which to replace NaNs. If this is None, no
        NaNs will be replaced and the original array is returned. If
        this is either 'min' or 'max', the NaNs will be replaced by
        the minimum or maximum of the array respectively.

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get the sag profile.
    '''
    def func(grid):
        r2 = _compute_r2(grid)

        curvature = 1 / radius_of_curvature
        alpha = (1 + conic_constant) * curvature**2 * r2

        with warnings.catch_warnings():
            # Suppress warnings about NaNs being produced if we're filling them in later.
            if fill_value is not None:
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")

            sag = r2 / (radius_of_curvature * (1 + np.sqrt(1 - alpha)))

        sag = _fill_nans(sag, fill_value)

        return Field(sag, grid)

    return func

def even_aspheric_surface_sag(radius_of_curvature, conic_constant=0, aspheric_coefficients=None, fill_value=None):
    r'''Makes a Field generator for the surface sag of an even aspherical surface.

    The surface profile is defined as:

    .. math:: z = \frac{cr^2}{1 + \sqrt{1-\left(1+k\right)c^2r^2}} + \sum_i=0 a_i r^{2i+4}

    With `z` the surface sag, `c` the curvature, `k` the conic constant and  :math:`a_i` the even aspheric coefficients.

    It is important to note that this definition deviates from the Zemax definition of an even aspheric surface.
    In Zemax the 2nd order term is also included in the expansion,
    which is unnessary because the conic surface itself already accounts for the 2nd order term.

    Parameters
    ----------
    radius_of_curvature : scalar
        The radius of curvature of the surface.
    conic_constant : scalar
        The conic constant of the surface.
    aspheric_coefficients : array_like
        Contains the high-order even aspheric coefficients.
    fill_value : scalar or {'min', 'max'} or None
        The value with which to replace NaNs. If this is None, no
        NaNs will be replaced and the original array is returned. If
        this is either 'min' or 'max', the NaNs will be replaced by
        the minimum or maximum of the array respectively.

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get the sag profile.
    '''
    if aspheric_coefficients is None:
        aspheric_coefficients = []

    def func(grid):
        r2 = _compute_r2(grid)

        # Start with a conic surface
        curvature = 1 / radius_of_curvature
        alpha = (1 + conic_constant) * curvature**2 * r2

        with warnings.catch_warnings():
            if fill_value is not None:
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")

            sag = r2 / (radius_of_curvature * (1 + np.sqrt(1 - alpha)))

        # Add aspheric coefficients
        # Only use the even modes and start at 4, because 0 is piston and 2 is the conic surface
        for ai, coef in enumerate(aspheric_coefficients):
            power_index = 4 + ai * 2
            sag += coef * r2**(power_index // 2)

        sag = _fill_nans(sag, fill_value)

        return Field(sag, grid)

    return func
