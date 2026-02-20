from __future__ import division
from collections.abc import Iterable

import numpy as np
from .coordinates import RegularCoords, SeparatedCoords, UnstructuredCoords
from .backends import _infer_xp
from .field import Field
from .cartesian_grid import CartesianGrid
from .grid import Grid

def _normalize_to_tuple(src, size, dtype=None):
    '''Normalize an input to a tuple.

    This function takes either a scalar or iterable as `src` and outputs a
    tuple with `size` elements. If the input is an iterable it is converted to
    a tuple. If the input is a scalar it is repeated `size` times.

    Parameters
    ----------
    src : scalar or iterable
        The input to normalize.
    size : int
        The number of elements the output tuple should have.
    dtype : type or None
        The type to cast the elements to. If None the type is not cast.

    Returns
    -------
    tuple
       The normalized input.
    '''
    if dtype is None:
        if isinstance(src, Iterable):
            return tuple(x for x in src)
        else:
            return (src,) * size
    else:
        if isinstance(src, Iterable):
            return tuple(dtype(x) for x in src)
        else:
            return (dtype(src),) * size

def _find_common_size(*args):
    '''Find the common size between a number of arguments.

    Parameters
    ----------
    args : scalars or iterables
        The arguments to find a common size for.

    Returns
    -------
    int
        The common size.
    '''
    return max(len(arg) if isinstance(arg, Iterable) else 1 for arg in args)

def make_uniform_grid(dims, extent, center=0, has_center=False, xp=None):
    '''Create a uniformly-spaced :class:`Grid` of a certain shape and size.

    Parameters
    ----------
    dims : int or tuple of ints
        The number of points in each dimension. If this is a scalar, it will
        be multiplexed over all dimensions.
    extent : scalar or array_like
        The total extent of the grid in each dimension. Can be an xp array for differentiability.
    center : scalar or array_like
        The center point. The grid will by symmetric around this point. Can be an xp array for differentiability.
    has_center : boolean
        Does the grid has to have the center as one of its points. If this is
        False, this does not mean that the grid will not have the center.
    xp : module, optional
        The array namespace to use. If not provided, will be inferred from extent/center if they are arrays,
        otherwise defaults to numpy.

    Returns
    -------
    Grid
        A :class:`Grid` with :class:`RegularCoords`.
    '''
    num_dims = _find_common_size(dims, extent, center)
    dims = _normalize_to_tuple(dims, num_dims, int)

    # Determine xp from inputs if not provided
    if xp is None:
        xp = _infer_xp(extent, center)

    # Convert to xp arrays (preserves differentiability if inputs are xp arrays)
    # Ensure at least 1D arrays for proper broadcasting
    extent = xp.asarray(extent, dtype=float)
    center = xp.asarray(center, dtype=float)

    # If scalars provided, broadcast to match num_dims
    if extent.ndim == 0:
        extent = xp.broadcast_to(extent, (num_dims,))
    if center.ndim == 0:
        center = xp.broadcast_to(center, (num_dims,))

    dims_arr = xp.asarray(dims)

    # Array math for differentiability
    delta = extent / dims_arr
    zero = -extent / 2 + center + delta / 2

    if has_center:
        zero = zero - delta / 2 * (1 - dims_arr % 2)

    return CartesianGrid(RegularCoords(delta, dims, zero))

def make_pupil_grid(dims, diameter=1, xp=None):
    '''Makes a new :class:`Grid`, meant for descretisation of a pupil-plane wavefront.

    This grid is symmetric around the origin, and therefore has no point exactly on
    the origin for an even number of pixels.

    Parameters
    ----------
    dims : tuple of ints or integer
        The number of pixels per dimension. If this is an integer, this number
        of pixels is used for all dimensions.
    diameter : scalar or array_like
        The diameter of the grid in each dimension. Can be an xp array for differentiability.
    xp : module, optional
        The array namespace to use. If not provided, will be inferred from diameter if it's an array,
        otherwise defaults to numpy.

    Returns
    -------
    Grid
        A :class:`CartesianGrid` with :class:`RegularCoords`.
    '''
    # Ensure 2D grid for pupil grids
    if xp is None:
        xp = _infer_xp(diameter)

    diameter = xp.asarray(diameter, dtype=float)
    if diameter.ndim == 0:
        diameter = xp.broadcast_to(diameter, (2,))

    return make_uniform_grid(dims, diameter, xp=xp)

def make_focal_grid_from_pupil_grid(pupil_grid, q=1, num_airy=None, focal_length=1, wavelength=1):
    '''Make a grid for a focal plane from a pupil grid.

    Calculate the focal grid corresponding to the pupil grid, using an FFT grid as a guide. The
    resulting grid focal will always contain the origin (0, 0) point. The extent of the pupil
    grid will be used as the diameter of the pupil. If the pupil is undersized on the pupil
    grid, the resulting focal grid needs to be rescaled manually.

    .. note::
        In almost all cases, it is preferable to use :func:`make_focal_grid`. This
        function allows you to directly set the diameter, and doesn't require the user to pass
        the pupil grid as an argument. `make_focal_grid_from_pupil_grid()` retains old functionality
        and serves as a backwards compatibility function, due to its ubiquitous usage in HCIPy code.

    Parameters
    ----------
    pupil_grid : Grid
        The pupil grid for which the focal grid needs to be calculated. The extent of this Grid
        is used as the diameter of the pupil.
    q : float or tuple of floats
        The number of pixels per resolution element (= lambda f / D).
    num_airy : float or tuple of floats
        The spatial extent of the grid in radius in resolution elements (= lambda f / D).
    focal_length : scalar
        The focal length used for calculating the spatial resolution at the focal plane.
    wavelength : scalar
        The reference wavelength used for calculating the spatial resolution at the focal plane.

    Returns
    -------
    Grid
        A Grid describing the sampling for a focal plane.
    '''
    from ..fourier import make_fft_grid

    if not pupil_grid.ndim == 2:
        raise ValueError('The given pupil grid has to be two-dimensional.')

    f_lambda = focal_length * wavelength

    q = _normalize_to_tuple(q, 2, float)

    if num_airy is None:
        fov = (1,) * pupil_grid.ndim
    else:
        num_airy = _normalize_to_tuple(num_airy, 2)
        fov = tuple(num_airy_i / (n / 2) for num_airy_i, n in zip(num_airy, pupil_grid.dims))

    if max(fov) > 1:
        import warnings
        warnings.warn('Focal grid is larger than the maximum allowed angle (fov=%.03f). You may see wrapping when doing propagations.' % max(fov), stacklevel=2)

    uv = make_fft_grid(pupil_grid, q, fov)
    focal_grid = uv.scaled(f_lambda / (2 * np.pi))

    return focal_grid

def make_focal_grid(q, num_airy, spatial_resolution=None, pupil_diameter=None, focal_length=None, f_number=None, reference_wavelength=None, xp=None):
    r'''Make a grid for a focal plane.

    This grid will be a CartesianGrid with RegularCoords, and supports different resolutions, samplings,
    or extent in x and y. If `spatial_resolution` is 1, then the grid will be returned in normalized
    (ie. homogenized) coordinates. Otherwise, it will be in physical units. The spatial resolution is
    defined by:

    .. math:: \Delta x = \lambda f / D = \lambda F

    where :math:`\lambda` is the wavelength, :math:`f` is the effective focal length before the focal plane,
    :math:`D` is the diameter of the pupil, and :math:`F` is the F-number of the incoming light beam.
    You can supply either the spatial resolution or a set of focal length, reference wavelength and pupil diameter,
    or a set of F-number and reference wavelength. If none are supplied, a spatial resolution of 1 will be
    assumed, meaning normalized units.

    The grid will always contain the origin (0, 0) point.

    Parameters
    ----------
    q : scalar or array_like
        The number of pixels per resolution element (= lambda f / D). Can be an xp array for differentiability.
    num_airy : scalar or array_like
        The spatial extent of the grid in radius in resolution elements (= lambda f / D). Can be an xp array for differentiability.
    spatial_resolution : scalar or array_like
        The physical size of a resolution element (= lambda f / D). Can be an xp array for differentiability.
    pupil_diameter : scalar or array_like
        The diameter of the pupil. Can be an xp array for differentiability.
    focal_length : scalar or array_like
        The focal length used for calculating the spatial resolution at the focal plane. Can be an xp array for differentiability.
    f_number : scalar or array_like
        The F number, also known as focal ratio, at the focal plane. Can be an xp array for differentiability.
    reference_wavelength : scalar
        The reference wavelength used for calculating the spatial resolution at the focal plane.
    xp : module, optional
        The array namespace to use. If not provided, will be inferred from input arrays if they are xp arrays,
        otherwise defaults to numpy.

    Returns
    -------
    Grid
        A Grid describing the sampling for a focal plane.

    Raises
    ------
    ValueError
        If both no spatial resolution and no complete set of (focal length, reference wavelength and pupil diameter) was supplied.
    '''
    if isinstance(q, Grid):
        raise ValueError('The function signature was changed as of HCIPy 0.3.0. Please use the new signature (prefered), or use make_focal_grid_from_pupil_grid() if you want to retain old behaviour.')

    # Determine xp from inputs if not provided
    if xp is None:
        xp = _infer_xp(q, num_airy, spatial_resolution, pupil_diameter, focal_length, f_number)

    # Convert all optical parameters to xp arrays (preserves differentiability)
    if focal_length is not None:
        focal_length = xp.asarray(focal_length, dtype=float)
    if spatial_resolution is not None:
        spatial_resolution = xp.asarray(spatial_resolution, dtype=float)
    if pupil_diameter is not None:
        pupil_diameter = xp.asarray(pupil_diameter, dtype=float)
    if f_number is not None:
        f_number = xp.asarray(f_number, dtype=float)

    if spatial_resolution is None:
        if f_number is None:
            if pupil_diameter is None or focal_length is None:
                if reference_wavelength is None:
                    spatial_resolution = xp.asarray([1, 1], dtype=float)
                else:
                    raise ValueError('You only supplied a reference wavelength and forgot to supply either an f_number or a (pupil_diameter, focal_length).')
            else:
                f_number = focal_length / pupil_diameter

        if spatial_resolution is None:
            if reference_wavelength is None:
                raise ValueError('You supplied an f_number or (pupil_diameter, focal_length), and forgot to supply a reference wavelength.')
            else:
                spatial_resolution = f_number * reference_wavelength

    q = xp.asarray(q, dtype=float)
    num_airy = xp.asarray(num_airy, dtype=float)
    spatial_resolution = xp.asarray(spatial_resolution, dtype=float)

    delta = spatial_resolution / q

    # Determine the dimensionality
    num_dims = max(q.ndim, num_airy.ndim)
    if num_dims == 0:
        # Both scalars - assume 2D for focal grids
        num_dims = 2
        dims = (int(2 * float(num_airy) * float(q)),) * num_dims
        delta = xp.broadcast_to(delta, (num_dims,))
    else:
        # At least one is an array
        if q.ndim == 0:
            # q is scalar, broadcast to match num_airy
            q_vals = [float(q)] * len(num_airy)
            num_airy_vals = num_airy.tolist()
        elif num_airy.ndim == 0:
            # num_airy is scalar, broadcast to match q
            q_vals = q.tolist()
            num_airy_vals = [float(num_airy)] * len(q)
        else:
            # Both are arrays
            q_vals = q.tolist()
            num_airy_vals = num_airy.tolist()
        dims = tuple(int(2 * n * q_i) for n, q_i in zip(num_airy_vals, q_vals))
        if delta.ndim == 0:
            delta = xp.broadcast_to(delta, (len(dims),))

    dims_arr = xp.asarray(dims)
    zero = delta * (-dims_arr / 2 + (dims_arr % 2) * 0.5)

    return CartesianGrid(RegularCoords(delta, dims, zero))

def make_hexagonal_grid(circum_diameter, n_rings, pointy_top=False, center=None, xp=None):
    '''Make a regular hexagonal grid.

    Parameters
    ----------
    circum_diameter : scalar or array_like
        The circum diameter of the hexagons in the grid. Can be an xp array for differentiability.
    n_rings : integer
        The number of rings in the grid.
    pointy_top : boolean
        If the hexagons contained in the grid.
    center : array_like
        The center of the grid in cartesian coordinates. Can be an xp array for differentiability.
    xp : module, optional
        The array namespace to use. If not provided, will be inferred from inputs if they are arrays,
        otherwise defaults to numpy.

    Returns
    -------
    Grid
        A :class:`CartesianGrid` with `UnstructuredCoords`, indicating the
        center of the hexagons.
    '''
    # Determine xp from inputs if not provided
    if xp is None:
        xp = _infer_xp(circum_diameter, center)

    if center is None:
        center = xp.asarray([0, 0], dtype=float)
    else:
        center = xp.asarray(center, dtype=float)

    circum_diameter = xp.asarray(circum_diameter, dtype=float)
    apothem = circum_diameter * xp.sqrt(3) / 4

    q = [0]
    r = [0]

    for n in range(1, n_rings + 1):
        # top
        q += list(range(n, 0, -1))
        r += list(range(0, n))
        # right top
        q += list(range(0, -n, -1))
        r += [n] * n
        # right bottom
        q += [-n] * n
        r += list(range(n, 0, -1))
        # bottom
        q += list(range(-n, 0))
        r += list(range(0, -n, -1))
        # left bottom
        q += list(range(0, n))
        r += [-n] * n
        # left top
        q += [n] * n
        r += list(range(-n, 0))

    q_arr = xp.asarray(q)
    r_arr = xp.asarray(r)
    x = (-q_arr + r_arr) * circum_diameter / 2 + center[0]
    y = (q_arr + r_arr) * apothem * 2 + center[1]

    weight = 2 * apothem**2 * xp.sqrt(3)

    if pointy_top:
        return CartesianGrid(UnstructuredCoords((x, y), xp=xp), weight)
    else:
        return CartesianGrid(UnstructuredCoords((y, x), xp=xp), weight)

def make_chebyshev_grid(dims, minimum=None, maximum=None, xp=None):
    '''Make a Chebyshev grid.

    Parameters
    ----------
    dims : tuple of ints
        The number of points in each dimension.
    minimum : scalar or array_like
        The minimum value in each dimension. Can be an xp array for differentiability.
        The default is -1.
    maximum : scalar or array_like
        The maximum value in each dimension. Can be an xp array for differentiability.
        The default is 1.
    xp : module, optional
        The array namespace to use. If not provided, will be inferred from inputs if they are arrays,
        otherwise defaults to numpy.

    Returns
    -------
    Grid
        A :class:`CartesianGrid` with `SeparatedCoords`.
    '''
    if minimum is None:
        minimum = -1

    if maximum is None:
        maximum = 1

    # Determine xp from inputs if not provided
    if xp is None:
        xp = _infer_xp(minimum, maximum)

    # Convert to xp arrays (preserves differentiability)
    minimum = xp.asarray(minimum, dtype=float)
    maximum = xp.asarray(maximum, dtype=float)

    middles = (minimum + maximum) / 2
    intervals = (maximum - minimum) / 2

    sep_coords = []
    for i, dim in enumerate(dims):
        c = xp.cos(xp.pi * (2 * xp.arange(dim) + 1) / (2.0 * dim))
        # Handle both scalar and array middles/intervals
        middle = middles if middles.ndim == 0 else middles[i]
        interval = intervals if intervals.ndim == 0 else intervals[i]
        c = middle + interval * c
        sep_coords.append(c)

    return CartesianGrid(SeparatedCoords(sep_coords, xp=xp))

def make_supersampled_grid(grid, oversampling):
    '''Make a new grid that oversamples by a factor `oversampling`.

    .. note ::
        The Grid `grid` must be a grid with separable coordinates.

    Parameters
    ----------
    grid : Grid
        The grid that we want to oversample.
    oversampling : integer or scalar or tuple of ints
        The factor by which to oversample. If this is a scalar, it will be rounded to
        the nearest integer. If this is an array, a different oversampling factor will
        be used for each dimension.

    Returns
    -------
    Grid
        The oversampled grid.
    '''
    oversampling = _normalize_to_tuple(oversampling, grid.ndim, round)

    if grid.is_regular:
        xp = grid.xp
        oversampling_arr = xp.asarray(oversampling)
        delta_new = grid.delta / oversampling_arr
        zero_new = grid.zero - grid.delta / 2 + delta_new / 2
        dims_new = tuple(d * o for d, o in zip(grid.dims, oversampling))

        return grid.__class__(RegularCoords(delta_new, dims_new, zero_new))
    elif grid.is_separated:
        raise NotImplementedError()

    raise ValueError('Cannot create a supersampled grid from a non-separated grid.')

def make_subsampled_grid(grid, undersampling):
    '''Make a new grid that undersamples by a factor `undersampling`.

    .. note ::
        The dimensions of the `grid` must be divisible by `undersampling`.

    Parameters
    ----------
    grid : Grid
        The grid that we want to oversample.
    undersampling : integer or scalar or tuple of ints
        The factor by which to undersample. If this is a scalar, it will be rounded to
        the nearest integer. If this is an array, a different undersampling factor will
        be used for each dimension.

    Returns
    -------
    Grid
        The undersampled grid.
    '''
    undersampling = _normalize_to_tuple(undersampling, grid.ndim, round)

    if grid.is_regular:
        xp = grid.xp
        undersampling_arr = xp.asarray(undersampling)
        delta_new = grid.delta * undersampling_arr
        zero_new = grid.zero - grid.delta / 2 + delta_new / 2
        dims_new = tuple(d // u for d, u in zip(grid.dims, undersampling))

        return grid.__class__(RegularCoords(delta_new, dims_new, zero_new))
    elif grid.is_separated:
        raise NotImplementedError()

    raise ValueError("Cannot create a subsampled grid from a non-separated grid.")

def subsample_field(field, subsampling, new_grid=None, statistic='mean'):
    '''Average the field over subsampling pixels in each dimension.

    .. note ::
        The dimensions of the grid of `field` must be divisible by `subsampling`.

    Parameters
    ----------
    field : Field
        The field to subsample. The grid of this field must have the right
        dimensions to be able to be subsampled.
    subsampling : integer or scalar or tuple of ints
        The subsampling factor. If this is a scalar, it will be rounded to the
        nearest integer. If this is an array, the subsampling factor will be
        different for each dimension.
    new_grid : Grid
        If this grid is given, no new grid will be calculated and this grid will
        be used instead. This saves on calculation time if your new grid is already
        known beforehand.
    statistic : string
        The statistic to compute (default is 'mean').
        The following statistics are available:
        * 'mean': compute the mean of values for points within each superpixel.
        * 'sum': compute the sum of values for points within each superpixel. This is identical to a weighted histogram.
        * 'min': compute the minimum of values for points within each superpixel.
        * 'max': compute the maximum of values for points within each superpixel.
        * 'median': compute the median of values for points within each superpixel.
        * 'nanmedian': compute the median of values for points within each superpixel while ignoring NaN values.

    Returns
    -------
    Field
        The subsampled field.
    '''
    subsampling = _normalize_to_tuple(subsampling, field.grid.ndim, round)

    if new_grid is None:
        new_grid = make_subsampled_grid(field.grid, subsampling)

    reshape = tuple()
    axes = tuple()
    for i, (s, sub) in enumerate(zip(new_grid.shape, reversed(subsampling))):
        reshape = reshape + (s, sub)
        axes = axes + (2 * i + 1,)

    if field.tensor_order > 0:
        reshape = field.tensor_shape + reshape
        axes = tuple(ax + field.tensor_order for ax in axes)
        new_shape = field.tensor_shape + (-1,)
    else:
        new_shape = (-1,)

    available_statistics = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'sum': np.sum,
        'median': np.median,
        'nanmedian': np.nanmedian
    }

    if statistic not in available_statistics:
        raise ValueError('This statistic is not recognized.')

    if field.grid.is_regular:
        # All weights will be the same, so the array can be combined without taking the weights into account.
        return Field(available_statistics[statistic](field.reshape(reshape), axis=axes).reshape(new_shape), new_grid)
    else:
        # Some weights will be different so calculate weighted mean instead.
        if statistic in ['min', 'max', 'sum', 'median', 'nanmedian']:
            f = available_statistics[statistic](field.reshape(reshape), axis=axes)
            return Field(f.reshape(new_shape), new_grid)
        else:
            # Statistic is mean
            weights = field.grid.weights
            w = weights.reshape(reshape).sum(axis=axes)
            f = np.sum((field * weights).reshape(reshape), axis=axes)
            return Field((f / w).reshape(new_shape), new_grid)

def evaluate_supersampled(field_generator, grid, oversampling, statistic='mean', make_sparse=True):
    '''Evaluate a Field generator on `grid`, with an oversampling.

    Parameters
    ----------
    field_generator : Field generator or list of Field generators
        The field generator to evaluate. If this is a list of Field generators,
        each Field generator will be evaluated and stored in a ModeBasis.
    grid : Grid
        The grid on which to evaluate `field_generator`.
    oversampling : integer or scalar or tuple of ints
        The factor by which to oversample. If this is a scalar, it will be rounded to
        the nearest integer. If this is an array, a different oversampling factor will
        be used for each dimension.
    statistic : string
        The statistic to compute (default is 'mean').
        The following statistics are available:
        * 'mean': compute the mean of values for points within each superpixel.
        * 'sum': compute the sum of values for points within each superpixel. This is identical to a weighted histogram.
        * 'min': compute the minimum of values for points within each superpixel.
        * 'max': compute the maximum of values for point within each superpixel.
    make_sparse : boolean
        If the resulting ModeBasis needs to be sparsified. This is ignored if
        only a single Field generator is provided.

    Returns
    -------
    Field or ModeBasis
        The evaluated field or mode basis.
    '''
    import scipy.sparse
    from ..mode_basis import ModeBasis

    if isinstance(field_generator, (list, tuple)):
        modes = []

        for fg in field_generator:
            field = evaluate_supersampled(fg, grid, oversampling, statistic)

            if make_sparse:
                field = scipy.sparse.csr_matrix(field)
                field.eliminate_zeros()

            modes.append(field)

        return ModeBasis(modes, grid)

    oversampling = _normalize_to_tuple(oversampling, grid.ndim, round)

    if grid.is_separated:
        # Use dithered sub grids to get the final supersampled field.
        deltas = []
        for i in range(grid.ndim):
            x = grid.coords.separated_coords[i]
            d = (x[2:] - x[:-2]) / 2.
            d = np.concatenate(([x[1] - x[0]], d, [x[-1] - x[-2]]))
            deltas.append(d)

        dithers = make_uniform_grid(oversampling, 1)

        separated_coords = grid.separated_coords

        if statistic in ['mean', 'sum']:
            field = 0
        else:
            field = None

        for dither in dithers.points:
            dithered_separated_coords = [c + d * delta for c, d, delta in zip(separated_coords, dither, deltas)]
            dithered_grid = grid.__class__(SeparatedCoords(dithered_separated_coords))

            if statistic in ['mean', 'sum']:
                field += field_generator(dithered_grid)
            else:
                if field is None:
                    field = field_generator(dithered_grid)
                elif statistic == 'min':
                    field = np.minimum(field, field_generator(dithered_grid))
                else:
                    field = np.maximum(field, field_generator(dithered_grid))

        if statistic == 'mean':
            field /= len(dithers)

        field.grid = grid
        return field
    else:
        # Cannot use sub grids, so fall back to evaluation of generator on the full
        # supersampled grid.
        supersampled_grid = make_supersampled_grid(grid, oversampling)
        field = field_generator(supersampled_grid)
        return subsample_field(field, oversampling, grid, statistic)

def make_uniform_vector_field(field, jones_vector):
    '''Make an uniform vector field from a scalar field and a jones vector.

    Parameters
    ----------
    field : Field
        An input scalar field that is expanded to a vector field
    jones_vector : array_like
        The output vector at every pixel

    Returns
    -------
    Field
        The expanded vector field
    '''
    if field.is_scalar_field:
        return Field([ei * field for ei in jones_vector], field.grid)

def make_uniform_vector_field_generator(field_generator, jones_vector):
    '''Make an uniform vector field generator from a scalar field generator and a jones vector.

    Parameters
    ----------
    field_generator : Field generator
        The field generator to evaluate.
    jones_vector : array_like
        The output vector at every grid coordinate

    Returns
    -------
    Field generator
        This function can be evaluated on a grid to get a Field.
    '''
    def func(grid):
        scalar_field = field_generator(grid)
        return Field([ei * scalar_field for ei in jones_vector], grid)

    return func
