from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import numpy as np
from ..field import Field

def make_linear_interpolator_separated(field, grid=None, fill_value=np.nan):
    '''Make a linear interpolator for a Field on a separated grid.

    Parameters
    ----------
    field : Field or ndarray
        The field to interpolate.
    grid : Grid or None
        The grid of the field. If it is given, the grid of `field` is replaced by this grid.
    fill_value : scalar
        The value to use for points outside of the domain of the input field. If this is None, the values
        outside the domain are extrapolated.

    Returns
    -------
    Field generator
        The interpolator, as a Field generator. The grid on which this field generator will evaluated, does
        not have to be separated.
    '''
    if grid is None:
        grid = field.grid
    else:
        field = Field(field, grid)

    # RegularGridInterpolator expects data to be in ij indexing rather than xy. We
    # need to reverse the axes of the data and the coordinates.
    data = np.moveaxis(field.shaped, range(-1, -grid.ndim - 1, -1), range(-grid.ndim, 0, 1))

    interp = RegularGridInterpolator(grid.separated_coords, data, 'linear', False, fill_value)

    def interpolator(evaluated_grid):
        res = interp(evaluated_grid.points)

        return Field(res.ravel(), evaluated_grid)

    return interpolator

def make_linear_interpolator_unstructured(field, grid=None, fill_value=np.nan):
    '''Make a linear interpolator for a Field on an unstructured grid.

    Parameters
    ----------
    field : Field or array_like
        The field to interpolate.
    grid : Grid or None
        The grid of the field. If it is given, the grid of `field` is replaced by this grid.
    fill_value : scalar
        The value to use for points outside of the domain of the input field. Linear extrapolation
        is not supported.

    Returns
    -------
    Field generator
        The interpolator as a Field generator. The grid on which this field generator will be evaluated does
        not need to have any structure.
    '''
    if grid is None:
        grid = field.grid
    else:
        field = Field(field, grid)

    interp = LinearNDInterpolator(grid.points, np.asarray(field), fill_value)

    def interpolator(evaluated_grid):
        res = interp(evaluated_grid.points)

        if fill_value is not np.nan:
            res[~np.isfinite(res)] = fill_value

        return Field(res, evaluated_grid)

    return interpolator

def make_linear_interpolator(field, grid=None, fill_value=np.nan):
    '''Make a linear interpolator for a Field on any type of grid.

    Parameters
    ----------
    field : Field or array_like
        The field to interpolate.
    grid : Grid or None
        The grid of the field. If it is given, the grid of `field` is replaced by this grid.
    fill_value : scalar or None
        The value to use for points outside of the domain of the input field. Extrapolation is not supported.
        If it is None, a numpy.nan value will be used for points outside of the domain.

    Returns
    -------
    Field generator
        The interpolator as a Field generator. The grid on which this field generator will be evaluated does
        not need to have any structure.
    '''
    if grid is None:
        grid = field.grid

    if grid.is_unstructured:
        return make_linear_interpolator_unstructured(field, grid, fill_value)
    else:
        return make_linear_interpolator_separated(field, grid, fill_value)
