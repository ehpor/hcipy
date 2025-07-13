from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
import numpy as np
from ..field import Field

def make_nearest_interpolator_separated(field, grid=None):
    '''Make a nearest interpolator for a field on a separated grid.

    Parameters
    ----------
    field : Field or ndarray
        The field to interpolate.
    grid : Grid or None
        The grid of the field. If it is given, the grid of `field` is replaced by this grid.

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

    interp = RegularGridInterpolator(grid.separated_coords, data, 'nearest', False)

    def interpolator(evaluated_grid):
        res = interp(evaluated_grid.points)

        return Field(res.ravel(), evaluated_grid)

    return interpolator

def make_nearest_interpolator_unstructured(field, grid=None):
    '''Make a nearest interpolator for an unstructured grid.

    Parameters
    ----------
    field : Field or array_like
        The field to interpolate.
    grid : Grid or None
        The grid of the field. If it is given, the grid of `field` is replaced by this grid.

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

    interp = NearestNDInterpolator(grid.points, field)

    def interpolator(evaluated_grid):
        res = interp(grid.points)
        return Field(res, evaluated_grid)

    return interpolator

def make_nearest_interpolator(field, grid=None):
    '''Make a nearest interpolator for any type of grid.

    Parameters
    ----------
    field : Field or array_like
        The field to interpolate.
    grid : Grid or None
        The grid of the field. If it is given, the grid of `field` is replaced by this grid.

    Returns
    -------
    Field generator
        The interpolator as a Field generator. The grid on which this field generator will be evaluated does
        not need to have any structure.
    '''
    if grid is None:
        grid = field.grid

    if grid.is_unstructured:
        return make_nearest_interpolator_unstructured(field, grid)
    else:
        return make_nearest_interpolator_separated(field, grid)
