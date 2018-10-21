from scipy.interpolate import RegularGridInterpolator
import numpy as np
from ..field import Field

def make_linear_interpolator_separated(field, grid=None, fill_value=None):
	'''Make a linear interpolators for a separated grid.

	Eventually these functions will be replaced by a single `make_linear_interpolator` function, that
	can operate on all types of coordinates.

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
	
	axes_reversed = np.array(grid.separated_coords)
	interp = RegularGridInterpolator(axes_reversed, field.shaped, 'linear', False, fill_value)

	def interpolator(evaluated_grid):
		evaluated_coords = np.flip(np.array(evaluated_grid.coords), 0)
		res = interp(evaluated_coords.T)
		return Field(res.ravel(), evaluated_grid)

	return interpolator