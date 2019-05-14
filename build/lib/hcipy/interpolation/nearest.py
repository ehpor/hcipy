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
	
	axes_reversed = np.array(grid.separated_coords)
	interp = RegularGridInterpolator(axes_reversed, field.shaped, 'nearest', False)

	def interpolator(evaluated_grid):
		evaluated_coords = np.flip(np.array(evaluated_grid.coords), 0)
		res = interp(evaluated_coords.T)
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
	
	interp = NearestNDInterpolator(grid.points, field, fill_value)

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