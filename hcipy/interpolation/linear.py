from scipy.interpolate import RegularGridInterpolator
import numpy as np
from ..field import Field

def make_linear_interpolator_separated(field, grid=None):
	if grid is None:
		grid = field.grid
	
	axes_reversed = np.array(grid.separated_coords)
	interp = RegularGridInterpolator(axes_reversed, field.shaped, 'linear', False, None)

	def interpolator(evaluated_grid):
		evaluated_coords = np.flip(np.array(evaluated_grid.coords), 0)
		res = interp(evaluated_coords.T)
		return Field(res.ravel(), evaluated_grid)

	return interpolator