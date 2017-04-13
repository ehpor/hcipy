from __future__ import division
import numpy as np
from ..field import Field

def linear_gradient(n0, dn):
	def func(grid, z):
		x, y = grid.as_('cartesian').coords
		refractive_index = n0 + dn * (x-x.min())/(x.max()-x.min())
		return Field(refractive_index.astype('float'), grid)
	return func