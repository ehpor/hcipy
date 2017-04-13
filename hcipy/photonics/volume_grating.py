from __future__ import division
import numpy as np
from ..field import Field

def volume_grating(n0, dn, pitch, kg):
	def func(grid, z):
		x, y = grid.as_('cartesian').coords
		refractive_index = n0 + dn * np.cos(2.0*np.pi / pitch  * (kg[0] * x + kg[1] * y + kg[2] * z)) 
		return Field(refractive_index.astype('float'), grid)
	return func