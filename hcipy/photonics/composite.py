from __future__ import division
import numpy as np
from ..field import Field

def composite_material(materials,thicknesses):
	material_boundary = np.cumsum(thicknesses)
	def func(grid, z):
		material_index = np.argmax( z < material_boundary )
		return materials[material_index](grid,z)
	return func