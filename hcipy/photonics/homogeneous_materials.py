from __future__ import division
import numpy as np
from ..field import Field

def free_space(n):
	def func(grid, z):
		return Field(n * np.ones(grid.size), grid)
	return func