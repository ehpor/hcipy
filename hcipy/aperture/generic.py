from __future__ import division
import numpy as np
from ..field import Field

def circular_aperture(diameter):
	def func(grid):
		if grid.is_('cartesian'):
			x, y = grid.coords
			f = (x**2 + y**2) <= (diameter / 2)**2
		else:
			f = grid.r <= (diameter / 2)
	
		return Field(f.astype('float'), grid)
	
	return func

def rectangular_aperture(size, center=None):
	dim = size * np.ones(2)

	def func(grid):
		x, y = grid.as_('cartesian').coords
		f = (np.abs(x) <= (dim[0]/2)) * (np.abs(y) <= (dim[1]/2))

		return Field(f.astype('float'), grid)
	
	return func

def regular_polygon_aperture(num_sides, circum_diameter):
	if num_sides < 3:
		raise ValueError('The number of sides for a regular polygon has to greater or equal to 3.')
	
	apothem = np.cos(np.pi / num_sides) * circum_diameter / 2

	# Make use of symmetry
	if num_sides % 2 == 0:
		thetas = np.arange(int(num_sides / 4) + 1) * 2*np.pi
	else:
		thetas = np.arange(int(num_sides / 2) + 1) * 2*np.pi

	mask = rectangular_aperture(circum_diameter)

	def func(grid):
		f = np.zeros(grid.size, dtype='float')
		g = grid.as_('cartesian')
		m = mask(g) != 0

		x, y = g.coords
		x = x[m]
		y = y[m]

		# Make use of symmetry
		if num_sides % 2 == 0:
			for theta in thetas:
				f[m] = np.abs(np.cos(theta) * x + np.sin(theta) * y) < (circum_diameter / 2)
		else:
			for theta in thetas:
				f[m] *= (np.cos(theta) * x + np.abs(np.sin(theta) * y)) < (circum_diameter / 2)
		
		return Field(f, grid)
	
	return func

# Convenience function
def hexagonal_aperture(circum_diameter):
	return regular_polygon_aperture(6, circum_diameter)