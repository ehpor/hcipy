import numpy as np
from math import sqrt, floor
from scipy.special import eval_hermite
from .zernike import ansi_to_zernike

def ansi_to_hermite(i):
	# Get the max order
	o = int((np.sqrt(8*i+1)-1)/2)
	# Get the y order
	m = int( i - o*(o+1)/2 )
	# Get the x order
	n = o-m
	return n,m
	
def gaussian_hermite(n, m, mode_field_diameter=1, grid=None):
	from ..field import Field

	#if grid is None:
	#	return lambda grid: zernike(n, m, grid)
	
	if grid.is_separated and grid.is_('cartesian'):
		#x, y = grid.separated_coords
		x = grid.x
		y = grid.y
		r2 = (x**2 + y**2)/(mode_field_diameter/2)**2
	else:
		x, y = grid.as_('cartesian').coords
		r2 = (2*grid.r/mode_field_diameter)**2

	# The mode
	hg = np.exp(-r2) * eval_hermite(n, 2*sqrt(2)*x/mode_field_diameter) * eval_hermite(m, 2*sqrt(2)*y/mode_field_diameter)
	# Numerically norm the modes
	# TODO : change to analytical normalization
	hg /= np.sum(np.abs(hg)**2 * grid.weights)
	
	return Field(hg, grid)

def gaussian_hermite_ansi(i, mode_field_diameter=1, grid=None):
	# Map mode index to p, l coordinates
	n,m = ansi_to_hermite(i)
	return gaussian_hermite(n, m, mode_field_diameter, grid)

# High level functions
def make_gaussian_hermite_basis( grid, num_modes, mode_field_diameter, starting_mode=0):
	from .mode_basis import ModeBasis

	modes = [gaussian_hermite_ansi(i, mode_field_diameter, grid) for i in range(starting_mode, starting_mode+num_modes)]
	return ModeBasis(modes)