import numpy as np
from math import sqrt, floor
from scipy.special import eval_genlaguerre
from .zernike import ansi_to_zernike

def gaussian_laguerre(p, l, mode_field_diameter=1, grid=None):
	from ..field import Field

	#if grid is None:
	#	return lambda grid: zernike(n, m, grid)
	
	if grid.is_separated and grid.is_('polar'):
		R, Theta = grid.separated_coords
		# Easy acces
		r = 2*R/mode_field_diameter
		r2 = r**2
		# The mode
		lg = (r*sqrt(2))**(abs(l)) * np.exp(-r2) * np.exp(-1j*l*phi) * eval_genlaguerre(p, abs(l), 2*r2)
	else:
		r, theta = grid.as_('polar').coords
		# Easy acces
		r = 2*R/mode_field_diameter
		r2 = r**2
		# The mode
		lg = (r*sqrt(2))**(abs(l)) * np.exp(-r2) * np.exp(-1j*l*phi) * eval_genlaguerre(p, abs(l), 2*r2)
	
	# Numerically norm the modes
	lg /= np.sum(np.abs(lg)**2 * grid.weights)

	return Field(lg, grid)

def gaussian_laguerre_ansi(i, mode_field_diameter=1, grid=None):
	# Map mode index to p, l coordinates
	p,l = ansi_to_zernike(i)
	return gaussian_laguerre(n, m, mode_field_diameter, grid)

# High level functions
def make_gaussian_laguerre_basis(num_modes, mode_field_diameter, grid, starting_mode=1):
	from .mode_basis import ModeBasis

	modes = [gaussian_laguerre_ansi(i, mode_field_diameter, grid) for i in range(starting_mode, starting_mode+num_modes)]
	return ModeBasis(modes)