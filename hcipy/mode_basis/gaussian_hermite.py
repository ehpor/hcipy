import numpy as np
from math import sqrt, floor
from scipy.special import eval_hermite

def index_to_hermite(i):
	'''Converts a one-dimensional mode index to the two-dimensional mode index.

	The one-dimensional mode index is converted to a two-dimensional index.
	The two-dimensional index determines the order in the x and y directions.

	Parameters
	----------
	int : i
		The one-dimensional index.
	
	Returns
	-------
	tuple : n, m
		The two-dimensional indeces.
	'''

	# Get the max order
	o = int((np.sqrt(8*i+1)-1)/2)
	# Get the y order
	m = int( i - o*(o+1)/2 )
	# Get the x order
	n = o-m
	return n,m
	
def gaussian_hermite(n, m, mode_field_diameter=1, grid=None):
	'''Creates a Gaussian-Hermite mode.

	This function evaluates a (n,m) order Gaussian-Hermite mode on a grid.
	The definition of the modes are the following,
	.. math:: 
		exp^{-\frac{r^2}{w_0^2}} H_n\left(\sqrt{2}\frac{x}{w_0}\right) H_m\left(\sqrt{2}\frac{y}{w_0}\right).
	Here :math: `w_0` is the mode_field_radius, which is :math: `\mathrm{MFD}/2`.
	This defintion follows the Physicists definition of the Hermite polynomials.
	The modes are numerical normalized to have a total power of 1.

	More details on the Hermite Polynomials can be found on:
	http://mathworld.wolfram.com/HermitePolynomial.html
	

	Parameters
	----------
	int : i
		The one-dimensional index.
	
	Returns
	-------
	tuple : n, m
		The two-dimensional indeces.
	'''
	from ..field import Field

	if grid.is_separated and grid.is_('cartesian'):
		x = grid.x
		y = grid.y
		r2 = (x**2 + y**2)/(mode_field_diameter/2)**2
	else:
		x, y = grid.as_('cartesian').coords
		r2 = (2*grid.r/mode_field_diameter)**2

	# The mode
	hg = np.exp(-r2) * eval_hermite(n, 2*sqrt(2)*x/mode_field_diameter) * eval_hermite(m, 2*sqrt(2)*y/mode_field_diameter)
	# Numerically norm the modes
	hg /= np.sum(np.abs(hg)**2 * grid.weights)
	
	return Field(hg, grid)

def gaussian_hermite_index(i, mode_field_diameter=1, grid=None):
	'''Creates a Gaussian-Hermite mode.

	This function evaluates the ith order Gaussian-Hermite mode on a grid.	

	Parameters
	----------
	int : i
		The one-dimensional mode index.
	
	float : mode_field_diameter
		The mode field diameter of the Gaussian-Hermite mode.
	
	grid : grid
		The grid on which to evaluate the Gaussian-Hermite mode.

	Returns
	-------
	Field : field
		The Gaussian-Hermite mode.
	'''

	n,m = index_to_hermite(i)
	return gaussian_hermite(n, m, mode_field_diameter, grid)

def make_gaussian_hermite_basis( grid, num_modes, mode_field_diameter, starting_mode=0):
	'''Creates a Gaussian-Hermite mode basis.

	This function evaluates the starting_mode to num_modes + starting_modes Gaussian-Hermite modes.
	And returns a ModeBasis made out of these Gaussian-Hermite modes.

	Parameters
	----------
	grid : grid
		The grid on which to evaluate the Gaussian-Hermite mode.
	
	int : num_modes
		The number of modes to create

	float : mode_field_diameter
		The mode field diameter of the Gaussian-Hermite mode.
	
	int : starting_mode
		The starting point of the mode indices.

	Returns
	-------
	ModeBasis : modes
		The Gaussian-Hermite modes
	'''
	from .mode_basis import ModeBasis

	modes = [gaussian_hermite_index(i, mode_field_diameter, grid) for i in range(starting_mode, starting_mode+num_modes)]
	return ModeBasis(modes)