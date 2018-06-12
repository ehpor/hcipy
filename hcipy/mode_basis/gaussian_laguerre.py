import numpy as np
from math import sqrt, floor
from scipy.special import eval_genlaguerre

def gaussian_laguerre(p, l, mode_field_diameter=1, grid=None):
	'''Creates a Gaussian-Hermite mode.

	This function evaluates a (p,l) order Gaussian-Laguerre mode on a grid.
	The definition of the modes are the following,
	.. math:: 
		exp^{-\frac{r^2}{w_0^2}} L_p^{|l|}\left(\frac{2r^2}{w_0^2} \right) \left(\sqrt{2}\frac{r}{w_0}\right)
	Here :math: `w_0` is the mode_field_radius, which is :math: `\mathrm{MFD}/2`.
	And :mat: `L_p^{|l|}` are the generalized Laguerre Polynomials.
	The modes are numerical normalized to have a total power of 1.

	More details on generalized Laguerre Polynomials can be found on:
	http://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
	

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

	if grid.is_separated and grid.is_('polar'):
		R, Theta = grid.separated_coords
	else:
		R, Theta = grid.as_('polar').coords

	# Easy acces
	r = 2*R/mode_field_diameter
	r2 = r**2
	# The mode
	lg = (r*sqrt(2))**(abs(l)) * np.exp(-r2) * np.exp(-1j*l*Theta) * eval_genlaguerre(p, abs(l), 2*r2)
	# Numerically norm the modes
	lg /= np.sum(np.abs(lg)**2 * grid.weights)

	return Field(lg, grid)

# High level functions
def make_gaussian_laguerre_basis( grid, pmax, lmax, mode_field_diameter, pmin=0):
	'''Creates a Gaussian-Laguerre mode basis.

	This function evaluates Gaussian-Laguerre modes.
	For each radial order within [pmin, pmax] it will calculate the 
	aziumthal order [-lmax, lmax].
	This function returns a ModeBasis made out of these Gaussian-Laguerre modes.

	Parameters
	----------
	grid : grid
		The grid on which to evaluate the Gaussian-Laguerre mode.
	
	int : pmax
		The maximum radial order of the modes
	
	int : lmax
		The maximum azimuthal order

	float : mode_field_diameter
		The mode field diameter of the Gaussian-Laguerre mode.

	int : pmin
		The minimal radial order

	Returns
	-------
	ModeBasis : modes
		The Gaussian-Laguerre modes
	'''
	from .mode_basis import ModeBasis

	modes = [gaussian_laguerre(pi, li, mode_field_diameter, grid) for li in range(-lmax, lmax) for pi in range(pmin, pmax)]
	return ModeBasis(modes)