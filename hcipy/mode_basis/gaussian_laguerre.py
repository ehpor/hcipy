import numpy as np
from math import sqrt, floor
from scipy.special import eval_genlaguerre

def gaussian_laguerre(p, l, mode_field_diameter=1, grid=None):
	r'''Creates a Gaussian-Hermite mode.

	This function evaluates a (p,l) order Gaussian-Laguerre mode on a grid.
	The definition of the modes are the following,

	.. math::
		\exp{\left(-\frac{r^2}{w_0^2}\right)} L_p^{|l|}\left(\frac{2r^2}{w_0^2} \right) \left(\sqrt{2}\frac{r}{w_0}\right)

	Here :math:`w_0` is the mode_field_radius, which is :math:`\mathrm{MFD}/2`.
	And :math:`L_p^{|l|}` are the generalized Laguerre Polynomials.
	All modes are numerical normalized to have a total power of 1.

	More details on generalized Laguerre Polynomials can be found on:
	http://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html

	Parameters
	----------
	p : int
		The radial order.
	l : int
		The azimuthal order.
	mode_field_diameter : scalar
		The mode field diameter of the mode.
	grid : Grid
		The grid on which to evaluate the mode.
	
	Returns
	-------
	Field
		The evaluated mode.
	'''
	from ..field import Field

	if grid.is_separated and grid.is_('polar'):
		R, Theta = grid.separated_coords
	else:
		R, Theta = grid.as_('polar').coords

	# Easy access
	r = 2*R/mode_field_diameter
	r2 = r**2
	# The mode
	lg = (r*sqrt(2))**(abs(l)) * np.exp(-r2) * np.exp(-1j*l*Theta) * eval_genlaguerre(p, abs(l), 2*r2)
	# Numerically normalize the modes
	lg /= np.sum(np.abs(lg)**2 * grid.weights)

	return Field(lg, grid)

# High level functions
def make_gaussian_laguerre_basis(grid, pmax, lmax, mode_field_diameter, pmin=0):
	'''Creates a Gaussian-Laguerre mode basis.

	This function evaluates Gaussian-Laguerre modes. For each radial order 
	within [pmin, pmax] it will calculate the azimuthal order [-lmax, lmax] inclusive.
	This function returns a ModeBasis made out of these Gaussian-Laguerre modes.

	Parameters
	----------
	grid : Grid
		The grid on which to evaluate the Gaussian-Laguerre mode.
	pmax : int
		The maximum radial order of the modes.
	lmax : int
		The maximum azimuthal order.
	mode_field_diameter : scalar
		The mode field diameter of the Gaussian-Laguerre mode.
	pmin : int
		The minimal radial order.

	Returns
	-------
	ModeBasis
		The Gaussian-Laguerre modes.
	'''
	from .mode_basis import ModeBasis

	modes = [gaussian_laguerre(pi, li, mode_field_diameter, grid) for li in range(-lmax, lmax + 1) for pi in range(pmin, pmax)]
	return ModeBasis(modes)