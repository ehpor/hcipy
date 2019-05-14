import numpy as np
from math import sqrt, floor
from scipy.special import binom

def noll_to_zernike(i):
	'''Get the Zernike index from a Noll index.

	Parameters
	----------
	i : int
		The Noll index.
	
	Returns
	-------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	'''
	n = int(sqrt(2 * i - 1) + 0.5) - 1
	if n % 2:
		m = 2 * int((2 * (i + 1) - n * (n + 1)) // 4) - 1
	else:
		m = 2 * int((2 * i + 1 - n * (n + 1)) // 4)
	return n, m * (-1)**(i % 2)

def ansi_to_zernike(i):
	'''Get the Zernike index from an ANSI index.

	Parameters
	----------
	i : int
		The ANSI index.
	
	Returns
	-------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	'''
	n = int((sqrt(8 * i + 1) - 1) / 2)
	m = 2 * i - n * (n + 2)
	return (n, m)

def zernike_to_ansi(n, m):
	'''Get the ANSI index for a pair of Zernike indices.

	Parameters
	----------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	
	Returns
	-------
	int
		The ANSI index.
	'''
	return (m + n * n) // 2 + n

def zernike_to_noll(n, m):
	'''Get the Noll index for a pair of Zernike indices.

	Parameters
	----------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	
	Returns
	-------
	int
		The Noll index.
	'''
	i = int(((n + 0.5)**2 + 1) / 2) + 1
	Nn = (n + 1) * (n + 2) // 2 + 1

	# Brute force search
	for j in range(i, i+Nn):
		nn, mm = noll_to_zernike(j)
		if nn == n and mm == m:
			return j
	raise ValueError('Could not find noll index for (%d,%d)' % n, m)

def zernike_radial(n, m, r):
	'''The radial component of a Zernike polynomial.

	Parameters
	----------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	r : array_like
		The (normalized) radial coordinates on which to calculate the polynomial.
	
	Returns
	-------
	array_like
		The radial component of the evaluated Zernike polynomial.
	'''
	R = np.zeros_like(r)
	for k in range((n - m) // 2 + 1):
		R += (-1)**k * binom(n - k, k) * binom(n - 2 * k, (n - m) // 2 - k) * r**(n - 2 * k)
	return R

def zernike_azimuthal(m, theta):
	'''The azimuthal component of a Zernike polynomial.

	Parameters
	----------
	m : int
		The azimuthal Zernike order.
	theta : array_like
		The azimuthal coordinates on which to calculate the polynomial.
	
	Returns
	-------
	array_like
		The azimuthal component of the evaluated Zernike polynomial.
	'''
	if m < 0:
		return sqrt(2) * np.sin(-m * theta)
	elif m == 0:
		return 1
	else:
		return sqrt(2) * np.cos(m * theta)

def zernike(n, m, D=1, grid=None, radial_cutoff=True):
	'''Evaluate the Zernike polynomial on a grid.

	Parameters
	----------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	D : scalar
		The diameter of the Zernike polynomial.
	grid : Grid
		The grid on which to evaluate the Zernike polynomial. If this is None,
		a Field generator will be returned.
	radial_cutoff : boolean
		Whether to apply a circular aperture to cutoff the modes.
	
	Returns
	-------
	Field or Field generator
		The evaluated Zernike polynomial. If `grid` is None, a Field generator is returned,
		which evaluates the Zernike polynomial on the supplied grid.
	'''
	from ..field import Field

	if grid is None:
		return lambda grid: zernike(n, m, D, grid)
	
	if grid.is_separated and grid.is_('polar'):
		R, Theta = grid.separated_coords
		z_r = zernike_radial(n, abs(m), 2 * R / D)
		if radial_cutoff:
			z_r *= (2 * R) < D
		z = sqrt(n + 1) * np.outer(zernike_azimuthal(m, Theta), z_r).flatten()
	else:
		r, theta = grid.as_('polar').coords
		z = sqrt(n + 1) * zernike_azimuthal(m, theta) * zernike_radial(n, abs(m), 2 * r / D)
		if radial_cutoff:
			z *= (2 * r) < D
	
	return Field(z, grid)

def zernike_ansi(i, D=1, grid=None, radial_cutoff=True):
	'''Evaluate the Zernike polynomial on a grid using an ANSI index.

	Parameters
	----------
	i : int
		The ANSI index.
	D : scalar
		The diameter of the Zernike polynomial.
	grid : Grid
		The grid on which to evaluate the Zernike polynomial. If this is None,
		a Field generator will be returned.
	radial_cutoff : boolean
		Whether to apply a circular aperture to cutoff the modes.
	
	Returns
	-------
	Field or Field generator
		The evaluated Zernike polynomial. If `grid` is None, a Field generator is returned,
		which evaluates the Zernike polynomial on the supplied grid.
	'''
	n, m = ansi_to_zernike(i)
	return zernike(n, m, D, grid, radial_cutoff)

def zernike_noll(i, D=1, grid=None, radial_cutoff=True):
	'''Evaluate the Zernike polynomial on a grid using a Noll index.

	Parameters
	----------
	i : int
		The Noll index.
	D : scalar
		The diameter of the Zernike polynomial.
	grid : Grid
		The grid on which to evaluate the Zernike polynomial. If this is None,
		a Field generator will be returned.
	radial_cutoff : boolean
		Whether to apply a circular aperture to cutoff the modes.
	
	Returns
	-------
	Field or Field generator
		The evaluated Zernike polynomial. If `grid` is None, a Field generator is returned,
		which evaluates the Zernike polynomial on the supplied grid.
	'''
	n, m = noll_to_zernike(i)
	return zernike(n, m, D, grid, radial_cutoff)

def make_zernike_basis(num_modes, D, grid, starting_mode=1, ansi=False, radial_cutoff=True):
	'''Make a ModeBasis of Zernike polynomials.

	Parameters
	----------
	num_modes : int
		The number of Zernike polynomials to generate.
	D : scalar
		The diameter of the Zernike polynomial.
	grid : Grid
		The grid on which to evaluate the Zernike polynomials.
	starting_mode : int
		The first mode to evaluate.
	ansi : boolean
		If this is True, the modes will be indexed using ANSI indices. Othewise, a Noll 
		indexing scheme is used.
	radial_cutoff : boolean
		Whether to apply a circular aperture to cutoff the modes.
	
	Returns
	-------
	ModeBasis
		The evaluated mode basis of Zernike polynomials.
	'''
	from .mode_basis import ModeBasis
	f = zernike_ansi if ansi else zernike_noll

	modes = [f(i, D, grid, radial_cutoff) for i in range(starting_mode, starting_mode+num_modes)]
	return ModeBasis(modes)