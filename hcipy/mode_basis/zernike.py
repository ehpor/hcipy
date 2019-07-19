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

def zernike_radial(n, m, r, cache=None):
	'''The radial component of a Zernike polynomial.

	We use the q-recursive method, which uses recurrence relations to calculate the radial
	Zernike polynomials without using factorials. A description of the method can be found
	in [1]_. Additionally, this function optionally caches results of previous calls.

	.. [1] Chong, C. W., Raveendran, P., & Mukundan, R. (2003). A comparative analysis of algorithms for fast computation of Zernike moments. Pattern Recognition, 36(3), 731-742.

	Parameters
	----------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	r : array_like
		The (normalized) radial coordinates on which to calculate the polynomial.
	cache : dictionary or None
		A dictionary containing previously calculated Zernike modes on the same grid.
		This function is for speedup only, and therefore the cache is expected to be 
		valid. You can reuse the cache for future calculations on the same exact grid.
		The given dictionary is updated with the current calculation.
	
	Returns
	-------
	array_like
		The radial component of the evaluated Zernike polynomial.
	'''
	m = abs(m)

	if cache is not None:
		if ('rad', n, m) in cache:
			return cache[('rad', n, m)]
	
	if n == m:
		res = r**n
	elif (n - m) == 2:
		z1 = zernike_radial(n, n, r, cache)
		z2 = zernike_radial(n - 2, n - 2, r, cache)

		res = n * z1 - (n - 1) * z2
	else:
		p = n
		q = m + 4

		h3 = -4 * (q - 2) * (q - 3) / float((p + q - 2) * (p - q + 4))
		h2 = h3 * (p + q) * (p - q + 2) / float(4 * (q - 1)) + (q - 2)
		h1 = q * (q - 1) / 2.0 - q * h2 + h3 * (p + q + 2) * (p - q) / 8.0

		r2 = zernike_radial(2, 2, r, cache)
		res = h1 * zernike_radial(p, q, r, cache) + (h2 + h3 / r2) * zernike_radial(n, q - 2, r, cache)

	if cache is not None:
		cache[('rad', n, m)] = res
	
	return res

def zernike_azimuthal(m, theta, cache=None):
	'''The azimuthal component of a Zernike polynomial.

	This function optionally caches results of previous calls.

	Parameters
	----------
	m : int
		The azimuthal Zernike order.
	theta : array_like
		The azimuthal coordinates on which to calculate the polynomial.
	cache : dictionary or None
		A dictionary containing previously calculated Zernike modes on the same grid.
		This function is for speedup only, and therefore the cache is expected to be 
		valid. You can reuse the cache for future calculations on the same exact grid.
		The given dictionary is updated with the current calculation.
	
	Returns
	-------
	array_like
		The azimuthal component of the evaluated Zernike polynomial.
	'''
	if cache is not None:
		if ('azim', m) in cache:
			return cache[('azim', m)]
	
	if m < 0:
		res = sqrt(2) * np.sin(-m * theta)
	elif m == 0:
		return 1
	else:
		res = sqrt(2) * np.cos(m * theta)
	
	if cache is not None:
		cache[('azim', m)] = res
	
	return res
	
def zernike(n, m, D=1, grid=None, radial_cutoff=True, cache=None):
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
	cache : dictionary or None
		A dictionary containing previously calculated Zernike modes on the same grid.
		This function is for speedup only, and therefore the cache is expected to be 
		valid. You can reuse the cache for future calculations on the same exact grid.
		The given dictionary is updated with the current calculation.
	
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
		z_r = zernike_radial(n, m, 2 * R / D, cache)
		if radial_cutoff:
			z_r *= (2 * R) < D
		z = sqrt(n + 1) * np.outer(zernike_azimuthal(m, Theta, cache), z_r).flatten()
	else:
		r, theta = grid.as_('polar').coords
		z = sqrt(n + 1) * zernike_azimuthal(m, theta, cache) * zernike_radial(n, m, 2 * r / D, cache)
		if radial_cutoff:
			z *= (2 * r) < D
	
	return Field(z, grid)

def zernike_ansi(i, D=1, grid=None, radial_cutoff=True, cache=None):
	'''Evaluate the Zernike polynomial on a grid using an ANSI index.

	Parameters
	----------
	i : int
		The ANSI index.
	D : scalar
		The diameter of the Zernike polynomial.
	grid : Grid or None
		The grid on which to evaluate the Zernike polynomial. If this is None,
		a Field generator will be returned.
	radial_cutoff : boolean
		Whether to apply a circular aperture to cutoff the modes.
	cache : dictionary or None
		A dictionary containing previously calculated Zernike modes on the same grid.
		This function is for speedup only, and therefore the cache is expected to be 
		valid. You can reuse the cache for future calculations on the same exact grid.
		The given dictionary is updated with the current calculation.
	
	Returns
	-------
	Field or Field generator
		The evaluated Zernike polynomial. If `grid` is None, a Field generator is returned,
		which evaluates the Zernike polynomial on the supplied grid.
	'''
	n, m = ansi_to_zernike(i)
	return zernike(n, m, D, grid, radial_cutoff, cache)

def zernike_noll(i, D=1, grid=None, radial_cutoff=True, cache=None):
	'''Evaluate the Zernike polynomial on a grid using a Noll index.

	Parameters
	----------
	i : int
		The Noll index.
	D : scalar
		The diameter of the Zernike polynomial.
	grid : Grid or None
		The grid on which to evaluate the Zernike polynomial. If this is None,
		a Field generator will be returned.
	radial_cutoff : boolean
		Whether to apply a circular aperture to cutoff the modes.
	cache : dictionary or None
		A dictionary containing previously calculated Zernike modes on the same grid.
		This function is for speedup only, and therefore the cache is expected to be 
		valid. You can reuse the cache for future calculations on the same exact grid.
		The given dictionary is updated with the current calculation.
	
	Returns
	-------
	Field or Field generator
		The evaluated Zernike polynomial. If `grid` is None, a Field generator is returned,
		which evaluates the Zernike polynomial on the supplied grid.
	'''
	n, m = noll_to_zernike(i)
	return zernike(n, m, D, grid, radial_cutoff, cache)

def make_zernike_basis(num_modes, D, grid, starting_mode=1, ansi=False, radial_cutoff=True, use_cache=True):
	'''Make a ModeBasis of Zernike polynomials.

	Parameters
	----------
	num_modes : int
		The number of Zernike polynomials to generate.
	D : scalar
		The diameter of the Zernike polynomial.
	grid : Grid or None
		The grid on which to evaluate the Zernike polynomials. If this is None,
		a list of Field generators will be returned.
	starting_mode : int
		The first mode to evaluate.
	ansi : boolean
		If this is True, the modes will be indexed using ANSI indices. Otherwise, a Noll 
		indexing scheme is used.
	radial_cutoff : boolean
		Whether to apply a circular aperture to cutoff the modes.
	use_cache : boolean
		Whether to use a cache while calculating the modes. A cache uses memory, so turn it
		off when you are limited on memory.
	
	Returns
	-------
	ModeBasis or list of Field generators
		The evaluated mode basis of Zernike polynomials, or a list of Field generators for
		each of the Zernike polynomials.
	'''
	from .mode_basis import ModeBasis
	f = zernike_ansi if ansi else zernike_noll

	if grid is None:
		polar_grid = None
	else:
		polar_grid = grid.as_('polar')
	
	if use_cache:
		cache = {}
	else:
		cache = None

	modes = [f(i, D, polar_grid, radial_cutoff, cache) for i in range(starting_mode, starting_mode + num_modes)]

	if grid is None:
		return modes
	else:
		return ModeBasis(modes)
