import numpy as np
from math import sqrt, floor
from scipy.special import binom

# Indexing
def noll_to_zernike(i):
	n = int(sqrt(2*i-1) + 0.5) - 1
	if n % 2:
		m = 2*int((2*(i+1) - n*(n+1))//4) - 1
	else:
		m = 2*int((2*i+1 - n*(n+1))//4)
	return n, m*(-1)**(i%2)

def ansi_to_zernike(i):
	n = int((sqrt(8*i+1)-1)/2)
	m = 2*i - n*(n+2)
	return (n, m)

def zernike_to_ansi(n, m):
	return (m+n*n)//2 + n

def zernike_to_noll(n, m):
	i = int(((n+0.5)**2 + 1) / 2) + 1
	Nn = (n+1)*(n+2)//2+1

	# Brute force search
	for j in range(i, i+Nn):
		nn, mm = noll_to_zernike(j)
		if nn==n and mm==m:
			return j
	raise ValueError('Could not find noll index for (%d,%d)' % n, m)

# Zernike functions
def zernike_radial(n, m, r):
	R = np.zeros_like(r)
	for k in range((n-m)//2+1):
		R += (-1)**k * binom(n-k, k) * binom(n-2*k, (n-m)//2 - k) * r**(n-2*k)
	return R

def zernike_azimuthal(m, theta):
	if m < 0:
		return sqrt(2)*np.sin(-m*theta)
	elif m == 0:
		return 1
	else:
		return sqrt(2)*np.cos(m*theta)

def zernike(n, m, D=1, grid=None):
	from ..field import Field

	if grid is None:
		return lambda grid: zernike(n, m, D, grid)
	
	if grid.is_separated and grid.is_('polar'):
		R, Theta = grid.separated_coords
		z = sqrt(n+1) * np.outer(zernike_azimuthal(m, Theta), zernike_radial(n, abs(m), 2*R/D) * (2*R<D)).flatten()
	else:
		r, theta = grid.as_('polar').coords
		z = sqrt(n+1) * zernike_azimuthal(m, theta) * zernike_radial(n, abs(m), 2*r/D) * (2*r<D)
	
	return Field(z, grid)

def zernike_ansi(i, D=1, grid=None):
	n, m = ansi_to_zernike(i)
	return zernike(n, m, D, grid)

def zernike_noll(i, D=1, grid=None):
	n, m = noll_to_zernike(i)
	return zernike(n, m, D, grid)

# High level functions
def make_zernike_basis(num_modes, D, grid, starting_mode=1, ansi=False):
	from .mode_basis import ModeBasis
	f = zernike_ansi if ansi else zernike_noll

	modes = [f(i, D, grid) for i in range(starting_mode, starting_mode+num_modes)]
	return ModeBasis(modes)