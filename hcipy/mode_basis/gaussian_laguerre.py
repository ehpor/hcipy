import numpy as np
from math import sqrt, floor
from scipy.special import eval_genlaguerre

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
	
	return Field(z, grid)

def gaussian_laguerre_ansi(i, mode_field_diameter=1, grid=None):
	# Map mode index to p, l coordinates
	p,l = ansi_to_zernike(i)
	return gaussian_laguerre(n, m, mode_field_diameter, grid)

# High level functions
def make_gaussian_laguerre_basis(num_modes, mode_field_diameter, grid, starting_mode=1):
	from .mode_basis import ModeBasis

	modes = [gaussian_laguerre_ansi(i, mode_field_diameter, grid) for i in range(starting_mode, starting_mode+num_modes)]
	return ModeBasis(modes)