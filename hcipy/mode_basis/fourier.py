from .mode_basis import ModeBasis
from ..field import Field
import numpy as np

_epsilon = 1e-12

def make_cosine_basis(grid, fourier_grid, sort_by_energy=True):
	'''Make a cosine basis.

	Repeated frequencies will not be repeated in this mode basis. This means that opposite points in the `fourier_grid` will be silently ignored.

	Parameters
	----------
	grid : Grid
		The :class:`Grid` on which to calculate the modes.
	fourier_grid : Grid
		The grid defining all frequencies.
	sort_by_energy : bool
		Whether to sort by increasing energy or not.

	Returns
	-------
	ModeBasis
		The mode basis containing all cosine modes.
	'''
	modes = []
	energies = []
	ignore_list = []

	c = np.array(grid.coords)
	
	for i, p in enumerate(fourier_grid.points):
		if i in ignore_list:
			continue
		
		mode = Field(np.cos(np.dot(p, c)), grid)
		modes.append(mode)

		j = fourier_grid.closest_to(-p)

		dist = fourier_grid.points[j] + p
		dist2 = np.dot(dist, dist)

		p_length2 = np.dot(p, p)
		energies.append(p_length2)

		if dist2 < (_epsilon * p_length2):
			ignore_list.append(j)
	
	if sort_by_energy:
		ind = np.argsort(energies)
		modes = [modes[i] for i in ind]
	
	return ModeBasis(modes)

def make_sine_basis(grid, fourier_grid, sort_by_energy=True):
	'''Make a sine basis.

	Repeated frequencies will not be repeated in this mode basis. This means that opposite points in the `fourier_grid` will be silently ignored.

	Parameters
	----------
	grid : Grid
		The :class:`Grid` on which to calculate the modes.
	fourier_grid : Grid
		The grid defining all frequencies.
	sort_by_energy : bool
		Whether to sort by increasing energy or not.

	Returns
	-------
	ModeBasis
		The mode basis containing all sine modes.
	'''
	modes = []
	energies = []
	ignore_list = []

	c = np.array(grid.coords)

	for i, p in enumerate(fourier_grid.points):
		if i in ignore_list:
			continue
		
		mode = Field(np.sin(np.dot(p, c)), grid)
		modes.append(mode)

		j = fourier_grid.closest_to(-p)

		dist = fourier_grid.points[j] + p
		dist2 = np.dot(dist, dist)

		p_length2 = np.dot(p, p)
		energies.append(p_length2)

		if dist2 < (_epsilon * p_length2):
			ignore_list.append(j)
	
	if sort_by_energy:
		ind = np.argsort(energies)
		modes = [modes[i] for i in ind]
	
	return ModeBasis(modes)

def make_fourier_basis(grid, fourier_grid, sort_by_energy=True):
	'''Make a Fourier basis.

	Fourier modes this function are defined to be real. This means that for each point, both a sine and cosine mode is returned.
	
	Repeated frequencies will not be repeated in this mode basis. This means that opposite points in the `fourier_grid` will be silently ignored.

	Parameters
	----------
	grid : Grid
		The :class:`Grid` on which to calculate the modes.
	fourier_grid : Grid
		The grid defining all frequencies.
	sort_by_energy : bool
		Whether to sort by increasing energy or not.

	Returns
	-------
	ModeBasis
		The mode basis containing all Fourier modes.
	'''
	modes_cos = []
	modes_sin = []
	energies = []
	ignore_list = []

	c = np.array(grid.coords)

	for i, p in enumerate(fourier_grid.points):
		if i in ignore_list:
			continue

		mode_cos = Field(np.cos(np.dot(p, c)), grid)
		mode_sin = Field(np.sin(np.dot(p, c)), grid)

		modes_cos.append(mode_cos)
		modes_sin.append(mode_sin)

		j = fourier_grid.closest_to(-p)

		dist = fourier_grid.points[j] + p
		dist2 = np.dot(dist, dist)

		p_length2 = np.dot(p, p)
		energies.append(p_length2)

		if dist2 < (_epsilon * p_length2):
			ignore_list.append(j)
	
	if sort_by_energy:
		ind = np.argsort(energies)
		modes_sin = [modes_sin[i] for i in ind]
		modes_cos = [modes_cos[i] for i in ind]
		energies = np.array(energies)[ind]
	
	modes = []
	for i, E in enumerate(energies):
		modes.append(modes_cos[i])
		if E > _epsilon:
			modes.append(modes_sin[i])
	
	return ModeBasis(modes)

def make_complex_fourier_basis(grid, fourier_grid, sort_by_energy=True):
	'''Make a complex Fourier basis.

	Fourier modes this function are defined to be complex. For each point in `fourier_grid` the complex Fourier mode is contained in the output.

	Parameters
	----------
	grid : Grid
		The :class:`Grid` on which to calculate the modes.
	fourier_grid : Grid
		The grid defining all frequencies.
	sort_by_energy : bool
		Whether to sort by increasing energy or not.

	Returns
	-------
	ModeBasis
		The mode basis containing all Fourier modes.
	'''
	c = np.array(grid.coords)
	
	modes = [Field(np.exp(1j * np.dot(p, c)), grid) for p in fourier_grid.points]
	energies = [np.dot(p, p) for p in fourier_grid.points]

	if sort_by_energy:
		ind = np.argsort(energies)
		modes = [modes[i] for i in ind]
	
	return ModeBasis(modes)
