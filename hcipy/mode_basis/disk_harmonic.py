import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros

from .mode_basis import ModeBasis
from ..aperture import circular_aperture
from ..field import Field

def disk_harmonic(n, m, D=1, bc='dirichlet', grid=None):
	'''Create a disk harmonic.

	Parameters
	----------
	n : int
		Radial order
	m : int
		Azimuthal order
	D : scalar
		The diameter of the pupil.
	bc : string
		The boundary conditions to use. This can be either 'dirichlet', or 
		'neumann' for a Dirichlet or Neumann boundary condition respectively.
	grid : Grid
		The grid on which to evaluate the function.
	
	Returns
	-------
	Field
		The disk harmonic function evaluated on `grid`.
	'''
	polar_grid = grid.as_('polar')
	r = 2 * polar_grid.r / D
	theta = polar_grid.theta

	m_negative = m < 0
	m = abs(m)

	if bc == 'dirichlet':
		lambda_mn = jn_zeros(m, n)[-1]
		norm = 1
	elif bc == 'neumann':
		lambda_mn = jnp_zeros(m, n)[-1]
		norm = 1
	else:
		raise RuntimeError('Boundary condition not recongnized.')
	
	if m_negative:
		z = norm * jv(m, lambda_mn * r) * np.sin(m * theta)
	else:
		z = norm * jv(m, lambda_mn * r) * np.cos(m * theta)

	# Do manual normalization for now...
	mask = circular_aperture(D)(grid) > 0.5
	norm = np.sqrt(np.sum(z[mask]**2 * grid.weights[mask]))

	return Field(z * mask / norm, grid)

def disk_harmonic_energy(n, m, bc='dirichlet'):
	'''Get the energy of a disk harmonic function.

	This allows for functions to sort a disk harmonic mode basis on energy.

	Parameters
	----------
	n : int
		Radial order
	m : int
		Azimuthal order
	bc : string
		The boundary conditions to use. This can be either 'dirichlet', or 
		'neumann' for a Dirichlet or Neumann boundary condition respectively.
	
	Returns
	-------
	scalar
		The energy corresponding to the mode.
	'''
	m = abs(m)

	if bc == 'dirichlet':
		lambda_mn = jn_zeros(m, n)[-1]
	elif bc == 'neumann':
		lambda_mn = jnp_zeros(m, n)[-1]
	else:
		raise RuntimeError('Boundary condition not recognized.')
	
	return lambda_mn**2

def get_disk_harmonic_orders_sorted(num_modes, bc='dirichlet'):
	'''Return the orders for the first `num_modes` modes.

	Parameters
	----------
	num_modes : int
		The number of modes to return.
	bc : string
		The boundary conditions to use. This can be either 'dirichlet', or 
		'neumann' for a Dirichlet or Neumann boundary condition respectively.
	
	Returns
	-------
	list of tuples
		A list of orders (n, m).
	'''
	orders = [(1,0)]
	energies = [disk_harmonic_energy(1, 0, bc)]

	results = []
	while len(results) < num_modes:
		k = np.argmin(energies)
		order = orders[k]
		
		if order[1] != 0:
			results.append((order[0], -order[1]))
		results.append(order)

		del orders[k]
		del energies[k]

		new_order = (order[0], order[1] + 1)
		if new_order not in results and new_order not in orders:
			orders.append(new_order)
			energies.append(disk_harmonic_energy(new_order[0], new_order[1], bc))
		new_order = (order[0] + 1, order[1])
		if new_order not in results and new_order not in orders:
			orders.append(new_order)
			energies.append(disk_harmonic_energy(new_order[0], new_order[1], bc))

	return results[:num_modes]

def make_disk_harmonic_basis(grid, num_modes, D=1, bc='dirichlet'):
	'''Create a disk harmonic mode basis.

	Parameters
	----------
	grid : Grid
		The grid on which to evaluate the disk harmonic modes.
	num_modes : int
		The number of modes to create.
	D : scalar
		The diameter of the disk.
	bc : string
		The boundary conditions to use. This can be either 'dirichlet', or 
		'neumann' for a Dirichlet or Neumann boundary condition respectively.
	
	Returns
	-------
	ModeBasis
		The evaluated disk harmonic modes.
	'''
	orders = get_disk_harmonic_orders_sorted(num_modes, bc)

	modes = [disk_harmonic(order[0], order[1], D, bc, grid) for order in orders]
	return ModeBasis(modes)