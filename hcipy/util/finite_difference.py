import numpy as np
from scipy import sparse

from ..field import Field

def generate_convolution_matrix(grid, kernel):
	'''Create the matrix that applies a convolution with kernel.

	The created matrix is a sparse matrix.
	
	Parameters
	----------
	grid : Grid
		The :class:`Grid` for which the convolution matrix will be created.
	kernel : array_like
		The convolution kernel
	
	Returns
	-------
	array_like
		The matrix that applies the convolution.
	'''
	num_x = kernel.shape[1]
	num_y = kernel.shape[0]

	yy, xx = np.meshgrid(np.arange(num_y), np.arange(num_y))
	offsets = ((xx - num_x // 2) + (yy - num_y // 2) * grid.shape[0] ).ravel()
	convolution_matrix = sparse.diags(kernel, offsets, shape=(grid.size, grid.size))
	return convolution_matrix

def make_laplacian_matrix(grid):
	'''Make the Laplacian operator using the 5-point stencil approximation
	
	Parameters
	----------
	grid : Grid
		The grid for which the derivative matrix is calculated.
	
	Returns
	-------
	array_like
		The convolution matrix.
	'''
	if grid.is_('cartesian') and grid.is_separated and grid.is_regular:
		num_x = 3
		num_y = 3
		kernel = np.zeros((num_y, num_x))
		kernel[1, 1] = 4
		kernel[1, 0] = -1
		kernel[1, 2] = -1
		kernel[0, 1] = -1
		kernel[2, 1] = -1
		kernel = kernel.ravel()

		return generate_convolution_matrix(grid, kernel)
	else:
		raise NotImplementedError()

def make_derivative_matrix(grid, axis='x'):
	'''Make the derivative operator using the central difference approximation.

	Parameters
	----------
	grid : Grid
		The grid for which the derivative matrix is calculated.
	axis : string
		The axis for which the convolution kernel is calculated default is 'x'.

	Returns
	-------
	array_like
		The convolution matrix.
	'''
	if grid.is_('cartesian') and grid.is_separated and grid.is_regular:
		num_x = 3
		num_y = 3
		kernel = np.zeros((num_y, num_x))

		if axis == 'x':
			kernel[1, 0] = -1 / (2 * grid.delta[1])
			kernel[1, 2] = 1 / (2 * grid.delta[1])
		elif axis == 'y':
			kernel[0, 1] = -1 / (2 * grid.delta[0])
			kernel[2, 1] = 1 / (2 * grid.delta[0])
		else:
			raise NotImplementedError()

		kernel = kernel.ravel()
		return generate_convolution_matrix(grid, kernel)
	else:
		raise NotImplementedError()
