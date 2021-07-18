from ..field import Field, make_uniform_grid
import numpy as np
from scipy import sparse

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
	if hasattr(kernel, 'grid'):
		if np.all(kernel.grid.delta == grid.delta):
			num_x = kernel.grid.shape[1]
			num_y = kernel.grid.shape[0]
		else:
			raise ValueError("Kernel and grid are sampled with different grid spacings.")
	else:
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
		kernel_grid = make_uniform_grid((num_x, num_y), (1, 1))
		kernel = kernel_grid.zeros().shaped
		kernel[1, 1] = 4
		kernel[1, 0] = -1
		kernel[1, 2] = -1
		kernel[0, 1] = -1
		kernel[2, 1] = -1
		kernel = Field(kernel.ravel(), kernel_grid)

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
		kernel_grid = make_uniform_grid((num_x, num_y), (1, 1))
		kernel = kernel_grid.zeros()
		kernel = kernel.shaped

		if axis == 'x':
			kernel[1, 0] = -1 / (2 * grid.delta[1])
			kernel[1, 2] = 1 / (2 * grid.delta[1])
		elif axis == 'y':
			kernel[0, 1] = -1 / (2 * grid.delta[0])
			kernel[2, 1] = 1 / (2 * grid.delta[0])
		else:
			raise NotImplementedError()

		kernel = Field(kernel.ravel(), kernel_grid)
		return generate_convolution_matrix(grid, kernel)
	else:
		raise NotImplementedError()
