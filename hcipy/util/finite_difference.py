import numpy as np
from scipy import sparse

from ..field import Field

def generate_convolution_matrix(grid, kernel):
	'''Create the matrix that applies a convolution with kernel.

	The created matrix is a sparse matrix.
	
	Parameters
	----------
	grid : Grid
		The :class:`Grid` on which to calculate the modes.
	kernel : array_like
		The convolution kernel
	
	Returns
	-------
	array_like
		The matrix that applies the convolution.
	'''

	Nx = kernel.shape[1]
	Ny = kernel.shape[0]

	YY, XX = np.meshgrid(np.arange(Ny), np.arange(Ny))
	offsets = ( (XX - Nx//2) + (YY-Ny//2) * grid.shape[0] ).ravel()
	convolution_matrix = sparse.diags(kernel, offsets, shape=(grid.size, grid.size))
	return convolution_matrix



def make_laplacian_matrix(grid):
	'''Make Laplacian operator using the 5-point stencil approximation
	'''
	if grid.is_('cartesian') and grid.is_separated and grid.is_regular:
		Nx = 3
		Ny = 3
		kernel = np.zeros((Ny,Nx))
		kernel[1,1] = 4
		kernel[1,0] = -1
		kernel[1,2] = -1
		kernel[0,1] = -1
		kernel[2,1] = -1
		kernel = kernel.ravel()

		return generate_convolution_matrix(grid, kernel)

	else:
		raise NotImplementedError()


def make_derivative_matrix(grid, axis='x'):
	'''Make Derivative operator using the 5-point stencil approximation
	'''
	if grid.is_('cartesian') and grid.is_separated and grid.is_regular:
		Nx = 3
		Ny = 3
		kernel = np.zeros((Ny,Nx))

		if axis == 'x':
			kernel[1,0] = -1 / (2 * grid.delta[1])
			kernel[1,2] = 1 / (2 * grid.delta[1])
		elif axis == 'y':
			kernel[0,1] = -1 / (2 * grid.delta[0])
			kernel[2,1] = 1 / (2 * grid.delta[0])
		kernel = kernel.ravel()

		return generate_convolution_matrix(grid, kernel)

	else:
		raise NotImplementedError()