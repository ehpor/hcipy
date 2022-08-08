import numpy as np
import os
from hcipy import *

def test_grid_io():
	grid1 = make_pupil_grid(128)
	grid2 = CartesianGrid(SeparatedCoords(grid1.separated_coords))
	grid3 = grid1.as_('polar')

	grids = [grid1, grid2, grid3]
	formats = ['asdf', 'fits', 'fits.gz', 'pkl', 'pickle']
	filenames = ['grid_test.' + fmt for fmt in formats]

	for g in grids:
		for fname in filenames:
			write_grid(g, fname)
			new_grid = read_grid(fname)

			assert hash(g) == hash(new_grid)

			os.remove(fname)

def test_field_io():
	grid = make_pupil_grid(128)
	field = make_circular_aperture(1)(grid)

	formats = ['asdf', 'fits', 'fits.gz', 'pkl', 'pickle']
	filenames = ['field_test.' + fmt for fmt in formats]

	for fname in filenames:
		write_field(field, fname)
		new_field = read_field(fname)

		assert np.allclose(field, new_field)
		assert hash(field.grid) == hash(new_field.grid)

		os.remove(fname)

def test_mode_basis_io():
	grid = make_pupil_grid(128)
	mode_bases = [
		make_zernike_basis(20, 1, grid, 1),
		make_xinetics_influence_functions(grid, 8, 1 / 8)
	]

	formats = ['asdf', 'fits', 'fits.gz', 'pkl', 'pickle']
	filenames = ['mode_basis_test.' + fmt for fmt in formats]

	for mode_basis in mode_bases:
		for fname in filenames:
			write_mode_basis(mode_basis, fname)
			new_mode_basis = read_mode_basis(fname)

			assert hash(new_mode_basis.grid) == hash(mode_basis.grid)
			assert new_mode_basis.is_sparse == mode_basis.is_sparse

			for i in range(mode_basis.num_modes):
				assert np.allclose(mode_basis[i], new_mode_basis[i])

			os.remove(fname)


def test_finite_difference():

	grid = make_pupil_grid(16)
	surface = grid.ones()

	Dx = make_derivative_matrix(grid, axis='x')
	Dy = make_derivative_matrix(grid, axis='y')

	surface_dx = Dx.dot(surface)
	surface_dy = Dy.dot(surface)

	assert abs(np.median(surface_dx)) < 1e-10
	assert abs(np.median(surface_dy)) < 1e-10

	laplacian = make_laplacian_matrix(grid)
	surface_lap = laplacian.dot(surface)
	assert abs(np.median(surface_lap)) < 1e-10

	# Test if numpy array is handled correctly b
	laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
	laplacian_operator = generate_convolution_matrix(grid, laplacian_kernel)
	surface_lap = laplacian_operator.dot(surface)
	assert abs(np.median(surface_lap)) < 1e-10
