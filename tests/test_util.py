import numpy as np
import os
from hcipy import *

def test_grid_io():
	grid1 = make_pupil_grid(128)
	grid2 = CartesianGrid(SeparatedCoords(grid1.separated_coords))
	grid3 = grid1.as_('polar')

	grids = [grid1, grid2, grid3]
	formats = ['asdf', 'fits', 'fits.gz']
	filenames = ['grid_test.' + fmt for fmt in formats]

	for g in grids:
		for fname in filenames:
			write_grid(g, fname)
			new_grid = read_grid(fname)

			assert hash(g) == hash(new_grid)

	for fname in filenames:
		os.remove(fname)

def test_field_io():
	grid = make_pupil_grid(256, [1, 1])
	field = circular_aperture(1)(grid)

	formats = ['asdf', 'fits', 'fits.gz']
	filenames = ['field_test.' + fmt for fmt in formats]

	for fname in filenames:
		write_field(field, fname)
		new_field = read_field(fname)

		assert np.allclose(field, new_field)
		assert hash(field.grid) == hash(new_field.grid)

	for fname in filenames:
		os.remove(fname)

"""
def test_write_mode_basis():
	# grid for the test mode basis
	pupil_grid = make_pupil_grid(128)

	# generating a test mode basis
	test_mode_basis = make_zernike_basis(num_modes=20, D=1, grid=pupil_grid, starting_mode=1, ansi=False, radial_cutoff=True)

	file_name = 'write_mode_basis_test.fits'

	# saving the mode basis
	write_mode_basis(test_mode_basis, file_name)

	# loading it
	test_mode_basis_array_read = read_fits(file_name)

	# comparing the arrays
	test_mode_basis_array = np.array([test_mode_basis])
	test_mode_basis_array = np.reshape(test_mode_basis_array, [20,128,128])
	assert np.isclose(test_mode_basis_array, test_mode_basis_array_read, rtol=1e-02, atol=1e-05).all()

	# Remove temporary file.
	os.remove(file_name)

def test_read_mode_basis_1():
	#-------------------------------
	# testing a square mode basis that we read without providing a grid
	#-------------------------------

	# grid for the test mode basis
	pupil_grid = make_pupil_grid(128)

	# testing a square mode basis defined in the pupil plane
	test_mode_basis = make_zernike_basis(num_modes=20, D=1, grid=pupil_grid, starting_mode=1, ansi=False, radial_cutoff=True)

	# writing the mode basis
	file_name = 'read_mode_basis_test_1.fits'
	write_mode_basis(test_mode_basis, file_name)

	# and loading it again
	test_mode_basis_read = read_mode_basis(file_name, grid=None)

	# checking if the modes are still the same
	for mode, mode_read in zip(test_mode_basis, test_mode_basis_read):
		assert np.isclose(mode, mode_read, rtol=1e-02, atol=1e-05).all()

	# checking if the grid is correct
	assert np.isclose(pupil_grid.x, test_mode_basis_read[0].grid.x, rtol=1e-02, atol=1e-05).all()
	assert np.isclose(pupil_grid.y, test_mode_basis_read[0].grid.y, rtol=1e-02, atol=1e-05).all()

	# Remove temporary file.
	os.remove(file_name)

def test_read_mode_basis_2():
	#-------------------------------
	# testing a square mode basis that we read with providing a grid
	#-------------------------------

	# grid for the test mode basis
	pupil_grid = make_pupil_grid(128, 3)

	# testing a square mode basis defined in the pupil plane
	test_mode_basis = make_zernike_basis(num_modes=20, D=3, grid=pupil_grid, starting_mode=1, ansi=False, radial_cutoff=True)

	# writing the mode basis
	file_name = 'read_mode_basis_test_2.fits'
	write_mode_basis(test_mode_basis, file_name)

	# and loading it again
	test_mode_basis_read = read_mode_basis(file_name, grid=pupil_grid)

	# checking if the modes are still the same
	for mode, mode_read in zip(test_mode_basis, test_mode_basis_read):
		assert np.isclose(mode, mode_read, rtol=1e-02, atol=1e-05).all()

	# checking if the grid is correct
	assert np.isclose(pupil_grid.x, test_mode_basis_read[0].grid.x, rtol=1e-02, atol=1e-05).all()
	assert np.isclose(pupil_grid.y, test_mode_basis_read[0].grid.y, rtol=1e-02, atol=1e-05).all()

	# Remove temporary file.
	os.remove(file_name)

def test_read_mode_basis_3():
	#-------------------------------
	# testing a non-square mode basis that we read with providing a grid
	#-------------------------------

	# grid for the test mode basis
	pupil_grid = make_uniform_grid([128,256], [128,256], center=0, has_center=False)

	# testing a square mode basis defined in the pupil plane
	test_mode_basis = []
	for i in np.arange(20):
		test_mode_basis.append(Field(np.random.rand(128*256), pupil_grid))
	test_mode_basis = ModeBasis(test_mode_basis)

	# writing the mode basis
	file_name = 'read_mode_basis_test_3.fits'
	write_mode_basis(test_mode_basis, file_name)

	# and loading it again
	test_mode_basis_read = read_mode_basis(file_name, grid=pupil_grid)

	# checking if the modes are still the same
	for mode, mode_read in zip(test_mode_basis, test_mode_basis_read):
		assert np.isclose(mode, mode_read, rtol=1e-02, atol=1e-05).all()

	# checking if the grid is correct
	assert np.isclose(pupil_grid.x, test_mode_basis_read[0].grid.x, rtol=1e-02, atol=1e-05).all()
	assert np.isclose(pupil_grid.y, test_mode_basis_read[0].grid.y, rtol=1e-02, atol=1e-05).all()

	# Remove temporary file.
	os.remove(file_name)
"""
