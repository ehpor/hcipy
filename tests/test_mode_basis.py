from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

def test_zernike_modes():
	grid = make_pupil_grid(256)
	aperture_mask = circular_aperture(1)(grid) > 0

	modes = make_zernike_basis(200, 1, grid)
	
	assert np.abs(np.std(modes[0][aperture_mask])) < 2e-2

	for m in modes[1:]:
		assert np.abs(np.std(m[aperture_mask]) - 1) < 2e-2
	
	for i, m in enumerate(modes):
		zn, zm = noll_to_zernike(i+1)
		assert np.allclose(m, zernike(zn, zm, grid=grid))

def test_zernike_indices():
	for i in range(1,500):
		n, m = noll_to_zernike(i)
		assert i == zernike_to_noll(n, m)

		n, m = ansi_to_zernike(i)
		assert i == zernike_to_ansi(n, m)

def test_disk_harmonic_modes():
	grid = make_pupil_grid(128)
	aperture_mask = circular_aperture(1)(grid) > 0

	num_modes = 20

	for bc in ['dirichlet', 'neumann']:
		modes = make_disk_harmonic_basis(grid, num_modes, bc=bc)
		
		for i, m1 in enumerate(modes):
			for j, m2 in enumerate(modes):
				product = np.sum((m1 * m2)[aperture_mask] * grid.weights[aperture_mask])
				print(product)

				assert np.abs(product - np.eye(num_modes)[i,j]) < 1e-2