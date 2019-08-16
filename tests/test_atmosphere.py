from hcipy import *
import numpy as np
from math import *
import mpmath
import scipy
import matplotlib.pyplot as plt

import pytest

def zernike_variance_von_karman(n, m, R, k0, Cn_squared, wavelength):
	'''Calculate the variance of the Zernike mode (`n`,`m`), using a von Karman turbulence spectrum.

	Parameters
	----------
	n : int
		The radial Zernike order.
	m : int
		The azimuthal Zernike order.
	R : scalar
		The radius of the aperture.
	k0 : scalar
		The spatial frequency of the outer scale (1/L0).
	Cn_squared : scalar
		The integrated Cn^2 profile.
	wavelength : scalar
		The wavelength at which to calculate the variance.
	
	Returns
	-------
	scalar
		The variance of the specific Zernike mode.
	'''
	A = 0.00969 * (2*np.pi / wavelength)**2 * Cn_squared
	coeffs_all = (-1)**(n - m) * 2 * (2 * np.pi)**(11./3) * (n + 1) * A * R**(5./3) / (sqrt(np.pi) * np.sin(np.pi * (n + 1./6)))

	term11 = mpmath.hyper([n + (3./2), n + 2, n + 1],[n + (1./6), n + 2, n + 2, 2 * n + 3], (2*np.pi * R * k0)**2)
	term12 = sqrt(np.pi) * (2*np.pi * R * k0)**(2 * n - 5./3) * scipy.special.gamma(n + 1) / (2**(2 * n + 3) * scipy.special.gamma(11./6) * scipy.special.gamma(n + 1./6) * scipy.special.gamma(n + 2)**2)
	term21 = -1 * scipy.special.gamma(7./3) * scipy.special.gamma(17./6) / (2 * scipy.special.gamma(-n + 11./6) * scipy.special.gamma(17./6)**2 * scipy.special.gamma(n + 23./6))
	term22 = mpmath.hyper([11./6, 7./3, 17./6], [-n + 11./6, 17./6, 17./6, n + 23./6], (2*np.pi * R * k0)**2)

	return coeffs_all * (term11 * term12 + term21 * term22)

def check_total_variance(wavelength, D_tel, fried_parameter, outer_scale, propagate_phase_screen):
	velocity = 10.0 # meters/sec
	num_modes = 1000

	pupil_grid = make_pupil_grid(64, D_tel)
	aperture = circular_aperture(D_tel)(pupil_grid)
	wf = Wavefront(aperture, wavelength)

	Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)
	layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, [velocity / np.sqrt(2), velocity / np.sqrt(2)])

	num_iterations = 2000
	total_variance = []

	for it in range(num_iterations):
		layer.reset()
		if propagate_phase_screen:
			layer.t = np.sqrt(2) * D_tel / velocity
		
		phase = layer.phase_for(wavelength)
		total_variance.append(np.var(phase[aperture>0]))
	
	variance_measured = np.mean(total_variance)
	
	variance_theory = 0
	for i in range(num_modes):
		n, m = noll_to_zernike(i + 2)
		variance_theory += zernike_variance_von_karman(n, m, D_tel / 2., 1. / outer_scale, layer.Cn_squared, wavelength)
	
	assert (variance_measured / variance_theory - 1) < 0.1

def check_zernike_variances(wavelength, D_tel, fried_parameter, outer_scale, propagate_phase_screen):
	velocity = 10.0 # meters/sec
	num_modes = 50

	pupil_grid = make_pupil_grid(128, D_tel)
	aperture = circular_aperture(D_tel)(pupil_grid)

	Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)
	layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, [velocity / np.sqrt(2), velocity / np.sqrt(2)])

	zernike_modes = make_zernike_basis(num_modes + 20, D_tel, pupil_grid, starting_mode=2, radial_cutoff=False)
	
	weights = evaluate_supersampled(circular_aperture(D_tel), pupil_grid, 32)
	zernike_modes = ModeBasis([z * np.sqrt(weights) for z in zernike_modes])
	
	transformation_matrix = zernike_modes.transformation_matrix
	projection_matrix = inverse_tikhonov(transformation_matrix, 1e-9)

	num_iterations = 300
	mode_coeffs = []

	for it in range(num_iterations):
		layer.reset()
		if propagate_phase_screen:
			layer.t = np.sqrt(2) * D_tel / velocity
		
		phase = layer.phase_for(wavelength)
		coeffs = projection_matrix.dot(phase * np.sqrt(weights))[:num_modes]
		mode_coeffs.append(coeffs)
		
	variances_simulated = np.var(mode_coeffs, axis=0)

	variances_theory = []
	for j in range(num_modes):
		n, m = noll_to_zernike(j + 2)
		variances_theory.append(zernike_variance_von_karman(n, m, D_tel / 2., 1. / outer_scale, layer.Cn_squared, wavelength))
	variances_theory = np.array(variances_theory)
	'''
	plt.plot(variances_simulated, label='simulated')
	plt.plot(variances_theory, label='theory')
	plt.yscale('log')
	plt.xlabel('Noll index')
	plt.ylabel('Variance (rad^2)')
	plt.legend()
	plt.show()
	'''
	assert np.all(np.abs(variances_simulated / variances_theory - 1) < 1)

def test_finite_atmosphere_total_variance():
	check_total_variance(0.5e-6, 1, 0.1, 10, False)

@pytest.mark.slow
def test_finite_atmosphere_total_variance_in_depth():
	# Check some selected parameter sets.
	check_total_variance(2e-6, 1, 0.1, 10, False)
	check_total_variance(0.5e-6, 8, 0.1, 10, False)
	check_total_variance(0.5e-6, 1, 0.3, 10, False)
	check_total_variance(0.5e-6, 1, 0.1, 40, False)

def test_infinite_atmosphere_total_variance():
	check_total_variance(0.5e-6, 1, 0.1, 10, True)

@pytest.mark.slow
def test_infinite_atmosphere_total_variance_in_depth():
	# Check some selected parameter sets.
	check_total_variance(2e-6, 1, 0.1, 10, True)
	check_total_variance(0.5e-6, 8, 0.1, 10, True)
	check_total_variance(0.5e-6, 1, 0.3, 10, True)
	check_total_variance(0.5e-6, 1, 0.1, 40, True)

def test_finite_atmosphere_zernike_variances():
	check_zernike_variances(0.5e-6, 1, 0.1, 10, False)

@pytest.mark.slow
def test_finite_atmosphere_zernike_variances_in_depth():
	# Check some selected parameter sets.
	check_zernike_variances(2e-6, 1, 0.1, 10, False)
	check_zernike_variances(0.5e-6, 8, 0.1, 10, False)
	check_zernike_variances(0.5e-6, 1, 0.3, 10, False)
	check_zernike_variances(0.5e-6, 1, 0.1, 40, False)

def test_infinite_atmosphere_zernike_variances():
	check_zernike_variances(0.5e-6, 1, 0.1, 10, True)

@pytest.mark.slow
def test_infinite_atmosphere_zernike_variances_in_depth():
	# Check some selected parameter sets.
	check_zernike_variances(2e-6, 1, 0.1, 10, True)
	check_zernike_variances(0.5e-6, 8, 0.1, 10, True)
	check_zernike_variances(0.5e-6, 1, 0.3, 10, True)
	check_zernike_variances(0.5e-6, 1, 0.1, 40, True)
