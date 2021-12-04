from hcipy import *
import numpy as np

def test_optical_differentiation_wavefront_sensor():
	pupil_grid = make_pupil_grid(128, 1)
	wfs_grid = make_pupil_grid(256, 2)
	amplitude_filter = make_polarization_odwfs_amplitude_filter(0.0)

	odwfs = OpticalDifferentiationWavefrontSensorOptics(amplitude_filter, pupil_grid, wfs_grid)

	zernike_modes = make_zernike_basis(20, 1, pupil_grid)
	aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

	wf = Wavefront(circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
	odwfs(wf).intensity

def test_pyramid_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)
	wfs_grid = make_pupil_grid(256, 2)

	pywfs = PyramidWavefrontSensorOptics(pupil_grid, wfs_grid, 1.0)

	zernike_modes = make_zernike_basis(20, 1, pupil_grid)
	aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

	wf = Wavefront(circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
	pywfs(wf).intensity

def test_modulated_pyramid_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)
	wfs_grid = make_pupil_grid(256, 2)

	num_steps = 12

	pywfs = PyramidWavefrontSensorOptics(pupil_grid, wfs_grid, 1.0)
	mpywfs = ModulatedPyramidWavefrontSensorOptics(pywfs, 2, num_steps=num_steps)

	zernike_modes = make_zernike_basis(20, 1, pupil_grid)
	aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

	wf = Wavefront(circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
	imgs = mpywfs(wf)

	assert len(imgs) == num_steps

def test_zernike_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)

	zwfs = ZernikeWavefrontSensorOptics(pupil_grid)

	zernike_modes = make_zernike_basis(20, 1, pupil_grid)
	aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

	wf = Wavefront(circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
	zwfs(wf).intensity

def test_vector_zernike_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)

	vzwfs = VectorZernikeWavefrontSensorOptics(pupil_grid)

	zwfs_plus = ZernikeWavefrontSensorOptics(pupil_grid, np.pi / 2)
	zwfs_neg = ZernikeWavefrontSensorOptics(pupil_grid, -np.pi / 2)

	zernike_modes = make_zernike_basis(20, 1, pupil_grid)
	aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

	wf = Wavefront(circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))

	vector_img = vzwfs(wf).intensity
	ref_img = (zwfs_plus(wf).intensity + zwfs_neg(wf).intensity) / 2

	assert np.allclose(vector_img, ref_img)
