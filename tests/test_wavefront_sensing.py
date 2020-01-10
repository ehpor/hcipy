from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

def test_pyramid_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)

	pywfs = PyramidWavefrontSensorOptics(pupil_grid, 1.2)
	
	zernike_modes = make_zernike_basis(20, 1, pupil_grid)
	aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

	wf = Wavefront(circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
	img = pywfs(wf).intensity

def test_modulated_pyramid_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)

	pywfs = PyramidWavefrontSensorOptics(pupil_grid, 1.2)
	mpywfs = ModulatedPyramidWavefrontSensorOptics(pywfs, 2)
	
	zernike_modes = make_zernike_basis(20, 1, pupil_grid)
	aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

	wf = Wavefront(circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
	imgs = mpywfs(wf)
	img = np.sum([im.intensity for im in mpywfs(wf)], axis=0)
	img = Field(img, imgs[0].grid)

def test_zernike_wavefront_sensor():
	pupil_grid = make_pupil_grid(128)

	zwfs = ZernikeWavefrontSensorOptics(pupil_grid)

	zernike_modes = make_zernike_basis(20, 1, pupil_grid)
	aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

	wf = Wavefront(circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
	img = zwfs(wf).intensity

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
