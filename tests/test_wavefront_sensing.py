from hcipy import *
import numpy as np

def test_optical_differentiation_wavefront_sensor():
    pupil_grid = make_pupil_grid(128, 1)
    wfs_grid = make_pupil_grid(256, 2)
    amplitude_filter = make_polarization_odwfs_amplitude_filter(0.0)

    odwfs = OpticalDifferentiationWavefrontSensorOptics(amplitude_filter, pupil_grid, wfs_grid)

    zernike_modes = make_zernike_basis(20, 1, pupil_grid)
    aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

    wf = Wavefront(make_circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
    odwfs(wf).intensity

def test_pyramid_wavefront_sensor():
    pupil_grid = make_pupil_grid(128)
    wfs_grid = make_pupil_grid(256, 2)

    pywfs = PyramidWavefrontSensorOptics(pupil_grid, wfs_grid, 1.0)

    zernike_modes = make_zernike_basis(20, 1, pupil_grid)
    aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

    wf = Wavefront(make_circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
    pywfs(wf).intensity

def test_modulated_pyramid_wavefront_sensor():
    pupil_grid = make_pupil_grid(128)
    wfs_grid = make_pupil_grid(256, 2)

    num_steps = 12

    pywfs = PyramidWavefrontSensorOptics(pupil_grid, wfs_grid, 1.0, q=3)
    mpywfs = ModulatedPyramidWavefrontSensorOptics(pywfs, 2, num_steps=num_steps)
    fast_mpywfs = ModulatedPyramidWavefrontSensorOptics(pywfs, 2, num_steps=num_steps, fast_modulation_method=True)

    zernike_modes = make_zernike_basis(20, 1, pupil_grid)
    aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

    wf = Wavefront(make_circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
    modulated_wfs = mpywfs(wf)

    assert len(modulated_wfs) == num_steps

    total_image = 0
    for wi in modulated_wfs:
        total_image += wi.power / num_steps

    total_image_fast = 0
    for wi in fast_mpywfs(wf):
        total_image_fast += wi.power / num_steps

    assert (abs(total_image - total_image_fast).max() / total_image.max()) < 2e-2

def test_zernike_wavefront_sensor():
    pupil_grid = make_pupil_grid(128)

    zwfs = ZernikeWavefrontSensorOptics(pupil_grid)

    zernike_modes = make_zernike_basis(20, 1, pupil_grid)
    aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

    wf = Wavefront(make_circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))
    zwfs(wf).intensity

def test_vector_zernike_wavefront_sensor():
    pupil_grid = make_pupil_grid(128)

    vzwfs = VectorZernikeWavefrontSensorOptics(pupil_grid)

    zwfs_plus = ZernikeWavefrontSensorOptics(pupil_grid, np.pi / 2)
    zwfs_neg = ZernikeWavefrontSensorOptics(pupil_grid, -np.pi / 2)

    zernike_modes = make_zernike_basis(20, 1, pupil_grid)
    aberration = zernike_modes.linear_combination(np.random.randn(20)) * 0.1

    wf = Wavefront(make_circular_aperture(1)(pupil_grid) * np.exp(1j * aberration))

    vector_img = vzwfs(wf).intensity
    ref_img = (zwfs_plus(wf).intensity + zwfs_neg(wf).intensity) / 2

    assert np.allclose(vector_img, ref_img)
