from hcipy import *
import numpy as np
from math import *
import mpmath
import scipy

import pytest
from unittest.mock import patch

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
    A = 0.00969 * (2 * np.pi / wavelength)**2 * Cn_squared
    coeffs_all = (-1)**(n - m) * 2 * (2 * np.pi)**(11 / 3) * (n + 1) * A * R**(5 / 3) / (sqrt(np.pi) * np.sin(np.pi * (n + 1 / 6)))

    term11 = mpmath.hyper([n + (3 / 2), n + 2, n + 1], [n + (1 / 6), n + 2, n + 2, 2 * n + 3], (2 * np.pi * R * k0)**2)
    term12 = sqrt(np.pi) * (2 * np.pi * R * k0)**(2 * n - 5 / 3) * scipy.special.gamma(n + 1) / (2**(2 * n + 3) * scipy.special.gamma(11 / 6) * scipy.special.gamma(n + 1 / 6) * scipy.special.gamma(n + 2)**2)
    term21 = -1 * scipy.special.gamma(7 / 3) * scipy.special.gamma(17 / 6) / (2 * scipy.special.gamma(-n + 11 / 6) * scipy.special.gamma(17 / 6)**2 * scipy.special.gamma(n + 23 / 6))
    term22 = mpmath.hyper([11 / 6, 7 / 3, 17 / 6], [-n + 11 / 6, 17 / 6, 17 / 6, n + 23 / 6], (2 * np.pi * R * k0)**2)

    return coeffs_all * (term11 * term12 + term21 * term22)

@pytest.mark.parametrize('propagate_phase_screen', [
    pytest.param(True, id='infinite'),
    pytest.param(False, id='finite')
])
@pytest.mark.parametrize('wavelength,pupil_diameter,fried_parameter,outer_scale', [
    pytest.param(0.5e-6, 1, 0.1, 10, id='baseline'),
    pytest.param(2e-6, 1, 0.1, 10, id='longer_wavelength', marks=pytest.mark.slow),
    pytest.param(0.5e-6, 8, 0.1, 10, id='larger_pupil_diameter', marks=pytest.mark.slow),
    pytest.param(0.5e-6, 1, 0.3, 10, id='larger_fried_parameter', marks=pytest.mark.slow),
    pytest.param(0.5e-6, 1, 0.1, 40, id='larger_outer_diameter', marks=pytest.mark.slow)
])
def test_atmosphere_total_variance(wavelength, pupil_diameter, fried_parameter, outer_scale, propagate_phase_screen):
    velocity = 10.0  # meters/sec
    num_modes = 1000

    pupil_grid = make_pupil_grid(64, pupil_diameter)
    aperture = make_circular_aperture(pupil_diameter)(pupil_grid)

    Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, [velocity / np.sqrt(2), velocity / np.sqrt(2)], use_interpolation=False)

    num_iterations = 2000
    total_variance = []

    for it in range(num_iterations):
        layer.reset(make_independent_realization=True)
        if propagate_phase_screen:
            layer.t = np.sqrt(2) * pupil_diameter / velocity

        phase = layer.phase_for(wavelength)
        total_variance.append(np.var(phase[aperture > 0]))

    variance_measured = np.mean(total_variance)

    variance_theory = 0
    for i in range(num_modes):
        n, m = noll_to_zernike(i + 2)
        variance_theory += zernike_variance_von_karman(n, m, pupil_diameter / 2., 1. / outer_scale, layer.Cn_squared, wavelength)

    assert (variance_measured / variance_theory - 1) < 0.1

@pytest.mark.parametrize('propagate_phase_screen', [
    pytest.param(True, id='infinite'),
    pytest.param(False, id='finite')
])
@pytest.mark.parametrize('wavelength,pupil_diameter,fried_parameter,outer_scale', [
    pytest.param(0.5e-6, 1, 0.1, 10, id='baseline'),
    pytest.param(2e-6, 1, 0.1, 10, id='longer_wavelength', marks=pytest.mark.slow),
    pytest.param(0.5e-6, 8, 0.1, 10, id='larger_pupil_diameter', marks=pytest.mark.slow),
    pytest.param(0.5e-6, 1, 0.3, 10, id='larger_fried_parameter', marks=pytest.mark.slow),
    pytest.param(0.5e-6, 1, 0.1, 40, id='larger_outer_diameter', marks=pytest.mark.slow)
])
def test_atmosphere_zernike_variances(wavelength, pupil_diameter, fried_parameter, outer_scale, propagate_phase_screen):
    velocity = 10.0  # meters/sec
    num_modes = 50

    pupil_grid = make_pupil_grid(128, pupil_diameter)

    Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, [velocity / np.sqrt(2), velocity / np.sqrt(2)], use_interpolation=False)

    zernike_modes = make_zernike_basis(num_modes + 20, pupil_diameter, pupil_grid, starting_mode=2, radial_cutoff=False)

    weights = evaluate_supersampled(make_circular_aperture(pupil_diameter), pupil_grid, 32)
    zernike_modes = ModeBasis([z * np.sqrt(weights) for z in zernike_modes])

    transformation_matrix = zernike_modes.transformation_matrix
    projection_matrix = inverse_tikhonov(transformation_matrix, 1e-9)

    num_iterations = 300
    mode_coeffs = []

    for it in range(num_iterations):
        layer.reset(make_independent_realization=True)
        if propagate_phase_screen:
            layer.t = np.sqrt(2) * pupil_diameter / velocity

        phase = layer.phase_for(wavelength)
        coeffs = projection_matrix.dot(phase * np.sqrt(weights))[:num_modes]
        mode_coeffs.append(coeffs)

    variances_simulated = np.var(mode_coeffs, axis=0)

    variances_theory = []
    for j in range(num_modes):
        n, m = noll_to_zernike(j + 2)
        variances_theory.append(zernike_variance_von_karman(n, m, pupil_diameter / 2., 1. / outer_scale, layer.Cn_squared, wavelength))
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

@pytest.mark.parametrize('layer_cls', [InfiniteAtmosphericLayer, FiniteAtmosphericLayer])
def test_atmospheric_layer_reset(layer_cls):
    fried_parameter = 0.3  # meter
    wavelength = 1e-6  # meter
    velocity = 10.0  # meter / sec
    pupil_diameter = 8  # meter
    outer_scale = 20  # meter

    pupil_grid = make_pupil_grid(32, pupil_diameter)

    Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)
    layer = layer_cls(pupil_grid, Cn_squared, outer_scale, [velocity / np.sqrt(2), velocity / np.sqrt(2)])

    phase_1 = layer.phase_for(wavelength)
    layer.reset(make_independent_realization=False)
    phase_2 = layer.phase_for(wavelength)
    layer.reset(make_independent_realization=False)
    phase_3 = layer.phase_for(wavelength)
    layer.reset(make_independent_realization=True)
    phase_4 = layer.phase_for(wavelength)
    layer.reset(make_independent_realization=False)
    phase_5 = layer.phase_for(wavelength)
    layer.evolve_until(1)
    phase_6 = layer.phase_for(wavelength)
    layer.reset(make_independent_realization=False)
    layer.evolve_until(1)
    phase_7 = layer.phase_for(wavelength)

    assert np.allclose(phase_1, phase_2)
    assert np.allclose(phase_2, phase_3)
    assert not np.allclose(phase_3, phase_4)
    assert np.allclose(phase_4, phase_5)
    assert not np.allclose(phase_5, phase_6)
    assert np.allclose(phase_6, phase_7)

def make_modal_ao_layer(framerate):
    grid = make_pupil_grid(32)
    cn_squared = Cn_squared_from_fried_parameter(1)

    layer = InfiniteAtmosphericLayer(grid, cn_squared, 1, 1, 1)
    ao_layer = ModalAdaptiveOpticsLayer(layer, make_zernike_basis(1, 1, grid), lag=1, framerate=framerate)

    return layer, ao_layer

def test_modal_ao_evolve_until_none_framerate():
    layer, ao_layer = make_modal_ao_layer(framerate=None)

    with patch.object(ao_layer, '_reconstruct_wavefront', wraps=ao_layer._reconstruct_wavefront) as mock_reconstruct,  \
            patch.object(layer, 'evolve_until', wraps=layer.evolve_until) as mock_evolve:
        phase_before = ao_layer.phase_for(1)
        assert phase_before.grid == layer.input_grid

        ao_layer.evolve_until(10)

        phase_after = ao_layer.phase_for(1)
        assert phase_after.grid == layer.input_grid
        assert not np.allclose(phase_before, phase_after)

        # Should evolve the underlying layer to the final time
        assert mock_evolve.call_args == ((10,),)

        # Should reconstruct wavefront once
        assert mock_reconstruct.call_count == 1

def test_modal_ao_evolve_until_fixed_framerate():
    layer, ao_layer = make_modal_ao_layer(framerate=1)

    with patch.object(ao_layer, '_reconstruct_wavefront', wraps=ao_layer._reconstruct_wavefront) as mock_reconstruct, \
            patch.object(layer, 'evolve_until', wraps=layer.evolve_until) as mock_evolve:
        phase_before = ao_layer.phase_for(1)
        assert phase_before.grid == ao_layer.input_grid

        ao_layer.evolve_until(2.5)

        phase_after = ao_layer.phase_for(1)
        assert phase_after.grid == ao_layer.input_grid
        assert not np.allclose(phase_before, phase_after)

        # Should evolve the underlying layer to 1, 2, 2.5
        assert mock_evolve.call_args_list == [((1,),), ((2,),), ((2.5,),)]

        # Should reconstruct wavefront twice (at t=1 and t=2)
        assert mock_reconstruct.call_count == 2


def test_multi_layer_atmosphere():
    r0 = 0.1
    wavelength = 500e-9

    pupil_grid = make_pupil_grid(32, 1.5)
    layers = make_mauna_kea_atmospheric_layers(pupil_grid)

    atmospheric_model = MultiLayerAtmosphere(layers, scintillation=False)
    atmospheric_model.Cn_squared = Cn_squared_from_fried_parameter(r0, wavelength)
    atmospheric_model.reset()

    aperture = Field(np.exp(-(pupil_grid.as_('polar').r / 0.65)**20), pupil_grid)
    wf = Wavefront(aperture, wavelength)

    # Model with no scintillation should only not modify amplitude
    wf_out = atmospheric_model.forward(wf)
    assert np.allclose(wf_out.amplitude, aperture)

    # It only should modify phase
    electric_field_compare = aperture * np.exp(1j * atmospheric_model.phase_for(wavelength))
    electric_field_simulated = wf_out.electric_field.copy()

    electric_field_compare /= electric_field_compare.mean()
    electric_field_simulated /= electric_field_simulated.mean()
    assert np.allclose(electric_field_compare, electric_field_simulated)

    wf_in = atmospheric_model.backward(wf_out)
    assert np.allclose(wf_in.electric_field, aperture)

    # Scintillation should also modify amplitude of the wavefront
    atmospheric_model.scintillation = True
    wf_out = atmospheric_model.forward(wf)
    assert not np.allclose(wf_out.amplitude, aperture)

    # Trying to get the phase for a model with scintillation should throw an exception
    with pytest.raises(ValueError):
        atmospheric_model.phase_for(wavelength)

    # Evolving a wavefront should modify the underlying layers
    atmospheric_model.evolve_until(1)
    wf_out2 = atmospheric_model.forward(wf)
    assert not np.allclose(wf_out.electric_field, wf_out2.electric_field)

@pytest.mark.parametrize('site', ['mauna_kea', 'las_campanas', 'keck'])
def test_standard_atmosphere_factory(site):
    grid = make_pupil_grid(32, 1.5)

    # Test with different sites.
    atmosphere = make_standard_atmosphere(grid, site=site)
    assert isinstance(atmosphere, MultiLayerAtmosphere)
    assert len(atmosphere.layers) > 0

def test_standard_atmosphere_factory_unknown():
    grid = make_pupil_grid(32, 1.5)

    with pytest.raises(ValueError):
        make_standard_atmosphere(grid, site='unknown_site')

def test_fried_parameter_seeing():
    np.random.seed(0)

    for i in range(10):
        seeing = np.random.uniform(0.5, 2)  # arcsec
        wavelength = np.random.uniform(500e-9, 800e-9)

        fried_parameter = seeing_to_fried_parameter(seeing, wavelength)
        seeing_recovered = fried_parameter_to_seeing(fried_parameter, wavelength)

        assert np.allclose(seeing_recovered, seeing)

@pytest.mark.parametrize('L0', [10, 20])
@pytest.mark.parametrize('r0', [0.1, 0.2])
def test_phase_covariance_phase_structure_function_von_karman(r0, L0):
    grid = make_uniform_grid([256, 256], [1, 1], has_center=True)

    phase_structure_function = phase_structure_function_von_karman(r0, L0)(grid)
    phase_covariance = phase_covariance_von_karman(r0, L0)(grid)

    phase_structure_function_from_covariance = 2 * (phase_covariance.max() - phase_covariance)

    assert np.allclose(phase_structure_function_from_covariance, phase_structure_function)
