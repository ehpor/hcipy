import numpy as np

import pytest

import hcipy


class FourierTransformElement(hcipy.OpticalElement):
    def __init__(self, input_grid, output_grid):
        self.transform = hcipy.make_fourier_transform(input_grid, output_grid)

    def forward(self, wavefront):
        return hcipy.Wavefront(self.transform.forward(wavefront.electric_field), wavefront.wavelength)


class RecordingElement(hcipy.OpticalElement):
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def forward(self, wavefront):
        self.log.append(self.name)
        return wavefront

    def backward(self, wavefront):
        self.log.append(self.name)
        return wavefront


def make_system():
    pupil_grid = hcipy.make_pupil_grid(64, 1)
    focal_grid = hcipy.make_focal_grid(64, 8, 1)
    system = hcipy.OpticalSystem([hcipy.FresnelPropagator(pupil_grid, 1), FourierTransformElement(pupil_grid, focal_grid)])
    return pupil_grid, system


def perturbed_input(aperture, mode, mode_type, coefficient, wavelength):
    k = 2 * np.pi / wavelength

    if mode_type == 'opd':
        return aperture * np.exp(1j * k * coefficient * mode)
    elif mode_type == 'phase':
        return aperture * np.exp(1j * coefficient * mode)
    elif mode_type == 'amplitude':
        return aperture * (1 + coefficient * mode)
    elif mode_type == 'field':
        return aperture + coefficient * mode
    else:
        raise ValueError('Unknown mode type.')


def central_difference_response(optical_system, aperture, mode, mode_type, wavelength, step):
    plus = perturbed_input(aperture, mode, mode_type, step, wavelength)
    minus = perturbed_input(aperture, mode, mode_type, -step, wavelength)

    response = optical_system(hcipy.Wavefront(plus, wavelength)).electric_field - optical_system(hcipy.Wavefront(minus, wavelength)).electric_field

    return response / (2 * step)


def make_modes(pupil_grid, aperture_field, mode_type):
    zernike_modes = hcipy.make_zernike_basis(4, 1, pupil_grid)

    if mode_type == 'field':
        return hcipy.ModeBasis([hcipy.Field(mode * aperture_field, pupil_grid) for mode in zernike_modes])
    else:
        return zernike_modes


@pytest.mark.parametrize('mode_type', ['opd', 'phase', 'amplitude', 'field'])
def test_static_field(mode_type):
    pupil_grid, system = make_system()
    aperture_field = hcipy.make_circular_aperture(1)(pupil_grid)

    modes = make_modes(pupil_grid, aperture_field, mode_type)

    model = hcipy.linearize_optical_system(system, aperture_field, modes, wavelength=1, mode_type=mode_type)

    E0 = system(hcipy.Wavefront(aperture_field, 1)).electric_field

    assert np.allclose(model.static_field, E0)


@pytest.mark.parametrize('mode_type', ['opd', 'phase', 'amplitude', 'field'])
def test_field_response(mode_type):
    pupil_grid, system = make_system()
    aperture_field = hcipy.make_circular_aperture(1)(pupil_grid)

    modes = make_modes(pupil_grid, aperture_field, mode_type)

    step = 1e-3
    if mode_type == 'opd':
        step = 1e-6

    model = hcipy.linearize_optical_system(system, aperture_field, modes, wavelength=1, mode_type=mode_type)

    G = model.field_response.transformation_matrix

    for j in range(len(modes)):
        G_fd = central_difference_response(system, aperture_field, modes[j], mode_type, 1, step)
        assert np.allclose(G[..., j], G_fd, rtol=1e-5, atol=1e-9 * np.abs(G[..., j]).max())


@pytest.mark.parametrize('mode_type', ['opd', 'phase', 'amplitude', 'field'])
def test_superposition(mode_type):
    rng = np.random.default_rng(0)

    pupil_grid, system = make_system()
    aperture = hcipy.make_circular_aperture(1)(pupil_grid)

    modes = make_modes(pupil_grid, aperture, mode_type)

    wavelength = 1
    coefficients = 1e-4 * rng.normal(size=len(modes))

    model = hcipy.linearize_optical_system(system, aperture, modes, wavelength=wavelength, mode_type=mode_type)

    field = aperture.copy()
    if mode_type == 'field':
        field = aperture + modes.linear_combination(coefficients)
    elif mode_type == 'opd':
        opd = modes.linear_combination(coefficients)
        field = aperture * np.exp(1j * 2 * np.pi / wavelength * opd)
    elif mode_type == 'phase':
        phase = modes.linear_combination(coefficients)
        field = aperture * np.exp(1j * phase)
    else:  # amplitude
        field = aperture + modes.linear_combination(coefficients)

    E = system(hcipy.Wavefront(field, wavelength)).electric_field
    E_lin = model(coefficients)

    assert np.allclose(E, E_lin, rtol=1e-4, atol=1e-10)


def test_chaining():
    pupil_grid, system = make_system()
    aperture_field = hcipy.make_circular_aperture(1)(pupil_grid)

    modes = hcipy.make_zernike_basis(4, 1, pupil_grid)

    tel_system, coro_system = system.optical_elements
    wavelength = 1

    tel = hcipy.linearize_optical_system(tel_system, aperture_field, modes, wavelength=wavelength)
    coro = hcipy.linearize_optical_system(coro_system, aperture=tel.static_field, modes=tel.field_response, mode_type='field', wavelength=wavelength)

    single = hcipy.linearize_optical_system(system, aperture_field, modes, wavelength=wavelength)

    assert np.allclose(coro.static_field, single.static_field, rtol=1e-8, atol=1e-12)
    assert np.allclose(coro.field_response.transformation_matrix, single.field_response.transformation_matrix, rtol=1e-8, atol=1e-12)


def test_unknown_mode_type():
    pupil_grid, system = make_system()
    aperture_field = hcipy.make_circular_aperture(1)(pupil_grid)
    modes = hcipy.make_zernike_basis(4, 1, pupil_grid)

    with pytest.raises(ValueError):
        hcipy.linearize_optical_system(system, aperture_field, modes, mode_type='bogus')


def test_bookkeeping():
    pupil_grid, system = make_system()
    aperture_field = hcipy.make_circular_aperture(1)(pupil_grid)
    modes = hcipy.make_zernike_basis(4, 1, pupil_grid)

    model = hcipy.linearize_optical_system(system, aperture_field, modes, wavelength=2, mode_type='phase')

    assert model.mode_type == 'phase'
    assert model.wavelength == 2
    assert model.modes is modes


def make_elements():
    pupil_grid = hcipy.make_pupil_grid(32, 1)
    apodizer = hcipy.Apodizer(hcipy.make_circular_aperture(1)(pupil_grid))
    phase_apodizer = hcipy.PhaseApodizer(hcipy.zernike(2, 0)(pupil_grid))
    return pupil_grid, apodizer, phase_apodizer


def make_recording_elements(*names):
    log = []
    return [RecordingElement(name, log) for name in names], log


def make_wavefront():
    return hcipy.Wavefront(hcipy.make_pupil_grid(1, 1).ones(), 1)


def test_matmul_composition():
    wf = make_wavefront()

    (a, b), log = make_recording_elements('a', 'b')
    system = a @ b
    assert system.optical_elements == [b, a]
    system(wf)
    assert log == ['b', 'a']

    (a, b, c), log = make_recording_elements('a', 'b', 'c')
    inner = hcipy.OpticalSystem([b, c])
    composed = a @ inner
    assert composed.optical_elements == [b, c, a]
    assert inner.optical_elements == [b, c]
    composed(wf)
    assert log == ['b', 'c', 'a']

    (a, b, c), log = make_recording_elements('a', 'b', 'c')
    inner = hcipy.OpticalSystem([a, b])
    composed = inner @ c
    assert composed.optical_elements == [c, a, b]
    assert inner.optical_elements == [a, b]
    composed(wf)
    assert log == ['c', 'a', 'b']

    (a, b, c, d), log = make_recording_elements('a', 'b', 'c', 'd')
    system1 = hcipy.OpticalSystem([a, b])
    system2 = hcipy.OpticalSystem([c, d])
    composed = system1 @ system2
    assert composed.optical_elements == [c, d, a, b]
    assert system1.optical_elements == [a, b]
    assert system2.optical_elements == [c, d]
    composed(wf)
    assert log == ['c', 'd', 'a', 'b']


def test_matmul_associativity():
    (a, b, c), log = make_recording_elements('a', 'b', 'c')

    left = (a @ b) @ c
    right = a @ (b @ c)

    assert left.optical_elements == right.optical_elements == [c, b, a]


def test_matmul_application_order():
    (a, b), log = make_recording_elements('a', 'b')
    system = a @ b

    system.forward(make_wavefront())
    assert log == ['b', 'a']

    log.clear()

    system.backward(make_wavefront())
    assert log == ['a', 'b']


def test_matmul_not_implemented():
    _, apodizer, _ = make_elements()

    with pytest.raises(TypeError):
        apodizer @ 5


def make_linear_model():
    pupil_grid = hcipy.make_pupil_grid(12, 1)
    aperture = hcipy.make_circular_aperture(1)(pupil_grid)
    system = hcipy.FresnelPropagator(pupil_grid, 1)
    modes = hcipy.make_zernike_basis(3, 1, pupil_grid)

    return hcipy.linearize_optical_system(system, aperture, modes, wavelength=1)


def test_linearized_model_call():
    model = make_linear_model()
    rng = np.random.default_rng(0)

    phi = rng.normal(size=3) * 1e-3

    E = model(phi)

    assert isinstance(E, hcipy.Field)
    assert E.grid == model.output_grid
    assert np.allclose(E, model.static_field + model.field_response.transformation_matrix @ phi)


def test_intensity_from_covariance():
    model = make_linear_model()
    rng = np.random.default_rng(1)

    r = model.modes.num_modes
    P = rng.normal(size=(r, r))
    P = P @ P.T + np.eye(r)

    I = model.intensity_from_covariance(P)

    assert isinstance(I, hcipy.Field)
    assert I.grid == model.output_grid

    # Brute force: diag(G P G^H) plus the static intensity.
    G = model.field_response.transformation_matrix
    expected = np.abs(model.static_field)**2 + np.diag(G @ P @ G.conj().T).real
    assert np.allclose(I, expected)

    # A diagonal covariance given as a vector must equal the full-matrix version.
    p = rng.uniform(0.5, 2, r)
    assert np.allclose(model.intensity_from_covariance(p), model.intensity_from_covariance(np.diag(p)))

    with pytest.raises(ValueError):
        model.intensity_from_covariance(np.zeros((r + 1, r)))


def test_intensity_from_covariance_monte_carlo():
    model = make_linear_model()
    rng = np.random.default_rng(2)

    r = model.modes.num_modes
    P = np.diag(rng.uniform(0.5, 2, r))

    I_true = model.intensity_from_covariance(P)

    samples = rng.multivariate_normal(np.zeros(r), P, size=20000)
    I_mc = np.mean([np.abs(model(phi))**2 for phi in samples], axis=0)

    assert np.allclose(I_mc, I_true, rtol=0.05, atol=0.01 * np.max(I_true))


def test_pastis_matrix():
    model = make_linear_model()
    rng = np.random.default_rng(3)

    G = model.field_response.transformation_matrix
    n_out = model.output_grid.size
    weights = hcipy.Field(rng.uniform(size=n_out), model.output_grid)

    M = model.pastis_matrix(weights)

    assert M.shape == (model.modes.num_modes, model.modes.num_modes)

    # Brute force: M = sum_i w_i Re(conj(G_i) G_i^T)
    M_brute = sum(weights[i] * np.outer(np.conj(G[i]), G[i]) for i in range(n_out)).real
    assert np.allclose(M, M_brute)

    # M is real and symmetric.
    assert np.allclose(M, M.T)

    # trace(M P) equals the weighted sum of the covariance contribution.
    r = model.modes.num_modes
    P = np.diag(rng.uniform(0.5, 2, r))
    variance = model.intensity_from_covariance(P) - np.abs(model.static_field)**2
    assert np.allclose(np.sum(weights * variance), np.trace(M @ P))
