import numpy as np
import pytest

from hcipy import (
    Field, ModeBasis, PhotonicLantern, SingleModeFiberArray,
    SingleModeFiberInjection, Wavefront, make_gaussian_fiber_mode,
    make_pupil_grid,
)

def make_coupler(kind, grid, mode):
    if kind == 'injection':
        return SingleModeFiberInjection(grid, mode)
    if kind == 'array':
        return SingleModeFiberArray(grid, make_pupil_grid(1), mode)
    return PhotonicLantern(ModeBasis([mode(grid)]))

def polarized_wavefront(field, polarization, wavelength):
    if polarization == 'scalar':
        return Wavefront(field, wavelength)
    if polarization == 'jones':
        return Wavefront(Field(np.array([1, 1j])[:, None] * field / np.sqrt(2), field.grid), wavelength)
    jones_matrix = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    return Wavefront(Field(jones_matrix[..., None] * field, field.grid), wavelength, [1, 0.3, 0.2, 0.1])

@pytest.mark.parametrize('kind', ['injection', 'array', 'lantern'])
@pytest.mark.parametrize('polarization', ['scalar', 'jones', 'stokes'])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_coupling_preserves_mode_phase_and_polarization(kind, polarization, dtype):
    grid = make_pupil_grid(32, 4)

    def mode(grid):
        # A nonconstant phase also checks which factor is conjugated in the overlap.
        return (make_gaussian_fiber_mode(1)(grid) * np.exp(0.8j * grid.x)).astype(dtype)

    coupler = make_coupler(kind, grid, mode)
    normalized_mode = mode(grid)
    normalized_mode /= np.sqrt(np.sum(abs(normalized_mode)**2 * grid.weights))
    wavefront = polarized_wavefront((normalized_mode * np.exp(0.7j)).astype(dtype), polarization, 700e-9)
    original_field = wavefront.electric_field.copy()

    coupled = coupler.forward(wavefront)
    recovered = coupler.backward(coupled)

    assert coupled.electric_field.shape == original_field.shape[:-1] + (1,)
    np.testing.assert_allclose(recovered.electric_field, original_field, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(coupled.total_power, wavefront.total_power, rtol=2e-6)
    np.testing.assert_allclose(recovered.stokes_vector, wavefront.stokes_vector, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(wavefront.electric_field, original_field)
    assert coupled.wavelength == wavefront.wavelength == recovered.wavelength
    assert recovered.grid == grid
    if polarization == 'stokes':
        np.testing.assert_array_equal(coupled.input_stokes_vector, wavefront.input_stokes_vector)
        np.testing.assert_array_equal(recovered.input_stokes_vector, wavefront.input_stokes_vector)

@pytest.mark.parametrize('kind', ['injection', 'array', 'lantern'])
@pytest.mark.parametrize('polarization', ['scalar', 'jones', 'stokes'])
@pytest.mark.parametrize('nonuniform_weights', [False, True])
def test_coupling_is_linear_and_weighted_adjoint(kind, polarization, nonuniform_weights):
    grid = make_pupil_grid(8, 2)
    if nonuniform_weights:
        grid.weights = np.linspace(0.01, 0.08, grid.size)
    mode = make_gaussian_fiber_mode(1)
    coupler = make_coupler(kind, grid, mode)
    rng = np.random.default_rng(832)
    field = Field(rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size), grid)
    wavefront = polarized_wavefront(field, polarization, 600e-9)
    scale = 0.4 + 1.2j
    scaled = wavefront.copy()
    scaled.electric_field *= scale

    coupled = coupler.forward(wavefront)
    np.testing.assert_allclose(coupler.forward(scaled).electric_field, scale * coupled.electric_field, atol=1e-14)

    modal_field = Field(rng.normal(size=coupled.electric_field.shape) +
                        1j * rng.normal(size=coupled.electric_field.shape), coupled.grid)
    modal_wavefront = Wavefront(modal_field, wavefront.wavelength, wavefront.input_stokes_vector)
    reconstructed = coupler.backward(modal_wavefront)
    lhs = np.vdot(coupled.electric_field, modal_field)
    rhs = np.vdot(wavefront.electric_field * grid.weights, reconstructed.electric_field)
    np.testing.assert_allclose(lhs, rhs, atol=1e-14)

@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('polarization', ['scalar', 'jones', 'stokes'])
def test_lantern_recovers_complex_modal_superposition(sparse, polarization):
    grid = make_pupil_grid(8, 2)
    # Two disjoint modes are exactly orthonormal under the physical grid weights.
    modes = np.zeros((grid.size, 2), dtype=complex)
    modes[:grid.size // 2, 0] = 1
    modes[grid.size // 2:, 1] = 1j
    modes /= np.sqrt(np.sum(abs(modes)**2 * grid.weights, axis=0))
    basis = ModeBasis(modes, grid)
    if sparse:
        basis = basis.to_sparse()
    lantern = PhotonicLantern(basis, normalize_modes=False)
    amplitudes = np.array([0.6 + 0.2j, -0.3 + 0.5j])
    wavefront = polarized_wavefront(Field(modes @ amplitudes, grid), polarization, 800e-9)

    coupled = lantern.forward(wavefront)
    expected = polarized_wavefront(Field(amplitudes, lantern.output_grid), polarization, 800e-9)
    np.testing.assert_allclose(coupled.electric_field, expected.electric_field, atol=1e-14)
    np.testing.assert_allclose(coupled.total_power, wavefront.total_power, atol=1e-14)
    np.testing.assert_allclose(lantern.backward(coupled).electric_field, wavefront.electric_field, atol=1e-14)

@pytest.mark.parametrize('kind', ['injection', 'array', 'lantern'])
def test_coupling_preserves_interference_with_a_coherent_reference(kind):
    grid = make_pupil_grid(32, 4)
    mode = make_gaussian_fiber_mode(1)
    coupler = make_coupler(kind, grid, mode)
    wavefront = Wavefront(1j * mode(grid))
    wavefront.total_power = 1
    amplitude = coupler.forward(wavefront).electric_field[0]
    # Equal-phase unit-amplitude fields interfere constructively at the output.
    np.testing.assert_allclose(abs(amplitude + 1j)**2, 4, atol=1e-14)

@pytest.mark.parametrize('polarization', ['scalar', 'jones', 'stokes'])
def test_fiber_array_projects_each_polarization_onto_each_mode(polarization):
    grid = make_pupil_grid(16, 3)
    fiber_grid = make_pupil_grid([3, 1], [2, 1])
    array = SingleModeFiberArray(grid, fiber_grid, make_gaussian_fiber_mode(0.7))
    field = Field(np.exp(1j * grid.x) * (1 + 0.2j * grid.y), grid)
    wavefront = polarized_wavefront(field, polarization, 1e-6)
    expected = np.stack([
        np.sum(wavefront.electric_field * mode.conj() * grid.weights, axis=-1)
        for mode in array.fiber_modes
    ], axis=-1)

    output = array.forward(wavefront)
    np.testing.assert_allclose(output.electric_field, expected, atol=1e-14)
    expected_field = sum(expected[..., i, None] * mode for i, mode in enumerate(array.fiber_modes))
    np.testing.assert_allclose(array.backward(output).electric_field, expected_field, atol=1e-14)

@pytest.mark.parametrize('kind', ['injection', 'array', 'lantern'])
@pytest.mark.parametrize('amplitude', [1, 0.5])
def test_real_scalar_coupling_control(kind, amplitude):
    grid = make_pupil_grid(16, 4)
    mode = make_gaussian_fiber_mode(1)
    coupler = make_coupler(kind, grid, mode)
    wavefront = Wavefront(mode(grid))
    wavefront.total_power = amplitude**2

    output = coupler.forward(wavefront)
    np.testing.assert_allclose(output.electric_field, amplitude, atol=1e-14)
    np.testing.assert_allclose(output.total_power, amplitude**2, atol=1e-14)
