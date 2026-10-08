"""Direct-quadrature and residency tests for backend MatrixFourierTransform fields."""

import copy

import numpy as np
import pytest

import hcipy as hp
from hcipy.field import NewStyleField
from hcipy._math.backends import to_numpy
from array_api_compat import device
from conftest import get_backend


@pytest.fixture(params=['numpy', 'cupy', 'array_api_strict'])
def backend(request):
    return get_backend(request.param)


def make_separated_grids(ndim):
    """Make small nonuniform grids with nonconstant quadrature weights."""
    if ndim == 1:
        input_coords = [[-0.31, -0.13, 0.02, 0.24, 0.51]]
        output_coords = [[-8.1, -2.7, 0.4, 3.2, 7.5, 11.0]]
    else:
        input_coords = [[-0.31, -0.10, 0.07, 0.36], [-0.22, -0.03, 0.29]]
        output_coords = [[-7.0, -1.2, 2.1, 6.8, 10.0], [-5.2, 0.3, 4.7, 8.4]]
    input_grid = hp.CartesianGrid(hp.SeparatedCoords(input_coords))
    output_grid = hp.CartesianGrid(hp.SeparatedCoords(output_coords))
    return input_grid, output_grid


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('ndim,regular', [(1, True), (1, False), (2, True), (2, False)])
def test_mft_direct_quadrature(backend, dtype, ndim, regular):
    if regular:
        dims_input = 5 if ndim == 1 else [4, 3]
        dims_output = 6 if ndim == 1 else [5, 4]
        input_grid = hp.make_uniform_grid(dims_input, 0.7).shifted(0.03)
        output_grid = hp.make_uniform_grid(dims_output, 17).shifted(-0.2)
    else:
        input_grid, output_grid = make_separated_grids(ndim)
    mft = hp.MatrixFourierTransform(input_grid, output_grid)
    rng = np.random.default_rng(19)
    original = rng.normal(size=input_grid.size)
    if dtype.startswith('complex'):
        original = original + 1j * rng.normal(size=original.shape)
    original = original.astype(dtype)
    data = backend.asarray(original)
    field = NewStyleField(data, input_grid)

    transformed = mft.forward(field)
    actual = to_numpy(transformed.data)
    input_points = np.asarray(input_grid.points)
    output_points = np.asarray(output_grid.points)
    kernel = np.exp(-1j * output_points @ input_points.T)
    expected = (original * np.asarray(input_grid.weights)) @ kernel.T
    tolerance = 3e-6 if dtype in ('float32', 'complex64') else 3e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert transformed.shape == (output_grid.size,)
    expected_dtype = backend.complex64 if dtype in ('float32', 'complex64') else backend.complex128
    assert transformed.dtype == expected_dtype
    assert transformed.grid is output_grid

    recovered = mft.backward(transformed)
    backward_reference = (actual * np.asarray(output_grid.weights) / (2 * np.pi) ** ndim) @ kernel.conj()
    np.testing.assert_allclose(to_numpy(recovered.data), backward_reference, rtol=tolerance, atol=tolerance)
    assert recovered.shape == (input_grid.size,)
    assert recovered.grid is input_grid

    # Inputs and earlier outputs must not alias reusable internal storage.
    saved = actual.copy()
    mft.forward(field)
    np.testing.assert_array_equal(to_numpy(data), original)
    np.testing.assert_array_equal(to_numpy(transformed.data), saved)


@pytest.mark.parametrize('precompute_matrices', [False, True])
@pytest.mark.parametrize('allocate_intermediate', [False, True])
def test_mft_backend_cache_flags(precompute_matrices, allocate_intermediate):
    cp = get_backend('cupy')
    input_grid = hp.make_uniform_grid([5, 4], [0.7, 0.5])
    output_grid = hp.make_uniform_grid([6, 3], [12, 9])
    mft = hp.MatrixFourierTransform(input_grid, output_grid, precompute_matrices, allocate_intermediate)
    field = NewStyleField(cp.asarray(np.arange(input_grid.size), dtype=cp.float64), input_grid)
    mft.forward(field)
    assert (mft._array_api_M1 is not None) == precompute_matrices
    assert (mft._array_api_M2 is not None) == precompute_matrices
    assert getattr(mft, 'intermediate_array', None) is None
    mft.forward(hp.Field(np.arange(input_grid.size), input_grid))
    assert (mft.intermediate_array is not None) == allocate_intermediate


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_mft_compatible_grid_inverse_and_parseval(backend, dtype):
    input_grid = hp.make_uniform_grid([5, 6], [0.7, 0.9]).shifted([0.02, -0.04])
    output_grid = hp.make_fft_grid(input_grid)
    mft = hp.MatrixFourierTransform(input_grid, output_grid)
    rng = np.random.default_rng(31)
    original = (rng.normal(size=input_grid.size) + 1j * rng.normal(size=input_grid.size)).astype(dtype)
    field = NewStyleField(backend.asarray(original), input_grid)
    transformed = mft.forward(field)
    recovered = mft.backward(transformed)
    actual = to_numpy(transformed.data)
    tolerance = 3e-5 if dtype == 'complex64' else 3e-12
    np.testing.assert_allclose(to_numpy(recovered.data), original, rtol=tolerance, atol=tolerance)
    power_input = np.sum(abs(original) ** 2) * input_grid.weights
    power_output = np.sum(abs(actual) ** 2) * output_grid.weights / (2 * np.pi) ** 2
    assert abs(power_output / power_input - 1) < tolerance


def test_warmed_mft_deepcopy(backend):
    input_grid = hp.make_uniform_grid([4, 3], [0.7, 0.5])
    output_grid = hp.make_uniform_grid([5, 4], [11, 8])
    mft = hp.MatrixFourierTransform(input_grid, output_grid)
    field = NewStyleField(backend.asarray(np.random.default_rng(8).normal(size=input_grid.size)), input_grid)
    expected = mft.forward(field)
    cloned = copy.deepcopy(mft)
    assert cloned._array_api_M1 is not mft._array_api_M1
    np.testing.assert_allclose(to_numpy(cloned.forward(field).data), to_numpy(expected.data), atol=1e-13)


def test_mft_resident_cache_switching_and_deepcopy(monkeypatch):
    cp = get_backend('cupy')
    input_grid, output_grid = make_separated_grids(2)
    mft = hp.MatrixFourierTransform(input_grid, output_grid)
    original = np.random.default_rng(7).normal(size=input_grid.size)
    field = NewStyleField(cp.asarray(original), input_grid)
    expected = mft.forward(field)
    matrices = (mft._array_api_M1, mft._array_api_M2, mft._array_api_weights_input, mft._array_api_weights_output)
    asarray = cp.asarray

    def no_host_upload(array, *args, **kwargs):
        assert not isinstance(array, np.ndarray)
        return asarray(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asarray', no_host_upload)
        result = mft.forward(field)
        recovered = mft.backward(result)
    assert all(
        old is new for old, new in zip(matrices, (mft._array_api_M1, mft._array_api_M2, mft._array_api_weights_input, mft._array_api_weights_output))
    )
    np.testing.assert_allclose(cp.asnumpy(result.data), cp.asnumpy(expected.data), atol=1e-13)

    cloned = copy.deepcopy(mft)
    assert cloned._array_api_M1 is not mft._array_api_M1
    np.testing.assert_allclose(cp.asnumpy(cloned.forward(field).data), cp.asnumpy(expected.data), atol=1e-13)

    # One instance can switch backend and precision, then return to legacy Fields.
    for xp, dtype in [(np, np.complex64), (cp, cp.complex64), (cp, cp.complex128)]:
        switched = mft.forward(NewStyleField(xp.asarray(original, dtype=dtype), input_grid))
        assert switched.dtype == dtype
        np.testing.assert_allclose(to_numpy(switched.data), cp.asnumpy(expected.data), rtol=2e-5, atol=1e-8)
    legacy = mft.forward(hp.Field(original, input_grid))
    np.testing.assert_allclose(legacy, cp.asnumpy(expected.data), atol=1e-13)
    np.testing.assert_allclose(cp.asnumpy(recovered.data), to_numpy(mft.backward(hp.Field(cp.asnumpy(result.data), output_grid))), atol=1e-12)


def test_mft_cupy_device_switching():
    cp = get_backend('cupy')
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip('Two CUDA devices are required for device-switch coverage.')
    input_grid = hp.make_uniform_grid([4, 3], [0.7, 0.5])
    output_grid = hp.make_uniform_grid([5, 4], [11, 8])
    original = np.random.default_rng(43).normal(size=input_grid.size)
    mft = hp.MatrixFourierTransform(input_grid, output_grid)

    results = []
    for device_id in [0, 1, 0]:
        with cp.cuda.Device(device_id):
            field = NewStyleField(cp.asarray(original), input_grid)
            result = mft.forward(field)
            assert device(result.data).id == device_id
            assert device(mft._array_api_M1).id == device_id
            results.append(cp.asnumpy(result.data))
    np.testing.assert_allclose(results[0], results[1], atol=1e-13)
    np.testing.assert_array_equal(results[0], results[2])
