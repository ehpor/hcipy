import array_api_strict as xp
import numpy as np
import pytest

import hcipy
from hcipy import CartesianGrid, Field, MatrixFourierTransform, SeparatedCoords, make_fft_grid, make_uniform_grid
from hcipy._math.backends import to_numpy

if not hcipy.Configuration().use_array_api:
    pytest.skip('Array API mode is required for these temporary tests.', allow_module_level=True)


_GRID_CASES = [
    pytest.param('regular', 8, 1.0, 1.0, id='1d-regular-even'),
    pytest.param('regular', 9, 1.3, 0.8, id='1d-regular-odd'),
    pytest.param('regular', [8, 8], 1.0, 1.0, id='2d-regular-even-even'),
    pytest.param('regular', [9, 9], 1.3, 0.8, id='2d-regular-odd-odd'),
    pytest.param('regular', [8, 9], 1.0, 1.0, id='2d-regular-even-odd'),
    pytest.param('regular', [9, 18], 1.3, 0.8, id='2d-regular-odd-even'),
    pytest.param('separated', 8, None, None, id='1d-separated-even'),
    pytest.param('separated', 9, None, None, id='1d-separated-odd'),
    pytest.param('separated', [8, 8], None, None, id='2d-separated-even-even'),
    pytest.param('separated', [9, 9], None, None, id='2d-separated-odd-odd'),
    pytest.param('separated', [8, 9], None, None, id='2d-separated-even-odd'),
    pytest.param('separated', [9, 18], None, None, id='2d-separated-odd-even'),
]


def _make_grids(kind, shape, q, fov, backend):
    if kind == 'regular':
        input_grid = make_uniform_grid(shape, 1, has_center=True, xp=backend)
        output_grid = make_fft_grid(input_grid, q=q, fov=fov)
    elif kind == 'separated':
        if isinstance(shape, int):
            x = np.linspace(-0.5, 0.5, shape)
            u = np.linspace(-1.0, 1.0, 2 * shape + 1)

            input_grid = CartesianGrid(SeparatedCoords([backend.asarray(x)]))
            output_grid = CartesianGrid(SeparatedCoords([backend.asarray(u)]))
        else:
            nx, ny = shape
            x = np.linspace(-0.5, 0.5, nx)
            y = np.linspace(-0.7, 0.7, ny)
            u = np.linspace(-1.0, 1.0, nx + 3)
            v = np.linspace(-0.9, 0.9, ny + 2)

            input_grid = CartesianGrid(SeparatedCoords([backend.asarray(x), backend.asarray(y)]))
            output_grid = CartesianGrid(SeparatedCoords([backend.asarray(u), backend.asarray(v)]))
    else:
        raise ValueError(kind)

    return input_grid, output_grid


def _make_data(rng, shape, dtype):
    if dtype.startswith('complex'):
        data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    else:
        data = rng.normal(size=shape)

    return data.astype(dtype)


@pytest.mark.parametrize('kind, shape, q, fov', _GRID_CASES)
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('tensor_shape', [(), (3,), (2, 3)])
@pytest.mark.parametrize('precompute_matrices', [True, False])
def test_mft_matches_numpy(kind, shape, q, fov, dtype, tensor_shape, precompute_matrices):
    rng = np.random.default_rng(seed=42)

    input_grid_np, output_grid_np = _make_grids(kind, shape, q, fov, np)
    input_grid_xp, output_grid_xp = _make_grids(kind, shape, q, fov, xp)

    data = _make_data(rng, tensor_shape + (input_grid_np.size,), dtype)

    field_np = Field(data, input_grid_np)
    field_xp = Field(xp.asarray(data, dtype=getattr(xp, dtype)), input_grid_xp)

    mft_np = MatrixFourierTransform(
        input_grid_np, output_grid_np,
        precompute_matrices=precompute_matrices,
    )
    mft_xp = MatrixFourierTransform(
        input_grid_xp, output_grid_xp,
        precompute_matrices=precompute_matrices,
    )

    result_np = mft_np.forward(field_np)
    result_xp = mft_xp.forward(field_xp)

    tol = 1e-5 if dtype in ('float32', 'complex64') else 1e-12
    complex_dtype = 'complex64' if dtype in ('float32', 'complex64') else 'complex128'

    assert result_xp.dtype == getattr(xp, complex_dtype)
    assert result_np.dtype == np.dtype(complex_dtype)
    np.testing.assert_allclose(to_numpy(result_xp.data), np.asarray(result_np.data), atol=tol, rtol=tol)

    back_np = mft_np.backward(result_np)
    back_xp = mft_xp.backward(result_xp)

    assert back_xp.dtype == getattr(xp, complex_dtype)
    assert back_np.dtype == np.dtype(complex_dtype)
    np.testing.assert_allclose(to_numpy(back_xp.data), np.asarray(back_np.data), atol=tol, rtol=tol)
