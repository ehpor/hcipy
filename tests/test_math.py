import pytest
import hcipy
import numpy as np
from hcipy._math.random import make_random_generator
from hcipy._math.stats import median, nanmedian
import math


def _parameters():
    for func in ['fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft', 'rfftn', 'irfftn', 'hfft', 'ihfft']:
        for dtype_in in ['float32', 'float64', 'complex64', 'complex128']:
            real_out = func.startswith('ir') or func.startswith('h')

            if dtype_in == 'float32' or dtype_in == 'complex64':
                dtype_out = 'float32' if real_out else 'complex64'
            else:
                dtype_out = 'float64' if real_out else 'complex128'

            if func.startswith('r') and dtype_in.startswith('complex'):
                continue

            if func.startswith('ih') and dtype_in.startswith('complex'):
                continue

            for method in ['numpy', 'scipy', 'fftw', 'mkl']:
                if ('r' in func or 'h' in func) and method == 'mkl':
                    # MKL doesn't implement rffts or hffts.
                    continue

                yield (func, method, dtype_in, dtype_out)

@pytest.mark.parametrize('func,method,dtype_in,dtype_out', list(_parameters()))
def test_fft_acceleration(func, method, dtype_in, dtype_out):
    rng = np.random.default_rng(seed=0)

    if method == 'fftw':
        pytest.importorskip('pyfftw')

    if method == 'mkl':
        pytest.importorskip('mkl_fft')

    for N in [32, 128, 1024]:
        x = rng.standard_normal((N, N)).astype(dtype_in)

        if dtype_in.startswith('complex'):
            x = x + 1j * rng.standard_normal((N, N)).astype(dtype_in)

        numpy_func = getattr(np.fft, func)
        hcipy_func = getattr(hcipy._math.fft, func)

        y_numpy = numpy_func(x).astype(dtype_out)
        y_method = hcipy_func(x, method=method)

        if dtype_out == 'complex64' or dtype_out == 'float32':
            rtol = 5e-4
        else:
            rtol = 1e-8

        assert np.allclose(y_numpy, y_method, rtol=rtol, atol=rtol)
        assert y_method.dtype == dtype_out

@pytest.mark.parametrize('mean, std', ((0.0, 1.0), (3.0, 2.0)))
def test_random_generator_distribution(xp, mean, std):
    # Get samples from the Standard Normal distribution.
    rng = make_random_generator(xp, seed=42)
    samples = rng.normal(mean, std, size=(10000,))

    # Basic statistical tests for normal distribution
    assert np.isclose(float(xp.mean(samples)), mean, atol=0.05)
    assert np.isclose(float(xp.std(samples)), std, atol=0.05)


@pytest.mark.parametrize('lam', (1.0, 3.0))
def test_random_generator_poisson(xp, lam):
    # Get samples from the Poisson distribution.
    rng = make_random_generator(xp, seed=42)
    samples = rng.poisson(lam=lam, size=(10000,))

    # Basic statistical tests for poisson distribution
    assert np.isclose(float(xp.mean(samples)), lam, atol=0.1)
    assert np.isclose(float(xp.std(samples)), math.sqrt(lam), atol=0.1)


@pytest.mark.parametrize('scale, shape', ((2.0, 2.0), (1.0, 3.0), (1.0, 0.5)))
def test_random_generator_gamma(xp, scale, shape):
    # Get samples from the Gamma distribution.
    rng = make_random_generator(xp, seed=42)
    samples = rng.gamma(scale=scale, shape_param=shape, size=(10000,))

    # Basic statistical tests (mean should be shape * scale)
    assert np.isclose(float(xp.mean(samples)), shape * scale, atol=0.1)
    assert np.isclose(float(xp.std(samples)), math.sqrt(shape) * scale, atol=0.1)


@pytest.mark.parametrize('distribution', ['normal', 'gamma', 'poisson'])
def test_random_generator_reproducible(xp, distribution):
    # Create two make_random_generator objects with same seed
    rng1 = make_random_generator(xp, seed=123)
    rng2 = make_random_generator(xp, seed=123)

    # Generate samples
    samples1 = getattr(rng1, distribution)(size=(100,))
    samples2 = getattr(rng2, distribution)(size=(100,))

    # Check that samples are identical
    assert np.allclose(samples1, samples2)


@pytest.mark.parametrize('distribution', ['normal', 'gamma', 'poisson'])
def test_random_generator_copy(xp, distribution):
    # Create initial rngs
    rng1 = make_random_generator(xp, seed=42)
    rng2 = rng1.copy()

    # Generate some samples
    samples1 = getattr(rng1, distribution)(size=(10,))
    samples2 = getattr(rng2, distribution)(size=(10,))

    # Should be identical
    assert np.allclose(samples1, samples2)


@pytest.mark.parametrize('distribution', ['normal', 'gamma', 'poisson'])
def test_random_generator_different_sizes(xp, distribution):
    rng = make_random_generator(xp, seed=42)

    generation_func = getattr(rng, distribution)

    # Test 1D array
    arr_1d = generation_func(size=5)
    assert arr_1d.shape == (5,)

    # Test 2D array
    arr_2d = generation_func(size=(3, 4))
    assert arr_2d.shape == (3, 4)

    # Test 3D array
    arr_3d = generation_func(size=(2, 3, 4))
    assert arr_3d.shape == (2, 3, 4)

_MEDIAN_PARAMS = [
    ((7,), None, False),
    ((6,), None, False),
    ((7,), None, True),
    ((3, 5), 0, False),
    ((3, 5), 1, False),
    ((3, 5), -1, False),
    ((2, 3, 4), (0, 2), False),
    ((2, 3, 4), None, False),
    ((2, 3, 4), 1, True),
    ((4, 6), (0, 1), True),
]

def _check_median(xp, shape, axis, keepdims, *, inject_nans=False):
    rng = np.random.default_rng(seed=0)
    x_np = rng.standard_normal(shape)

    if inject_nans and x_np.size > 0:
        nan_indices = rng.choice(x_np.size, size=max(1, x_np.size // 5), replace=False)
        x_np.flat[nan_indices] = np.nan

    x = xp.asarray(x_np)

    if inject_nans:
        result = nanmedian(x, axis=axis, keepdims=keepdims)
        expected = np.nanmedian(x_np, axis=axis, keepdims=keepdims)
    else:
        result = median(x, axis=axis, keepdims=keepdims)
        expected = np.median(x_np, axis=axis, keepdims=keepdims)

    result_np = np.asarray(result)
    assert result_np.shape == expected.shape
    assert np.allclose(result_np, expected, equal_nan=inject_nans)

@pytest.mark.parametrize('shape, axis, keepdims', _MEDIAN_PARAMS)
def test_median(xp, shape, axis, keepdims):
    _check_median(xp, shape, axis, keepdims)

@pytest.mark.parametrize('shape, axis, keepdims', _MEDIAN_PARAMS)
def test_nanmedian(xp, shape, axis, keepdims):
    _check_median(xp, shape, axis, keepdims, inject_nans=True)

def test_median_0d(xp):
    x_np = np.array(5.0)
    x = xp.asarray(x_np)
    result = median(x)
    assert np.allclose(np.asarray(result), x_np)

def test_median_int_dtype(xp):
    rng = np.random.default_rng(seed=1)
    x_np = rng.integers(0, 100, size=10, dtype=np.int32)
    x = xp.asarray(x_np)

    result = median(x)
    expected = np.median(x_np)

    assert np.allclose(np.asarray(result), expected)

def test_median_even_uses_mean(xp):
    """Regression test: even N must return the mean of the two middles."""
    x_np = np.array([1.0, 2.0, 3.0, 4.0])
    x = xp.asarray(x_np)
    result = median(x)
    assert np.allclose(np.asarray(result), 2.5)

def test_median_axis_empty_tuple(xp):
    """axis=() is a no-op; result should equal input."""
    rng = np.random.default_rng(seed=2)
    x_np = rng.standard_normal((3, 4))
    x = xp.asarray(x_np)
    result = median(x, axis=())
    assert np.allclose(np.asarray(result), x_np)

def test_nanmedian_all_nan_slice(xp):
    """A row/column that is all-NaN should return NaN, not error."""
    x_np = np.array([[1.0, np.nan, 3.0],
                     [np.nan, np.nan, np.nan],
                     [4.0, 5.0, 6.0]], dtype=np.float64)
    x = xp.asarray(x_np)

    result = nanmedian(x, axis=0)
    expected = np.nanmedian(x_np, axis=0)

    result_np = np.asarray(result)
    assert result_np.shape == expected.shape
    assert np.allclose(result_np, expected, equal_nan=True)
