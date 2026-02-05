import pytest
import hcipy
import numpy as np
from hcipy._math.random import RandomGenerator


def _parameters():
    for func in ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 'hfft', 'ihfft']:
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


def test_random_generator_distribution(xp):
    # Get samples from the Standard Normal distribution.
    rng = RandomGenerator(xp, seed=42)
    samples = rng.normal(size=(10000,))

    # Basic statistical tests for normal distribution
    assert np.isclose(float(xp.mean(samples)), 0.0, atol=0.05)
    assert np.isclose(float(xp.std(samples)), 1.0, atol=0.05)


def test_random_generator_poisson(xp):
    # Get samples from the Poisson distribution.
    rng = RandomGenerator(xp, seed=42)
    samples = rng.poisson(lam=2.0, size=(10000,))

    # Basic statistical tests for poisson distribution
    assert np.isclose(float(xp.mean(samples)), 2.0, atol=0.1)


def test_random_generator_gamma(xp):
    # Get samples from the Gamma distribution.
    rng = RandomGenerator(xp, seed=42)
    samples = rng.gamma(scale=2.0, shape_param=2.0, size=(10000,))

    # Basic statistical tests (mean should be shape * scale = 2 * 2 = 4)
    assert np.isclose(float(xp.mean(samples)), 4.0, atol=0.5)


def test_random_generator_reproducible(xp):
    # Create two RandomGenerator objects with same seed
    rng1 = RandomGenerator(xp, seed=123)
    rng2 = RandomGenerator(xp, seed=123)

    # Generate samples
    samples1 = rng1.normal(size=(100,))
    samples2 = rng2.normal(size=(100,))

    assert np.allclose(samples1, samples2)


def test_random_generator_copy(xp):
    # Create initial rngs
    rng1 = RandomGenerator(xp, seed=42)
    rng2 = rng1.copy()

    # Generate some samples
    samples1 = rng1.normal(size=(10,))

    # Generate samples from copied rng
    samples2 = rng2.normal(size=(10,))

    # Should be identical
    assert np.allclose(samples1, samples2)


def test_random_generator_different_sizes(xp):
    rng = RandomGenerator(xp, seed=42)

    # Test scalar output
    arr_0d = rng.normal()
    assert arr_0d.shape == tuple()

    # Test 1D array
    arr_1d = rng.normal(size=5)
    assert arr_1d.shape == (5,)

    # Test 2D array
    arr_2d = rng.normal(size=(3, 4))
    assert arr_2d.shape == (3, 4)

    # Test 3D array
    arr_3d = rng.normal(size=(2, 3, 4))
    assert arr_3d.shape == (2, 3, 4)
