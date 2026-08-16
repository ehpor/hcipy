import pytest
import hcipy
import numpy as np
from hcipy._math.random import make_random_generator
from hcipy._math.stats import median, nanmedian
from hcipy._math.backends import to_numpy, array_namespace
from hcipy._math.subpixel_shift import separable_convolve, subpixel_shift, _row_pass, _col_pass
from hcipy._math import cpu
import math
from scipy.ndimage import affine_transform


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
    samples = xp.astype(samples, xp.float32)

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


@pytest.mark.parametrize('low, high', ((0.0, 1.0), (-2.0, 5.0)))
def test_random_generator_uniform(xp, low, high):
    # Get samples from the Uniform distribution.
    rng = make_random_generator(xp, seed=42)
    samples = rng.uniform(low, high, size=(10000,))

    # Basic statistical tests for uniform distribution
    mean = (low + high) / 2
    std = (high - low) / math.sqrt(12)
    assert np.isclose(float(xp.mean(samples)), mean, atol=0.05)
    assert np.isclose(float(xp.std(samples)), std, atol=0.05)


@pytest.mark.parametrize('scale', (1.0, 3.0))
def test_random_generator_exponential(xp, scale):
    # Get samples from the Exponential distribution.
    rng = make_random_generator(xp, seed=42)
    samples = rng.exponential(scale=scale, size=(10000,))

    # Basic statistical tests for exponential distribution
    assert np.isclose(float(xp.mean(samples)), scale, atol=0.1)
    assert np.isclose(float(xp.std(samples)), scale, atol=0.1)


@pytest.mark.parametrize('replace', (True, False))
def test_random_generator_choice(xp, replace):
    rng = make_random_generator(xp, seed=42)

    n = 100 if replace else 3

    # Sample from integer range
    samples = rng.choice(10, size=(n,), replace=replace)
    assert samples.shape == (n,)
    assert all(0 <= s < 10 for s in samples)

    # Sample from array
    a = xp.asarray([5, 10, 15, 20])
    samples = rng.choice(a, size=(n,), replace=replace)
    assert samples.shape == (n,)
    assert all(s in [5, 10, 15, 20] for s in np.asarray(samples))

    # Sample with weights (always with replacement)
    p = xp.asarray([0.1, 0.2, 0.3, 0.4])
    samples = rng.choice(a, size=(100,), replace=True, p=p)
    assert samples.shape == (100,)
    assert all(s in [5, 10, 15, 20] for s in np.asarray(samples))


@pytest.mark.parametrize('distribution, args', [
    ('normal', (dict(),)),
    ('gamma', (dict(),)),
    ('poisson', (dict(),)),
    ('uniform', (dict(),)),
    ('exponential', (dict(),)),
    ('choice', ({'a': 10, 'replace': True},)),
])
def test_random_generator_reproducible(xp, distribution, args):
    # Create two make_random_generator objects with same seed
    rng1 = make_random_generator(xp, seed=123)
    rng2 = make_random_generator(xp, seed=123)

    # Generate samples
    samples1 = getattr(rng1, distribution)(size=(100,), **args[0])
    samples2 = getattr(rng2, distribution)(size=(100,), **args[0])

    # Check that samples are identical
    assert np.allclose(samples1, samples2)


@pytest.mark.parametrize('distribution, args', [
    ('normal', (dict(),)),
    ('gamma', (dict(),)),
    ('poisson', (dict(),)),
    ('uniform', (dict(),)),
    ('exponential', (dict(),)),
    ('choice', ({'a': 10, 'replace': True},)),
])
def test_random_generator_copy(xp, distribution, args):
    # Create initial rngs
    rng1 = make_random_generator(xp, seed=42)
    rng2 = rng1.copy()

    # Generate some samples
    samples1 = getattr(rng1, distribution)(size=(10,), **args[0])
    samples2 = getattr(rng2, distribution)(size=(10,), **args[0])

    # Should be identical
    assert np.allclose(samples1, samples2)


@pytest.mark.parametrize('distribution, args', [
    ('normal', (dict(),)),
    ('gamma', (dict(),)),
    ('poisson', (dict(),)),
    ('uniform', (dict(),)),
    ('exponential', (dict(),)),
    ('choice', ({'a': 10, 'replace': True},)),
])
def test_random_generator_different_sizes(xp, distribution, args):
    rng = make_random_generator(xp, seed=42)

    generation_func = getattr(rng, distribution)

    # Test 1D array
    arr_1d = generation_func(size=5, **args[0])
    assert arr_1d.shape == (5,)

    # Test 2D array
    arr_2d = generation_func(size=(3, 4), **args[0])
    assert arr_2d.shape == (3, 4)

    # Test 3D array
    arr_3d = generation_func(size=(2, 3, 4), **args[0])
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

def test_to_numpy(xp):
    arr = xp.zeros(10)
    assert isinstance(to_numpy(arr), np.ndarray)

def test_array_namespace(xp):
    arr = xp.zeros(10)
    _ = array_namespace(arr)

def _naive_separable_convolve(img, kernel_row, kernel_col):
    """Per-pixel nearest-boundary reference implementation."""
    rx = kernel_row.shape[0] // 2
    ry = kernel_col.shape[0] // 2
    H, W = img.shape

    tmp = np.empty_like(img)
    for y in range(H):
        for x in range(W):
            acc = 0.0

            for k in range(-rx, rx + 1):
                xx = x + k
                if xx < 0:
                    xx = 0
                elif xx >= W:
                    xx = W - 1

                acc += img[y, xx] * kernel_row[k + rx]

            tmp[y, x] = acc

    out = np.empty_like(tmp)
    for y in range(H):
        for x in range(W):
            acc = 0.0

            for k in range(-ry, ry + 1):
                yy = y + k
                if yy < 0:
                    yy = 0
                elif yy >= H:
                    yy = H - 1

                acc += tmp[yy, x] * kernel_col[k + ry]

            out[y, x] = acc

    return out


def _random_kernel(rng, radius):
    n = 2 * radius + 1
    k = rng.normal(size=n)

    xp = array_namespace(k)
    return k / xp.sum(k)


@pytest.mark.parametrize('H,W,radius', [
    (32, 32, 3),
    (64, 64, 3),
    (128, 32, 5),
    (31, 31, 2),
    (33, 33, 2),
])
def test_separable_convolve(xp, H, W, radius):
    rng = make_random_generator(xp)

    img = rng.normal(size=(H, W))
    kx = _random_kernel(rng, radius)
    ky = _random_kernel(rng, radius)

    ref = _naive_separable_convolve(np.asarray(img), np.asarray(kx), np.asarray(ky))

    result = separable_convolve(img, kx, ky)
    result_np = np.asarray(result)

    tol = 1e-5 if result_np.dtype == np.float32 else 1e-12
    assert np.allclose(result_np, ref, atol=tol), f"max diff: {np.abs(result_np - ref).max()}"


@pytest.mark.parametrize('row_shift,col_shift', [
    (0.3, 0.3),
    (-0.4, 0.1),
    (0.0, -0.3),
    (-0.5, -0.5),
    (0.5, 0.5),
])
def test_subpixel_shift(xp, row_shift, col_shift):
    rng = make_random_generator(xp)

    img = rng.normal(size=(64, 64))

    result = np.asarray(subpixel_shift(img, row_shift, col_shift))

    expected = affine_transform(
        np.asarray(img),
        np.eye(2),
        offset=np.array([row_shift, col_shift]),
        order=5,
        prefilter=False,
        mode='nearest',
    )

    tol = 1e-5 if result.dtype == np.float32 else 1e-12
    assert np.allclose(result, expected, atol=tol)

def test_zero_kernel(xp):
    rng = make_random_generator(xp, seed=0)
    img = rng.normal(size=(32, 32))
    k = xp.asarray([0.0, 0.0, 1.0, 0.0, 0.0])

    result = separable_convolve(img, k, k)

    assert np.allclose(np.asarray(result), np.asarray(img))


def test_row_col_pass_compiled():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64))
    kernel = rng.standard_normal(5)
    kernel /= kernel.sum()
    r = len(kernel) // 2

    out_py = np.zeros_like(img)
    out_jit = np.ones_like(img)

    _row_pass.py_func(img, kernel, r, out_py)
    _row_pass(img, kernel, r, out_jit)

    assert np.allclose(out_py, out_jit)

    out_py = np.zeros_like(img)
    out_jit = np.ones_like(img)

    _col_pass.py_func(img, kernel, r, out_py)
    _col_pass(img, kernel, r, out_jit)

    assert np.allclose(out_py, out_jit)

@pytest.mark.parametrize(
    "func",
    [
        "_cgroup_v2_quota",
        "_cgroup_v1_quota",
        "_cgroup_quota_cpus",
    ],
)
def test_quota_functions_return_sensible_value(func):
    value = getattr(cpu, func)()
    assert value is None or value > 0

@pytest.mark.parametrize(
    "func",
    [
        "_blas_threadpool_count",
        "_blas_env_override_count",
        "_scheduler_env_override_count",
        "_linux_affinity_count",
        "_windows_affinity_count",
    ],
)
def test_count_functions_return_sensible_value(func):
    value = getattr(cpu, func)()
    assert value is None or value >= 1

def test_get_num_available_cores():
    assert cpu.get_num_available_cores() >= 1
