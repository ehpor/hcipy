import pytest
import hcipy
import numpy as np
from hcipy._math.random import make_random_generator
from hcipy._math.stats import median, nanmedian
from hcipy._math.backends import to_numpy, array_namespace
from hcipy._math.einsum import einsum
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

@pytest.mark.parametrize('subscripts, shapes', [
    # basic contractions
    pytest.param("ij,jk->ik", [(4, 5), (5, 6)], id="matmul"),
    pytest.param("bij,bjk->bik", [(3, 4, 5), (3, 5, 6)], id="batched-matmul"),
    pytest.param("ii->", [(5, 5)], id="trace"),
    pytest.param("ii->i", [(5, 5)], id="diagonal"),
    pytest.param("ij->ji", [(4, 5)], id="transpose"),
    pytest.param("i,j->ij", [(4,), (5,)], id="outer-product"),
    pytest.param("i,i->", [(6,), (6,)], id="dot-product"),
    pytest.param("ij,ij->ij", [(4, 5), (4, 5)], id="hadamard"),
    pytest.param("abcd,cdef->abef", [(3, 4, 5, 6), (5, 6, 7, 8)], id="multi-index-contraction"),
    pytest.param("iij,jkk->ik", [(4, 4, 5), (5, 6, 6)], id="diagonals-plus-contraction"),
    # implicit output
    pytest.param("ij,jk", [(4, 5), (5, 6)], id="implicit-matmul"),
    pytest.param("i,j", [(4,), (5,)], id="implicit-outer"),
    pytest.param("ii", [(5, 5)], id="implicit-trace"),
    # scalar (0-d) operands
    pytest.param(",i->i", [(), (5,)], id="scalar-times-vector"),
    pytest.param(",ij->ij", [(), (4, 5)], id="scalar-times-matrix"),
    pytest.param("i,->i", [(5,), ()], id="vector-times-scalar"),
    pytest.param(",->", [(), ()], id="scalar-scalar"),
    # empty dimensions
    pytest.param("ij,jk->ik", [(0, 5), (5, 6)], id="empty-dimension"),
    pytest.param("ij,kj->ik", [(0, 5), (4, 5)], id="empty-result"),
    # multi-operand chains
    pytest.param("ij,jk,kl->il", [(2, 3), (3, 4), (4, 5)], id="three-operand-chain"),
    pytest.param("ij,jk,kl,lm->im", [(2, 3), (3, 4), (4, 5), (5, 6)], id="four-operand"),
    pytest.param("pi,qj,ijkl,rk,sl->pqrs", [(4, 4), (4, 4), (4, 4, 4, 4), (4, 4), (4, 4)], id="five-operand"),
    # single-operand einsum
    pytest.param("ij->", [(3, 4)], id="sum-all"),
    pytest.param("ij->i", [(3, 4)], id="sum-axis-0"),
    pytest.param("ij->j", [(3, 4)], id="sum-axis-1"),
    # diagonals within one operand
    pytest.param("iij->j", [(4, 4, 5)], id="trace-with-extra-axes"),
    pytest.param("iijj->", [(3, 3, 4, 4)], id="double-trace"),
    pytest.param("iii->i", [(3, 3, 3)], id="triple-diagonal"),
    pytest.param("ij,jkk->ik", [(4, 5), (5, 6, 6)], id="diagonal-on-right"),
    pytest.param("iij,jkk->ik", [(4, 4, 5), (5, 6, 6)], id="diagonals-on-both"),
    # broadcast / batch over several dims
    pytest.param("abij,abjk->abik", [(2, 3, 4, 5), (2, 3, 5, 6)], id="multi-batch-dims"),
    pytest.param("bi,cj->bcij", [(2, 3), (4, 5)], id="batch-outer"),
    # stress / high-dimensional
    pytest.param("abcdef,bcdefg->ag", [(2, 3, 4, 5, 2, 3), (3, 4, 5, 2, 3, 6)], id="6d-contraction"),
    pytest.param("abcij,abcjk->abcik", [(2, 3, 4, 5, 6), (2, 3, 4, 6, 7)], id="high-dim-batched"),
    pytest.param("ij,ij->", [(6, 7), (6, 7)], id="all-batch"),
    pytest.param("dc,ba->abcd", [(7, 6), (5, 4)], id="shuffled-index-order"),
    # edge cases
    pytest.param("ij,jk->ik", [(1, 1), (1, 1)], id="size-one-dims"),
    pytest.param("i->", [(1,)], id="single-element-tensor"),
    pytest.param("i,j,k->ijk", [(3,), (4,), (5,)], id="outer-product-3way"),
    pytest.param("bij,bjk->bik", [(20, 10, 12), (20, 12, 8)], id="large-batched"),
    pytest.param("ijk->kji", [(4, 5, 6)], id="chain-transpose"),
    # ellipsis
    pytest.param("...i,...i->...", [(3, 4), (4,)], id="ellipsis-matvec-broadcast"),
    pytest.param("...ij,...jk->...ik", [(2, 3, 4), (4, 5)], id="ellipsis-matmul-broadcast"),
    pytest.param("...ij,...ij->...", [(2, 3, 4), (3, 4)], id="ellipsis-common-batch"),
    pytest.param("i...i->...", [(3, 5, 3)], id="ellipsis-middle"),
    pytest.param("ij->...ij", [(4, 5)], id="out-only-ellipsis"),
    pytest.param("...i->...", [(3, 4)], id="ellipsis-reduce-index"),
    # implicit output with ellipsis
    pytest.param("...i", [(3, 4)], id="implicit-ellipsis"),
    pytest.param("...i,...i", [(3, 4), (4,)], id="implicit-ellipsis-broadcast"),
    pytest.param("i...i", [(3, 5, 3)], id="implicit-ellipsis-middle"),
])
@pytest.mark.parametrize('optimize', [False, True])
def test_einsum(xp, subscripts, shapes, optimize):
    rng = make_random_generator(xp)
    arrays = [xp.astype(rng.normal(size=s), xp.float64) for s in shapes]

    arrays_numpy = [np.asarray(arr) for arr in arrays]
    expected = np.einsum(subscripts, *arrays_numpy)

    got = einsum(subscripts, *arrays, optimize=optimize)

    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-11)

@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
def test_einsum_dtype(xp, dtype):
    dtype_xp = getattr(xp, dtype)
    rng = make_random_generator(xp)

    a = xp.astype(rng.normal(size=(4, 5)), dtype_xp)
    b = xp.astype(rng.normal(size=(5, 6)), dtype_xp)

    if xp.isdtype(dtype_xp, 'complex floating'):
        a = a + xp.asarray(1j, dtype=dtype_xp) * xp.astype(rng.normal(size=(4, 5)), dtype_xp)
        b = b + xp.asarray(1j, dtype=dtype_xp) * xp.astype(rng.normal(size=(5, 6)), dtype_xp)

    rtol = 1e-5 if xp.finfo(dtype_xp).bits <= 32 else 1e-11

    expected = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
    got = einsum("ij,jk->ik", a, b)

    assert got.dtype == dtype_xp
    np.testing.assert_allclose(np.asarray(got), expected, rtol=rtol)

def test_einsum_int_input(xp):
    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[5, 6], [7, 8]])

    expected = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
    got = einsum("ij,jk->ik", a, b)
    np.testing.assert_array_equal(np.asarray(got), expected)

def test_einsum_result_is_scalar(xp):
    result = einsum("i,i->", xp.asarray([1.0, 2.0]), xp.asarray([3.0, 4.0]))
    assert result.ndim == 0
