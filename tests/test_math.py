import pytest
import hcipy
import numpy as np

@pytest.mark.parametrize('func', ['fftn', 'ifftn', 'rfftn', 'irfftn'])
@pytest.mark.parametrize('dtype', ['complex128', 'complex64'])
@pytest.mark.parametrize('method', ['numpy', 'scipy', 'fftw', 'mkl'])
def test_fft_acceleration(func, method, dtype):
    if method == 'fftw':
        pytest.importorskip('pyfftw')

    if method == 'mkl':
        pytest.importorskip('mkl_fft')

    N = 128

    x = np.random.randn(N, N).astype(dtype)

    if 'r' not in func and 'complex' in dtype:
        x = x + 1j * np.random.randn(N, N).astype(dtype)

    if 'r' in func:
        x = x.real

    numpy_func = getattr(np.fft, func)
    hcipy_func = getattr(hcipy._math.fft, func)

    y_numpy = numpy_func(x)
    y_method = hcipy_func(x, method=method)

    if dtype == 'complex64' or dtype == 'float32':
        rtol = 1e-4
    else:
        rtol = 1e-8

    assert np.allclose(y_numpy, y_method, rtol=rtol)
