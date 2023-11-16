import pytest
import hcipy
import numpy as np

@pytest.mark.parametrize('dtype', ['complex128', 'complex64'])
@pytest.mark.parametrize('method', ['numpy', 'scipy', 'fftw', 'mkl'])
def test_fft_method(method, dtype):
    if method == 'fftw':
        pytest.importorskip('pyfftw')

    if method == 'mkl':
        pytest.importorskip('mkl_fft')

    N = 128

    x = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    x = x.astype(dtype)

    y_numpy = np.fft.fftn(x).astype(dtype)
    y_method = hcipy.fftn(x, method=method)

    if dtype == 'complex64':
        rtol = 1e-4
    else:
        rtol = 1e-8

    assert np.allclose(y_numpy, y_method, rtol=rtol)
