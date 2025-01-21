import pytest
import hcipy
import numpy as np


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
        hcipy_func = getattr(hcipy.math._implementation.fft, func)

        y_numpy = numpy_func(x).astype(dtype_out)
        y_method = hcipy_func(x, method=method)

        if dtype_out == 'complex64' or dtype_out == 'float32':
            rtol = 5e-4
        else:
            rtol = 1e-8

        assert np.allclose(y_numpy, y_method, rtol=rtol, atol=rtol)
        assert y_method.dtype == dtype_out

def test_backend_math():
    import time
    import hcipy.math.numpy as np

    a = np.eye(1024, dtype='complex64')

    print('Dot product:')
    b = np.dot(a, a)
    print(np.to_numpy(b).dtype)

    N = 10000
    start = time.perf_counter()
    for i in range(N):
        b = np.dot(a, a)
    np.to_numpy(b)
    end = time.perf_counter()
    print((end - start) / N * 1e6, 'us per iteration')

    print('FFT:')
    b = np.fft.fft2(a)
    print(np.to_numpy(b).dtype)

    N = 10000
    start = time.perf_counter()
    for i in range(N):
        b = np.fft.fft2(a)
    np.to_numpy(b)
    end = time.perf_counter()
    print((end - start) / N * 1e6, 'us per iteration')
