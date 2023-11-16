from ..config import Configuration

import functools
import numpy as np
import scipy
import multiprocessing

_CPU_COUNT = multiprocessing.cpu_count()

try:
    import mkl_fft
except ImportError:
    mkl_fft = None

try:
    import pyfftw
except ImportError:
    pyfftw = None


def _make_func(func_name):
    if mkl_fft is not None:
        mkl_func = getattr(mkl_fft, func_name)

    if pyfftw is not None:
        pyfftw_func = getattr(pyfftw.interfaces.scipy_fft, func_name)

    scipy_func = getattr(scipy.fft, func_name)
    numpy_func = getattr(np.fft, func_name)

    def func(*args, overwrite_x=False, method=None, threads=None, **kwargs):
        if method is None:
            methods = Configuration().fourier.fft.method
        else:
            methods = [method]

        if threads is None:
            threads = _CPU_COUNT

        for method in methods:
            if method == 'mkl' and mkl_fft is not None:
                return mkl_func(*args, **kwargs, overwrite_x=overwrite_x)
            elif method == 'fftw' and pyfftw is not None:
                return pyfftw_func(*args, **kwargs, workers=threads, overwrite_x=overwrite_x)
            elif method == 'scipy':
                return scipy_func(*args, **kwargs, workers=threads, overwrite_x=overwrite_x)
            elif method == 'numpy':
                return numpy_func(*args, **kwargs)
        else:
            raise ValueError('No FFT method could be found.')

    return func

fft = _make_func('fft')
fft2 = _make_func('fft2')
fftn = _make_func('fftn')
ifft = _make_func('ifft')
ifft2 = _make_func('ifft2')
ifftn = _make_func('ifftn')
