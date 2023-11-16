from ..config import Configuration
from .cpu import get_num_available_cores

import functools
import numpy as np
import scipy

_CPU_COUNT = get_num_available_cores()

try:
    import mkl_fft
except ImportError:
    mkl_fft = None

try:
    import pyfftw

    # Set default cache variables for PyFFTW.
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)
except ImportError:
    pyfftw = None

def _make_func(func_name):
    if mkl_fft is not None:
        mkl_func = getattr(mkl_fft, func_name)

    if pyfftw is not None:
        pyfftw_func = getattr(pyfftw.interfaces.scipy_fft, func_name)

    scipy_func = getattr(scipy.fft, func_name)
    numpy_func = getattr(np.fft, func_name)

    @functools.wraps(numpy_func)
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
