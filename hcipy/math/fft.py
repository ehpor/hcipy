from ..config import Configuration

import functools
import numpy as np
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

@functools.wraps(np.fft.fftn)
def fftn(*args, method=None, **kwargs):
    if method is None:
        methods = Configuration().fourier.fft.method
    else:
        methods = [method]

    for method in methods:
        if method == 'mkl' and mkl_fft is not None:
            return mkl_fft._numpy_fft.fftn(*args, **kwargs)
        elif method == 'fftw' and pyfftw is not None:
            return pyfftw.interfaces.numpy_fft.fftn(*args, threads=_CPU_COUNT, **kwargs)
        elif method == 'numpy':
            return np.fft.fftn(*args, **kwargs)
    else:
        raise ValueError('No FFT method could be found.')

@functools.wraps(np.fft.ifftn)
def ifftn(*args, **kwargs):
    for method in Configuration().fourier.fft.method:
        if method == 'mkl' and mkl_fft is not None:
            return mkl_fft._numpy_fft.ifftn(*args, **kwargs)
        elif method == 'fftw' and pyfftw is not None:
            return pyfftw.interfaces.numpy_fft.ifftn(*args, threads=_CPU_COUNT, **kwargs)
        elif method == 'numpy':
            return np.fft.ifftn(*args, **kwargs)
    else:
        raise ValueError('No FFT method could be found.')
