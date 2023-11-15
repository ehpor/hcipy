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

@functools.wraps(np.fft.fftn)
def fftn(*args, threads=None, axes=None, overwrite_x=False, method=None):
    if method is None:
        methods = Configuration().fourier.fft.method
    else:
        methods = [method]

    if threads is None:
        threads = _CPU_COUNT

    for method in methods:
        if method == 'mkl' and mkl_fft is not None:
            return mkl_fft.fftn(*args, axes=axes, overwrite_x=overwrite_x)
        elif method == 'fftw' and pyfftw is not None:
            return pyfftw.interfaces.scipy_fft.fftn(*args, axes=axes, workers=threads, overwrite_x=overwrite_x)
        elif method == 'scipy':
            return scipy.fft.fftn(*args, axes=axes, workers=threads, overwrite_x=overwrite_x)
        elif method == 'numpy':
            return np.fft.fftn(*args, axes=axes)
    else:
        raise ValueError('No FFT method could be found.')

@functools.wraps(np.fft.ifftn)
def ifftn(*args, threads=None, axes=None, overwrite_x=False, method=None):
    if method is None:
        methods = Configuration().fourier.fft.method
    else:
        methods = [method]

    if threads is None:
        threads = _CPU_COUNT

    for method in methods:
        if method == 'mkl' and mkl_fft is not None:
            return mkl_fft.ifftn(*args, axes=axes, overwrite_x=overwrite_x)
        elif method == 'fftw' and pyfftw is not None:
            return pyfftw.interfaces.scipy_fft.ifftn(*args, axes=axes, workers=threads, overwrite_x=overwrite_x)
        elif method == 'scipy':
            return scipy.fft.ifftn(*args, axes=axes, workers=threads, overwrite_x=overwrite_x)
        elif method == 'numpy':
            return np.fft.ifftn(*args, axes=axes)
    else:
        raise ValueError('No FFT method could be found.')
