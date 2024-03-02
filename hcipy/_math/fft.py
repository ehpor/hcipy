from ..config import Configuration
from .cpu import get_num_available_cores

import functools
import numpy as np
import scipy
import warnings

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
        try:
            mkl_func = getattr(mkl_fft, func_name)
        except AttributeError:
            mkl_func = None

    if pyfftw is not None:
        pyfftw_func = getattr(pyfftw.interfaces.scipy_fft, func_name)

    scipy_func = getattr(scipy.fft, func_name)
    numpy_func = getattr(np.fft, func_name)

    @functools.wraps(numpy_func)
    def func(x, *args, overwrite_x=False, method=None, threads=None, **kwargs):
        if method is None:
            methods = Configuration().fourier.fft.method
        else:
            methods = [method]

        if threads is None:
            # Use a single thread for small FFTs by default. The exact cutoff is very
            # rough changes from computer to computer. We choose 256x256 as a general guide.
            if x.size < 256**2:
                threads_attempts = [1]
            else:
                threads_attempts = [_CPU_COUNT, 1]

        # Try multithreaded first, but fall back upon single-threaded if that doesn't work.
        for threads in threads_attempts:
            for method in methods:
                try:
                    if method == 'mkl' and mkl_fft is not None and mkl_func is not None:
                        return mkl_func(x, *args, **kwargs, overwrite_x=overwrite_x)
                    elif method == 'fftw' and pyfftw is not None:
                        return pyfftw_func(x, *args, **kwargs, workers=threads, overwrite_x=overwrite_x)
                    elif method == 'scipy':
                        return scipy_func(x, *args, **kwargs, workers=threads, overwrite_x=overwrite_x)
                    elif method == 'numpy':
                        # Numpy doesn't always retain the correct bit depth. Cast the results to the correct bit depth here.
                        dtype_in = x.dtype
                        real_out = func_name.startswith('ir') or func_name.startswith('h')

                        if dtype_in == 'float32' or dtype_in == 'complex64':
                            dtype_out = 'float32' if real_out else 'complex64'
                        else:
                            dtype_out = 'float64' if real_out else 'complex128'

                        return numpy_func(x, *args, **kwargs).astype(dtype_out, copy=False)
                except Exception as e:
                    warnings.warn(f'Method {method} raised an exception "{e}". Falling back to other methods.')
            else:
                warnings.warn(f'No suitable/working FFT method could be found using {threads} threads.')
        else:
            raise ValueError('No suitable/working FFT method could be found.')

    return func

# Make all individual functions using this function generator.
fft = _make_func('fft')
fft2 = _make_func('fft2')
fftn = _make_func('fftn')

ifft = _make_func('ifft')
ifft2 = _make_func('ifft2')
ifftn = _make_func('ifftn')

rfft = _make_func('rfft')
rfft2 = _make_func('rfft2')
rfftn = _make_func('rfftn')

irfft = _make_func('irfft')
irfft2 = _make_func('irfft2')
irfftn = _make_func('irfftn')

hfft = _make_func('hfft')

ihfft = _make_func('ihfft')

# For completeness, we include fftshift and fftfreq functions.
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

fftfreq = np.fft.fftfreq
rfftfreq = np.fft.rfftfreq
