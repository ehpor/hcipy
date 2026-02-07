from ..config import Configuration
from .cpu import get_num_available_cores

import numpy as np
import scipy
import warnings
import array_api_compat

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

def _is_numpy_array(x):
    """Check if the input is a NumPy array."""
    return x.__class__.__module__.startswith("numpy")

def _make_1d_fft(func_name):
    """Create a 1D FFT function with minimal runtime overhead.
    """
    mkl_func = None
    if mkl_fft is not None:
        try:
            mkl_func = getattr(mkl_fft, func_name)
        except AttributeError:
            pass

    if pyfftw is not None:
        fftw_func = getattr(pyfftw.interfaces.scipy_fft, func_name)

    scipy_func = getattr(scipy.fft, func_name)
    numpy_func = getattr(np.fft, func_name)

    real_out = func_name.startswith('ir') or func_name.startswith('h')

    def fft_1d(x, /, *, n=None, axis=-1, norm='backward', overwrite_x=False, method=None, threads=None):
        # If the array is not a Numpy array, defer to the native implementation.
        if not _is_numpy_array(x):
            xp = array_api_compat.array_namespace(x)
            return getattr(xp.fft, func_name)(x, n=n, axis=axis, norm=norm)

        methods = Configuration().fourier.fft.method if method is None else [method]
        threads_attempts = ([1] if x.size < 256**2 else [_CPU_COUNT, 1]) if threads is None else [threads]

        dtype_in = x.dtype
        if dtype_in == 'float32' or dtype_in == 'complex64':
            expected_dtype = np.dtype('float32' if real_out else 'complex64')
        else:
            expected_dtype = np.dtype('float64' if real_out else 'complex128')

        for threads in threads_attempts:
            for method in methods:
                try:
                    if method == 'mkl' and mkl_func is not None:
                        if overwrite_x and dtype_in == expected_dtype:
                            return mkl_func(x, n=n, axis=axis, norm=norm, out=x)
                        else:
                            return mkl_func(x, n=n, axis=axis, norm=norm)
                    elif method == 'fftw' and fftw_func is not None:
                        return fftw_func(x, n=n, axis=axis, norm=norm, workers=threads, overwrite_x=overwrite_x)
                    elif method == 'scipy':
                        return scipy_func(x, n=n, axis=axis, norm=norm, workers=threads, overwrite_x=overwrite_x)
                    elif method == 'numpy':
                        if overwrite_x and dtype_in == expected_dtype:
                            return numpy_func(x, n=n, axis=axis, norm=norm, out=x)
                        else:
                            return numpy_func(x, n=n, axis=axis, norm=norm).astype(expected_dtype, copy=False)
                except Exception as e:
                    warnings.warn(f'Method {method} raised an exception "{e}". Falling back to other methods.')
            else:
                warnings.warn(f'No suitable/working FFT method could be found using {threads} threads.')
        else:
            raise ValueError('No suitable/working FFT method could be found.')

    fft_1d.__name__ = func_name
    fft_1d.__qualname__ = func_name
    fft_1d.__doc__ = f"{func_name}(x, /, *, n=None, axis=-1, norm='backward', overwrite_x=False, method=None, threads=None)"

    return fft_1d

def _make_nd_fft(func_name):
    """Create an N-D FFT function with minimal runtime overhead.
    """
    mkl_func = None
    if mkl_fft is not None:
        try:
            mkl_func = getattr(mkl_fft, func_name)
        except AttributeError:
            pass

    if pyfftw is not None:
        fftw_func = getattr(pyfftw.interfaces.scipy_fft, func_name)

    scipy_func = getattr(scipy.fft, func_name)
    numpy_func = getattr(np.fft, func_name)

    real_out = func_name.startswith('ir') or func_name.startswith('h')

    def fft_nd(x, /, *, s=None, axes=None, norm='backward', overwrite_x=False, method=None, threads=None):
        # If the array is not a Numpy array, defer to the native implementation.
        if not _is_numpy_array(x):
            xp = array_api_compat.array_namespace(x)
            return getattr(xp.fft, func_name)(x, s=s, axes=axes, norm=norm)

        methods = Configuration().fourier.fft.method if method is None else [method]
        threads_attempts = ([1] if x.size < 256**2 else [_CPU_COUNT, 1]) if threads is None else [threads]

        dtype_in = x.dtype
        if dtype_in == 'float32' or dtype_in == 'complex64':
            expected_dtype = np.dtype('float32' if real_out else 'complex64')
        else:
            expected_dtype = np.dtype('float64' if real_out else 'complex128')

        for threads in threads_attempts:
            for method in methods:
                try:
                    if method == 'mkl' and mkl_func is not None:
                        if overwrite_x and dtype_in == expected_dtype:
                            return mkl_func(x, s=s, axes=axes, norm=norm, out=x)
                        else:
                            return mkl_func(x, s=s, axes=axes, norm=norm)
                    elif method == 'fftw' and fftw_func is not None:
                        return fftw_func(x, s=s, axes=axes, norm=norm, workers=threads, overwrite_x=overwrite_x)
                    elif method == 'scipy':
                        return scipy_func(x, s=s, axes=axes, norm=norm, workers=threads, overwrite_x=overwrite_x)
                    elif method == 'numpy':
                        if overwrite_x and dtype_in == expected_dtype:
                            return numpy_func(x, s=s, axes=axes, norm=norm, out=x)
                        else:
                            return numpy_func(x, s=s, axes=axes, norm=norm).astype(expected_dtype, copy=False)
                except Exception as e:
                    warnings.warn(f'Method {method} raised an exception "{e}". Falling back to other methods.')
            else:
                warnings.warn(f'No suitable/working FFT method could be found using {threads} threads.')
        else:
            raise ValueError('No suitable/working FFT method could be found.')

    fft_nd.__name__ = func_name
    fft_nd.__qualname__ = func_name
    fft_nd.__doc__ = f"{func_name}(x, /, *, s=None, axes=None, norm='backward', overwrite_x=False, method=None, threads=None)"

    return fft_nd

# Create all 1D FFT functions.
fft = _make_1d_fft('fft')
ifft = _make_1d_fft('ifft')
rfft = _make_1d_fft('rfft')
irfft = _make_1d_fft('irfft')
hfft = _make_1d_fft('hfft')
ihfft = _make_1d_fft('ihfft')

# Create all N-D FFT functions.
fftn = _make_nd_fft('fftn')
ifftn = _make_nd_fft('ifftn')
rfftn = _make_nd_fft('rfftn')
irfftn = _make_nd_fft('irfftn')

# Shift functions.
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
