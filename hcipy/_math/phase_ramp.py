import math

import numpy as np
from numba import njit, prange
from array_api_compat import is_numpy_array
from .backends import array_namespace


@njit(cache=True)
def _ramp_1d(A, f, out):
    for i in range(A.shape[0]):
        out[i] = A[i] * f[i]


@njit(cache=True, parallel=True)
def _ramp_2d(A, f0, f1, out):
    for i in prange(A.shape[0]):
        for j in range(A.shape[1]):
            out[i, j] = A[i, j] * f0[i] * f1[j]


_KERNEL_CACHE = {}


def _kernel(ndim):
    '''Return the fused kernel for `ndim` axes, generating it on first use.

    Kernels for 1 and 2 dimensions are written by hand; higher-dimensional
    kernels are generated at runtime. Generated kernels cannot be cached on
    disk by numba (no source locator), so they are cached in memory here.
    '''
    kernel = _KERNEL_CACHE.get(ndim)
    if kernel is not None:
        return kernel

    param_str = ', '.join(f'f{d}' for d in range(ndim))
    index_str = ', '.join(f'i{d}' for d in range(ndim))
    factor_str = ' * '.join(f'f{d}[i{d}]' for d in range(ndim))

    lines = [f'def _ramp_nd_{ndim}(A, {param_str}, out):']
    lines.append('    for i0 in prange(A.shape[0]):')
    for d in range(1, ndim):
        lines.append('    ' * (d + 1) + f'for i{d} in range(A.shape[{d}]):')
    lines.append('    ' * (ndim + 1) + f'out[{index_str}] = A[{index_str}] * {factor_str}')
    src = '\n'.join(lines)

    ns = {}
    exec(src, {'prange': prange}, ns)
    kernel = njit(cache=False)(ns[f'_ramp_nd_{ndim}'])
    _KERNEL_CACHE[ndim] = kernel
    return kernel


def _factor_view(f, a, ndim):
    return f[(None,) * a + (slice(None),) + (None,) * (ndim - 1 - a)]


def prepare_phase_ramp(slopes, grid, threshold=256 * 256):
    '''Prepare the factors of a phase ramp for use with `apply_phase_ramp`.

    The phase ramp is `exp(1j * sum_a (k_a * x_a[i_a]))`, which is separable
    because the phase is a sum of one term per axis, each depending only on
    one coordinate. The ramp is applied to arrays with shape `grid.shape`.

    For small arrays, the full N-D phase array is precomputed and returned.
    For large arrays, the 1-D factors are returned as a tuple to save
    memory and improve performance.

    Parameters
    ----------
    slopes : array_like
        The phase slopes, one scalar per axis, in the order (x, y, z, ...).
    grid : Grid
        The separated grid on which the phase ramp is evaluated. The
        coordinates per axis are taken from `grid.separated_coords`.
    threshold : int, optional
        The maximum total number of points for which the full N-D phase
        array is precomputed. Above this, the tuple of 1-D factors is
        returned. By default 65536.

    Returns
    -------
    ndarray or tuple of ndarray
        If the total number of points is at most `threshold`: the
        full N-D phase array. Otherwise: a tuple of 1-D factors
        `exp(1j * k_a * x_a)`, one per axis of the shaped array.

    Raises
    ------
    ValueError
        If the grid is not separated.
    '''
    if not grid.is_separated:
        raise ValueError('The grid must be separated to prepare a phase ramp.')

    x_coords = grid.separated_coords

    xp = array_namespace(*x_coords)
    factors = [xp.exp(1j * (slope * x)) for slope, x in zip(slopes, x_coords)]

    # The shaped array on a separated grid has its axes reversed with
    # respect to `separated_coords`: axis 0 corresponds to the last
    # coordinate and the last axis to the first coordinate (x).
    if math.prod(x.shape[0] for x in x_coords) <= threshold:
        result = _factor_view(factors[-1], 0, len(factors))
        for a in range(1, len(factors)):
            result = result * _factor_view(factors[-1 - a], a, len(factors))
        return result
    else:
        return tuple(reversed(factors))


def apply_phase_ramp_numpy(arr, ramp, out=None):
    '''Apply a prepared phase ramp to an array, numpy backend.

    Returns `arr * exp(1j * sum_a (k_a * x_a[i_a]))`. If `ramp` is a tuple
    of factors, a fused numba kernel is used with a single pass; if it is
    the full N-D phase array, a single numpy multiplication is used.

    Parameters
    ----------
    arr : ndarray
        The complex array to multiply.
    ramp : ndarray or tuple of ndarray
        The output of `prepare_phase_ramp`: either the full N-D phase array
        or a tuple of 1-D factors.
    out : ndarray, optional
        Buffer in which to place the result, of the same shape and dtype as
        `arr`. It is trusted as is; no checks are performed. If `out` is
        `arr`, the operation is performed in-place.

    Returns
    -------
    ndarray
        The array multiplied by the phase ramp.
    '''
    if not is_numpy_array(arr):
        raise TypeError('apply_phase_ramp_numpy only accepts numpy arrays')

    if isinstance(ramp, tuple):
        if len(ramp) == 0 or not is_numpy_array(ramp[0]):
            result = arr
            for a, f in enumerate(ramp):
                result = result * _factor_view(f, a, len(ramp))
            if out is None:
                return result
            out[:] = result
            return out

        factors = ramp
        ndim = arr.ndim

        if ndim == 1:
            kernel = _ramp_1d
        elif ndim == 2:
            kernel = _ramp_2d
        else:
            kernel = _kernel(ndim)

        if out is None:
            out = np.empty_like(arr)

        kernel(arr, *factors, out)
        return out
    else:
        if not is_numpy_array(ramp):
            result = arr * ramp
            if out is None:
                return result
            out[:] = result
            return out

        if out is None:
            return arr * ramp

        np.multiply(arr, ramp, out=out)
        return out


def apply_phase_ramp(arr, ramp):
    '''Apply a prepared phase ramp to an array.

    Returns `arr * exp(1j * sum_a (k_a * x_a[i_a]))`.

    Works with any array API compliant library (numpy, JAX, PyTorch, CuPy,
    ...). For numpy arrays this is equivalent to
    `apply_phase_ramp_numpy` (which supports the `out` parameter).

    Parameters
    ----------
    arr : ndarray
        The complex array to multiply.
    ramp : ndarray or tuple of ndarray
        The output of `prepare_phase_ramp`: either the full N-D phase array
        or a tuple of 1-D factors.

    Returns
    -------
    ndarray
        The array multiplied by the phase ramp.
    '''
    if is_numpy_array(arr):
        return apply_phase_ramp_numpy(arr, ramp)

    if isinstance(ramp, tuple):
        result = arr
        for a, f in enumerate(ramp):
            result = result * _factor_view(f, a, len(ramp))
        return result

    return arr * ramp
