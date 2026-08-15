import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from array_api_compat import is_numpy_array
from .backends import array_namespace


# ---------------------------------------------------------------- kernels
@njit(cache=True)
def _filter_1d(A, f, out):
    for i in range(A.shape[0]):
        out[i] = A[i] * f[i]


@njit(cache=True, parallel=True)
def _filter_2d(A, f0, f1, out):
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

    lines = [f'def _filter_nd_{ndim}(A, {param_str}, out):']
    lines.append('    for i0 in prange(A.shape[0]):')
    for d in range(1, ndim):
        lines.append('    ' * (d + 1) + f'for i{d} in range(A.shape[{d}]):')
    lines.append('    ' * (ndim + 1) + f'out[{index_str}] = A[{index_str}] * {factor_str}')
    src = '\n'.join(lines)

    ns = {}
    exec(src, {'prange': prange}, ns)
    kernel = njit(cache=False)(ns[f'_filter_nd_{ndim}'])
    _KERNEL_CACHE[ndim] = kernel
    return kernel


# ---------------------------------------------------------------- helpers
def _factor_view(f, a, ndim):
    return f[(None,) * a + (slice(None),) + (None,) * (ndim - 1 - a)]


def _expand_factors(factors):
    '''Expand a tuple of 1-D factors to the full N-D filter array.'''
    result = _factor_view(factors[0], 0, len(factors))
    for a in range(1, len(factors)):
        result = result * _factor_view(factors[a], a, len(factors))
    return result


# ---------------------------------------------------------------- filter
@dataclass
class SeparableFilter:
    '''A separable multiplication filter, one 1-D factor per axis.

    Applying the filter to an array multiplies each element by the product
    of the factors along the corresponding axes. The filter is typically
    the product of a phase ramp and per-axis amplitude factors.

    The `threshold` controls how the filter is applied: for at most
    `threshold` total points, the factors are expanded to the full N-D
    filter array and a single multiplication is used; above it, a fused
    kernel applies the factors per axis in a single pass.

    Parameters
    ----------
    factors : tuple of ndarray
        One 1-D factor per axis of the array the filter is applied to, in
        the order of the array's axes.
    threshold : int, optional
        The maximum total number of points for which the full N-D filter
        array is used. By default 65536.

    Attributes
    ----------
    full
        The full N-D filter array, computed lazily and cached.
    '''
    factors: tuple = field(repr=False)
    threshold: int = 256 * 256

    _full: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._full = None

    @property
    def full(self):
        '''The full N-D filter array, computed lazily and cached.'''
        if self._full is None:
            self._full = _expand_factors(self.factors)
        return self._full

    @property
    def _use_full(self):
        return math.prod(f.shape[0] for f in self.factors) <= self.threshold

    def apply(self, arr, inverse=False):
        '''Multiply `arr` by the filter.

        Parameters
        ----------
        arr : ndarray
            The array to multiply. Its shape must match the lengths of the
            factors.
        inverse : bool, optional
            If True, divide by the filter (multiply by its reciprocal)
            instead of multiplying. By default False.

        Returns
        -------
        ndarray
            The filtered array.
        '''
        if is_numpy_array(arr):
            return self.apply_numpy(arr, inverse=inverse)
        return self._apply_generic(arr, inverse)

    def apply_numpy(self, arr, out=None, inverse=False):
        '''Multiply `arr` by the filter, numpy backend.

        Parameters
        ----------
        arr : ndarray
            The array to multiply. Its shape must match the lengths of the
            factors.
        out : ndarray, optional
            Buffer in which to place the result, of the same shape and
            dtype as `arr`. It is trusted as is; no checks are performed.
            If `out` is `arr`, the operation is performed in-place. If
            None (default), a new array is allocated and returned.
        inverse : bool, optional
            If True, divide by the filter (multiply by its reciprocal)
            instead of multiplying. By default False.

        Returns
        -------
        ndarray
            The filtered array, i.e. `out` if given.
        '''
        if self._use_full:
            full = self.full
            if out is None:
                return arr / full if inverse else arr * full
            if inverse:
                np.divide(arr, full, out=out)
            else:
                np.multiply(arr, full, out=out)
            return out

        if not is_numpy_array(self.factors[0]):
            result = arr
            for a, f in enumerate(self.factors):
                f_view = _factor_view(f, a, len(self.factors))
                result = result * f_view if not inverse else result / f_view
            if out is None:
                return result
            out[:] = result
            return out

        factors = self.factors
        if inverse:
            factors = tuple(1 / f for f in factors)
        ndim = arr.ndim

        if ndim == 1:
            kernel = _filter_1d
        elif ndim == 2:
            kernel = _filter_2d
        else:
            kernel = _kernel(ndim)

        if out is None:
            out = np.empty_like(arr)

        kernel(arr, *factors, out)
        return out

    def _apply_generic(self, arr, inverse):
        if self._use_full:
            full = self.full
            return arr / full if inverse else arr * full

        result = arr
        for a, f in enumerate(self.factors):
            f_view = _factor_view(f, a, len(self.factors))
            result = result * f_view if not inverse else result / f_view
        return result


# ---------------------------------------------------------------- factories
def make_phase_ramp(slopes, grid, threshold=256 * 256):
    '''A `SeparableFilter` implementing the phase ramp `exp(1j * sum_a (slope_a * x_a))`.

    Works with any array API compliant library (numpy, JAX, PyTorch, CuPy,
    ...). The filter is applied to arrays with shape `grid.shape`.

    Parameters
    ----------
    slopes : array_like
        The phase slopes, one scalar per axis, in the order (x, y, z, ...).
    grid : Grid
        The separated grid on which the phase ramp is evaluated. The
        coordinates per axis are taken from `grid.separated_coords`.
    threshold : int, optional
        See `SeparableFilter`. By default 65536.

    Returns
    -------
    SeparableFilter
        The phase ramp filter.

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
    return SeparableFilter(tuple(reversed(factors)), threshold)


def make_separable_filter(factors, threshold=256 * 256):
    '''A `SeparableFilter` from one 1-D factor array per axis.

    This can be used to build filters that are more complicated than a
    phase ramp, such as an apodization times a phase ramp.

    Parameters
    ----------
    factors : tuple of ndarray
        One 1-D factor per axis of the array the filter is applied to, in
        the order of the array's axes.
    threshold : int, optional
        See `SeparableFilter`. By default 65536.

    Returns
    -------
    SeparableFilter
        The filter.
    '''
    return SeparableFilter(tuple(factors), threshold)
