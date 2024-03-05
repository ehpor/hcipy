import numpy as np
from scipy.fft import next_fast_len

from .fourier_transform import _get_float_and_complex_dtype
from .._math import fft as _fft_module

class ChirpZTransform:
    '''The Chirp Z-transform (CZT).

    This class evaluates:

    .. math:: X_k = \sum_{n=0}^{N-1} x(n) z_k^{-n}

    where :math:`N` is the number of points along the real space, :math:`M`
    is the number of points in the Z-plane. The Chirp Z-transform samples
    the Z-plane on equidistant points along spiral arcs, where the arcs are
    given by the starting point :math:`A` and the complex ratio between points
    :math:`W`:

    .. math:: z_k = A * W^{-k}, k=0, 1, \ldots, M - 1

    This implementation uses Bluestein's algorithm to compute the CZT using Fast
    Fourier Transforms.

    Parameters
    ----------
    n : integer
        The number of points in the real plane.
    m : integer
        The number of points in the Z-plane.
    w : scalar
        The complex ratio between points in the Z-plane. Note: ensure that its
        absolute value is close to one, so that w**(max(n, m)**2) does not
        overflow.
    a : scalar
        The starting point in the complex Z-plane. Note: ensure that its absolute
        value is close to one, so that a**max(n, m) does not overflow.
    '''
    def __init__(self, n, m, w, a):
        # Ensure that w and a are complex scalars.
        w = complex(w)
        a = complex(a)

        self.n = n
        self.m = m
        self.w = w
        self.a = a

        self._current_dtype = None

    def _compute_kernel_and_weights(self, dtype):
        _, complex_dtype = _get_float_and_complex_dtype(dtype)

        if complex_dtype != self._current_dtype:
            k = np.arange(max(self.m, self.n))
            self.k = k

            wk2 = self.w**(self.k**2 / 2)

            self.nfft = next_fast_len(self.n + self.m - 1)

            self._Awk2 = self.a**-k[:self.n] * wk2[:self.n]
            self._Fwk2 = _fft_module.fft(1 / np.hstack((wk2[self.n - 1:0:-1], wk2[:self.m])), self.nfft)
            self._wk2 = wk2[:self.m]
            self._yidx = slice(self.n - 1, self.n + self.m - 1)

            self._Awk2 = self._Awk2.astype(complex_dtype, copy=False)
            self._Fwk2 = self._Fwk2.astype(complex_dtype, copy=False)
            self._wk2 = self._wk2.astype(complex_dtype, copy=False)

            self._current_dtype = complex_dtype

    def __call__(self, x):
        '''Compute the Chirp Z-transform along the last axis.

        The input is expected to have `n` points along the last axis.
        This is assumed but not checked. The transform is multiplexed
        along all other axes.

        The return value has the same number of axes, with the last axis
        having `m` elements.

        Parameters
        ----------
        x : array_like
            The array to compute the chirp Z-transform for.

        Returns
        -------
        array_like
            The chirp Z-transformed array.
        '''
        self._compute_kernel_and_weights(x.dtype)

        # Perform the CZT.
        x = x * self._Awk2

        intermediate = _fft_module.fft(x, self.nfft)
        intermediate *= self._Fwk2
        res = _fft_module.ifft(intermediate)

        res = res[..., self._yidx] * self._wk2

        _, complex_dtype = _get_float_and_complex_dtype(x.dtype)
        return res.astype(complex_dtype, copy=False)
