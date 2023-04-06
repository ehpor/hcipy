import numpy as np
from scipy.fft import next_fast_len

try:
	import mkl_fft._numpy_fft as _fft_module
except ImportError:
	_fft_module = np.fft

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

		k = np.arange(max(m, n))
		self.k = k

		wk2 = w**(k**2 / 2)

		self.nfft = next_fast_len(n + m - 1)

		self._Awk2 = a**-k[:n] * wk2[:n]
		self._Fwk2 = _fft_module.fft(1 / np.hstack((wk2[n - 1:0:-1], wk2[:m])), self.nfft)
		self._wk2 = wk2[:m]
		self._yidx = slice(n - 1, n + m - 1)

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
		x = x * self._Awk2

		intermediate = _fft_module.fft(x, self.nfft)
		intermediate *= self._Fwk2
		res = _fft_module.ifft(intermediate)

		res = res[..., self._yidx] * self._wk2

		return res
