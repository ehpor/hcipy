import numpy as np

from .chirp_z_transform import ChirpZTransform
from .fourier_transform import FourierTransform, ComputationalComplexity, _get_float_and_complex_dtype
from ..field import Field

class ZoomFastFourierTransform(FourierTransform):
	'''A Zoom Fast Fourier transform (ZoomFFT) object.

	This Fourier transform is a specialization of the Chirp Z-transform. It requires
	both the input and output grid to be regularly spaced in Cartesian coordinates. However,
	contrary to the Fast Fourier Transform (FFT), the spacing can be arbitrary and a small
	region of Fourier space can be efficiently evaluated.

	The ZoomFFT is asymptotically faster than a Matrix Fourier Transform (MFT) in cases where
	both input and output grids are large, typically at 1k x 1k or bigger in each grid. It also
	supports arbitrary dimenionality of the input and output grids.

	Parameters
	----------
	input_grid : Grid
		The grid that is expected for the input field.
	output_grid : Grid
		The grid that is produced by the Fourier transform.

	Raises
	------
	ValueError
		If the input grid is not separated in Cartesian coordinates, if it's not one- or two-
		dimensional, or if the output grid has a different dimension than the input grid.
	'''
	def __init__(self, input_grid, output_grid):
		self.check_if_supported(input_grid, output_grid)

		self.input_grid = input_grid
		self.output_grid = output_grid

		self._current_dtype = None

	def _compute_shifts_and_weights(self, dtype):
		float_dtype, complex_dtype = _get_float_and_complex_dtype(dtype)

		if complex_dtype != self._current_dtype:
			w = np.exp(-1j * self.output_grid.delta * self.input_grid.delta)
			a = np.exp(1j * self.output_grid.zero * self.input_grid.delta)

			inv_w = np.exp(1j * self.input_grid.delta * self.output_grid.delta)
			inv_a = np.exp(-1j * self.input_grid.zero * self.output_grid.delta)

			self.czts = [ChirpZTransform(n, m, ww, aa) for n, m, ww, aa in zip(self.input_grid.dims, self.output_grid.dims, w, a)]
			self.inv_czts = [ChirpZTransform(n, m, ww, aa) for n, m, ww, aa in zip(self.output_grid.dims, self.input_grid.dims, inv_w, inv_a)]

			self.shifts = [np.exp(-1j * x * x0) for x, x0 in zip(self.output_grid.separated_coords, self.input_grid.zero)]
			self.inv_shifts = [np.exp(1j * x * x0) for x, x0 in zip(self.input_grid.separated_coords, self.output_grid.zero)]

			self.shifts = [s.astype(complex_dtype, copy=False) for s in self.shifts]
			self.inv_shifts = [s.astype(complex_dtype, copy=False) for s in self.inv_shifts]

			self.input_weights = self.input_grid.weights.astype(float_dtype)
			self.output_weights = (self.output_grid.weights / (2 * np.pi)**self.output_grid.ndim).astype(float_dtype)

			self._current_dtype = complex_dtype

	def forward(self, field):
		'''Returns the forward Fourier transform of the :class:`Field` field.

		Parameters
		----------
		field : Field
			The field to Fourier transform.

		Returns
		--------
		Field
			The Fourier transform of the field.
		'''
		self._compute_shifts_and_weights(field.dtype)

		f = (field * self.input_weights).shaped

		for i, (czt, shift) in enumerate(zip(self.czts, self.shifts)):
			f = np.moveaxis(f, -i, 0)
			f = czt(f) * shift
			f = np.moveaxis(f, -i, 0)

		shape = tuple(field.tensor_shape) + (-1,)

		return Field(f.reshape(shape), self.output_grid)

	def backward(self, field):
		'''Returns the inverse Fourier transform of the :class:`Field` field.

		Parameters
		----------
		field : Field
			The field to inverse Fourier transform.

		Returns
		--------
		Field
			The inverse Fourier transform of the field.
		'''
		self._compute_shifts_and_weights(field.dtype)

		f = (field * self.output_weights).shaped

		for i, (czt, shift) in enumerate(zip(self.inv_czts, self.inv_shifts)):
			f = np.moveaxis(f, -i, 0)
			f = czt(f) * shift
			f = np.moveaxis(f, -i, 0)

		shape = tuple(field.tensor_shape) + (-1,)

		return Field(f.reshape(shape), self.input_grid)

	@classmethod
	def check_if_supported(cls, input_grid, output_grid):
		'''Check if the specified grids are supported by the Zoom Fast Fourier transform.

		Parameters
		----------
		input_grid : Grid
			The grid that is expected for the input field.
		output_grid : Grid
			The grid that is produced by the Zoom Fast Fourier transform.

		Raises
		------
		ValueError
			If the grids are not supported. The message will indicate why
			the grids are not supported.
		'''
		if not input_grid.is_regular or not input_grid.is_('cartesian'):
			raise ValueError('The input grid should be regularly spaced in Cartesian coordinates.')

		if not output_grid.is_regular or not output_grid.is_('cartesian'):
			raise ValueError('The output grid should be regularly spaced in Cartesian coordinates.')

		if input_grid.ndim != output_grid.ndim:
			raise ValueError('The input_grid must have the same dimensions as the output_grid.')

	@classmethod
	def compute_complexity(cls, input_grid, output_grid):
		'''Compute the algorithmic complexity for the Zoom Fast Fourier transform.

		Parameters
		----------
		input_grid : Grid
			The grid that is expected for the input field.
		output_grid : Grid
			The grid that is produced by the Zoom Fast Fourier transform.

		Returns
		-------
		AlgorithmicComplexity
			The algorithmic complexity for this Fourier transform.

		Raises
		------
		ValueError
			If the grids are not supported. The message will indicate why
			the grids are not supported.
		'''
		cls.check_if_supported(input_grid, output_grid)

		num_complex_multiplications = 0
		num_complex_additions = 0

		for i, (in_len, out_len) in enumerate(zip(input_grid.shape, output_grid.shape)):
			czt_len = in_len + out_len
			size_per_czt = np.prod(output_grid.shape[:i]) * np.prod(input_grid.shape[i + 1:])

			size_before_czt = np.prod(output_grid.shape[:i]) * np.prod(input_grid.shape[i:])
			size_after_czt = np.prod(output_grid.shape[:i + 1]) * np.prod(input_grid.shape[i + 1:])

			print(i, f'{czt_len=}, {size_per_czt=}, {size_before_czt=}, {size_after_czt=}')


			# Multiplication 1
			num_complex_multiplications += size_before_czt

			# FFT
			num_complex_multiplications += 0.5 * czt_len * size_per_czt * np.log2(czt_len)
			num_complex_additions += czt_len * size_per_czt * np.log2(czt_len)

			# Multiplication 2
			num_complex_multiplications += czt_len * size_per_czt

			# Inverse FFT
			num_complex_multiplications += 0.5 * czt_len * size_per_czt * np.log2(czt_len)
			num_complex_additions += czt_len * size_per_czt * np.log2(czt_len)

			# Multiplication 3
			num_complex_multiplications += size_after_czt

			# Shift
			num_complex_multiplications += size_after_czt

		num_multiplications = 4 * num_complex_multiplications
		num_additions = 2 * num_complex_multiplications + 2 * num_complex_additions

		return ComputationalComplexity(
			num_multiplications=num_multiplications,
			num_additions=num_additions
		)
