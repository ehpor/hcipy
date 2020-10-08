import numpy as np

from .fast_fourier_transform import FastFourierTransform
from ..field import Field, TensorFlowField, field_dot, field_conjugate_transpose

class FourierFilter(object):
	'''A filter in the Fourier domain.

	The filtering is performed by Fast Fourier Transforms, but is quicker than
	the equivalent multiplication in the Fourier domain using the FastFourierTransform
	classes. It does this by avoiding redundant field multiplications that limit performance.

	Parameters
	----------
	input_grid : Grid
		The grid that is expected for the input field.
	transfer_function : Field generator or Field
		The transfer function to use for the filter. If this is a Field, the user is responsible
		for checking that its grid corresponds to the internal grid used by this filter. The Grid
		is not checked.
	q : scalar
		The amount of zeropadding to perform in the real domain. A value
		of 1 denotes no zeropadding. Zeropadding increases the resolution in the
		Fourier domain and therefore reduces aliasing/wrapping effects.
	'''
	def __init__(self, input_grid, transfer_function, q=1):
		fft = FastFourierTransform(input_grid, q)

		self.input_grid = input_grid
		self.internal_grid = fft.output_grid
		self.internal_shape = fft.output_grid.shape
		self.ndim = self.internal_grid.ndim
		self.cutout = fft.cutout_input
		self.shape_in = input_grid.shape

		self.transfer_function = transfer_function

		self._transfer_function = None
		self.internal_array = None
		self._cached_tf_transfer_function = None

	def _compute_functions(self, field):
		# Set the correct complex and real data type, based on the input data type.
		if field.dtype == np.dtype('float32') or field.dtype == np.dtype('complex64'):
			complex_dtype = 'complex64'
		else:
			complex_dtype = 'complex128'

		if self._transfer_function is None or self._transfer_function.dtype != complex_dtype:
			if hasattr(self.transfer_function, '__call__'):
				h = self.transfer_function(self.internal_grid)
			else:
				h = self.transfer_function.copy()

			h = np.fft.ifftshift(h.shaped, axes=tuple(range(-self.input_grid.ndim, 0)))
			self._transfer_function = h.astype(complex_dtype, copy=False)

		recompute_internal_array = self.internal_array is None
		recompute_internal_array = recompute_internal_array or (self.internal_array.ndim != (field.grid.ndim + field.tensor_order))
		recompute_internal_array = recompute_internal_array or (self.internal_array.dtype != complex_dtype)
		recompute_internal_array = recompute_internal_array or np.all(self.internal_array.shape[:field.tensor_order] == field.tensor_shape)

		if recompute_internal_array:
			self.internal_array = self.internal_grid.zeros(field.tensor_shape, complex_dtype).shaped

	def forward(self, field):
		'''Return the forward filtering of the input field.

		Parmeters
		---------
		field : Field
			The field to filter.

		Returns
		-------
		Field
			The filtered field.
		'''
		return self._operation(field, adjoint=False)

	def backward(self, field):
		'''Return the backward (adjoint) filtering of the input field.

		Parmeters
		---------
		field : Field
			The field to filter.

		Returns
		-------
		Field
			The adjoint filtered field.
		'''
		return self._operation(field, adjoint=True)

	def _operation(self, field, adjoint):
		'''The internal filtering operation.

		Parameters
		----------
		field : Field
			The input field.
		adjoint : boolean
			Whether to perform a forward or adjoint filter.

		Returns
		-------
		Field
			The filtered field.
		'''
		self._compute_functions(field)

		if field.backend == 'tensorflow':
			import tensorflow as tf
		elif field.backend not in ['numpy']:
			raise ValueError('Fields with this backend are not supported.')

		if field.backend == 'numpy':
			if self.cutout is None:
				f = field.shaped
			else:
				f = self.internal_array
				f[:] = 0
				c = tuple([slice(None)] * field.tensor_order) + self.cutout
				f[c] = field.shaped
		elif field.backend == 'tensorflow':
			if self.cutout is None:
				f = field.shaped.arr
			else:
				padding = np.array([np.array([self.cutout[i].start, self.internal_shape[i] - self.cutout[i].stop]) for i in range(self.ndim)])
				f = tf.pad(field.shaped.arr, padding)

		if field.backend == 'numpy':
			f = np.fft.fftn(f, axes=tuple(range(-self.input_grid.ndim, 0)))
		else:
			if self.ndim == 1:
				f = tf.signal.fft(f)
			elif self.ndim == 2:
				f = tf.signal.fft2d(f)
			elif self.ndim == 3:
				f = tf.signal.fft3d(f)
			else:
				raise ValueError('FourierFilters with >3 dimensions are not supported with this backend.')

		if (self._transfer_function.ndim - self.internal_grid.ndim) == 2:
			# The transfer function is a matrix field.
			if field.backend == 'numpy':
				s1 = f.shape[:-self.internal_grid.ndim] + (self.internal_grid.size,)
				s2 = self._transfer_function.shape[:-self.internal_grid.ndim] + (self.internal_grid.size,)

				f = Field(f.reshape(s1), self.internal_grid)
				h = Field(self._transfer_function.reshape(s2), self.internal_grid)

				if adjoint:
					h = field_conjugate_transpose(h)

				f = field_dot(h, f).shaped
			elif field.backend == 'tensorflow':
				h = tf.reshape(self._tf_transfer_function, s2)

				if adjoint:
					n = self._transfer_function.ndim
					perm = [1, 0] + list(range(2, n))
					h = tf.transpose(h, perm=perm, conjugate=True)

				if field.is_vector_field:
					f = tf.einsum('...i,...i->...', h, f)
				elif field.tensor_order == 2:
					f = tf.einsum('...ij,...jk->...ik', h, f)
				else:
					raise ValueError('Transfer function combination not implemented.')
		else:
			# The transfer function is a scalar field.
			if field.backend == 'numpy':
				if adjoint:
					h = self._transfer_function.conj()
				else:
					h = self._transfer_function
			elif field.backend == 'tensorflow':
				if adjoint:
					h = tf.conj(self._tf_transfer_function)
				else:
					h = self._tf_transfer_function

			f *= h

		s = tuple(f.shape)[:-self.internal_grid.ndim] + (-1,)

		if field.backend == 'numpy':
			f = np.fft.ifftn(f, axes=tuple(range(-self.input_grid.ndim, 0)))

			if self.cutout is None:
				res = f.reshape(s)
			else:
				res = f[c].reshape(s)

			return Field(res, self.input_grid)
		elif field.backend == 'tensorflow':
			if self.ndim == 1:
				f = tf.signal.ifft(f)
			elif self.ndim == 2:
				f = tf.signal.ifft2d(f)
			elif self.ndim == 3:
				f = tf.signal.ifft3d(f)

			if self.cutout is None:
				f = tf.reshape(f, s)
			else:
				f = tf.reshape(f[self.cutout], s)

			return TensorFlowField(f, self.input_grid)

	@property
	def _tf_transfer_function(self):
		import tensorflow as tf

		if self._cached_tf_transfer_function is None:
			self._cached_tf_transfer_function = tf.convert_to_tensor(self._transfer_function)
		return self._cached_tf_transfer_function
