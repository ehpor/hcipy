import numpy as np

from .fast_fourier_transform import FastFourierTransform, make_fft_grid, _numexpr_grid_shift
from ..field import Field, field_dot, field_conjugate_transpose, CartesianGrid, RegularCoords
from .._math import fft as _fft_module


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
		self.cutout = fft.cutout_input
		self.shape_in = input_grid.shape

		self.transfer_function = transfer_function

		self._transfer_function = None
		self.internal_array = None

	def _compute_functions(self, field):
		if self._transfer_function is None or self._transfer_function.dtype != field.dtype:
			if hasattr(self.transfer_function, '__call__'):
				tf = self.transfer_function(self.internal_grid)
			else:
				tf = self.transfer_function.copy()

			tf = np.fft.ifftshift(tf.shaped, axes=tuple(range(-self.input_grid.ndim, 0)))
			self._transfer_function = tf.astype(field.dtype, copy=False)

		recompute_internal_array = self.internal_array is None
		recompute_internal_array = recompute_internal_array or (self.internal_array.ndim != (field.grid.ndim + field.tensor_order))
		recompute_internal_array = recompute_internal_array or (self.internal_array.dtype != field.dtype)
		recompute_internal_array = recompute_internal_array or not np.array_equal(self.internal_array.shape[:field.tensor_order], field.tensor_shape)

		if recompute_internal_array:
			self.internal_array = self.internal_grid.zeros(field.tensor_shape, field.dtype).shaped

	def forward(self, field):
		'''Return the forward filtering of the input field.

		Parameters
		----------
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

		Parameters
		----------
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

		if self.cutout is None:
			f = field.shaped
		else:
			f = self.internal_array
			f[:] = 0
			c = tuple([slice(None)] * field.tensor_order) + self.cutout
			f[c] = field.shaped

		# Don't overwrite f if it shares memory with the input field.
		overwrite_x = self.cutout is not None
		axes = tuple(range(-self.input_grid.ndim, 0))

		f = _fft_module.fftn(f, axes=axes, overwrite_x=overwrite_x)

		if (self._transfer_function.ndim - self.internal_grid.ndim) == 2:
			# The transfer function is a matrix field.
			s1 = f.shape[:-self.internal_grid.ndim] + (self.internal_grid.size,)
			f = Field(f.reshape(s1), self.internal_grid)

			s2 = self._transfer_function.shape[:-self.internal_grid.ndim] + (self.internal_grid.size,)
			tf = Field(self._transfer_function.reshape(s2), self.internal_grid)

			if adjoint:
				tf = field_conjugate_transpose(tf)

			f = field_dot(tf, f).shaped
		else:
			# The transfer function is a scalar field.
			if adjoint:
				tf = self._transfer_function.conj()
			else:
				tf = self._transfer_function

			# This is faster than f *= tf for Numpy due to it going back to C ordering.
			f = f * tf

		# Since f is now guaranteed to not share memory, always allow overwriting.
		f = _fft_module.ifftn(f, axes=axes, overwrite_x=True)

		s = f.shape[:-self.internal_grid.ndim] + (-1,)
		if self.cutout is None:
			res = f.reshape(s)
		else:
			res = f[c].reshape(s)

		return Field(res, self.input_grid)

class FourierShift:
	'''An image shifting operator implemented in the Fourier domain.

	This operator is smart enough to ignore dimensions where the
	shift is zero.

	Parameters
	----------
	input_grid : Grid
		The grid that is expected for the input field.
	shift : array_like
		The shift to apply to any input field.

	Attributes
	----------
	shift : array_like
		The shift to apply to any input field.
	'''
	def __init__(self, input_grid, shift):
		self.input_grid = input_grid
		self.output_grid = input_grid

		self.shift = shift

	@property
	def shift(self):
		return self._shift

	@shift.setter
	def shift(self, shift):
		shift = np.array(shift)
		self._shift = shift

		# Find out which axes to Fourier transform.
		mask = shift != 0
		self._ft_axes = tuple(np.flatnonzero(shift) - 1)

		if not self._ft_axes:
			# All shifts are zero. We can exit out early.
			self._shift_filter = None
			return

		broadcasting_slice = tuple(slice(None) if m else np.newaxis for m in mask[::-1])

		# Compute the Fourier transform grid for these axes.
		fft_grid = make_fft_grid(self.input_grid, q=1, fov=1)
		fft_grid = CartesianGrid(RegularCoords(fft_grid.delta[mask], fft_grid.dims[mask], fft_grid.zero[mask]))

		# Compute the shift filter on this grid and broadcast to the original field shape.
		shift_filter = _numexpr_grid_shift(-shift, fft_grid).reshape(fft_grid.shape)
		shift_filter = np.fft.ifftshift(shift_filter)
		shift_filter = shift_filter[broadcasting_slice]

		self._shift_filter = shift_filter

	def forward(self, field):
		'''Return the forward filtering of the input field.

		Parameters
		----------
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

		Parameters
		----------
		field : Field
			The field to filter.

		Returns
		-------
		Field
			The adjoint filtered field.
		'''
		return self._operation(field, adjoint=True)

	def _operation(self, field, adjoint):
		if self._shift_filter is None:
			return field.astype('complex')

		# Never overwrite the input, so don't use kwargs here.
		f = _fft_module.fftn(field.shaped, axes=self._ft_axes)

		if adjoint:
			f *= np.conj(self._shift_filter)
		else:
			f *= self._shift_filter

		# Fine to overwrite the input, if supported.
		f = _fft_module.ifftn(f, axes=self._ft_axes, overwrite_x=True)

		shape = f.shape[:-field.grid.ndim] + (-1,)

		return Field(f.reshape(shape), field.grid)

class FourierShear:
	'''An image shearing operator implemented in the Fourier domain.

	When given an image I(x, y), this operator will return a new
	image I(x + a * y, y) when a shearing along the x axis is
	requested.

	Parameters
	----------
	input_grid : Grid
		The grid that is expected for the input field.
	shear : scalar
		The amount of shear to apply to th image.
	shear_dim : integer
		The dimension to which to apply the shear. A shear along x would
		be a value of 0, a shear along y would be 1.

	Attributes
	----------
	input_grid : Grid
		The grid assumed for the input of this operator. Read-only.
	shear : scalar
		The amount of shear along the axis.
	shear_dim : integer
		The dimension along which the shear is applied. Read-only.

	Raises
	------
	ValueError
		When the input grid is not 2D and regularly spaced.
	'''
	def __init__(self, input_grid, shear, shear_dim=0):
		if not input_grid.is_regular or input_grid.ndim != 2:
			raise ValueError('The input grid should be 2D and regularly spaced.')

		self._input_grid = input_grid
		self._shear_dim = shear_dim
		self.shear = shear

	@property
	def input_grid(self):
		return self._input_grid

	@property
	def shear_dim(self):
		return self._shear_dim

	@property
	def fourier_dim(self):
		return 1 if self.shear_dim == 0 else 0

	@property
	def shear(self):
		return self._shear

	@shear.setter
	def shear(self, shear):
		fft_grid = make_fft_grid(self.input_grid)
		fx = np.fft.ifftshift(fft_grid.separated_coords[self.shear_dim])

		y = self.input_grid.separated_coords[self.fourier_dim]

		self._filter = np.exp(-1j * shear * np.outer(y, fx))

		if self.shear_dim == 1:
			self._filter = self._filter.T

		# Make sure the ordering of the filter is the same as the FFT output.
		self._filter = np.ascontiguousarray(self._filter)

		self._shear = shear

	def forward(self, field):
		'''Return the forward shear of the input field.

		Parameters
		----------
		field : Field
			The field to shear.

		Returns
		-------
		Field
			The sheared field.
		'''
		return self._operation(field, adjoint=False)

	def backward(self, field):
		'''Return the backward (adjoint) shear of the input field.

		Parameters
		----------
		field : Field
			The field to shear.

		Returns
		-------
		Field
			The adjoint sheared field.
		'''
		return self._operation(field, adjoint=True)

	def _operation(self, field, adjoint):
		# Never overwrite the input, so don't use kwargs here.
		f = _fft_module.fft(field.shaped, axis=-self.shear_dim - 1)

		if adjoint:
			f *= np.conj(self._filter)
		else:
			f *= self._filter

		# Fine to overwrite the input, if supported.
		f = _fft_module.ifft(f, axis=-self.shear_dim - 1, overwrite_x=True)

		shape = f.shape[:-field.grid.ndim] + (-1,)

		return Field(f.reshape(shape), field.grid)

class FourierRotation:
	'''An image rotation operator implemented in the Fourier domain.

	Parameters
	----------
	input_grid : Grid
		The grid that is expected for the input field.
	angle : scalar
		The rotation angle.

	Raises
	------
	ValueError
		When the input grid is not 2D and regularly spaced.
	'''
	def __init__(self, input_grid, angle):
		self._input_grid = input_grid

		if not input_grid.is_regular or input_grid.ndim != 2:
			raise ValueError('The input grid should be 2D and regularly spaced.')

		self._shear_x = FourierShear(input_grid, shear=0, shear_dim=0)
		self._shear_y = FourierShear(input_grid, shear=0, shear_dim=1)

		self.angle = angle

	@property
	def angle(self):
		return self._angle

	@angle.setter
	def angle(self, angle):
		self._shear_x.shear = np.tan(angle / 2)
		self._shear_y.shear = -np.sin(angle)

		self._angle = angle

	def forward(self, field):
		'''Return the forward rotation of the input field.

		Parameters
		----------
		field : Field
			The field to rotate.

		Returns
		-------
		Field
			The rotated field.
		'''
		return self._operation(field, adjoint=False)

	def backward(self, field):
		'''Return the backward (adjoint) rotation of the input field.

		Parameters
		----------
		field : Field
			The field to rotate.

		Returns
		-------
		Field
			The adjoint rotated field.
		'''
		return self._operation(field, adjoint=True)

	def _operation(self, field, adjoint):
		f1 = self._shear_x._operation(field, adjoint)
		f2 = self._shear_y._operation(f1, adjoint)
		f3 = self._shear_x._operation(f2, adjoint)

		return f1, f2, f3
