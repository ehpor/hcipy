import numpy as np

from .fast_fourier_transform import FastFourierTransform
from .fourier_transform import multiplex_for_tensor_fields
from ..field import Field

class FourierFilter(object):
	'''A filter in the Fourier domain.

	The filtering is performed by Fast Fourier Transforms, but is quicker than
	the equivalent multiplication in the Fourier domain using the FastFourierTransform
	classes. It does this by avoiding redundant field multiplications that limit performance.

	Parameters
	----------
	input_grid : Grid
		The grid that is expected for the input field.
	transfer_function : Field generator
		The transfer function to use for the filter.
	q : scalar
		The amount of zeropadding to perform in the real domain. A value
		of 1 denotes no zeropadding. Zeropadding increases the resolution in the
		Fourier domain and therefore reduces aliasing/wrapping effects.
	'''
	def __init__(self, input_grid, transfer_function, q=1):
		fft = FastFourierTransform(input_grid, q)

		self.input_grid = input_grid
		self.cutout = fft.cutout_input
		self.shape_in = input_grid.shape

		self._transfer_function = np.fft.ifftshift(transfer_function(fft.output_grid).shaped)
		self.internal_array = np.zeros_like(self._transfer_function)

	@multiplex_for_tensor_fields
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

	@multiplex_for_tensor_fields
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
		if self.cutout is None:
			f = field.reshape(self.shape_in)
		else:
			f = self.internal_array
			f[:] = 0
			f[self.cutout] = field.reshape(self.shape_in)

		f = np.fft.fftn(f)

		if adjoint:
			f *= self._transfer_function.conj()
		else:
			f *= self._transfer_function

		f = np.fft.ifftn(f)

		if self.cutout is None:
			res = f.ravel()
		else:
			res = f[self.cutout].ravel()

		return Field(res, self.input_grid)
