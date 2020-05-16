import numpy as np

from .fast_fourier_transform import FastFourierTransform
from .matrix_fourier_transform import MatrixFourierTransform
from .fourier_transform import multiplex_for_tensor_fields
from ..field import Field

class ConvolveFFT(object):
	def __init__(self, input_grid, kernel):
		pass

class ShearFFT(object):
	pass

class RotateFFT(object):
	pass

class FourierFilter(object):
	def __init__(self, input_grid, transfer_function, q=1):
		fft = FastFourierTransform(input_grid, q)

		self.input_grid = input_grid
		self.cutout = fft.cutout_input
		self.shape_in = input_grid.shape

		self._transfer_function = np.fft.ifftshift(transfer_function(fft.output_grid).shaped)
		self.internal_array = np.zeros_like(self._transfer_function)

	@multiplex_for_tensor_fields
	def forward(self, field):
		return self._operation(field, adjoint=False)

	@multiplex_for_tensor_fields
	def backward(self, field):
		return self._operation(field, adjoint=True)

	def _operation(self, field, adjoint):
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
