import numpy as np
from scipy.linalg import blas
from .fourier_transform import FourierTransform, multiplex_for_tensor_fields
from ..field import Field

class MatrixFourierTransform(FourierTransform):
	def __init__(self, input_grid, output_grid):
		# Check input grid assumptions
		if not input_grid.is_separated or not input_grid.is_('cartesian'):
			raise ValueError('The input_grid must be separable in cartesian coordinates.')
		if not output_grid.is_separated or not output_grid.is_('cartesian'):
			raise ValueError('The output_grid must be separable in cartesian coordinates.')
		if input_grid.ndim not in [1, 2]:
			raise ValueError('The input_grid must be one- or two-dimensional.')
		if input_grid.ndim != output_grid.ndim:
			raise ValueError('The input_grid must have the same dimensions as the output_grid.')

		self.input_grid = input_grid
		self.output_grid = output_grid

		self.shape_input = input_grid.shape
		self.shape_output = output_grid.shape

		self.ndim = input_grid.ndim

		self.weights_input = input_grid.weights
		self.weights_output = output_grid.weights / (2 * np.pi)**self.ndim

		# If all input weights are all the same, use a scalar instead.
		if np.all(self.weights_input == self.weights_input[0]):
			self.weights_input = self.weights_input[0]

		# If all output weights are all the same, use a scalar instead.
		if np.all(self.weights_output == self.weights_output[0]):
			self.weights_output = self.weights_output[0]

		if self.ndim == 1:
			self.M = np.exp(-1j * np.outer(output_grid.x, input_grid.x))
		elif self.ndim == 2:
			self.M1 = np.exp(-1j * np.outer(output_grid.coords.separated_coords[1], input_grid.separated_coords[1]))
			self.M2 = np.exp(-1j * np.outer(input_grid.coords.separated_coords[0], output_grid.separated_coords[0]))

	@multiplex_for_tensor_fields
	def forward(self, field):
		if self.ndim == 1:
			f = field * self.weights_input
			res = np.dot(self.M, f)
		elif self.ndim == 2:
			if np.isscalar(self.weights_input):
				f = field.reshape(self.shape_input)
				if np.dtype('complex128') == field.dtype:
					# Use handcoded BLAS call. BLAS is better when all inputs are Fortran ordered,
					# so we apply matrix multiplications on the transpose of each of the arrays
					# (which are C ordered). Weights are included in the call as that multiplication
					# happens anyway (and it saves an array copy).
					res = blas.zgemm(self.weights_input, blas.zgemm(1, self.M2.T, f.T), self.M1.T).T.reshape(-1)
				elif self.input_grid.size > self.output_grid.size:
					# Apply weights to the output as that array is smaller than the input.
					res = np.dot(np.dot(self.M1, f), self.M2).reshape(-1) * self.weights_input
				else:
					# Apply weights to the input as that array is smaller than the output.
					res = np.dot(np.dot(self.M1, f * self.weights_input), self.M2).reshape(-1)
			else:
				# Fallback in case the weights is not a scalar.
				f = (field * self.weights_input).reshape(self.shape_input)
				res = np.dot(np.dot(self.M1, f), self.M2).reshape(-1)

		return Field(res, self.output_grid).astype(field.dtype)

	@multiplex_for_tensor_fields
	def backward(self, field):
		if self.ndim == 1:
			f = field * self.weights_output
			res = np.dot(self.M.conj().T, f)
		elif self.ndim == 2:
			if np.dtype('complex128') == field.dtype:
				if np.isscalar(self.weights_output):
					# Use handcoded BLAS call. BLAS is better when all inputs are Fortran ordered,
					# so we apply matrix multiplications on the transpose of each of the arrays
					# (which are C ordered). Weights are included in the call as that multiplication
					# happens anyway (and it saves an array copy). Adjoint is handled by GEMM, which
					# avoids an array copy for these array as well.
					f = field.reshape(self.shape_output)
					res = blas.zgemm(1, blas.zgemm(self.weights_output, self.M2.T, f.T, trans_a=2), self.M1.T, trans_b=2).T.reshape(-1)
				else:
					# Fallback in case the weights is not a scalar.
					f = (field * self.weights_output).reshape(self.shape_output)
					res = blas.zgemm(1, blas.zgemm(1, self.M2.T, f.T, trans_a=2), self.M1.T, trans_b=2).T.reshape(-1)
			else:
				if np.isscalar(self.weights_output) and self.input_grid.size < self.output_grid.size:
					# Apply weights in the output, as that array is smaller than the input.
					f = field.reshape(self.shape_output)
					res = np.dot(np.dot(self.M1.conj().T, f), self.M2.conj().T).reshape(-1) * self.weights_output
				else:
					# Apply weights in the input, as that array is smaller than the output or if the
					# weights is not a scalar.
					f = (field * self.weights_output).reshape(self.shape_output)
					res = np.dot(np.dot(self.M1.conj().T, f), self.M2.conj().T).reshape(-1)

		return Field(res, self.input_grid).astype(field.dtype)
