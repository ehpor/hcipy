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

		self.weights_input = input_grid.weights.ravel()
		if np.all(self.weights_input == self.weights_input[0]):
			self.weights_input = self.weights_input[0]

		self.weights_output = output_grid.weights.ravel()
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
					res = blas.zgemm(self.weights_input, blas.zgemm(1, self.M2.T, f.T), self.M1.T)
				elif self.input_grid.size > self.output_grid.size:
					res = np.dot(np.dot(self.M1, f), self.M2).reshape(-1) * self.weights_input
				else:
					res = np.dot(np.dot(self.M1, f * self.weights_input), self.M2).reshape(-1)
			else:
				f = (field * self.weights_input).reshape(self.shape_input)
				res = np.dot(np.dot(self.M1, f), self.M2).reshape(-1)

		return Field(res, self.output_grid)

	@multiplex_for_tensor_fields
	def backward(self, field):
		if self.ndim == 1:
			f = field * self.weights_output
			res = np.dot(self.M.conj().T, f) / (2 * np.pi)
		elif self.ndim == 2:
			if np.dtype('complex128') == field.dtype:
				if np.isscalar(self.weights_output):
					f = field.reshape(self.shape_output)
					w = self.weights_output / (2 * np.pi)**2
					res = blas.zgemm(1, blas.zgemm(w, self.M2.T, f.T, trans_a=2), self.M1.T, trans_b=2).T.reshape(-1)
				else:
					f = (field * self.weights_output).reshape(self.shape_output)
					w = 1 / (2 * np.pi)**2
					res = blas.zgemm(1, blas.zgemm(w, self.M2.T, f.T, trans_a=2), self.M1.T, trans_b=2).T.reshape(-1)
			else:
				if np.isscalar(self.weights_output) and self.input_grid.size < self.output_grid.size:
					f = field.reshape(self.shape_output)
					res = np.dot(np.dot(self.M1.conj().T, f), self.M2.conj().T).reshape(-1) * self.weights_output
				else:
					f = (field * self.weights_output).reshape(self.shape_output)
					res = np.dot(np.dot(self.M1.conj().T, f), self.M2.conj().T).reshape(-1)

		return Field(res, self.input_grid)
