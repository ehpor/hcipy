import numpy as np
from scipy.linalg import blas
from .fourier_transform import FourierTransform, multiplex_for_tensor_fields
from ..field import Field
from ..config import Configuration

class MatrixFourierTransform(FourierTransform):
	def __init__(self, input_grid, output_grid, precompute_matrices=None, allocate_intermediate=None):
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

		if precompute_matrices is None:
			precompute_matrices = Configuration().fourier.mft.precompute_matrices
		self.precompute_matrices = precompute_matrices

		if allocate_intermediate is None:
			allocate_intermediate = Configuration().fourier.mft.allocate_intermediate
		self.allocate_intermediate = allocate_intermediate

		self.matrices_dtype = None
		self.intermediate_dtype = None
		self._remove_matrices()

	def _compute_matrices(self, dtype):
		if dtype == 'float32' or dtype == 'complex64':
			complex_dtype = 'complex64'
			float_dtype = 'float32'
		else:
			complex_dtype = 'complex128'
			float_dtype = 'float64'

		if self.matrices_dtype != complex_dtype:
			self.weights_input = (self.input_grid.weights).astype(float_dtype)
			self.weights_output = (self.output_grid.weights / (2 * np.pi)**self.ndim).astype(float_dtype)

			# If all input weights are all the same, use a scalar instead.
			if np.all(self.weights_input == self.weights_input[0]):
				self.weights_input = self.weights_input[0]

			# If all output weights are all the same, use a scalar instead.
			if np.all(self.weights_output == self.weights_output[0]):
				self.weights_output = self.weights_output[0]

			if self.ndim == 1:
				self.M = np.exp(-1j * np.outer(self.output_grid.x, self.input_grid.x)).astype(complex_dtype)
			elif self.ndim == 2:
				self.M1 = np.exp(-1j * np.outer(self.output_grid.coords.separated_coords[1], self.input_grid.separated_coords[1])).astype(complex_dtype)
				self.M2 = np.exp(-1j * np.outer(self.input_grid.coords.separated_coords[0], self.output_grid.separated_coords[0])).astype(complex_dtype)

			self.matrices_dtype = complex_dtype

		if self.intermediate_dtype != complex_dtype:
			if self.ndim == 2:
				self.intermediate_array = np.empty((self.M1.shape[0], self.input_grid.shape[1]), dtype=complex_dtype)

				self.intermediate_dtype = complex_dtype

	def _remove_matrices(self):
		if not self.precompute_matrices:
			if self.ndim == 1:
				self.M = None
			elif self.ndim == 2:
				self.M1 = None
				self.M2 = None

			self.matrices_dtype = None

		if not self.allocate_intermediate:
			if self.ndim == 2:
				self.intermediate_array = None

				self.intermediate_dtype = None

	@multiplex_for_tensor_fields
	def forward(self, field):
		self._compute_matrices(field.dtype)
		field = field.astype(self.matrices_dtype, copy=False)

		if self.ndim == 1:
			f = field * self.weights_input
			res = np.dot(self.M, f)
		elif self.ndim == 2:
			if np.isscalar(self.weights_input):
				f = field.reshape(self.shape_input)
				if field.dtype in [np.dtype('complex64'), np.dtype('complex128')]:
					# Use handcoded BLAS call. BLAS is better when all inputs are Fortran ordered,
					# so we apply matrix multiplications on the transpose of each of the arrays
					# (which are C ordered). Weights are included in the call as that multiplication
					# happens anyway (and it saves an array copy).
					if field.dtype == 'complex64':
						gemm = blas.cgemm
					else:
						gemm = blas.zgemm
					gemm(1, f.T, self.M1.T, c=self.intermediate_array.T, overwrite_c=True)
					res = gemm(self.weights_input, self.M2.T, self.intermediate_array.T).T.reshape(-1)
				elif self.input_grid.size > self.output_grid.size:
					# Apply weights to the output as that array is smaller than the input.
					np.dot(self.M1, f, out=self.intermediate_array)
					res = np.dot(self.intermediate_array, self.M2).reshape(-1) * self.weights_input
				else:
					# Apply weights to the input as that array is smaller than the output.
					np.dot(self.M1, f * self.weights_input, out=self.intermediate_array)
					res = np.dot(self.intermediate_array, self.M2).reshape(-1)
			else:
				# Fallback in case the weights is not a scalar.
				f = (field * self.weights_input).reshape(self.shape_input)
				np.dot(self.M1, f, out=self.intermediate_array)
				res = np.dot(self.intermediate_array, self.M2).reshape(-1)

		self._remove_matrices()

		return Field(res, self.output_grid)

	@multiplex_for_tensor_fields
	def backward(self, field):
		self._compute_matrices(field.dtype)
		field = field.astype(self.matrices_dtype, copy=False)

		if self.ndim == 1:
			f = field * self.weights_output
			res = np.dot(self.M.conj().T, f)
		elif self.ndim == 2:
			if field.dtype in [np.dtype('complex64'), np.dtype('complex128')]:
				if field.dtype == 'complex64':
					gemm = blas.cgemm
				else:
					gemm = blas.zgemm

				if np.isscalar(self.weights_output):
					# Use handcoded BLAS call. BLAS is better when all inputs are Fortran ordered,
					# so we apply matrix multiplications on the transpose of each of the arrays
					# (which are C ordered). Weights are included in the call as that multiplication
					# happens anyway (and it saves an array copy). Adjoint is handled by GEMM, which
					# avoids an array copy for these array as well.

					f = field.reshape(self.shape_output)
					gemm(1, self.M2.T, f.T, trans_a=2, c=self.intermediate_array.T, overwrite_c=True)
					res = gemm(self.weights_output, self.intermediate_array.T, self.M1.T, trans_b=2).T.reshape(-1)
				else:
					# Fallback in case the weights is not a scalar.
					f = (field * self.weights_output).reshape(self.shape_output)
					gemm(1, M2.T, f.T, trans_a=2, c=self.intermediate_array.T, overwrite_c=True)
					res = gemm(1, self.intermediate_array.T, self.M1.T, trans_b=2).T.reshape(-1)
			else:
				if np.isscalar(self.weights_output) and self.input_grid.size < self.output_grid.size:
					# Apply weights in the output, as that array is smaller than the input.
					f = field.reshape(self.shape_output)
					np.dot(f, self.M2.conj().T, out=self.intermediate_array)
					res = np.dot(self.M1.conj().T, self.intermediate_array).reshape(-1) * self.weights_output
				else:
					# Apply weights in the input, as that array is smaller than the output or if the
					# weights is not a scalar.
					f = (field * self.weights_output).reshape(self.shape_output)
					np.dot(f, self.M2.conj().T, out=self.intermediate_array)
					res = np.dot(self.M1.conj().T, self.intermediate_array).reshape(-1)

		self._remove_matrices()

		return Field(res, self.input_grid)
