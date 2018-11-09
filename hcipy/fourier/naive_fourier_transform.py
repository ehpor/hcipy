import numpy as np

from .fourier_transform import FourierTransform, multiplex_for_tensor_fields
from ..field import Field

class NaiveFourierTransform(FourierTransform):
	def __init__(self, input_grid, output_grid, cache_matrices=True):
		self.input_grid = input_grid
		self.output_grid = output_grid

		self._T_forward = None
		self._T_backward = None
		self.cache_matrices = cache_matrices

		self.coords_in = np.array(self.input_grid.as_('cartesian').coords)
		self.coords_out = np.array(self.output_grid.as_('cartesian').coords)
	
	@property
	def T_forward(self):
		if not self.cache_matrices:
			return self.get_transformation_matrix_forward()

		if self._T_forward is None:
			self._T_forward = self.get_transformation_matrix_forward()
		
		return self._T_forward
	
	@property
	def T_backward(self):
		if not self.cache_matrices:
			self.get_transformation_matrix_backward()
		
		if self._T_backward is None:
			self._T_backward = self.get_transformation_matrix_backward()
		
		return self._T_backward

	@multiplex_for_tensor_fields
	def forward(self, field):
		if self.cache_matrices:
			res = self.T_forward.dot(field.ravel())
			return Field(res, self.output_grid)
		else:
			res = np.array([(field * self.input_grid.weights).dot(np.exp(-1j * np.dot(p, self.coords_in))) for p in self.coords_out.T])
			return Field(res, self.output_grid)
	
	@multiplex_for_tensor_fields
	def backward(self, field):
		if self.cache_matrices:
			res = self.T_backward.dot(field.ravel())
			return Field(res, self.input_grid)
		else:
			res = np.array([(field * self.output_grid.weights).dot(np.exp(1j * np.dot(p, self.coords_out))) for p in self.coords_in.T])
			res /= (2*np.pi)**self.input_grid.ndim
			return Field(res, self.input_grid)