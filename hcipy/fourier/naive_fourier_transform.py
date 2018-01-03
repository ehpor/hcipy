import numpy as np
from .fourier_transform import FourierTransform, multiplex_for_tensor_fields

class NaiveFourierTransform(FourierTransform):
	def __init__(self, input_grid, output_grid):
		self.input_grid = input_grid
		self.output_grid = output_grid
	
	@multiplex_for_tensor_fields
	def forward(self, field):
		T = self.get_transformation_matrix_forward()
		res = T.dot(field.ravel())

		from ..field import Field
		return Field(res, self.output_grid)
	
	@multiplex_for_tensor_fields
	def backward(self, field):
		T = self.get_transformation_matrix_backward()
		res = T.dot(field.ravel())

		from ..field import Field
		return Field(res, self.input_grid)