import numpy as np
from .fourier_transform import FourierTransform, multiplex_for_tensor_fields

class MatrixFourierTransform(FourierTransform):
	def __init__(self, input_grid, output_grid):
		# Check input grid assumptions
		if not input_grid.is_separated or not input_grid.is_('cartesian'):
			raise ValueError('The input_grid must be separable in cartesian coordinates.')
		if not output_grid.is_separated or not output_grid.is_('cartesian'):
			raise ValueError('The output_grid must be separable in cartesian coordinates.')
		if not input_grid.ndim in [1,2]:
			raise ValueError('The input_grid must be one- or two-dimensional.')
		if input_grid.ndim != output_grid.ndim:
			raise ValueError('The input_grid must have the same dimensions as the output_grid.')
		
		self.input_grid = input_grid
		
		self.shape = input_grid.shape
		self.weights = input_grid.weights.ravel()
		self.output_grid = output_grid
		self.ndim = input_grid.ndim
		
		self.output_grid = output_grid
		
		if self.ndim == 1:
			self.M = np.exp(-1j * np.dot(output_grid.x[:,np.newaxis], input_grid.x[np.newaxis,:]))
		elif self.ndim == 2:
			self.M1 = np.exp(-1j * np.dot(output_grid.coords.separated_coords[1][:,np.newaxis], input_grid.coords.separated_coords[1][np.newaxis,:]))
			self.M2 = np.exp(-1j * np.dot(output_grid.coords.separated_coords[0][:,np.newaxis], input_grid.coords.separated_coords[0][np.newaxis,:])).T
	
	@multiplex_for_tensor_fields
	def forward(self, field):
		from ..field import Field

		if self.ndim == 1:
			f = field.ravel() * self.weights
			res = np.dot(self.M, f)
		elif self.ndim == 2:
			f = (field.ravel() * self.weights).reshape(self.shape)
			res = np.dot(np.dot(self.M1, f), self.M2).ravel()
		
		return Field(res, self.output_grid)
	
	@multiplex_for_tensor_fields
	def backward(self, field):
		from ..field import Field
		
		if self.ndim == 1:
			f = field.ravel() * self.output_grid.weights
			res = np.dot(self.M.conj().T, f)
		elif self.ndim == 2:
			f = (field.ravel() * self.output_grid.weights).reshape(self.output_grid.shape)
			res = np.dot(np.dot(self.M1.conj().T, f), self.M2.conj().T).ravel()
			
		res /= (2*np.pi)**self.ndim
		
		return Field(res, self.input_grid)