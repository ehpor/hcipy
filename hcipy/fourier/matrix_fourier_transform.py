import numpy as np

class MatrixFourierTransform(object):
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
			self.M1 = np.exp(-1j * np.dot(output_grid.coords.separated_coords[0][:,np.newaxis], input_grid.coords.separated_coords[0][np.newaxis,:]))
			self.M2 = np.exp(-1j * np.dot(output_grid.coords.separated_coords[1][:,np.newaxis], input_grid.coords.separated_coords[1][np.newaxis,:])).T
			
	def __call__(self, field):
		return self.forward(field)
	
	def forward(self, field):
		if self.ndim == 1:
			f = field.ravel() * self.weights
			res = np.dot(self.M, f)
		elif self.ndim == 2:
			f = (field.ravel() * self.weights).reshape(self.shape)
			res = np.dot(np.dot(self.M1, f), self.M2).ravel()
		
		if hasattr(field, 'grid'):
			from ..field import Field
			return Field(res, self.output_grid)
		else:
			return res
	
	def backward(self, field):
		if self.ndim == 1:
			f = field.ravel() * self.output_grid.weights
			res = np.dot(self.M.conj().T, f)
		elif self.ndim == 2:
			f = (field.ravel() * self.output_grid.weights).reshape(self.output_grid.shape)
			res = np.dot(np.dot(self.M1.conj().T, f), self.M2.conj().T).ravel()
			
		res /= (2*np.pi)**self.ndim
		
		if hasattr(field, 'grid'):
			from ..field import Field
			return Field(res, self.input_grid)
		else:
			return res