import numpy as np

class Field(np.ndarray):
	def __new__(cls, arr, grid):
		obj = np.asarray(arr).view(cls)
		obj.grid = grid
		return obj
	
	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.grid = getattr(obj, 'grid', None)
	
	@property
	def tensor_order(self):
		return self.ndim - 1
	
	@property
	def is_scalar_field(self):
		return self.tensor_order == 0
	
	@property
	def is_vector_field(self):
		return self.tensor_order == 1
	
	@property
	def is_valid_field(self):
		return self.shape[0] == self.grid.size
	
	@property
	def shaped(self):
		if self.tensor_order > 0:
			new_shape = np.concatenate([self.grid.shape, np.array(self.shape)[1:]])
			return self.reshape(new_shape)
		return self.reshape(self.grid.shape)
	
	def at(self, p):
		i = self.grid.closest_to(p)
		return self[i]