import numpy as np

class Field(np.ndarray):
	'''The value of some physical quantity for each point in some coordinate system.

	Parameters
	----------
	arr : array_like
		An array of values or tensors for each point in the :class:`Grid`.
	grid : Grid
		The corresponding :class:`Grid` on which the values are set.

	Attributes
	----------
	grid : Grid
		The grid on which the values are defined.
	'''
	def __new__(cls, arr, grid):
		obj = np.asarray(arr).view(cls)
		obj.grid = grid
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.grid = getattr(obj, 'grid', None)

	@classmethod
	def from_dict(cls, tree):
		from .grid import Grid

		return cls(np.array(tree['values']), Grid.from_dict(tree['grid']))

	def to_dict(self):
		tree = {
			'values': np.asarray(self),
			'grid': self.grid.to_dict()
		}

		return tree

	@property
	def tensor_order(self):
		'''The order of the tensor of the field.
		'''
		return self.ndim - 1

	@property
	def tensor_shape(self):
		'''The shape of the tensor of the field.
		'''
		return np.array(self.shape)[:-1]

	@property
	def is_scalar_field(self):
		'''True if this field is a scalar field (ie. a tensor order of 0), False otherwise.
		'''
		return self.tensor_order == 0

	@property
	def is_vector_field(self):
		'''True if this field is a vector field (ie. a tensor order of 1), False otherwise.
		'''
		return self.tensor_order == 1

	@property
	def is_valid_field(self):
		'''True if the field corresponds with its grid.
		'''
		return self.shape[-1] == self.grid.size

	@property
	def shaped(self):
		'''The reshaped version of this field.

		Raises
		------
		ValueError
			If this field isn't separated, no reshaped version can be made.
		'''
		if not self.grid.is_separated:
			raise ValueError('This field doesn\'t have a shape.')

		if self.tensor_order > 0:
			new_shape = np.concatenate([np.array(self.shape)[:-1], self.grid.shape])
			return self.reshape(new_shape)

		return self.reshape(self.grid.shape)

	def at(self, p):
		'''The value of this field closest to point p.

		Parameters
		----------
		p : array_like
			The point at which the closest value should be returned.

		Returns
		-------
		array_like
			The value, potentially tensor, closest to point p.
		'''
		i = self.grid.closest_to(p)
		return self[..., i]
