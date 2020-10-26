import numpy as np

from ..config import Configuration

_field_backends = {}

def field_backend(backend):
	def decorator(cls):
		cls.backend = backend
		_field_backends[backend] = cls
		setattr(FieldBase, backend, lambda self: self.as_backend(backend))

		return cls

	return decorator

class FieldBase(object):
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

	def as_backend(self, backend):
		if self.backend == backend:
			return self

		return _field_backends[backend](self, self.grid)

def Field(arr, grid):
	backend = Configuration().field.default_backend

	return _field_backends[backend](arr, grid)

def is_field(arg):
	return isinstance(arg, FieldBase)
