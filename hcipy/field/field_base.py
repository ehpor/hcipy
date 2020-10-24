import numpy as np

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

		return _field_backends[backend]._from_numpy(self._to_numpy())

	'''
	def numpy(self):
		return self.as_backend('numpy')

	def tensorflow(self):
		return self.as_backend('tensorflow')
	'''

from .field_numpy import NumpyField
Field = NumpyField

'''

def _operator(operator):
	def f(self, *args):
		if args:
			# Binary operator
			b = args[0]
			try:
				return self.__class__(getattr(self.arr, operator)(b.arr), self.grid)
			except AttributeError:
				return self.__class__(getattr(self.arr, operator)(b), self.grid)
		else:
			# Unary operator
			return self.__class__(getattr(self.arr, operator)(), self.grid)
	return f

def _inplace_operator(inplace_operator):
	non_inplace_operator = inplace_operator[:2] + inplace_operator[3:]

	def f(self, b):
		try:
			self.arr = getattr(self.arr, non_inplace_operator)(b.arr)
		except AttributeError:
			self.arr = getattr(self.arr, non_inplace_operator)(b)

		return self
	return f

@field_backend('tensorflow')
class TensorFlowField(FieldBase):
	def __new__(cls, arr, grid):
		if not hasattr(cls, '__add__'):
			import tensorflow as tf

			for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
				setattr(cls, operator, _operator(operator))

			for operator in ['__iadd__', '__isub__', '__imul__', '__imatmul__', '__itruediv__',
					'__ifloordiv__', '__imod__', '__ipow__', '__iand__', '__ixor__', '__ior__']:
				setattr(cls, operator, _inplace_operator(operator))

		return super().__new__(cls)

	def __init__(self, arr, grid):
		self.arr = arr
		self.grid = grid

	def __copy__(self):
		return self.__class__(self.arr, self.grid)

	def __deepcopy__(self, memo):
		import copy
		return self.__class__(copy.deepcopy(self.arr, memo), self.grid)

	def __getattr__(self, attr):
		return getattr(self.arr, attr)

	def reshape(self, shape, order='C'):
		import tensorflow as tf
		return self.__class__(tf.reshape(self.arr, shape), self.grid)

	def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
		import tensorflow as tf
		return self.__class__(tf.cast(self.arr, dtype), self.grid)

	def __str__(self):
		return self.arr.__str__()

	def __repr__(self):
		return self.arr.__repr__()

	def __array__(self, dtype=None):
		res = self.numpy()

		if dtype is not None:
			res = res.astype(dtype)

		return res

	@property
	def real(self):
		return self.__class__(tf.math.real(self.arr), self.grid)

	@property
	def imag(self):
		return self.__class__(tf.math.imag(self.arr), self.grid)

	def max(self, axis=None, out=None, keepdims=False):
		import tensorflow as tf
		maximum = tf.reduce_max(self.arr, axis=axis, keepdims=keepdims)

		if out is not None:
			out[:] = maximum

		return maximum

	def min(self, axis=None, out=None, keepdims=False):
		import tensorflow as tf
		minimum = tf.reduce_min(self.arr, axis=axis, keepdims=keepdims)

		if out is not None:
			out[:] = minimum

		return minimum

	@property
	def ndim(self):
		return len(self.shape)

	@classmethod
	def _from_numpy(cls, field):
		import tensorflow as tf
		return cls(tf.convert_to_tensor(field), field.grid)

	def _to_numpy(self):
		import tensorflow as tf
		return NumpyField(self.arr.numpy(), self.grid)

Field = NumpyField
'''