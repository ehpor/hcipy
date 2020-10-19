import numpy as np

_field_backends = {}

def field_backend(backend):
	def decorator(cls):
		cls.backend = backend
		_field_backends[backend] = cls

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

	def numpy(self):
		return self.as_backend('numpy')

	def tensorflow(self):
		return self.as_backend('tensorflow')

@field_backend('numpy')
class NumpyField(np.ndarray, FieldBase):
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
		'''Make a Field from a dictionary, previously created by `to_dict()`.

		Parameters
		----------
		tree : dictionary
			The dictionary from which to make a new Field object.

		Returns
		-------
		Field
			The created object.

		Raises
		------
		ValueError
			If the dictionary is not formatted correctly.
		'''
		from .grid import Grid

		return cls(np.array(tree['values']), Grid.from_dict(tree['grid']))

	def to_dict(self):
		'''Convert the object to a dictionary for serialization.

		Returns
		-------
		dictionary
			The created dictionary.
		'''
		tree = {
			'values': np.asarray(self),
			'grid': self.grid.to_dict()
		}

		return tree

	def __getstate__(self):
		'''Get the internal state for pickling.

		Returns
		-------
		tuple
			The state of the Field.
		'''
		data_state = super().__reduce__(self)[2]
		return data_state + (self.grid,)

	def __setstate__(self, state):
		'''Set the internal state for pickling.

		Parameters
		----------
		state : tuple
			The state coming from a __getstate__().
		'''
		_, shp, typ, isf, raw, grid = state

		super().__setstate__((shp, typ, isf, raw))
		self.grid = grid

	def __reduce__(self):
		'''Return a 3-tuple for pickling the Field.

		Returns
		-------
		tuple
			The reduced version of the Field.
		'''
		return (_field_reconstruct,
			(self.__class__, np.ndarray, (0,), 'b',),
			self.__getstate__())

	@classmethod
	def _from_numpy(cls, field):
		return field

	def _to_numpy(self):
		return self

def _field_reconstruct(subtype, baseclass, baseshape, basetype):
	'''Internal function for building a new Field object for pickling.

	Parameters
	----------
	subtype
		The class of Field.
	baseclass
		The array class that was used for the Field.
	baseshape
		The shape of the Field.
	basetype
		The data type of the Field.

	Returns
	-------
	Field
		The built Field object.
	'''
	data = np.ndarray.__new__(baseclass, baseshape, basetype)
	grid = None

	return subtype.__new__(subtype, data, grid)

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
