import numpy as np

from ..math import infer_backend_from_array

def _unwrap(arg):
	if isinstance(arg, Field):
		return arg.array

	if isinstance(arg, list):
		return [_unwrap(x) for x in arg]

	if isinstance(arg, dict):
		return {key: _unwrap(val) for key, val in arg.items()}

	if isinstance(arg, tuple):
		return tuple(_unwrap(x) for x in arg)

	return arg

class Field(np.lib.mixins.NDArrayOperatorsMixin):
	def __init__(self, array, grid):
		self.array = np.asarray(array)
		self.grid = grid

	@property
	def backend(self):
		return infer_backend_from_array(self.array.__class__)

	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		out = kwargs.get('out', ())

		inputs = _unwrap(inputs)

		if out:
			kwargs['out'] = tuple(x.array if isinstance(x, Field) else x for x in out)

		result = getattr(ufunc, method)(*inputs, **kwargs)

		if isinstance(result, np.ndarray):
			return Field(result, self.grid)
		else:
			return result

	def __array__(self, dtype=None):
		if dtype is None:
			return self.array

		return self.array.astype(dtype, copy=False)

	def __array_function__(self, func, types, args, kwargs):
		args = _unwrap(args)

		result = func(*args, **kwargs)

		if isinstance(result, np.ndarray):
			return Field(result, self.grid)
		return result

	def __getitem__(self, indices):
		res = self.array[indices]
		if np.isscalar(res):
			return res

		return Field(res, self.grid)

	def __setitem__(self, indices, values):
		self.array[indices] = values

		return self

	@property
	def T(self):
		return np.transpose(self)

	@property
	def dtype(self):
		return self.array.dtype

	@property
	def imag(self):
		return np.imag(self)

	@property
	def real(self):
		return np.real(self)

	@property
	def size(self):
		return np.prod(self.shape)

	@property
	def ndim(self):
		return len(self.shape)

	@property
	def shape(self):
		return self.array.shape

	def astype(self, dtype, *args, **kwargs):
		return Field(self.array.astype(dtype, *args, **kwargs), self.grid)

	def __len__(self):
		return len(self.array)

	def conj(self):
		return np.conj(self)

	def conjugate(self):
		return np.conjugate(self)

	all = np.all
	any = np.any
	argmax = np.argmax
	argmin = np.argmin
	argpartition = np.argpartition
	argsort = np.argsort
	clip = np.clip
	compress = np.compress
	#conj = np.conj
	#conjugate = np.conjugate
	copy = np.copy
	cumprod = np.cumprod
	cumsum = np.cumsum
	dot = np.dot
	flatten = np.ravel
	max = np.max
	mean = np.mean
	min = np.min
	nonzero = np.nonzero
	prod = np.prod
	ptp = np.ptp
	ravel = np.ravel
	repeat = np.repeat
	reshape = np.reshape
	round = np.round
	sort = np.sort
	squeeze = np.squeeze
	std = np.std
	sum = np.sum
	trace = np.trace
	transpose = np.transpose
	var = np.var

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

	def __repr__(self):
		'''The text representation of the field.

		Returns
		-------
		str
			The text representation of the field.
		'''
		return self.__class__.__name__ + '(' + repr(self.array) + ')'

	def __str__(self):
		'''The field converted to a string.

		Returns
		-------
		str
			The field converted to a string.
		'''
		return self.__class__.__name__ + '(' + str(self.array) + ')'

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
			'values': np.asarray(self.array),
			'backend': self.backend,
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
		data_state = self.array.__reduce__()[2]
		return data_state + (self.grid,)

	def __setstate__(self, state):
		'''Set the internal state for pickling.

		Parameters
		----------
		state : tuple
			The state coming from a __getstate__().
		'''
		_, shp, typ, isf, raw, grid = state

		self.array.__setstate__((shp, typ, isf, raw))
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

	return subtype(data, grid)

def is_field(obj):
	return isinstance(obj, Field)
