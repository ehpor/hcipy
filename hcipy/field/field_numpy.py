import numpy as np

from .field import FieldBase, field_backend

@field_backend('numpy', ['np'])
class NumpyField(FieldBase, np.lib.mixins.NDArrayOperatorsMixin):
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
	def __init__(self, array, grid):
		self.array = np.asarray(array)
		self.grid = grid

	def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
		out = kwargs.get('out', ())

		inputs = tuple(x.array if isinstance(x, NumpyField) else x for x in inputs)

		if out:
			kwargs['out'] = tuple(x.array if isinstance(x, NumpyField) else x for x in out)

		result = getattr(ufunc, method)(*inputs, **kwargs)

		if isinstance(result, np.ndarray):
			return NumpyField(result, self.grid)
		else:
			return result

	def __array__(self, dtype=None):
		if dtype is None:
			return self.array

		return self.array.astype(dtype, copy=False)

	def __array_function__(self, func, types, args, kwargs):
		args = tuple(x.array if isinstance(x, NumpyField) else x for x in args)

		result = func(*args, **kwargs)

		if isinstance(result, np.ndarray):
			return NumpyField(result, self.grid)
		return result

	@classmethod
	def is_native_array(cls, array):
		return True

	def __getitem__(self, indices):
		return NumpyField(self.array[indices], self.grid)

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
		return NumpyField(self.array.astype(dtype, *args, **kwargs), self.grid)

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
		return (
			_field_reconstruct,
			(self.__class__, np.ndarray, (0,), 'b',),
			self.__getstate__()
		)

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
