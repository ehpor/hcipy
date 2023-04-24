import numpy as np
from ..config import Configuration

class Field:
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
		if Configuration().core.use_new_style_fields:
			return super().__new__(NewStyleField)
		else:
			return OldStyleField.__new__(OldStyleField, arr, grid)

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

class OldStyleField(Field, np.ndarray):
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

class NewStyleField:
	pass
