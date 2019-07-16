import numpy as np
import copy
import warnings
import xxhash

class Grid(object):
	'''A set of points on some coordinate system.

	Parameters
	----------
	coords : CoordsBase
		The actual definition of the coordinate values.
	weights : array_like or None
		The interval size, area, volume or hypervolume of each point, depending on the number of dimensions.
		If this is None (default), the weights will be attempted to be calculated on the fly when needed.
	
	Attributes
	----------
	coords
		The coordinate values for each dimension.
	'''

	_coordinate_system = 'none'
	_coordinate_system_transformations = {}
	
	def __init__(self, coords, weights=None):
		self.coords = coords
		self.weights = weights
	
	def copy(self):
		'''Create a copy.
		'''
		return copy.deepcopy(self)
	
	def subset(self, criterium):
		'''Construct a subset of the current sampling, based on `criterium`.

		Parameters
		----------
		criterium : function or array_like
			The criterium used to select points. A function will be evaluated for every point. 
			Otherwise, this must be a boolean array of integer array, used for slicing the points.
		
		Returns
		-------
		Grid
			A new grid with UnstructuredCoords that includes only the points for which the criterium 
			was true.
		'''
		from .coordinates import UnstructuredCoords
		
		if hasattr(criterium, '__call__'):
			indices = criterium(self) != 0
		else:
			indices = criterium
		new_coords = [c[indices] for c in self.coords]
		new_weights = self.weights[indices]
		return self.__class__(UnstructuredCoords(new_coords), new_weights)
	
	@property
	def ndim(self):
		'''The number of dimensions.
		'''
		return len(self.coords)
	
	@property
	def size(self):
		'''The number of points in this grid.
		'''
		return self.coords.size
	
	def __len__(self):
		'''The number of points in this grid.
		'''
		return self.size

	@property
	def dims(self):
		'''The number of elements in each dimension for a separated grid.

		Raises
		------
		ValueError
			If the grid is not separated.
		'''
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have dims.')
		return self.coords.dims
	
	@property
	def shape(self):
		'''The shape of a reshaped ``numpy.ndarray`` using this grid.

		Raises
		------
		ValueError
			If the grid is not separated.
		'''
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have a shape.')
		return self.coords.shape

	@property
	def delta(self):
		'''The spacing between points in regularly-spaced grid.

		Raises
		------
		ValueError
			If the grid is not regular.
		'''
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have a delta.')
		return self.coords.delta
	
	@property
	def zero(self):
		'''The zero point of a regularly-spaced grid.

		Raises
		------
		ValueError
			If the grid is not regular.
		'''
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have a zero.')
		return self.coords.zero
	
	@property
	def separated_coords(self):
		'''A list of coordinates for each dimension in a separated grid.

		Raises
		------
		ValueError
			If the grid is not separated.
		'''
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have separated coordinates.')
		return self.coords.separated_coords
	
	@property
	def regular_coords(self):
		'''The tuple (delta, dims, zero) for a regularly-spaced grid.

		Raises
		------
		ValueError
			If the grid is not regular.
		'''
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have regular coordinates.')
		return self.coords.regular_coords
	
	@property
	def weights(self):
		'''The interval size, area, volume or hypervolume of each point, depending on the number of dimensions.

		The weights are attempted to be calculated on the fly if not set. If this fails, a warning is emitted and all points will be given an equal weight of one. 
		'''
		if self._weights is None:
			self._weights = self.__class__._get_automatic_weights(self.coords)

			if self._weights is None:
				self._weights = 1
				warnings.warn('No automatic weights could be calculated for this grid.', stacklevel=2)
		
		if np.isscalar(self._weights):
			return np.ones(self.size) * self._weights
		else:
			return self._weights
	
	@weights.setter
	def weights(self, weights):
		self._weights = weights
	
	@property
	def points(self):
		'''A list of points of this grid.

		This can be used for easier iteration::
		
			for p in grid.points:
				print(p)
		'''
		return np.array(self.coords).T
	
	@property
	def is_separated(self):
		'''True if the grid is separated, False otherwise.
		'''
		return self.coords.is_separated

	@property
	def is_regular(self):
		'''True if the grid is regularly-spaced, False otherwise.
		'''
		return self.coords.is_regular
	
	@property
	def is_unstructured(self):
		'''True if the grid is unstructured, False otherwise.
		'''
		return self.coords.is_unstructured
	
	def is_(self, system):
		'''Check if the coordinate system is `system`.

		Parameters
		----------
		system : str
			The name of the coordinate system to check for.

		Returns
		-------
		bool
			If the coordinate system of the grid is equal to `system`.
		'''
		return system == self._coordinate_system

	def as_(self, system):
		'''Convert the grid to the new coordinate system `system`.

		If the grid is already in the right coordinate system, this function doesn't do anything.

		Parameters
		----------
		system : str
			The name of the coordinate system to check for.

		Returns
		-------
		Grid
			A new :class:`Grid` in the required coordinate system.
		
		Raises
		------
		ValueError
			If the conversion to the coordinate system `system` isn't known.
		'''
		if self.is_(system):
			return self
		else:
			return Grid._coordinate_system_transformations[self._coordinate_system][system](self)
	
	@staticmethod
	def _add_coordinate_system_transformation(source, dest, func):
		if source in Grid._coordinate_system_transformations:
			Grid._coordinate_system_transformations[source][dest] = func
		else:
			Grid._coordinate_system_transformations[source] = {dest: func}
	
	def __getitem__(self, i):
		'''The `i`-th point in this grid.
		'''
		return self.points[i]

	def scale(self, scale):
		'''Scale the grid in-place.

		Parameters
		----------
		scale : array_like
			The factor with which to scale the grid.
		
		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		raise NotImplementedError()
	
	def scaled(self, scale):
		'''A scaled copy of this grid.

		Parameters
		----------
		scale : array_like
			The factor with which to scale the grid.
		
		Returns
		-------
		Grid
			The scaled grid.
		'''
		grid = self.copy()
		grid.scale(scale)
		return grid
	
	def shift(self, shift):
		'''Shift the grid in-place.

		Parameters
		----------
		shift : array_like
			The amount with which to shift the grid.
		
		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		raise NotImplementedError()
	
	def shifted(self, shift):
		'''A shifted copy of this grid.

		Parameters
		----------
		shift : array_like
			The amount with which to shift the grid.
		
		Returns
		-------
		Grid
			The scaled grid.
		'''
		grid = self.copy()
		grid.shift(shift)
		return grid
	
	def rotate(self, angle, axis=None):
		'''Rotate the grid in-place.

		Parameters
		----------
		angle : scalar
			The angle in radians.
		axis : ndarray or None
			The axis of rotation. For two-dimensional grids, it is ignored. For
			three-dimensional grids it is required.
		
		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		raise NotImplementedError()
	
	def rotated(self, angle, axis=None):
		'''A rotated copy of this grid.

		Parameters
		----------
		angle : scalar
			The angle in radians.
		axis : ndarray or None
			The axis of rotation. For two-dimensional grids, it is ignored. For
			three-dimensional grids it is required.
		
		Returns
		-------
		Grid
			The rotated grid.
		'''
		grid = self.copy()
		grid.rotate(angle, axis)
		return grid

	def reverse(self):
		'''Reverse the order of the points in-place.

		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		self.coords.reverse()
		return self

	def reversed(self):
		'''Make a copy of the grid with the order of the points reversed.

		Returns
		-------
		Grid
			The reversed grid.
		'''
		grid = self.copy()
		grid.reverse()
		return grid
	
	@staticmethod
	def _get_automatic_weights(coords):
		raise NotImplementedError()
	
	def __str__(self):
		return str(self.__class__.__name__) + '(' + str(self.coords.__class__.__name__) + ')'
	
	def __hash__(self):
		h = xxhash.xxh64()
		h.update(self._coordinate_system)

		if self.is_regular:
			h.update(self.delta)
			h.update(self.dims)
			h.update(self.zero)
		elif self.is_separated:
			for s in self.separated_coords:
				h.update(s)
		else:
			for s in self.coords:
				h.update(s)
		
		return h.intdigest()

	def closest_to(self, p):
		'''Get the index of the point closest to point `p`.

		Point `p` is assumed to have the same coordinate system as the grid itself.

		Parameters
		----------
		p : array_like
			The point at which to search for.
		
		Returns
		-------
		int
			The index of the closest point.
		'''
		rel_points = self.points - np.array(p) * np.ones(self.ndim)
		return np.argmin(np.sum(rel_points**2, axis=-1))
	
	def zeros(self, tensor_shape=None, dtype=None):
		'''Create a field of zeros from this `Grid`.

		Parameters
		----------
		tensor_shape : array_like or None
			The shape of the tensors in the to be created field. If this is None, 
			a scalar field will be created.
		dtype : data-type
			The numpy data-type with which to create the field.
		
		Returns
		-------
		Field
			A zeros field.
		'''
		from .field import Field

		if tensor_shape is None:
			shape = [self.size]
		else:
			shape = np.concatenate((self.size, tensor_shape))

		return Field(np.zeros(shape, dtype), self)
	
	def ones(self, tensor_shape=None, dtype=None):
		'''Create a field of ones from this `Grid`.

		Parameters
		----------
		tensor_shape : array_like or None
			The shape of the tensors in the to be created field. If this is None, 
			a scalar field will be created.
		dtype : data-type
			The numpy data-type with which to create the field.
		
		Returns
		-------
		Field
			A ones field.
		'''
		from .field import Field

		if tensor_shape is None:
			shape = [self.size]
		else:
			shape = np.concatenate((self.size, tensor_shape))

		return Field(np.ones(shape, dtype=dtype), self)

	def empty(self, tensor_shape=None, dtype=None):
		'''Create an empty Field from this `Grid`.

		Parameters
		----------
		tensor_shape : array_like or None
			The shape of the tensors in the to be created field. If this is None, 
			a scalar field will be created.
		dtype : data-type
			The numpy data-type with which to create the field.
		
		Returns
		-------
		Field
			A empty field.
		'''
		from .field import Field

		if tensor_shape is None:
			shape = [self.size]
		else:
			shape = np.concatenate((self.size, tensor_shape))

		return Field(np.empty(shape, dtype=dtype), self)
