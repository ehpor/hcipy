import numpy as np
import copy
import warnings

class Grid(object):
	coordinate_system = 'none'
	coordinate_system_transformations = {}
	
	def __init__(self, coords, weights=None):
		self.coords = coords
		self.weights = weights
	
	def copy(self):
		return copy.deepcopy(self)
	
	def subset(self, criterium):
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
		return len(self.coords)
	
	@property
	def size(self):
		return self.coords.size
	
	@property
	def dims(self):
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have dims.')
		return self.coords.dims
	
	@property
	def shape(self):
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have a shape.')
		return self.coords.shape

	@property
	def delta(self):
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have a delta.')
		return self.coords.delta
	
	@property
	def zero(self):
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have a zero.')
		return self.coords.zero
	
	@property
	def separated_coords(self):
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have separated coordinates.')
		return self.coords.separated_coords
	
	@property
	def regular_coords(self):
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have regular coordinates.')
		return self.coords.regular_coords
	
	@property
	def weights(self):
		if self._weights is None:
			self._weights = self.__class__.get_automatic_weights(self.coords)

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
		return np.array(self.coords).T
	
	@property
	def is_separated(self):
		return self.coords.is_separated

	@property
	def is_regular(self):
		return self.coords.is_regular
	
	def is_(self, system):
		return system == self.coordinate_system

	def as_(self, system):
		if self.is_(system):
			return self
		else:
			return Grid.coordinate_system_transformations[self.coordinate_system][system](self)
	
	@staticmethod
	def add_coordinate_system_transformation(source, dest, func):
		if source in Grid.coordinate_system_transformations:
			Grid.coordinate_system_transformations[source][dest] = func
		else:
			Grid.coordinate_system_transformations[source] = {dest: func}
	
	def __getitem__(self, i):
		return self.points[i]

	def scale(self, scale):
		raise NotImplementedError
	
	def scaled(self, scale):
		grid = self.copy()
		grid.scale(scale)
		return grid
	
	def shift(self, shift):
		raise NotImplementedError
	
	def shifted(self, shift):
		grid = self.copy()
		grid.shift(shift)
		return grid
	
	def reverse(self):
		self.coords.reverse()
		return self

	def reversed(self):
		grid = self.copy()
		grid.reverse()
		return grid
	
	@staticmethod
	def get_automatic_weights(coords):
		raise NotImplementedError()
	
	def __str__(self):
		return str(self.__class__) + '(' + str(self.coords.__class__) + ')'

	def closest_to(self, p):
		rel_points = self.points - np.array(p) * np.ones(self.ndim)
		return np.argmin(np.sum(rel_points**2, axis=-1))