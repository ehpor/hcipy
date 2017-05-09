import numpy as np
import copy
from .grid import Grid

class CartesianGrid(Grid):
	'''A grid representing a N-dimensional Cartesian coordinate system.
	'''
	
	_coordinate_system = 'cartesian'

	@property
	def x(self):
		'''The x-coordinate (dimension 0).
		'''
		return self.coords[0]

	@property
	def y(self):
		'''The y-coordinate (dimension 1).
		'''
		return self.coords[1]

	@property
	def z(self):
		'''The z-coordinate (dimension 2).
		'''
		return self.coords[2]

	@property
	def w(self):
		'''The w-coordinate (dimension 3).
		'''
		return self.coords[3]
	
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
		self.weights *= np.abs(scale)**self.ndim
		self.coords *= scale
		return self

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
		self.coords += shift
		return self
	
	@staticmethod
	def _get_automatic_weights(coords):
		if coords.is_regular:
			return np.prod(coords.delta)
		elif coords.is_separated:
			weights = []
			for i in range(len(coords)):
				x = coords.separated_coords[i]
				w = (x[2:] - x[:-2]) / 2.
				w = np.concatenate(([x[1] - x[0]], w, [x[-1] - x[-2]]))
				weights.append(w)
			
			return np.multiply.reduce(np.ix_(*weights[::-1])).ravel()