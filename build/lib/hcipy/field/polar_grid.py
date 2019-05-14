import numpy as np
import copy
from .grid import Grid

class PolarGrid(Grid):
	'''A grid representing a two-dimensional Polar coordinate system.
	'''
	
	_coordinate_system = 'polar'

	@property
	def r(self):
		'''The radial coordinate (dimension 0).
		'''
		return self.coords[0]

	@property
	def theta(self):
		''' The angular coordinate (dimension 1).
		'''
		return self.coords[1]
	
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
		self.coords *= np.array([scale, 1])
		self.weights *= np.abs(scale)**(self.ndim)
		return self

	def shift(self, shift):
		'''Shift the grid in-place.

		.. caution::
		   All structure in the coordinates will be destroyed.

		Parameters
		----------
		shift : array_like
			The amount with which to shift the grid.
		
		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		
		grid = PolarGrid(UnstructuredCoords(self.coords)).as_('cartesian')
		self.coords = new_grid.shift(shift).as_('polar').coords

		return self
	
	def shifted(self, shift):
		'''A shifted copy of this grid.

		.. caution::
		   The returned grid is a Cartesian grid.

		Parameters
		----------
		shift : array_like
			The amount with which to shift the grid.
		
		Returns
		-------
		Grid
			The scaled grid.
		'''
		grid = self.as_('cartesian')
		grid.shift(shift)
		return grid
	
	def rotate(self, angle, axis=None):
		'''Rotate the grid in-place.

		Parameters
		----------
		angle : scalar
			The angle in radians.
		axis : ndarray or None
			The axis of rotation. For this (polar) grid, it is ignored.
		
		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		self.theta += angle
		return self
	
	@staticmethod
	def _get_automatic_weights(coords):
		return None

def _cartesian_to_polar(self):
	from .coordinates import UnstructuredCoords
	
	x = self.x
	y = self.y
	r = np.hypot(x, y)
	theta = np.arctan2(y, x)
	return PolarGrid(UnstructuredCoords([r, theta]))

def _polar_to_cartesian(self):
	from .coordinates import UnstructuredCoords
	from .cartesian_grid import CartesianGrid
	
	r = self.r
	theta = self.theta
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	return CartesianGrid(UnstructuredCoords([x, y]))

Grid._add_coordinate_system_transformation('cartesian', 'polar', _cartesian_to_polar)
Grid._add_coordinate_system_transformation('polar', 'cartesian', _polar_to_cartesian)
