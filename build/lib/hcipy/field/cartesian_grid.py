import numpy as np
import copy
from .grid import Grid
from .coordinates import UnstructuredCoords

def _get_rotation_matrix(ndim, angle, axis=None):
	if ndim == 1:
		raise ValueError('Rotation of a one-dimensional grid is not possible.')
	elif ndim > 3:
		raise NotImplementedError()
	
	if ndim == 2:
		return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
	elif ndim == 3:
		if axis is None:
			raise ValueError('An axis must be supplied when rotating a three-dimensional grid.')

		axis = np.array(axis).astype('float')
		if np.all(np.array(axis.shape) != (3,)):
			raise ValueError('The axis must be a 3-vector.')
		axis /= np.sqrt(axis.dot(axis))
		
		K = np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
		return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K.dot(K)

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
		scale : array_like or scalar
			The factor with which to scale the grid.
		
		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		if np.isscalar(scale):
			self.weights *= np.abs(scale)**self.ndim
		else:
			self.weights *= np.prod(np.abs(scale))
			
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
	
	def rotate(self, angle, axis=None):
		'''Rotate the grid in-place.

		.. caution::
			All structure in the coordinates will be destroyed.

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
		R = _get_rotation_matrix(self.ndim, angle, axis)
		
		coords = np.einsum('ik,kn->in', R, np.array(self.coords))
		self.coords = UnstructuredCoords(coords)
		return self
	
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
		R = _get_rotation_matrix(self.ndim, angle, axis)
		
		coords = np.einsum('ik,kn->in', R, np.array(self.coords))
		return CartesianGrid(UnstructuredCoords(coords))
	
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