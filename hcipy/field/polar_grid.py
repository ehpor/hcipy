import numpy as np
import copy
from .grid import Grid

class PolarGrid(Grid):
	coordinate_system = 'polar'

	@property
	def r(self):
		return self.coords[0]

	@property
	def theta(self):
		return self.coords[1]
	
	def scale(self, scale):
		self.coords *= np.array([scale, 1])
		self.weights *= np.abs(scale)**(self.ndim)

	def shift(self, shift):
		raise ValueError("Can't shift a PolarGrid inplace.");
	
	def shifted(self, shift):
		grid = self.as_('cartesian')
		grid.shift(shift)
		return grid
	
	@staticmethod
	def get_automatic_weights(coords):
		return None

def cartesian_to_polar(self):
	from .coordinates import UnstructuredCoords
	
	x = self.x
	y = self.y
	r = np.hypot(x, y)
	theta = np.arctan2(y, x)
	return PolarGrid(UnstructuredCoords([r, theta]))

def polar_to_cartesian(self):
	from .coordinates import UnstructuredCoords
	from .cartesian_grid import CartesianGrid
	
	r = self.r
	theta = self.theta
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	return CartesianGrid(UnstructuredCoords([x, y]))

Grid.add_coordinate_system_transformation('cartesian', 'polar', cartesian_to_polar)
Grid.add_coordinate_system_transformation('polar', 'cartesian', polar_to_cartesian)
