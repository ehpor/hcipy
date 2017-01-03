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
	
	def __imul__(self, f):
		self.coords *= np.array([f, 1])
		self.weights *= np.abs(f)**(self.ndim)
		return self
	
	@staticmethod
	def get_automatic_weights(coords):
		return None

def cartesian_to_polar(self):
	x = self.x
	y = self.y
	r = np.hypot(x, y)
	theta = np.arctan2(y, x)
	return PolarGrid(Coords([r, theta]))

def polar_to_cartesian(self):
	r = self.r
	theta = self.theta
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	return CartesianGrid(Coords([x, y]))

Grid.add_coordinate_system_transformation('cartesian', 'polar', cartesian_to_polar)
Grid.add_coordinate_system_transformation('polar', 'cartesian', polar_to_cartesian)
