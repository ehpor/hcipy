import numpy as np
from .optical_element import OpticalElement
from ..field import Field

def closest_points(point_grid, evaluated_grid):
	from scipy import spatial

	tree = spatial.KDTree(point_grid.points)
	d, i = tree.query(evaluated_grid.points)

	return Field(i, evaluated_grid), Field(d, evaluated_grid)

class MicroLensArray(OpticalElement):
	def __init__(self, input_grid, lenslet_grid, focal_length, lenslet_shape=None):
		self.input_grid = input_grid
		self.focal_length = focal_length
		
		self.mla_grid = lenslet_grid
		
		if lenslet_shape is None:
			indices, distances = closest_points(lenslet_grid, input_grid)

			self.mla_index = indices
			self.mla_opd = (-1 / (2*focal_length)) * distances**2
		else:
			self.mla_index = Field(-np.ones(self.input_grid.size), self.input_grid)
			self.mla_opd = Field(np.zeros(self.input_grid.size), self.input_grid)

			for i, (x,y) in enumerate(lenslet_grid.as_('cartesian').points):
				shifted_grid = input_grid.shifted((x,y))
				mask = lenslet_shape(shifted_grid) != 0
				
				self.mla_opd[mask] = (-1 / (2*focal_length)) * (shifted_grid.x[mask]**2 + shifted_grid.y[mask]**2)
				self.mla_index[mask] = i
			
	def forward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field *= np.exp(1j * self.mla_opd * 2*np.pi / wf.wavelength)
		return wf
		
	def backward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field /= np.exp(1j * self.mla_opd * 2*np.pi / wf.wavelength)
		return wf