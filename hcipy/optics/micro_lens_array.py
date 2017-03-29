import numpy as np
from .optical_element import OpticalElement
from ..field import Field

class MicroLensArray(OpticalElement):
	def __init__(self, input_grid, micro_lens_grid, micro_lens_shape, focal_length):
		self.input_grid = input_grid

		self.mla_index = Field(-np.ones(self.input_grid.size), self.input_grid)
		self.mla_opd = Field(np.zeros(self.input_grid.size), self.input_grid)

		for i, (x,y) in enumerate(micro_lens_grid.as_('cartesian').points):
			shifted_grid = input_grid.shifted((x,y))
			mask = micro_lens_shape(shifted_grid) != 0

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