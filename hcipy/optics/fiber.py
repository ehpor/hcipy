import numpy as np
from .optical_element import Detector

def fiber_mode_gaussian(grid, mode_field_diameter):
	g = grid.as_('polar')
	return np.exp(-(g.r/(0.5 * mode_field_diameter))**2)

class SingleModeFiber(Detector):
	def __init__(self, input_grid, mode_field_diameter, mode=None):
		self.input_grid = input_grid
		self.mode_field_diameter = mode_field_diameter

		if mode is None:
			mode = gaussian_mode
		
		self.mode = mode(self.input_grid, mode_field_diameter)
		self.mode /= np.sum(np.abs(self.mode)**2 * self.input_grid.weights)
		self.intensity = 0

	def integrate(self, wavefront, dt, weight=1):
		self.intensity += weight * dt * (np.dot(wavefront.electric_field * wavefront.electric_field.grid.weights, self.mode))**2
	
	def read_out(self):
		intensity = self.intensity
		self.intensity = 0
		return intensity