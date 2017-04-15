import numpy as np
from ..field.field import Field
class Detector(object):
	def integrate(self, wavefront, dt, weight=1):
		raise NotImplementedError()
	
	def read_out(self):
		raise NotImplementedError()
	
	def __call__(self, wavefront, dt=1, weight=1):
		self.integrate(wavefront, dt, weight)
		return self.read_out()

# TODO: add the detector noise
class CCD(Detector):
	def __init__(self, pixel_grid):
		self.intensity = Field(np.zeros((pixel_grid.size,)), pixel_grid)

	def integrate(self, wavefront, dt, weight=1):
		self.intensity += wavefront.intensity * dt * weight

	def read_out(self):
		# Get current counts
		output_field = self.intensity.copy()
		
		# Empty pixels
		self.intensity = 0

		# Return read out
		return output_field