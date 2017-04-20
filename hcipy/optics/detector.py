import numpy as np
from ..field import Field

class Detector(object):
	def integrate(self, wavefront, dt, weight=1):
		raise NotImplementedError()
	
	def read_out(self):
		raise NotImplementedError()
	
	def __call__(self, wavefront, dt=1, weight=1):
		self.integrate(wavefront, dt, weight)
		return self.read_out()

class PerfectDetector(Detector):
	def __init__(self):
		self.intensity = 0

	def integrate(self, wavefront, dt, weight=1):
		self.intensity += wavefront.intensity * dt * weight

	def read_out(self):
		# Make sure not to overwrite output
		output_field = self.intensity.copy()

		# Reset detector
		self.intensity = 0

		return output_field

# TODO: add detector noise models
class NoisyDetector(Detector):
	pass