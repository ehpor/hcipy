import numpy as np
from .optical_element import OpticalElement

class Apodizer(OpticalElement):
	def __init__(self, apodization):
		self.apodization = apodization
	
	def forward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field *= self.apodization
		return wf
	
	def backward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field /= self.apodization
		return wf

class PhaseApodizer(Apodizer):
	def __init__(self, phase):
		self.phase = phase
	
	@property
	def apodization(self):
		return np.exp(1j * self.phase)

class ThinLens(OpticalElement):
	def __init__(self, focal_length):
		self.focal_length = focal_length
	
	def forward(self, wavefront):
		pass
	
	def backward(self, wavefront):
		pass