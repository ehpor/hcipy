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

class SurfaceApodizer(OpticalElement):
	def __init__(self, surface, refractive_index):
		self.surface = surface
		self.refractive_index = refractive_index
	
	def forward(self, wavefront):
		opd = (refractive_index(wavefront.wavelength) - 1) * self.surface
		
		wf = wavefront.copy()
		wf.electric_field *= np.exp(1j * opd * wf.wavenumber)

		return wf
	
	def backward(self, wavefront):
		opd = (refractive_index(wavefront.wavelength) - 1) * self.surface
		
		wf = wavefront.copy()
		wf.electric_field *= np.exp(-1j * opd * wf.wavenumber)

		return wf