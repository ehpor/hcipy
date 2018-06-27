import numpy as np
from .optical_element import OpticalElement, make_polychromatic

class ApodizerMonochromatic(object):
	'''A monochromatic thin apodizer.
	
	This apodizer can apodize both in phase and amplitude. It assumes a 
	
	
	Parameters
	----------
	apodization : Field or scalar
		The apodization that we want to apply to any input wavefront.
	wavelength : scalar
		The wavelength at which the apodization is defined.
	'''
	def __init__(self, apodization, wavelength=1):
		self.apodization = apodization
		self.wavelength = wavelength
	
	def forward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field *= self.apodization
		return wf
	
	def backward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field /= self.apodization
		return wf

Apodizer = make_polychromatic(["apodization"])(ApodizerMonochromatic)

class PhaseApodizerMonochromatic(object):
	def __init__(self, phase, wavelength=1):
		self.phase = phase
		self.wavelength = wavelength
	
	def forward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field *= np.exp(1j * self.phase)
		return wf
	
	def backward(self, wavefront):
		wf = wavefront.copy()
		wf.electric_field *= np.exp(-1j * self.phase)
		return wf

PhaseApodizer = make_polychromatic(["phase"])(PhaseApodizerMonochromatic)

class ThinLens(OpticalElement):
	def __init__(self, focal_length):
		self.focal_length = focal_length
	
	def forward(self, wavefront):
		pass
	
	def backward(self, wavefront):
		pass

class SurfaceApodizerMonochromatic(OpticalElement):
	def __init__(self, surface, refractive_index, wavelength):
		self.surface = surface
		self.refractive_index = refractive_index
	
	def forward(self, wavefront):
		opd = (self.refractive_index - 1) * self.surface
		
		wf = wavefront.copy()
		wf.electric_field *= np.exp(1j * opd * wf.wavenumber)

		return wf
	
	def backward(self, wavefront):
		opd = (self.refractive_index - 1) * self.surface
		
		wf = wavefront.copy()
		wf.electric_field *= np.exp(-1j * opd * wf.wavenumber)

		return wf

SurfaceApodizer = make_polychromatic(["surface", "refractive_index"])(SurfaceApodizerMonochromatic)

class ComplexSurfaceApodizer(OpticalElement):
	def __init__(self, amplitude, surface, refractive_index):
		self.amplitude = amplitude
		self.surface = surface
		self.refractive_index = refractive_index
	
	def forward(self, wavefront):
		opd = (self.refractive_index(wavefront.wavelength) - 1) * self.surface
		
		wf = wavefront.copy()
		wf.electric_field *= self.amplitude * np.exp(1j * opd * wf.wavenumber)

		return wf
	
	def backward(self, wavefront):
		opd = (self.refractive_index(wavefront.wavelength) - 1) * self.surface
		
		wf = wavefront.copy()
		wf.electric_field *= np.exp(-1j * opd * wf.wavenumber) / self.amplitude

		return wf

class MultiplexedComplexSurfaceApodizer(OpticalElement):
	def __init__(self, amplitude, surface, refractive_index):
		self.amplitude = amplitude
		self.surface = surface
		self.refractive_index = refractive_index
	
	def forward(self, wavefront):
		apodizer_mask = 0
		for amplitude, surface in zip(self.amplitude, self.surface):
			opd = (self.refractive_index(wavefront.wavelength) - 1) * surface
			apodizer_mask += amplitude * np.exp(1j * opd * wavefront.wavenumber)

		wf = wavefront.copy()
		wf.electric_field *= apodizer_mask
		return wf
	
	def backward(self, wavefront):
		apodizer_mask = 0
		for amplitude, surface in zip(self.amplitude, self.surface):
			opd = (self.refractive_index(wavefront.wavelength) - 1) * surface
			apodizer_mask += amplitude * np.exp(1j * opd * wavefront.wavenumber)
			
		wf = wavefront.copy()
		wf.electric_field /= apodizer_mask
		return wf
