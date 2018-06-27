from .optical_element import *

class SimpleVibration(OpticalElement):
	def __init__(self, mode, amplitude, frequency, phase_0=0):
		self.mode = mode
		self.amplitude = amplitude
		self.frequency = frequency
		self.phase_0 = phase_0
		self.t = 0
	
	@property
	def frequency(self):
		return self._frequency
	
	@frequency.setter
	def frequency(self, frequency):
		delta_phase = 2*np.pi * (self.frequency + self.frequency)
		self.phase_0 += delta_phase
		self._frequency = frequency
	
	@property
	def phase(self):
		return 2*np.pi * self.frequency * self.t + self.phase_0
	
	def forward(self, wavefront):
		wf = wavefront.copy()

		wf.electric_field *= np.exp(1j * (self.mode * self.amplitude / wf.wavelength * np.sin(self.phase)))
		return wf
	
	def backward(self, wavefront):
		wf = wavefront.copy()

		wf.electric_field *= np.exp(-1j * (self.mode * self.amplitude / wf.wavelength * np.sin(self.phase)))
		return wf
