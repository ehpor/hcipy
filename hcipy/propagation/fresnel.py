import numpy as np
from .propagator import MonochromaticPropagator, make_propagator
from ..optics import Wavefront
from ..field import Field
from ..fourier import FastFourierTransform

class FresnelPropagatorMonochromatic(object):
	def __init__(self, input_grid, distance, wavelength=1):
		self.fft = FastFourierTransform(input_grid)
		
		k = 2*np.pi / wavelength
		k_squared = self.fft.output_grid.as_('polar').r**2
		phase_factor = np.exp(1j * k * distance)
		self.transfer_function = np.exp(-0.5j * distance * k_squared / k)# * phase_factor
	
	def forward(self, wavefront):
		ft = self.fft.forward(wavefront.electric_field)
		ft *= self.transfer_function
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)
	
	def backward(self, wavefront):
		ft = self.fft.forward(wavefront.electric_field)
		ft /= self.transfer_function
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)

FresnelPropagator = make_propagator(FresnelPropagatorMonochromatic)