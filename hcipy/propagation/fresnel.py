import numpy as np
from .propagator import MonochromaticPropagator, make_propagator
from ..optics import Wavefront
from ..field import Field
from ..fourier import FastFourierTransform

class FresnelPropagatorMonochromatic(object):
	def __init__(self, input_grid, distance, wavelength=1):
		# Maintain symmetry in the fourier grid:
		# shifts by half a pixel if the input grid has even dimensions.
		shift = (1 - np.mod(input_grid.dims, 2)) * input_grid.delta / 2
		self.fft = FastFourierTransform(input_grid, shift=shift)
		
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