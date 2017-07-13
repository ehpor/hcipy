import numpy as np
from .propagator import MonochromaticPropagator, make_propagator
from ..optics import Wavefront
from ..field import Field
from ..fourier import FastFourierTransform


# TODO: Add change sampling domain depending on distance.
# Real domain impulse response function is:
#
class AngularSpectrumPropagatorMonochromatic(object):
	def __init__(self, input_grid, distance, wavelength=1):
		# Maintain symmetry in the fourier grid:
		# shifts by half a pixel if the input grid has even dimensions.
		shift = (1 - np.mod(input_grid.dims, 2)) * input_grid.delta / 2
		self.fft = FastFourierTransform(input_grid, shift=shift)
		
		k = 2*np.pi / wavelength
		k_squared = self.fft.output_grid.as_('polar').r**2
		k_z = np.sqrt(k**2 - k_squared + 0j)
		self.transfer_function = np.exp(1j * k_z * distance)
	
	def forward(self, wavefront):
		ft = self.fft.forward(wavefront.electric_field)
		ft *= self.transfer_function
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)
	
	def backward(self, wavefront):
		ft = self.fft.forward(wavefront.electric_field)
		ft /= self.transfer_function
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)

AngularSpectrumPropagator = make_propagator(AngularSpectrumPropagatorMonochromatic)