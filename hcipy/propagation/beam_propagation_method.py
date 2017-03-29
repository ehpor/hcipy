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
		k_z = numpy.sqrt( k**2 - k_squared + 0j )
		self.transfer_function = np.exp( 1j*k_z*distance )
	
	def forward(self, wavefront):
		ft = self.fft.forward(wavefront.electric_field)
		ft *= self.transfer_function
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)
	
	def backward(self, wavefront):
		ft = self.fft.forward(wavefront.electric_field)
		ft /= self.transfer_function
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)

WideAnglePropagator = make_propagator(WideAnglePropagatorMonochromatic)