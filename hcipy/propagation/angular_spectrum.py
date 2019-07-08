import numpy as np
from .propagator import MonochromaticPropagator
from ..optics import Wavefront, make_polychromatic
from ..field import Field
from ..fourier import FastFourierTransform
from ..plotting import imshow_field
from matplotlib import pyplot as plt

class AngularSpectrumPropagatorMonochromatic(object):
	'''The monochromatic propagator for scalar fields.

	Parameters
	----------
		
	Attributes
	----------
	
	'''
	def __init__(self, input_grid, distance, wavelength=1, refractive_index=1):

		self.fft = FastFourierTransform(input_grid)

		k = 2*np.pi / wavelength * refractive_index
		
		# The FFT requires a regular cartesian grid there we can use the cartesian coordinates.
		Lmax = np.max([coord.ptp() for coord in input_grid.coords])
		
		# This condition is violated we have a phase that varies to much
		# and we need to sample the transfer function in real space.
		if np.any(input_grid.delta > wavelength * distance / Lmax):
			
			# Distance from the origin to the evaluation point
			r = np.sqrt(input_grid.as_('polar').r**2 + distance**2)
			# Angle of the evaluation point w.r.t. propagation direction
			cos_theta = distance/r
			# Evaluate the impulse response equation 6 of "Vector Fourier optics of anisotropicmaterials", Mcleod 2014.
			impulse_response = cos_theta/(2*np.pi) * np.exp(1j * k * r) * (1/r**2 - 1j*k/r)
			
			self.transfer_function = self.fft.forward(impulse_response)
		else:
			k_squared = self.fft.output_grid.as_('polar').r**2
			k_z = np.real( np.sqrt(k**2 - k_squared + 0j) )
			self.transfer_function = np.exp(1j * k_z * distance)

	def forward(self, wavefront):
		ft = self.fft.forward(wavefront.electric_field)
		ft *= self.transfer_function
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)
	
	def backward(self, wavefront):
		ft = self.fft.forward(wavefront.electric_field)
		ft *= np.conj(self.transfer_function)
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)

AngularSpectrumPropagator = make_polychromatic(["refractive_index"])(AngularSpectrumPropagatorMonochromatic)