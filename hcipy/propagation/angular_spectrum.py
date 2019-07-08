import numpy as np
from .propagator import MonochromaticPropagator
from ..optics import Wavefront, make_polychromatic
from ..field import Field
from ..fourier import FastFourierTransform
from ..plotting import imshow_field
from matplotlib import pyplot as plt

class AngularSpectrumPropagatorMonochromatic(object):
	'''The monochromatic angular spectrum propagator for scalar fields.

	The scalar Angular Spectrum propagator is implemented as described by
	 "Vector Fourier optics of anisotropicmaterials", Mcleod 2014. The propagation
	 of an electric field can be described as a transfer function in frequency space.
	 The transfer function is taken from equation 9 of Mcleod 2014, and the related
	 impulse response is taken from equation 6.

	Parameters
	----------
	input_grid : Grid
		The grid on which the incoming wavefront is defined.
	distance : scalar
		The distance to propagate
	wavelength : scalar
		The wavelength of the wavefront.
	refractive_index : scalar
		The refractive index of the medium that the wavefront is propagating in.
	'''
	def __init__(self, input_grid, distance, wavelength=1, refractive_index=1):

		self.fft = FastFourierTransform(input_grid)

		k = 2*np.pi / wavelength * refractive_index
		
		# The FFT requires a regular cartesian grid there we can use the cartesian coordinates.
		Lmax = np.max([coord.ptp() for coord in input_grid.coords])
		
		if np.any(input_grid.delta > wavelength * distance / Lmax):

			r = np.sqrt(input_grid.as_('polar').r**2 + distance**2)
			cos_theta = distance/r
			impulse_response = cos_theta/(2*np.pi) * np.exp(1j * k * r) * (1/r**2 - 1j*k/r)
			
			self.transfer_function = self.fft.forward(impulse_response)
		else:
			k_squared = self.fft.output_grid.as_('polar').r**2
			k_z = np.real( np.sqrt(k**2 - k_squared + 0j) )
			self.transfer_function = np.exp(1j * k_z * distance)

	def forward(self, wavefront):
		'''Propagate a wavefront forward by a certain distance.
	
		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The wavefront after the propagation.
		'''
		ft = self.fft.forward(wavefront.electric_field)
		ft *= self.transfer_function
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)
	
	def backward(self, wavefront):
		'''Propagate a wavefront backward by a certain distance.
	
		Parameters
		----------
		wavefront : Wavefront
			The incoming wavefront.
		
		Returns
		-------
		Wavefront
			The wavefront after the propagation.
		'''
		ft = self.fft.forward(wavefront.electric_field)
		ft *= np.conj(self.transfer_function)
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)

AngularSpectrumPropagator = make_polychromatic(["refractive_index"])(AngularSpectrumPropagatorMonochromatic)