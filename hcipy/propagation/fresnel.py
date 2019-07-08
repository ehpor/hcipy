import numpy as np
from .propagator import MonochromaticPropagator
from ..optics import Wavefront, make_polychromatic
from ..field import Field
from ..fourier import FastFourierTransform

class FresnelPropagatorMonochromatic(object):
	'''The monochromatic Fresnel propagator for scalar fields.

	The Fresnel propagator is implemented as described in "Fourier Optics" by J. W. Goodman.

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
		
		if np.any(input_grid.delta < wavelength * distance / Lmax):
			r_squared = input_grid.as_('polar').r**2
			impulse_response = Field(np.exp(1j * k * distance)/(1j * wavelength * distance) * np.exp(1j * k * r_squared / (2*distance)), input_grid)
			self.transfer_function = self.fft.forward(impulse_response)
		else:
			k_squared = self.fft.output_grid.as_('polar').r**2
			phase_factor = np.exp(1j * k * distance)
			self.transfer_function = np.exp(-0.5j * distance * k_squared / k) * phase_factor
	
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
		ft *= np.conj(self.transfer_function)
		return Wavefront(self.fft.backward(ft), wavefront.wavelength)

FresnelPropagator = make_polychromatic(["refractive_index"])(FresnelPropagatorMonochromatic)