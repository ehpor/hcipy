import numpy as np
from .propagator import MonochromaticPropagator
from ..optics import Wavefront, make_polychromatic
from ..field import Field
from ..fourier import FastFourierTransform

class FresnelPropagatorMonochromatic(object):
	'''The monochromatic Fresnel propagator for scalar fields.

	The Fresnel propagator is implemented as described in [1]_.

	.. [1] Goodman, J.W., 2005 Introduction to Fourier optics. Roberts and Company Publishers.

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

	Raises
	------
	ValueError
		If the `input_grid` is not regular and Cartesian.
	'''
	def __init__(self, input_grid, distance, wavelength=1, refractive_index=1):
		# The FFT requires a regular cartesian grid
		if not input_grid.is_regular or not input_grid.is_('cartesian'):
			raise ValueError('The input grid must be a regular, Cartesian grid.')
		
		self.fft = FastFourierTransform(input_grid)
		
		k = 2 * np.pi / wavelength * refractive_index
		L_max = np.max(input_grid.dims * input_grid.delta)
		
		if np.any(input_grid.delta < wavelength * distance / L_max):
			r_squared = input_grid.x**2 + input_grid.y**2
			impulse_response = Field(np.exp(1j * k * distance) / (1j * wavelength * distance) * np.exp(1j * k * r_squared / (2 * distance)), input_grid)
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
