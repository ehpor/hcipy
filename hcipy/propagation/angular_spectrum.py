import numpy as np
from .propagator import MonochromaticPropagator
from ..optics import Wavefront, make_polychromatic
from ..field import Field
from ..fourier import FastFourierTransform

class AngularSpectrumPropagatorMonochromatic(object):
	'''The monochromatic angular spectrum propagator for scalar fields.

	The scalar Angular Spectrum propagator is implemented as described by
	[1]_. The propagation of an electric field can be described as a transfer 
	function in frequency space. The transfer function is taken from 
	equation 9 of [1]_, and the related impulse response is taken from 
	equation 6 of [1]_.

	.. [1] Robert R. McLeod and Kelvin H. Wagner 2014, "Vector Fourier optics of anisotropic materials," Adv. Opt. Photon. 6, 368-412 (2014)

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
		
		if np.any(input_grid.delta < (wavelength * distance / L_max)):
			r_squared = input_grid.x**2 + input_grid.y**2 + distance**2
			r = np.sqrt(r_squared)
			
			cos_theta = distance / r
			impulse_response = Field(cos_theta / (2 * np.pi) * np.exp(1j * k * r) * (1 / r_squared - 1j * k / r), input_grid)
			self.transfer_function = self.fft.forward(impulse_response)
		else:
			k_squared = self.fft.output_grid.as_('polar').r**2
			k_z = np.sqrt(k**2 - k_squared + 0j)
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
