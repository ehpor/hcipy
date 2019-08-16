from ..optics import Apodizer, OpticalElement, Wavefront
from ..propagation import FraunhoferPropagator
from ..field import Field
from ..fourier import FastFourierTransform, make_fft_grid

import numpy as np

class KnifeEdgeLyotCoronagraph(OpticalElement):
	'''A Lyot-style coronagraph with a centered, knife-edge focal-plane mask.

	As the knife-edge is invariant along one axis, the Fourier transforms can
	be sped up by only taking them along one axis. Also, the zero-padding only
	needs to be done along that axis, reducing memory usage while doing the
	transform.

	The knife-edge is set along the y-axis, so perpendicular to the x-axis.

	Parameters
	----------
	input_grid : Grid
		The grid of the wavefront that is to be propagated.
	q : scalar
		The amount of oversampling used for the knife edge.
	apodizer : OpticalElement or Field or None
		The pre-apodizer in the pupil before the focal-plane mask.
		If this is a Field, it will be converted into an apodizer.
		If this is None, no apodizer will be used.
	lyot_mask : OpticalElement or Field
		The Lyot stop in the pupil after the focal-plane mask. If
		this is a Field, it will be converted to an apodizer.
		If this is None, no Lyot stop will be used.
	'''
	def __init__(self, input_grid, q=8, apodizer=None, lyot_stop=None):
		fft = FastFourierTransform(input_grid, q, 1)
		self.cutout_input = (Ellipsis, fft.cutout_input[1])
		self.internal_shape = input_grid.shape[0], fft.internal_shape[1]
		
		# Create the knife-edge focal-plane mask along the x-axis.
		focal_mask_grid = make_fft_grid(input_grid, q).scaled(1.0 / (2 * np.pi))
		x = focal_mask_grid.separated_coords[0]
		self.focal_mask = (x > 0).astype('float')
		self.focal_mask[np.abs(x) < 1e-9] = 0.5

		# Pre-shift the mask to speed up propagations
		# FIXME: not necessary when 1D FFTs on 2D grids are implemented.
		self.focal_mask = np.fft.fftshift(self.focal_mask)
		
		if apodizer is not None and not hasattr(apodizer, 'input_grid'):
			apodizer = Apodizer(apodizer)
		self.apodizer = apodizer

		if lyot_stop is not None and not hasattr(lyot_stop, 'input_grid'):
			lyot_stop = Apodizer(lyot_stop)
		self.lyot_stop = lyot_stop
	
	def forward(self, wavefront):
		'''Propagate a wavefront forward through the knife-edge coronagraph.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		if self.apodizer is not None:
			wavefront = self.apodizer.forward(wavefront)

		ap = np.zeros(self.internal_shape, dtype='complex')
		ap[tuple(self.cutout_input)] = wavefront.electric_field.shaped
		
		# FIXME: use 1D FFTs on 2D grids implementation when available, instead of this.
		post_coro = np.fft.ifft(np.fft.fft(ap, axis=1) * self.focal_mask[np.newaxis,:], axis=1)
		post_coro = Field(post_coro[self.cutout_input].ravel(), wavefront.electric_field.grid)

		wavefront = Wavefront(post_coro, wavefront.wavelength)

		if self.lyot_stop is not None:
			wavefront = self.lyot_stop.forward(wavefront)
		
		return wavefront
	
	def backward(self, wavefront):
		'''Propagate a wavefront backward through the knife-edge coronagraph.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront to propagate.
		
		Returns
		-------
		Wavefront
			The propagated wavefront.
		'''
		if self.lyot_stop is not None:
			wavefront = self.lyot_stop.backward(wavefront)

		ap = np.zeros(self.internal_shape, dtype='complex')
		ap[tuple(self.cutout_input)] = wavefront.electric_field.shaped
		
		# FIXME: use 1D FFTs on 2D grids implementation when available, instead of this.
		post_coro = np.fft.ifft(np.fft.fft(ap, axis=1) * self.focal_mask[np.newaxis,:], axis=1)
		post_coro = Field(post_coro[self.cutout_input].ravel(), wavefront.electric_field.grid)

		wavefront = Wavefront(post_coro, wavefront.wavelength)

		if self.apodizer is not None:
			wavefront = self.apodizer.backward(wavefront)

		return wavefront
