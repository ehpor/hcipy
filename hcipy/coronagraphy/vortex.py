import numpy as np
from ..optics import OpticalElement, Wavefront
from ..propagation import FraunhoferPropagator
from ..field import make_focal_grid, Field, make_pupil_grid
from ..aperture import circular_aperture, rectangular_aperture
from ..fourier import FastFourierTransform, MatrixFourierTransform
from scipy.special import jv

import matplotlib.pyplot as plt
from ..plotting import imshow_field, complex_field_to_rgb

class VortexCoronagraph(OpticalElement):
	def __init__(self, input_grid, pupil_diameter=1, charge=2, scalings=4, scaling_factor=4):
		self.input_grid = input_grid
		
		q_outer = 2
		num_airy_outer = input_grid.shape[0]/2

		focal_grids = []
		self.focal_masks = []
		self.props = []

		for i in range(scalings):
			q = q_outer * scaling_factor**i
			if i == 0:
				num_airy = num_airy_outer
			else:
				num_airy = 32 / (q_outer * scaling_factor**(i-1))

			focal_grid = make_focal_grid(input_grid, q, num_airy)
			print(focal_grid.shape)
			focal_mask = Field(np.exp(1j * charge * focal_grid.as_('polar').theta), focal_grid)

			focal_mask *= 1 - circular_aperture(1e-9)(focal_grid)

			for j in range(i):
				fft = FastFourierTransform(focal_grids[j])
				mft = MatrixFourierTransform(focal_grid, fft.output_grid)

				focal_mask -= mft.backward(fft.forward(self.focal_masks[j]))
			
			prop = FraunhoferPropagator(input_grid, focal_grid)

			focal_grids.append(focal_grid)
			self.focal_masks.append(focal_mask)
			self.props.append(prop)

	def forward(self, wavefront):
		wavelength = wavefront.wavelength
		wavefront.wavelength = 1

		for i, (mask, prop) in enumerate(zip(self.focal_masks, self.props)):
			focal = prop(wavefront)
			focal.electric_field *= mask
			if i == 0:
				lyot = prop.backward(focal)
			else:
				lyot.electric_field += prop.backward(focal).electric_field

		lyot.wavelength = wavelength
		return lyot
	
	def backward(self, wavefront):
		wavelength = wavefront.wavelength
		wavefront.wavelength = 1

		for i, (mask, prop) in enumerate(zip(self.focal_masks, self.props)):
			focal = prop(wavefront)
			focal.electric_field *= mask.conj()
			if i == 0:
				pup = prop.backward(focal)
			else:
				pup.electric_field += prop.backward(focal).electric_field

		pup.wavelength = wavelength
		return pup