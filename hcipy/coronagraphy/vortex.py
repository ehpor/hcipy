import numpy as np
from ..optics import OpticalElement, Wavefront
from ..propagation import FraunhoferPropagator
from ..field import make_focal_grid, Field
from ..aperture import circular_aperture
from ..fourier import FastFourierTransform
from scipy.special import jv

import matplotlib.pyplot as plt
from ..plotting import imshow_field, complex_field_to_rgb

class VortexCoronagraph(OpticalElement):
	def __init__(self, pupil_grid, pupil_diameter=1, charge=2, method='gs', q=8, num_airy=128):
		self.pupil_grid = pupil_grid
		self.focal_grid = make_focal_grid(pupil_grid, q, num_airy)
		self.prop = FraunhoferPropagator(pupil_grid, self.focal_grid)

		self.method = method.lower()

		if self.method == 'gs':
			aperture = circular_aperture(pupil_diameter)(pupil_grid)
			wf = Wavefront(aperture)

			self.vortex = np.exp(1j * charge * self.focal_grid.as_('polar').theta)
			self.vortex = Field(self.vortex, self.focal_grid)
			i = self.focal_grid.closest_to((0,0))
			self.vortex[i] = 0

			foc = self.prop(wf)

			for i in range(100):
				foc2 = foc.copy()
				foc2.electric_field *= self.vortex
				foc2.electric_field *= circular_aperture(num_airy*2)(self.focal_grid)

				pup = self.prop.backward(foc2)
				pup.electric_field[aperture < 0.5] = 0

				foc3 = self.prop(pup)
				#foc3.electric_field /= self.vortex

				self.vortex -= foc3.electric_field / foc.electric_field
				self.vortex = np.exp(1j * np.angle(self.vortex))

			#plt.subplot(1,2,1)
			#imshow_field(np.abs(self.vortex), self.focal_grid, vmax=1.2)
			#plt.colorbar()
			#plt.subplot(1,2,2)
			#imshow_field(np.angle(self.vortex), self.focal_grid)
			#plt.colorbar()
			#plt.show()
		elif self.method.lower() == 'none':
			self.vortex = np.exp(1j * charge * self.focal_grid.as_('polar').theta)
			self.vortex = Field(self.vortex, self.focal_grid)
			i = self.focal_grid.closest_to((0,0))
			self.vortex[i] = 0
		elif self.method.lower() == 'convolution':
			pup_pol = pupil_grid.as_('polar')
			L = 512 # size of the focal-plane mask in lambda/D
			r_scaled = 2*np.pi * L * pup_pol.r

			if charge == 2:
				F = np.exp(2j * pup_pol.theta) / (np.pi * pup_pol.r**2) * (-1 + jv(0,r_scaled) + 0.5 * r_scaled * jv(1, r_scaled))
			elif charge == 4:
				F = np.exp(4j * pup_pol.theta) / (np.pi * pup_pol.r**2) * (2 + 4 * jv(0, r_scaled) + (0.5 * r_scaled - 12 / r_scaled) * jv(1, r_scaled))
			elif charge > 4:
				pass
			
			F *= circular_aperture(pupil_diameter)(pupil_grid)
			self.fourier = FastFourierTransform(pupil_grid, 4, 1)
			self.F = self.fourier.forward(F)

			plt.subplot(1,2,1)
			imshow_field(np.abs(F), self.fourier.input_grid)
			plt.colorbar()
			plt.subplot(1,2,2)
			imshow_field(np.angle(F), self.fourier.input_grid)
			plt.colorbar()
			plt.figure()

			plt.subplot(1,2,1)
			imshow_field(np.abs(self.F), self.fourier.output_grid)
			plt.colorbar()
			plt.subplot(1,2,2)
			imshow_field(np.angle(self.F), self.fourier.output_grid)
			plt.colorbar()
			plt.show()
		elif method.lower() == 'inversion':
			aperture = circular_aperture(pupil_diameter)(pupil_grid)
			wf = Wavefront(aperture)
			wf.total_power = 1

			self.vortex = np.exp(1j * charge * self.focal_grid.as_('polar').theta)
			self.vortex = Field(self.vortex, self.focal_grid)
			self.vortex *= circular_aperture(num_airy*2)(self.focal_grid)

			i = self.focal_grid.closest_to((0,0))
			self.vortex[i] = 0

			foc = self.prop.forward(wf)
			foc_copy = foc.copy()
			foc.electric_field *= self.vortex

			pup = self.prop.backward(foc)
			pup.electric_field[aperture < 0.5] = 0
			
			self.reference = pup.electric_field

	def forward(self, wavefront):
		if self.method == 'gs' or self.method == 'none' or self.method == 'inversion':
			foc = self.prop(wavefront)
			foc.electric_field *= self.vortex
			res = self.prop.backward(foc)
			if self.method == 'inversion':
				res.electric_field -= self.reference * np.sqrt(wavefront.total_power)
			return res
		elif self.method == 'convolution':
			ft = self.fourier.forward(wavefront.electric_field)
			ft *= self.F
			conv = self.fourier.backward(ft)

			return Wavefront(conv, wavefront.wavelength)
