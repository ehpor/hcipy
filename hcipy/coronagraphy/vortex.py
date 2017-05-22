import numpy as np
from ..optics import OpticalElement, Wavefront
from ..propagation import FraunhoferPropagator
from ..field import make_focal_grid, Field
from ..aperture import circular_aperture

import matplotlib.pyplot as plt
from ..plotting import imshow_field

class VortexCoronagraph(OpticalElement):
	def __init__(self, pupil_grid, pupil_diameter=1, charge=2, method='GS', q=16, num_airy=32):
		self.pupil_grid = pupil_grid
		self.focal_grid = make_focal_grid(pupil_grid, q, num_airy)
		self.prop = FraunhoferPropagator(pupil_grid, self.focal_grid)

		if method == 'GS':
			aperture = circular_aperture(pupil_diameter)(pupil_grid)
			wf = Wavefront(aperture)

			self.vortex = np.exp(1j * charge * self.focal_grid.as_('polar').theta)
			self.vortex = Field(self.vortex, self.focal_grid)

			for i in range(30):
				foc = self.prop(wf)
				foc.electric_field *= self.vortex

				pup = self.prop.backward(foc)
				pup.electric_field[aperture < 0.5] = 0

				foc2 = self.prop(pup)
				self.vortex *= 1 - foc2.electric_field / foc.electric_field

			imshow_field(np.angle(self.vortex), self.focal_grid)
			plt.colorbar()
			plt.show()
		elif method is 'none':
			self.vortex = np.exp(1j * charge * self.focal_grid.as_('polar').theta)

	def forward(self, wavefront):
		foc = self.prop(wavefront)

		foc.electric_field *= self.vortex

		return self.prop.backward(foc)