from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..plotting import imshow_field
from ..optics import SurfaceApodizer, PhaseApodizer
from ..field import make_pupil_grid, make_focal_grid, Field
import numpy as np
from matplotlib import pyplot as plt

def phase_step_mask(diameter=1, phase_step=np.pi/2):
	def func(grid):
		radius = grid.as_('polar').r
		phase_mask = Field(phase_step * (radius <= diameter/2), grid)
		return PhaseApodizer(phase_mask)
	return func

class ZernikeWavefrontSensorOptics(WavefrontSensorOptics):
	def __init__(self, pupil_grid, num_pupil_pixels=32, q=2, wavelength_0=1, diameter=1.06, num_airy=None):

		output_grid_size = pupil_grid.x.ptp()

		# Create the intermediate and final grids
		self.output_grid = make_pupil_grid(num_pupil_pixels, output_grid_size)
		self.focal_grid = make_focal_grid(pupil_grid, q=q, num_airy=num_airy, wavelength=wavelength_0)

		# Make all the optical elements
		self.pupil_to_focal = FraunhoferPropagator(pupil_grid, self.focal_grid, wavelength_0=wavelength_0)
		self.phase_step = phase_step_mask(diameter * wavelength_0, np.pi/2)(self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid, wavelength_0=wavelength_0)

	def forward(self, wavefront):
		wf = wavefront.copy()

		wf = self.pupil_to_focal.forward(wf)
		wf = self.phase_step.forward(wf)
		wf = self.focal_to_pupil(wf)

		return wf

class ZernikeWavefrontSensorEstimator(WavefrontSensorEstimator):
	def __init__(self, aperture, output_grid, reference):
		self.measurement_grid = output_grid
		self.pupil_mask = aperture(self.measurement_grid)
		self.reference = reference

	def estimate(self, images):
		image = images[0]

		intensity_measurements = (image-self.reference).ravel() * self.pupil_mask
		res = Field(intensity_measurements[self.pupil_mask>0], self.pupil_mask.grid)
		return res