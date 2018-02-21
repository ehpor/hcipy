from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..propagation import FraunhoferPropagator
from ..plotting import imshow_field
from ..optics import SurfaceApodizer, PhaseApodizer
from ..field import make_pupil_grid, make_focal_grid, Field
import numpy as np
from matplotlib import pyplot as plt

def pyramid_surface(refractive_index, separation, wavelength_0):
	def func(grid):
		surf = -separation / (refractive_index(wavelength_0) - 1) * (np.abs(grid.x) + np.abs(grid.y))
		surf = Field(surf, grid)
		return SurfaceApodizer(surf, refractive_index)
	return func

class PyramidWavefrontSensorOptics(WavefrontSensorOptics):
	def __init__(self, pupil_grid, wavelength_0=1, pupil_separation=1.5, pupil_diameter=None, num_pupil_pixels=32, q=4, refractive_index=lambda x : 1.5, num_airy=None):
		
		if pupil_diameter is None:
			pupil_diameter = pupil_grid.x.ptp()
		
		# Make mask
		sep = 0.5 * pupil_separation * pupil_diameter
		
		# Multiply by 2 because we want to have two pupils next to each other
		output_grid_size = (pupil_separation + 1) * pupil_diameter
		output_grid_pixels = np.ceil(num_pupil_pixels * (pupil_separation + 1))

		# Need at least two times over sampling in the focal plane because we want to separate two pupils completely
		if q < 2 * pupil_separation:
			q = 2 * pupil_separation

		# Create the intermediate and final grids
		self.output_grid = make_pupil_grid(output_grid_pixels, output_grid_size)
		self.focal_grid = make_focal_grid(pupil_grid, q=q, num_airy=num_airy, wavelength=wavelength_0)

		# Make all the optical elements
		self.pupil_to_focal = FraunhoferPropagator(pupil_grid, self.focal_grid, wavelength_0=wavelength_0)
		self.pyramid = pyramid_surface(refractive_index, sep, wavelength_0)(self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid, wavelength_0=wavelength_0)

	def forward(self, wavefront):
		wf = wavefront.copy()
		
		wf = self.pupil_to_focal.forward(wf)
		wf = self.pyramid.forward(wf)		
		wf = self.focal_to_pupil(wf)

		return wf

class PyramidWavefrontSensorEstimator(WavefrontSensorEstimator):
	def __init__(self, aperture, output_grid):
		self.measurement_grid = make_pupil_grid(output_grid.shape[0]/2, output_grid.x.ptp()/2)
		self.pupil_mask = aperture(self.measurement_grid)
		self.num_measurements = 2 * int(np.sum(self.pupil_mask > 0))

	def estimate(self, images):
		image = images.shaped
		sub_shape = image.grid.shape // 2

		# Subpupils
		I_a = image[:sub_shape[0], :sub_shape[1]]
		I_b = image[sub_shape[0]:2*sub_shape[0], :sub_shape[1]]
		I_c = image[sub_shape[0]:2*sub_shape[0], sub_shape[1]:2*sub_shape[1]]
		I_d = image[:sub_shape[0], sub_shape[1]:2*sub_shape[1]]

		norm = I_a + I_b + I_c + I_d

		I_x = (I_a + I_b - I_c - I_d) / norm
		I_y = (I_a - I_b - I_c + I_d) / norm

		I_x = I_x.ravel()[self.pupil_mask>0]
		I_y = I_y.ravel()[self.pupil_mask>0]

		res = np.column_stack((I_x, I_y))
		res = Field(res, self.pupil_mask.grid)
		return res