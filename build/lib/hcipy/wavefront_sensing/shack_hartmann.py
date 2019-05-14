from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..optics import *
from ..field import *
from ..aperture import *
from ..propagation import FresnelPropagator
from scipy import ndimage

import numpy as np

class ShackHartmannWavefrontSensorOptics(WavefrontSensorOptics):
	def __init__(self, input_grid, micro_lens_array):
		# Make propagator
		sh_prop = FresnelPropagator(input_grid, micro_lens_array.focal_length)
		
		# Make optical system
		OpticalSystem.__init__(self, (micro_lens_array, sh_prop))
		self.mla_index = micro_lens_array.mla_index
		self.mla_grid = micro_lens_array.mla_grid
		self.micro_lens_array = micro_lens_array

class SquareShackHartmannWavefrontSensorOptics(ShackHartmannWavefrontSensorOptics):
	## Helper class to create a Shack-Hartmann WFS with square microlens array
	def __init__(self, input_grid, f_number, num_lenslets, pupil_diameter):
		lenslet_diameter = float(pupil_diameter) / num_lenslets
		x = np.arange(-pupil_diameter, pupil_diameter, lenslet_diameter)
		self.mla_grid = CartesianGrid(SeparatedCoords((x, x)))

		focal_length = f_number * lenslet_diameter
		self.micro_lens_array = MicroLensArray(input_grid, self.mla_grid, focal_length)
		
		ShackHartmannWavefrontSensorOptics.__init__(self, input_grid, self.micro_lens_array)

class ShackHartmannWavefrontSensorEstimator(WavefrontSensorEstimator):
	def __init__(self, mla_grid, mla_index, estimation_subapertures=None):
		self.mla_grid = mla_grid
		self.mla_index = mla_index
		if estimation_subapertures is None:
			self.estimation_subapertures = np.unique(self.mla_index)
		else:
			self.estimation_subapertures = np.flatnonzero(np.array(estimation_subapertures))
		self.estimation_grid = self.mla_grid.subset(estimation_subapertures)
		
	def estimate(self, images, use_par_map=True):
		image = images[0]
		
		fluxes = ndimage.measurements.sum(image, self.mla_index, self.estimation_subapertures)
		sum_x = ndimage.measurements.sum(image * image.grid.x, self.mla_index, self.estimation_subapertures)
		sum_y = ndimage.measurements.sum(image * image.grid.y, self.mla_index, self.estimation_subapertures)

		centroid_x = sum_x / fluxes
		centroid_y = sum_y / fluxes

		centroids = np.array((centroid_x, centroid_y)) - np.array(self.mla_grid.points[self.estimation_subapertures,:]).T
		return Field(centroids, self.estimation_grid)