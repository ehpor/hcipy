from .wavefront_sensor import WavefrontSensorOptics, WavefrontSensorEstimator
from ..optics import *
from ..field import *
from ..aperture import *
from ..propagation import FresnelPropagator

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
	def __init__(self, mla_grid, mla_index):
		self.mla_grid = mla_grid
		self.mla_index = mla_index
		self.unique_indices = np.unique(self.mla_index)
		
	def estimate(self, images):
		image = images[0]

		centroids = np.empty([self.unique_indices.size, 2])
		
		for i, index in enumerate(self.unique_indices):
			# Select individual subapertures based on mla_index
			mask = self.mla_index == index

			# Mask off this part
			subimage = image[mask]
			x = image.grid.x[mask]
			y = image.grid.y[mask]

			# Find centroid
			centroid_x = np.sum(subimage * x) / np.sum(subimage)
			centroid_y = np.sum(subimage * y) / np.sum(subimage)

			# Subtract lenslet position and store.
			centroids[i,:] = np.array((centroid_x, centroid_y)) - self.mla_grid[index]

		return Field(centroids, self.mla_grid)